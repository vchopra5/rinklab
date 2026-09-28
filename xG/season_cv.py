"""Chronological xG evaluation. Models implement fit(X, goal) and predict_proba(X).

Features are always raw aligned (x, y); transformations belong inside each model.
Factories must return fresh unfitted models. No exceptions are caught here.
"""

import time

import numpy as np
import pandas as pd


class SeasonCV:
    def __init__(self, data, validation_years=(2019, 2020, 2021, 2022, 2023),
                 test_years=(2024, 2025)):
        """Years are season starting years: 2024 means 2024–25."""
        self.data = data[["game_id", "x", "y", "goal"]].copy().reset_index(drop=True)
        self.data["season"] = self.data["game_id"].astype("int64") // 1_000_000
        self.validation_years = tuple(validation_years)
        self.test_years = tuple(test_years)
        assert self.validation_years and self.test_years
        assert list(self.validation_years) == sorted(set(self.validation_years))
        assert list(self.test_years) == sorted(set(self.test_years))
        assert max(self.validation_years) < min(self.test_years)
        assert self.data["goal"].isin([0, 1]).all()
        assert np.isfinite(self.data[["x", "y"]].to_numpy()).all()
        available = set(self.data["season"])
        assert set(self.validation_years + self.test_years) <= available

    @staticmethod
    def _probabilities(model, X):
        """Accept P(goal), or two columns explicitly ordered [non-goal, goal]."""
        p = np.asarray(model.predict_proba(X), dtype=np.float64)
        if p.ndim == 2:
            assert p.shape == (len(X), 2)
            if hasattr(model, "classes_"):
                assert np.array_equal(model.classes_, [0, 1]), "Expected class order [0, 1]"
            p = p[:, 1]
        assert p.shape == (len(X),)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        return p

    @staticmethod
    def _metrics(goal, probability):
        # The same clipping convention applies to every model's log loss.
        p_log = np.clip(probability, 1e-7, 1 - 1e-7)
        return {
            "log_loss": -np.mean(goal * np.log(p_log) + (1 - goal) * np.log1p(-p_log)),
            "brier": np.mean((probability - goal) ** 2),
            "observed_goal_rate": np.mean(goal),
            "predicted_goal_rate": np.mean(probability),
        }

    def _evaluate(self, factories, train, evaluation, stage):
        assert len(train) > 0 and len(evaluation) > 0
        assert train["goal"].nunique() == 2
        assert train["season"].max() < evaluation["season"].min()
        assert set(train["game_id"]).isdisjoint(evaluation["game_id"])
        X_train = train[["x", "y"]].to_numpy(dtype=np.float64)
        goal_train = train["goal"].to_numpy(dtype=np.uint8)
        X_eval = evaluation[["x", "y"]].to_numpy(dtype=np.float64)
        records, predictions = [], []
        for name, factory in factories.items():
            model = factory()
            start = time.perf_counter()
            model.fit(X_train.copy(), goal_train.copy())
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            probability = self._probabilities(model, X_eval.copy())
            predict_seconds = time.perf_counter() - start
            output = evaluation[["game_id", "season", "goal"]].copy()
            # Stable row ID for matching predictions between models and folds.
            output.insert(0, "row_id", evaluation.index.to_numpy())
            output["probability"] = probability
            output["model"] = name
            output["stage"] = stage
            predictions.append(output)
            for year, group in output.groupby("season"):
                scores = self._metrics(group["goal"].to_numpy(), group["probability"].to_numpy())
                records.append({
                    "model": name, "stage": stage, "evaluation_season": int(year),
                    "train_start": int(train["season"].min()), "train_end": int(train["season"].max()),
                    "n_train": len(train), "n_evaluation": len(group),
                    "fit_seconds": fit_seconds,
                    # For final testing, timing covers BOTH test seasons together.
                    "predict_seconds": predict_seconds, **scores,
                })
                print(f"{name} | {stage} {year}–{year + 1} | "
                      f"log loss={scores['log_loss']:.5f} | Brier={scores['brier']:.5f}", flush=True)
        return pd.DataFrame(records), pd.concat(predictions, ignore_index=True)

    def run_cv(self, factories):
        """Return per-season metrics and held-out predictions; never score test years."""
        assert factories
        results, predictions = [], []
        for year in self.validation_years:
            train = self.data.loc[self.data["season"] < year]
            validation = self.data.loc[self.data["season"] == year]
            metrics, output = self._evaluate(factories, train, validation, "validation")
            results.append(metrics)
            predictions.append(output)
        return pd.concat(results, ignore_index=True), pd.concat(predictions, ignore_index=True)

    def run_final_test(self, selected_factories):
        """After tuning, fit once before the test window; no updates between test years."""
        assert selected_factories
        train = self.data.loc[self.data["season"] < min(self.test_years)]
        test = self.data.loc[self.data["season"].isin(self.test_years)]
        return self._evaluate(selected_factories, train, test, "test")
