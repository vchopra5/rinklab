"""Verify time boundaries, fresh models, probability handling, and loss calculations."""

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from season_cv import SeasonCV


def dataset():
    return pd.DataFrame([
        {"game_id": year * 1_000_000 + 20001, "x": year, "y": lateral, "goal": goal}
        for year in range(2018, 2026)
        for lateral, goal in [(-10, 0), (10, 1)]
    ])


class ConstantModel:
    def fit(self, X, goal):
        self.training_years = set(X[:, 0].astype(int))
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.25)


class SeasonCVTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset()
        self.cv = SeasonCV(self.data)
        self.created = []

    def factory(self):
        model = ConstantModel()
        self.created.append(model)
        return model

    def test_expanding_folds_and_metrics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions = self.cv.run_cv({"constant": self.factory})
        self.assertEqual(len(self.created), 5)
        for model, year in zip(self.created, range(2019, 2024)):
            self.assertEqual(model.training_years, set(range(2018, year)))
        self.assertEqual(set(predictions.season), set(range(2019, 2024)))
        self.assertTrue(predictions.row_id.is_unique)
        np.testing.assert_allclose(metrics.brier, 0.3125)
        np.testing.assert_allclose(metrics.log_loss, -0.5 * (np.log(.25) + np.log(.75)))
        pd.testing.assert_frame_equal(self.data, dataset())

    def test_final_test_fits_once_without_test_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions = self.cv.run_final_test({"constant": self.factory})
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.created[0].training_years, set(range(2018, 2024)))
        self.assertEqual(set(predictions.season), {2024, 2025})
        self.assertEqual(len(metrics), 2)

    def test_two_column_probabilities_and_invalid_predictions(self):
        class TwoColumn(ConstantModel):
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                return np.tile([.75, .25], (len(X), 1))

        X = np.zeros((2, 2))
        np.testing.assert_allclose(self.cv._probabilities(TwoColumn(), X), [.25, .25])
        model = TwoColumn()
        model.classes_ = np.array([1, 0])
        with self.assertRaises(AssertionError):
            self.cv._probabilities(model, X)
        for invalid in [np.array([np.nan, .5]), np.array([-.1, 1.1]), np.zeros((2, 1))]:
            model.predict_proba = lambda X, value=invalid: value
            with self.assertRaises(AssertionError):
                self.cv._probabilities(model, X)

    def test_model_errors_propagate(self):
        class Broken(ConstantModel):
            def fit(self, X, goal):
                raise RuntimeError("training failed")

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.cv.run_cv({"broken": Broken})

    def test_overlapping_windows_rejected(self):
        with self.assertRaises(AssertionError):
            SeasonCV(self.data, validation_years=(2024,), test_years=(2024, 2025))


if __name__ == "__main__":
    unittest.main()
