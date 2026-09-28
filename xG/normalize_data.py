import numpy as np
import pandas as pd
from pathlib import Path

class LoadHockeyShots:
    def __init__(self):
        pass
    
    def load_data(self):
        folder = Path(__file__).resolve().parent / "data" / "shots"

        columns = [
            "game_id",
            "game_type",
            "period",
            "period_type",
            "x_coord",
            "y_coord",
            "shot_type",
            "event_type",
            "zone_code",
            "home_team_id",
            "away_team_id",
            "situationCode",
            "event_owner_team_id",
            "home_team_defending_side"
        ]

        frames = []

        for path in sorted(folder.glob("shots_*.csv")):
            frame = pd.read_csv(
                path,
                usecols=columns,
                dtype={"x_coord": "float32", "y_coord": "float32", "situationCode": "string"},
            )

            frames.append(frame.loc[
                frame["period_type"].eq("REG")
                & frame["game_type"].eq(2)
            ].copy())

        df = pd.concat(frames, ignore_index=True)
        del frames, frame

        df["shot_type"] = df["shot_type"].astype("category")

        evidence = df.loc[
            df["event_type"].isin(["goal", "shot-on-goal", "missed-shot"])
            & df["zone_code"].eq("O")
            & df["x_coord"].abs().gt(25)
        ].copy()

        evidence["vote"] = (
            np.sign(evidence["x_coord"])
            * np.where(
                evidence["event_owner_team_id"].eq(evidence["home_team_id"]),
                1,
                -1
            )
        )

        vote_sum = evidence.groupby(["game_id", "period"])["vote"].sum()
        inferred = np.sign(vote_sum).replace(0, np.nan)

        df = df.join(
            inferred.rename("inferred_home_direction"),
            on=["game_id", "period"],
        )

        del evidence

        api_direction = df["home_team_defending_side"].map(
            {
                "left": 1,
                "right": -1,
            }
        )

        home_direction = api_direction.fillna(df["inferred_home_direction"])

        shooting_direction = home_direction * np.where(
            df["event_owner_team_id"].eq(df["home_team_id"]),
            1,
            -1
        )

        df["x"] = (df["x_coord"] * shooting_direction).astype("float32")
        df["y"] = (df["y_coord"] * shooting_direction).astype("float32")
        df["goal"] = df["event_type"].eq("goal").astype("uint8")
        five_on_five = df["situationCode"].eq("1551").fillna(False)

        # 4. Report and exclude unresolved directions or missing coordinates.
        unresolved = home_direction.isna()
        missing_coordinates = df[["x_coord", "y_coord"]].isna().any(axis=1)
        exclude = unresolved | missing_coordinates | (~five_on_five)

        print(f"Regulation attempts loaded: {len(df):,}")
        print(f"Unresolved direction:       {unresolved.sum():,}")
        print(f"Missing coordinates:        {missing_coordinates.sum():,}")
        print(f"Not 5v5:                    {(~five_on_five).sum():,}")
        print(f"Total rows excluded:        {exclude.sum():,}")

        # Keep game_id so training/test splits can be grouped by game.
        data = df.loc[
            ~exclude,
            ["game_id", "x", "y", "shot_type", "goal"],
        ].dropna(subset=["game_id", "x", "y", "goal"]).reset_index(drop=True)

        del df

        print(f"Final attempts:             {len(data):,}")
        print(f"Memory: {data.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(data.head())

        return data
