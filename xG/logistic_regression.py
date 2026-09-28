import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from normalize_data import LoadHockeyShots
from season_cv import SeasonCV

class LogisticRegression2D:
    def __init__(self, C = 1.0):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=1000),
        )
    
    def chg_features(self, X):
        X = np.asarray(X)
        # y^2 to allow the model to pick up on data near boards
        return np.column_stack([X[:, 0], X[:, 1] ** 2])
    
    def fit(self, X, y):
        self.model.fit(self.chg_features(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.chg_features(X))[:, 1]
    
    def gradient(self, coords):
        scaler, lr = self.model.steps[0][1], self.model.steps[1][1]
        p = self.predict_proba(np.array([coords]))[0]

        gradient = p * (1 - p) * (lr.coef_[0] / scaler.scale_) * np.array([1, 2*coords[1]])

        return gradient

if __name__ == '__main__':
    data = LoadHockeyShots().load_data()
    cv = SeasonCV(data)

    metrics, predictions = cv.run_cv({
        "logistic": lambda: LogisticRegression2D(),
    })

    print(
        metrics.groupby("model")[["log_loss", "brier"]]
        .agg(["mean", "std"])
    )
