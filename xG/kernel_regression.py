import numpy as np
from scipy.spatial import cKDTree
from normalize_data import LoadHockeyShots
from season_cv import SeasonCV

class KernelRegression2D:
    def __init__(self, bandwidth=4, radius=1.0):
        self.h = bandwidth
        self.radius = radius
    
    def fit(self, X, y):
        # X shape: (n_samples, 2)
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        self.tree = cKDTree(self.X)
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X)

        locations, inverse = np.unique(
            X, axis=0, return_inverse=True
        )

        probabilities = np.empty(len(locations))

        for i, x in enumerate(locations):
            idx = self.tree.query_ball_point(x, r=self.radius)

            Xn = self.X[idx]
            yn = self.y[idx]

            diff = Xn - x
            dist2 = np.sum(diff**2, axis=1)

            w = np.exp(-dist2 / (2 * self.h**2))

            probabilities[i] = np.sum(w * yn) / np.sum(w)

        return probabilities[inverse]


    def gradient(self, coord):
        idx = self.tree.query_ball_point(coord, r=self.radius)
        Xn = self.X[idx]
        yn = self.y[idx]
        diff = Xn - coord
        dist2 = np.sum(diff**2, axis=1)

        w = np.exp(-dist2 / (2 * self.h**2))
        p_hat = np.sum(w * yn) / np.sum(w)

        weighted_residual = w * (yn - p_hat)
        denominator = self.h**2 * np.sum(w)

        grad_x = np.sum(weighted_residual * diff[:, 0]) / denominator
        grad_y = np.sum(weighted_residual * diff[:, 1]) / denominator

        return np.array([grad_x, grad_y])

if __name__ == '__main__':
    data = LoadHockeyShots().load_data()
    cv = SeasonCV(data)

    bandwidths = [2]

    metrics, predictions = cv.run_cv({
        f"kernel_{bw}": lambda bw=bw: KernelRegression2D(bandwidth=bw)
        for bw in bandwidths
    })

    print(
        metrics.groupby("model")[["log_loss", "brier"]]
        .agg(["mean", "std"])
    )
