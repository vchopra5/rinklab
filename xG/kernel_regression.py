import numpy as np
from scipy.spatial import cKDTree
from normalize_data import LoadHockeyShots
from season_cv import SeasonCV

class KernelRegression2D:
    def __init__(self, bandwidth=2, radius=None):
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite")
        if radius is not None and (not np.isfinite(radius) or radius <= 0):
            raise ValueError("radius must be positive and finite, or None")
        self.h = bandwidth
        # None uses the full Gaussian. A radius retains the legacy hard cutoff.
        self.radius = radius

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0 or y.shape != (len(X),):
            raise ValueError("Expected nonempty coordinates (n, 2) and outcomes (n,)")
        if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Coordinates must be finite and outcomes binary")
        # Exact compression of repeated locations, not spatial binning.
        self.locations, inverse, self.counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        self.goals = np.bincount(inverse, weights=y, minlength=len(self.locations))
        self.tree = cKDTree(self.locations)
        return self

    def _evaluate(self, X, derivatives=False):
        X = np.asarray(X, dtype=float).reshape(-1, 2)
        result = []
        for q in X:
            idx = (slice(None) if self.radius is None
                   else self.tree.query_ball_point(q, r=self.radius))
            diff = self.locations[idx] - q
            if len(diff) == 0:
                raise ValueError("No attempts inside the requested hard radius")
            distance2 = np.sum(diff**2, axis=1)
            # Common rescaling cancels in the probability and gradient ratios.
            w = np.exp(-(distance2 - distance2.min()) / (2 * self.h**2))
            counts, goals = self.counts[idx], self.goals[idx]
            denominator = np.sum(w * counts)
            probability = np.sum(w * goals) / denominator
            if derivatives:
                residual = w * (goals - probability * counts)
                result.append(np.sum(residual[:, None] * diff, axis=0) / (self.h**2 * denominator))
            else:
                result.append(probability)
        return np.asarray(result).reshape((-1, 2) if derivatives else (-1,))

    def predict_proba(self, X):
        locations, inverse = np.unique(np.asarray(X, dtype=float).reshape(-1, 2), axis=0, return_inverse=True)
        return self._evaluate(locations)[inverse]

    def gradient(self, coord):
        coord = np.asarray(coord, dtype=float)
        gradient = self._evaluate(coord, derivatives=True)
        return gradient[0] if coord.ndim == 1 else gradient

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
