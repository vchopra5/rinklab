import torch
from torch import nn
import numpy as np

from normalize_data import LoadHockeyShots
from season_cv import SeasonCV

class Sine(nn.Module):
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega

    def forward(self, X):
        return torch.sin(self.omega * X)


class SIREN2D(nn.Module):
    def __init__(self, omega=30.0, hidden_sizes=(32, 32, 32), features="xy"):
        super().__init__()
        if features not in ("xy", "net", "xy_net", "radial", "xy_radial"):
            raise ValueError("features must be xy, net, xy_net, radial, or xy_radial")
        self.features = features
        self.hidden_sizes = tuple(hidden_sizes)
        if not self.hidden_sizes or any(not isinstance(n, int) or n <= 0 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integer widths")
        layers = []
        incoming = 4 if features in ("xy_net", "xy_radial") else 2
        for width in self.hidden_sizes:
            layers.extend([nn.Linear(incoming, width), Sine(omega)])
            incoming = width
        layers.append(nn.Linear(incoming, 1))
        self.layers = nn.Sequential(*layers)

        # First-layer and hidden/output-layer SIREN weight initialization.
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    bound = (1 / layer.in_features if layer is self.layers[0]
                             else np.sqrt(6 / layer.in_features) / omega)
                    layer.weight.uniform_(-bound, bound)
    
    def encode_features(self, X):
        scaled = X / X.new_tensor([100.0, 42.5])
        if self.features == "xy":
            return scaled
        distance = 89.0 - X[..., 0]
        lateral = X[..., 1]
        # Bearing from net center: 0 straight in front, +/-pi straight behind.
        # Angle is undefined at net center; use 0 there for training values.
        origin = (distance == 0) & (lateral == 0)
        safe_distance = torch.where(origin, torch.ones_like(distance), distance)
        angle = torch.atan2(lateral, safe_distance)
        if self.features in ("radial", "xy_radial"):
            # Signed radial distance: positive on the goal line and in front,
            # negative behind it. Preserve legacy goal-line feature modes.
            radius = torch.linalg.vector_norm(torch.stack((distance, lateral), dim=-1), dim=-1)
            distance = torch.where(distance < 0, -radius, radius)
        net = torch.stack((distance / 100.0, angle / torch.pi), dim=-1)
        return torch.cat((scaled, net), dim=-1) if self.features in ("xy_net", "xy_radial") else net

    def forward(self, X):
        return self.layers(self.encode_features(X)).squeeze(-1)
    
    def fit(self, X, y, epochs=20, batch_size=2048, lr=1e-4):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.BCEWithLogitsLoss()

        self.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for coords, goals in loader:
                optimizer.zero_grad()
                logits = self(coords)
                loss = loss_function(logits, goals)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(goals)
            
            print(
                f"Epoch {epoch + 1}: Loss: {total_loss / len(dataset):.6f}"
            )
        
        return self

    def predict_proba(self, X, batch_size=4096):
        X = torch.tensor(X, dtype=torch.float32)
        self.eval()

        with torch.no_grad():
            probabilities = [
                torch.sigmoid(self(batch))
                for batch in X.split(batch_size)
            ]

        return torch.cat(probabilities).numpy()

    def gradient(self, coords):
        self.eval()
        point = torch.tensor(coords, dtype=torch.float32, requires_grad=True)
        probability = torch.sigmoid(self(point))
        gradient, = torch.autograd.grad(probability.sum(), point)
        return gradient.detach().numpy()


if __name__ == "__main__":
    data = LoadHockeyShots().load_data()
    cv = SeasonCV(data)

    metrics, predictions = cv.run_cv({
        "siren": SIREN2D,
    })

    print(
        metrics.groupby("model")[["log_loss", "brier"]]
        .agg(["mean", "std"])
    )

    print(
        metrics[
            ["evaluation_season", "fit_seconds", "predict_seconds"]
        ]
    )
