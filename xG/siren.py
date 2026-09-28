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
    def __init__(self, omega=30.0):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            Sine(omega),
            nn.Linear(128, 128),
            Sine(omega),
            nn.Linear(128, 1),
        )

        # First-layer and hidden/output-layer SIREN weight initialization.
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    bound = (1 / layer.in_features if layer is self.layers[0]
                             else np.sqrt(6 / layer.in_features) / omega)
                    layer.weight.uniform_(-bound, bound)
    
    def forward(self, X):
        scaled = X / X.new_tensor([100.0, 42.5])
        return self.layers(scaled).squeeze(-1)
    
    def fit(self, X, y, epochs=20, batch_size=2048, lr=1e-3):
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
