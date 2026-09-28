"""Fit four location-only xG models and compare probability/gradient geometry.

Run: python compare_geometry.py
Uses seasons through 2023–24; final test seasons are not evaluated.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np
from scipy.spatial import cKDTree
import torch

from normalize_data import LoadHockeyShots
from logistic_regression import LogisticRegression2D
from kernel_regression import KernelRegression2D
from mlp import MLP2D
from siren import SIREN2D


def rink(ax):
    ax.add_patch(FancyBboxPatch(
        (-100, -42.5), 200, 85, boxstyle="round,pad=0,rounding_size=28",
        fill=False, edgecolor="#526474", linewidth=.8))
    for y in (-22, 22):
        ax.add_patch(Circle((69, y), 15, fill=False, edgecolor="#526474", linewidth=.6))
    ax.plot([89, 89], [-37, 37], color="#526474", linewidth=.6)
    ax.add_patch(Rectangle((89, -3), 4, 6, fill=False, edgecolor="#283849", linewidth=1))
    ax.set(xlim=(25, 100), ylim=(-42.5, 42.5), aspect="equal",
           xticks=[25, 50, 75, 100], yticks=[-40, -20, 0, 20, 40])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.set_facecolor("#ededed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--bandwidth", type=float, default=2)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--siren-only", action="store_true",
                        help="Retrain SIREN and reuse the other saved model surfaces.")
    selection.add_argument("--mlp-only", action="store_true",
                           help="Retrain MLP and reuse the other saved model surfaces.")
    args = parser.parse_args()
    only = "SIREN" if args.siren_only else "MLP" if args.mlp_only else None
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    output = root / "analysis" / "geometry"
    output.mkdir(parents=True, exist_ok=True)

    data = LoadHockeyShots().load_data()
    data = data.loc[data.game_id.astype("int64") // 1_000_000 <= 2023]
    X = data[["x", "y"]].to_numpy(dtype=float)
    goal = data.goal.to_numpy(dtype=float)
    print(f"Geometry training set: {len(X):,} attempts, {data.game_id.nunique():,} games", flush=True)

    factories = {
        "Logistic": LogisticRegression2D,
        "Kernel": lambda: KernelRegression2D(bandwidth=args.bandwidth, radius=args.radius),
        "MLP": MLP2D,
        "SIREN": lambda: SIREN2D(omega=args.omega),
    }
    xx, yy = np.meshgrid(np.arange(25, 100.1, 2.5), np.arange(-40, 40.1, 2.5))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(X)
    support = tree.query_ball_point(grid, r=5, return_length=True)
    local_support = tree.query_ball_point(grid, r=args.radius, return_length=True)
    in_rink = ((grid[:, 0] <= 72) | (np.abs(grid[:, 1]) <= 14.5)
               | ((grid[:, 0] - 72)**2 + (np.abs(grid[:, 1]) - 14.5)**2 <= 28**2))
    keep = (support >= 100) & (local_support >= 20) & in_rink
    points = grid[keep]
    probabilities, gradients = {}, {}
    report = {"training_end_season": "2023–24", "n_train": len(X),
              "n_games": int(data.game_id.nunique()), "settings": vars(args),
              "grid_step_ft": 2.5, "support": "At least 100 attempts within 5 ft and 20 within the kernel radius; inside rink",
              "neural_hidden_layers": 2, "neural_width": {"MLP": 128, "SIREN": 128},
              "supported_grid_points": len(points), "gradient_units": "probability per foot",
              "caveats": ["Single seed, fixed training schedule; no claim of tuned predictive superiority.",
                          "Kernel gradient holds its radius-based neighbor set fixed; cutoff boundaries can jump.",
                          "Logistic uses (x, y squared); the other models use (x, y)."],
              "models": {}, "pairwise": []}
    if only:
        previous = json.loads((output / "comparison.json").read_text())
        with np.load(output / "surfaces.npz") as saved:
            for name in factories:
                if name != only:
                    probabilities[name] = saved[f"{name}_probability"]
                    gradients[name] = saved[f"{name}_gradient"]
                    report["models"][name] = previous["models"][name]
    for name, factory in factories.items():
        if only and name != only:
            continue
        print(f"Fitting {name}", flush=True)
        torch.manual_seed(args.seed)
        model = factory()
        if isinstance(model, torch.nn.Module):
            model.fit(X, goal, epochs=args.epochs, lr=args.lr)
            torch.save(model.state_dict(), output / f"{name.lower()}_weights.pt")
        else:
            model.fit(X, goal)
        probabilities[name] = model.predict_proba(points)
        gradients[name] = (model.gradient(points) if isinstance(model, torch.nn.Module)
                           else np.array([model.gradient(q) for q in points]))
        # Finite-difference spot check away from exact recorded coordinates.
        q = np.array([71.3, 7.2])
        eps = .001 if name in ("Logistic", "Kernel") else .02
        numeric = np.array([(model.predict_proba(np.array([q + eps*a]))[0]
                             - model.predict_proba(np.array([q - eps*a]))[0]) / (2*eps)
                            for a in np.eye(2)])
        analytic = model.gradient(q)
        magnitude = np.linalg.norm(gradients[name], axis=1)
        report["models"][name] = {
            "xg_min": float(probabilities[name].min()), "xg_max": float(probabilities[name].max()),
            "median_gradient": float(np.median(magnitude)), "p95_gradient": float(np.quantile(magnitude, .95)),
            "finite_difference_check": {"point": q.tolist(), "step_ft": eps,
                "analytic": analytic.tolist(), "numeric": numeric.tolist(),
                "max_absolute_error": float(np.abs(analytic - numeric).max())}}
        print(f"Finished {name}; gradient check error {np.abs(analytic-numeric).max():.2g}", flush=True)

    for a, b in combinations(factories, 2):
        ga, gb = gradients[a], gradients[b]
        na, nb = np.linalg.norm(ga, axis=1), np.linalg.norm(gb, axis=1)
        nonflat = (na > 1e-5) & (nb > 1e-5)
        cosine = np.sum(ga[nonflat] * gb[nonflat], axis=1) / (na[nonflat] * nb[nonflat])
        report["pairwise"].append({"models": [a, b],
            "probability_rmse_percentage_points": float(100*np.sqrt(np.mean((probabilities[a]-probabilities[b])**2))),
            "median_gradient_angle_degrees": float(np.median(np.degrees(np.arccos(np.clip(cosine,-1,1))))),
            "direction_comparison_points": int(nonflat.sum())})

    np.savez_compressed(output / "surfaces.npz", locations=points, support=support[keep], local_support=local_support[keep],
                        **{f"{k}_probability": v for k,v in probabilities.items()},
                        **{f"{k}_gradient": v for k,v in gradients.items()})
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.subplots_adjust(left=.055, right=.89, top=.83, bottom=.15, wspace=.16, hspace=.22)
    max_p = max(p.max() for p in probabilities.values()) * 100
    max_g = max(np.linalg.norm(g,axis=1).max() for g in gradients.values()) * 100
    labels = {"Kernel": "Kernel", "Logistic": "Logistic",
              "MLP": "MLP", "SIREN": "SIREN"}
    for column, name in enumerate(labels):
        for row, values in enumerate([probabilities[name]*100, np.linalg.norm(gradients[name],axis=1)*100]):
            surface = np.full(len(grid), np.nan)
            surface[keep] = values
            im = axes[row,column].imshow(surface.reshape(xx.shape), origin="lower",
                extent=(23.75,101.25,-41.25,41.25), interpolation="nearest", cmap="Blues" if row==0 else "Purples",
                norm=plt.Normalize(0,max_p) if row==0 else PowerNorm(.5,vmin=0,vmax=max_g))
            rink(axes[row,column])
            if row == 0:
                axes[row,column].set_title(labels[name], fontsize=13, pad=12)
                probability_image = im
            else:
                gradient_image = im
                axes[row,column].set_xlabel("Distance from center ice (ft)", fontsize=9)
            if column == 0:
                axes[row,column].set_ylabel("Across the rink (ft)", fontsize=9)
        # Unit-length arrows display direction only; bottom row supplies magnitudes.
        arrow = (np.isclose(points[:,0] % 10,0) & np.isclose(points[:,1] % 10,0))
        g = gradients[name];length=np.linalg.norm(g,axis=1);arrow &= length>1e-5
        axes[0,column].quiver(points[arrow,0],points[arrow,1],g[arrow,0]/length[arrow],g[arrow,1]/length[arrow],
            angles="xy",scale_units="xy",scale=.22,width=.006,color="#38434a",alpha=.8)
    fig.colorbar(probability_image,cax=fig.add_axes([.915,.525,.012,.28]),label="Chance of scoring (%)")
    fig.colorbar(gradient_image,cax=fig.add_axes([.915,.165,.012,.28]),label="How quickly the chance changes\nPercentage points per foot · square-root colors")
    fig.text(.055,.955,"Where each model sees a scoring chance",fontsize=23,weight="bold")
    fig.text(.055,.915,f"{len(X):,} shots and attempts · 5-on-5, regular season, no overtime · 2009–10 to 2023–24",fontsize=12)
    fig.text(.055,.875,"Top: scoring chance, with arrows toward higher chances. Bottom: how much a small move changes the prediction.",fontsize=11)
    fig.text(.055,.096,"All teams shoot right. Colors use the same scale; arrows show direction only. Gray areas have too few shots or are outside the rink.",fontsize=9)
    fig.text(.055,.067,f"Both networks: two hidden layers of 128 neurons. Trained for {args.epochs} passes. Kernel: {args.radius:g}-ft radius.",fontsize=9)
    fig.text(.055,.038,"Source: NHL play-by-play. Final test seasons are held back. The kernel can jump when a shot enters or leaves its neighborhood.",fontsize=9)
    fig.savefig(output / "scoring_geometry.png", dpi=170, facecolor="white")
    fig.savefig(output / "scoring_geometry.pdf", facecolor="white")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
