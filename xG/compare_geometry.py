"""Fit four location-only xG models and compare probability/gradient geometry.

Run: python compare_geometry.py
Uses seasons through 2023–24; final test seasons are not evaluated.
"""
import argparse
from copy import deepcopy
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


def gradient_check(model):
    """Use double precision and a fine step to resolve rapid sine oscillations."""
    q = np.array([71.3, 7.2])
    eps = 1e-7 if isinstance(model, torch.nn.Module) else .001
    if isinstance(model, torch.nn.Module):
        checked = deepcopy(model).double().eval()
        point = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        derivative, = torch.autograd.grad(torch.sigmoid(checked(point)), point)
        analytic = derivative.detach().numpy()
        def predict(coords):
            with torch.no_grad():
                return torch.sigmoid(checked(torch.tensor(coords, dtype=torch.float64))).numpy()
    else:
        analytic = model.gradient(q)
        predict = model.predict_proba
    numeric = np.array([(predict(np.array([q + eps*a]))[0]
                         - predict(np.array([q - eps*a]))[0]) / (2*eps)
                        for a in np.eye(2)])
    return {"point": q.tolist(), "step_ft": eps,
            "analytic": analytic.tolist(), "numeric": numeric.tolist(),
            "max_absolute_error": float(np.abs(analytic - numeric).max())}


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


def render(folder, destination=None):
    folder = Path(folder)
    destination = Path(destination) if destination else folder
    destination.mkdir(parents=True, exist_ok=True)
    report = json.loads((folder / 'comparison.json').read_text())
    settings = report['settings']
    names = ['Kernel','Logistic','MLP','SIREN']
    with np.load(folder / 'surfaces.npz') as saved:
        points = saved['locations'].copy()
        probabilities = {n:saved[f'{n}_probability'].copy()*100 for n in names}
        gradients = {n:saved[f'{n}_gradient'].copy()*100 for n in names}
    pmax = max(p.max() for p in probabilities.values())
    gmax = max(np.linalg.norm(g,axis=1).max() for g in gradients.values())
    widths = settings.get('siren_hidden_sizes',[32,32,32])
    kernel_label = ("Gaussian · no cutoff" if settings.get("radius") is None else f"radius {settings['radius']:g} ft")
    subtitles = [f"{kernel_label} · h {settings['bandwidth']:g} ft",
                 'features: x and y²', '128 → 128 · tanh',
                 ' → '.join(map(str,widths)) + f" · lr {settings.get('siren_lr',1e-4):g}"]
    fig, axes = plt.subplots(2,4,figsize=(14,9))
    fig.subplots_adjust(left=.055,right=.875,top=.79,bottom=.18,wspace=.13,hspace=.22)
    for col,name in enumerate(names):
        g = gradients[name]
        magnitude = np.linalg.norm(g,axis=1)
        for row,values in enumerate((probabilities[name],magnitude)):
            raster=np.full((33,31),np.nan)
            ix=np.rint((points[:,0]-25)/2.5).astype(int)
            iy=np.rint((points[:,1]+40)/2.5).astype(int)
            raster[iy,ix]=values
            im=axes[row,col].imshow(raster,origin='lower',extent=(23.75,101.25,-41.25,41.25),
                interpolation='nearest',cmap='Blues' if row==0 else 'Purples',
                norm=plt.Normalize(0,pmax) if row==0 else PowerNorm(.5,vmin=0,vmax=gmax))
            rink(axes[row,col])
            if col==0:
                axes[row,col].set_ylabel('Across the rink (ft)',fontsize=9)
            if row==0:
                axes[row,col].set_title(f'{name}\n{subtitles[col]}',fontsize=11,pad=12)
                probability_image=im
            else:
                gradient_image=im
                axes[row,col].set_xlabel('x (ft; goal line at 89)',fontsize=9)
        # One physical threshold across all models; do not amplify almost-flat regions.
        arrow=(np.isclose(points[:,0]%10,0)&np.isclose(points[:,1]%10,0)&(magnitude>=.1))
        axes[0,col].quiver(points[arrow,0],points[arrow,1],g[arrow,0]/magnitude[arrow],g[arrow,1]/magnitude[arrow],
            angles='xy',scale_units='xy',scale=.22,width=.006,color='#38434a',alpha=.85)
    fig.colorbar(probability_image,cax=fig.add_axes([.90,.52,.012,.25]),label='Scoring probability (%)')
    fig.colorbar(gradient_image,cax=fig.add_axes([.90,.19,.012,.25]),label='Gradient magnitude (percentage points / ft)')
    fig.text(.055,.96,'How scoring probability changes across the rink',fontsize=20,weight='bold')
    fig.text(.055,.915,f"{report['n_train']:,} attempts · NHL 5-on-5 regulation · 2009–10 through 2023–24",fontsize=11)
    fig.text(.055,.875,'Top: predicted scoring probability and direction of increase. Bottom: local gradient magnitude.',fontsize=10)
    fig.text(.055,.125,'Shared scales within each row; gradient colors use a square-root scale. Gray = outside rink or insufficient data.',fontsize=9)
    fig.text(.055,.096,'Arrows show direction only and appear at magnitudes ≥ 0.1 percentage points/ft. All teams shoot right.',fontsize=9)
    fig.text(.055,.067,f"Neural models: {settings['epochs']} epochs, seed {settings['seed']}. SIREN: raw x, y, omega {settings['omega']:g}. Held-out seasons not evaluated.",fontsize=9)
    kernel_note = ('Gaussian kernel includes all recorded locations; no hard neighborhood cutoff.' if settings.get('radius') is None else 'Kernel gradients hold neighbors fixed; radius boundaries can jump.')
    fig.text(.055,.038,kernel_note + ' Associations among recorded attempts.',fontsize=9)
    fig.savefig(destination/'scoring_geometry.png',dpi=180,facecolor='white')
    fig.savefig(destination/'scoring_geometry.pdf',facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=.001, help="MLP learning rate")
    parser.add_argument("--siren-lr", type=float, default=.0001)
    parser.add_argument("--siren-hidden-sizes", type=int, nargs="+", default=[32, 32, 32])
    parser.add_argument("--bandwidth", type=float, default=2)
    parser.add_argument("--radius", type=float, default=None, help="Optional legacy hard cutoff; default is full Gaussian")
    parser.add_argument("--omega", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--siren-only", action="store_true",
                        help="Retrain SIREN and reuse the other saved model surfaces.")
    selection.add_argument("--mlp-only", action="store_true",
                           help="Retrain MLP and reuse the other saved model surfaces.")
    selection.add_argument("--kernel-only", action="store_true", help="Refit kernel and reuse saved neural surfaces")
    selection.add_argument("--reuse-neural", action="store_true", help="Re-evaluate saved neural weights on the current grid")
    parser.add_argument("--plot-only", action="store_true", help="Redraw saved results without fitting")
    args = parser.parse_args()
    only = "SIREN" if args.siren_only else "MLP" if args.mlp_only else "Kernel" if args.kernel_only else None
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    output = root / "analysis" / "geometry"
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        render(output)
        return

    data = LoadHockeyShots().load_data()
    data = data.loc[data.game_id.astype("int64") // 1_000_000 <= 2023]
    X = data[["x", "y"]].to_numpy(dtype=float)
    goal = data.goal.to_numpy(dtype=float)
    print(f"Geometry training set: {len(X):,} attempts, {data.game_id.nunique():,} games", flush=True)

    factories = {
        "Logistic": LogisticRegression2D,
        "Kernel": lambda: KernelRegression2D(bandwidth=args.bandwidth, radius=args.radius),
        "MLP": MLP2D,
        "SIREN": lambda: SIREN2D(omega=args.omega, hidden_sizes=args.siren_hidden_sizes),
    }
    xx, yy = np.meshgrid(np.arange(25, 100.1, 2.5), np.arange(-40, 40.1, 2.5))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(X)
    support = tree.query_ball_point(grid, r=5, return_length=True)
    in_rink = ((grid[:, 0] <= 72) | (np.abs(grid[:, 1]) <= 14.5)
               | ((grid[:, 0] - 72)**2 + (np.abs(grid[:, 1]) - 14.5)**2 <= 28**2))
    keep = (support >= 100) & in_rink
    points = grid[keep]
    probabilities, gradients = {}, {}
    report = {"training_end_season": "2023–24", "n_train": len(X),
              "n_games": int(data.game_id.nunique()), "settings": vars(args),
              "grid_step_ft": 2.5, "support": "At least 100 attempts within 5 ft; inside rink. No kernel-neighborhood restriction.",
              "neural_hidden_layers": {"MLP": 2, "SIREN": len(args.siren_hidden_sizes)},
              "neural_width": {"MLP": 128, "SIREN": args.siren_hidden_sizes},
              "supported_grid_points": len(points), "gradient_units": "probability per foot",
              "caveats": ["Single seed, fixed training schedule; no claim of tuned predictive superiority.",
                          ("Kernel uses a full Gaussian without a hard cutoff." if args.radius is None else "Kernel gradient holds its neighbor set fixed; cutoff boundaries can jump."),
                          "Logistic uses (x, y squared); the other models use (x, y)."],
              "models": {}, "pairwise": []}
    if args.reuse_neural:
        previous = json.loads((output / "comparison.json").read_text())
        for setting in ("epochs", "lr", "siren_lr", "siren_hidden_sizes", "omega", "seed"):
            if previous["settings"].get(setting) != getattr(args, setting):
                raise ValueError(f"Saved neural {setting} differs; rerun training.")
    if only:
        previous = json.loads((output / "comparison.json").read_text())
        for setting in ("radius", "bandwidth", "epochs", "lr", "seed", "omega"):
            if only == "Kernel" and setting in ("radius", "bandwidth"):
                continue
            if previous["settings"][setting] != getattr(args, setting):
                raise ValueError(f"Saved {setting} differs; rerun all models to keep the comparison consistent.")
        if only in ("MLP", "Kernel") and (previous["settings"].get("siren_hidden_sizes") != args.siren_hidden_sizes
                               or previous["settings"].get("siren_lr") != args.siren_lr):
            raise ValueError("Saved SIREN settings differ; rerun all models.")
        with np.load(output / "surfaces.npz") as saved:
            if not np.array_equal(saved["locations"], points):
                raise ValueError("Saved surfaces use a different grid; rerun all models.")
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
            if args.reuse_neural:
                model.load_state_dict(torch.load(output / f"{name.lower()}_weights.pt", weights_only=True))
            else:
                model.fit(X, goal, epochs=args.epochs, lr=args.siren_lr if name == "SIREN" else args.lr)
                torch.save(model.state_dict(), output / f"{name.lower()}_weights.pt")
        else:
            model.fit(X, goal)
        probabilities[name] = model.predict_proba(points)
        gradients[name] = (model.gradient(points) if isinstance(model, torch.nn.Module)
                           else np.array([model.gradient(q) for q in points]))
        check = gradient_check(model)
        magnitude = np.linalg.norm(gradients[name], axis=1)
        report["models"][name] = {
            "xg_min": float(probabilities[name].min()), "xg_max": float(probabilities[name].max()),
            "median_gradient": float(np.median(magnitude)), "p95_gradient": float(np.quantile(magnitude, .95)),
            "finite_difference_check": check}
        print(f"Finished {name}; gradient check error {check['max_absolute_error']:.2g}", flush=True)

    for a, b in combinations(factories, 2):
        ga, gb = gradients[a], gradients[b]
        na, nb = np.linalg.norm(ga, axis=1), np.linalg.norm(gb, axis=1)
        nonflat = (na > 1e-5) & (nb > 1e-5)
        cosine = np.sum(ga[nonflat] * gb[nonflat], axis=1) / (na[nonflat] * nb[nonflat])
        report["pairwise"].append({"models": [a, b],
            "gradient_vector_rmse_percentage_points_per_ft": float(100*np.sqrt(np.mean(np.sum((ga-gb)**2,axis=1)))),
            "probability_rmse_percentage_points": float(100*np.sqrt(np.mean((probabilities[a]-probabilities[b])**2))),
            "median_gradient_angle_degrees": float(np.median(np.degrees(np.arccos(np.clip(cosine,-1,1))))),
            "direction_comparison_points": int(nonflat.sum())})

    np.savez_compressed(output / "surfaces.npz", locations=points, support=support[keep],
                        **{f"{k}_probability": v for k,v in probabilities.items()},
                        **{f"{k}_gradient": v for k,v in gradients.items()})
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    render(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
