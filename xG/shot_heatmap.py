import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from scipy.ndimage import gaussian_filter
from normalize_data import LoadHockeyShots
from matplotlib.colors import PowerNorm

def plot_shot_heatmap(data, sigma_ft=2.5):
    """Plot already-aligned x/y coordinates. No coordinate normalization."""

    xy = data[["x", "y"]].to_numpy(dtype=float)
    assert len(xy) > 0, "No shots to plot."
    assert np.isfinite(xy).all(), "Coordinates contain missing/infinite values."

    # One-foot cells; values are raw attempt counts.
    counts, _, _ = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=[
            np.arange(-100, 101, 1),
            np.arange(-42.5, 43, 1),
        ],
    )

    assert counts.sum() == len(xy), "Some coordinates fall outside the rink grid."

    # Optional display smoothing; does not modify input coordinates.
    heatmap = (
        gaussian_filter(counts, sigma=sigma_ft, mode="constant")
        if sigma_ft > 0
        else counts
    )

    fig, ax = plt.subplots(figsize=(8.5, 8))

    image = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[-100, 100, -42.5, 42.5],
        cmap="Blues",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=heatmap.max()),
        aspect="equal",
        interpolation="nearest",
    )

    ax.add_patch(FancyBboxPatch(
        (-100, -42.5), 200, 85,
        boxstyle="round,pad=0,rounding_size=28",
        fill=False, edgecolor="black", linewidth=1.2,
    ))

    ax.axvline(0, color="firebrick", linewidth=1, alpha=0.6)

    for x in [-25, 25]:
        ax.axvline(x, color="steelblue", linewidth=1, alpha=0.7)

    ax.add_patch(Circle(
        (0, 0), 15,
        fill=False, edgecolor="steelblue", alpha=0.6,
    ))

    for x in [-69, 69]:
        for y in [-22, 22]:
            ax.add_patch(Circle(
                (x, y), 15,
                fill=False, edgecolor="gray", linewidth=0.8,
            ))

    for x in [-89, 89]:
        ax.plot(
            [x, x], [-37, 37],
            color="firebrick", linewidth=0.8, alpha=0.6,
        )
        ax.add_patch(Rectangle(
            (x if x > 0 else x - 4, -3), 4, 6,
            fill=False, edgecolor="black", linewidth=1.2,
        ))

    ax.set(
        xlim=(-100, 100),
        ylim=(-42.5, 42.5),
        xlabel="Aligned x coordinate (feet)",
        ylabel="Aligned y coordinate (feet)",
        title=f"Shot locations · {len(xy):,} attempts",
    )
    ax.spines[:].set_visible(False)
    ax.set_xlim(25, 100)

    label = (
        "Smoothed attempts per square foot"
        if sigma_ft > 0
        else "Attempts per square foot"
    )
    fig.colorbar(image, ax=ax, shrink=0.8, label=label)

    fig.text(
        0.5, 0.02,
        f"1-foot cells · Gaussian smoothing σ = {sigma_ft:g} feet",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])


    return fig, ax


if __name__ == '__main__':
    data_loader = LoadHockeyShots()
    fig, ax = plot_shot_heatmap(data_loader.load_data())
    plt.show(block=True)