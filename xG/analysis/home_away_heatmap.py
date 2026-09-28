"""Rank home/away spatial distributions and plot the largest observed gap."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parent
groups = defaultdict(lambda: [[], []])
names = {}
game_ids = defaultdict(lambda: [set(), set()])
with (ROOT.parent / "data" / "shots" / "shots_20252026.csv").open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["is_shootout"] == "1":
            continue
        team = r["event_owner_team_id"]
        home = team == r["home_team_id"]
        assert home or team == r["away_team_id"]
        assert r["home_team_defending_side"] in ("left", "right")
        names[team] = r["home_team_abbrev"] if home else r["away_team_abbrev"]
        # Rotate 180 degrees when attacking left; preserve lateral orientation.
        sign = -1 if (r["home_team_defending_side"] == "right") == home else 1
        groups[team][0 if home else 1].append([sign * float(r["x_coord"]), sign * float(r["y_coord"])])
        game_ids[team][0 if home else 1].add(r["game_id"])

xedges = np.arange(-100, 101, 5)
yedges = np.arange(-42.5, 43, 5)
surfaces = {}
ranking = []
for team, samples in groups.items():
    density = []
    for sample in samples:
        histogram = np.histogram2d(*np.array(sample).T, bins=[xedges, yedges])[0]
        assert histogram.sum() == len(sample)
        smooth = gaussian_filter(histogram, sigma=1, mode="constant")
        density.append(smooth / smooth.sum())
    surfaces[team] = density
    score = float(jensenshannon(density[0].ravel(), density[1].ravel(), base=2) ** 2)
    ranking.append({"team_id": team, "team": names[team], "js_divergence_bits": score,
                    "home_attempts": len(samples[0]), "away_attempts": len(samples[1])})
ranking.sort(key=lambda r: r["js_divergence_bits"], reverse=True)
winner = ranking[0]
team = winner["team_id"]
home, away = [100 * h for h in surfaces[team]]
difference = home - away
assert np.isclose(home.sum(), 100) and np.isclose(away.sum(), 100)
assert np.isclose(difference.sum(), 0)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(3, 1, figsize=(11, 11.5))
fig.subplots_adjust(left=.09, right=.82, top=.855, bottom=.15, hspace=.37)
full_name = "Seattle Kraken" if names[team] == "SEA" else names[team]
fig.text(.09, .965, f"{full_name}: largest home–away location gap", fontsize=21, weight="bold")
fig.text(.09, .933, "2025–26 · All shot attempts, excluding shootouts · Ranked across 32 teams", fontsize=11)
fig.text(.09, .908, "Each map is normalized to 100 attempts. Both teams’ attacking directions are rotated to the right.", fontsize=10, color="#444444")

def rink(ax):
    boundary = FancyBboxPatch((-100, -42.5), 200, 85,
                             boxstyle="round,pad=0,rounding_size=28", fill=False,
                             edgecolor="#4b5563", linewidth=.9, zorder=4)
    ax.add_patch(boundary)
    for x in (-25, 25):
        ax.axvline(x, color="#788696", lw=.7, alpha=.65)
    ax.axvline(0, color="#788696", lw=.7, alpha=.65)
    for x in (-89, 89):
        ax.plot([x, x], [-37, 37], color="#788696", lw=.65, alpha=.6)
        ax.add_patch(Rectangle((x if x > 0 else x-4, -3), 4, 6, fill=False,
                               edgecolor="#252525", linewidth=1.1, zorder=5))
    for x in (-69, 69):
        for y in (-22, 22):
            ax.add_patch(Circle((x, y), 15, fill=False, color="#788696", lw=.6, alpha=.6))
    ax.set(xlim=(-100, 100), ylim=(-42.5, 42.5), aspect="equal", yticks=[-40, 0, 40])
    ax.set_ylabel("Lateral position (ft)", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=8)

vmax = max(home.max(), away.max())
limit = np.abs(difference).max()
for ax, data, title in zip(axes, [home, away, difference], [
    f"Home · {winner['home_attempts']:,} attempts / {len(game_ids[team][0])} games",
    f"Away · {winner['away_attempts']:,} attempts / {len(game_ids[team][1])} games",
    "Home minus away · Orange = more at home; purple = more away",
]):
    diff = ax is axes[2]
    im = ax.imshow(data.T, origin="lower", extent=[-100, 100, -42.5, 42.5],
                   cmap="PuOr_r" if diff else "YlOrRd", interpolation="bilinear",
                   vmin=-limit if diff else 0, vmax=limit if diff else vmax)
    rink(ax)
    ax.set_title(title, loc="left", fontsize=12, pad=9)
    if not diff:
        ax.set_xticklabels([])
    box = ax.get_position()
    cax = fig.add_axes([.85, box.y0, .016, box.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.outline.set_visible(False)
    cb.set_label("Percentage-point difference" if diff else "Attempts per 100", fontsize=9)
axes[-1].set_xlabel("Rink position (ft) · Attacking goal at +89 →", fontsize=10, labelpad=8)
fig.text(.09, .087, f"Selection: largest Jensen–Shannon divergence on the displayed distributions ({winner['js_divergence_bits']:.4f} bits).", fontsize=9)
fig.text(.09, .066, "5 × 5 ft bins; Gaussian smoothing σ = 5 ft. Color values are per bin. Source: cached NHL play-by-play.", fontsize=9)
fig.text(.09, .045, "Descriptive difference, not proof of recording bias. Tactics, opponents and game situations also vary.", fontsize=9)
fig.savefig(ROOT / "largest_home_away_heatmap.png", dpi=180, facecolor="white")
fig.savefig(ROOT / "largest_home_away_heatmap.pdf", facecolor="white")
summary = {"method": "Jensen–Shannon divergence (bits), 5 ft bins, Gaussian sigma 5 ft; full-rink normalized densities",
           "filters": "Regular season and playoffs; all four attempt types; no shootouts",
           "coordinate_transform": "Rotate x and y by 180 degrees when shooting team attacks left; CSV unchanged",
           "caveat": "Blocked-shot coordinates are API event locations, not necessarily release locations. No causal bias claim.",
           "ranking": ranking}
(ROOT / "home_away_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(winner, indent=2))
for label, data in [("home excess", difference), ("away excess", -difference)]:
    i, j = np.unravel_index(np.argmax(data), data.shape)
    print(label, "near", (float(xedges[i]+2.5), float(yedges[j]+2.5)), "delta", float(data[i,j]))
