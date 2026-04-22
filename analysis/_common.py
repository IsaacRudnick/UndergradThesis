"""Shared helpers for analysis/ plotting scripts."""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Canonical phase order — index in this list is the training order.
PHASE_ORDER = ["reach", "reach_hold", "grasp", "pick_place"]

PHASE_TITLES = {
    "reach":      "Phase 1A · Reach",
    "reach_hold": "Phase 1B · Reach & Hold",
    "grasp":      "Phase 2 · Grasp",
    "pick_place": "Phase 3 · Pick-and-Place",
}

# Colorblind-safe palette (Okabe–Ito), consistent per curriculum everywhere.
CURRICULUM_COLORS = {
    "all":     "#0072B2",
    "ordered": "#E69F00",
    "random":  "#009E73",
}


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.45,
        "grid.color": "#BBBBBB",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 170,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
    })


def _find_event_file(phase, curriculum, ppo_n):
    log_dir = os.path.join(LOG_ROOT, f"{phase}_{curriculum}", f"PPO_{ppo_n}")
    matches = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not matches:
        print(f"  [WARN] No event file in {log_dir}")
        return None
    return max(matches, key=os.path.getsize)


def load_run(phase, curriculum, ppo_n, metric):
    """Return {steps, values, rel_hours, first_wall, last_wall} or None."""
    path = _find_event_file(phase, curriculum, ppo_n)
    if path is None:
        return None
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    if metric not in ea.Tags().get("scalars", []):
        print(f"  [WARN] '{metric}' missing in {path}")
        return None
    sc = ea.Scalars(metric)
    steps  = np.array([s.step      for s in sc])
    walls  = np.array([s.wall_time for s in sc])
    values = np.array([s.value     for s in sc])
    return {
        "steps": steps,
        "values": values,
        "rel_hours": (walls - walls[0]) / 3600.0,
        "first_wall": float(walls[0]),
        "last_wall":  float(walls[-1]),
    }


def phase_duration_hours(phase, curriculum, ppo_n, probe_metric="rollout/ep_rew_mean"):
    """Wall-clock hours spanned by a phase's event file (first scalar → last scalar)."""
    run = load_run(phase, curriculum, ppo_n, probe_metric)
    if run is None:
        return 0.0
    return (run["last_wall"] - run["first_wall"]) / 3600.0


def smooth_ema(values, weight):
    """TensorBoard-style EMA."""
    v = np.asarray(values, dtype=float)
    if weight == 0 or len(v) == 0:
        return v.copy()
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = weight * out[i - 1] + (1 - weight) * v[i]
    return out


def _fmt_steps(x, _pos):
    if abs(x) >= 1e6:
        return f"{x/1e6:.4g}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.4g}K"
    return f"{int(x)}"


def _fmt_hours(x, _pos):
    return f"{int(x)}"


def _fmt_percent(x, _pos):
    return f"{x * 100:.0f}%"


STEP_FORMATTER    = mticker.FuncFormatter(_fmt_steps)
HOUR_FORMATTER    = mticker.FuncFormatter(_fmt_hours)
PERCENT_FORMATTER = mticker.FuncFormatter(_fmt_percent)


def style_reward_axis(ax):
    """Reward values are arbitrary — hide the numeric tick labels on that axis."""
    ax.tick_params(axis="y", labelleft=False, length=0)


def style_success_axis(ax):
    """Format success-rate axis as percentages with fixed [0, 100%] range."""
    ax.yaxis.set_major_formatter(PERCENT_FORMATTER)
    ax.set_ylim(-0.03, 1.03)


def draw_series(ax, x, values, smooth, color, label, end_marker=True):
    """Faint raw trace + bold smoothed trace + subtle end dot."""
    sm = smooth_ema(values, smooth)
    ax.plot(x, values, color=color, alpha=0.12, linewidth=0.7,  zorder=1)
    ax.plot(x, sm,     color=color, alpha=1.0,  linewidth=2.2, label=label, zorder=3)
    if end_marker and len(x) > 0:
        ax.plot(
            x[-1], sm[-1],
            marker="o", markersize=4.8,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.9, zorder=4,
        )


def resolve_color(run):
    return run.get("color") or CURRICULUM_COLORS.get(run["curriculum"], "#444444")


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    return out
