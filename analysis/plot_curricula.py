"""
plot_curricula.py — Per-phase comparison of the three sensor curricula.

For each phase in PHASES, produces a 2x2 figure:
    ┌ Reward vs steps      │ Success rate vs steps       ┐
    │ Reward vs wall-hours │ Success rate vs wall-hours  │

The wall-clock x-axis here is each run's *own* elapsed time (resets to 0 per
phase). For the cumulative-time view of the final task, see plot_cumulative.py.

Usage (from the project root):
    .venv/bin/python analysis/plot_curricula.py
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    HOUR_FORMATTER,
    PHASE_TITLES,
    STEP_FORMATTER,
    draw_series,
    load_run,
    resolve_color,
    save,
    setup_style,
    style_reward_axis,
    style_success_axis,
)

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

PHASES = ["reach", "reach_hold", "grasp", "pick_place"]

# Per-curriculum run selection. Override `ppo_n` to pick a specific attempt.
RUNS = [
    dict(curriculum="all",     ppo_n=1, label="all"),
    dict(curriculum="ordered", ppo_n=1, label="ordered"),
    dict(curriculum="random",  ppo_n=1, label="random"),
]

METRICS = [
    dict(tag="rollout/ep_rew_mean", title="Reward",            kind="reward"),
    dict(tag="eval/success_rate",   title="Eval success rate", kind="success"),
]

SMOOTH_DEFAULT   = 0.9
SMOOTH_PER_PHASE = {"reach": 0.4, "reach_hold": 0.4}

# ════════════════════════════════════════════════════════════════════════════


def _plot_panel(ax, phase, runs, metric, smooth, x_kind):
    plotted = False
    for run in runs:
        r = load_run(phase, run["curriculum"], run["ppo_n"], metric["tag"])
        if r is None:
            continue
        x = r["steps"] if x_kind == "steps" else r["rel_hours"]
        draw_series(ax, x, r["values"], smooth, resolve_color(run), run["label"])
        plotted = True

    if x_kind == "steps":
        ax.xaxis.set_major_formatter(STEP_FORMATTER)
        ax.set_xlabel("Timesteps")
    else:
        ax.xaxis.set_major_formatter(HOUR_FORMATTER)
        ax.set_xlabel("Real time (hours) (this phase)")

    if metric["kind"] == "reward":
        style_reward_axis(ax)
    else:
        style_success_axis(ax)
    ax.margins(x=0.02)
    return plotted


def make_figure(phase, runs, metrics):
    smooth = SMOOTH_PER_PHASE.get(phase, SMOOTH_DEFAULT)

    fig, axes = plt.subplots(
        2, len(metrics),
        figsize=(5.8 * len(metrics), 7.2),
        constrained_layout=True,
        sharey="col",
    )
    if len(metrics) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, m in enumerate(metrics):
        axes[0, col].set_title(m["title"], pad=8)
        _plot_panel(axes[0, col], phase, runs, m, smooth, "steps")
        _plot_panel(axes[1, col], phase, runs, m, smooth, "hours")

    # Single figure-level legend below the axes, handles pulled from any axis.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="outside lower center",
            ncols=len(labels),
            frameon=False,
            fontsize=10.5,
            handlelength=1.8,
            columnspacing=2.2,
        )

    fig.suptitle(
        f"{PHASE_TITLES.get(phase, phase)} · curricula comparison"
        f"   (EMA α = {smooth:.2f})",
        fontsize=13.5, fontweight="semibold",
    )
    return save(fig, f"curricula_{phase}.png")


if __name__ == "__main__":
    setup_style()
    for phase in PHASES:
        make_figure(phase, RUNS, METRICS)
