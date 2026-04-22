"""
plot_cumulative.py — Final-task comparison against *cumulative* wall-clock time.

For the chosen FINAL_PHASE, each curriculum's curve is offset on the x-axis by
the total wall-clock hours that curriculum spent on all preceding phases. So if
'all' took 3 h across phases 1A+1B+2, its pick_place curve starts at x = 3 h.

This makes the x-axis "total training wall-clock hours so far", which captures
the speed differences between curricula (some run faster per timestep).

Usage (from project root):
    .venv/bin/python analysis/plot_cumulative.py
"""

import matplotlib.pyplot as plt

from _common import (
    HOUR_FORMATTER,
    PHASE_ORDER,
    PHASE_TITLES,
    draw_series,
    load_run,
    phase_duration_hours,
    resolve_color,
    save,
    setup_style,
    style_reward_axis,
    style_success_axis,
)

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

FINAL_PHASE = "pick_place"

# Each RUN configures one curriculum across *all* phases, because we need
# the preceding phases' durations to compute the offset.
RUNS = [
    dict(
        curriculum="all", label="all",
        phases={"reach": 1, "reach_hold": 1, "grasp": 1, "pick_place": 1},
    ),
    dict(
        curriculum="ordered", label="ordered",
        phases={"reach": 1, "reach_hold": 1, "grasp": 1, "pick_place": 1},
    ),
    dict(
        curriculum="random", label="random",
        phases={"reach": 1, "reach_hold": 1, "grasp": 1, "pick_place": 1},
    ),
]

METRICS = [
    dict(tag="rollout/ep_rew_mean", title="Reward",            kind="reward"),
    dict(tag="eval/success_rate",   title="Eval success rate", kind="success"),
]

SMOOTH = 0.9

# ════════════════════════════════════════════════════════════════════════════


def prior_phase_hours(run):
    """Sum of wall-clock durations for this curriculum across phases before FINAL_PHASE."""
    final_idx = PHASE_ORDER.index(FINAL_PHASE)
    total = 0.0
    for phase in PHASE_ORDER[:final_idx]:
        total += phase_duration_hours(phase, run["curriculum"], run["phases"].get(phase, 1))
    return total


def _plot_panel(ax, runs, metric, smooth, offsets):
    handoff_marks = []
    for run in runs:
        r = load_run(FINAL_PHASE, run["curriculum"], run["phases"][FINAL_PHASE], metric["tag"])
        if r is None:
            continue
        offset = offsets[run["curriculum"]]
        x = r["rel_hours"] + offset
        color = resolve_color(run)
        draw_series(ax, x, r["values"], smooth, color,
                    f"{run['label']}  (prior: {offset:.1f} h)")
        handoff_marks.append((offset, color))

    for x_off, color in handoff_marks:
        ax.axvline(x_off, color=color, alpha=0.28, linewidth=1.0, linestyle=":")

    ax.xaxis.set_major_formatter(HOUR_FORMATTER)
    ax.set_xlabel("Cumulative real time (hours)  (incl. preceding phases)")

    if metric["kind"] == "reward":
        style_reward_axis(ax)
    else:
        style_success_axis(ax)
    ax.margins(x=0.02)


def main():
    setup_style()

    offsets = {run["curriculum"]: prior_phase_hours(run) for run in RUNS}
    print("Preceding-phase durations (hours):")
    for k, v in offsets.items():
        print(f"  {k:>8s}: {v:6.2f} h")

    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(6.6 * len(METRICS), 4.8),
        constrained_layout=True,
    )
    if len(METRICS) == 1:
        axes = [axes]
    for ax, m in zip(axes, METRICS):
        ax.set_title(m["title"], pad=8)
        _plot_panel(ax, RUNS, m, SMOOTH, offsets)

    handles, labels = axes[0].get_legend_handles_labels()
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
        f"{PHASE_TITLES.get(FINAL_PHASE, FINAL_PHASE)} · cumulative training time",
        fontsize=13.5, fontweight="semibold",
    )
    save(fig, f"cumulative_{FINAL_PHASE}.png")


if __name__ == "__main__":
    main()
