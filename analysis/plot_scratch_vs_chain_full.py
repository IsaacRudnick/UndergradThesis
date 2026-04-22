"""
plot_scratch_vs_chain_full.py — Pick-and-Place: chain-trained vs. full-length
from-scratch run (matched to the total timestep budget of all phases combined).

Compares two PPO runs under logs/pick_place_all/:
    PPO_1 — properly trained via the full curriculum chain
            (reach → reach_hold → grasp → pick_place).
            Plotted with its x-axis offset by the total timesteps of the
            preceding phases, so the x-axis reads "cumulative timesteps".
    PPO_3 — trained from scratch on pick_place only, but for as many
            timesteps as all phases combined.

Produces a 1x2 figure (reward | success) on a timesteps-only x-axis.

Usage (from project root):
    .venv/bin/python analysis/plot_scratch_vs_chain_full.py
"""

import matplotlib.pyplot as plt

from _common import (
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

PHASE = "pick_place"
CURRICULUM = "all"

# Preceding-phase timestep budgets for the chain-trained run (reach +
# reach_hold + grasp). PPO_1's pick_place curve is shifted right by this
# amount so its x-axis represents cumulative training timesteps.
PRIOR_PHASE_STEPS = 500_000 + 1_000_000 + 5_000_000  # 6,500,000

RUNS = [
    dict(curriculum=CURRICULUM, ppo_n=1,
         label=f"chain-trained (offset +{PRIOR_PHASE_STEPS:,} prior steps)",
         color="#0072B2",
         step_offset=PRIOR_PHASE_STEPS),
    dict(curriculum=CURRICULUM, ppo_n=3,
         label="from scratch, full-length (pick_place only)",
         color="#D55E00",
         step_offset=0),
]

METRICS = [
    dict(tag="rollout/ep_rew_mean", title="Reward",            kind="reward"),
    dict(tag="eval/success_rate",   title="Eval success rate", kind="success"),
]

SMOOTH = 0.9

# ════════════════════════════════════════════════════════════════════════════


def _plot_panel(ax, runs, metric, smooth):
    handoff_marks = []
    for run in runs:
        r = load_run(PHASE, run["curriculum"], run["ppo_n"], metric["tag"])
        if r is None:
            continue
        offset = run.get("step_offset", 0)
        x = r["steps"] + offset
        color = resolve_color(run)
        draw_series(ax, x, r["values"], smooth, color, run["label"])
        if offset > 0:
            handoff_marks.append((offset, color))

    for x_off, color in handoff_marks:
        ax.axvline(x_off, color=color, alpha=0.28, linewidth=1.0, linestyle=":")

    ax.xaxis.set_major_formatter(STEP_FORMATTER)
    ax.set_xlabel("Cumulative timesteps  (incl. preceding phases)")

    if metric["kind"] == "reward":
        style_reward_axis(ax)
    else:
        style_success_axis(ax)
    ax.margins(x=0.02)


def make_figure():
    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(5.8 * len(METRICS), 4.2),
        constrained_layout=True,
    )
    if len(METRICS) == 1:
        axes = [axes]

    for ax, m in zip(axes, METRICS):
        ax.set_title(m["title"], pad=8)
        _plot_panel(ax, RUNS, m, SMOOTH)

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
        f"{PHASE_TITLES.get(PHASE, PHASE)} · chain-trained vs. full-length from-scratch"
        f"   (EMA α = {SMOOTH:.2f})",
        fontsize=13.5, fontweight="semibold",
    )
    return save(fig, f"scratch_vs_chain_full_{PHASE}.png")


if __name__ == "__main__":
    setup_style()
    make_figure()
