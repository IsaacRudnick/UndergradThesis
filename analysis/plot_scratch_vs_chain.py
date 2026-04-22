"""
plot_scratch_vs_chain.py — Pick-and-Place: chain-trained vs. from-scratch.

Compares two PPO runs under logs/pick_place_all/:
    PPO_1 — properly trained via the full curriculum chain
            (reach → reach_hold → grasp → pick_place)
    PPO_2 — trained from scratch on pick_place only, no preceding phases.

Produces a 1x2 figure (same style as plot_curricula.py):
    ┌ Reward vs steps │ Success rate vs steps ┐

Usage (from project root):
    .venv/bin/python analysis/plot_scratch_vs_chain.py
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

# Two runs to compare. Colors override the curriculum palette so the two
# runs read as distinct rather than both blue.
RUNS = [
    dict(curriculum=CURRICULUM, ppo_n=1,
         label="chain-trained (reach → hold → grasp → pick_place)",
         color="#0072B2"),
    dict(curriculum=CURRICULUM, ppo_n=2,
         label="from scratch (pick_place only)",
         color="#D55E00"),
]

METRICS = [
    dict(tag="rollout/ep_rew_mean", title="Reward",            kind="reward"),
    dict(tag="eval/success_rate",   title="Eval success rate", kind="success"),
]

SMOOTH = 0.9

# ════════════════════════════════════════════════════════════════════════════


def _plot_panel(ax, runs, metric, smooth):
    for run in runs:
        r = load_run(PHASE, run["curriculum"], run["ppo_n"], metric["tag"])
        if r is None:
            continue
        draw_series(ax, r["steps"], r["values"], smooth, resolve_color(run), run["label"])

    ax.xaxis.set_major_formatter(STEP_FORMATTER)
    ax.set_xlabel("Timesteps")

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
        f"{PHASE_TITLES.get(PHASE, PHASE)} · chain-trained vs. from-scratch"
        f"   (EMA α = {SMOOTH:.2f})",
        fontsize=13.5, fontweight="semibold",
    )
    return save(fig, f"scratch_vs_chain_{PHASE}.png")


if __name__ == "__main__":
    setup_style()
    make_figure()
