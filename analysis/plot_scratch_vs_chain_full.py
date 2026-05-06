"""
plot_scratch_vs_chain_full.py — Pick-and-Place: chain-trained vs. full-length
from-scratch run (matched to the total timestep budget of all phases combined).

Compares two PPO runs under logs/pick_place_all/:
    chain   — properly trained via the full curriculum chain
              (reach → reach_hold → grasp → pick_place, ~10M steps).
              Plotted with its x-axis offset by the total timesteps of the
              preceding phases, so the x-axis reads "cumulative timesteps".
    scratch — trained from scratch on pick_place only, but for as many
              timesteps as all phases combined (~14M steps).

The two PPO_<n> directories are auto-detected by final step count: the
scratch run exceeds 10M steps, the chain run does not. This means the
plot keeps working regardless of which PPO_<n> slot each run lands in.

Produces a 1x2 figure (reward | success) on a timesteps-only x-axis.

Usage (from project root):
    .venv/bin/python analysis/plot_scratch_vs_chain_full.py
"""

import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _common import (
    LOG_ROOT,
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
# reach_hold + grasp). The chain run's pick_place curve is shifted right
# by this amount so its x-axis represents cumulative training timesteps.
PRIOR_PHASE_STEPS = 500_000 + 1_000_000 + 5_000_000  # 6,500,000

# Used to distinguish chain (~10M) from scratch (~14M) runs by their final
# step count. Anything above this threshold is the scratch run.
SCRATCH_STEP_THRESHOLD = 12_000_000


def _detect_runs(phase, curriculum):
    """Identify chain and scratch PPO_<n> dirs under logs/<phase>_<curriculum>/.

    Returns (chain_ppo_n, scratch_ppo_n); either may be None if not present.
    """
    d = os.path.join(LOG_ROOT, f"{phase}_{curriculum}")
    candidates = []
    for run_dir in sorted(glob.glob(os.path.join(d, "PPO_*"))):
        try:
            ppo_n = int(os.path.basename(run_dir).split("_")[1])
        except (IndexError, ValueError):
            continue
        events = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
        if not events:
            continue
        ea = EventAccumulator(max(events, key=os.path.getsize),
                              size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        sc = ea.Scalars(tags[0])
        if not sc:
            continue
        candidates.append({
            "ppo_n":      ppo_n,
            "max_step":   sc[-1].step,
            "first_wall": sc[0].wall_time,
        })

    scratch = next(
        (c for c in sorted(candidates, key=lambda c: -c["max_step"])
         if c["max_step"] > SCRATCH_STEP_THRESHOLD),
        None,
    )
    chain = next(
        (c for c in sorted(candidates, key=lambda c: c["first_wall"])
         if c is not scratch),
        None,
    )
    return (chain["ppo_n"] if chain else None,
            scratch["ppo_n"] if scratch else None)


_chain_n, _scratch_n = _detect_runs(PHASE, CURRICULUM)
print(f"  [auto-detect] chain PPO_n={_chain_n}, scratch PPO_n={_scratch_n}")

RUNS = []
if _chain_n is not None:
    RUNS.append(dict(
        curriculum=CURRICULUM, ppo_n=_chain_n,
        label=f"chain-trained (offset +{PRIOR_PHASE_STEPS:,} prior steps)",
        color="#0072B2",
        step_offset=PRIOR_PHASE_STEPS))
if _scratch_n is not None:
    RUNS.append(dict(
        curriculum=CURRICULUM, ppo_n=_scratch_n,
        label="from scratch, full-length (pick_place only)",
        color="#D55E00",
        step_offset=0))

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
