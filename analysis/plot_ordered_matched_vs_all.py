"""
plot_ordered_matched_vs_all.py — Pick-and-Place: 'all' vs. 'ordered' on a
cumulative wall-clock axis, with the ordered curve extended (via the
--max-hours-budgeted resume) until it has matched 'all's total hours.

PPO_1 already contains the stitched data for ordered (the resume writes more
event files into the same dir). If future resumes land in new PPO_<n> dirs,
append them to ORDERED_PPO_NS and every segment will be stitched in order,
with wall-clock idle gaps between segments removed (idle time is not training
time).

Up to step DASHED_AFTER_STEPS the ordered curve is solid (= original budget);
beyond that it is dashed (= bonus wall-clock granted by the resume).

Usage (from project root):
    .venv/bin/python analysis/plot_ordered_matched_vs_all.py
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _common import (
    CURRICULUM_COLORS,
    HOUR_FORMATTER,
    LOG_ROOT,
    PHASE_ORDER,
    PHASE_TITLES,
    draw_series,
    load_run,
    phase_duration_hours,
    save,
    setup_style,
    smooth_ema,
    style_reward_axis,
    style_success_axis,
)

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

PHASE = "pick_place"

ALL_PPO_N = 1
ORDERED_PPO_NS = [1]  # append more if future resumes land in new PPO_<n> dirs

# Original pick_place budget (steps). Curve is solid up to here, dashed after.
DASHED_AFTER_STEPS = 7_500_000

SMOOTH = 0.9

METRICS = [
    dict(tag="rollout/ep_rew_mean", title="Reward",            kind="reward"),
    dict(tag="eval/success_rate",   title="Eval success rate", kind="success"),
]

# ════════════════════════════════════════════════════════════════════════════


def _event_files(phase, curriculum, ppo_n):
    d = os.path.join(LOG_ROOT, f"{phase}_{curriculum}", f"PPO_{ppo_n}")
    return sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))


def _load_event(path, metric):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    if metric not in ea.Tags().get("scalars", []):
        return None
    sc = ea.Scalars(metric)
    if not sc:
        return None
    return (
        np.array([s.step      for s in sc]),
        np.array([s.wall_time for s in sc]),
        np.array([s.value     for s in sc]),
    )


def load_stitched(phase, curriculum, ppo_ns, metric):
    """Concatenate every event file across the given PPO_<n> dirs into one
    continuous series. Wall-clock gaps between segments are removed so
    rel_hours measures active training time only."""
    segments = []
    for ppo_n in ppo_ns:
        for f in _event_files(phase, curriculum, ppo_n):
            seg = _load_event(f, metric)
            if seg is not None:
                segments.append(seg)
    if not segments:
        return None

    steps_chunks, hours_chunks, value_chunks = [], [], []
    cum = 0.0
    for steps, walls, values in segments:
        rel = (walls - walls[0]) / 3600.0
        steps_chunks.append(steps)
        hours_chunks.append(rel + cum)
        value_chunks.append(values)
        cum += (walls[-1] - walls[0]) / 3600.0

    return {
        "steps":     np.concatenate(steps_chunks),
        "rel_hours": np.concatenate(hours_chunks),
        "values":    np.concatenate(value_chunks),
    }


def prior_phase_hours(curriculum):
    """Sum of wall-clock durations for all phases preceding PHASE."""
    final_idx = PHASE_ORDER.index(PHASE)
    total = 0.0
    for phase in PHASE_ORDER[:final_idx]:
        # One PPO_<n> per preceding phase; walk them like train_pick_place does.
        n = 1
        while True:
            d = os.path.join(LOG_ROOT, f"{phase}_{curriculum}", f"PPO_{n}")
            if not os.path.isdir(d):
                break
            total += phase_duration_hours(phase, curriculum, n)
            n += 1
    return total


def _draw_solid_dashed(ax, x, steps, values, split_step, smooth, color, label):
    """draw_series-style rendering but the post-split tail is dashed."""
    sm = smooth_ema(values, smooth)
    ax.plot(x, values, color=color, alpha=0.12, linewidth=0.7, zorder=1)

    split = int(np.searchsorted(steps, split_step, side="right"))
    if split <= 0 or split >= len(x):
        ax.plot(x, sm, color=color, linewidth=2.2, label=label, zorder=3)
    else:
        ax.plot(x[:split + 1], sm[:split + 1],
                color=color, linewidth=2.2, label=label, zorder=3)
        ax.plot(x[split:], sm[split:],
                color=color, linewidth=2.2, linestyle=(0, (5, 2)), zorder=3)

    if len(x) > 0:
        ax.plot(
            x[-1], sm[-1],
            marker="o", markersize=4.8,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.9, zorder=4,
        )


def _plot_panel(ax, metric, offsets, ordered_data, all_data):
    handoff_marks = []

    # 'all' — solid curve.
    if all_data is not None:
        c = CURRICULUM_COLORS["all"]
        off = offsets["all"]
        x = all_data["rel_hours"] + off
        draw_series(ax, x, all_data["values"], SMOOTH, c,
                    f"all  (prior: {off:.1f} h)")
        handoff_marks.append((off, c, ":"))

    # 'ordered' — solid up to DASHED_AFTER_STEPS, dashed after.
    if ordered_data is not None:
        c = CURRICULUM_COLORS["ordered"]
        off = offsets["ordered"]
        x = ordered_data["rel_hours"] + off
        _draw_solid_dashed(
            ax, x, ordered_data["steps"], ordered_data["values"],
            DASHED_AFTER_STEPS, SMOOTH, c,
            f"ordered  (prior: {off:.1f} h; dashed after {DASHED_AFTER_STEPS/1e6:.1f}M steps)",
        )
        handoff_marks.append((off, c, ":"))

        # Faint vertical marker at the solid→dashed swap.
        split = int(np.searchsorted(ordered_data["steps"], DASHED_AFTER_STEPS, side="right"))
        if 0 < split < len(x):
            ax.axvline(x[split], color=c, alpha=0.35, linewidth=1.0,
                       linestyle=(0, (5, 2)))

    for x_off, color, ls in handoff_marks:
        ax.axvline(x_off, color=color, alpha=0.28, linewidth=1.0, linestyle=ls)

    ax.xaxis.set_major_formatter(HOUR_FORMATTER)
    ax.set_xlabel("Cumulative real time (hours)  (incl. preceding phases)")

    if metric["kind"] == "reward":
        style_reward_axis(ax)
    else:
        style_success_axis(ax)
    ax.margins(x=0.02)


def main():
    setup_style()

    offsets = {
        "all":     prior_phase_hours("all"),
        "ordered": prior_phase_hours("ordered"),
    }

    print("Preceding-phase durations (hours):")
    for k, v in offsets.items():
        print(f"  {k:>8s}: {v:6.2f} h")

    probe = METRICS[0]["tag"]
    ordered_probe = load_stitched(PHASE, "ordered", ORDERED_PPO_NS, probe)
    all_probe     = load_run(PHASE, "all", ALL_PPO_N, probe)

    if ordered_probe is not None:
        total_h = float(ordered_probe["rel_hours"][-1])
        split = int(np.searchsorted(ordered_probe["steps"], DASHED_AFTER_STEPS, side="right"))
        pre_h  = float(ordered_probe["rel_hours"][min(split, len(ordered_probe["rel_hours"]) - 1)])
        post_h = total_h - pre_h
        print(f"ordered pick_place (stitched across PPO_N={ORDERED_PPO_NS}):")
        print(f"  total active training time     : {total_h:6.2f} h "
              f"({len(ordered_probe['steps'])} points, "
              f"{ordered_probe['steps'][0]:,} → {ordered_probe['steps'][-1]:,} steps)")
        print(f"  up to {DASHED_AFTER_STEPS:,} steps (solid): {pre_h:6.2f} h")
        print(f"  beyond (dashed, from resume)   : {post_h:6.2f} h")
        print(f"  with prior-phase offset        : {offsets['ordered'] + total_h:6.2f} h cumulative")

    if all_probe is not None:
        all_h = float(all_probe["rel_hours"][-1])
        print(f"all pick_place (PPO_{ALL_PPO_N}):")
        print(f"  pick_place wall time           : {all_h:6.2f} h")
        print(f"  with prior-phase offset        : {offsets['all'] + all_h:6.2f} h cumulative")

    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(5.8 * len(METRICS), 4.2),
        constrained_layout=True,
    )
    if len(METRICS) == 1:
        axes = [axes]

    for ax, m in zip(axes, METRICS):
        ax.set_title(m["title"], pad=8)
        ordered_data = load_stitched(PHASE, "ordered", ORDERED_PPO_NS, m["tag"])
        all_data     = load_run(PHASE, "all", ALL_PPO_N, m["tag"])
        _plot_panel(ax, m, offsets, ordered_data, all_data)

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
        f"{PHASE_TITLES[PHASE]} · ordered matched to all by wall-clock "
        f"(EMA α = {SMOOTH:.2f})",
        fontsize=13.5, fontweight="semibold",
    )
    save(fig, f"ordered_matched_vs_all_{PHASE}.png")


if __name__ == "__main__":
    main()
