"""Print total wall-clock training hours for a curriculum across all phases.

Used by the Makefile to set --max-hours for the 'ordered' pick_place resume
so it matches the total wall-clock time the 'all' curriculum spent across
reach + reach_hold + grasp + pick_place (including any resumes).

Scans logs/<phase>_<curriculum>/PPO_*/events.out.tfevents.* and for each
event file sums (last_scalar.wall_time - first_scalar.wall_time). Idle gaps
between runs are excluded because each event file measures only its own run.

Usage (from project root):
    .venv/bin/python analysis/compute_match_hours.py           # defaults to 'all'
    .venv/bin/python analysis/compute_match_hours.py ordered

Prints a single float (hours, 4 decimals) to stdout. All progress messages
go to stderr so the value is safe to capture in a shell subshell, e.g.:
    HOURS=$(.venv/bin/python analysis/compute_match_hours.py all)
"""

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PHASES = ("reach", "reach_hold", "grasp", "pick_place")

# The scratch-full baseline (plot_scratch_vs_chain_full.py) trains for ~14M
# steps and also lives under logs/pick_place_<curriculum>/; the chain-trained
# run is ~10M. Skip any pick_place run whose final step exceeds this threshold
# so the scratch baseline doesn't double-count toward the curriculum's total.
SCRATCH_STEP_THRESHOLD = 10_000_000


def _event_stats(event_path):
    """Return (hours_spanned, final_step) for a TB event file, or (0, 0) if unreadable."""
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return 0.0, 0
    sc = ea.Scalars(tags[0])
    if len(sc) < 2:
        return 0.0, 0
    hours = (sc[-1].wall_time - sc[0].wall_time) / 3600.0
    return hours, sc[-1].step


def curriculum_hours(curriculum, phases=PHASES):
    total = 0.0
    for phase in phases:
        phase_dir = os.path.join("logs", f"{phase}_{curriculum}")
        if not os.path.isdir(phase_dir):
            continue
        for run_dir in sorted(glob.glob(os.path.join(phase_dir, "PPO_*"))):
            events = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
            if not events:
                continue
            hours, final_step = _event_stats(max(events, key=os.path.getsize))
            if phase == "pick_place" and final_step > SCRATCH_STEP_THRESHOLD:
                print(f"  {phase:>10s} {os.path.basename(run_dir):>8s}: "
                      f"{hours:6.2f} h  [SKIP — scratch baseline, {final_step:,} steps]",
                      file=sys.stderr)
                continue
            print(f"  {phase:>10s} {os.path.basename(run_dir):>8s}: {hours:6.2f} h",
                  file=sys.stderr)
            total += hours
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("curriculum", nargs="?", default="all",
                   choices=["all", "ordered", "random"])
    args = p.parse_args()
    total = curriculum_hours(args.curriculum)
    print(f"  TOTAL {args.curriculum:>12s}: {total:6.2f} h", file=sys.stderr)
    print(f"{total:.4f}")


if __name__ == "__main__":
    main()
