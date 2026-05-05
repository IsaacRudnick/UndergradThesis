"""Shared training-time helpers used by the per-phase train scripts."""

from stable_baselines3.common.callbacks import BaseCallback

try:
    from tqdm.rich import tqdm, FractionColumn, RateColumn
    from rich.progress import (
        BarColumn,
        ProgressColumn,
        TimeElapsedColumn,
    )
    from rich.text import Text
except ImportError:
    tqdm = None


def _format_hms(seconds: float) -> str:
    if seconds == float("inf"):
        return "?:??:??"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if tqdm is not None:
    class _WallTimeRemainingColumn(ProgressColumn):
        """Replacement for rich's TimeRemainingColumn.

        ETA is computed from average rate over the whole run
        (elapsed * remaining / done) so pauses don't skew it.  When a
        cumulative wall-time budget is given, ETA is capped at the remaining
        budget unless the timestep estimate finishes first.
        """

        def __init__(self, max_seconds=None, prior_seconds=0.0):
            super().__init__()
            self.max_seconds = max_seconds
            self.prior_seconds = prior_seconds

        def render(self, task) -> "Text":
            elapsed = task.elapsed
            done = task.completed
            total = task.total
            if elapsed and done and total:
                ts_eta = elapsed * (total - done) / done
            else:
                ts_eta = float("inf")
            if self.max_seconds is not None and elapsed is not None:
                budget_eta = max(
                    0.0, self.max_seconds - self.prior_seconds - elapsed
                )
                eta = min(ts_eta, budget_eta)
            else:
                eta = ts_eta
            return Text(_format_hms(eta), style="progress.remaining")


class WallTimeProgressBarCallback(BaseCallback):
    """Drop-in replacement for SB3's ProgressBarCallback with a fixed ETA.

    The default tqdm.rich ETA divides remaining steps by recent it/s, so any
    pause (eval rollouts, GPU contention, the user stepping away) skews the
    estimate.  This callback instead computes
        ETA = elapsed_run_time * (total - done) / done
    which is just the average rate over the whole run — pauses already baked in.

    When ``max_hours`` is set, ETA is additionally capped at the remaining
    cumulative budget (``max_hours - prior_hours - elapsed``).  If the
    timestep-based estimate finishes before the budget runs out, that one wins
    instead.

    Visually identical to ``stable_baselines3.common.callbacks.ProgressBarCallback``
    (also tqdm.rich-backed) — only the time-remaining column is swapped out.
    """

    def __init__(self, max_hours: float | None = None,
                 prior_hours: float = 0.0):
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "WallTimeProgressBarCallback requires tqdm and rich — "
                "install them with `pip install tqdm rich`."
            )
        self.max_seconds = max_hours * 3600.0 if max_hours is not None else None
        self.prior_seconds = prior_hours * 3600.0
        self.pbar = None

    def _on_training_start(self) -> None:
        total = self.locals["total_timesteps"] - self.model.num_timesteps
        # Mirror tqdm.rich's default columns, but swap TimeRemainingColumn
        # for our wall-time-based equivalent.
        unit_scale = False
        unit_divisor = 1000
        progress = (
            "[progress.description]{task.description}"
            "[progress.percentage]{task.percentage:>4.0f}%",
            BarColumn(bar_width=None),
            FractionColumn(unit_scale=unit_scale, unit_divisor=unit_divisor),
            "[",
            TimeElapsedColumn(),
            "<",
            _WallTimeRemainingColumn(
                max_seconds=self.max_seconds,
                prior_seconds=self.prior_seconds,
            ),
            ",",
            RateColumn(unit="it", unit_scale=unit_scale, unit_divisor=unit_divisor),
            "]",
        )
        self.pbar = tqdm(total=total, progress=progress)

    def _on_step(self) -> bool:
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()
