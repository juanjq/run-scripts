"""
run_analyzer.py
---------------
Drop this file in the same folder as your notebook.

Usage:
    from run_analyzer import RunAnalyzer, PR_TARGETS
    analyzer = RunAnalyzer(runs)
    analyzer.flag_by_date("2025-03-15")
    analyzer.personal_records(PR_TARGETS)   # full target list
    analyzer.best_for("5 km")
    analyzer.best_for("+VK")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PR target definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistanceTarget:
    """A flat-distance PR target. distance_km is the target in kilometres."""
    label: str
    distance_km: float
    kind: Literal["distance"] = "distance"


@dataclass(frozen=True)
class ElevationTarget:
    """
    An elevation PR target.

    elev_m    : cumulative elevation to accumulate inside the sliding window
    direction : "gain" (uphill) or "loss" (downhill)
    Metric    : fastest time to accumulate elev_m in the given direction.
    """
    label: str
    elev_m: float
    direction: Literal["gain", "loss"]
    kind: Literal["elevation"] = "elevation"


# Convenience constructors ─────────────────────────────────────────────────
def dist(label: str, km: float) -> DistanceTarget:
    return DistanceTarget(label=label, distance_km=km)

def climb(label: str, m: float) -> ElevationTarget:
    return ElevationTarget(label=label, elev_m=m, direction="gain")

def descent(label: str, m: float) -> ElevationTarget:
    return ElevationTarget(label=label, elev_m=m, direction="loss")


# ---------------------------------------------------------------------------
# Full default target list  ── edit freely in your notebook via PR_TARGETS
# ---------------------------------------------------------------------------

PR_TARGETS: list[DistanceTarget | ElevationTarget] = [
    # ── Flat distances ──────────────────────────────────────────────────────
    dist("400m",        0.400),
    dist("½ mile",      0.80467),
    dist("1 km",        1.0),
    dist("1 mile",      1.60934),
    dist("2 miles",     3.21869),
    dist("5 km",        5.0),
    dist("10 km",       10.0),
    dist("15 km",       15.0),
    dist("10 miles",    16.0934),
    dist("20 km",       20.0),
    dist("½ marathon",  21.0975),
    dist("30 km",       30.0),
    dist("marathon",    42.195),
    dist("50 km",       50.0),
    # ── Climbing: fastest time to accumulate N metres of gain ───────────────
    climb("+100 m",    100),
    climb("+300 m",    300),
    climb("+500 m",    500),
    climb("+VK",      1000),   # Vertical Kilometre
    climb("+1500 m",  1500),
    climb("+2VK",     2000),
    # ── Descending: fastest time to accumulate N metres of loss ────────────
    descent("-100 m",  100),
    descent("-300 m",  300),
    descent("-500 m",  500),
    descent("-VK",    1000),
    descent("-1500 m",1500),
    descent("-2VK",   2000),
]


# ---------------------------------------------------------------------------
# Sport type filter
# ---------------------------------------------------------------------------

RUNNING_SPORT_TYPES: set[str] = {
    "running",
    "run",
    "trail_running",
    "track_running",
    "road_running",
    "treadmill",
    "indoor_running",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_time(secs: float) -> str:
    """Seconds → H:MM:SS or M:SS."""
    if secs is None or not np.isfinite(secs):
        return "—"
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_pace(secs_per_km: float) -> str:
    """Seconds/km → M:SS /km."""
    if secs_per_km is None or not np.isfinite(secs_per_km) or secs_per_km <= 0:
        return "—"
    m, s = divmod(int(round(secs_per_km)), 60)
    return f"{m}:{s:02d} /km"


def fmt_mh(secs: float, elev_m: float) -> str:
    """Compute and format m/h climb/descent rate."""
    if secs is None or not np.isfinite(secs) or secs <= 0:
        return "—"
    rate = elev_m / (secs / 3600.0)
    return f"{rate:.0f} m/h"


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RunAnalyzer:
    """
    Analyse a list of running DataFrames (one per activity).

    Each DataFrame must have at minimum:
        - timestamp  : datetime (UTC or tz-aware)
        - distance   : cumulative metres (Coros/Garmin FIT format)

    For elevation PRs also needed (one of):
        - enhanced_altitude   (preferred)
        - altitude

    Optional extras used in the summary table:
        - heart_rate, cadence, power, enhanced_speed / speed
    """

    def __init__(self, runs: list[pd.DataFrame]):
        self.runs: list[dict] = [self._preprocess(df, i) for i, df in enumerate(runs)]
        self.flagged: set[int] = set()

    # ──────────────────────────────────────────────────────────────────
    # Pre-processing
    # ──────────────────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame, idx: int) -> dict:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = (
            df.dropna(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True)
        )

        # ── distance ──────────────────────────────────────────────────
        dist_s = pd.to_numeric(df.get("distance", pd.Series(dtype=float)), errors="coerce")
        total_km = dist_s.max() / 1000.0 if dist_s.notna().any() else 0.0

        # ── elapsed time ──────────────────────────────────────────────
        elapsed_s = (
            df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        ).total_seconds()

        # ── altitude ──────────────────────────────────────────────────
        alt_col = (
            "enhanced_altitude" if "enhanced_altitude" in df.columns else
            "altitude"          if "altitude"          in df.columns else
            None
        )
        elev_gain = np.nan
        elev_loss = np.nan
        if alt_col:
            alt = pd.to_numeric(df[alt_col], errors="coerce").to_numpy(dtype=float)
            diffs = np.diff(np.nan_to_num(alt, nan=0.0))
            elev_gain = float(diffs[diffs > 0].sum())
            elev_loss = float(abs(diffs[diffs < 0].sum()))

        # ── scalar helpers ────────────────────────────────────────────
        def _mean(col: str) -> float:
            return (
                pd.to_numeric(df[col], errors="coerce").mean()
                if col in df.columns else np.nan
            )

        return {
            "id":          idx,
            "date":        df["timestamp"].iloc[0].date(),
            "df":          df,
            "alt_col":     alt_col,
            "total_km":    round(total_km, 3),
            "elapsed_s":   elapsed_s,
            "avg_pace":    elapsed_s / total_km if total_km > 0 else np.nan,
            "avg_hr":      _mean("heart_rate"),
            "avg_cadence": _mean("cadence"),
            "elev_gain_m": elev_gain,
            "elev_loss_m": elev_loss,
        }

    # ──────────────────────────────────────────────────────────────────
    # Flagging API
    # ──────────────────────────────────────────────────────────────────

    def flag(self, *run_ids: int) -> None:
        """Flag runs by index to exclude from PR calculations."""
        self.flagged.update(run_ids)

    def unflag(self, *run_ids: int) -> None:
        """Remove flag from runs by index."""
        self.flagged -= set(run_ids)

    def flag_by_date(self, date_str: str) -> list[int]:
        """Flag all runs on YYYY-MM-DD. Returns the list of IDs flagged."""
        from datetime import date
        target = date.fromisoformat(date_str)
        ids = [r["id"] for r in self.runs if r["date"] == target]
        self.flagged.update(ids)
        if not ids:
            print(f"[RunAnalyzer] No runs found on {date_str}")
        return ids

    def toggle_flag(self, run_id: int) -> bool:
        """Toggle flag state. Returns True if now flagged."""
        if run_id in self.flagged:
            self.flagged.discard(run_id)
            return False
        self.flagged.add(run_id)
        return True

    @property
    def active_runs(self) -> list[dict]:
        return [r for r in self.runs if r["id"] not in self.flagged]

    # ──────────────────────────────────────────────────────────────────
    # Sliding-window: flat distance
    # ──────────────────────────────────────────────────────────────────

    def _best_effort_distance(
        self, run: dict, target_km: float, tolerance: float = 0.01
    ) -> float | None:
        """
        Minimum seconds to cover target_km anywhere within the run.
        Returns None if the run is too short (below tolerance threshold).
        """
        if run["total_km"] < target_km * (1 - tolerance):
            return None

        df = (
            run["df"]
            .dropna(subset=["distance", "timestamp"])
            .sort_values("timestamp")
        )
        if len(df) < 2:
            return None

        dist_km = df["distance"].to_numpy(dtype=float) / 1000.0
        times_s = df["timestamp"].astype("int64").to_numpy() / 1e9

        best: float | None = None
        lo = 0
        for hi in range(1, len(dist_km)):
            # Shrink left edge while window is wider than needed
            while lo < hi - 1 and (dist_km[hi] - dist_km[lo + 1]) >= target_km:
                lo += 1
            covered = dist_km[hi] - dist_km[lo]
            if covered >= target_km * (1 - tolerance):
                elapsed = times_s[hi] - times_s[lo]
                if elapsed > 0:
                    scaled = elapsed * (target_km / covered)
                    if best is None or scaled < best:
                        best = scaled
        return best

    # ──────────────────────────────────────────────────────────────────
    # Sliding-window: elevation
    # ──────────────────────────────────────────────────────────────────

    def _best_effort_elevation(
        self,
        run: dict,
        target_m: float,
        direction: Literal["gain", "loss"],
        noise_filter_m: float = 1.0,
    ) -> float | None:
        """
        Minimum seconds to accumulate target_m of cumulative elevation
        gain or loss anywhere within the run.

        Algorithm
        ---------
        1. Compute per-step altitude deltas; filter steps < noise_filter_m
           to suppress GPS noise creating spurious micro-climbs/descents.
        2. Keep only directional steps (gain or loss), build prefix-sum.
        3. Two-pointer on prefix-sum to find shortest time window where
           the cumulative directional elevation >= target_m.

        noise_filter_m : altitude steps smaller than this are treated as 0.
                         Default 1 m is conservative; raise to 2–3 m if your
                         GPS altitude is noisy.
        """
        alt_col = run["alt_col"]
        if alt_col is None or alt_col not in run["df"].columns:
            return None

        df = run["df"].dropna(subset=["timestamp"]).sort_values("timestamp")
        alt = pd.to_numeric(df[alt_col], errors="coerce").to_numpy(dtype=float)
        times_s = df["timestamp"].astype("int64").to_numpy() / 1e9
        n = len(alt)
        if n < 3:
            return None

        # Per-step deltas with noise filter
        steps = np.diff(alt)                                          # length n-1
        steps = np.where(np.abs(steps) >= noise_filter_m, steps, 0.0)

        # Directional component
        if direction == "gain":
            directional = np.where(steps > 0, steps, 0.0)
        else:
            directional = np.where(steps < 0, -steps, 0.0)

        total = directional.sum()
        if total < target_m:
            return None  # not enough total gain/loss in this run

        # Prefix sum: prefix[i] = sum of directional[0..i-1]
        # prefix[0] = 0, prefix[k] = directional[0]+...+directional[k-1]
        prefix = np.zeros(n, dtype=float)
        prefix[1:] = np.cumsum(directional)

        # Two-pointer: find min (times_s[hi] - times_s[lo])
        # such that prefix[hi] - prefix[lo] >= target_m
        # Points correspond to df rows; directional[i] is the step lo→lo+1.
        best: float | None = None
        lo = 0
        for hi in range(1, n):
            # Advance lo as far right as window still satisfies the constraint
            while lo < hi - 1 and (prefix[hi] - prefix[lo + 1]) >= target_m:
                lo += 1
            if prefix[hi] - prefix[lo] >= target_m:
                elapsed = times_s[hi] - times_s[lo]
                if elapsed > 0 and (best is None or elapsed < best):
                    best = elapsed
        return best

    # ──────────────────────────────────────────────────────────────────
    # Unified dispatcher
    # ──────────────────────────────────────────────────────────────────

    def _best_effort(
        self, run: dict, target: DistanceTarget | ElevationTarget
    ) -> float | None:
        if isinstance(target, DistanceTarget):
            return self._best_effort_distance(run, target.distance_km)
        return self._best_effort_elevation(run, target.elev_m, target.direction)

    # ──────────────────────────────────────────────────────────────────
    # Personal records
    # ──────────────────────────────────────────────────────────────────

    def personal_records(
        self,
        targets: list[DistanceTarget | ElevationTarget] | None = None,
    ) -> dict[str, dict | None]:
        """
        Compute PRs for all targets using only non-flagged runs.

        Parameters
        ----------
        targets : list of DistanceTarget / ElevationTarget, default PR_TARGETS.

        Returns
        -------
        dict  {label → PR dict | None}

        PR dict keys (distance)  : time_s, time, pace, date, run_id
        PR dict keys (elevation) : time_s, time, mh, elev_m, date, run_id
        """
        if targets is None:
            targets = PR_TARGETS

        active = self.active_runs
        prs: dict[str, dict | None] = {}

        for target in targets:
            best_time: float | None = None
            best_run:  dict | None  = None

            for run in active:
                t = self._best_effort(run, target)
                if t is not None and (best_time is None or t < best_time):
                    best_time, best_run = t, run

            if best_time is not None and best_run is not None:
                entry: dict = {
                    "time_s": round(best_time, 1),
                    "time":   fmt_time(best_time),
                    "date":   best_run["date"],
                    "run_id": best_run["id"],
                }
                if isinstance(target, DistanceTarget):
                    entry["pace"] = fmt_pace(best_time / target.distance_km)
                else:
                    entry["mh"]     = fmt_mh(best_time, target.elev_m)
                    entry["elev_m"] = target.elev_m
                prs[target.label] = entry
            else:
                prs[target.label] = None

        return prs

    def best_for(self, label: str) -> dict | None:
        """
        Look up a single PR by label.
        e.g.  analyzer.best_for("5 km")  or  analyzer.best_for("+VK")
        """
        return self.personal_records().get(label)

    # ──────────────────────────────────────────────────────────────────
    # Summary tables
    # ──────────────────────────────────────────────────────────────────

    def summary(self, include_flagged: bool = True) -> pd.DataFrame:
        """One row per run with key stats."""
        rows = []
        for run in self.runs:
            if not include_flagged and run["id"] in self.flagged:
                continue
            rows.append({
                "id":           run["id"],
                "date":         run["date"],
                "distance_km":  round(run["total_km"], 2),
                "time":         fmt_time(run["elapsed_s"]),
                "avg_pace":     fmt_pace(run["avg_pace"]),
                "avg_hr":       round(run["avg_hr"])      if np.isfinite(run["avg_hr"])      else None,
                "avg_cadence":  round(run["avg_cadence"]) if np.isfinite(run["avg_cadence"]) else None,
                "elev_gain_m":  round(run["elev_gain_m"]) if np.isfinite(run["elev_gain_m"]) else None,
                "elev_loss_m":  round(run["elev_loss_m"]) if np.isfinite(run["elev_loss_m"]) else None,
                "flagged":      run["id"] in self.flagged,
            })
        return pd.DataFrame(rows)

    def pr_table(
        self,
        targets: list[DistanceTarget | ElevationTarget] | None = None,
    ) -> pd.DataFrame:
        """
        Tidy DataFrame of all PRs.

        Columns (distance rows) : label, kind, target, time, pace, date, run_id
        Columns (elevation rows): label, kind, target, time, m_h,  date, run_id
        """
        if targets is None:
            targets = PR_TARGETS
        prs = self.personal_records(targets)
        rows = []
        for target in targets:
            pr = prs.get(target.label)
            row: dict = {
                "label":   target.label,
                "kind":    target.kind,
                "time":    pr["time"]   if pr else "—",
                "date":    pr["date"]   if pr else None,
                "run_id":  pr["run_id"] if pr else None,
            }
            if isinstance(target, DistanceTarget):
                row["pace"]   = pr["pace"] if pr else "—"
                row["target"] = f"{target.distance_km:.4g} km"
            else:
                row["m_h"]    = pr["mh"]     if pr else "—"
                row["target"] = f"{target.elev_m:.0f} m {target.direction}"
            rows.append(row)
        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"RunAnalyzer({len(self.runs)} runs, {len(self.flagged)} flagged)"
