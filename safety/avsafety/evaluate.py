"""Run the scenario matrix (noise x GPS fault x odom fault) for every estimator and seed, checkpointing each
scenario to results/episodes.csv, then aggregate into results/*.md."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import ESTIMATORS, run_episode
from .scenario import scenario_matrix

OUT = Path("results")


def _job(args):
    return run_episode(*args)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g); k = int((g.outcome == "collision").sum())
        lo, hi = wilson(k, n)
        rows.append({**dict(zip(by, key)), "episodes": n, "collision_rate": k / n, "collision_ci95": f"[{lo:.3f}, {hi:.3f}]",
                     "success_rate": (g.outcome == "success").mean(), "timeout_rate": (g.outcome == "timeout").mean(),
                     "min_clear_p05": g.min_clearance.quantile(0.05), "min_clear_median": g.min_clearance.median(),
                     "loc_rmse_median": g.loc_rmse.median(), "loc_max_p95": g.loc_max.quantile(0.95),
                     "time_success_median": g.loc[g.outcome == "success", "time"].median(),
                     "nees_mean": g.nees_mean.mean(), "frac_nees_gt_chi2_95": g.nees_gt_chi2_95.mean()})
    return pd.DataFrame(rows)


def to_md(t: pd.DataFrame) -> str:
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    ap.add_argument("--resume", action="store_true", help="skip scenarios already in results/episodes.csv")
    a = ap.parse_args()
    if a.seeds < 1 or a.workers < 1:
        ap.error("seeds and workers must be positive")
    OUT.mkdir(exist_ok=True)
    csv = OUT / "episodes.csv"
    done = set(pd.read_csv(csv).scenario) if a.resume and csv.exists() else set()
    if not a.resume and csv.exists():
        csv.unlink()
    t0 = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    scs = scenario_matrix()
    with Pool(a.workers) as pool:
        for i, sc in enumerate(scs):
            if sc.name in done:
                continue
            jobs = [(sc, est, s) for est in ESTIMATORS for s in range(a.seeds)]
            res = pd.DataFrame(pool.map(_job, jobs, chunksize=4))
            res.to_csv(csv, mode="a", header=not csv.exists(), index=False)      # checkpoint per scenario
            print(f"[{i + 1}/{len(scs)}] {sc.name}: {time.time() - t0:.0f}s", flush=True)
    df = pd.read_csv(csv)
    expected = {(sc.name, estimator, seed) for sc in scs for estimator in ESTIMATORS for seed in range(a.seeds)}
    actual = list(df[["scenario", "estimator", "seed"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("Episode coverage is incomplete or duplicated; aggregation requires the requested full matrix")
    cols = ["episodes", "collision_rate", "collision_ci95", "success_rate", "timeout_rate", "min_clear_p05",
            "min_clear_median", "loc_rmse_median", "loc_max_p95", "time_success_median", "nees_mean",
            "frac_nees_gt_chi2_95"]
    fmt = to_md
    by_est = summarize(df, ["estimator"])
    by_fault = summarize(df, ["gps_fault", "estimator"])
    by_noise = summarize(df, ["gps_sigma", "estimator"])
    by_odom = summarize(df, ["odom_fault", "estimator"])
    cells = summarize(df, ["scenario", "estimator"])
    cells.to_csv(OUT / "cells.csv", index=False)
    n_cells = df.scenario.nunique()
    unsafe = cells[(cells.estimator.isin(["ekf", "ekf_monitor"])) & (cells.collision_rate > 0)]
    exposure = df[df.gps_fault.ne("none") | df.odom_fault.ne("ok")].fault_at_closest.mean()
    cov = (f"Scenario coverage: {n_cells}/{len(scs)} matrix cells x {len(ESTIMATORS)} estimators x "
           f"{df.seed.nunique()} seeds = {len(df)} episodes. Fault active at the closest obstacle approach in "
           f"{exposure:.1%} of faulted episodes. Cells with any EKF collision: "
           f"ekf {int((unsafe.estimator == 'ekf').sum())}, ekf_monitor {int((unsafe.estimator == 'ekf_monitor').sum())} "
           f"of {n_cells}. Wall time {time.time() - t0:.0f}s with {a.workers} workers.")
    md = ["# AV safety evaluation (synthetic 2-D robot, see README)", "", cov, "",
          "Localization errors compare the updated estimate and truth at the same simulator time. NEES summaries are equal-weight means of per-episode summaries; the exceedance column averages each episode's sampled exceedance fraction.", "",
          "## By estimator (all scenarios)", fmt(by_est[["estimator"] + cols]), "",
          "## By GPS fault", fmt(by_fault[["gps_fault", "estimator"] + cols]), "",
          "## By GPS noise sigma [m]", fmt(by_noise[["gps_sigma", "estimator"] + cols]), "",
          "## By odometry fault", fmt(by_odom[["odom_fault", "estimator"] + cols]), "",
          "## Worst cells for ekf / ekf_monitor (collision rate > 0)",
          fmt(unsafe.sort_values("collision_rate", ascending=False)[["scenario", "estimator", "episodes", "collision_rate",
                                                                    "collision_ci95", "min_clear_p05"]].head(20))]
    (OUT / "safety_report.md").write_text("\n".join(md) + "\n")
    metadata = {"started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - t0, "workers": a.workers, "seeds_per_cell_estimator": a.seeds,
                "scenario_cells": len(scs), "estimators": list(ESTIMATORS), "episodes": len(df),
                "unique_requested_episode_keys_verified": True, "resume": a.resume}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n".join(md[:11]))


if __name__ == "__main__":
    main()
