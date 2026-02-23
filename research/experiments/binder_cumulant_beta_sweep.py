"""
Binder Cumulant Beta Sweep Experiment (Erdős-Rényi Model)
=========================================================
Run ErdosRenyiModel for multiple (N, beta) combinations to study the
effect of beta on the critical temperature via the Binder Cumulant.

Fixed parameters:
  alpha = 20

Sweep parameters:
  beta  : 0.20, 0.22, 0.24, 0.26, 0.28, 0.30
  N     : 5000, 10000, 50000  (to remove finite-size effects)

Each combination is run for 100 000 Monte Carlo sweeps.
Results are saved to:
  results/erdos_renyi/binder_cumulant_beta_sweep_<timestamp>/
    magnetization.csv   — multi-index (N, beta) columns
    metadata.json
"""

import os
import json
import time
import random
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

from src.model.erdos_renyi import ErdosRenyiModel


# ---------------------------------------------------------------------------
# Worker function (executed in a separate process)
# ---------------------------------------------------------------------------

def run_simulation(N: int, alpha: float, beta: float, n_steps: int, seed: int):
    """Initialise and run one ErdosRenyiModel instance; return M_t time series."""

    t0 = time.perf_counter()
    print(f"[START] N={N:>6d}, alpha={alpha}, beta={beta:.2f}", flush=True)

    # Seed RNGs for reproducibility inside the worker process
    random.seed(seed)
    np.random.seed(seed)

    market = ErdosRenyiModel(alpha=alpha, beta=beta, N=N, seed=seed)
    market.initialize()

    for _ in range(n_steps):
        market.step()

    elapsed = time.perf_counter() - t0
    print(f"[DONE]  N={N:>6d}, alpha={alpha}, beta={beta:.2f}  ({elapsed:.1f}s)", flush=True)
    return (N, beta, market.M_t_values, elapsed)


def _worker(args: tuple):
    """Top-level wrapper so imap_unordered can pickle the callable."""
    return run_simulation(*args)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Experiment parameters ----
    ALPHA   = 20
    N_RANGE = [5_000, 10_000, 50_000]
    BETA_RANGE = [round(b, 2) for b in np.arange(0.20, 0.32, 0.02)]  # 0.20 … 0.30
    N_STEPS = 100_000

    print("=" * 60)
    print("Binder Cumulant Beta Sweep — Erdős-Rényi Model")
    print("=" * 60)
    print(f"alpha    : {ALPHA}")
    print(f"N values : {N_RANGE}")
    print(f"beta     : {BETA_RANGE}")
    print(f"n_steps  : {N_STEPS:,}")
    print(f"# tasks  : {len(N_RANGE) * len(BETA_RANGE)}")
    print("=" * 60)

    # Assign a unique reproducible seed to every (N, beta) pair
    base_seed = 42
    param_list = [
        (N, ALPHA, beta, N_STEPS, base_seed + i)
        for i, (N, beta) in enumerate(
            (N, beta) for N in N_RANGE for beta in BETA_RANGE
        )
    ]

    n_workers = min(mp.cpu_count(), len(param_list))

    # ---- Create output directory BEFORE starting pool ----
    # This ensures partial results can be saved immediately as each task completes.
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.join(script_dir, "../../results/erdos_renyi")
    output_dir = os.path.join(base_dir, f"binder_cumulant_beta_sweep_{timestamp}")
    partial_dir = os.path.join(output_dir, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    print(f"Output directory : {output_dir}")
    print(f"Launching pool with {n_workers} workers …\n")

    wall_start    = time.perf_counter()
    timing_records: list = []
    results:        list = []

    with mp.Pool(n_workers) as pool:
        for N, beta, M_t_values, elapsed in tqdm(
            pool.imap_unordered(_worker, param_list),
            total=len(param_list),
            desc="Simulations",
            unit="task",
            dynamic_ncols=True,
        ):
            # ── Immediately persist this result ──────────────────────────
            npy_path = os.path.join(partial_dir, f"N{N}_beta{beta:.2f}.npy")
            np.save(npy_path, np.array(M_t_values, dtype=np.int32))

            # ── Update incremental timing log ────────────────────────────
            m_e, s_e = divmod(int(elapsed), 60)
            timing_records.append({
                "N": N, "beta": beta,
                "elapsed_s": round(elapsed, 2),
                "elapsed_fmt": f"{m_e:02d}m {s_e:02d}s",
            })
            timing_records.sort(key=lambda r: (r["N"], r["beta"]))
            with open(os.path.join(output_dir, "timing_log.json"), "w") as f:
                json.dump(timing_records, f, indent=2)

            results.append((N, beta, M_t_values, elapsed))
            tqdm.write(f"  [SAVED] N={N:>6,}, beta={beta:.2f}  ({elapsed:.1f}s)  → {npy_path}")

    wall_elapsed = time.perf_counter() - wall_start

    # ---- Per-simulation timing summary ----
    print(f"\n{'─' * 44}")
    print(f"{'Per-simulation timing':^44}")
    print(f"{'─' * 44}")
    print(f"{'N':>10}  {'beta':>6}  {'time':>10}  {'seconds':>10}")
    print(f"{'─' * 44}")
    for r in timing_records:
        print(f"{r['N']:>10,}  {r['beta']:>6.2f}  {r['elapsed_fmt']:>10}  {r['elapsed_s']:>10.2f}")
    print(f"{'─' * 44}")
    wall_m, wall_s = divmod(int(wall_elapsed), 60)
    wall_h, wall_m = divmod(wall_m, 60)
    print(f"Total wall-clock : {wall_h:02d}h {wall_m:02d}m {wall_s:02d}s  ({wall_elapsed:.1f}s)")
    print(f"{'─' * 44}\n")

    # ---- Merge partials → magnetization.csv ----
    print("Merging partial results …")
    M_t_data: dict = {}
    for N, beta, M_t_values, _ in results:
        M_t_data[(N, beta)] = M_t_values

    M_t_df = pd.DataFrame(M_t_data)
    M_t_df.columns = pd.MultiIndex.from_tuples(
        M_t_df.columns, names=["N", "beta"]
    )

    # Save merged magnetisation time series
    M_t_df.to_csv(os.path.join(output_dir, "magnetization.csv"))
    print(f"Saved magnetization.csv → {output_dir}")

    # ---- Experiment metadata ----
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=script_dir
        ).decode().strip()
    except Exception:
        git_rev = "N/A"

    metadata = {
        "experiment_name": "binder_cumulant_beta_sweep",
        "timestamp": datetime.now().isoformat(),
        "git_revision": git_rev,
        "model": "ErdosRenyiModel",
        "parameters": {
            "alpha": ALPHA,
            "beta_range": BETA_RANGE,
            "N_range": N_RANGE,
            "n_steps": N_STEPS,
            "base_seed": base_seed,
            "total_simulations": len(param_list),
            "n_workers": n_workers,
        },
        "output_files": {
            "magnetization": "magnetization.csv",
            "partial_results": "partial/N{N}_beta{beta:.2f}.npy",
            "timing_log": "timing_log.json",
        },
        "timing": {
            "wall_clock_seconds": round(wall_elapsed, 2),
            "per_simulation": [
                {"N": r["N"], "beta": r["beta"], "elapsed_seconds": r["elapsed_s"]}
                for r in timing_records
            ],
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata.json → {output_dir}")
    print("\nAll results saved successfully.")
    print(f"Output directory: {output_dir}")
