'''
Compare Bornholdt model: default M(t) vs. |M(t)| in the local field.
Runs both variants with the same seed and parameters, writes separate outputs.
'''
import os
import json
import subprocess
import pandas as pd
from datetime import datetime
from src.model.bornholdt import BornholdtModel


def run_one(use_abs_M: bool, alpha: float, beta: float, L: int, p: float, frames: int, seed: int):
    model = BornholdtModel(alpha=alpha, beta=beta, L=L, p=p, use_abs_M=use_abs_M, seed=seed)
    df = model.run(frames, output_dir=None)
    return df


def main():
    seed = 42
    L, p = 50, 0.5
    alpha, beta = 20, 2
    frames = 3000

    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../results/bornholdt_abs_M_comparison'
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 1) Default: M(t) (signed)
    df_signed = run_one(use_abs_M=False, alpha=alpha, beta=beta, L=L, p=p, frames=frames, seed=seed)
    signed_dir = os.path.join(run_dir, 'signed_M')
    os.makedirs(signed_dir, exist_ok=True)
    df_signed.to_csv(os.path.join(signed_dir, 'timeseries.csv'), index=False)
    with open(os.path.join(signed_dir, 'params.json'), 'w') as f:
        json.dump({
            "use_abs_M": False,
            "alpha": alpha, "beta": beta, "L": L, "p": p,
            "frames": frames, "seed": seed,
        }, f, indent=2)

    # 2) Variant: |M(t)|
    df_abs = run_one(use_abs_M=True, alpha=alpha, beta=beta, L=L, p=p, frames=frames, seed=seed)
    abs_dir = os.path.join(run_dir, 'abs_M')
    os.makedirs(abs_dir, exist_ok=True)
    df_abs.to_csv(os.path.join(abs_dir, 'timeseries.csv'), index=False)
    with open(os.path.join(abs_dir, 'params.json'), 'w') as f:
        json.dump({
            "use_abs_M": True,
            "alpha": alpha, "beta": beta, "L": L, "p": p,
            "frames": frames, "seed": seed,
        }, f, indent=2)

    git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    meta = {
        "experiment": "bornholdt_abs_M_comparison",
        "timestamp": datetime.now().isoformat(),
        "git_revision": git_rev,
        "parameters": {"alpha": alpha, "beta": beta, "L": L, "p": p, "frames": frames, "seed": seed},
        "output_dirs": {"signed_M": "signed_M", "abs_M": "abs_M"},
    }
    with open(os.path.join(run_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Results written to {run_dir}")
    return run_dir


if __name__ == '__main__':
    main()
