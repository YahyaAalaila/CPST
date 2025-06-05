#!/usr/bin/env python3
"""
Batch-simulate multivariate spatio-temporal Hawkes data.

Usage
-----
python run_simulations.py -c configs/sweep.yaml --jobs 8
"""

from pathlib import Path
import logging, datetime, json, uuid
from functools import partial
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib
from omegaconf import OmegaConf
from tqdm import tqdm

from simulators.Hawkes import MultiMarkHawkesDGP            # ← your class

# ----------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _build_branching(m: int, self_exc: float, cross_exc: float) -> np.ndarray:
    """Utility to create an m×m matrix with given diagonals / off-diagonals."""
    A = np.full((m, m), cross_exc)
    np.fill_diagonal(A, self_exc)
    return A

# ----------------------------------------------------------------------

def _events_to_df(events: list[list[dict]], run_id: str) -> pd.DataFrame:
    """
    Convert a list of 200 event‐lists into one DataFrame.
    Each inner list is one independent sequence.
    """
    dfs = []
    for seq_idx, seq in enumerate(events):
        df_seq = pd.DataFrame(seq)
        df_seq["seq"]    = seq_idx
        df_seq["run_id"] = run_id
        dfs.append(df_seq)
    return pd.concat(dfs, ignore_index=True)

# ───────────────────────────────────────────────────────────────────────
def simulate_one(cfg: Dict[str, Any],
                 run_id: str,
                 out_root: Path | None = None) -> pd.DataFrame:
    """
    Generate *one* configuration of data and return it as a pandas DataFrame.
    Optionally persists the parquet file to disk (if `out_root` is given).
    """
    rng = np.random.default_rng(cfg["seed"])

    sim = MultiMarkHawkesDGP(
        T                 = cfg["simulation"]["T"],
        domain            = cfg["simulation"]["domain"],
        A                 = np.array(cfg["branching"]),
        Lambda            = cfg["simulation"]["Lambda"],
        background_config = cfg["background"],
        kernel_config     = cfg["kernel"],
        mean              = cfg["normalization"]["mean"],
        cov               = cfg["normalization"]["cov"],
        rng               = rng,
    )

    events = [sim.generate_data() for _ in range(200)]
    df     = _events_to_df(events, run_id)

    # ── optional persistence ──────────────────────────────────────────
    if out_root is not None:
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(run_dir / "events.parquet", index=False)  # compact + fast
        # you can still call save_run_output(events, run_dir) if you need the
        # original pickled list for backward compatibility.

    return df

# ----------------------------------------------------------------------
def expand_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one concrete config per branching matrix."""
    base = {k: v for k, v in cfg.items() if k != "branching_list"}
    out  = []
    for idx, mat in enumerate(cfg["branching_list"]):
        one = json.loads(json.dumps(base))    # deep copy
        one["branching"] = mat
        one["seed"]      = int(base.get("seeds", [0])[0]) + idx
        out.append(one)
    return out
def events_to_df(events, run_id):
    if isinstance(events, list) and isinstance(events[0], list):
        events = events[0]                   # num_seqs==1
    df = pd.DataFrame(events)
    df["run_id"] = run_id
    return df

# ----------------------------------------------------------------------
def main():
    import argparse, subprocess, json, logging
    from omegaconf import OmegaConf

    # ─────────────────── CLI ──────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML sweep file")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--out_dir", default="results", help="Root output folder")
    args = parser.parse_args()

    # ─────────────────── config + output dirs ─────────────────────────
    cfg_raw  = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    job_cfgs = expand_grid(cfg_raw)                 # one cfg per branching matrix

    out_root = Path(args.out_dir) / f"sweep_{_timestamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    # provenance
    (out_root / "git_hash.txt").write_text(
        subprocess.getoutput("git rev-parse HEAD").strip() + "\n"
    )
    OmegaConf.save(config=OmegaConf.create(cfg_raw), f=out_root / "config.yaml")

    logging.basicConfig(filename=out_root / "run.log",
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting sweep with %d configs on %d workers",
                 len(job_cfgs), args.jobs)

    # ─────────────────── parallel execution ───────────────────────────
    worker = partial(simulate_one, out_root=out_root)   # keeps per-run Parquet
    with joblib.Parallel(n_jobs=args.jobs, backend="loky") as parallel:
        dfs = list(tqdm(
            parallel(joblib.delayed(worker)(cfg, f"run{idx:02d}")
                     for idx, cfg in enumerate(job_cfgs)),
            total=len(job_cfgs),
            desc="Simulating"
        ))

    # ─────────────────── aggregate outputs ────────────────────────────
    big_df = pd.concat(dfs, ignore_index=True)
    big_df.to_parquet(out_root / "events_all.parquet", index=False)

    # summary table (one row per run_id)
    summary = (
        big_df.groupby("run_id").size()
              .rename("n_events")
              .reset_index()
              .merge(
                  pd.DataFrame({
                      "run_id": [f"run{idx:02d}" for idx in range(len(job_cfgs))],
                      "seed"  : [cfg["seed"]      for cfg in job_cfgs],
                      "branching": [json.dumps(cfg["branching"])
                                    for cfg in job_cfgs],
                  }),
                  on="run_id",
                  how="left"
              )
              .sort_values("run_id")
    )
    summary.to_csv(out_root / "summary.csv", index=False)
    logging.info("Finished – results in %s", out_root)


if __name__ == "__main__":
    main()
