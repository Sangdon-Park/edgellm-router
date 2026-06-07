#!/usr/bin/env python3
"""Summarise energy, cost, and throughput from GPU trace/replay outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path.home() / "codex-experiments" / "edgellm-router" / "results"


def default_output_dir() -> Path:
    return Path(os.environ.get("EDGELLM_EXPERIMENT_DIR", DEFAULT_RESULTS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, default=default_output_dir() / "gpu_latency_trace_raw.csv")
    parser.add_argument("--replay-csv", type=Path, default=default_output_dir() / "gpu_trace_replay_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=0.15)
    parser.add_argument("--amortized-gpu-usd-per-hour", type=float, default=0.20,
                        help="Optional local hardware amortisation cost proxy.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.trace_csv.exists():
        df = pd.read_csv(args.trace_csv)
        df["energy_j"] = pd.to_numeric(df["energy_j"], errors="coerce")
        df["full_latency_ms"] = pd.to_numeric(df["full_latency_ms"], errors="coerce")
        df["electricity_cost_usd"] = df["energy_j"] / 3_600_000.0 * args.electricity_usd_per_kwh
        df["amortized_cost_usd"] = (df["full_latency_ms"] / 3_600_000.0) * args.amortized_gpu_usd_per_hour
        df["local_cost_usd"] = df["electricity_cost_usd"] + df["amortized_cost_usd"]
        trace_summary = df.groupby(["tier", "model_id", "batch_size", "target_new_tokens"], as_index=False).agg({
            "full_latency_ms": ["mean", "median"],
            "ttft_ms": "median",
            "tokens_per_s": "mean",
            "energy_j": "mean",
            "local_cost_usd": "mean",
            "peak_vram_mb": "max",
        })
        trace_summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in trace_summary.columns
        ]
        trace_summary.to_csv(args.output_dir / "gpu_energy_trace_summary.csv", index=False)

    if args.replay_csv.exists():
        replay = pd.read_csv(args.replay_csv)
        replay["local_energy_kwh"] = replay["energy_j_per_request"] * replay["n_requests"] / 3_600_000.0
        replay["electricity_cost_usd"] = replay["local_energy_kwh"] * args.electricity_usd_per_kwh
        replay_summary = replay.groupby(["domain", "deadline_ms", "stress", "policy"], as_index=False).agg({
            "quality": "mean",
            "sla_violation": "mean",
            "latency_p99": "mean",
            "energy_j_per_request": "mean",
            "electricity_cost_usd": "mean",
            "total_cost": "mean",
        })
        replay_summary["quality_per_joule"] = replay_summary["quality"] / np.maximum(
            replay_summary["energy_j_per_request"], 1e-9
        )
        replay_summary.to_csv(args.output_dir / "gpu_energy_replay_summary.csv", index=False)

    print(f"Energy summaries written to {args.output_dir}")


if __name__ == "__main__":
    main()
