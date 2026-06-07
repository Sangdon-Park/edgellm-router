#!/usr/bin/env python3
"""Replay routing policies on empirical GPU latency traces.

The main simulator samples latency from parametric distributions.  This script
uses measured local-GPU inference rows from experiments_gpu_latency_trace.py and
replays routing policies on top of those empirical distributions.  Quality is
kept comparable with the paper's calibrated quality model unless a separate
quality CSV is supplied.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_simulator import (
    ComplexityPredictor,
    DATSPolicy,
    EpsilonGreedyPolicy,
    ExecutionTier,
    FrugalGPTCascadePolicy,
    QoSStaticPolicy,
    Request,
    RequestGenerator,
    Response,
    ThompsonSamplingPolicy,
    UCB1Policy,
    Domain,
)


DEFAULT_RESULTS = Path.home() / "codex-experiments" / "edgellm-router" / "results"


def default_output_dir() -> Path:
    return Path(os.environ.get("EDGELLM_EXPERIMENT_DIR", DEFAULT_RESULTS))


def percentile(xs: List[float], p: float) -> float:
    return float(np.percentile(np.array(xs, dtype=float), p))


@dataclass
class TraceSample:
    tier: str
    domain: str
    complexity: int
    length_bucket: str
    target_new_tokens: int
    batch_size: int
    ttft_ms: float
    full_latency_ms: float
    tokens_per_s: float
    energy_j: Optional[float]


class EmpiricalTrace:
    def __init__(self, rows: List[TraceSample], seed: int = 0):
        self.rows = rows
        self.rng = random.Random(seed)
        self.by_tier: Dict[str, List[TraceSample]] = {}
        self.by_key: Dict[Tuple[str, str, int], List[TraceSample]] = {}
        for row in rows:
            self.by_tier.setdefault(row.tier, []).append(row)
            self.by_key.setdefault((row.tier, row.domain, row.complexity), []).append(row)

    @classmethod
    def from_csv(cls, path: Path, seed: int = 0) -> "EmpiricalTrace":
        df = pd.read_csv(path)
        rows = []
        for _, r in df.iterrows():
            rows.append(TraceSample(
                tier=str(r["tier"]),
                domain=str(r.get("domain", "narrative")),
                complexity=int(r.get("complexity", 1)),
                length_bucket=str(r.get("length_bucket", "medium")),
                target_new_tokens=int(r.get("target_new_tokens", 64)),
                batch_size=int(r.get("batch_size", 1)),
                ttft_ms=float(r["ttft_ms"]),
                full_latency_ms=float(r["full_latency_ms"]),
                tokens_per_s=float(r.get("tokens_per_s", 0.0)),
                energy_j=None if pd.isna(r.get("energy_j", None)) else float(r.get("energy_j", 0.0)),
            ))
        return cls(rows, seed=seed)

    def sample(self, tier: str, request: Request) -> TraceSample:
        dom = request.domain.value if hasattr(request.domain, "value") else str(request.domain)
        key = (tier, dom, int(request.complexity))
        pool = (
            self.by_key.get(key)
            or self.by_key.get((tier, "narrative", int(request.complexity)))
            or self.by_tier.get(tier)
            or next(iter(self.by_tier.values()))
        )
        return self.rng.choice(pool)


class EmpiricalReplaySimulator:
    def __init__(
        self,
        trace: EmpiricalTrace,
        deadline_ms: float,
        latency_multiplier: float = 1.0,
        burst_prob: float = 0.0,
        burst_multiplier: float = 1.0,
        seed: int = 0,
    ):
        self.trace = trace
        self.deadline_ms = deadline_ms
        self.latency_multiplier = latency_multiplier
        self.burst_prob = burst_prob
        self.burst_multiplier = burst_multiplier
        self.rng = np.random.default_rng(seed)

    def _quality(self, request: Request, tier: ExecutionTier) -> float:
        c = int(request.complexity)
        if tier == ExecutionTier.EDGE:
            base = [0.98, 0.97, 0.95][c]
        elif tier == ExecutionTier.CLOUD:
            base = [0.99, 0.98, 0.97][c]
        else:
            base = [0.98, 0.97, 0.96][c]
        if request.domain == Domain.CODE and tier == ExecutionTier.EDGE and c == 2:
            base -= 0.03
        if request.domain == Domain.CUSTOMER_QA and tier == ExecutionTier.EDGE and c == 2:
            base -= 0.015
        return float(np.clip(base + self.rng.normal(0, 0.025), 0, 1))

    def _latency(self, request: Request, tier: ExecutionTier) -> Tuple[float, float]:
        if tier == ExecutionTier.EDGE:
            s = self.trace.sample("edge", request)
            latency = s.full_latency_ms
            energy = s.energy_j or 0.0
        elif tier == ExecutionTier.CLOUD:
            s = self.trace.sample("cloud", request)
            latency = s.full_latency_ms
            energy = s.energy_j or 0.0
        else:
            edge = self.trace.sample("edge", request)
            cloud_pool = self.trace.by_tier.get("cloud")
            if cloud_pool:
                cloud = self.trace.sample("cloud", request)
                draft = edge.full_latency_ms * 0.55
                budget = self.deadline_ms - draft
                if budget > 180:
                    refine = min(cloud.full_latency_ms * 0.65, budget)
                    latency = draft + refine
                    energy = (edge.energy_j or 0.0) * 0.55 + (cloud.energy_j or 0.0) * 0.65
                else:
                    latency = draft
                    energy = (edge.energy_j or 0.0) * 0.55
            else:
                latency = edge.full_latency_ms * 0.75
                energy = (edge.energy_j or 0.0) * 0.75
        if self.rng.random() < self.burst_prob:
            latency *= self.burst_multiplier
        return latency * self.latency_multiplier, energy

    def run(self, requests: List[Request], policy) -> List[Response]:
        policy.reset()
        responses: List[Response] = []
        context = {"recent_latency_ema": 100.0}
        for req in requests:
            tier = policy.select_tier(req, context)
            latency, energy = self._latency(req, tier)
            q = self._quality(req, tier)
            if tier == ExecutionTier.EDGE:
                cost = 0.0
            elif tier == ExecutionTier.CLOUD:
                cost = (req.input_tokens + req.expected_output_tokens) / 1000 * 0.00125
            else:
                cost = req.expected_output_tokens / 1000 * 0.00125 * 0.5
            resp = Response(
                request_id=req.id,
                tier=tier,
                quality=q,
                latency_ms=float(latency),
                output_tokens=req.expected_output_tokens,
                cost=float(cost),
                deadline_met=latency <= self.deadline_ms,
            )
            setattr(resp, "energy_j", energy)
            policy.update(req, resp)
            responses.append(resp)
            context["recent_latency_ema"] = 0.9 * context["recent_latency_ema"] + 0.1 * resp.latency_ms
        return responses

    def metrics(self, responses: List[Response]) -> dict:
        lat = [r.latency_ms for r in responses]
        qual = [r.quality for r in responses]
        cost = [r.cost for r in responses]
        energy = [float(getattr(r, "energy_j", 0.0)) for r in responses]
        return {
            "quality": float(np.mean(qual)),
            "latency_p50": percentile(lat, 50),
            "latency_p95": percentile(lat, 95),
            "latency_p99": percentile(lat, 99),
            "sla_violation": float(np.mean([not r.deadline_met for r in responses])),
            "total_cost": float(np.sum(cost)),
            "energy_j_per_request": float(np.mean(energy)),
            "edge_share": float(np.mean([r.tier == ExecutionTier.EDGE for r in responses])),
            "cloud_share": float(np.mean([r.tier == ExecutionTier.CLOUD for r in responses])),
            "hybrid_share": float(np.mean([r.tier == ExecutionTier.HYBRID for r in responses])),
        }


def train_predictor(seed: int) -> ComplexityPredictor:
    predictor = ComplexityPredictor(seed=seed)
    x, y = ComplexityPredictor.build_dataset(n_samples=6000, seed=seed, label_noise=0.05)
    predictor.fit(x, y, epochs=40)
    return predictor


def make_requests(n: int, seed: int, domain: str) -> List[Request]:
    # document_qa is a heavier customer-support-like workload used only in
    # the empirical replay layer; the base simulator currently exposes three
    # domains, so we borrow CUSTOMER_QA request statistics for this fourth
    # prompt family instead of changing the original paper experiments.
    dom = Domain.CUSTOMER_QA if domain == "document_qa" else Domain(domain)
    return RequestGenerator(seed=seed, domain=dom).generate_poisson(n)


def policies(deadline: float, predictor: ComplexityPredictor) -> Dict[str, object]:
    return {
        "FrugalGPT-cascade": FrugalGPTCascadePolicy(),
        "QoS-static": QoSStaticPolicy(deadline_ms=deadline),
        "UCB1": UCB1Policy(),
        "Thompson": ThompsonSamplingPolicy(),
        "DATS (ours)": DATSPolicy(deadline_ms=deadline, predictor=predictor),
        "epsilon-greedy": EpsilonGreedyPolicy(epsilon=0.1),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, default=default_output_dir() / "gpu_latency_trace_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--n-requests", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--deadlines", default="300,500,750,1000,1500")
    parser.add_argument("--domains", default="narrative,customer_qa,code,document_qa")
    parser.add_argument("--stress", default="baseline:1.0:0.0:1.0,heavy_tail:1.0:0.05:3.0,degraded:1.35:0.02:2.0")
    args = parser.parse_args()

    if not args.trace_csv.exists():
        raise SystemExit(f"Trace CSV not found: {args.trace_csv}")
    trace = EmpiricalTrace.from_csv(args.trace_csv)
    deadlines = [float(x) for x in args.deadlines.split(",") if x.strip()]
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    stress_specs = []
    for spec in args.stress.split(","):
        name, mult, burst_p, burst_m = spec.split(":")
        stress_specs.append((name, float(mult), float(burst_p), float(burst_m)))

    rows: List[dict] = []
    for seed in range(args.seeds):
        predictor = train_predictor(seed)
        for domain in domains:
            requests = make_requests(args.n_requests, seed, domain)
            for deadline in deadlines:
                for stress_name, mult, burst_p, burst_m in stress_specs:
                    sim = EmpiricalReplaySimulator(
                        trace=trace,
                        deadline_ms=deadline,
                        latency_multiplier=mult,
                        burst_prob=burst_p,
                        burst_multiplier=burst_m,
                        seed=seed,
                    )
                    for policy_name, policy in policies(deadline, predictor).items():
                        responses = sim.run(requests, policy)
                        row = sim.metrics(responses)
                        row.update({
                            "seed": seed,
                            "domain": domain,
                            "deadline_ms": deadline,
                            "stress": stress_name,
                            "policy": policy_name,
                            "n_requests": args.n_requests,
                        })
                        rows.append(row)
        print(f"seed {seed} done")

    raw_path = args.output_dir / "gpu_trace_replay_raw.csv"
    write_csv(raw_path, rows)
    df = pd.DataFrame(rows)
    summary = df.groupby(["domain", "deadline_ms", "stress", "policy"], as_index=False).agg({
        "quality": "mean",
        "latency_p50": "mean",
        "latency_p95": "mean",
        "latency_p99": "mean",
        "sla_violation": "mean",
        "total_cost": "mean",
        "energy_j_per_request": "mean",
        "edge_share": "mean",
        "cloud_share": "mean",
        "hybrid_share": "mean",
    })
    summary_path = args.output_dir / "gpu_trace_replay_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote raw replay to {raw_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
