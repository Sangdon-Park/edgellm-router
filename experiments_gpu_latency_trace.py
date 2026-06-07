#!/usr/bin/env python3
r"""Collect empirical local-GPU LLM latency and energy traces.

This script is intentionally self-contained and writes outside Dropbox by
default.  Use it to calibrate the simulator with real RTX inference traces:

  python experiments_gpu_latency_trace.py ^
      --model Qwen/Qwen3-4B:edge --model Qwen/Qwen3-14B:cloud ^
      --prompt-csv C:\Users\User\codex-experiments\edgellm-router\results\extended_prompt_bench.csv

The script records TTFT proxy latency (one generated token), full generation
latency, generated tokens, tokens/s, peak VRAM, and NVML power samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = Path.home() / "codex-experiments"
DEFAULT_RESULTS = DEFAULT_ROOT / "edgellm-router" / "results"
DEFAULT_HF_HOME = DEFAULT_ROOT / "hf-cache"


@dataclass
class PowerStats:
    avg_w: Optional[float]
    max_w: Optional[float]
    samples: int


class NvmlPowerSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._ok = False

    def __enter__(self) -> "NvmlPowerSampler":
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._ok = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._ok = False
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append(float(mw) / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ok:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def stats(self) -> PowerStats:
        if not self.samples:
            return PowerStats(None, None, 0)
        return PowerStats(float(statistics.mean(self.samples)), float(max(self.samples)), len(self.samples))


def default_output_dir() -> Path:
    return Path(os.environ.get("EDGELLM_EXPERIMENT_DIR", DEFAULT_RESULTS))


def configure_external_caches() -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def parse_model_arg(value: str) -> Tuple[str, str]:
    if ":" in value:
        model_id, tier = value.rsplit(":", 1)
    else:
        model_id, tier = value, "edge"
    tier = tier.lower().strip()
    if tier not in {"edge", "cloud"}:
        raise argparse.ArgumentTypeError("model tier must be edge or cloud")
    return model_id.strip(), tier


def load_prompts(path: Path, limit: int) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def ensure_prompt_bench(path: Path, size: int) -> None:
    if path.exists():
        return
    from experiments_extended_prompt_bench import make_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(make_rows(size))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def percentile(xs: List[float], pct: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    idx = min(len(xs_sorted) - 1, max(0, round((pct / 100.0) * (len(xs_sorted) - 1))))
    return float(xs_sorted[idx])


@contextmanager
def maybe_inference_mode(torch):
    with torch.inference_mode():
        yield


def run_one_generation(torch, model, tokenizer, prompts: List[str], max_new_tokens: int) -> Tuple[float, int]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_tokens = int(encoded["input_ids"].shape[1])
    torch.cuda.synchronize()
    start = time.perf_counter()
    with maybe_inference_mode(torch):
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    total_new = int(out.shape[1] - input_tokens) * len(prompts)
    return elapsed_ms, total_new


def benchmark_model(
    model_id: str,
    tier: str,
    prompts: List[dict],
    batch_sizes: List[int],
    output_tokens: List[int],
    warmup: int,
    dtype: str,
) -> List[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    warm_prompt = apply_chat_template(tokenizer, prompts[0]["prompt"])
    for _ in range(warmup):
        run_one_generation(torch, model, tokenizer, [warm_prompt], 4)

    rows: List[dict] = []
    for bs in batch_sizes:
        for out_tok in output_tokens:
            for prompt_row in prompts:
                text = apply_chat_template(tokenizer, prompt_row["prompt"])
                batch = [text] * bs

                torch.cuda.reset_peak_memory_stats()
                with NvmlPowerSampler() as power:
                    ttft_ms, _ = run_one_generation(torch, model, tokenizer, batch, 1)
                    full_ms, new_tokens = run_one_generation(torch, model, tokenizer, batch, out_tok)
                pstats = power.stats()
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                tps = (new_tokens / (full_ms / 1000.0)) if full_ms > 0 else 0.0
                joules = (pstats.avg_w * full_ms / 1000.0) if pstats.avg_w is not None else None

                rows.append({
                    "model_id": model_id,
                    "tier": tier,
                    "prompt_id": prompt_row.get("id"),
                    "domain": prompt_row.get("domain"),
                    "complexity": int(prompt_row.get("complexity", 1)),
                    "category": prompt_row.get("category"),
                    "length_bucket": prompt_row.get("length_bucket"),
                    "batch_size": bs,
                    "target_new_tokens": out_tok,
                    "ttft_ms": ttft_ms,
                    "full_latency_ms": full_ms,
                    "new_tokens": new_tokens,
                    "tokens_per_s": tps,
                    "avg_power_w": pstats.avg_w,
                    "max_power_w": pstats.max_w,
                    "power_samples": pstats.samples,
                    "energy_j": joules,
                    "peak_vram_mb": peak_vram_mb,
                })
    del model
    torch.cuda.empty_cache()
    return rows


def summarise(rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, int, int], List[dict]] = {}
    for r in rows:
        key = (r["model_id"], r["tier"], int(r["batch_size"]), int(r["target_new_tokens"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (model_id, tier, bs, out_tok), vals in groups.items():
        lat = [float(v["full_latency_ms"]) for v in vals]
        ttft = [float(v["ttft_ms"]) for v in vals]
        tps = [float(v["tokens_per_s"]) for v in vals]
        energy = [float(v["energy_j"]) for v in vals if v["energy_j"] not in (None, "")]
        out.append({
            "model_id": model_id,
            "tier": tier,
            "batch_size": bs,
            "target_new_tokens": out_tok,
            "n": len(vals),
            "latency_p50": percentile(lat, 50),
            "latency_p95": percentile(lat, 95),
            "latency_p99": percentile(lat, 99),
            "ttft_p50": percentile(ttft, 50),
            "tokens_per_s_mean": statistics.mean(tps),
            "energy_j_mean": statistics.mean(energy) if energy else None,
            "peak_vram_mb_max": max(float(v["peak_vram_mb"]) for v in vals),
        })
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_external_caches()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model_arg,
                        default=None,
                        help="Model spec as HF_MODEL_ID:tier. Tier is edge or cloud.")
    parser.add_argument("--prompt-csv", type=Path,
                        default=default_output_dir() / "extended_prompt_bench.csv")
    parser.add_argument("--prompt-limit", type=int, default=96)
    parser.add_argument("--bench-size", type=int, default=900)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--output-tokens", default="32,64,128")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--raw-name", default="gpu_latency_trace_raw.csv")
    parser.add_argument("--summary-name", default="gpu_latency_trace_summary.csv")
    args = parser.parse_args()

    models = args.model or [("Qwen/Qwen3-4B", "edge"), ("Qwen/Qwen3-14B", "cloud")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    output_tokens = [int(x) for x in args.output_tokens.split(",") if x.strip()]

    ensure_prompt_bench(args.prompt_csv, args.bench_size)
    prompts = load_prompts(args.prompt_csv, args.prompt_limit)
    all_rows: List[dict] = []
    for model_id, tier in models:
        print(f"Benchmarking {model_id} as {tier} on {len(prompts)} prompts")
        all_rows.extend(benchmark_model(model_id, tier, prompts, batch_sizes, output_tokens, args.warmup, args.dtype))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / args.raw_name
    summary_path = args.output_dir / args.summary_name
    write_csv(raw_path, all_rows)
    write_csv(summary_path, summarise(all_rows))
    meta = {
        "models": [{"model_id": m, "tier": t} for m, t in models],
        "prompt_csv": str(args.prompt_csv),
        "prompt_limit": args.prompt_limit,
        "batch_sizes": batch_sizes,
        "output_tokens": output_tokens,
        "hf_home": os.environ.get("HF_HOME"),
        "output_dir": str(args.output_dir),
    }
    with open(args.output_dir / "gpu_latency_trace_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote raw trace to {raw_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
