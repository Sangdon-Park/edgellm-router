# EdgeLLM-Router

Source code and artifacts for:

**Latency-Constrained Request Routing for Interactive LLM Applications: A Multi-Armed Bandit Approach**

This repository contains the simulator, policies, complexity predictor, experiment scripts, generated results, and paper artifacts used in the revised manuscript.

## Contents

- `edge_simulator.py` - discrete-event edge/cloud simulator, routing policies, DATS, baselines, and the lightweight MLP complexity predictor.
- `experiments_mab_routing.py` - main policy comparison, non-stationary experiment, sensitivity analysis, multi-domain evaluation, lambda sweep, predictor ablation, and statistical test.
- `experiments_quality_validation.py` - simulator quality-assumption validation, including synthetic and API-backed paths.
- `experiments_predictor_realistic.py` - realistic-query transfer evaluation for the complexity predictor.
- `experiments_gpu_latency_trace.py` and `experiments_trace_replay_routing.py` - local-GPU latency tracing and empirical replay utilities.
- `generate_figures.py` - regenerates the manuscript figures from `results/`.
- `models/complexity_predictor_weights.npz` - trained MLP weights used for the released predictor artifact.
- `models/complexity_predictor_metadata.json` - training/evaluation metadata for the released predictor artifact.
- `results/` - CSV/JSON/NPZ outputs used to generate the manuscript tables and figures.
- `figures/` - PDF figures referenced by the manuscript.
- `paper/` - manuscript PDF/source and response-to-reviewers PDF.

## Quick Start

Install the minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a fast smoke reproduction:

```bash
python experiments_mab_routing.py --output results_smoke --fast --seeds 1
python generate_figures.py
```

Run the main simulator experiments:

```bash
python experiments_mab_routing.py --output results --seeds 5
python experiments_predictor_realistic.py
python generate_figures.py
```

The API-backed validation scripts require provider API keys and are not required for the deterministic simulator reproduction.

## GPU Experiments

GPU experiments were run outside the synchronized manuscript directory to avoid Dropbox sync contention. The helper script is:

```powershell
.\run_gpu_experiments_outside_dropbox.ps1
```

See `GPU_EXPERIMENTS.md` for the intended workflow. GPU tracing depends on local hardware, drivers, and model availability, so the released `results/` directory includes the generated traces and replay summaries used in the paper.

## Reproducibility Notes

- The simulator and predictor are deterministic under the seeds in the scripts.
- The complexity predictor is implemented without PyTorch/TensorFlow; the released weights are a NumPy `.npz` file.
- The main manuscript table/figure outputs are already included under `results/` and `figures/`.
- API-backed Gemini validation can be rerun with appropriate keys, but the controlled simulator experiments are the primary reproducible evidence.

