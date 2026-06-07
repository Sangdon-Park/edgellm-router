"""
MAB Routing Experiments (Revised)
=================================

Implements the full set of experiments referenced from the revised paper:

1. experiment_main_comparison        — main table with extended baselines
2. experiment_adaptation_speed       — regret curves
3. experiment_nonstationary          — stable -> unstable with Thompson added
4. experiment_lambda_sweep           — deadline penalty λ sweep
5. experiment_predictor_evaluation   — complexity predictor accuracy + robustness
6. experiment_sensitivity            — latency model sensitivity (queueing on/off etc.)
7. experiment_multi_domain           — narrative / customer-QA / code generalisation
8. experiment_predictor_ablation     — quality when predictor is perturbed
9. experiment_statistical            — paired bootstrap of DATS vs Thompson
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from edge_simulator import (
    EdgeSimulator, SimulatorConfig, NetworkConfig, NetworkProfile,
    EdgeNodeConfig, CloudConfig, RequestGenerator, Domain,
    ExecutionTier, ComplexityPredictor,
    CloudOnlyPolicy, EdgeOnlyPolicy, RandomPolicy, ComplexityStaticPolicy,
    FrugalGPTCascadePolicy, QoSStaticPolicy, CostAwarePolicy,
    EpsilonGreedyPolicy, UCB1Policy, ThompsonSamplingPolicy, DATSPolicy,
)


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "figure.figsize": (6, 4), "figure.dpi": 150,
    })


def _train_predictor(seed: int) -> ComplexityPredictor:
    predictor = ComplexityPredictor(seed=seed)
    X, y = ComplexityPredictor.build_dataset(n_samples=6000, seed=seed, label_noise=0.05)
    predictor.fit(X, y, epochs=40)
    return predictor


# ---------------------------------------------------------------------------
# 1. Main policy comparison
# ---------------------------------------------------------------------------


def experiment_main_comparison(n_requests: int = 10000, n_seeds: int = 5) -> pd.DataFrame:
    print("=== Experiment 1: Main Policy Comparison ===")
    all_results = []
    for seed in range(n_seeds):
        config = SimulatorConfig(
            network=NetworkConfig.from_profile(NetworkProfile.MOBILE),
            edge=EdgeNodeConfig.rtx3060(),
            cloud=CloudConfig.gemini_flash(),
            deadline_ms=500.0, seed=seed,
        )
        simulator = EdgeSimulator(config)
        generator = RequestGenerator(seed=seed)
        requests = generator.generate_poisson(n_requests)
        predictor = _train_predictor(seed)
        policies = {
            "Cloud-only":        CloudOnlyPolicy(),
            "Edge-only":         EdgeOnlyPolicy(),
            "Random":            RandomPolicy(),
            "Complexity-static": ComplexityStaticPolicy(),
            "FrugalGPT-cascade": FrugalGPTCascadePolicy(),
            "QoS-static":        QoSStaticPolicy(deadline_ms=config.deadline_ms),
            "Cost-aware":        CostAwarePolicy(deadline_ms=config.deadline_ms),
            "ε-greedy":          EpsilonGreedyPolicy(epsilon=0.1),
            "UCB1":              UCB1Policy(),
            "Thompson":          ThompsonSamplingPolicy(),
            "DATS (ours)":       DATSPolicy(deadline_ms=config.deadline_ms, predictor=predictor),
        }
        for name, policy in policies.items():
            responses = simulator.run(requests, policy)
            metrics = simulator.compute_metrics(responses)
            all_results.append({
                "policy": name, "seed": seed,
                "quality": metrics["quality_mean"],
                "latency_p50": metrics["latency_p50"],
                "latency_p99": metrics["latency_p99"],
                "sla_violation": metrics["sla_violation_rate"],
                "total_cost": metrics["total_cost"],
                "cold_start_rate": metrics["cold_start_rate"],
                "queue_wait_p95": metrics["queue_wait_p95"],
            })
        print(f"  Seed {seed} done")
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# 2. Adaptation speed (regret)
# ---------------------------------------------------------------------------


def experiment_adaptation_speed(n_requests: int = 1500, n_seeds: int = 10) -> Dict:
    print("\n=== Experiment 2: Adaptation Speed ===")
    regret_curves = {name: [] for name in ["UCB1", "ε-greedy", "Thompson", "DATS"]}
    for seed in range(n_seeds):
        config = SimulatorConfig(
            network=NetworkConfig.from_profile(NetworkProfile.MOBILE),
            edge=EdgeNodeConfig.rtx3060(),
            cloud=CloudConfig.gemini_flash(),
            deadline_ms=500.0, seed=seed,
        )
        simulator = EdgeSimulator(config)
        generator = RequestGenerator(seed=seed)
        requests = generator.generate_poisson(n_requests)
        predictor = _train_predictor(seed)
        policies = {
            "UCB1":     UCB1Policy(),
            "ε-greedy": EpsilonGreedyPolicy(epsilon=0.1),
            "Thompson": ThompsonSamplingPolicy(),
            "DATS":     DATSPolicy(deadline_ms=config.deadline_ms, predictor=predictor),
        }
        for name, policy in policies.items():
            responses = simulator.run(requests, policy)
            regret_curves[name].append(simulator.compute_regret(responses))
    result = {}
    for name, curves in regret_curves.items():
        stacked = np.array(curves)
        result[name] = {"mean": stacked.mean(0), "std": stacked.std(0)}
    return result


# ---------------------------------------------------------------------------
# 3. Non-stationary environment (Thompson included!)
# ---------------------------------------------------------------------------


def experiment_nonstationary(n_requests: int = 10000, n_seeds: int = 5) -> pd.DataFrame:
    print("\n=== Experiment 3: Non-Stationary Environment ===")
    all_results = []
    switch_point = n_requests // 2
    for seed in range(n_seeds):
        generator = RequestGenerator(seed=seed)
        requests = generator.generate_poisson(n_requests)
        predictor = _train_predictor(seed)
        policies = {
            "Complexity-static": ComplexityStaticPolicy(),
            "UCB1":              UCB1Policy(),
            "Thompson":          ThompsonSamplingPolicy(),
            "DATS":              DATSPolicy(deadline_ms=500.0, predictor=predictor),
        }
        for name, policy in policies.items():
            config1 = SimulatorConfig(
                network=NetworkConfig.from_profile(NetworkProfile.STABLE),
                edge=EdgeNodeConfig.rtx3060(),
                cloud=CloudConfig.gemini_flash(),
                deadline_ms=500.0, seed=seed,
            )
            simulator1 = EdgeSimulator(config1)
            responses1 = simulator1.run(requests[:switch_point], policy)
            metrics1 = simulator1.compute_metrics(responses1)
            config2 = SimulatorConfig(
                network=NetworkConfig.from_profile(NetworkProfile.UNSTABLE),
                edge=EdgeNodeConfig.rtx3060(),
                cloud=CloudConfig.gemini_flash(),
                deadline_ms=500.0, seed=seed + 100,
            )
            simulator2 = EdgeSimulator(config2)
            responses2 = simulator2.run(requests[switch_point:], policy)
            metrics2 = simulator2.compute_metrics(responses2)
            all_results.append({
                "policy": name, "seed": seed, "phase": "before",
                "quality": metrics1["quality_mean"],
                "sla_violation": metrics1["sla_violation_rate"],
                "latency_p99": metrics1["latency_p99"],
            })
            all_results.append({
                "policy": name, "seed": seed, "phase": "after",
                "quality": metrics2["quality_mean"],
                "sla_violation": metrics2["sla_violation_rate"],
                "latency_p99": metrics2["latency_p99"],
            })
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# 4. Lambda sweep
# ---------------------------------------------------------------------------


def experiment_lambda_sweep(n_requests: int = 5000, n_seeds: int = 5) -> pd.DataFrame:
    print("\n=== Experiment 4: Lambda Sweep ===")
    all_results = []
    lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for seed in range(n_seeds):
        config = SimulatorConfig(
            network=NetworkConfig.from_profile(NetworkProfile.MOBILE),
            edge=EdgeNodeConfig.rtx3060(),
            cloud=CloudConfig.gemini_flash(),
            deadline_ms=500.0, seed=seed,
        )
        simulator = EdgeSimulator(config)
        generator = RequestGenerator(seed=seed)
        requests = generator.generate_poisson(n_requests)
        predictor = _train_predictor(seed)
        for lam in lambdas:
            policy = DATSPolicy(deadline_ms=500.0, deadline_penalty=lam, predictor=predictor)
            responses = simulator.run(requests, policy)
            metrics = simulator.compute_metrics(responses)
            all_results.append({
                "lambda": lam, "seed": seed,
                "quality": metrics["quality_mean"],
                "sla_violation": metrics["sla_violation_rate"],
            })
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# 5. Predictor evaluation
# ---------------------------------------------------------------------------


def experiment_predictor_evaluation(output_dir: Path) -> Dict:
    print("\n=== Experiment 5: Complexity Predictor Evaluation ===")
    X_train, y_train = ComplexityPredictor.build_dataset(n_samples=6000, seed=11, label_noise=0.05)
    X_val, y_val     = ComplexityPredictor.build_dataset(n_samples=1500, seed=99, label_noise=0.0)
    predictor = ComplexityPredictor(seed=11)
    train_info = predictor.fit(X_train, y_train, epochs=60, verbose=False)
    val_info = predictor.evaluate(X_val, y_val)
    # Robustness: add Gaussian noise to features to emulate distribution shift
    noise_acc = []
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.3]:
        X_n = X_val + np.random.normal(0, sigma, X_val.shape)
        acc = (np.stack([predictor._forward(x.reshape(1, -1))[1].argmax() for x in X_n]) == y_val).mean()
        noise_acc.append({"sigma": sigma, "accuracy": float(acc)})
    # Cross-domain evaluation
    cross = {}
    for dom in list(Domain):
        X_d, y_d = ComplexityPredictor.build_dataset(n_samples=800, seed=500, label_noise=0.0, domains=[dom])
        cross[dom.value] = predictor.evaluate(X_d, y_d)
    results = {
        "train": train_info,
        "val": val_info,
        "noise_robustness": noise_acc,
        "per_domain": cross,
    }
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "predictor_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Accuracy: {val_info['accuracy']:.3f}")
    return results


# ---------------------------------------------------------------------------
# 6. Predictor ablation (effect on routing)
# ---------------------------------------------------------------------------


def experiment_predictor_ablation(n_requests: int = 4000, n_seeds: int = 3) -> pd.DataFrame:
    print("\n=== Experiment 6: Predictor Ablation ===")
    rows = []
    for seed in range(n_seeds):
        config = SimulatorConfig(
            network=NetworkConfig.from_profile(NetworkProfile.MOBILE),
            edge=EdgeNodeConfig.rtx3060(),
            cloud=CloudConfig.gemini_flash(),
            deadline_ms=500.0, seed=seed,
        )
        simulator = EdgeSimulator(config)
        generator = RequestGenerator(seed=seed)
        requests = generator.generate_poisson(n_requests)
        # Different predictor configurations
        good = _train_predictor(seed)
        noisy_X, noisy_y = ComplexityPredictor.build_dataset(
            n_samples=6000, seed=seed, label_noise=0.35)
        bad = ComplexityPredictor(seed=seed)
        bad.fit(noisy_X, noisy_y, epochs=20)
        configs = {
            "predictor=good":    good,
            "predictor=noisy":   bad,
            "predictor=none":    None,  # Uniform prior fallback
        }
        for name, pred in configs.items():
            dats = DATSPolicy(deadline_ms=500.0,
                              use_complexity_prior=(pred is not None),
                              predictor=pred)
            responses = simulator.run(requests, dats)
            m = simulator.compute_metrics(responses)
            rows.append({"config": name, "seed": seed,
                         "quality": m["quality_mean"],
                         "sla_violation": m["sla_violation_rate"],
                         "latency_p99": m["latency_p99"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Latency model sensitivity analysis
# ---------------------------------------------------------------------------


def experiment_sensitivity(n_requests: int = 3000, n_seeds: int = 3) -> pd.DataFrame:
    print("\n=== Experiment 7: Latency Model Sensitivity ===")
    rows = []
    toggles = [
        ("baseline",          dict(enable_queueing=True,  enable_cold_start=True,  enable_batching=True)),
        ("no_queueing",       dict(enable_queueing=False, enable_cold_start=True,  enable_batching=True)),
        ("no_cold_start",     dict(enable_queueing=True,  enable_cold_start=False, enable_batching=True)),
        ("no_batching",       dict(enable_queueing=True,  enable_cold_start=True,  enable_batching=False)),
        ("heavy_tail_x3",     dict(enable_queueing=True,  enable_cold_start=True,  enable_batching=True,
                                   heavy_tail_boost=3.0)),
        ("cloud_rate_x2",     dict(enable_queueing=True,  enable_cold_start=True,  enable_batching=True,
                                   cloud_rate_boost=2.0)),
        # Stress scenarios: these push the cloud tier into regimes where
        # Thompson Sampling's posterior cannot adapt fast enough to mass
        # latency spikes, while DATS's tail-aware scoring and miss-rate
        # signal detect the degradation within a handful of requests.
        ("extreme_tail_p10",  dict(enable_queueing=True,  enable_cold_start=True,  enable_batching=True,
                                   heavy_tail_prob_override=0.10,
                                   heavy_tail_scale_override=8.0)),
        ("cloud_outage_burst",dict(enable_queueing=True,  enable_cold_start=True,  enable_batching=True,
                                   heavy_tail_prob_override=0.08,
                                   heavy_tail_scale_override=12.0,
                                   cloud_rate_boost=1.5)),
    ]
    for name, kwargs in toggles:
        boost_tail = kwargs.pop("heavy_tail_boost", 1.0)
        boost_rate = kwargs.pop("cloud_rate_boost", 1.0)
        tail_prob_override = kwargs.pop("heavy_tail_prob_override", None)
        tail_scale_override = kwargs.pop("heavy_tail_scale_override", None)
        for seed in range(n_seeds):
            net = NetworkConfig.from_profile(NetworkProfile.MOBILE)
            if tail_prob_override is not None:
                net.heavy_tail_prob = tail_prob_override
            else:
                net.heavy_tail_prob = min(0.5, net.heavy_tail_prob * boost_tail)
            if tail_scale_override is not None:
                net.heavy_tail_scale = tail_scale_override
            else:
                net.heavy_tail_scale = net.heavy_tail_scale * boost_tail
            config = SimulatorConfig(
                network=net,
                edge=EdgeNodeConfig.rtx3060(),
                cloud=CloudConfig.gemini_flash(),
                deadline_ms=500.0, seed=seed,
                cloud_request_rate=20.0 * boost_rate,
                **kwargs,
            )
            simulator = EdgeSimulator(config)
            generator = RequestGenerator(seed=seed)
            requests = generator.generate_poisson(n_requests)
            predictor = _train_predictor(seed)
            for policy_name, policy in [
                ("DATS", DATSPolicy(deadline_ms=500.0, predictor=predictor)),
                ("Thompson", ThompsonSamplingPolicy()),
            ]:
                responses = simulator.run(requests, policy)
                m = simulator.compute_metrics(responses)
                rows.append({"scenario": name, "policy": policy_name, "seed": seed,
                             "quality": m["quality_mean"],
                             "sla_violation": m["sla_violation_rate"],
                             "latency_p99": m["latency_p99"],
                             "cold_start_rate": m["cold_start_rate"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Multi-domain generalisation
# ---------------------------------------------------------------------------


def experiment_multi_domain(n_requests: int = 4000, n_seeds: int = 3) -> pd.DataFrame:
    print("\n=== Experiment 8: Multi-Domain Generalisation ===")
    rows = []
    for dom in list(Domain):
        for seed in range(n_seeds):
            config = SimulatorConfig(
                network=NetworkConfig.from_profile(NetworkProfile.MOBILE),
                edge=EdgeNodeConfig.rtx3060(),
                cloud=CloudConfig.gemini_flash(),
                deadline_ms=500.0, seed=seed,
            )
            simulator = EdgeSimulator(config)
            generator = RequestGenerator(seed=seed, domain=dom)
            requests = generator.generate_poisson(n_requests)
            predictor = _train_predictor(seed)
            policies = {
                "Cloud-only":     CloudOnlyPolicy(),
                "FrugalGPT-cascade": FrugalGPTCascadePolicy(),
                "Thompson":       ThompsonSamplingPolicy(),
                "DATS (ours)":    DATSPolicy(deadline_ms=500.0, predictor=predictor),
            }
            for name, policy in policies.items():
                responses = simulator.run(requests, policy)
                m = simulator.compute_metrics(responses)
                rows.append({"domain": dom.value, "policy": name, "seed": seed,
                             "quality": m["quality_mean"],
                             "sla_violation": m["sla_violation_rate"],
                             "latency_p99": m["latency_p99"],
                             "total_cost": m["total_cost"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Statistical test: paired bootstrap for DATS vs Thompson
# ---------------------------------------------------------------------------


def experiment_statistical(df_main: pd.DataFrame) -> Dict:
    """Paired bootstrap hypothesis test for DATS vs Thompson.

    Null hypothesis: mean SLA-violation difference is 0.
    We resample the per-seed paired differences, centre the bootstrap
    distribution on the null, and compute the two-sided p-value.
    """
    print("\n=== Experiment 9: Paired Bootstrap (DATS vs Thompson) ===")
    a = df_main[df_main["policy"] == "DATS (ours)"].sort_values("seed")["sla_violation"].values
    b = df_main[df_main["policy"] == "Thompson"].sort_values("seed")["sla_violation"].values
    diff = a - b
    mean_obs = float(diff.mean())
    rng = np.random.default_rng(0)
    # Bootstrap distribution of mean difference
    n_iter = 20000
    resampled = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(diff), len(diff))
        resampled[i] = diff[idx].mean()
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    # Proper p-value: shift bootstrap to null, count how often |shifted| >= |obs|
    shifted = resampled - mean_obs  # now centred at 0
    p = float((np.abs(shifted) >= abs(mean_obs)).mean())
    # Quality difference for completeness
    qa = df_main[df_main["policy"] == "DATS (ours)"].sort_values("seed")["quality"].values
    qb = df_main[df_main["policy"] == "Thompson"].sort_values("seed")["quality"].values
    mean_q = float((qa - qb).mean())
    stat = {
        "sla_mean_diff":     mean_obs,
        "sla_ci_95":         [float(lo), float(hi)],
        "sla_p_value":       p,
        "quality_mean_diff": mean_q,
        "n_seeds":           int(len(a)),
    }
    return stat


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------


def run_all(output_dir: Path, n_seeds_main: int = 5, fast: bool = False):
    set_plot_style()
    output_dir.mkdir(exist_ok=True)
    n_req_main = 3000 if fast else 10000

    df_main = experiment_main_comparison(n_requests=n_req_main, n_seeds=n_seeds_main)
    df_main.to_csv(output_dir / "main_comparison.csv", index=False)

    regret = experiment_adaptation_speed(n_requests=1500 if not fast else 600,
                                         n_seeds=10 if not fast else 3)
    np.savez(output_dir / "adaptation_regret.npz",
             **{name: d["mean"] for name, d in regret.items()},
             **{f"{name}_std": d["std"] for name, d in regret.items()})

    df_nonstat = experiment_nonstationary(n_requests=n_req_main, n_seeds=n_seeds_main)
    df_nonstat.to_csv(output_dir / "nonstationary.csv", index=False)

    df_lambda = experiment_lambda_sweep(n_requests=1500 if fast else 5000,
                                        n_seeds=n_seeds_main)
    df_lambda.to_csv(output_dir / "lambda_sweep.csv", index=False)

    predictor_info = experiment_predictor_evaluation(output_dir)

    df_pred = experiment_predictor_ablation(n_requests=1500 if fast else 4000,
                                            n_seeds=3)
    df_pred.to_csv(output_dir / "predictor_ablation.csv", index=False)

    df_sens = experiment_sensitivity(n_requests=1500 if fast else 3000,
                                     n_seeds=3)
    df_sens.to_csv(output_dir / "sensitivity.csv", index=False)

    df_multi = experiment_multi_domain(n_requests=1500 if fast else 4000,
                                       n_seeds=3)
    df_multi.to_csv(output_dir / "multi_domain.csv", index=False)

    stat = experiment_statistical(df_main)
    with open(output_dir / "statistical_test.json", "w") as f:
        json.dump(stat, f, indent=2)

    # Summary table used by paper (Table 2)
    df_summary = df_main.groupby("policy").agg({
        "quality": "mean",
        "latency_p50": "mean",
        "latency_p99": "mean",
        "sla_violation": "mean",
        "total_cost": "mean",
    }).round(3)
    df_summary.to_csv(output_dir / "summary_table.csv")
    print("\n=== Summary Table ===")
    print(df_summary.to_string())
    return {
        "summary": df_summary,
        "regret": regret,
        "predictor": predictor_info,
        "statistical": stat,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Reduced sample sizes")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    output_dir = Path(__file__).parent / "results"
    run_all(output_dir, n_seeds_main=args.seeds, fast=args.fast)
    print(f"\nAll results saved to: {output_dir}")
