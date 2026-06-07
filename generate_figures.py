#!/usr/bin/env python3
"""Generate all figures for the revised MAB routing paper.

Regenerates the existing figures (fixing legibility issues raised by the
reviewers) and adds new figures required by the revision:

- predictor_accuracy.pdf        — confusion matrix + per-class metrics
- predictor_robustness.pdf      — accuracy under feature noise
- cascade_comparison.pdf        — new baselines (FrugalGPT / QoS / cost-aware)
- multi_domain.pdf              — generalisation across domains
- sensitivity.pdf               — latency model sensitivity
- nonstationary_adaptation.pdf  — stable->unstable with Thompson included
"""

from __future__ import annotations

import json
import os
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_FONT = 11
AXIS_FONT = 12
TITLE_FONT = 12
LEGEND_FONT = 9.5
TICK_FONT = 10
ANNOT_FONT = 9
SMALL_ANNOT_FONT = 8.5
SUPTITLE_FONT = 13

plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.labelsize": AXIS_FONT,
    "axes.titlesize": TITLE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": LEGEND_FONT,
    "figure.dpi": 150,
})

RESULTS_DIR = Path("results")
GPU_RESULTS_DIR = Path(os.environ.get(
    "EDGELLM_EXPERIMENT_DIR",
    str(Path.home() / "codex-experiments" / "edgellm-router" / "results"),
))
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

COLORS = {
    "Cloud-only":         "#2ecc71",
    "Edge-only":          "#e74c3c",
    "Random":             "#95a5a6",
    "Complexity-static":  "#9b59b6",
    "FrugalGPT-cascade":  "#16a085",
    "QoS-static":         "#d35400",
    "Cost-aware":         "#7f8c8d",
    "UCB1":               "#3498db",
    "Thompson":           "#f39c12",
    "epsilon-greedy":     "#1abc9c",
    "DATS":               "#e91e63",
    "DATS (ours)":        "#e91e63",
}
MARKERS = {
    "Cloud-only": "o", "Edge-only": "s", "Random": "D",
    "Complexity-static": "^", "FrugalGPT-cascade": "P",
    "QoS-static": "X", "Cost-aware": "h",
    "UCB1": "v", "Thompson": "<", "epsilon-greedy": ">",
    "DATS (ours)": "*", "DATS": "*",
}


def display_policy(name):
    """Normalize policy names that can be mojibake in older CSV outputs."""
    if not isinstance(name, str):
        return name
    if name.endswith("-greedy") and name != "epsilon-greedy":
        return "epsilon-greedy"
    return name


def normalize_policy_frame(df):
    if "policy" in df.columns:
        df = df.copy()
        df["policy"] = df["policy"].map(display_policy)
    return df


def normalize_regret_key(key):
    suffix = "_std" if key.endswith("_std") else ""
    base = key[:-4] if suffix else key
    return f"{display_policy(base)}{suffix}"


LABEL_BBOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_all():
    out = {}
    out["summary"] = normalize_policy_frame(pd.read_csv(RESULTS_DIR / "summary_table.csv"))
    out["main"] = normalize_policy_frame(pd.read_csv(RESULTS_DIR / "main_comparison.csv"))
    out["nonstat"] = normalize_policy_frame(pd.read_csv(RESULTS_DIR / "nonstationary.csv"))
    out["lambda"] = normalize_policy_frame(pd.read_csv(RESULTS_DIR / "lambda_sweep.csv"))
    with open(RESULTS_DIR / "policy_comparison.json") as f:
        out["policy_json"] = json.load(f)
    for opt in ["predictor_ablation.csv", "sensitivity.csv", "multi_domain.csv"]:
        if (RESULTS_DIR / opt).exists():
            out[opt.replace(".csv", "")] = normalize_policy_frame(pd.read_csv(RESULTS_DIR / opt))
    if (RESULTS_DIR / "predictor_eval.json").exists():
        with open(RESULTS_DIR / "predictor_eval.json") as f:
            out["predictor_eval"] = json.load(f)
    if (RESULTS_DIR / "adaptation_regret.npz").exists():
        arr = np.load(RESULTS_DIR / "adaptation_regret.npz")
        out["regret"] = {normalize_regret_key(k): arr[k] for k in arr.files}
    if (RESULTS_DIR / "statistical_test.json").exists():
        with open(RESULTS_DIR / "statistical_test.json") as f:
            out["statistical"] = json.load(f)
    gpu_files = {
        "gpu_latency": "gpu_latency_trace_summary.csv",
        "gpu_replay": "gpu_trace_replay_summary.csv",
        "gpu_energy": "gpu_energy_replay_summary.csv",
    }
    for key, filename in gpu_files.items():
        path = GPU_RESULTS_DIR / filename
        if path.exists():
            out[key] = normalize_policy_frame(pd.read_csv(path))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_quality_sla_scatter(summary):
    """Single-column zoom of the low-SLA Pareto region.

    A two-panel full-range/zoom plot becomes illegible when placed in one
    Elsevier column. Table 2 carries the full-range numbers, while this plot
    keeps the policy cluster around the SLA target readable in print.
    """
    fig, ax = plt.subplots(figsize=(4.3, 2.75))
    rng = np.random.default_rng(0)
    for _, row in summary.iterrows():
        policy = row["policy"]
        c = COLORS.get(policy, "#333333")
        m = MARKERS.get(policy, "o")
        size = 145 if "DATS" in policy else 80
        sla = row["sla_violation"] * 100
        q = row["quality"]
        if sla <= 6.5 and 0.9665 <= q <= 0.9715:
            jitter_x = rng.normal(0, 0.08)
            jitter_y = rng.normal(0, 0.00015)
            ax.scatter(sla + jitter_x, q + jitter_y,
                       c=c, marker=m, s=size, label=policy,
                       edgecolors="black", linewidths=0.6, zorder=3)

    ax.set_xlim(-0.6, 6.5)
    ax.set_ylim(0.9665, 0.9715)
    ax.set_xlabel("SLA Violation Rate (%)")
    ax.set_ylabel("Response Quality")
    ax.set_title("Low-SLA Pareto Region")
    ax.grid(True, alpha=0.3)
    # Stagger label offsets so near-overlapping points (DATS, Thompson,
    # QoS-static clustered around SLA<1%) don't print on top of each other.
    label_offsets = {
        "DATS (ours)":      (8, 24),
        "Thompson":         (12, -14),
        "QoS-static":       (15, -2),
        "epsilon-greedy":   (12, 14),
        "UCB1":             (10, 4),
        "Cost-aware":       (-42, 9),
        "Edge-only":        (-44, -2),
    }
    for _, row in summary.iterrows():
        sla = row["sla_violation"] * 100
        q = row["quality"]
        if sla <= 6.5 and 0.9665 <= q <= 0.9715:
            off = label_offsets.get(row["policy"], (8, 4))
            ha = "right" if off[0] < 0 else "left"
            ax.annotate(row["policy"], (sla, q),
                        textcoords="offset points", xytext=off,
                        fontsize=ANNOT_FONT, ha=ha, bbox=LABEL_BBOX)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quality_sla_scatter.pdf", bbox_inches="tight")
    plt.close()
    print("  quality_sla_scatter.pdf")


def fig_tier_distribution(policy_json):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    mapping = [
        ("Cloud-only", "cloud_only"), ("Edge-only", "edge_only"),
        ("Random", "random"), ("Complexity-static", "complexity_static"),
        ("FrugalGPT-cascade", "frugalgpt_cascade"),
        ("QoS-static", "qos_static"), ("Cost-aware", "cost_aware"),
        ("UCB1", "ucb1"), ("Thompson", "thompson_sampling"),
        ("epsilon-greedy", "epsilon_greedy"), ("DATS (ours)", "dats"),
    ]
    labels = [m[0] for m in mapping]
    edge, cloud, hybrid = [], [], []
    for _, key in mapping:
        dist = policy_json[key]["metrics"]["tier_distribution"]
        edge.append(dist["EDGE"] * 100)
        cloud.append(dist["CLOUD"] * 100)
        hybrid.append(dist["HYBRID"] * 100)
    x = np.arange(len(labels))
    ax.bar(x, edge, 0.65, label="Edge", color="#3498db")
    ax.bar(x, cloud, 0.65, bottom=edge, label="Cloud", color="#e74c3c")
    ax.bar(x, hybrid, 0.65, bottom=np.array(edge) + np.array(cloud),
           label="Hybrid", color="#2ecc71")
    ax.set_ylabel("Routing Distribution (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Tier Selection Distribution by Policy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("  tier_distribution.pdf")


def fig_nonstationary(nonstat):
    """3-panel layout: Quality (all 4 policies), then SLA split by
    magnitude — the high-SLA group (Complexity-static, UCB1 spanning
    3–23%) and the low-SLA group (Thompson, DATS below 1%). Without the
    split, DATS at 0.38% is visually indistinguishable from Thompson at
    0.56% next to Complexity-static at 22.9%."""
    wanted = ["Complexity-static", "UCB1", "Thompson", "DATS"]
    policies = [p for p in wanted if p in nonstat["policy"].unique()]
    colors = {"Complexity-static": "#9b59b6", "UCB1": "#3498db",
              "Thompson": "#f39c12", "DATS": "#e91e63"}

    def phase_stats(p, phase, col):
        sub = nonstat[(nonstat["policy"] == p) & (nonstat["phase"] == phase)][col]
        if col == "sla_violation":
            sub = sub * 100
        return sub.mean(), sub.std()

    high = ["Complexity-static", "UCB1"]
    low = ["Thompson", "DATS"]
    high = [p for p in high if p in policies]
    low = [p for p in low if p in policies]

    fig = plt.figure(figsize=(14, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.28)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_hi = fig.add_subplot(gs[0, 1])
    ax_lo = fig.add_subplot(gs[0, 2])

    phases = ["before", "after"]
    phase_labels = ["Before (Stable)", "After (Unstable)"]
    x = np.arange(2)

    def _grouped_bars(ax, sub_policies, col, title, ylabel, ylim=None, ref1pct=False):
        width = 0.8 / max(len(sub_policies), 1)
        offsets = np.linspace(-(len(sub_policies) - 1) / 2 * width,
                              (len(sub_policies) - 1) / 2 * width,
                              len(sub_policies))
        for i, p in enumerate(sub_policies):
            mus = [phase_stats(p, ph, col)[0] for ph in phases]
            sds = [phase_stats(p, ph, col)[1] for ph in phases]
            bars = ax.bar(x + offsets[i], mus, width, yerr=sds, capsize=2.5,
                          label=p, color=colors[p], edgecolor="black", linewidth=0.4)
            if col == "sla_violation":
                for rect, v, sd in zip(bars, mus, sds):
                    label = f"{v:.2f}" if v < 10 else f"{v:.1f}"
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            v + sd + (max(mus) * 0.03),
                            label, ha="center", va="bottom", fontsize=SMALL_ANNOT_FONT)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(phase_labels)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=LEGEND_FONT)
        if ref1pct:
            ax.axhline(y=1.0, color="#888", linestyle=":", linewidth=1, alpha=0.8)

    _grouped_bars(ax_q, policies, "quality",
                  "Quality Under Network Change", "Response Quality",
                  ylim=(0.965, 0.976))
    _grouped_bars(ax_hi, high, "sla_violation",
                  "SLA — high-violation policies", "SLA Violation Rate (%)",
                  ylim=(0, 28))
    _grouped_bars(ax_lo, low, "sla_violation",
                  "SLA — low-violation policies", "SLA Violation Rate (%)",
                  ylim=(0, 1.0), ref1pct=True)

    plt.savefig(FIGURES_DIR / "nonstationary_adaptation.pdf", bbox_inches="tight")
    plt.close()
    print("  nonstationary_adaptation.pdf")


def fig_cost_quality_pareto(policy_json):
    """Full-range and zoomed panels. Without the zoom, the cheap-and-good
    cluster near cost ≈ 0 is indistinguishable."""
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.8),
                                           gridspec_kw={"width_ratios": [1, 1.3]})
    mapping = {
        "cloud_only": "Cloud-only", "edge_only": "Edge-only",
        "random": "Random", "complexity_static": "Complexity-static",
        "frugalgpt_cascade": "FrugalGPT-cascade", "qos_static": "QoS-static",
        "cost_aware": "Cost-aware", "ucb1": "UCB1",
        "thompson_sampling": "Thompson", "epsilon_greedy": "epsilon-greedy",
        "dats": "DATS (ours)",
    }
    for key, name in mapping.items():
        m = policy_json[key]["metrics"]
        c = COLORS.get(name, "#333333"); mk = MARKERS.get(name, "o")
        s = 230 if "DATS" in name else 120
        for ax in (ax_full, ax_zoom):
            ax.scatter(m["total_cost"], m["quality_mean"], c=c, marker=mk, s=s,
                       label=name, edgecolors="black", linewidths=0.6, alpha=0.88)

    ax_full.set_xlabel("API Cost (USD per 10K requests)")
    ax_full.set_ylabel("Response Quality")
    ax_full.set_title("Full range")
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc="lower right", ncol=2, fontsize=LEGEND_FONT)
    ax_full.axvspan(-0.01, 0.15, color="#e91e63", alpha=0.06, zorder=0)

    ax_zoom.set_xlim(-0.01, 0.15)
    ax_zoom.set_ylim(0.9665, 0.9715)
    ax_zoom.set_xlabel("API Cost (USD per 10K requests)")
    ax_zoom.set_ylabel("Response Quality")
    ax_zoom.set_title("Zoom on low-cost region")
    ax_zoom.grid(True, alpha=0.3)
    label_offsets = {
        "Edge-only": (10, 10),
        "Cost-aware": (10, -16),
        "DATS (ours)": (12, 8),
        "QoS-static": (12, 6),
        "UCB1": (10, 4),
        "Thompson": (10, 2),
        "epsilon-greedy": (10, 8),
    }
    for key, name in mapping.items():
        m = policy_json[key]["metrics"]
        if m["total_cost"] <= 0.15 and 0.9665 <= m["quality_mean"] <= 0.9715:
            ax_zoom.annotate(name, (m["total_cost"], m["quality_mean"]),
                             textcoords="offset points",
                             xytext=label_offsets.get(name, (8, 4)),
                             fontsize=ANNOT_FONT, bbox=LABEL_BBOX)

    fig.suptitle("Cost vs. Quality Trade-off", y=1.02, fontsize=SUPTITLE_FONT)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cost_quality_pareto.pdf", bbox_inches="tight")
    plt.close()
    print("  cost_quality_pareto.pdf")


def fig_latency_comparison(summary):
    """Sorted by p99 so ranking is immediately visible. Numeric labels on
    bars help when p99 is capped at 500 ms for several policies."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ordered = summary.sort_values("latency_p99").reset_index(drop=True)
    policies = ordered["policy"].tolist()
    x = np.arange(len(policies))
    width = 0.38
    bars50 = ax.bar(x - width / 2, ordered["latency_p50"], width, label="p50",
                    color="#3498db", edgecolor="black", linewidth=0.4)
    bars99 = ax.bar(x + width / 2, ordered["latency_p99"], width, label="p99",
                    color="#e74c3c", edgecolor="black", linewidth=0.4)
    ax.axhline(y=500, color="black", linestyle="--", linewidth=1.3,
               label="SLA Threshold (500 ms)")
    for rect, v in zip(bars99, ordered["latency_p99"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 25,
                f"{v:.0f}", ha="center", va="bottom", fontsize=SMALL_ANNOT_FONT, color="#7b1d10")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Latency Distribution by Policy (sorted by $p_{99}$)")
    ax.set_ylim(0, max(ordered["latency_p99"]) * 1.12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "latency_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  latency_comparison.pdf")


def fig_policy_radar(summary):
    """Dashed line for Thompson so it does not visually hide behind DATS
    (the two policies are nearly identical on every axis)."""
    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw=dict(polar=True))
    metrics = ["Quality", "Low p99", "Low SLA", "Low p50"]
    # Draw Thompson LAST (after DATS) so its dashed outline is visible on
    # top of DATS's solid line — otherwise Thompson would be occluded.
    plot_order = ["Cloud-only", "Complexity-static", "FrugalGPT-cascade",
                  "UCB1", "DATS (ours)", "Thompson"]
    s = summary.copy()
    s["Quality"] = ((s["quality"] - 0.80) / 0.15).clip(0, 1)
    s["Low p99"] = (1 - (s["latency_p99"] - 500) / 700).clip(0, 1)
    s["Low SLA"] = (1 - s["sla_violation"] / 0.30).clip(0, 1)
    s["Low p50"] = (1 - (s["latency_p50"] - 300) / 200).clip(0, 1)
    angles = [i / float(len(metrics)) * 2 * pi for i in range(len(metrics))]
    angles += angles[:1]
    styles = {"Thompson": "--", "DATS (ours)": "-"}
    widths = {"DATS (ours)": 2.8, "Thompson": 2.0}
    for policy in plot_order:
        if policy not in s["policy"].values:
            continue
        row = s[s["policy"] == policy].iloc[0]
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        c = COLORS.get(policy, "#333333")
        ls = styles.get(policy, "-")
        lw = widths.get(policy, 1.8)
        ax.plot(angles, vals, marker="o", linestyle=ls, linewidth=lw,
                label=policy, color=c, alpha=0.9 if policy != "Thompson" else 1.0)
        ax.fill(angles, vals, alpha=0.05 if policy == "Thompson" else 0.08, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=LEGEND_FONT)
    ax.set_title("Multi-Metric Policy Comparison\n(DATS and Thompson nearly overlap; Thompson drawn dashed)",
                 y=1.08, fontsize=TITLE_FONT)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "policy_radar.pdf", bbox_inches="tight")
    plt.close()
    print("  policy_radar.pdf")


def fig_variance_analysis(main_comp):
    """3-panel layout. Panel 1: Quality across all policies. Panels 2 & 3
    split SLA by magnitude — high-SLA baselines (0–60% axis) and low-SLA
    policies (0–3.5% axis) — so that DATS at 0.36% stays visually distinct
    from Thompson at 0.44% and QoS-static at 0.91%. Stacking these on one
    shared axis (as earlier revisions attempted) made the low-SLA cluster
    collapse into identical-looking near-zero bars."""
    all_policies = [
        "Cloud-only", "Complexity-static", "FrugalGPT-cascade", "QoS-static",
        "Cost-aware", "UCB1", "Thompson", "epsilon-greedy", "DATS (ours)",
    ]
    policies = [p for p in all_policies if p in main_comp["policy"].unique()]

    def agg(p, col):
        vals = main_comp[main_comp["policy"] == p][col]
        if col == "sla_violation":
            vals = vals * 100
        return vals.mean(), vals.std()

    # SLA grouping threshold: 4% (cleanly separates our two clusters).
    high = [p for p in policies if agg(p, "sla_violation")[0] >= 4]
    low = [p for p in policies if agg(p, "sla_violation")[0] < 4]

    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.3], wspace=0.28)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_hi = fig.add_subplot(gs[0, 1])
    ax_lo = fig.add_subplot(gs[0, 2])

    # --- Quality (all policies) ---
    x = np.arange(len(policies))
    q_mu = [agg(p, "quality")[0] for p in policies]
    q_sd = [agg(p, "quality")[1] for p in policies]
    bars_q = ax_q.bar(x, q_mu, yerr=q_sd, capsize=3,
                      color=[COLORS.get(p, "#555") for p in policies],
                      edgecolor="black", linewidth=0.5, alpha=0.85)
    for rect, v in zip(bars_q, q_mu):
        ax_q.text(rect.get_x() + rect.get_width() / 2, v + 0.0004,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=SMALL_ANNOT_FONT)
    ax_q.set_ylim(0.965, 0.982)
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(policies, rotation=30, ha="right")
    ax_q.set_ylabel("Response Quality")
    ax_q.set_title("Quality Across Seeds (mean ± 1σ, n=5)")
    ax_q.grid(True, alpha=0.3, axis="y")

    def _sla_panel(ax, sub_policies, title, ymax_pad=1.25):
        xs = np.arange(len(sub_policies))
        mus = [agg(p, "sla_violation")[0] for p in sub_policies]
        sds = [agg(p, "sla_violation")[1] for p in sub_policies]
        bars = ax.bar(xs, mus, yerr=sds, capsize=3,
                      color=[COLORS.get(p, "#555") for p in sub_policies],
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        for rect, v, sd in zip(bars, mus, sds):
            label = f"{v:.2f}" if v < 10 else f"{v:.1f}"
            ax.text(rect.get_x() + rect.get_width() / 2,
                    v + sd + (max(mus) * 0.025),
                    label, ha="center", va="bottom", fontsize=SMALL_ANNOT_FONT)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub_policies, rotation=30, ha="right")
        ax.set_ylabel("SLA Violation Rate (%)")
        ax.set_title(title)
        ax.set_ylim(0, max(mus) * ymax_pad)
        ax.grid(True, alpha=0.3, axis="y")

    _sla_panel(ax_hi, high, "SLA — high-violation baselines")
    _sla_panel(ax_lo, low, "SLA — low-violation policies")
    # 1% reference line on the low-SLA panel for context.
    ax_lo.axhline(y=1.0, color="#888", linestyle=":", linewidth=1,
                  alpha=0.8, label="1% reference")
    ax_lo.legend(loc="upper right", fontsize=LEGEND_FONT)

    plt.savefig(FIGURES_DIR / "variance_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("  variance_analysis.pdf")


def fig_adaptation(regret):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"UCB1": "#3498db", "epsilon-greedy": "#1abc9c", "Thompson": "#f39c12", "DATS": "#e91e63"}
    for name in ["UCB1", "epsilon-greedy", "Thompson", "DATS"]:
        mean = regret.get(name)
        std = regret.get(f"{name}_std")
        if mean is None:
            continue
        x = np.arange(len(mean))
        ax.plot(x, mean, label=name, color=colors[name], linewidth=2)
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[name])
    ax.set_xlabel("Requests"); ax.set_ylabel("Cumulative Regret")
    ax.set_title("Adaptation Speed Comparison")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "adaptation_regret.pdf", bbox_inches="tight")
    plt.close()
    print("  adaptation_regret.pdf")


def fig_lambda_sweep(df_lambda):
    """Twin-axis line with shaded 1-sigma bands. Effects are small (mostly
    within noise), so the band must be visible for the plot to be honest."""
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax2 = ax1.twinx()
    agg = df_lambda.groupby("lambda").agg(
        quality_mean=("quality", "mean"), quality_std=("quality", "std"),
        sla_mean=("sla_violation", "mean"), sla_std=("sla_violation", "std"),
    ).reset_index()
    lam = agg["lambda"].values
    q_mu = agg["quality_mean"].values
    q_sd = agg["quality_std"].values
    s_mu = (agg["sla_mean"] * 100).values
    s_sd = (agg["sla_std"] * 100).values

    ax1.plot(lam, q_mu, marker="o", color="C0", linewidth=2, label="Quality")
    ax1.fill_between(lam, q_mu - q_sd, q_mu + q_sd, color="C0", alpha=0.18)
    ax2.plot(lam, s_mu, marker="s", color="C3", linewidth=2, label="SLA Violation")
    ax2.fill_between(lam, s_mu - s_sd, s_mu + s_sd, color="C3", alpha=0.18)

    ax1.set_xlabel("Deadline Penalty ($\\lambda$)")
    ax1.set_ylabel("Quality", color="C0")
    ax2.set_ylabel("SLA Violation Rate (%)", color="C3")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Effect of Deadline Penalty $\\lambda$ (shaded: $\\pm 1\\sigma$ across seeds)")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lambda_sweep.pdf", bbox_inches="tight")
    plt.close()
    print("  lambda_sweep.pdf")


# ---------------------------------------------------------------------------
# New figures
# ---------------------------------------------------------------------------


def fig_predictor(predictor_eval):
    cm = np.array(predictor_eval["val"]["confusion_matrix"])
    noise = predictor_eval["noise_robustness"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1, 2]); ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(["Simple", "Moderate", "Complex"])
    ax1.set_yticklabels(["Simple", "Moderate", "Complex"])
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title(f"Confusion Matrix (acc={predictor_eval['val']['accuracy']:.2f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax1, fraction=0.046)

    sigmas = [p["sigma"] for p in noise]
    accs = [p["accuracy"] for p in noise]
    ax2.plot(sigmas, accs, "o-", color="#e91e63", linewidth=2)
    ax2.set_xlabel("Feature Noise $\\sigma$")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Robustness Under Feature Perturbation")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predictor_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print("  predictor_accuracy.pdf")


def fig_multi_domain(df_multi):
    """2-row layout.
    Row 1: Quality across all four policies (shared linear axis, zoomed).
    Row 2: Per-domain SLA panels showing Thompson vs DATS ONLY, so each
    panel zooms into the sub-3% (narrative, customer-QA) or sub-60% (code)
    range where the head-to-head comparison is actually visible. Including
    Cloud-only and FrugalGPT in these SLA panels forces the axis to 0–60%
    or 0–100%, which compresses Thompson and DATS into near-zero bars.
    Their SLA behaviour is tabulated in the text and plotted with the rest
    of the baselines in Figure~variance_analysis."""
    all_policies = ["Cloud-only", "FrugalGPT-cascade", "Thompson", "DATS (ours)"]
    compare_policies = ["Thompson", "DATS (ours)"]
    domains = df_multi["domain"].unique().tolist()
    n_dom = len(domains)

    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(2, n_dom, height_ratios=[1.1, 1.0], hspace=0.85,
                          wspace=0.28)

    # --- Row 1: Quality (spans all columns) ---
    ax_q = fig.add_subplot(gs[0, :])
    x_dom = np.arange(n_dom)
    width = 0.2
    for i, p in enumerate(all_policies):
        sub = df_multi[df_multi["policy"] == p]
        q = [sub[sub["domain"] == d]["quality"].mean() for d in domains]
        qe = [sub[sub["domain"] == d]["quality"].std() for d in domains]
        ax_q.bar(x_dom + (i - 1.5) * width, q, width, yerr=qe, capsize=2.5,
                 label=p, color=COLORS.get(p, "#333"),
                 edgecolor="black", linewidth=0.4)
    ax_q.set_title("Quality per Domain")
    ax_q.set_ylabel("Quality")
    ax_q.set_xticks(x_dom); ax_q.set_xticklabels(domains)
    ax_q.set_ylim(0.960, 0.985)
    ax_q.grid(True, alpha=0.3, axis="y")
    ax_q.legend(fontsize=LEGEND_FONT, ncol=4, loc="upper center",
                bbox_to_anchor=(0.5, -0.18))

    # --- Row 2: per-domain SLA (Thompson vs DATS only, zoomed) ---
    for di, d in enumerate(domains):
        ax = fig.add_subplot(gs[1, di])
        vals_local = []
        for i, p in enumerate(compare_policies):
            sub = df_multi[df_multi["policy"] == p]
            v = sub[sub["domain"] == d]["sla_violation"].mean() * 100
            se = sub[sub["domain"] == d]["sla_violation"].std() * 100
            ax.bar(i, v, 0.55, yerr=se, capsize=2.5,
                   color=COLORS.get(p, "#333"),
                   edgecolor="black", linewidth=0.4)
            vals_local.append((v, se))
        for i, (v, se) in enumerate(vals_local):
            label = f"{v:.2f}" if v < 10 else f"{v:.1f}"
            ax.text(i, v + se + 0.04 * max(x[0] for x in vals_local),
                    label, ha="center", va="bottom", fontsize=ANNOT_FONT)
        ax.set_title(f"SLA — {d} (TS vs DATS)")
        ax.set_ylabel("SLA Viol. (%)" if di == 0 else "")
        ax.set_xticks(range(len(compare_policies)))
        ax.set_xticklabels(["TS", "DATS"])
        ymax = max(v for v, _ in vals_local) * 1.30
        ax.set_ylim(0, max(ymax, 1.0))
        ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(FIGURES_DIR / "multi_domain.pdf", bbox_inches="tight")
    plt.close()
    print("  multi_domain.pdf")


def fig_sensitivity(df_sens):
    """Left: SLA violations across scenarios. Right: DATS - Thompson gap
    (in pp) per scenario, which is the actionable signal. The earlier
    version's p99 panel was useless because p99 is capped at 500 ms for
    both policies in every scenario."""
    scenarios = df_sens["scenario"].unique().tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(scenarios))
    width = 0.35
    data = {}
    for policy in ["Thompson", "DATS"]:
        sub = df_sens[df_sens["policy"] == policy]
        s_mu = np.array([sub[sub["scenario"] == sc]["sla_violation"].mean() * 100
                         for sc in scenarios])
        s_sd = np.array([sub[sub["scenario"] == sc]["sla_violation"].std() * 100
                         for sc in scenarios])
        data[policy] = (s_mu, s_sd)

    for i, policy in enumerate(["Thompson", "DATS"]):
        s_mu, s_sd = data[policy]
        color = COLORS.get(policy, "#333")
        bars = ax1.bar(x + (i - 0.5) * width, s_mu, width, yerr=s_sd,
                       capsize=2.5, label=policy, color=color,
                       edgecolor="black", linewidth=0.4)
        # Place value label above the error bar so it isn't crossed out.
        for rect, v, sd in zip(bars, s_mu, s_sd):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + sd + 0.04,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=SMALL_ANNOT_FONT)
    ax1.axhline(y=1.0, color="#777", linestyle=":", linewidth=1, alpha=0.8,
                label="1% reference")
    ax1.set_xticks(x); ax1.set_xticklabels(scenarios, rotation=30, ha="right")
    ax1.set_ylabel("SLA Violation Rate (%)")
    ax1.set_title("SLA Violations Across Sensitivity Scenarios")
    ax1.set_ylim(0, 1.75)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.16), ncol=3, fontsize=LEGEND_FONT)

    gap = data["Thompson"][0] - data["DATS"][0]
    gap = np.where(np.abs(gap) < 0.005, 0.0, gap)
    colors_gap = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gap]
    bars = ax2.bar(x, gap, 0.6, color=colors_gap, edgecolor="black", linewidth=0.4)
    gap_max = max(abs(v) for v in gap) or 1.0
    for rect, v in zip(bars, gap):
        y = gap_max * 0.05 if v == 0 else v + (gap_max * 0.04 if v > 0 else -gap_max * 0.04)
        ax2.text(rect.get_x() + rect.get_width() / 2,
                 y, f"{v:+.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=ANNOT_FONT)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(scenarios, rotation=30, ha="right")
    ax2.set_ylabel("Thompson − DATS SLA gap (pp)")
    ax2.set_title("DATS Advantage by Scenario (green = DATS wins)")
    ax2.set_ylim(-gap_max * 0.16, gap_max * 1.35)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensitivity.pdf", bbox_inches="tight")
    plt.close()
    print("  sensitivity.pdf")


def fig_gpu_latency_calibration(df_gpu):
    """Empirical local-GPU latency/energy trace summary."""
    df = df_gpu.copy()
    df["label"] = df["tier"] + "\n" + df["target_new_tokens"].astype(str) + " tok\nB" + df["batch_size"].astype(str)
    df = df.sort_values(["tier", "target_new_tokens", "batch_size"])
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    colors = ["#2c7fb8" if t == "edge" else "#d95f0e" for t in df["tier"]]
    ax1.bar(x, df["latency_p50"], color=colors, edgecolor="black", linewidth=0.4, label="p50")
    ax1.scatter(x, df["latency_p99"], color="#111111", s=36, marker="D", label="p99", zorder=4)
    ax1.axhline(500, color="#777", linestyle=":", linewidth=1, label="500 ms SLA")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=0, fontsize=TICK_FONT)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Measured RTX 5090 Inference Latency")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    ax2.scatter(df["latency_p50"], df["energy_j_mean"],
                s=df["peak_vram_mb_max"] / 20,
                color=colors, alpha=0.75, edgecolors="black", linewidths=0.5)
    label_offsets = {
        ("edge", 16, 1): (-18, 14),
        ("edge", 16, 2): (18, -22),
        ("edge", 32, 1): (10, 8),
        ("edge", 32, 2): (10, 14),
        ("cloud", 16, 1): (10, -16),
        ("cloud", 16, 2): (10, 8),
        ("cloud", 32, 1): (-72, 10),
        ("cloud", 32, 2): (-72, -16),
    }
    for _, row in df.iterrows():
        key = (row["tier"], int(row["target_new_tokens"]), int(row["batch_size"]))
        off = label_offsets.get(key, (5, 4))
        ha = "right" if off[0] < 0 else "left"
        ax2.annotate(f"{row['tier']} {int(row['target_new_tokens'])}t B{int(row['batch_size'])}",
                     (row["latency_p50"], row["energy_j_mean"]),
                     textcoords="offset points", xytext=off, fontsize=SMALL_ANNOT_FONT,
                     ha=ha, bbox=LABEL_BBOX)
    xr = df["latency_p50"].max() - df["latency_p50"].min()
    yr = df["energy_j_mean"].max() - df["energy_j_mean"].min()
    ax2.set_xlim(df["latency_p50"].min() - xr * 0.10, df["latency_p50"].max() + xr * 0.18)
    ax2.set_ylim(df["energy_j_mean"].min() - yr * 0.12, df["energy_j_mean"].max() + yr * 0.16)
    ax2.set_xlabel("p50 latency (ms)")
    ax2.set_ylabel("Energy per request (J)")
    ax2.set_title("Latency-Energy Trade-off")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpu_latency_calibration.pdf", bbox_inches="tight")
    plt.close()
    print("  gpu_latency_calibration.pdf")


def fig_gpu_trace_phase_map(df_replay):
    """Deadline-by-domain phase map for empirical trace replay."""
    df = df_replay.copy()
    df = df[df["stress"] == "baseline"].copy()
    pivot = df.pivot_table(index="domain", columns=["deadline_ms", "policy"],
                           values="sla_violation", aggfunc="mean")
    domains = list(pivot.index)
    deadlines = sorted({c[0] for c in pivot.columns})
    policies = ["DATS (ours)", "Thompson", "QoS-static", "FrugalGPT-cascade"]

    mats = {}
    vmax = 1.0
    for policy in policies:
        mat = []
        for dom in domains:
            row = []
            for dl in deadlines:
                row.append(float(pivot.get((dl, policy), pd.Series(index=domains, dtype=float)).get(dom, np.nan)) * 100)
            mat.append(row)
        mats[policy] = mat
        vmax = max(vmax, float(np.nanmax(mat)))

    fig, axes = plt.subplots(1, len(policies), figsize=(15.8, 4.2), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    for ax, policy in zip(axes, policies):
        mat = mats[policy]
        im = ax.imshow(mat, aspect="auto", cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(policy)
        ax.set_xticks(range(len(deadlines)))
        ax.set_xticklabels([str(int(d)) for d in deadlines], rotation=45)
        ax.set_xlabel("Deadline (ms)")
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains)
        for i in range(len(domains)):
            for j in range(len(deadlines)):
                v = mat[i][j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=SMALL_ANNOT_FONT,
                            color="white" if v > 8 else "black")
    axes[0].set_ylabel("Domain")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.20, top=0.78, wspace=0.18)
    cax = fig.add_axes([0.925, 0.20, 0.015, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SLA violation (%)")
    fig.suptitle("Empirical GPU Trace Replay: Deadline Phase Map", y=0.96)
    plt.savefig(FIGURES_DIR / "gpu_trace_phase_map.pdf", bbox_inches="tight")
    plt.close()
    print("  gpu_trace_phase_map.pdf")


def fig_gpu_energy_pareto(df_energy):
    """Quality/SLA/energy Pareto view from empirical replay."""
    df = df_energy.copy()
    df = df[(df["stress"] == "baseline") & (df["deadline_ms"] == 500.0)].copy()
    agg = df.groupby("policy", as_index=False).agg({
        "quality": "mean",
        "sla_violation": "mean",
        "energy_j_per_request": "mean",
        "total_cost": "mean",
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    label_offsets = {
        "DATS (ours)": (10, 12),
        "epsilon-greedy": (10, -16),
        "UCB1": (10, 8),
        "Thompson": (10, -2),
        "QoS-static": (-58, 10),
        "FrugalGPT-cascade": (10, 10),
    }
    for _, row in agg.iterrows():
        policy = row["policy"]
        size = 80 + row["energy_j_per_request"] * 5
        ax.scatter(row["sla_violation"] * 100, row["quality"], s=size,
                   color=COLORS.get(policy, "#555"), edgecolors="black",
                   linewidths=0.6, alpha=0.82)
        off = label_offsets.get(policy, (6, 4))
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(policy, (row["sla_violation"] * 100, row["quality"]),
                    textcoords="offset points", xytext=off, fontsize=ANNOT_FONT,
                    ha=ha, bbox=LABEL_BBOX)
    xr = (agg["sla_violation"] * 100).max() - (agg["sla_violation"] * 100).min()
    yr = agg["quality"].max() - agg["quality"].min()
    ax.set_xlim((agg["sla_violation"] * 100).min() - xr * 0.08,
                (agg["sla_violation"] * 100).max() + xr * 0.10)
    ax.set_ylim(agg["quality"].min() - yr * 0.10,
                agg["quality"].max() + yr * 0.16)
    ax.set_xlabel("SLA violation (%)")
    ax.set_ylabel("Quality")
    ax.set_title("Empirical Replay Pareto: Quality, SLA, Energy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpu_energy_pareto.pdf", bbox_inches="tight")
    plt.close()
    print("  gpu_energy_pareto.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading data...")
    data = load_all()
    print("Generating figures...")
    fig_quality_sla_scatter(data["summary"])
    fig_tier_distribution(data["policy_json"])
    fig_nonstationary(data["nonstat"])
    fig_cost_quality_pareto(data["policy_json"])
    fig_latency_comparison(data["summary"])
    fig_policy_radar(data["summary"])
    fig_variance_analysis(data["main"])
    if "regret" in data:
        fig_adaptation(data["regret"])
    fig_lambda_sweep(data["lambda"])
    if "predictor_eval" in data:
        fig_predictor(data["predictor_eval"])
    if "multi_domain" in data:
        fig_multi_domain(data["multi_domain"])
    if "sensitivity" in data:
        fig_sensitivity(data["sensitivity"])
    if "gpu_latency" in data:
        fig_gpu_latency_calibration(data["gpu_latency"])
    if "gpu_replay" in data:
        fig_gpu_trace_phase_map(data["gpu_replay"])
    if "gpu_energy" in data:
        fig_gpu_energy_pareto(data["gpu_energy"])
    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print(f"Total: {len(list(FIGURES_DIR.glob('*.pdf')))} PDF files")


if __name__ == "__main__":
    main()
