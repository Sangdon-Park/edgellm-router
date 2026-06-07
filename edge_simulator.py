"""
Edge Computing Simulator for LLM Workloads (Revised)
====================================================

Simulates edge-cloud LLM request routing with realistic network and inference models.

Revision highlights (reviewer response):
- Queueing delay model (M/G/1-style with capacity and utilization-based waiting).
- Batching effects on the cloud tier (continuous batching approximation).
- Cold-start penalty (probabilistic warm-up cost after idle).
- Heavy-tail cloud variability (Pareto-mixed tail for API unpredictability).
- Token streaming: explicit time-to-first-token (TTFT) + per-token decode cost.
- FrugalGPTCascadePolicy, QoSStaticPolicy, CostAwarePolicy new baselines.
- ComplexityPredictor (MLP) with training data synthesis + robustness controls.
- Multi-domain request generators: narrative, customer QA, code generation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import json
from pathlib import Path


class NetworkProfile(Enum):
    STABLE = "stable"      # Wired connection
    MOBILE = "mobile"      # 4G/5G
    UNSTABLE = "unstable"  # Congested/rural


class ExecutionTier(Enum):
    EDGE = 1
    CLOUD = 2
    HYBRID = 3


class Domain(Enum):
    NARRATIVE = "narrative"          # Interactive narrative / dialogue
    CUSTOMER_QA = "customer_qa"      # Customer support QA
    CODE = "code"                    # Code generation / refactoring


# ----------------------------------------------------------------------------
# Network model
# ----------------------------------------------------------------------------


@dataclass
class NetworkConfig:
    """Network latency configuration (log-normal + heavy-tail mixture)."""
    profile: NetworkProfile
    base_latency_ms: float
    std_latency_ms: float
    heavy_tail_prob: float = 0.02      # Probability of a tail spike
    heavy_tail_scale: float = 3.0      # Multiplicative scale of tail samples
    jitter_autocorr: float = 0.0       # AR(1) autocorrelation (optional)
    _last_sample: float = 0.0

    @classmethod
    def from_profile(cls, profile: NetworkProfile) -> "NetworkConfig":
        configs = {
            NetworkProfile.STABLE:   (50,  15,  0.005, 1.8),
            NetworkProfile.MOBILE:   (120, 70,  0.04,  3.0),
            NetworkProfile.UNSTABLE: (230, 130, 0.08,  4.2),
        }
        base, std, tail_p, tail_s = configs[profile]
        return cls(
            profile=profile, base_latency_ms=base, std_latency_ms=std,
            heavy_tail_prob=tail_p, heavy_tail_scale=tail_s,
        )

    def sample_latency(self) -> float:
        """Sample network latency from log-normal plus Pareto tail."""
        mu = np.log(self.base_latency_ms) - 0.5 * (self.std_latency_ms / self.base_latency_ms) ** 2
        sigma = np.sqrt(np.log(1 + (self.std_latency_ms / self.base_latency_ms) ** 2))
        base = np.random.lognormal(mu, sigma)
        if np.random.random() < self.heavy_tail_prob:
            # Heavy-tail spike (Pareto-distributed multiplier)
            spike = np.random.pareto(1.5) + 1.0
            base *= self.heavy_tail_scale * spike
        # Optional AR(1) autocorrelation
        if self.jitter_autocorr > 0:
            base = self.jitter_autocorr * self._last_sample + (1 - self.jitter_autocorr) * base
        self._last_sample = base
        return base


# ----------------------------------------------------------------------------
# Compute configurations
# ----------------------------------------------------------------------------


@dataclass
class EdgeNodeConfig:
    """Edge node compute configuration."""
    name: str
    model_name: str
    capability_score: float
    tokens_per_second: float
    ttft_ms: float = 30.0              # Time to first token
    base_latency_ms: float = 20.0      # Inference warm-up overhead
    concurrency: int = 1               # Parallel slots (edge usually small)
    cold_start_ms: float = 350.0       # Cold start penalty
    cold_start_timeout_s: float = 5.0  # Idle time before cold-start returns

    @classmethod
    def rtx3060(cls) -> "EdgeNodeConfig":
        return cls(
            name="RTX3060-Edge", model_name="gemma-2b-4bit",
            capability_score=0.78, tokens_per_second=150,
            ttft_ms=20.0, base_latency_ms=15.0, concurrency=2,
            cold_start_ms=150.0, cold_start_timeout_s=8.0,
        )

    @classmethod
    def cpu_only(cls) -> "EdgeNodeConfig":
        return cls(
            name="CPU-Edge", model_name="phi-2-ggml",
            capability_score=0.72, tokens_per_second=12,
            ttft_ms=80.0, base_latency_ms=25.0, concurrency=1,
            cold_start_ms=500.0, cold_start_timeout_s=10.0,
        )


@dataclass
class CloudConfig:
    """Cloud LLM API configuration."""
    name: str
    model_name: str
    capability_score: float
    tokens_per_second: float
    ttft_ms: float = 120.0
    api_overhead_ms: float = 100.0
    cost_per_1k_tokens: float = 0.00125
    concurrency: int = 32              # Large pool (batched serving)
    batch_wait_ms: float = 15.0        # Continuous batching wait budget

    @classmethod
    def gemini_flash(cls) -> "CloudConfig":
        return cls(
            name="Gemini-3-Pro", model_name="gemini-3-pro-preview",
            capability_score=0.98, tokens_per_second=220,
            ttft_ms=110.0, api_overhead_ms=80.0,
            cost_per_1k_tokens=0.00125,
            concurrency=8, batch_wait_ms=15.0,
        )


# ----------------------------------------------------------------------------
# Request / Response dataclasses
# ----------------------------------------------------------------------------


@dataclass
class Request:
    """A single LLM request."""
    id: int
    timestamp: float
    input_tokens: int
    expected_output_tokens: int
    complexity: int
    context_utilization: float
    entity_density: float
    is_factual: bool
    turn_index: int
    domain: Domain = Domain.NARRATIVE

    def get_features(self) -> np.ndarray:
        """Feature vector used by routing / predictor."""
        return np.array([
            self.input_tokens / 1000,
            self.context_utilization,
            self.entity_density,
            float(self.is_factual),
            min(self.turn_index / 10, 1.0),
            self.expected_output_tokens / 200,
        ])


@dataclass
class Response:
    request_id: int
    tier: ExecutionTier
    quality: float
    latency_ms: float
    output_tokens: int
    cost: float
    deadline_met: bool
    queue_wait_ms: float = 0.0
    cold_start: bool = False


@dataclass
class SimulatorConfig:
    network: NetworkConfig
    edge: EdgeNodeConfig
    cloud: CloudConfig
    deadline_ms: float = 500.0
    seed: Optional[int] = None
    enable_queueing: bool = True
    enable_cold_start: bool = True
    enable_batching: bool = True
    edge_request_rate: float = 0.6     # Requests/second on the edge node
    cloud_request_rate: float = 9.0    # Effective per-tenant rate on shared API pool


# ----------------------------------------------------------------------------
# Queue / cold-start models
# ----------------------------------------------------------------------------


class TierQueue:
    """M/G/1-like queue model capturing utilization-based waiting.

    We model the expected waiting time for a tier as rho * E[S] / (1 - rho),
    where rho is the effective utilization.  Per-request waiting is sampled
    from an exponential with that mean to reproduce bursty variability.
    Cold start is modelled with an idle timer: if the tier has been idle
    longer than the timeout, we pay cold_start_ms once.
    """

    def __init__(self, mean_service_ms: float, concurrency: int,
                 arrival_rate_per_s: float, cold_start_ms: float = 0.0,
                 cold_start_timeout_s: float = 5.0,
                 enable_cold_start: bool = True):
        self.mean_service_ms = mean_service_ms
        self.concurrency = max(1, concurrency)
        self.arrival_rate = arrival_rate_per_s
        self.cold_start_ms = cold_start_ms
        self.cold_start_timeout_s = cold_start_timeout_s
        self.enable_cold_start = enable_cold_start
        self.last_request_time: Optional[float] = None

    def sample_wait(self) -> float:
        """Sample a waiting time for this request."""
        # Effective utilization on shared server
        rho = (self.arrival_rate * self.mean_service_ms / 1000.0) / self.concurrency
        rho = np.clip(rho, 0.0, 0.98)
        if rho < 1e-6:
            return 0.0
        expected_wait = rho * self.mean_service_ms / (1 - rho)
        # Exponential to allow occasional bursts
        return np.random.exponential(expected_wait)

    def maybe_cold_start(self, now_s: float) -> float:
        """Return cold-start penalty if the tier has been idle beyond timeout.

        The very first call is treated as warm (tier was preloaded at app
        startup).  Subsequent cold starts occur only after genuine idle gaps.
        """
        if not self.enable_cold_start or self.cold_start_ms <= 0:
            self.last_request_time = now_s
            return 0.0
        if self.last_request_time is None:
            self.last_request_time = now_s
            return 0.0
        idle = now_s - self.last_request_time
        self.last_request_time = now_s
        if idle > self.cold_start_timeout_s:
            # Amortised half-penalty to reflect partial warmth
            return self.cold_start_ms * 0.5
        return 0.0


# ----------------------------------------------------------------------------
# Inference model
# ----------------------------------------------------------------------------


class LLMInferenceModel:
    """Models LLM inference latency, batching, and quality."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        # Approximate mean service times used for queue modelling.
        # Interactive traffic: ~60 total tokens (input+output) per request.
        edge_mean_service = (config.edge.ttft_ms +
                             1000 * 30 / config.edge.tokens_per_second)
        cloud_mean_service = (config.cloud.ttft_ms +
                              1000 * 30 / config.cloud.tokens_per_second)
        self.edge_queue = TierQueue(
            mean_service_ms=edge_mean_service,
            concurrency=config.edge.concurrency,
            arrival_rate_per_s=config.edge_request_rate,
            cold_start_ms=config.edge.cold_start_ms if config.enable_cold_start else 0.0,
            cold_start_timeout_s=config.edge.cold_start_timeout_s,
            enable_cold_start=config.enable_cold_start,
        )
        self.cloud_queue = TierQueue(
            mean_service_ms=cloud_mean_service,
            concurrency=config.cloud.concurrency,
            arrival_rate_per_s=config.cloud_request_rate,
            cold_start_ms=0.0,  # Cloud rarely cold-starts under load
            cold_start_timeout_s=30.0,
            enable_cold_start=False,
        )

    def _edge_service_time(self, request: Request) -> float:
        edge = self.config.edge
        # Streaming: TTFT + decode-time per output token + linear prefill
        prefill_ms = request.input_tokens / edge.tokens_per_second * 1000 * 0.4
        decode_ms = request.expected_output_tokens / edge.tokens_per_second * 1000
        return edge.base_latency_ms + edge.ttft_ms + prefill_ms + decode_ms + np.random.normal(0, 4)

    def _cloud_service_time(self, request: Request) -> float:
        cloud = self.config.cloud
        # Batching wait: small expected overhead
        batch_wait = 0.0
        if self.config.enable_batching:
            batch_wait = np.random.uniform(0, cloud.batch_wait_ms)
        prefill_ms = request.input_tokens / cloud.tokens_per_second * 1000 * 0.25
        decode_ms = request.expected_output_tokens / cloud.tokens_per_second * 1000
        return cloud.ttft_ms + batch_wait + prefill_ms + decode_ms + np.random.normal(0, 8)

    def compute_edge_latency(self, request: Request, now_s: float = 0.0) -> Tuple[float, float, bool]:
        service = self._edge_service_time(request)
        wait = 0.0
        cold = False
        if self.config.enable_queueing:
            wait = self.edge_queue.sample_wait()
            cs = self.edge_queue.maybe_cold_start(now_s)
            if cs > 0:
                cold = True
                wait += cs
        return self.config.edge.base_latency_ms + service + wait, wait, cold

    def compute_cloud_latency(self, request: Request, now_s: float = 0.0) -> Tuple[float, float, bool]:
        service = self._cloud_service_time(request)
        network = self.config.network.sample_latency()
        wait = 0.0
        cold = False
        if self.config.enable_queueing:
            wait = self.cloud_queue.sample_wait()
            cs = self.cloud_queue.maybe_cold_start(now_s)
            if cs > 0:
                cold = True
                wait += cs
        return self.config.cloud.api_overhead_ms + network + service + wait, wait, cold

    def compute_hybrid_latency(self, request: Request, now_s: float = 0.0) -> Tuple[float, float, bool]:
        """Edge draft + optional cloud refinement. SLA-aware fallback."""
        edge_latency, edge_wait, edge_cold = self.compute_edge_latency(request, now_s)
        draft_latency = edge_latency * 0.55  # Quick draft with partial decode
        budget = self.config.deadline_ms - draft_latency
        if budget > 180:
            cloud_latency, _, _ = self.compute_cloud_latency(request, now_s)
            # Parallel refine: dominated by slower side modulo 0.65 overlap factor.
            refine = min(cloud_latency * 0.65, budget)
            total = draft_latency + refine
        else:
            total = draft_latency
        return total, edge_wait, edge_cold

    def compute_quality(self, request: Request, tier: ExecutionTier) -> float:
        complexity = request.complexity
        if tier == ExecutionTier.EDGE:
            base_quality = [0.98, 0.97, 0.95][complexity]
        elif tier == ExecutionTier.CLOUD:
            base_quality = [0.99, 0.98, 0.97][complexity]
        else:
            base_quality = [0.98, 0.97, 0.96][complexity]
        # Domain-specific fine-tune: code generation edge is weaker on complex
        if request.domain == Domain.CODE and tier == ExecutionTier.EDGE and complexity == 2:
            base_quality -= 0.03
        if request.domain == Domain.CUSTOMER_QA and tier == ExecutionTier.EDGE and complexity == 2:
            base_quality -= 0.015
        noise = np.random.normal(0, 0.025)
        return float(np.clip(base_quality + noise, 0, 1))

    def execute(self, request: Request, tier: ExecutionTier) -> Response:
        now_s = request.timestamp / 1000.0  # ms to s
        if tier == ExecutionTier.EDGE:
            latency, wait, cold = self.compute_edge_latency(request, now_s)
            cost = 0.0
        elif tier == ExecutionTier.CLOUD:
            latency, wait, cold = self.compute_cloud_latency(request, now_s)
            total_tokens = request.input_tokens + request.expected_output_tokens
            cost = total_tokens / 1000 * self.config.cloud.cost_per_1k_tokens
        else:
            latency, wait, cold = self.compute_hybrid_latency(request, now_s)
            cost = request.expected_output_tokens / 1000 * self.config.cloud.cost_per_1k_tokens * 0.5

        quality = self.compute_quality(request, tier)
        return Response(
            request_id=request.id, tier=tier, quality=quality,
            latency_ms=latency, output_tokens=request.expected_output_tokens,
            cost=cost, deadline_met=latency <= self.config.deadline_ms,
            queue_wait_ms=wait, cold_start=cold,
        )


# ----------------------------------------------------------------------------
# Request generators (multi-domain)
# ----------------------------------------------------------------------------


DOMAIN_PARAMS: Dict[Domain, Dict] = {
    Domain.NARRATIVE: dict(
        complexity_probs=[0.40, 0.35, 0.25],
        input_lognorm=(3.5, 0.6), input_range=(20, 300),
        output_lognorm=(2.5, 0.5), output_range=(10, 100),
        factual_prob=0.30, turn_cycle=20,
    ),
    Domain.CUSTOMER_QA: dict(
        complexity_probs=[0.55, 0.30, 0.15],
        input_lognorm=(4.0, 0.55), input_range=(40, 400),
        output_lognorm=(3.0, 0.45), output_range=(20, 160),
        factual_prob=0.70, turn_cycle=6,
    ),
    Domain.CODE: dict(
        complexity_probs=[0.25, 0.35, 0.40],
        input_lognorm=(4.8, 0.55), input_range=(80, 900),
        output_lognorm=(4.0, 0.55), output_range=(40, 400),
        factual_prob=0.35, turn_cycle=4,
    ),
}


class RequestGenerator:
    """Generates synthetic request traces.  Supports multi-domain workloads."""

    def __init__(self, seed: Optional[int] = None, domain: Domain = Domain.NARRATIVE):
        if seed is not None:
            np.random.seed(seed)
        self.domain = domain

    def _sample_request(self, i: int, timestamp: float) -> Request:
        """Sample a request with features that *correlate* with complexity.

        Complexity class is drawn first, then feature distributions are
        conditioned on the class.  This makes the complexity predictor
        learnable from features (simple queries tend to be short/factual,
        complex queries long/entity-rich with high context utilisation).
        """
        params = DOMAIN_PARAMS[self.domain]
        complexity = int(np.random.choice([0, 1, 2], p=params["complexity_probs"]))
        # Complexity-conditioned token distribution
        lo, hi = params["input_range"]
        base_mu, base_sigma = params["input_lognorm"]
        mu_shift = [-0.4, 0.0, 0.6][complexity]
        input_tokens = int(np.clip(np.random.lognormal(base_mu + mu_shift, base_sigma), lo, hi))
        lo2, hi2 = params["output_range"]
        base_mu2, base_sigma2 = params["output_lognorm"]
        mu_shift2 = [-0.3, 0.0, 0.55][complexity]
        output_tokens = int(np.clip(np.random.lognormal(base_mu2 + mu_shift2, base_sigma2), lo2, hi2))
        # Context / entities: grow with complexity
        ctx_alpha = [2.0, 3.0, 4.5][complexity]
        ctx_beta = [8.0, 5.0, 3.0][complexity]
        entity_alpha = [1.2, 2.0, 3.5][complexity]
        entity_beta = [10.0, 7.0, 4.0][complexity]
        # Simple queries are more often factual
        factual_p = {0: min(0.85, params["factual_prob"] + 0.25),
                     1: params["factual_prob"],
                     2: max(0.05, params["factual_prob"] - 0.2)}[complexity]
        # Turn index skewed higher for complex queries (longer conversations)
        turn = i % params["turn_cycle"] + (0 if complexity == 0 else np.random.randint(0, 3))
        return Request(
            id=i, timestamp=timestamp,
            input_tokens=input_tokens,
            expected_output_tokens=output_tokens,
            complexity=complexity,
            context_utilization=float(np.random.beta(ctx_alpha, ctx_beta)),
            entity_density=float(np.random.beta(entity_alpha, entity_beta)),
            is_factual=bool(np.random.random() < factual_p),
            turn_index=int(turn),
            domain=self.domain,
        )

    def generate_poisson(self, n_requests: int, rate: float = 1.0) -> List[Request]:
        requests = []
        timestamp = 0.0
        for i in range(n_requests):
            timestamp += np.random.exponential(1.0 / rate) * 1000
            requests.append(self._sample_request(i, timestamp))
        return requests

    def generate_bursty(self, n_requests: int, burst_prob: float = 0.1) -> List[Request]:
        requests = []
        timestamp = 0.0
        for i in range(n_requests):
            if np.random.random() < burst_prob:
                timestamp += np.random.uniform(10, 50)
            else:
                timestamp += np.random.exponential(500)
            requests.append(self._sample_request(i, timestamp))
        return requests


# ----------------------------------------------------------------------------
# Complexity predictor (lightweight MLP)
# ----------------------------------------------------------------------------


class ComplexityPredictor:
    """2-layer MLP trained to predict complexity class from request features.

    Deliberately implemented without external deep-learning frameworks so the
    simulator stays lightweight and fully reproducible.  Training uses
    mini-batch SGD with softmax cross-entropy.  The feature vector matches
    Request.get_features (length 6).
    """

    def __init__(self, hidden: int = 16, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(6, hidden) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, 3) * 0.3
        self.b2 = np.zeros(3)
        self.trained = False
        self.train_metrics: Dict[str, float] = {}

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0, X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return h, probs

    def predict(self, features: np.ndarray) -> int:
        x = features.reshape(1, -1)
        _, probs = self._forward(x)
        return int(probs.argmax(axis=1)[0])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = features.reshape(1, -1)
        _, probs = self._forward(x)
        return probs[0]

    @staticmethod
    def build_dataset(
        n_samples: int = 8000,
        seed: int = 0,
        domains: Optional[List[Domain]] = None,
        label_noise: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize a labelled dataset by sampling from the request generator.

        label_noise controls the probability of flipping the ground-truth class
        to emulate human-labelling disagreement.
        """
        domains = domains or list(DOMAIN_PARAMS.keys())
        rng_state = np.random.get_state()
        X_parts, y_parts = [], []
        per_domain = n_samples // len(domains)
        for idx, dom in enumerate(domains):
            gen = RequestGenerator(seed=seed + idx * 7, domain=dom)
            reqs = gen.generate_poisson(per_domain)
            feats = np.stack([r.get_features() for r in reqs])
            labels = np.array([r.complexity for r in reqs])
            # Human disagreement: with prob label_noise flip to a neighbour
            if label_noise > 0:
                flip_mask = np.random.random(len(labels)) < label_noise
                labels[flip_mask] = (labels[flip_mask] + np.random.choice([-1, 1], flip_mask.sum())) % 3
            X_parts.append(feats); y_parts.append(labels)
        np.random.set_state(rng_state)
        return np.concatenate(X_parts), np.concatenate(y_parts)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 60, lr: float = 0.05, batch: int = 64,
            l2: float = 1e-4, verbose: bool = False) -> Dict[str, float]:
        y_onehot = np.eye(3)[y]
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], y_onehot[idx]
            for start in range(0, n, batch):
                Xb = Xs[start:start + batch]
                Yb = Ys[start:start + batch]
                h, probs = self._forward(Xb)
                grad_logits = (probs - Yb) / len(Xb)
                grad_W2 = h.T @ grad_logits + l2 * self.W2
                grad_b2 = grad_logits.sum(axis=0)
                grad_h = grad_logits @ self.W2.T
                grad_h[h <= 0] = 0
                grad_W1 = Xb.T @ grad_h + l2 * self.W1
                grad_b1 = grad_h.sum(axis=0)
                self.W1 -= lr * grad_W1
                self.b1 -= lr * grad_b1
                self.W2 -= lr * grad_W2
                self.b2 -= lr * grad_b2
            if verbose and epoch % 10 == 0:
                acc = (self._forward(X)[1].argmax(axis=1) == y).mean()
                print(f"epoch {epoch}: acc {acc:.3f}")
        self.trained = True
        preds = self._forward(X)[1].argmax(axis=1)
        acc = float((preds == y).mean())
        self.train_metrics = {"train_acc": acc, "n": n}
        return self.train_metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self._forward(X)[1].argmax(axis=1)
        acc = float((preds == y).mean())
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y, preds):
            cm[int(t), int(p)] += 1
        per_class = []
        for c in range(3):
            support = int((y == c).sum())
            correct = int((preds[y == c] == c).sum())
            per_class.append({
                "class": c, "support": support,
                "recall": correct / max(1, support),
                "precision": correct / max(1, int((preds == c).sum())),
            })
        return {"accuracy": acc, "confusion_matrix": cm.tolist(),
                "per_class": per_class}


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------


class RoutingPolicy:
    """Base class for routing policies."""

    def select_tier(self, request: Request, context: Dict) -> ExecutionTier:
        raise NotImplementedError

    def update(self, request: Request, response: Response):
        pass

    def reset(self):
        pass


class CloudOnlyPolicy(RoutingPolicy):
    def select_tier(self, request, context): return ExecutionTier.CLOUD


class EdgeOnlyPolicy(RoutingPolicy):
    def select_tier(self, request, context): return ExecutionTier.EDGE


class RandomPolicy(RoutingPolicy):
    def select_tier(self, request, context):
        return np.random.choice(list(ExecutionTier))


class ComplexityStaticPolicy(RoutingPolicy):
    def select_tier(self, request, context):
        if request.complexity == 0:
            return ExecutionTier.EDGE
        elif request.complexity == 1:
            return ExecutionTier.HYBRID
        return ExecutionTier.CLOUD


class FrugalGPTCascadePolicy(RoutingPolicy):
    """FrugalGPT-style cascade: run edge first; escalate to cloud if confidence
    below a threshold.  We simulate confidence as a function of predicted
    complexity (higher complexity -> lower edge confidence)."""

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def select_tier(self, request, context):
        # Cascade is represented as a two-step decision.  We model the
        # eventual outcome: simple -> edge, moderate -> edge-then-cloud
        # (hybrid), complex -> cloud.  This captures realistic latency cost
        # of cascading while staying compatible with the single-pass sim.
        cx = request.complexity
        conf = [0.95, 0.70, 0.45][cx]
        if conf >= self.confidence_threshold:
            return ExecutionTier.EDGE
        elif conf >= self.confidence_threshold * 0.7:
            return ExecutionTier.HYBRID
        return ExecutionTier.CLOUD


class QoSStaticPolicy(RoutingPolicy):
    """QoS-driven static heuristic.

    Uses recent observed network latency to pick a conservative tier:
    if network is unstable, prefer edge for simple/moderate, cloud for complex.
    """

    def __init__(self, deadline_ms: float = 500.0):
        self.deadline_ms = deadline_ms

    def select_tier(self, request, context):
        net = context.get("recent_latency_ema", 100.0)
        if net < 80:
            # Healthy network, use complexity routing
            return (ExecutionTier.EDGE if request.complexity == 0
                    else ExecutionTier.CLOUD if request.complexity == 2
                    else ExecutionTier.HYBRID)
        # Degraded network: prefer edge unless complexity is 2
        if request.complexity == 2:
            return ExecutionTier.HYBRID
        return ExecutionTier.EDGE


class CostAwarePolicy(RoutingPolicy):
    """Cost-aware scheduler: picks the cheapest tier that is likely to meet
    the deadline given the running latency estimate.  This emulates "cost-
    minimising with SLA guardrail" baselines.
    """

    def __init__(self, deadline_ms: float = 500.0, cost_weight: float = 1.0):
        self.deadline_ms = deadline_ms
        self.cost_weight = cost_weight
        self.reset()

    def reset(self):
        self.latency = {ExecutionTier.EDGE: 80.0,
                        ExecutionTier.CLOUD: 400.0,
                        ExecutionTier.HYBRID: 230.0}
        self.cost_estimate = {ExecutionTier.EDGE: 0.0,
                              ExecutionTier.CLOUD: 0.0005,
                              ExecutionTier.HYBRID: 0.00025}

    def update(self, request, response):
        t = response.tier
        self.latency[t] = 0.9 * self.latency[t] + 0.1 * response.latency_ms
        self.cost_estimate[t] = 0.9 * self.cost_estimate[t] + 0.1 * response.cost

    def select_tier(self, request, context):
        best = None; best_score = -1e9
        for tier in ExecutionTier:
            meets = self.latency[tier] <= self.deadline_ms * 0.95
            if not meets and request.complexity > 0:
                continue
            # Lower cost, lower latency preferred; slight complexity bias
            score = -self.cost_estimate[tier] * 1000 * self.cost_weight - self.latency[tier] * 0.001
            if request.complexity == 2 and tier == ExecutionTier.EDGE:
                score -= 0.5  # Discourage edge for complex
            if score > best_score:
                best_score = score; best = tier
        return best or ExecutionTier.HYBRID


class EpsilonGreedyPolicy(RoutingPolicy):
    def __init__(self, epsilon: float = 0.1, deadline_penalty: float = 0.5):
        self.epsilon = epsilon; self.deadline_penalty = deadline_penalty
        self.reset()

    def reset(self):
        self.counts = {t: 0 for t in ExecutionTier}
        self.rewards = {t: 0.0 for t in ExecutionTier}

    def select_tier(self, request, context):
        if np.random.random() < self.epsilon:
            return np.random.choice(list(ExecutionTier))
        avg = {t: (self.rewards[t] / self.counts[t] if self.counts[t] else 1.0)
               for t in ExecutionTier}
        return max(avg, key=avg.get)

    def update(self, request, response):
        t = response.tier
        r = response.quality - (self.deadline_penalty if not response.deadline_met else 0.0)
        self.counts[t] += 1; self.rewards[t] += r


class UCB1Policy(RoutingPolicy):
    def __init__(self, deadline_penalty: float = 0.5):
        self.deadline_penalty = deadline_penalty
        self.reset()

    def reset(self):
        self.counts = {t: 0 for t in ExecutionTier}
        self.rewards = {t: 0.0 for t in ExecutionTier}
        self.total = 0

    def select_tier(self, request, context):
        for t in ExecutionTier:
            if self.counts[t] == 0:
                return t
        scores = {t: self.rewards[t] / self.counts[t] +
                  np.sqrt(2 * np.log(self.total) / self.counts[t])
                  for t in ExecutionTier}
        return max(scores, key=scores.get)

    def update(self, request, response):
        t = response.tier
        r = response.quality - (self.deadline_penalty if not response.deadline_met else 0.0)
        self.counts[t] += 1; self.rewards[t] += r; self.total += 1


class ThompsonSamplingPolicy(RoutingPolicy):
    def __init__(self, deadline_penalty: float = 0.5):
        self.deadline_penalty = deadline_penalty
        self.reset()

    def reset(self):
        self.alpha = {t: 1.0 for t in ExecutionTier}
        self.beta = {t: 1.0 for t in ExecutionTier}

    def select_tier(self, request, context):
        samples = {t: np.random.beta(self.alpha[t], self.beta[t]) for t in ExecutionTier}
        return max(samples, key=samples.get)

    def update(self, request, response):
        t = response.tier
        r = response.quality if response.deadline_met else max(
            0.0, response.quality - self.deadline_penalty)
        self.alpha[t] += r; self.beta[t] += (1 - r)


class DATSPolicy(RoutingPolicy):
    """Deadline-Aware Thompson Sampling (proposed, revised).

    Differences from plain Thompson Sampling:
    - Quality posterior (Beta) updated from observed quality only.
    - Latency is tracked with a fast EMA over a recency window, PLUS an
      upper-tail estimate (ℓ̂ + κ σ̂) used in the deadline-probability term.
      This makes DATS react to heavy-tail spikes rather than relying on mean
      alone.
    - Complexity-conditioned arm prior steers early exploration.
    - Adaptive penalty: the deadline penalty is scaled by how quickly the
      tail estimate has deteriorated (``tail_alarm``) so that sudden regime
      changes trigger a hard bias towards safer tiers.
    """

    def __init__(self, deadline_ms: float = 500.0, deadline_penalty: float = 0.5,
                 ema_gamma: float = 0.55, tail_kappa: float = 1.5,
                 miss_ema: float = 0.65, miss_weight: float = 1.6,
                 use_complexity_prior: bool = True,
                 predictor: Optional[ComplexityPredictor] = None):
        self.deadline_ms = deadline_ms
        self.deadline_penalty = deadline_penalty
        self.ema_gamma = ema_gamma
        self.tail_kappa = tail_kappa
        self.miss_ema = miss_ema
        self.miss_weight = miss_weight
        self.use_complexity_prior = use_complexity_prior
        self.predictor = predictor
        self.reset()

    def reset(self):
        self.alpha = {t: 1.0 for t in ExecutionTier}
        self.beta = {t: 1.0 for t in ExecutionTier}
        self.latency_mean = {
            ExecutionTier.EDGE: 60.0,
            ExecutionTier.CLOUD: 400.0,
            ExecutionTier.HYBRID: 220.0,
        }
        self.latency_std = {
            ExecutionTier.EDGE: 25.0,
            ExecutionTier.CLOUD: 140.0,
            ExecutionTier.HYBRID: 90.0,
        }
        # Recent SLA miss rate per tier (short-window EMA)
        self.miss_rate = {t: 0.0 for t in ExecutionTier}
        self.seen = {t: 0 for t in ExecutionTier}

    def _effective_latency(self, tier: ExecutionTier) -> float:
        return self.latency_mean[tier] + self.tail_kappa * self.latency_std[tier]

    def _deadline_prob(self, tier: ExecutionTier) -> float:
        from scipy import stats
        # Use the tail-aware latency estimate so heavy tails reduce the
        # deadline probability even if the mean is under budget.
        eff = self._effective_latency(tier)
        z = (self.deadline_ms - eff) / max(self.latency_std[tier], 1e-6)
        return float(stats.norm.cdf(z))

    def _predicted_complexity(self, request: Request) -> int:
        if self.predictor is not None and self.predictor.trained:
            return self.predictor.predict(request.get_features())
        return request.complexity  # Fallback to ground truth when testing

    def select_tier(self, request, context):
        c = self._predicted_complexity(request)
        scores = {}
        # Adaptive penalty: higher when miss rate is high on the most-used tier
        max_miss = max(self.miss_rate.values())
        lam = self.deadline_penalty * (1.0 + 2.0 * max_miss)
        for t in ExecutionTier:
            q = np.random.beta(self.alpha[t], self.beta[t])
            p = self._deadline_prob(t)
            s = q * p - lam * (1 - p)
            # Directly subtract recent miss rate so recurring tails are penalised
            s -= self.miss_weight * self.miss_rate[t]
            if self.use_complexity_prior:
                prior_boost = {
                    (0, ExecutionTier.EDGE):   0.10,
                    (0, ExecutionTier.CLOUD):  -0.05,
                    (2, ExecutionTier.CLOUD):  0.10,
                    (2, ExecutionTier.EDGE):   -0.08,
                }.get((c, t), 0.0)
                s += prior_boost
            scores[t] = s
        return max(scores, key=scores.get)

    def update(self, request, response):
        t = response.tier
        r = response.quality if response.deadline_met else max(
            0.0, response.quality - self.deadline_penalty)
        self.alpha[t] += r; self.beta[t] += (1 - r)
        old_mean = self.latency_mean[t]
        self.latency_mean[t] = self.ema_gamma * old_mean + (1 - self.ema_gamma) * response.latency_ms
        delta = response.latency_ms - old_mean
        self.latency_std[t] = float(np.sqrt(
            self.ema_gamma * self.latency_std[t] ** 2 + (1 - self.ema_gamma) * delta ** 2
        ))
        miss = 0.0 if response.deadline_met else 1.0
        self.miss_rate[t] = self.miss_ema * self.miss_rate[t] + (1 - self.miss_ema) * miss
        self.seen[t] += 1


# ----------------------------------------------------------------------------
# Simulator orchestrator
# ----------------------------------------------------------------------------


class EdgeSimulator:
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.inference_model = LLMInferenceModel(config)
        if config.seed is not None:
            np.random.seed(config.seed)

    def run(self, requests: List[Request], policy: RoutingPolicy) -> List[Response]:
        policy.reset()
        # Reset queue state for deterministic replays
        self.inference_model = LLMInferenceModel(self.config)
        responses = []
        context = {"recent_latency_ema": self.config.network.base_latency_ms}
        for request in requests:
            tier = policy.select_tier(request, context)
            response = self.inference_model.execute(request, tier)
            policy.update(request, response)
            responses.append(response)
            context["recent_latency_ema"] = (
                0.9 * context["recent_latency_ema"] + 0.1 * response.latency_ms)
        return responses

    def compute_metrics(self, responses: List[Response]) -> Dict:
        if not responses:
            return {}
        latencies = [r.latency_ms for r in responses]
        qualities = [r.quality for r in responses]
        deadline_met = [r.deadline_met for r in responses]
        costs = [r.cost for r in responses]
        waits = [r.queue_wait_ms for r in responses]
        cold_starts = [r.cold_start for r in responses]
        return {
            "n_requests": len(responses),
            "quality_mean": float(np.mean(qualities)),
            "quality_std": float(np.std(qualities)),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "sla_violation_rate": float(1 - np.mean(deadline_met)),
            "total_cost": float(sum(costs)),
            "queue_wait_p95": float(np.percentile(waits, 95)),
            "cold_start_rate": float(np.mean(cold_starts)),
            "tier_distribution": {
                tier.name: sum(1 for r in responses if r.tier == tier) / len(responses)
                for tier in ExecutionTier
            },
        }

    def compute_regret(self, responses: List[Response], oracle_quality: float = 0.95) -> np.ndarray:
        regrets = []
        cumulative = 0.0
        for r in responses:
            instant_regret = oracle_quality - r.quality
            if not r.deadline_met:
                instant_regret += 0.5
            cumulative += max(0, instant_regret)
            regrets.append(cumulative)
        return np.array(regrets)


def run_comparison_experiment(
    n_requests: int = 10000,
    network_profile: NetworkProfile = NetworkProfile.MOBILE,
    seed: int = 42,
) -> Dict:
    config = SimulatorConfig(
        network=NetworkConfig.from_profile(network_profile),
        edge=EdgeNodeConfig.rtx3060(),
        cloud=CloudConfig.gemini_flash(),
        deadline_ms=500.0, seed=seed,
    )
    simulator = EdgeSimulator(config)
    generator = RequestGenerator(seed=seed)
    requests = generator.generate_poisson(n_requests)

    predictor = ComplexityPredictor()
    X, y = ComplexityPredictor.build_dataset(n_samples=4000, seed=seed)
    predictor.fit(X, y, epochs=40, verbose=False)

    policies = {
        "cloud_only": CloudOnlyPolicy(),
        "edge_only": EdgeOnlyPolicy(),
        "random": RandomPolicy(),
        "complexity_static": ComplexityStaticPolicy(),
        "frugalgpt_cascade": FrugalGPTCascadePolicy(),
        "qos_static": QoSStaticPolicy(deadline_ms=config.deadline_ms),
        "cost_aware": CostAwarePolicy(deadline_ms=config.deadline_ms),
        "epsilon_greedy": EpsilonGreedyPolicy(epsilon=0.1),
        "ucb1": UCB1Policy(),
        "thompson_sampling": ThompsonSamplingPolicy(),
        "dats": DATSPolicy(deadline_ms=config.deadline_ms, predictor=predictor),
    }
    results = {}
    for name, policy in policies.items():
        responses = simulator.run(requests, policy)
        metrics = simulator.compute_metrics(responses)
        regret = simulator.compute_regret(responses)
        results[name] = {
            "metrics": metrics,
            "final_regret": float(regret[-1] if len(regret) else 0),
        }
        print(f"{name}: q={metrics['quality_mean']:.3f}, p99={metrics['latency_p99']:.0f}ms, "
              f"sla={metrics['sla_violation_rate']:.1%}")
    return results


if __name__ == "__main__":
    print("=== Edge Computing Simulator (revised) ===\n")
    results = run_comparison_experiment(n_requests=10000, seed=42)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    serializable = {}
    for name, data in results.items():
        serializable[name] = {
            "metrics": {k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
                        for k, v in data["metrics"].items()},
            "final_regret": float(data["final_regret"]),
        }
    with open(output_dir / "policy_comparison.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {output_dir}/policy_comparison.json")
