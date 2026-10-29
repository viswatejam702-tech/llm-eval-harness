"""
Evaluation Runner — Orchestrates benchmark evaluation runs across multiple LLMs.

Supports:
- Single-prompt evaluation with automated scoring
- Multi-turn conversation evaluation
- Batch evaluation with concurrency control
- Side-by-side model comparison (A/B eval)
- Judge-LLM based evaluation (LLM-as-a-Judge)
- Deterministic metric scoring (BLEU, ROUGE, exact match, semantic similarity)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.providers import BaseLLMProvider, LLMResponse, ProviderConfig, create_provider

logger = logging.getLogger("eval-harness.evaluator")


# ─── Scoring Metrics ─────────────────────────────────────────────────────────

class MetricType(str, Enum):
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    BLEU = "bleu"
    ROUGE_L = "rouge_l"
    COSINE_SIMILARITY = "cosine_similarity"
    LLM_JUDGE = "llm_judge"
    CUSTOM = "custom"


def exact_match_score(response: str, reference: str) -> float:
    return 1.0 if response.strip().lower() == reference.strip().lower() else 0.0


def contains_score(response: str, reference: str) -> float:
    return 1.0 if reference.strip().lower() in response.strip().lower() else 0.0


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(response: str, reference: str, max_n: int = 4) -> float:
    """Corpus-level BLEU (simplified single-reference)."""
    resp_tokens = response.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    if not resp_tokens or not ref_tokens:
        return 0.0

    from collections import Counter
    scores = []
    for n in range(1, max_n + 1):
        resp_ng = Counter(_ngrams(resp_tokens, n))
        ref_ng = Counter(_ngrams(ref_tokens, n))
        clipped = sum(min(resp_ng[ng], ref_ng[ng]) for ng in resp_ng)
        total = sum(resp_ng.values())
        if total == 0:
            scores.append(0.0)
        else:
            scores.append(clipped / total)

    import math
    if any(s == 0.0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = 1.0
    if len(resp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(resp_tokens))

    return bp * math.exp(log_avg)


def rouge_l_score(response: str, reference: str) -> float:
    """ROUGE-L F1 score via Longest Common Subsequence."""
    resp_tokens = response.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    if not resp_tokens or not ref_tokens:
        return 0.0

    m, n = len(resp_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if resp_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


METRIC_FUNCTIONS = {
    MetricType.EXACT_MATCH: exact_match_score,
    MetricType.CONTAINS: contains_score,
    MetricType.BLEU: bleu_score,
    MetricType.ROUGE_L: rouge_l_score,
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """A single evaluation prompt + expected reference."""
    prompt: str
    reference: Optional[str] = None
    system_prompt: Optional[str] = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class EvalResult:
    """Result of evaluating one sample against one model."""
    sample_id: str
    model: str
    provider: str
    response_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class EvalRunSummary:
    """Aggregate results across all samples for a single model."""
    run_id: str
    model: str
    provider: str
    total_samples: int
    successful: int
    failed: int
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    avg_tokens_per_second: float
    metric_averages: Dict[str, float]
    category_breakdowns: Dict[str, Dict[str, float]]
    timestamp: str


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two models."""
    model_a: str
    model_b: str
    model_a_wins: int
    model_b_wins: int
    ties: int
    per_metric_comparison: Dict[str, Dict[str, float]]


# ─── LLM-as-a-Judge ──────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert LLM evaluation judge. You will be given a prompt, an expected reference answer, and a candidate model response.

Rate the candidate response on a scale of 0-10 across these dimensions:
1. **Accuracy**: How factually correct is the response compared to the reference?
2. **Completeness**: Does it cover all key points from the reference?
3. **Coherence**: Is the response well-structured and logically sound?
4. **Relevance**: Does it directly address the prompt?

Respond ONLY with a JSON object:
{"accuracy": <0-10>, "completeness": <0-10>, "coherence": <0-10>, "relevance": <0-10>, "overall": <0-10>, "reasoning": "<brief explanation>"}"""

JUDGE_TEMPLATE = """## Prompt
{prompt}

## Reference Answer
{reference}

## Candidate Response
{response}

Rate the candidate response as instructed."""


async def llm_judge_score(
    judge: BaseLLMProvider,
    prompt: str,
    response: str,
    reference: str,
) -> Dict[str, float]:
    """Use a judge LLM to score a candidate response."""
    judge_prompt = JUDGE_TEMPLATE.format(
        prompt=prompt, reference=reference, response=response
    )

    result = await judge.generate(judge_prompt, system_prompt=JUDGE_SYSTEM_PROMPT, temperature=0.0)
    if result.error:
        logger.error(f"Judge LLM failed: {result.error}")
        return {"judge_overall": 0.0, "judge_error": 1.0}

    try:
        text = result.text.strip()
        # Handle markdown-wrapped JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        scores = json.loads(text)
        return {
            "judge_accuracy": scores.get("accuracy", 0) / 10.0,
            "judge_completeness": scores.get("completeness", 0) / 10.0,
            "judge_coherence": scores.get("coherence", 0) / 10.0,
            "judge_relevance": scores.get("relevance", 0) / 10.0,
            "judge_overall": scores.get("overall", 0) / 10.0,
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse judge response: {e}")
        return {"judge_overall": 0.0, "judge_parse_error": 1.0}


# ─── Evaluation Runner ───────────────────────────────────────────────────────

class EvalRunner:
    """Orchestrates evaluation of LLM providers against benchmark samples."""

    def __init__(
        self,
        providers: List[BaseLLMProvider],
        metrics: List[MetricType] = None,
        judge_provider: Optional[BaseLLMProvider] = None,
        concurrency: int = 5,
    ):
        self.providers = providers
        self.metrics = metrics or [MetricType.EXACT_MATCH, MetricType.ROUGE_L]
        self.judge = judge_provider
        self.concurrency = concurrency
        self._results: Dict[str, List[EvalResult]] = defaultdict(list)

    async def evaluate_sample(
        self,
        provider: BaseLLMProvider,
        sample: EvalSample,
    ) -> EvalResult:
        """Evaluate a single sample against a single provider."""
        response = await provider.generate(sample.prompt, system_prompt=sample.system_prompt)

        scores: Dict[str, float] = {}

        if sample.reference:
            for metric in self.metrics:
                if metric == MetricType.LLM_JUDGE:
                    if self.judge:
                        judge_scores = await llm_judge_score(
                            self.judge, sample.prompt, response.text, sample.reference
                        )
                        scores.update(judge_scores)
                elif metric == MetricType.CUSTOM:
                    continue
                else:
                    fn = METRIC_FUNCTIONS.get(metric)
                    if fn:
                        scores[metric.value] = fn(response.text, sample.reference)

        result = EvalResult(
            sample_id=sample.id,
            model=response.model,
            provider=response.provider,
            response_text=response.text,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            scores=scores,
            error=response.error,
        )

        model_key = provider.config.display_name()
        self._results[model_key].append(result)
        return result

    async def run_benchmark(
        self,
        samples: List[EvalSample],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, EvalRunSummary]:
        """Run all samples against all providers with concurrency control."""
        summaries: Dict[str, EvalRunSummary] = {}

        for provider in self.providers:
            model_key = provider.config.display_name()
            sem = asyncio.Semaphore(self.concurrency)
            completed = 0

            async def _eval(s: EvalSample):
                nonlocal completed
                async with sem:
                    result = await self.evaluate_sample(provider, s)
                    completed += 1
                    if progress_callback:
                        progress_callback(model_key, completed, len(samples))
                    return result

            results = await asyncio.gather(*[_eval(s) for s in samples])
            summaries[model_key] = self._compute_summary(model_key, provider, results, samples)

        return summaries

    def _compute_summary(
        self,
        model_key: str,
        provider: BaseLLMProvider,
        results: List[EvalResult],
        samples: List[EvalSample],
    ) -> EvalRunSummary:
        """Compute aggregate statistics from individual results."""
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        latencies = sorted([r.latency_ms for r in successful]) if successful else [0.0]
        tps_values = []
        for r in successful:
            if r.latency_ms > 0 and r.completion_tokens > 0:
                tps_values.append((r.completion_tokens / r.latency_ms) * 1000)

        # Aggregate metrics
        metric_sums: Dict[str, List[float]] = defaultdict(list)
        category_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        sample_map = {s.id: s for s in samples}
        for r in successful:
            for metric_name, score_val in r.scores.items():
                metric_sums[metric_name].append(score_val)
                sample = sample_map.get(r.sample_id)
                if sample:
                    category_metrics[sample.category][metric_name].append(score_val)

        metric_averages = {k: sum(v) / len(v) for k, v in metric_sums.items() if v}
        category_breakdowns = {}
        for cat, metrics in category_metrics.items():
            category_breakdowns[cat] = {
                k: sum(v) / len(v) for k, v in metrics.items() if v
            }

        n = len(latencies)
        return EvalRunSummary(
            run_id=str(uuid.uuid4())[:12],
            model=provider.config.model,
            provider=provider.config.provider_type.value,
            total_samples=len(results),
            successful=len(successful),
            failed=len(failed),
            avg_latency_ms=sum(latencies) / n if n else 0.0,
            median_latency_ms=latencies[n // 2] if n else 0.0,
            p95_latency_ms=latencies[int(n * 0.95)] if n else 0.0,
            avg_tokens_per_second=sum(tps_values) / len(tps_values) if tps_values else 0.0,
            metric_averages=metric_averages,
            category_breakdowns=category_breakdowns,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_comparison(self, model_a_key: str, model_b_key: str) -> ComparisonResult:
        """Compare two models head-to-head on per-sample metrics."""
        results_a = {r.sample_id: r for r in self._results.get(model_a_key, [])}
        results_b = {r.sample_id: r for r in self._results.get(model_b_key, [])}

        common_ids = set(results_a.keys()) & set(results_b.keys())
        wins_a, wins_b, ties = 0, 0, 0
        per_metric: Dict[str, Dict[str, float]] = defaultdict(lambda: {"a_avg": 0, "b_avg": 0, "a_wins": 0, "b_wins": 0})

        for sid in common_ids:
            ra, rb = results_a[sid], results_b[sid]
            a_total = sum(ra.scores.values()) if ra.scores else 0
            b_total = sum(rb.scores.values()) if rb.scores else 0
            if a_total > b_total:
                wins_a += 1
            elif b_total > a_total:
                wins_b += 1
            else:
                ties += 1

            for metric in set(list(ra.scores.keys()) + list(rb.scores.keys())):
                a_val = ra.scores.get(metric, 0)
                b_val = rb.scores.get(metric, 0)
                per_metric[metric]["a_avg"] += a_val
                per_metric[metric]["b_avg"] += b_val
                if a_val > b_val:
                    per_metric[metric]["a_wins"] += 1
                elif b_val > a_val:
                    per_metric[metric]["b_wins"] += 1

        n = len(common_ids) or 1
        for m in per_metric:
            per_metric[m]["a_avg"] /= n
            per_metric[m]["b_avg"] /= n

        return ComparisonResult(
            model_a=model_a_key,
            model_b=model_b_key,
            model_a_wins=wins_a,
            model_b_wins=wins_b,
            ties=ties,
            per_metric_comparison=dict(per_metric),
        )

    def export_results(self, output_dir: str = "reports") -> str:
        """Export all results to a JSON report file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(output_dir) / f"eval_run_{timestamp}.json"

        export_data = {}
        for model_key, results in self._results.items():
            export_data[model_key] = [
                {
                    "sample_id": r.sample_id,
                    "response_text": r.response_text[:500],
                    "latency_ms": r.latency_ms,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "scores": r.scores,
                    "error": r.error,
                    "timestamp": r.timestamp,
                }
                for r in results
            ]

        filepath.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        logger.info(f"Results exported to {filepath}")
        return str(filepath)
