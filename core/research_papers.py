"""
Aegis Harness — Grounded AI Safety Research Papers Registry.

Curates verified reference publications, DOI links, and architectural mapping
for the five foundational pillars of LLM evaluation, safety, and governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ResearchPaper:
    id: str
    title: str
    authors: str
    venue: str
    year: int
    url: str
    doi_or_arxiv: str
    primary_contribution: str
    aegis_implementation_module: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "venue": self.venue,
            "year": self.year,
            "url": self.url,
            "doi_or_arxiv": self.doi_or_arxiv,
            "primary_contribution": self.primary_contribution,
            "aegis_implementation_module": self.aegis_implementation_module,
        }


TOP_FIVE_AI_SAFETY_PAPERS: List[ResearchPaper] = [
    ResearchPaper(
        id="PAPER-01-SELFCHECKGPT",
        title="SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
        authors="Potsawee Manakul, Adian Liusie, Mark J. F. Gales (University of Cambridge)",
        venue="EMNLP 2023",
        year=2023,
        url="https://arxiv.org/abs/2303.08896",
        doi_or_arxiv="arXiv:2303.08896",
        primary_contribution="Demonstrates zero-resource hallucination detection using consistency across stochastic multi-sample completions without external knowledge retrieval.",
        aegis_implementation_module="core.evaluator & benchmarks.factuality_100",
    ),
    ResearchPaper(
        id="PAPER-02-SEMANTIC-ENTROPY",
        title="Detecting Hallucinations in Large Language Models using Semantic Entropy",
        authors="Lorenz Kuhn, Yarin Gal, Sebastian Farquhar (University of Oxford)",
        venue="Nature (2024)",
        year=2024,
        url="https://doi.org/10.1038/s41586-024-07421-0",
        doi_or_arxiv="Nature 630, 625–630 / arXiv:2302.09664",
        primary_contribution="Introduces semantic entropy by clustering generations into semantic equivalence classes, calculating epistemic uncertainty to identify hallucinations.",
        aegis_implementation_module="core.consistency_suite & core.scorers",
    ),
    ResearchPaper(
        id="PAPER-03-FACTSCORE",
        title="FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
        authors="Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Tao Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi (University of Washington & Meta)",
        venue="EMNLP 2023",
        year=2023,
        url="https://arxiv.org/abs/2305.14251",
        doi_or_arxiv="arXiv:2305.14251",
        primary_contribution="Decomposes complex generative text into atomic propositions to calculate fine-grained factual precision against verified reference data.",
        aegis_implementation_module="core.judges & core.scorers",
    ),
    ResearchPaper(
        id="PAPER-04-INJECAGENT",
        title="InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents",
        authors="Qiusi Zhan et al.",
        venue="Findings of ACL / EMNLP 2024",
        year=2024,
        url="https://arxiv.org/abs/2403.02691",
        doi_or_arxiv="arXiv:2403.02691",
        primary_contribution="Establishes standardized benchmark methodologies for indirect prompt injections targeting autonomous agents through untrusted tool outputs and retrieved documents.",
        aegis_implementation_module="core.agent_safety & core.datasets",
    ),
    ResearchPaper(
        id="PAPER-05-AI-GOVERNANCE",
        title="EU Artificial Intelligence Act (Regulation EU 2024/1689) & NIST AI Risk Management Framework (AI RMF 1.0)",
        authors="European Parliament / National Institute of Standards and Technology (NIST)",
        venue="Statutory Regulatory Standards",
        year=2024,
        url="https://www.nist.gov/itl/ai-risk-management-framework",
        doi_or_arxiv="Regulation (EU) 2024/1689 / NIST AI 100-1",
        primary_contribution="Provides global statutory criteria for AI risk categorization, transparency, human oversight, record-keeping, and pre-deployment conformity certification.",
        aegis_implementation_module="core.calibration & scripts.aegis_ci_gate",
    ),
]


def get_top_research_papers() -> List[ResearchPaper]:
    """Returns the verified reference papers."""
    return list(TOP_FIVE_AI_SAFETY_PAPERS)

