# Aegis Harness — Deterministic Reproducibility Protocol

## 1. Random Seed Pinning
All stochastic operations, demographic template sampling, and bootstrap confidence interval iterations are seeded with fixed defaults (`seed=42`).

## 2. Model Checkpoint Pinning
Aegis records exact model version strings (e.g., `claude-3-5-sonnet-20241022`, `gpt-4o-2024-08-06`, `gemini-2.5-flash`) and provider system fingerprints to detect silent upstream API drift.

## 3. Configuration Freezing
Benchmark runs snapshot evaluation parameters (temperature, top_p, max_tokens, system prompt) into the output `ci_summary.json` alongside evaluation results.

