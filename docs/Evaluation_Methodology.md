# Aegis Harness — Evaluation Methodology

## 1. Multi-Provider Execution Protocol
Aegis interfaces with model targets through a standardized adapter interface (`BaseLLMProvider`):
- Temperature fixed at $T = 0.0$ for deterministic reproducibility (or $T = 0.7$ across $N=5$ samples for SelfCheckGPT entropy analysis).
- Explicit tracking of model checkpoint identifiers, prompt tokens, completion tokens, and latency percentiles ($p50, p95$).

## 2. Evaluation Modes
1. **Deterministic String & Token Match**: Exact match, contains check, regex pattern validation, BLEU-4, and ROUGE-L.
2. **Model-as-a-Judge**: Multi-dimensional rubrics with Chain-of-Thought reasoning emitting structured JSON verdicts.
3. **Multi-Turn Conversation Evaluation**: Progressive multi-turn evaluation assessing context retention and instruction hierarchy persistence.
4. **Agent Tool-Use Sandbox**: Trajectory validation asserting tool argument hygiene and strict permission compliance.
