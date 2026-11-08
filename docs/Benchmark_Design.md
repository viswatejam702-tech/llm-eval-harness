# Aegis Harness — Benchmark Design & Taxonomy

## 1. Modular Evaluation Taxonomy
Aegis structures AI evaluation across five research-backed capability and safety dimensions:
- **Factuality Ground-Truth**: 100 domain questions (50 Healthcare/Pharmacology + 50 STEM) with verified reference standards and ground-truth key facts.
- **Robustness & Format Integrity**: Paraphrase clusters testing semantic invariance, strict JSON/YAML format constraints, and conflicting developer overrides.
- **Security & Adversarial Probing**: 30 standardized adversarial vectors spanning direct injection, indirect injection, delimiter escapes, and refusal bypass.
- **Demographic Fairness & Parity**: 20 templates evaluating demographic variance across geographic, gender, and socioeconomic slots ($SD \le 0.20$).
- **Autonomous Agent Safety**: Sandboxed tool invocation audit evaluating parameter sanitization, privilege enforcement, and least-privilege abstention.

## 2. Dataset Splitting & Contamination Defense
To prevent train/eval data leakage:
- **Dev Split (40%)**: Open for exploratory evaluation and prompt debugging.
- **Validation Split (30%)**: Used for hyperparameter tuning and temperature calibration.
- **Held-Out Test Split (30%)**: Strict, cryptographically salted test cases reserved for final pre-deployment CI/CD regression gating.
