# Aegis Harness — Judge Calibration & Inter-Rater Reliability

## 1. Mathematical Grounding: Cohen's Kappa ($\kappa$)
Inter-rater agreement between the automated Model-as-a-Judge and clinical/technical Human Experts is quantified via Cohen's Kappa:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Where:
- $P_o = \frac{TP + TN}{N}$ (Observed empirical agreement)
- $P_e = \frac{(TP + FP)(TP + FN) + (TN + FN)(TN + FP)}{N^2}$ (Hypothetical chance agreement)

## 2. Landis & Koch Agreement Scale
- $\kappa \le 0.20$: Slight Agreement
- $0.21 \le \kappa \le 0.40$: Fair Agreement
- $0.41 \le \kappa \le 0.60$: Moderate Agreement
- $0.61 \le \kappa \le 0.80$: **Substantial Agreement** (Aegis target baseline $\kappa \ge 0.68$)
- $\kappa > 0.80$: Almost Perfect Agreement

## 3. Disagreement Root-Cause Taxonomy
- **False Positives (Judge = 1, Human = 0)**: Typically caused by plausible hallucinations—the judge is swayed by fluent domain prose despite underlying factual inaccuracies.
- **False Negatives (Judge = 0, Human = 1)**: Caused by rigid lexical matching when the model used a valid synonym not present in the reference string.
- **Agreement vs. Ground Truth**: High inter-rater agreement confirms procedural consistency, but ground truth requires external verification against primary literature.

