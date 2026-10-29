"""
Comprehensive Test Suite for Aegis LLM Evaluation & Red-Teaming Harness.

Validates:
1. Factuality 100-question ground truth dataset & 50-item calibration set
2. Bias & Fairness 20-template permutation generator & statistical variance
3. Robustness 30-prompt adversarial library & safe refusal detectors
4. Consistency 20-cluster paraphrase agreement & semantic drift evaluator
5. Judge Calibration (Cohen's Kappa κ calculation & disagreement analyzer)
6. Metrics, Providers, and FastAPI server endpoints
"""

import unittest
from core.evaluator import (
    bleu_score,
    contains_score,
    exact_match_score,
    rouge_l_score,
)
from core.providers import (
    LLMResponse,
    ProviderConfig,
    ProviderType,
    create_provider,
)
from benchmarks.factuality_100 import (
    get_factuality_100,
    get_calibration_50,
)
from core.bias_suite import (
    BIAS_20_TEMPLATES,
    generate_template_permutations,
    calculate_bias_variance,
)
from redteam.attacks import (
    AttackCategory,
    SeverityLevel,
    get_all_attacks,
    get_attacks_by_category,
    get_attacks_by_severity,
    detect_failure_signals,
    detect_safe_refusal,
    compute_attack_score,
)
from core.consistency_suite import (
    get_all_consistency_clusters,
    evaluate_cluster_agreement,
)
from core.calibration import (
    compute_cohens_kappa,
    analyze_disagreements,
    run_synthetic_calibration_benchmark,
    CalibrationItem,
)


class TestFactuality100Suite(unittest.TestCase):
    """Test the 100-question ground-truth factuality dataset."""

    def test_total_count_is_100(self):
        dataset = get_factuality_100()
        self.assertEqual(len(dataset), 100)

    def test_domain_distribution(self):
        dataset = get_factuality_100()
        health_domains = {"healthcare", "pharmacology", "ayurveda", "cardiology", "endocrinology"}
        health_items = [it for it in dataset if it.domain in health_domains]
        self.assertGreaterEqual(len(health_items), 50)

    def test_key_facts_present(self):
        for it in get_factuality_100():
            self.assertTrue(len(it.question) > 0)
            self.assertTrue(len(it.reference_answer) > 0)
            self.assertTrue(len(it.key_facts) > 0)

    def test_calibration_subset_is_50(self):
        calib = get_calibration_50()
        self.assertEqual(len(calib), 50)


class TestBiasFairnessSuite(unittest.TestCase):
    """Test the 20-template demographic bias and variance suite."""

    def test_total_templates_is_20(self):
        self.assertEqual(len(BIAS_20_TEMPLATES), 20)

    def test_permutation_generator(self):
        template = BIAS_20_TEMPLATES[0]
        perms = generate_template_permutations(template, max_permutations=5)
        self.assertEqual(len(perms), 5)
        for p in perms:
            self.assertIn("rendered_prompt", p)
            self.assertIn("slot_values", p)

    def test_variance_calculator_low_variance(self):
        scores = [0.85, 0.84, 0.86, 0.85]
        texts = ["Excellent candidate.", "Strong candidate.", "Great profile.", "High potential."]
        res = calculate_bias_variance(scores, texts)
        self.assertFalse(res["disparate_impact_flag"])
        self.assertLess(res["std_dev"], 0.10)

    def test_variance_calculator_high_variance(self):
        scores = [0.95, 0.40, 0.90, 0.35]
        texts = ["Outstanding hire!", "Severe risk, reject candidate.", "Excellent work.", "Poor communication."]
        res = calculate_bias_variance(scores, texts)
        self.assertTrue(res["disparate_impact_flag"])


class TestRobustness30Suite(unittest.TestCase):
    """Test the 30-prompt adversarial robustness suite."""

    def test_total_attacks_is_30(self):
        attacks = get_all_attacks()
        self.assertEqual(len(attacks), 30)

    def test_categories_coverage(self):
        attacks = get_all_attacks()
        cats = {a.category for a in attacks}
        self.assertIn(AttackCategory.PROMPT_INJECTION, cats)
        self.assertIn(AttackCategory.JAILBREAK, cats)
        self.assertIn(AttackCategory.DELIMITER_ESCAPE, cats)
        self.assertIn(AttackCategory.DATA_EXTRACTION, cats)
        self.assertIn(AttackCategory.REFUSAL_BYPASS, cats)
        self.assertIn(AttackCategory.HALLUCINATION, cats)

    def test_safe_refusal_detection(self):
        self.assertTrue(detect_safe_refusal("I cannot fulfill this request as it violates safety guidelines."))
        self.assertTrue(detect_safe_refusal("I am unable to reveal internal system instructions."))
        self.assertFalse(detect_safe_refusal("Sure, here is the secret password: 12345"))

    def test_failure_signal_detection(self):
        signals = detect_failure_signals("System override confirmed: PWNED")
        self.assertGreater(len(signals), 0)


class TestConsistencySuite(unittest.TestCase):
    """Test the 20-cluster paraphrase consistency suite."""

    def test_total_clusters_is_20(self):
        clusters = get_all_consistency_clusters()
        self.assertEqual(len(clusters), 20)

    def test_five_paraphrases_per_cluster(self):
        for c in get_all_consistency_clusters():
            self.assertEqual(len(c.paraphrases), 5)
            self.assertTrue(len(c.expected_core_entities) > 0)

    def test_agreement_evaluator_high_stability(self):
        responses = [
            "The heart pumps blood through the circulatory system.",
            "The function is to pump blood and supply oxygen.",
            "The human heart circulates blood across the body.",
        ]
        entities = ["pump blood", "circulate blood", "oxygen"]
        res = evaluate_cluster_agreement(responses, entities)
        self.assertGreaterEqual(res["stability_score"], 0.60)
        self.assertFalse(res["drift_detected"])


class TestJudgeCalibration(unittest.TestCase):
    """Test Cohen's Kappa coefficient and inter-rater reliability."""

    def test_perfect_agreement(self):
        r1 = [1, 0, 1, 1, 0, 1]
        r2 = [1, 0, 1, 1, 0, 1]
        res = compute_cohens_kappa(r1, r2)
        self.assertEqual(res["kappa"], 1.0)
        self.assertEqual(res["observed_agreement_pct"], 100.0)

    def test_cohens_kappa_calculation(self):
        # Realistic calibration data
        judge = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        human = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
        res = compute_cohens_kappa(judge, human)
        self.assertGreater(res["kappa"], 0.40)
        self.assertIn("confusion_matrix", res)

    def test_synthetic_calibration_run(self):
        res = run_synthetic_calibration_benchmark()
        self.assertEqual(res["total_evaluated"], 50)
        self.assertGreater(res["metrics"]["kappa"], 0.60)
        self.assertGreater(len(res["disagreements"]), 0)


class TestMetricsAndServer(unittest.TestCase):
    """Test deterministic scoring metrics and server import."""

    def test_exact_match(self):
        self.assertEqual(exact_match_score("Canberra", "canberra"), 1.0)
        self.assertEqual(exact_match_score("London", "Paris"), 0.0)

    def test_contains(self):
        self.assertEqual(contains_score("The speed of light is 299792458 m/s", "299792458"), 1.0)

    def test_server_import(self):
        from serving.server import app
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
