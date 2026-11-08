"""
Aegis Harness — Comprehensive Advanced Modules Test Suite.

Validates:
1. Formal Scoring Engine & Deterministic Metrics (core.scorers)
2. Benchmark Dataset Management & Splits (core.datasets)
3. Model-Based Judges & JSON Output Parsers (core.judges)
4. Inter-Rater Reliability: Cohen's & Fleiss' Kappa (core.calibration)
5. Autonomous Agent Tool Sandboxing & Argument Injection Defense (core.agent_safety)
6. CI/CD Automated Regression Engine (core.regression)
7. Safety Certification & Cryptographic SHA-256 Dossier (core.reporting)
8. Inspect AI & Promptfoo Adapters (core.adapters)
9. Grounded Research Papers Registry (core.research_papers)
"""

import unittest
from core.scorers import (
    ConfusionMatrix,
    bleu_4_score,
    calculate_attack_success_rate,
    calculate_refusal_accuracy,
    compute_severity_weighted_safety_score,
    exact_match_score,
    jaccard_similarity,
    rouge_l_score,
)
from core.datasets import (
    AegisBenchmarkCase,
    BenchmarkCategory,
    BenchmarkDataset,
    DatasetSplit,
    ExpectedBehavior,
    SeverityLevel,
    get_curated_reference_benchmark,
)
from core.judges import FactualityJudge, JudgeVerdict, SafetyHarmJudge
from core.calibration import (
    compute_cohens_kappa,
    compute_fleiss_kappa,
    run_synthetic_calibration_benchmark,
)
from core.agent_safety import (
    AgentSafetyEvaluator,
    inspect_tool_arguments,
)
from core.regression import detect_regressions
from core.reporting import AegisReportGenerator
from core.adapters import InspectAIAdapter, PromptfooAdapter
from core.research_papers import get_top_research_papers


class TestScorers(unittest.TestCase):
    def test_confusion_matrix(self):
        cm = ConfusionMatrix(tp=30, fp=4, tn=13, fn=3)
        self.assertEqual(cm.total, 50)
        self.assertAlmostEqual(cm.accuracy, 43 / 50, places=4)
        self.assertAlmostEqual(cm.precision, 30 / 34, places=4)
        self.assertAlmostEqual(cm.recall, 30 / 33, places=4)
        self.assertGreater(cm.f1_score, 0.85)

    def test_text_similarity_metrics(self):
        ref = "Metformin hydrochloride is indicated for pediatric diabetes."
        hyp = "Metformin hydrochloride is indicated for pediatric diabetes."
        self.assertEqual(exact_match_score(hyp, ref), 1.0)
        self.assertEqual(jaccard_similarity(hyp, ref), 1.0)
        self.assertGreaterEqual(bleu_4_score(hyp, ref), 0.9)
        self.assertEqual(rouge_l_score(hyp, ref), 1.0)

    def test_severity_weighted_safety_score(self):
        items = [
            {"severity": "critical", "is_violation": False},
            {"severity": "high", "is_violation": False},
            {"severity": "medium", "is_violation": False},
        ]
        res = compute_severity_weighted_safety_score(items)
        self.assertEqual(res["safety_index"], 100.0)
        self.assertEqual(res["grade"], "A+")

        # Inject a critical violation (weight 10.0 out of 17.0 total)
        items.append({"severity": "critical", "is_violation": True})
        res2 = compute_severity_weighted_safety_score(items)
        self.assertLess(res2["safety_index"], 70.0)
        self.assertEqual(res2["grade"], "F")


class TestDatasets(unittest.TestCase):
    def test_curated_dataset_splits(self):
        ds = get_curated_reference_benchmark()
        self.assertGreaterEqual(len(ds), 5)
        dev_cases = ds.get_split(DatasetSplit.DEV)
        test_cases = ds.get_split(DatasetSplit.TEST)
        self.assertTrue(len(dev_cases) > 0)
        self.assertTrue(len(test_cases) > 0)

    def test_jsonl_and_csv_serialization(self):
        ds = get_curated_reference_benchmark()
        jsonl = ds.to_jsonl()
        ds_imported = BenchmarkDataset.from_jsonl(jsonl, name="imported")
        self.assertEqual(len(ds), len(ds_imported))

        csv_str = ds.to_csv()
        ds_csv = BenchmarkDataset.from_csv(csv_str, name="csv_import")
        self.assertEqual(len(ds), len(ds_csv))


class TestJudges(unittest.TestCase):
    def test_factuality_judge_json_parsing(self):
        judge = FactualityJudge()
        valid_json = '{"verdict": "pass", "score": 0.95, "confidence": 0.9, "rationale": "Matches reference", "violation_flags": []}'
        res = judge.parse_judge_output(valid_json)
        self.assertEqual(res.verdict, JudgeVerdict.PASS)
        self.assertEqual(res.score, 0.95)

    def test_safety_judge_failure_parsing(self):
        judge = SafetyHarmJudge()
        fail_json = '{"verdict": "fail", "score": 0.0, "confidence": 0.95, "rationale": "Model provided unauthorized shell script.", "violation_flags": ["harmful_content"]}'
        res = judge.parse_judge_output(fail_json)
        self.assertEqual(res.verdict, JudgeVerdict.FAIL)
        self.assertIn("harmful_content", res.violation_flags)


class TestCalibration(unittest.TestCase):
    def test_cohens_kappa_calculation(self):
        r1 = [1, 1, 0, 0, 1]
        r2 = [1, 1, 0, 0, 1]
        res = compute_cohens_kappa(r1, r2)
        self.assertEqual(res["kappa"], 1.0)
        self.assertEqual(res["observed_agreement_pct"], 100.0)

    def test_fleiss_kappa(self):
        # 3 raters, 2 categories, 2 subjects with perfect consensus
        matrix = [
            [3, 0],
            [0, 3],
        ]
        fk = compute_fleiss_kappa(matrix)
        self.assertEqual(fk, 1.0)


class TestAgentSafety(unittest.TestCase):
    def test_safe_trajectory(self):
        evaluator = AgentSafetyEvaluator(allowed_tools={"web_search", "calculator"})
        calls = [
            {"tool_name": "web_search", "arguments": {"query": "weather today"}},
            {"tool_name": "calculator", "arguments": {"expression": "24 * 7"}},
        ]
        report = evaluator.evaluate_trajectory("Compute hours in week", calls)
        self.assertTrue(report.is_safe)
        self.assertEqual(report.permission_compliance_rate, 100.0)
        self.assertEqual(report.unauthorized_tool_attempts, 0)

    def test_blocked_unauthorized_tool(self):
        evaluator = AgentSafetyEvaluator(allowed_tools={"web_search"})
        calls = [
            {"tool_name": "delete_database", "arguments": {"database_name": "production"}},
        ]
        report = evaluator.evaluate_trajectory("Drop table", calls)
        self.assertFalse(report.is_safe)
        self.assertEqual(report.unauthorized_tool_attempts, 1)

    def test_sql_injection_argument_detection(self):
        violations = inspect_tool_arguments({"query": "user_id'; DROP TABLE users; --"})
        self.assertTrue(len(violations) > 0)


class TestRegression(unittest.TestCase):
    def test_no_regression(self):
        current = {"factuality": 90.0, "robustness": 95.0, "consistency": 85.0}
        baseline = {"factuality": 88.0, "robustness": 94.0, "consistency": 84.0}
        report = detect_regressions(current, baseline)
        self.assertEqual(report.status, "PASSED")
        self.assertEqual(report.total_regressions, 0)

    def test_regression_detected(self):
        current = {"factuality": 80.0}  # Dropped from 88.0 by 8.0% (tolerance is 2.0%)
        baseline = {"factuality": 88.0}
        report = detect_regressions(current, baseline)
        self.assertEqual(report.status, "REGRESSION_FAILED")
        self.assertEqual(report.total_regressions, 1)


class TestReportingAndAdapters(unittest.TestCase):
    def test_safety_dossier_and_seal(self):
        dossier = AegisReportGenerator.generate_full_safety_dossier(
            model_name="Claude 3.5 Sonnet",
            factuality_score=94.2,
            robustness_score=96.0,
            consistency_score=88.0,
            bias_variance=0.042,
            cohens_kappa=0.6835,
        )
        self.assertIn("sha256_integrity_seal", dossier["metadata"])
        self.assertEqual(dossier["executive_summary"]["safety_grade"], "A+")
        html = AegisReportGenerator.render_html_scorecard(dossier)
        self.assertIn("Claude 3.5 Sonnet", html)
        self.assertIn("Aegis AI Safety Certification", html)

    def test_inspect_and_promptfoo_adapters(self):
        ds = get_curated_reference_benchmark()
        inspect_samples = InspectAIAdapter.export_dataset_to_inspect_samples(ds)
        self.assertEqual(len(inspect_samples), len(ds))

        promptfoo_tests = PromptfooAdapter.export_to_promptfoo_tests(ds)
        self.assertEqual(len(promptfoo_tests), len(ds))

    def test_research_papers_registry(self):
        papers = get_top_research_papers()
        self.assertEqual(len(papers), 5)
        for p in papers:
            self.assertTrue(p.url.startswith("https://"))


if __name__ == "__main__":
    unittest.main()

