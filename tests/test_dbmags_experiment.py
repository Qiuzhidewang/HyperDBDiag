import json
import unittest
from pathlib import Path

from dbmags_experiment import _canonical_method


class DBMAGSExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(
            Path("runs/dbmags-ablation/full_report.json").read_text(encoding="utf-8")
        )

    def test_canonical_prediction_uses_registered_six_fold_report(self):
        method, dataset = _canonical_method(self.report, 20260802, "hypergraph")
        self.assertEqual(len(dataset["folds"]), 6)
        self.assertTrue(
            all(row["evaluation_count"] == 110 for row in dataset["folds"])
        )
        self.assertEqual(len(method["results"]), 660)
        self.assertAlmostEqual(method["overall"]["exact_set_accuracy"], 0.906060606060606)

    def test_canonical_prediction_rejects_seed_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "seeds differ"):
            _canonical_method(self.report, 1, "hypergraph")

    def test_canonical_method_accepts_completed_llm_ablation_report(self):
        report = json.loads(
            Path("runs/dbmags-ablation/full_report_llm_xhigh.json").read_text(
                encoding="utf-8"
            )
        )
        method, dataset = _canonical_method(report, 20260802, "hyperdbdiag")
        self.assertEqual(len(dataset["folds"]), 6)
        self.assertEqual(len(method["results"]), 660)
        self.assertAlmostEqual(method["overall"]["exact_set_accuracy"], 0.9196969696969697)

    def test_ablation_report_does_not_split_single_and_mixed_roots(self):
        dataset = self.report["datasets"]["dbmags_sql_interaction_subset"]
        for method in dataset["methods"].values():
            self.assertNotIn("by_root_cardinality", method)

    def test_main_comparison_keeps_root_cardinality_breakdown(self):
        report = json.loads(
            Path("runs/dbmags-main-comparison/full_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(report["methods"]), {"opdiag", "dbaiops", "epdg_hypergraph"}
        )
        self.assertNotIn("not_evaluated", report)
        for method in report["methods"].values():
            self.assertEqual(set(method["by_root_cardinality"]), {"1", "2"})
            self.assertEqual(len(method["results"]), 660)


if __name__ == "__main__":
    unittest.main()
