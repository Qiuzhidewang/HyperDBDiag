import json
import tempfile
import unittest
from pathlib import Path

import torch

from operator_bound_experiment import (
    LearnedHierarchicalRanker,
    _case_fold,
    _interpolate_single_row,
    _operator_key,
    _single_row_ig_scores,
    _summarize_rows,
    audit_dataset,
    load_cases,
)


class OperatorBoundExperimentTests(unittest.TestCase):
    SOURCE = Path("data/dbmags_operator_bound_v4")

    def test_formal_dataset_passes_integrity_audit(self):
        report = audit_dataset(self.SOURCE)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["collected_case_count"], 296)
        self.assertEqual(report["unsupported_case_count"], 12)
        self.assertEqual(report["collected_by_root_cardinality"], {"1": 96, "2": 120, "3": 80})
        self.assertEqual(report["distinct_root_combinations"], {"1": 6, "2": 15, "3": 20})

    def test_repeated_collections_are_balanced_across_two_folds(self):
        cases, _ = load_cases(self.SOURCE)
        by_combination = {}
        for case in cases:
            by_combination.setdefault(tuple(sorted(case.roots)), [0, 0])[_case_fold(case)] += 1
        self.assertEqual(set(map(tuple, by_combination.values())), {(8, 8), (4, 4), (2, 2)})

    def test_operator_identity_is_scoped_by_sql(self):
        self.assertNotEqual(_operator_key("sql-a", "op-x"), _operator_key("sql-b", "op-x"))

    def test_integrated_gradient_path_changes_only_target_row(self):
        matrix = torch.tensor([[2.0, 4.0], [3.0, 9.0], [5.0, 10.0]])
        interpolated, target = _interpolate_single_row(matrix, 1, 0.25)
        self.assertTrue(torch.equal(interpolated[0], matrix[0]))
        self.assertTrue(torch.equal(interpolated[2], matrix[2]))
        self.assertTrue(torch.equal(interpolated[1], matrix[1] * 0.25))
        interpolated.sum().backward()
        self.assertTrue(torch.equal(target.grad, torch.ones_like(target)))

    def test_integrated_gradient_integrates_gradient_norms_before_scoring(self):
        matrix = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        scores = _single_row_ig_scores(matrix, lambda value: torch.sum(value ** 2), steps=4)
        self.assertTrue(torch.allclose(scores, torch.tensor([31.25, 5.0])))

    def test_case_coverage_counts_multi_root_case_once(self):
        rows = [
            {
                "case_id": "c1", "root_cardinality": 2, "root": root,
                "sql_hit_at_1": True, "operator_hit_at_1": True,
                "all_sql_hit_at_1": True, "all_chain_hit_at_1": True,
            }
            for root in ("group_by", "missing_index")
        ]
        report = _summarize_rows(rows)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["by_root_cardinality"]["2"]["case_count"], 1)
        self.assertEqual(
            report["root_sql_pairs"], {"denominator": 2, "hit_at_1": 1.0}
        )
        self.assertNotIn("hit_at_2", report["root_sql_pairs"])

    def test_ranker_does_not_accept_test_truth(self):
        cases, _ = load_cases(self.SOURCE)
        training = [case for case in cases if _case_fold(case) != 0]
        evaluation = next(case for case in cases if _case_fold(case) == 0)
        ranker = LearnedHierarchicalRanker(seed=7).fit(training)
        blind = evaluation.blind_view()
        self.assertFalse(hasattr(blind, "truth"))
        result = ranker.rank(blind, evaluation.roots[0])
        self.assertTrue(result["sql_ranking"])
        self.assertTrue(result["operator_rankings_by_sql"])
        self.assertFalse(ranker.metadata()["test_ground_truth_read_by_ranker"])

    def test_unconditioned_binder_registers_one_shared_model_pair(self):
        ranker = LearnedHierarchicalRanker(seed=7, root_conditioning=False)
        self.assertEqual(ranker.metadata()["epdg_binding"], "not used")
        self.assertIn("one shared", ranker.metadata()["training"])

    def test_audit_rejects_target_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            source_case = self.SOURCE / "cases" / "obv4-variant-1-001-missing_index"
            case_dir = target / "cases" / source_case.name
            case_dir.mkdir(parents=True)
            for name in ("case.json", "plans.json", "metrics.json", "ground_truth.json"):
                (case_dir / name).write_bytes((source_case / name).read_bytes())
            blind = json.loads((source_case / "blind_candidates.json").read_text())
            blind["target"] = True
            (case_dir / "blind_candidates.json").write_text(json.dumps(blind))
            manifest = {
                "protocol": "dbmags-operator-bound-extension-v2",
                "sample_schedule_count": 1,
                "results": [{"status": "collected", "labels": ["missing_index"]}],
                "counts": {"collected": 1},
            }
            (target / "dataset_manifest.json").write_text(json.dumps(manifest))
            report = audit_dataset(target)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("forbidden" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
