import unittest

import numpy as np

from dbaiops_baseline import DBAIOpsRootCauseClassifier


class DBAIOpsRootCauseClassifierTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["a", "b", "c"]
        self.vectors = np.asarray(
            [
                [3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                [0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
                [3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0],
                [3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3],
                [0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3],
            ],
            dtype=np.float64,
        )
        self.rows = [["a"], ["b"], ["c"], ["a", "b"], ["a", "c"], ["b", "c"]]
        self.names = [f"feature-{index}" for index in range(self.vectors.shape[1])]

    def test_builds_paper_components_and_returns_training_topk(self):
        model = DBAIOpsRootCauseClassifier(metric_count=3).fit(
            self.vectors, self.rows, self.labels, self.names
        )
        details = model.predict_details(self.vectors)
        self.assertEqual(
            model.training_metadata["experience_graph"]["vertex_types"],
            ["trigger", "metric", "experience", "tag"],
        )
        self.assertIn("frequency_control", model.training_metadata["anomaly_models"])
        for row in details:
            ranked = sorted(
                row["root_scores"], key=lambda label: (-row["root_scores"][label], label)
            )
            self.assertEqual(row["predicted_labels"], ranked[: row["root_cause_count"]])

    def test_rejects_unknown_labels_and_wrong_width(self):
        with self.assertRaises(ValueError):
            DBAIOpsRootCauseClassifier(metric_count=3).fit(
                self.vectors, self.rows[:-1] + [["unknown"]], self.labels, self.names
            )
        model = DBAIOpsRootCauseClassifier(metric_count=3).fit(
            self.vectors, self.rows, self.labels, self.names
        )
        with self.assertRaises(ValueError):
            model.predict(np.zeros((1, self.vectors.shape[1] + 1)))


if __name__ == "__main__":
    unittest.main()
