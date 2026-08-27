import unittest

import numpy as np

from opdiag_baseline import OpDiagRootCauseClassifier


class OpDiagRootCauseClassifierTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["a", "b", "c"]
        self.vectors = np.asarray(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
                [2.5, 2.5, 0.0],
                [2.5, 0.0, 2.5],
                [0.0, 2.5, 2.5],
            ],
            dtype=np.float64,
        )
        self.label_rows = [
            ["a"],
            ["b"],
            ["c"],
            ["a", "b"],
            ["a", "c"],
            ["b", "c"],
        ]

    def test_uses_paper_rf_settings_and_training_topk(self):
        model = OpDiagRootCauseClassifier(seed=42).fit(
            self.vectors, self.label_rows, self.labels
        )
        details = model.predict_details(self.vectors)

        self.assertEqual(len(details), len(self.vectors))
        self.assertTrue(all(row["predicted_labels"] for row in details))
        for row in details:
            ranked = sorted(
                row["anomaly_probabilities"],
                key=lambda label: (-row["anomaly_probabilities"][label], label),
            )
            self.assertEqual(
                row["predicted_labels"], ranked[: row["root_cause_count"]]
            )
            self.assertEqual(
                len(row["predicted_labels"]), row["root_cause_count"]
            )
        self.assertEqual(
            model.training_metadata["anomaly_classifier"],
            {
                "type": "RandomForestClassifier",
                "count": 3,
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": 8,
                "random_state": 42,
            },
        )
        self.assertEqual(
            model.training_metadata["multi_root_output_adapter"]["selection"],
            "top-k anomaly probabilities",
        )
        self.assertFalse(model.training_metadata["sql_operator_attribution_evaluated"])

    def test_rejects_unknown_labels_and_mismatched_query_width(self):
        with self.assertRaises(ValueError):
            OpDiagRootCauseClassifier().fit(
                self.vectors, self.label_rows[:-1] + [["unknown"]], self.labels
            )

        model = OpDiagRootCauseClassifier().fit(
            self.vectors, self.label_rows, self.labels
        )
        with self.assertRaises(ValueError):
            model.predict(np.zeros((1, self.vectors.shape[1] + 1), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
