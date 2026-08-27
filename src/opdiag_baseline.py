"""OpDiag root detection with a training-only Top-k output adapter.

OpDiag trains one random-forest classifier per anomaly from observed KPIs
(paper Section IV-C and ``diag.py`` in the official release). SQL and operator
attribution is downstream of root detection and is outside the root-set metric
used here. The only multi-root extension predicts k from training data and
returns the k highest anomaly probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _as_2d(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError("metric vectors must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError("metric vectors must be finite")
    return result


def _positive_probability(
    model: Optional[RandomForestClassifier], vectors: np.ndarray, constant: int
) -> np.ndarray:
    if model is None:
        return np.full(len(vectors), float(constant), dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(vectors), dtype=np.float64)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(vectors), dtype=np.float64)
    return probabilities[:, classes.index(1)]


class OpDiagRootCauseClassifier:
    """Paper-aligned independent anomaly classifiers with learned Top-k."""

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = int(seed)
        self.labels: Tuple[str, ...] = ()
        self.models: Dict[str, Optional[RandomForestClassifier]] = {}
        self.constant_targets: Dict[str, int] = {}
        self.count_model: Optional[RandomForestClassifier] = None
        self.constant_root_count = 1
        self.training_metadata: Dict[str, Any] = {}

    def _new_model(self, *, seed: int) -> RandomForestClassifier:
        # These are the official release's anomaly-classifier settings. The
        # sklearn default for n_estimators is 100.
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=8,
            random_state=int(seed),
            n_jobs=-1,
        )

    def fit(
        self,
        vectors: np.ndarray,
        label_rows: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> "OpDiagRootCauseClassifier":
        vectors = _as_2d(vectors)
        if len(vectors) != len(label_rows):
            raise ValueError("metric vectors and label rows differ in length")
        self.labels = tuple(str(label) for label in labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be a nonempty unique sequence")
        known = set(self.labels)
        normalized_rows = [tuple(str(value) for value in row) for row in label_rows]
        if any(not row or not set(row) <= known for row in normalized_rows):
            raise ValueError("training rows must contain known root labels")

        self.models = {}
        self.constant_targets = {}
        for label in self.labels:
            targets = np.asarray(
                [int(label in row) for row in normalized_rows], dtype=np.int64
            )
            unique = np.unique(targets)
            if len(unique) == 1:
                self.models[label] = None
                self.constant_targets[label] = int(unique[0])
            else:
                self.models[label] = self._new_model(seed=self.seed).fit(
                    vectors, targets
                )
                self.constant_targets[label] = 0

        count_targets = np.asarray(
            [len(set(row)) for row in normalized_rows], dtype=np.int64
        )
        unique_counts = np.unique(count_targets)
        if len(unique_counts) == 1:
            self.count_model = None
            self.constant_root_count = int(unique_counts[0])
        else:
            self.count_model = self._new_model(seed=self.seed + 1).fit(
                vectors, count_targets
            )

        self.training_metadata = {
            "input_feature_count": int(vectors.shape[1]),
            "paper_component": "OpDiag independent anomaly classifiers (Section IV-C)",
            "anomaly_classifier": {
                "type": "RandomForestClassifier",
                "count": len(self.labels),
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": 8,
                "random_state": self.seed,
            },
            "multi_root_output_adapter": {
                "type": "training-fold RandomForestClassifier root-count model",
                "root_count_classes": [int(value) for value in unique_counts],
                "selection": "top-k anomaly probabilities",
                "changes_to_opdiag_core": "none; adapter changes only output cardinality",
            },
            "uses_kpi": True,
            "uses_sql_or_plan_for_root_detection": False,
            "sql_operator_attribution_evaluated": False,
            "test_labels_used": False,
        }
        return self

    def predict_details(self, vectors: np.ndarray) -> List[Dict[str, Any]]:
        vectors = _as_2d(vectors)
        if not self.training_metadata:
            raise RuntimeError("OpDiagRootCauseClassifier must be fit before prediction")
        if vectors.shape[1] != self.training_metadata["input_feature_count"]:
            raise ValueError("feature width differs from the fitted OpDiag classifiers")

        probability_matrix = np.column_stack(
            [
                _positive_probability(
                    self.models[label], vectors, self.constant_targets[label]
                )
                for label in self.labels
            ]
        )
        if self.count_model is None:
            root_counts = np.full(
                len(vectors), self.constant_root_count, dtype=np.int64
            )
        else:
            root_counts = np.asarray(self.count_model.predict(vectors), dtype=np.int64)

        results: List[Dict[str, Any]] = []
        for probabilities, raw_count in zip(probability_matrix, root_counts):
            root_count = max(1, min(int(raw_count), len(self.labels)))
            order = sorted(
                range(len(self.labels)),
                key=lambda index: (-float(probabilities[index]), self.labels[index]),
            )
            results.append(
                {
                    "predicted_labels": [
                        self.labels[index] for index in order[:root_count]
                    ],
                    "root_cause_count": root_count,
                    "anomaly_probabilities": {
                        label: float(probability)
                        for label, probability in zip(self.labels, probabilities)
                    },
                }
            )
        return results

    def predict(self, vectors: np.ndarray) -> List[List[str]]:
        return [list(row["predicted_labels"]) for row in self.predict_details(vectors)]
