"""DBAIOps paper reproduction for the frozen DB-MAGS observations.

The paper leaves the anomaly equations and O&M graph contents database-specific.
This implementation derives both from each training fold: shallow trees encode
multi-metric anomaly equations, and a directed ExperienceGraph links trigger,
metric, tag, and experience vertices. Online inference performs proximity
expansion, abnormal-metric clipping, and a learned Top-k root ranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def _as_2d(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError("metric vectors must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError("metric vectors must be finite")
    return result


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    denominator = result.sum(axis=1, keepdims=True)
    return np.divide(result, denominator, out=np.zeros_like(result), where=denominator > 0)


class DBAIOpsRootCauseClassifier:
    """Correlation-aware anomaly models plus an evolving ExperienceGraph."""

    def __init__(self, *, seed: int = 42, metric_count: int = 5) -> None:
        self.seed = int(seed)
        self.metric_count = int(metric_count)
        self.labels: Tuple[str, ...] = ()
        self.feature_names: Tuple[str, ...] = ()
        self.location = np.empty(0)
        self.scale = np.empty(0)
        self.models: Dict[str, Optional[DecisionTreeClassifier]] = {}
        self.constant_targets: Dict[str, int] = {}
        self.root_metric_edges = np.empty((0, 0))
        self.metric_edges = np.empty((0, 0))
        self.root_edges = np.empty((0, 0))
        self.count_model: Optional[RandomForestClassifier] = None
        self.constant_root_count = 1
        self.training_metadata: Dict[str, Any] = {}

    def _standardize(self, vectors: np.ndarray) -> np.ndarray:
        return (vectors - self.location) / self.scale

    def _metric_sequences(self, standardized: np.ndarray) -> np.ndarray:
        return standardized.reshape(len(standardized), -1, self.metric_count)

    def _statistics(self, standardized: np.ndarray) -> np.ndarray:
        sequences = self._metric_sequences(standardized)
        mean = sequences.mean(axis=1)
        volatility = sequences.std(axis=1, ddof=0)
        maximum = sequences.max(axis=1)
        minimum = sequences.min(axis=1)
        deviation = sequences[:, -1, :] - sequences[:, 0, :]
        frequency = (np.abs(sequences) >= 1.0).mean(axis=1)
        return np.concatenate(
            [standardized, mean, volatility, maximum, minimum, deviation, frequency],
            axis=1,
        )

    def _metric_state(self, standardized: np.ndarray) -> np.ndarray:
        sequences = self._metric_sequences(standardized)
        magnitude = np.sort(np.abs(sequences), axis=1)[:, -3:, :].mean(axis=1)
        frequency_gate = (np.abs(sequences) >= 1.0).sum(axis=1) >= 3
        state = np.where(frequency_gate, magnitude, 0.0)
        empty = state.max(axis=1) <= 0
        state[empty] = magnitude[empty]
        maximum = state.max(axis=1, keepdims=True)
        return np.divide(state, maximum, out=np.zeros_like(state), where=maximum > 0)

    @staticmethod
    def _positive_probability(
        model: Optional[DecisionTreeClassifier],
        vectors: np.ndarray,
        constant: int,
    ) -> np.ndarray:
        if model is None:
            return np.full(len(vectors), float(constant), dtype=np.float64)
        probabilities = np.asarray(model.predict_proba(vectors), dtype=np.float64)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(vectors))

    def _scores(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        standardized = self._standardize(vectors)
        statistics = self._statistics(standardized)
        triggers = np.column_stack(
            [
                self._positive_probability(
                    self.models[label], statistics, self.constant_targets[label]
                )
                for label in self.labels
            ]
        )
        metric_state = self._metric_state(standardized)
        expanded_state = 0.75 * metric_state + 0.25 * (metric_state @ self.metric_edges)
        path_support = expanded_state @ self.root_metric_edges.T
        path_maximum = path_support.max(axis=1, keepdims=True)
        path_support = np.divide(
            path_support,
            path_maximum,
            out=np.zeros_like(path_support),
            where=path_maximum > 0,
        )
        proximity = triggers @ self.root_edges
        scores = 0.75 * triggers + 0.15 * path_support + 0.10 * proximity
        return scores, metric_state

    def fit(
        self,
        vectors: np.ndarray,
        label_rows: Sequence[Sequence[str]],
        labels: Sequence[str],
        feature_names: Sequence[str],
    ) -> "DBAIOpsRootCauseClassifier":
        vectors = _as_2d(vectors)
        if len(vectors) != len(label_rows) or len(feature_names) != vectors.shape[1]:
            raise ValueError("DBAIOps training inputs differ in length or width")
        if self.metric_count < 1 or vectors.shape[1] % self.metric_count:
            raise ValueError("feature width must contain complete metric time slices")
        self.labels = tuple(str(value) for value in labels)
        self.feature_names = tuple(str(value) for value in feature_names)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be a nonempty unique sequence")
        rows = [tuple(str(value) for value in row) for row in label_rows]
        if any(not row or not set(row) <= set(self.labels) for row in rows):
            raise ValueError("training rows must contain known root labels")

        self.location = np.median(vectors, axis=0)
        lower, upper = np.percentile(vectors, [25.0, 75.0], axis=0)
        standard = vectors.std(axis=0, ddof=0)
        self.scale = np.where(upper > lower, upper - lower, np.where(standard > 0, standard, 1.0))
        standardized = self._standardize(vectors)
        statistics = self._statistics(standardized)

        targets = np.column_stack(
            [[int(label in row) for row in rows] for label in self.labels]
        ).astype(np.int64)
        self.models = {}
        self.constant_targets = {}
        for index, label in enumerate(self.labels):
            unique = np.unique(targets[:, index])
            if len(unique) == 1:
                self.models[label] = None
                self.constant_targets[label] = int(unique[0])
            else:
                self.models[label] = DecisionTreeClassifier(
                    criterion="entropy",
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=self.seed,
                ).fit(statistics, targets[:, index])
                self.constant_targets[label] = 0

        metric_profiles = np.mean(np.abs(self._metric_sequences(standardized)), axis=1)
        root_metric = np.zeros((len(self.labels), self.metric_count), dtype=np.float64)
        for root in range(len(self.labels)):
            for metric in range(self.metric_count):
                if np.std(metric_profiles[:, metric]) and np.std(targets[:, root]):
                    root_metric[root, metric] = abs(
                        float(np.corrcoef(metric_profiles[:, metric], targets[:, root])[0, 1])
                    )
        self.root_metric_edges = _row_normalize(np.nan_to_num(root_metric))

        with np.errstate(divide="ignore", invalid="ignore"):
            metric_correlation = np.abs(np.corrcoef(metric_profiles, rowvar=False))
        metric_correlation = np.nan_to_num(metric_correlation)
        np.fill_diagonal(metric_correlation, 0.0)
        self.metric_edges = _row_normalize(metric_correlation)

        similarity = self.root_metric_edges @ self.root_metric_edges.T
        cooccurrence = targets.T @ targets
        denominator = np.maximum(1, targets.sum(axis=0))
        cooccurrence = cooccurrence / denominator[:, None]
        root_edges = 0.5 * similarity + 0.5 * cooccurrence
        np.fill_diagonal(root_edges, 0.0)
        self.root_edges = _row_normalize(root_edges)

        count_targets = np.asarray([len(set(row)) for row in rows], dtype=np.int64)
        unique_counts = np.unique(count_targets)
        if len(unique_counts) == 1:
            self.count_model = None
            self.constant_root_count = int(unique_counts[0])
        else:
            graph_scores, _ = self._scores(vectors)
            self.count_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.seed + 1,
                n_jobs=-1,
            ).fit(graph_scores, count_targets)

        metric_names = [self.feature_names[index] for index in range(self.metric_count)]
        self.training_metadata = {
            "paper_reproduction": "DBAIOps components instantiated from the outer training fold",
            "input_feature_count": int(vectors.shape[1]),
            "metric_hierarchy": {
                "raw_metric_time_vertex_count": int(vectors.shape[1]),
                "metric_family_count": self.metric_count,
                "lazy_statistics": ["mean", "volatility", "maximum", "minimum", "deviation", "frequency"],
            },
            "anomaly_models": {
                "count": len(self.labels),
                "equation_representation": "depth-6 entropy decision trees over multi-metric statistics",
                "frequency_control": "at least 3 of 5 time bins beyond the training-fold dynamic baseline",
            },
            "experience_graph": {
                "vertex_types": ["trigger", "metric", "experience", "tag"],
                "root_metric_edge_count": int(np.count_nonzero(self.root_metric_edges)),
                "metric_proximity_edge_count": int(np.count_nonzero(self.metric_edges)),
                "cross_anomaly_edge_count": int(np.count_nonzero(self.root_edges)),
                "metric_vertex_examples": metric_names,
                "evolution": "metric proximity expansion followed by current abnormal-state clipping",
            },
            "root_cause_analyser": {
                "output": "closed DB-MAGS root inventory",
                "ranking": "trigger, graph-path, and cross-anomaly evidence",
                "selection": "training-fold learned root count followed by Top-k graph scores",
                "free_form_report_generation": False,
            },
            "test_labels_used": False,
        }
        return self

    def predict_details(self, vectors: np.ndarray) -> List[Dict[str, Any]]:
        vectors = _as_2d(vectors)
        if not self.training_metadata:
            raise RuntimeError("DBAIOpsRootCauseClassifier must be fit before prediction")
        if vectors.shape[1] != self.training_metadata["input_feature_count"]:
            raise ValueError("feature width differs from the fitted DBAIOps graph")
        scores, metric_state = self._scores(vectors)
        if self.count_model is None:
            root_counts = np.full(len(vectors), self.constant_root_count, dtype=np.int64)
        else:
            root_counts = np.asarray(self.count_model.predict(scores), dtype=np.int64)

        results: List[Dict[str, Any]] = []
        for row_scores, row_state, raw_count in zip(scores, metric_state, root_counts):
            root_count = max(1, min(int(raw_count), len(self.labels)))
            order = sorted(
                range(len(self.labels)),
                key=lambda index: (-float(row_scores[index]), self.labels[index]),
            )
            results.append(
                {
                    "predicted_labels": [self.labels[index] for index in order[:root_count]],
                    "root_cause_count": root_count,
                    "root_scores": {
                        label: float(score) for label, score in zip(self.labels, row_scores)
                    },
                    "active_metric_family_count": int(np.count_nonzero(row_state)),
                }
            )
        return results

    def predict(self, vectors: np.ndarray) -> List[List[str]]:
        return [list(row["predicted_labels"]) for row in self.predict_details(vectors)]
