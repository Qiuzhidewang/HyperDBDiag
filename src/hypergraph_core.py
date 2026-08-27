"""Shared EPDG-grounded lightweight-H diagnosis models.

The module deliberately contains no dataset loading, split logic, or result
reporting. A caller supplies a training-fold feature matrix and declared OOF
groups; the higher-order incidence structure and available EPDG paths are then
built solely from that training fold.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


DEFAULT_SEED = 20260802
DEFAULT_ACTIVATION_THRESHOLD = 1.0
DEFAULT_PROPAGATION_STEPS = 1
DEFAULT_DECODER_N_ESTIMATORS = 500
DEFAULT_BINARY_GRAPH_N_ESTIMATORS = 160
DEFAULT_BINARY_GRAPH_MAX_DEPTH = 7
DEFAULT_BINARY_GRAPH_MIN_SAMPLES_LEAF = 2
DEFAULT_BINARY_GRAPH_THRESHOLD = 0.50
DEFAULT_CANDIDATE_POOL_MAX_SIZE = 4
DEFAULT_CANDIDATE_POOL_TARGET_RECALL = 0.95
DEFAULT_EPDG_MIN_OOF_EXACT_GAIN = 0.01


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / float(len(materialized) or 1)


def _exact(expected: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]) -> float:
    return _mean(set(left) == set(right) for left, right in zip(expected, predicted))


def _component_f1(
    expected: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]], labels: Sequence[str]
) -> float:
    true_positive = sum(
        label in observed and label in predicted_row
        for observed, predicted_row in zip(expected, predicted)
        for label in labels
    )
    false_positive = sum(
        label not in observed and label in predicted_row
        for observed, predicted_row in zip(expected, predicted)
        for label in labels
    )
    false_negative = sum(
        label in observed and label not in predicted_row
        for observed, predicted_row in zip(expected, predicted)
        for label in labels
    )
    precision = true_positive / float(true_positive + false_positive or 1)
    recall = true_positive / float(true_positive + false_negative or 1)
    return 2.0 * precision * recall / float(precision + recall or 1)


def _as_2d(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError("metric values must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError("metric values must be finite")
    return result


def _candidate_sets(
    label_rows: Sequence[Sequence[str]], labels: Sequence[str]
) -> List[Tuple[int, ...]]:
    index = {label: position for position, label in enumerate(labels)}
    return sorted(
        {tuple(sorted(index[label] for label in row)) for row in label_rows},
        key=lambda candidate: (len(candidate), candidate),
    )


def _candidate_targets(
    label_rows: Sequence[Sequence[str]],
    labels: Sequence[str],
    candidates: Sequence[Tuple[int, ...]],
) -> np.ndarray:
    index = {label: position for position, label in enumerate(labels)}
    candidate_index = {candidate: position for position, candidate in enumerate(candidates)}
    return np.asarray(
        [candidate_index[tuple(sorted(index[label] for label in row))] for row in label_rows],
        dtype=np.int64,
    )


def _sets_from_candidate_indices(
    indices: np.ndarray, labels: Sequence[str], candidates: Sequence[Tuple[int, ...]]
) -> List[List[str]]:
    return [[labels[root] for root in candidates[int(index)]] for index in indices]


def _assert_candidate_coverage(
    expected: Sequence[Sequence[str]], candidates: Sequence[Tuple[int, ...]], labels: Sequence[str]
) -> None:
    label_index = {label: index for index, label in enumerate(labels)}
    unseen = {
        tuple(sorted(label_index[label] for label in expected_row)) for expected_row in expected
    } - set(candidates)
    if unseen:
        raise ValueError(
            "A held-out block contains a root set absent from the training candidate inventory: "
            f"{sorted(unseen)}"
        )


class HypergraphDiffusionResidualEncoder:
    """Encode a query through a training-fold incidence matrix.

    The first half of the output is the conventional static HGNN residual.
    The second half is a query-dependent aggregation over frozen training
    hyperedges.  Unlike a clique expansion, this retains each training sample
    as a higher-order relation until the query has selected relevant
    hyperedges.  The query never changes the fitted incidence structure.
    """

    def __init__(
        self,
        activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
        propagation_steps: int = DEFAULT_PROPAGATION_STEPS,
    ) -> None:
        if activation_threshold < 0.0:
            raise ValueError("activation_threshold must be nonnegative")
        if propagation_steps < 1:
            raise ValueError("propagation_steps must be positive")
        self.activation_threshold = float(activation_threshold)
        self.propagation_steps = int(propagation_steps)
        self.scaler: Optional[StandardScaler] = None
        self.propagation: Optional[np.ndarray] = None
        self.edge_unit_vectors: Optional[np.ndarray] = None
        self.reference_h_shape: Tuple[int, int] = ()
        self.reference_activation_rate = 0.0
        self.reference_nonzero_incidence_count = 0
        self.reference_equivalent_pairwise_occurrence_count = 0
        self.reference_unique_pair_count = 0
        self.input_feature_count = 0
        self.dynamic_neighbor_count = 0
        self.root_aware_training_incidence = False

    def _incidence(self, standardized: np.ndarray) -> np.ndarray:
        standardized = _as_2d(standardized)
        active = np.abs(standardized) >= self.activation_threshold
        # A conventional incidence matrix is nonnegative.  Splitting each KPI
        # into increase/decrease atoms preserves direction without cancellation.
        positive = np.maximum(standardized, 0.0) * active.astype(np.float64)
        negative = np.maximum(-standardized, 0.0) * active.astype(np.float64)
        incidence = np.concatenate([positive, negative], axis=1).T
        empty_columns = np.sum(np.abs(incidence), axis=0) <= 0.0
        if np.any(empty_columns):
            empty_rows = np.flatnonzero(empty_columns)
            strongest = np.argmax(np.abs(standardized[empty_rows]), axis=1)
            fallback = standardized[empty_rows, strongest]
            atoms = strongest + (fallback < 0.0).astype(np.int64) * standardized.shape[1]
            incidence[atoms, empty_rows] = np.where(np.abs(fallback) > 0.0, np.abs(fallback), 0.5)
        return incidence

    def fit(
        self,
        vectors: np.ndarray,
        training_metric_weights: Optional[np.ndarray] = None,
    ) -> "HypergraphDiffusionResidualEncoder":
        vectors = _as_2d(vectors)
        self.input_feature_count = int(vectors.shape[1])
        self.scaler = StandardScaler().fit(vectors)
        incidence = self._incidence(self.scaler.transform(vectors))
        if training_metric_weights is not None:
            weights = np.asarray(training_metric_weights, dtype=np.float64)
            if weights.shape != vectors.shape or not np.all(np.isfinite(weights)):
                raise ValueError("root-aware training weights must align to metric vectors")
            if np.any(weights <= 0.0):
                raise ValueError("root-aware training weights must be positive")
            incidence *= np.concatenate([weights, weights], axis=1).T
            self.root_aware_training_incidence = True
        node_degree = np.maximum(np.sum(np.abs(incidence), axis=1), 1e-12)
        edge_degree = np.maximum(np.sum(np.abs(incidence), axis=0), 1e-12)
        propagation = (incidence / edge_degree[None, :]) @ incidence.T
        propagation /= np.sqrt(node_degree[:, None])
        propagation /= np.sqrt(node_degree[None, :])
        self.propagation = propagation
        edge_vectors = incidence.T
        edge_norms = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        self.edge_unit_vectors = edge_vectors / np.maximum(edge_norms, 1e-12)
        self.reference_h_shape = tuple(int(value) for value in incidence.shape)
        active_incidence = np.abs(incidence) > 0.0
        self.reference_nonzero_incidence_count = int(np.count_nonzero(active_incidence))
        self.reference_activation_rate = self.reference_nonzero_incidence_count / float(
            incidence.size
        )
        hyperedge_degrees = np.count_nonzero(active_incidence, axis=0).astype(np.int64)
        self.reference_equivalent_pairwise_occurrence_count = int(
            np.sum(hyperedge_degrees * (hyperedge_degrees - 1) // 2)
        )
        pair_coactivation = active_incidence.astype(np.int32) @ active_incidence.T.astype(
            np.int32
        )
        self.reference_unique_pair_count = int(
            np.count_nonzero(np.triu(pair_coactivation, k=1))
        )
        self.dynamic_neighbor_count = min(16, int(edge_vectors.shape[0]))
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if (
            self.scaler is None
            or self.propagation is None
            or self.edge_unit_vectors is None
        ):
            raise RuntimeError("HypergraphDiffusionResidualEncoder must be fit before transform")
        vectors = _as_2d(vectors)
        if vectors.shape[1] != self.input_feature_count:
            raise ValueError("feature width differs from the reference hypergraph")
        query_incidence = self._incidence(self.scaler.transform(vectors))
        propagated = query_incidence
        for _ in range(self.propagation_steps):
            propagated = self.propagation @ propagated
        static_residual = (propagated - query_incidence).T

        query_vectors = query_incidence.T
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_units = query_vectors / np.maximum(query_norms, 1e-12)
        similarities = np.maximum(query_units @ self.edge_unit_vectors.T, 0.0)
        if self.dynamic_neighbor_count < similarities.shape[1]:
            selected = np.argpartition(
                similarities,
                -self.dynamic_neighbor_count,
                axis=1,
            )[:, -self.dynamic_neighbor_count :]
            weights = np.zeros_like(similarities)
            row_indices = np.arange(len(similarities))[:, None]
            weights[row_indices, selected] = similarities[row_indices, selected]
        else:
            weights = similarities
        # Squaring retains only coherent high-order matches while keeping the
        # operation permutation invariant over training hyperedges.
        weights = weights * weights
        context = weights @ self.edge_unit_vectors
        context_norms = np.linalg.norm(context, axis=1, keepdims=True)
        context *= query_norms / np.maximum(context_norms, 1e-12)
        dynamic_residual = context - query_vectors
        return np.concatenate([static_residual, dynamic_residual], axis=1)

    def metadata(self) -> Dict[str, Any]:
        if self.scaler is None or self.propagation is None or self.edge_unit_vectors is None:
            raise RuntimeError("HypergraphDiffusionResidualEncoder must be fit before metadata")
        return {
            "vertex_schema": "fixed_signed_metric_atoms",
            "reference_h_shape": list(self.reference_h_shape),
            "input_feature_count": self.input_feature_count,
            "signed_vertex_count": self.reference_h_shape[0],
            "training_hyperedge_count": self.reference_h_shape[1],
            "nonzero_incidence_count": self.reference_nonzero_incidence_count,
            "equivalent_pairwise_occurrence_count": (
                self.reference_equivalent_pairwise_occurrence_count
            ),
            "unique_pair_count_after_clique_deduplication": self.reference_unique_pair_count,
            "incidence_reduction_vs_pairwise_occurrences": (
                1.0
                - self.reference_nonzero_incidence_count
                / float(self.reference_equivalent_pairwise_occurrence_count)
                if self.reference_equivalent_pairwise_occurrence_count
                else 0.0
            ),
            "h_value": "nonnegative_signed_standardized_delta_times_activation_mask",
            "activation_threshold_standardized_units": self.activation_threshold,
            "reference_activation_rate": self.reference_activation_rate,
            "propagation": "Dv_inverse_half_H_De_inverse_H_T_Dv_inverse_half",
            "propagation_steps": self.propagation_steps,
            "dynamic_hyperedge_aggregation": "cosine_topk_over_frozen_training_hyperedges",
            "dynamic_neighbor_count": self.dynamic_neighbor_count,
            "output": "static_H_residual_concatenated_with_dynamic_hyperedge_residual",
            "uses_root_or_candidate_labels": self.root_aware_training_incidence,
            "root_aware_training_incidence": self.root_aware_training_incidence,
            "query_incidence_uses_root_labels": False,
            "test_query_in_propagation": False,
            "test_query_changes_training_hyperedges": False,
        }


class BinaryMetricGraphResidualEncoder:
    """Pairwise KPI graph control built only from a training-fold correlation matrix."""

    def __init__(self, neighbor_count: int = 3) -> None:
        if neighbor_count < 1:
            raise ValueError("neighbor_count must be positive")
        self.neighbor_count = int(neighbor_count)
        self.scaler: Optional[StandardScaler] = None
        self.propagation: Optional[np.ndarray] = None
        self.input_feature_count = 0
        self.edge_count = 0

    def fit(self, vectors: np.ndarray) -> "BinaryMetricGraphResidualEncoder":
        vectors = _as_2d(vectors)
        self.input_feature_count = int(vectors.shape[1])
        self.scaler = StandardScaler().fit(vectors)
        standardized = self.scaler.transform(vectors)
        if self.input_feature_count == 1:
            adjacency = np.ones((1, 1), dtype=np.float64)
        else:
            centered = standardized - np.mean(standardized, axis=0, keepdims=True)
            covariance = centered.T @ centered
            scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = np.abs(covariance / (scale[:, None] * scale[None, :]))
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 0.0)
            adjacency = np.zeros_like(correlation)
            width = min(self.neighbor_count, self.input_feature_count - 1)
            for node in range(self.input_feature_count):
                neighbors = np.argsort(correlation[node], kind="stable")[-width:]
                adjacency[node, neighbors] = correlation[node, neighbors]
            adjacency = np.maximum(adjacency, adjacency.T)
            self.edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))
            adjacency += np.eye(self.input_feature_count, dtype=np.float64)
        degree = np.maximum(np.sum(adjacency, axis=1), 1e-12)
        self.propagation = adjacency / np.sqrt(degree[:, None] * degree[None, :])
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.propagation is None:
            raise RuntimeError("BinaryMetricGraphResidualEncoder must be fit before transform")
        vectors = _as_2d(vectors)
        if vectors.shape[1] != self.input_feature_count:
            raise ValueError("feature width differs from the reference binary graph")
        standardized = self.scaler.transform(vectors)
        return self.aggregate(vectors) - standardized

    def aggregate(self, vectors: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.propagation is None:
            raise RuntimeError("BinaryMetricGraphResidualEncoder must be fit before aggregate")
        vectors = _as_2d(vectors)
        if vectors.shape[1] != self.input_feature_count:
            raise ValueError("feature width differs from the reference binary graph")
        return self.scaler.transform(vectors) @ self.propagation.T

    def metadata(self) -> Dict[str, Any]:
        if self.scaler is None or self.propagation is None:
            raise RuntimeError("BinaryMetricGraphResidualEncoder must be fit before metadata")
        return {
            "vertex_schema": "metric_time_atoms",
            "input_feature_count": self.input_feature_count,
            "vertex_count": self.input_feature_count,
            "pairwise_relation": "absolute_pearson_training_fold_knn",
            "neighbor_count": min(self.neighbor_count, max(self.input_feature_count - 1, 0)),
            "undirected_edge_count": self.edge_count,
            "propagation": "D_inverse_half_A_plus_I_D_inverse_half",
            "test_query_in_graph_construction": False,
        }


class EPDGPathPrior:
    """Project registered EPDG root-to-metric paths onto a query.

    The graph is deliberately small and explicit: callers provide the edge
    weights for the feature names that are actually available to the current
    dataset.  A path with no corresponding observed feature contributes no
    score, so the prior cannot manufacture SQL/operator evidence or silently
    treat a generic metric as root-specific.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        path_edges: Optional[Mapping[str, Mapping[str, float]]] = None,
        semantic_path_edges: Optional[Mapping[str, Sequence[str]]] = None,
        minimum_path_strength: float = 0.25,
        minimum_direction_stability: float = 0.75,
    ) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("EPDG feature names must be nonempty and unique")
        self.path_edges = {
            str(root): {str(feature): float(weight) for feature, weight in edges.items()}
            for root, edges in (path_edges or {}).items()
        }
        self.semantic_path_edges = {
            str(root): frozenset(str(atom) for atom in atoms if str(atom))
            for root, atoms in (semantic_path_edges or {}).items()
        }
        if any(weight < 0.0 or not np.isfinite(weight) for edges in self.path_edges.values() for weight in edges.values()):
            raise ValueError("EPDG path weights must be finite and nonnegative")
        if minimum_path_strength < 0.0:
            raise ValueError("EPDG minimum path strength must be nonnegative")
        if not 0.0 < minimum_direction_stability <= 1.0:
            raise ValueError("EPDG direction stability must lie in (0, 1]")
        self.minimum_path_strength = float(minimum_path_strength)
        self.minimum_direction_stability = float(minimum_direction_stability)
        self.labels: Tuple[str, ...] = ()
        self.edge_matrix: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.registered_edge_count = 0
        self.learned_edge_count = 0
        self.semantic_edge_count = sum(len(atoms) for atoms in self.semantic_path_edges.values())

    def fit(
        self,
        vectors: np.ndarray,
        labels: Sequence[str],
        train_labels: Optional[Sequence[Sequence[str]]] = None,
        groups: Optional[Sequence[Any]] = None,
    ) -> "EPDGPathPrior":
        vectors = _as_2d(vectors)
        if vectors.shape[1] != len(self.feature_names):
            raise ValueError("EPDG feature names and vectors differ in width")
        self.labels = tuple(str(label) for label in labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("EPDG root labels must be nonempty and unique")
        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        matrix = np.zeros((len(self.labels), len(self.feature_names)), dtype=np.float64)
        for root_index, root in enumerate(self.labels):
            for feature, weight in self.path_edges.get(root, {}).items():
                if feature in feature_index:
                    matrix[root_index, feature_index[feature]] = weight
        registered = matrix != 0.0
        self.registered_edge_count = int(np.count_nonzero(registered))
        self.scaler = StandardScaler().fit(vectors)
        if train_labels is not None or groups is not None:
            if train_labels is None or groups is None:
                raise ValueError("EPDG learned paths require both labels and groups")
            if len(train_labels) != len(vectors) or len(groups) != len(vectors):
                raise ValueError("EPDG learned path inputs differ in length")
            group_values = sorted(set(groups), key=str)
            if len(group_values) < 2:
                raise ValueError("EPDG learned paths require at least two groups")
            standardized = self.scaler.transform(vectors)
            group_array = np.asarray(groups, dtype=object)
            label_sets = tuple(frozenset(row) for row in train_labels)
            learned = np.zeros_like(matrix)
            for root_index, root in enumerate(self.labels):
                deltas: List[np.ndarray] = []
                for group in group_values:
                    in_group = group_array == group
                    present = in_group & np.asarray(
                        [root in row for row in label_sets], dtype=bool
                    )
                    absent = in_group & ~present
                    if np.any(present) and np.any(absent):
                        deltas.append(
                            np.mean(standardized[present], axis=0)
                            - np.mean(standardized[absent], axis=0)
                        )
                if not deltas:
                    continue
                rows = np.vstack(deltas)
                mean_delta = np.mean(rows, axis=0)
                direction = np.sign(mean_delta)
                stability = np.mean(np.sign(rows) == direction[None, :], axis=0)
                keep = (
                    (np.abs(mean_delta) >= self.minimum_path_strength)
                    & (stability >= self.minimum_direction_stability)
                )
                learned[root_index, keep] = mean_delta[keep]
            learned[registered] = matrix[registered]
            matrix = learned
            self.learned_edge_count = int(np.count_nonzero((matrix != 0.0) & ~registered))
        self.edge_matrix = matrix
        return self

    @property
    def active_edge_count(self) -> int:
        return int(np.count_nonzero(self.edge_matrix)) if self.edge_matrix is not None else 0

    def metric_root_scores(self, vectors: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.edge_matrix is None:
            raise RuntimeError("EPDGPathPrior must be fit before prediction")
        vectors = _as_2d(vectors)
        if vectors.shape[1] != len(self.feature_names):
            raise ValueError("EPDG feature width differs from the fitted prior")
        standardized = self.scaler.transform(vectors)
        alignment = standardized[:, None, :] * self.edge_matrix[None, :, :]
        matches = np.maximum(alignment, 0.0)
        edge_counts = np.maximum(np.count_nonzero(self.edge_matrix, axis=1), 1)
        return np.sum(matches, axis=2) / np.sqrt(edge_counts[None, :])

    def semantic_root_scores(
        self,
        row_count: int,
        semantic_observations: Optional[Sequence[Sequence[Any]]],
    ) -> np.ndarray:
        scores = np.zeros((int(row_count), len(self.labels)), dtype=np.float64)
        if semantic_observations is None:
            return scores
        observations = tuple(tuple(row) for row in semantic_observations)
        if len(observations) != int(row_count):
            raise ValueError("EPDG semantic observations and vectors differ in length")
        for row_index, row in enumerate(observations):
            observed = {
                str(atom)
                for observation in row
                for atom in getattr(observation, "atoms", ())
                if str(atom)
            }
            for root_index, root in enumerate(self.labels):
                scores[row_index, root_index] += float(
                    bool(observed & self.semantic_path_edges.get(root, frozenset()))
                )
        return scores

    def root_scores(
        self,
        vectors: np.ndarray,
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
        *,
        metric_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> np.ndarray:
        if metric_weight < 0.0 or semantic_weight < 0.0:
            raise ValueError("EPDG score weights must be nonnegative")
        metric = self.metric_root_scores(vectors)
        semantic = self.semantic_root_scores(len(metric), semantic_observations)
        return float(metric_weight) * metric + float(semantic_weight) * semantic

    def candidate_probabilities(
        self,
        vectors: np.ndarray,
        candidates: Sequence[Tuple[int, ...]],
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
        *,
        metric_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> np.ndarray:
        scores = self.root_scores(
            vectors,
            semantic_observations,
            metric_weight=metric_weight,
            semantic_weight=semantic_weight,
        )
        if not candidates:
            raise ValueError("EPDG candidate inventory must be nonempty")
        candidate_scores = np.column_stack(
            [np.sum(scores[:, list(candidate)], axis=1) for candidate in candidates]
        )
        if not self.active_edge_count and not self.semantic_edge_count:
            return np.full_like(candidate_scores, 1.0 / float(len(candidates)))
        centered = candidate_scores - np.max(candidate_scores, axis=1, keepdims=True)
        probabilities = np.exp(np.clip(centered, -50.0, 50.0))
        return probabilities / np.maximum(np.sum(probabilities, axis=1, keepdims=True), 1e-12)

    def metadata(self) -> Dict[str, Any]:
        if self.edge_matrix is None:
            raise RuntimeError("EPDGPathPrior must be fit before metadata")
        return {
            "graph": "registered_epdg_root_to_available_metric_paths",
            "feature_names": list(self.feature_names),
            "root_count": len(self.labels),
            "active_edge_count": self.active_edge_count,
            "registered_edge_count": self.registered_edge_count,
            "learned_edge_count": self.learned_edge_count,
            "semantic_edge_count": self.semantic_edge_count,
            "minimum_path_strength": self.minimum_path_strength,
            "minimum_direction_stability": self.minimum_direction_stability,
            "unavailable_path_types": ["sql_template", "execution_plan", "execution_operator"],
            "query_changes_graph": False,
        }


class MetricSetDecoder:
    """Nonlinear training-fold candidate-set decoder used by the H model."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        decoder_criterion: str = "gini",
        decoder_n_estimators: int = DEFAULT_DECODER_N_ESTIMATORS,
    ) -> None:
        if decoder_criterion not in {"gini", "entropy"}:
            raise ValueError("decoder_criterion must be 'gini' or 'entropy'")
        if int(decoder_n_estimators) < 1:
            raise ValueError("decoder_n_estimators must be positive")
        self.seed = int(seed)
        self.decoder_criterion = decoder_criterion
        self.decoder_n_estimators = int(decoder_n_estimators)
        self.candidates: List[Tuple[int, ...]] = []
        self.model: Any = None
        self.constant_target: Optional[int] = None
        self.training_metadata: Dict[str, Any] = {}

    def _new_model(self) -> Any:
        return RandomForestClassifier(
            n_estimators=self.decoder_n_estimators,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            criterion=self.decoder_criterion,
            random_state=self.seed,
            n_jobs=-1,
        )

    def _fit_target_model(self, vectors: np.ndarray, targets: np.ndarray) -> None:
        self.constant_target = None
        if len(self.candidates) == 1:
            self.constant_target = 0
            self.model = None
            return
        self.model = self._new_model()
        self.model.fit(vectors, targets)

    def fit(
        self,
        vectors: np.ndarray,
        train_labels: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> "MetricSetDecoder":
        vectors = _as_2d(vectors)
        self.candidates = _candidate_sets(train_labels, labels)
        self._fit_target_model(vectors, _candidate_targets(train_labels, labels, self.candidates))
        self.training_metadata = {
            "base_feature_count": int(vectors.shape[1]),
            "candidate_set_count": len(self.candidates),
            "decoder_criterion": self.decoder_criterion,
            "decoder_n_estimators": self.decoder_n_estimators,
            "uses_h_diffusion": False,
        }
        return self

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        vectors = _as_2d(vectors)
        if self.constant_target is not None:
            result = np.zeros((len(vectors), 1), dtype=np.float64)
            result[:, self.constant_target] = 1.0
            return result
        if self.model is None:
            raise RuntimeError("MetricSetDecoder must be fit before predict")
        return self._model_probabilities(self.model, vectors)

    def _model_probabilities(self, model: Any, vectors: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(vectors), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        expected = np.arange(len(self.candidates), dtype=np.int64)
        if not np.array_equal(classes, expected):
            raise ValueError("training candidate inventory is incomplete")
        return probabilities

    def predict(self, vectors: np.ndarray, labels: Sequence[str]) -> List[List[str]]:
        return _sets_from_candidate_indices(
            np.argmax(self.predict_proba(vectors), axis=1), labels, self.candidates
        )


class OrdinaryBinaryGraphClassifier:
    """A conventional pairwise-graph, independent multi-root baseline.

    This is intentionally a different output model from ``MetricSetDecoder``:
    each root is a separate binary classifier and predictions are assembled
    with one fixed threshold.  It has no candidate inventory, set argmax,
    fold-level residual stack, or hypergraph-specific state.  The feature for
    each metric is a single normalized aggregation over its pairwise-KPI
    neighbors in the current training fold.
    """

    def __init__(
        self,
        neighbor_count: int = 3,
        seed: int = DEFAULT_SEED,
        n_estimators: int = DEFAULT_BINARY_GRAPH_N_ESTIMATORS,
        max_depth: int = DEFAULT_BINARY_GRAPH_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_BINARY_GRAPH_MIN_SAMPLES_LEAF,
        threshold: float = DEFAULT_BINARY_GRAPH_THRESHOLD,
    ) -> None:
        if int(neighbor_count) < 1:
            raise ValueError("neighbor_count must be positive")
        if int(n_estimators) < 1 or int(max_depth) < 1 or int(min_samples_leaf) < 1:
            raise ValueError("ordinary graph RF settings are invalid")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("ordinary graph threshold must lie strictly between zero and one")
        self.neighbor_count = int(neighbor_count)
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.labels: Tuple[str, ...] = ()
        self.encoder: Optional[BinaryMetricGraphResidualEncoder] = None
        self.models: Dict[str, Any] = {}
        self.priors: Dict[str, float] = {}
        self.threshold = float(threshold)
        self.training_metadata: Dict[str, Any] = {}

    @staticmethod
    def _targets(label_rows: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [[int(label in row) for label in labels] for row in label_rows], dtype=np.int64
        )

    @staticmethod
    def _positive_probability(model: Any, vectors: np.ndarray, prior: float) -> np.ndarray:
        if model is None:
            return np.full(len(vectors), float(prior), dtype=np.float64)
        probabilities = np.asarray(model.predict_proba(vectors), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        if 1 not in classes:
            return np.zeros(len(vectors), dtype=np.float64)
        return probabilities[:, int(np.flatnonzero(classes == 1)[0])]

    def _new_model(self, root_index: int) -> Any:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            criterion="gini",
            random_state=self.seed + int(root_index),
            n_jobs=-1,
        )

    def _fit_root_models(
        self,
        features: np.ndarray,
        label_rows: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        targets = self._targets(label_rows, labels)
        models: Dict[str, Any] = {}
        priors: Dict[str, float] = {}
        for root_index, label in enumerate(labels):
            target = targets[:, root_index]
            prior = float(np.mean(target)) if len(target) else 0.0
            priors[label] = prior
            if len(np.unique(target)) < 2:
                models[label] = None
                continue
            model = self._new_model(root_index)
            model.fit(features, target)
            models[label] = model
        return models, priors

    def _predict_matrix(
        self,
        models: Mapping[str, Any],
        priors: Mapping[str, float],
        features: np.ndarray,
        labels: Sequence[str],
    ) -> np.ndarray:
        return np.column_stack(
            [
                self._positive_probability(models.get(label), features, float(priors.get(label, 0.0)))
                for label in labels
            ]
        )

    @staticmethod
    def _assemble(
        probabilities: np.ndarray, labels: Sequence[str], threshold: float
    ) -> List[List[str]]:
        predictions: List[List[str]] = []
        for row in probabilities:
            selected = [label for label, value in zip(labels, row) if float(value) >= threshold]
            if not selected:
                selected = [labels[int(np.argmax(row))]]
            selected.sort(key=lambda label: float(row[list(labels).index(label)]), reverse=True)
            predictions.append(selected)
        return predictions

    def fit(
        self,
        vectors: np.ndarray,
        train_labels: Sequence[Sequence[str]],
        labels: Sequence[str],
        oof_groups: Optional[Sequence[int]] = None,
    ) -> "OrdinaryBinaryGraphClassifier":
        vectors = _as_2d(vectors)
        if len(vectors) != len(train_labels):
            raise ValueError("ordinary graph vectors and labels differ in length")
        if oof_groups is not None and len(vectors) != len(oof_groups):
            raise ValueError("ordinary graph vectors and groups differ in length")
        self.labels = tuple(labels)
        self.encoder = BinaryMetricGraphResidualEncoder(self.neighbor_count).fit(vectors)
        self.models, self.priors = self._fit_root_models(
            self.encoder.aggregate(vectors), train_labels, self.labels
        )
        self.training_metadata = {
            "model_family": "independent_rootwise_binary_random_forests",
            "base_feature_count": int(vectors.shape[1]),
            "pairwise_graph_feature_count": int(vectors.shape[1]),
            "combined_feature_count": int(vectors.shape[1]),
            "candidate_inventory": "none",
            "candidate_set_count": 0,
            "set_decoder": "none; independent root probabilities plus threshold",
            "shares_fitted_state_with_hypergraph": False,
            "uses_hypergraph_specific_parameters": False,
            "parameter_selection": "fixed_shared_baseline_defaults",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "threshold": self.threshold,
            "encoder": self.encoder.metadata(),
        }
        return self

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        if self.encoder is None or not self.labels:
            raise RuntimeError("OrdinaryBinaryGraphClassifier must be fit before predict")
        vectors = _as_2d(vectors)
        features = self.encoder.aggregate(vectors)
        return self._predict_matrix(self.models, self.priors, features, self.labels)

    def predict(self, vectors: np.ndarray, labels: Optional[Sequence[str]] = None) -> List[List[str]]:
        requested = tuple(labels) if labels is not None else self.labels
        if requested != self.labels:
            raise ValueError("prediction labels differ from the labels used during training")
        return self._assemble(self.predict_proba(vectors), self.labels, self.threshold)


class LightweightHDiffusionClassifier(MetricSetDecoder):
    """Joint root-set decoder with fold-safe raw, H, and EPDG fusion.

    All weights are calibrated on caller-declared inner OOF folds without
    seeing the outer evaluation block. EPDG is disabled when the dataset has
    no registered root-metric or direct anonymous semantic path.
    """

    def __init__(
        self,
        activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
        propagation_steps: int = DEFAULT_PROPAGATION_STEPS,
        seed: int = DEFAULT_SEED,
        decoder_criterion: str = "gini",
        decoder_n_estimators: int = DEFAULT_DECODER_N_ESTIMATORS,
        adaptive_fusion: bool = True,
        fixed_fusion_weight: Optional[float] = None,
        fusion_weight_grid: Sequence[float] = tuple(
            round(value, 2) for value in np.arange(0.0, 1.01, 0.1)
        ),
        candidate_pool_max_size: int = DEFAULT_CANDIDATE_POOL_MAX_SIZE,
        candidate_pool_target_recall: float = DEFAULT_CANDIDATE_POOL_TARGET_RECALL,
        epdg_path_edges: Optional[Mapping[str, Mapping[str, float]]] = None,
        epdg_feature_names: Optional[Sequence[str]] = None,
        epdg_semantic_path_edges: Optional[Mapping[str, Sequence[str]]] = None,
        epdg_weight_grid: Sequence[float] = tuple(
            round(value, 2) for value in np.arange(0.0, 0.51, 0.1)
        ),
        enable_epdg: bool = False,
        epdg_min_oof_exact_gain: float = DEFAULT_EPDG_MIN_OOF_EXACT_GAIN,
    ) -> None:
        super().__init__(
            seed=seed,
            decoder_criterion=decoder_criterion,
            decoder_n_estimators=decoder_n_estimators,
        )
        self.activation_threshold = float(activation_threshold)
        self.propagation_steps = int(propagation_steps)
        self.encoder: Optional[HypergraphDiffusionResidualEncoder] = None
        self.base_model: Any = None
        self.adaptive_fusion = bool(adaptive_fusion)
        self.fixed_fusion_weight = (
            None if fixed_fusion_weight is None else float(fixed_fusion_weight)
        )
        if self.fixed_fusion_weight is not None and not 0.0 <= self.fixed_fusion_weight <= 1.0:
            raise ValueError("fixed fusion weight must lie between zero and one")
        self.fusion_weight_grid = tuple(float(value) for value in fusion_weight_grid)
        if any(not 0.0 <= value <= 1.0 for value in self.fusion_weight_grid):
            raise ValueError("fusion weights must lie between zero and one")
        if not self.fusion_weight_grid:
            raise ValueError("fusion weight grid must not be empty")
        if int(candidate_pool_max_size) < 1:
            raise ValueError("candidate pool maximum size must be positive")
        if not 0.0 < float(candidate_pool_target_recall) <= 1.0:
            raise ValueError("candidate pool target recall must lie in (0, 1]")
        self.candidate_pool_max_size = int(candidate_pool_max_size)
        self.candidate_pool_target_recall = float(candidate_pool_target_recall)
        self.epdg_path_edges = {
            str(root): {str(feature): float(weight) for feature, weight in edges.items()}
            for root, edges in (epdg_path_edges or {}).items()
        }
        self.epdg_feature_names = tuple(str(name) for name in (epdg_feature_names or ()))
        self.epdg_semantic_path_edges = {
            str(root): tuple(str(atom) for atom in atoms)
            for root, atoms in (epdg_semantic_path_edges or {}).items()
        }
        self.epdg_weight_grid = tuple(float(value) for value in epdg_weight_grid)
        if any(not 0.0 <= value <= 1.0 for value in self.epdg_weight_grid):
            raise ValueError("EPDG weights must lie between zero and one")
        if not self.epdg_weight_grid:
            raise ValueError("EPDG weight grid must not be empty")
        self.enable_epdg = bool(enable_epdg)
        if epdg_min_oof_exact_gain < 0.0:
            raise ValueError("EPDG minimum OOF Exact gain must be nonnegative")
        self.epdg_min_oof_exact_gain = float(epdg_min_oof_exact_gain)
        self.epdg_prior: Optional[EPDGPathPrior] = None
        self.epdg_weight = 0.0
        self.epdg_selection: Dict[str, Any] = {}
        self.candidate_pool_size = 1
        self.fusion_weight = 0.5
        self.fusion_selection: Dict[str, Any] = {}

    def _oof_residuals(
        self,
        vectors: np.ndarray,
        groups: Sequence[int],
        train_labels: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> np.ndarray:
        groups = list(groups)
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            raise ValueError("H-diffusion training requires at least two OOF groups")
        result: Optional[np.ndarray] = None
        for held_out in unique_groups:
            validation = np.asarray([group == held_out for group in groups], dtype=bool)
            encoder = HypergraphDiffusionResidualEncoder(
                activation_threshold=self.activation_threshold,
                propagation_steps=self.propagation_steps,
            )
            inner_weights = self._root_aware_metric_weights(
                vectors[~validation],
                [row for row, keep in zip(train_labels, validation) if not keep],
                labels,
                [group for group, keep in zip(groups, validation) if not keep],
            )
            encoder.fit(
                vectors[~validation],
                None if np.allclose(inner_weights, 1.0) else inner_weights,
            )
            transformed = encoder.transform(vectors[validation])
            if result is None:
                result = np.zeros((len(vectors), transformed.shape[1]), dtype=np.float64)
            result[validation] = transformed
        if result is None:
            raise RuntimeError("H-diffusion cross-fit did not produce residual features")
        return result

    def _root_aware_metric_weights(
        self,
        vectors: np.ndarray,
        train_labels: Sequence[Sequence[str]],
        labels: Sequence[str],
        groups: Sequence[int],
    ) -> np.ndarray:
        if not self.enable_epdg or not any(self.epdg_path_edges.get(label) for label in labels):
            return np.ones_like(vectors, dtype=np.float64)
        prior = EPDGPathPrior(
            self.epdg_feature_names,
            self.epdg_path_edges,
            self.epdg_semantic_path_edges,
        )
        if len(set(groups)) < 2:
            # The deepest residual cross-fit can leave one source group. In
            # that fold, retain registered paths but do not learn new paths.
            prior.fit(vectors, labels)
        else:
            prior.fit(vectors, labels, train_labels, groups)
        if prior.edge_matrix is None:
            raise RuntimeError("EPDG path prior did not expose learned metric edges")
        root_index = {label: index for index, label in enumerate(labels)}
        weights = np.ones_like(vectors, dtype=np.float64)
        for row_index, roots in enumerate(train_labels):
            path_strength = np.mean(
                np.abs(prior.edge_matrix[[root_index[root] for root in roots]]), axis=0
            )
            maximum = float(np.max(path_strength))
            if maximum > 0.0:
                weights[row_index] += path_strength / maximum
        return weights

    @staticmethod
    def _align_candidate_probabilities(
        probabilities: np.ndarray,
        source_candidates: Sequence[Tuple[int, ...]],
        target_candidates: Sequence[Tuple[int, ...]],
    ) -> np.ndarray:
        aligned = np.zeros((len(probabilities), len(target_candidates)), dtype=np.float64)
        target_index = {candidate: index for index, candidate in enumerate(target_candidates)}
        for source_index, candidate in enumerate(source_candidates):
            target_index_value = target_index.get(candidate)
            if target_index_value is not None:
                aligned[:, target_index_value] = probabilities[:, source_index]
        row_sums = np.sum(aligned, axis=1, keepdims=True)
        missing = row_sums[:, 0] <= 0.0
        if np.any(missing):
            aligned[missing] = 1.0 / float(len(target_candidates))
            row_sums = np.sum(aligned, axis=1, keepdims=True)
        return aligned / np.maximum(row_sums, 1e-12)

    def _select_fusion_weight(
        self,
        vectors: np.ndarray,
        train_labels: Sequence[Sequence[str]],
        labels: Sequence[str],
        groups: Sequence[int],
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Select raw/H and EPDG weights using the same grouped inner OOF rows."""
        group_values = sorted(set(groups))
        # A nested split with fewer than four source groups is too unstable to
        # estimate a branch weight; the fixed neutral prior is safer there.
        if len(group_values) < 4:
            return 0.5, 0.0, {
                "status": "fallback_insufficient_inner_groups",
                "group_count": len(group_values),
                "selected_raw_weight": 0.5,
                "selected_epdg_weight": 0.0,
                "candidate_pool_selection_status": "fallback_insufficient_inner_groups",
                "selected_candidate_pool_size": min(
                    self.candidate_pool_max_size, len(self.candidates)
                ),
            }
        fold_count = min(3, len(group_values))
        validation_group_sets = [
            set(group_values[index::fold_count]) for index in range(fold_count)
        ]
        raw_rows: List[np.ndarray] = []
        h_rows: List[np.ndarray] = []
        epdg_rows: List[np.ndarray] = []
        expected: List[Sequence[str]] = []
        epdg_grid = self.epdg_weight_grid if self.enable_epdg else (0.0,)
        semantic_rows = tuple(tuple(row) for row in (semantic_observations or ()))
        if semantic_rows and len(semantic_rows) != len(vectors):
            raise ValueError("EPDG semantic training rows and vectors differ in length")
        for inner_index, validation_groups in enumerate(validation_group_sets):
            validation = np.asarray(
                [group in validation_groups for group in groups], dtype=bool
            )
            inner_model = LightweightHDiffusionClassifier(
                activation_threshold=self.activation_threshold,
                propagation_steps=self.propagation_steps,
                seed=self.seed + inner_index + 1,
                decoder_criterion=self.decoder_criterion,
                decoder_n_estimators=self.decoder_n_estimators,
                adaptive_fusion=False,
                fusion_weight_grid=self.fusion_weight_grid,
                candidate_pool_max_size=self.candidate_pool_max_size,
                candidate_pool_target_recall=self.candidate_pool_target_recall,
                epdg_path_edges=self.epdg_path_edges,
                epdg_feature_names=self.epdg_feature_names,
                epdg_semantic_path_edges=self.epdg_semantic_path_edges,
                enable_epdg=self.enable_epdg,
            )
            inner_train_labels = [
                row for row, keep in zip(train_labels, validation) if not keep
            ]
            inner_groups = [group for group, keep in zip(groups, validation) if not keep]
            inner_model.fit(
                vectors[~validation],
                inner_train_labels,
                labels,
                inner_groups,
            )
            branch = inner_model.predict_branch_probabilities(vectors[validation])
            raw_rows.append(
                self._align_candidate_probabilities(
                    branch["raw"], inner_model.candidates, self.candidates
                )
            )
            h_rows.append(
                self._align_candidate_probabilities(
                    branch["h_residual"], inner_model.candidates, self.candidates
                )
            )
            if self.enable_epdg:
                epdg_rows.append(
                    EPDGPathPrior(
                        self.epdg_feature_names,
                        self.epdg_path_edges,
                        self.epdg_semantic_path_edges,
                    )
                    .fit(
                        vectors[~validation],
                        labels,
                        [row for row, keep in zip(train_labels, validation) if not keep],
                        [group for group, keep in zip(groups, validation) if not keep],
                    )
                    .candidate_probabilities(
                        vectors[validation],
                        self.candidates,
                        [row for row, keep in zip(semantic_rows, validation) if keep]
                        if semantic_rows
                        else None,
                        metric_weight=0.0,
                        semantic_weight=1.0,
                    )
                )
            expected.extend(row for row, keep in zip(train_labels, validation) if keep)
        raw_probabilities = np.vstack(raw_rows)
        h_probabilities = np.vstack(h_rows)
        epdg_probabilities = (
            np.vstack(epdg_rows)
            if epdg_rows
            else np.full_like(raw_probabilities, 1.0 / float(len(self.candidates)))
        )
        scores: Dict[Tuple[float, float], float] = {}
        for raw_weight in self.fusion_weight_grid:
            metric_probabilities = (
                raw_weight * raw_probabilities + (1.0 - raw_weight) * h_probabilities
            )
            for epdg_weight in epdg_grid:
                fused = (
                    (1.0 - epdg_weight) * metric_probabilities
                    + epdg_weight * epdg_probabilities
                )
                predictions = _sets_from_candidate_indices(
                    np.argmax(fused, axis=1), labels, self.candidates
                )
                scores[(raw_weight, epdg_weight)] = _exact(expected, predictions)
        selected_raw_weight, selected_epdg_weight = max(
            scores,
            key=lambda weights: (
                scores[weights],
                -weights[1],
                -abs(weights[0] - 0.5),
            ),
        )
        best_without_epdg = max(
            (score for (raw_weight, epdg_weight), score in scores.items() if epdg_weight == 0.0),
            default=0.0,
        )
        selected_gain = scores[(selected_raw_weight, selected_epdg_weight)] - best_without_epdg
        if selected_epdg_weight > 0.0 and selected_gain < self.epdg_min_oof_exact_gain:
            selected_raw_weight, selected_epdg_weight = max(
                (weights for weights in scores if weights[1] == 0.0),
                key=lambda weights: (scores[weights], -abs(weights[0] - 0.5)),
            )
            selected_gain = 0.0
        metric_probabilities = (
            selected_raw_weight * raw_probabilities
            + (1.0 - selected_raw_weight) * h_probabilities
        )
        fused = (
            (1.0 - selected_epdg_weight) * metric_probabilities
            + selected_epdg_weight * epdg_probabilities
        )
        target_indices = _candidate_targets(expected, labels, self.candidates)
        order = np.argsort(-fused, axis=1, kind="stable")
        maximum_pool_size = min(self.candidate_pool_max_size, len(self.candidates))
        pool_recall = {
            size: float(
                np.mean(
                    [
                        int(target in order[row, :size])
                        for row, target in enumerate(target_indices)
                    ]
                )
            )
            for size in range(1, maximum_pool_size + 1)
        }
        selected_pool_size = next(
            (
                size
                for size, recall in pool_recall.items()
                if recall >= self.candidate_pool_target_recall
            ),
            maximum_pool_size,
        )
        return float(selected_raw_weight), float(selected_epdg_weight), {
            "status": "selected_group_oof",
            "group_count": len(group_values),
            "selection_fold_count": fold_count,
            "objective": "candidate_set_exact_accuracy",
            "scores": {
                f"raw={raw_weight:.2f},epdg={epdg_weight:.2f}": score
                for (raw_weight, epdg_weight), score in scores.items()
            },
            "selected_raw_weight": float(selected_raw_weight),
            "selected_hypergraph_weight": float(1.0 - selected_raw_weight),
            "selected_epdg_weight": float(selected_epdg_weight),
            "epdg_minimum_oof_exact_gain": self.epdg_min_oof_exact_gain,
            "selected_epdg_oof_exact_gain": float(selected_gain),
            "epdg_status": "available" if self.enable_epdg else "disabled",
            "epdg_active_edge_count": int(
                EPDGPathPrior(
                    self.epdg_feature_names,
                    self.epdg_path_edges,
                    self.epdg_semantic_path_edges,
                )
                .fit(vectors, labels, train_labels, groups)
                .active_edge_count
                if self.enable_epdg
                else 0
            ),
            "candidate_pool_selection_status": "selected_group_oof",
            "candidate_pool_target_recall": self.candidate_pool_target_recall,
            "candidate_pool_oof_recall": {
                str(size): recall for size, recall in pool_recall.items()
            },
            "selected_candidate_pool_size": int(selected_pool_size),
        }

    def fit(
        self,
        vectors: np.ndarray,
        train_labels: Sequence[Sequence[str]],
        labels: Sequence[str],
        oof_groups: Sequence[int],
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
    ) -> "LightweightHDiffusionClassifier":
        vectors = _as_2d(vectors)
        if len(vectors) != len(oof_groups):
            raise ValueError("H-diffusion vectors and OOF groups differ in length")
        self.candidates = _candidate_sets(train_labels, labels)
        targets = _candidate_targets(train_labels, labels, self.candidates)
        if self.adaptive_fusion:
            (
                self.fusion_weight,
                self.epdg_weight,
                self.fusion_selection,
            ) = self._select_fusion_weight(
                vectors,
                train_labels,
                labels,
                oof_groups,
                semantic_observations,
            )
            self.candidate_pool_size = int(
                self.fusion_selection["selected_candidate_pool_size"]
            )
        elif self.fixed_fusion_weight is not None:
            self.fusion_weight = self.fixed_fusion_weight
            self.fusion_selection = {
                "status": "fixed_outer_training_weight",
                "selected_raw_weight": self.fusion_weight,
                "selected_epdg_weight": 0.0,
                "candidate_pool_selection_status": "fixed_for_nested_fit",
                "selected_candidate_pool_size": min(
                    self.candidate_pool_max_size, len(self.candidates)
                ),
            }
            self.candidate_pool_size = int(
                self.fusion_selection["selected_candidate_pool_size"]
            )
            self.epdg_weight = 0.0
        else:
            self.fusion_weight = 0.5
            self.fusion_selection = {
                "status": "disabled_for_nested_fit",
                "selected_raw_weight": 0.5,
                "selected_epdg_weight": 0.0,
                "candidate_pool_selection_status": "disabled_for_nested_fit",
                "selected_candidate_pool_size": min(
                    self.candidate_pool_max_size, len(self.candidates)
                ),
            }
            self.candidate_pool_size = int(
                self.fusion_selection["selected_candidate_pool_size"]
            )
            self.epdg_weight = 0.0
        self.epdg_selection = {
            "status": self.fusion_selection.get("epdg_status", "disabled_for_nested_fit"),
            "selected_epdg_weight": self.epdg_weight,
            "active_edge_count": self.fusion_selection.get("epdg_active_edge_count", 0),
        }
        residuals = self._oof_residuals(
            vectors, oof_groups, train_labels, labels
        )
        if len(self.candidates) == 1:
            self.base_model = None
        else:
            self.base_model = self._new_model()
            self.base_model.fit(vectors, targets)
        self._fit_target_model(np.concatenate([vectors, residuals], axis=1), targets)
        self.encoder = HypergraphDiffusionResidualEncoder(
            activation_threshold=self.activation_threshold,
            propagation_steps=self.propagation_steps,
        )
        training_weights = self._root_aware_metric_weights(
            vectors, train_labels, labels, oof_groups
        )
        self.encoder.fit(
            vectors,
            None if np.allclose(training_weights, 1.0) else training_weights,
        )
        if self.enable_epdg:
            self.epdg_prior = EPDGPathPrior(
                self.epdg_feature_names,
                self.epdg_path_edges,
                self.epdg_semantic_path_edges,
            ).fit(vectors, labels, train_labels, oof_groups)
            self.epdg_selection.update(
                {
                    "registered_metric_edge_count": self.epdg_prior.registered_edge_count,
                    "learned_metric_edge_count": self.epdg_prior.learned_edge_count,
                    "semantic_edge_count": self.epdg_prior.semantic_edge_count,
                }
            )
        self.training_metadata = {
            "base_feature_count": int(vectors.shape[1]),
            "h_diffusion_feature_count": int(residuals.shape[1]),
            "combined_feature_count": int(vectors.shape[1] + residuals.shape[1]),
            "candidate_set_count": len(self.candidates),
            "decoder_criterion": self.decoder_criterion,
            "decoder_n_estimators": self.decoder_n_estimators,
            "oof_group_field": "caller_provided_source_disjoint_group",
            "decoder_fusion": (
                "fixed_outer_training_probability_fusion"
                if self.fixed_fusion_weight is not None and not self.adaptive_fusion
                else "group_oof_adaptive_probability_fusion"
            ),
            "fusion_weight_raw": self.fusion_weight,
            "fusion_weight_hypergraph": 1.0 - self.fusion_weight,
            "fusion_selection": self.fusion_selection,
            "candidate_generation": "top_probability_root_sets_with_training_oof_selected_width",
            "candidate_pool_size": self.candidate_pool_size,
            "epdg_path_prior": self.epdg_selection,
            "epdg_weight": self.epdg_weight,
            "encoder": self.encoder.metadata(),
        }
        return self

    def predict_proba(
        self,
        vectors: np.ndarray,
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
    ) -> np.ndarray:
        return self.predict_branch_probabilities(vectors, semantic_observations)["fused"]

    def predict_candidate_pool_indices(
        self,
        vectors: np.ndarray,
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Return a broad root-set pool with width fixed by training OOF only."""

        probabilities = self.predict_proba(vectors, semantic_observations)
        width = min(self.candidate_pool_size, probabilities.shape[1])
        return np.argsort(-probabilities, axis=1, kind="stable")[:, :width]

    def predict_branch_probabilities(
        self,
        vectors: np.ndarray,
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return aligned raw, H-residual, and fused candidate posteriors."""
        if self.encoder is None:
            raise RuntimeError("LightweightHDiffusionClassifier must be fit before predict")
        vectors = _as_2d(vectors)
        h_probabilities = super().predict_proba(
            np.concatenate([vectors, self.encoder.transform(vectors)], axis=1)
        )
        if self.base_model is None:
            base_probabilities = h_probabilities.copy()
        else:
            base_probabilities = self._model_probabilities(self.base_model, vectors)
        fused = self.fusion_weight * base_probabilities + (
            1.0 - self.fusion_weight
        ) * h_probabilities
        epdg_probabilities = np.full_like(fused, 1.0 / float(len(self.candidates)))
        if self.epdg_prior is not None:
            epdg_probabilities = self.epdg_prior.candidate_probabilities(
                vectors,
                self.candidates,
                semantic_observations,
                metric_weight=0.0,
                semantic_weight=1.0,
            )
            fused = (1.0 - self.epdg_weight) * fused + self.epdg_weight * epdg_probabilities
        return {
            "raw": base_probabilities,
            "h_residual": h_probabilities,
            "epdg_prior": epdg_probabilities,
            "fused": fused,
        }
