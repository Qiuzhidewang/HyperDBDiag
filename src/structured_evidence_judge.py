"""Local evidence organization for broad HyperDBDiag candidate pools.

The hypergraph keeps several high-recall root-set candidates.  This module
does not train a second diagnosis model and does not memorize root-pair
failures.  It builds training-fold metric profiles, finds independent direct
evidence for roots that distinguish candidates, and contracts the broad pool
to a local fallback plus at most one evidence-supported LLM challenger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateEvidenceCard:
    """Compact, auditable evidence for one generated root-set candidate."""

    candidate_id: str
    root_labels: Tuple[str, ...]
    supporting_atoms: Tuple[str, ...]
    counterevidence_atoms: Tuple[str, ...]


@dataclass(frozen=True)
class StructuredCandidateConflict:
    """Local fallback plus one challenger retained from the broad pool."""

    local_candidate_id: str
    challenger_candidate_id: Optional[str]
    reviewable: bool
    status: str
    local_card: CandidateEvidenceCard
    challenger_card: Optional[CandidateEvidenceCard]


@dataclass(frozen=True)
class _ProfileState:
    mean: np.ndarray
    scale: np.ndarray
    roots: np.ndarray
    exact_sets: Mapping[Tuple[int, ...], np.ndarray]


@dataclass(frozen=True)
class _CandidateScore:
    candidate_index: int
    direct_hits: int
    direct_missing: int
    metric_alignment: float
    total: float


class StructuredEvidenceJudge:
    """Organize a broad H candidate pool without making a second diagnosis."""

    def __init__(self) -> None:
        self.root_labels: Tuple[str, ...] = ()
        self.feature_names: Tuple[str, ...] = ()
        self._profiles: Optional[_ProfileState] = None

    @staticmethod
    def _matrix(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            raise ValueError("metric values must be a nonempty two-dimensional matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("metric values must be finite")
        return matrix

    def _label_indexes(self, labels: Sequence[str]) -> Tuple[int, ...]:
        positions = {label: index for index, label in enumerate(self.root_labels)}
        if not labels or any(label not in positions for label in labels):
            raise ValueError("root set contains an unknown or empty label")
        result = tuple(sorted(positions[label] for label in labels))
        if len(result) != len(set(result)):
            raise ValueError("root set contains duplicate labels")
        return result

    def fit(
        self,
        train_x: np.ndarray,
        train_labels: Sequence[Sequence[str]],
        oof_groups: Sequence[Any],
        candidate_labels: Sequence[str],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "StructuredEvidenceJudge":
        train_x = self._matrix(train_x)
        if len(train_x) != len(train_labels) or len(train_x) != len(oof_groups):
            raise ValueError("structured evidence training inputs differ in length")
        self.root_labels = tuple(str(value) for value in candidate_labels)
        if not self.root_labels or len(set(self.root_labels)) != len(self.root_labels):
            raise ValueError("candidate labels must be nonempty and unique")
        self.feature_names = tuple(feature_names or (
            f"metric_{index:03d}" for index in range(train_x.shape[1])
        ))
        if len(self.feature_names) != train_x.shape[1]:
            raise ValueError("feature names must align to the metric matrix")

        groups = np.asarray(oof_groups, dtype=object)
        group_values = tuple(sorted(set(groups.tolist()), key=str))
        if len(group_values) < 2:
            raise ValueError("structured evidence requires at least two training groups")
        label_sets = tuple(self._label_indexes(row) for row in train_labels)
        mean = np.mean(train_x, axis=0)
        scale = np.maximum(np.std(train_x, axis=0), 1e-12)
        standardized = (train_x - mean) / scale

        def group_balanced_profile(present_mask: np.ndarray) -> np.ndarray:
            deltas: List[np.ndarray] = []
            for group in group_values:
                in_group = groups == group
                present = present_mask & in_group
                absent = ~present_mask & in_group
                if np.any(present) and np.any(absent):
                    deltas.append(
                        np.mean(standardized[present], axis=0)
                        - np.mean(standardized[absent], axis=0)
                    )
            return np.mean(deltas, axis=0) if deltas else np.zeros(train_x.shape[1])

        root_profiles = np.vstack(
            [
                group_balanced_profile(
                    np.asarray([root in row for row in label_sets], dtype=bool)
                )
                for root in range(len(self.root_labels))
            ]
        )
        exact_profiles = {
            candidate: group_balanced_profile(
                np.asarray([row == candidate for row in label_sets], dtype=bool)
            )
            for candidate in sorted(set(label_sets), key=lambda row: (len(row), row))
        }
        self._profiles = _ProfileState(mean, scale, root_profiles, exact_profiles)
        return self

    @staticmethod
    def _observed_atoms(observations: Sequence[Any]) -> frozenset[str]:
        atoms = {
            str(atom)
            for observation in observations
            for atom in getattr(observation, "atoms", ())
            if str(atom)
        }
        return frozenset(atoms)

    @staticmethod
    def _direct_atoms(
        label: str, mechanism_cards: Mapping[str, Mapping[str, Any]]
    ) -> frozenset[str]:
        card = mechanism_cards.get(label) or {}
        return frozenset(
            str(item.get("atom"))
            for item in card.get("semantic_observables") or ()
            if item.get("role") == "direct" and item.get("atom")
        )

    def _profile(self, candidate: Sequence[int]) -> np.ndarray:
        if self._profiles is None:
            raise RuntimeError("StructuredEvidenceJudge must be fit before prediction")
        key = tuple(int(index) for index in candidate)
        exact = self._profiles.exact_sets.get(key)
        return exact if exact is not None else np.mean(self._profiles.roots[list(key)], axis=0)

    def _metric_alignment(self, candidate: Sequence[int], query: np.ndarray) -> float:
        if self._profiles is None:
            raise RuntimeError("StructuredEvidenceJudge must be fit before prediction")
        query_z = (np.asarray(query, dtype=np.float64) - self._profiles.mean) / self._profiles.scale
        profile = self._profile(candidate)
        denominator = float(np.linalg.norm(query_z) * np.linalg.norm(profile))
        if denominator <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(query_z, profile) / denominator, -1.0, 1.0))

    def _semantic_counts(
        self,
        candidate: Sequence[int],
        observations: Sequence[Any],
        mechanism_cards: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[int, int]:
        observed = self._observed_atoms(observations)
        direct_hits = 0
        direct_missing = 0
        has_runtime_observations = bool(observations)
        for root in candidate:
            direct = self._direct_atoms(self.root_labels[int(root)], mechanism_cards)
            if direct & observed:
                direct_hits += 1
            elif has_runtime_observations and direct:
                direct_missing += 1
        return direct_hits, direct_missing

    def _score(
        self,
        candidate_index: int,
        candidate: Sequence[int],
        query: np.ndarray,
        observations: Sequence[Any],
        mechanism_cards: Mapping[str, Mapping[str, Any]],
    ) -> _CandidateScore:
        direct_hits, direct_missing = self._semantic_counts(
            candidate, observations, mechanism_cards
        )
        metric_alignment = self._metric_alignment(candidate, query)
        total = metric_alignment + 0.75 * float(direct_hits) - 0.10 * float(direct_missing)
        return _CandidateScore(
            candidate_index=int(candidate_index),
            direct_hits=int(direct_hits),
            direct_missing=int(direct_missing),
            metric_alignment=metric_alignment,
            total=float(total),
        )

    def _card(
        self,
        candidate_id: str,
        candidate: Sequence[int],
        query: np.ndarray,
        observations: Sequence[Any],
        mechanism_cards: Mapping[str, Mapping[str, Any]],
    ) -> CandidateEvidenceCard:
        if self._profiles is None:
            raise RuntimeError("StructuredEvidenceJudge must be fit before prediction")
        root_labels = tuple(self.root_labels[index] for index in candidate)
        observed = self._observed_atoms(observations)
        semantic_support: List[str] = []
        semantic_missing: List[str] = []
        for label in root_labels:
            direct = self._direct_atoms(label, mechanism_cards)
            matched = sorted(direct & observed)
            if matched:
                semantic_support.append(
                    f"{label}: current direct atom {', '.join(matched)}"
                )
            elif direct:
                semantic_missing.append(
                    f"{label}: no registered direct atom observed"
                )

        query_z = (np.asarray(query, dtype=np.float64) - self._profiles.mean) / self._profiles.scale
        alignment = query_z * self._profile(candidate)
        positive = np.flatnonzero(alignment > 0.0)
        negative = np.flatnonzero(alignment < 0.0)
        positive = positive[np.argsort(-alignment[positive], kind="stable")[:2]]
        negative = negative[np.argsort(alignment[negative], kind="stable")[:2]]
        metric_support = [
            f"{self.feature_names[index]} agrees with training profile"
            for index in positive
        ]
        metric_counter = [
            f"{self.feature_names[index]} opposes training profile"
            for index in negative
        ]
        return CandidateEvidenceCard(
            candidate_id=candidate_id,
            root_labels=root_labels,
            supporting_atoms=tuple((semantic_support + metric_support)[:4]),
            counterevidence_atoms=tuple((semantic_missing + metric_counter)[:4]),
        )

    def _challenger_choice(
        self,
        candidates: Sequence[Tuple[int, ...]],
        pool: Sequence[int],
        query: np.ndarray,
        observations: Sequence[Any],
        mechanism_cards: Mapping[str, Mapping[str, Any]],
    ) -> Optional[Tuple[int, str]]:
        if len(pool) < 2 or not mechanism_cards:
            return None
        local = self._score(
            int(pool[0]), candidates[int(pool[0])], query, observations, mechanism_cards
        )
        challengers: List[Tuple[_CandidateScore, int]] = []
        for rank, candidate_index in enumerate(pool[1:], start=1):
            score = self._score(
                int(candidate_index),
                candidates[int(candidate_index)],
                query,
                observations,
                mechanism_cards,
            )
            challengers.append((score, -rank))
        if not challengers:
            return None
        best, _ = max(
            challengers,
            key=lambda row: (
                row[0].direct_hits,
                row[0].total,
                row[0].metric_alignment,
                row[1],
            ),
        )
        if best.direct_hits > local.direct_hits and best.direct_hits > 0:
            return best.candidate_index, "direct_evidence_challenger"
        if best.total > local.total + 0.05 and best.metric_alignment > local.metric_alignment:
            return best.candidate_index, "profile_evidence_challenger"
        return None

    def propose(
        self,
        h_model: Any,
        eval_x: np.ndarray,
        candidate_ids: Sequence[str],
        semantic_observations: Optional[Sequence[Sequence[Any]]] = None,
        mechanism_cards: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> List[StructuredCandidateConflict]:
        if self._profiles is None:
            raise RuntimeError("StructuredEvidenceJudge must be fit before propose")
        eval_x = self._matrix(eval_x)
        candidates = tuple(tuple(int(value) for value in row) for row in h_model.candidates)
        identifiers = tuple(str(value) for value in candidate_ids)
        if len(candidates) != len(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate ids must align to the hypergraph inventory")
        try:
            pools = np.asarray(
                h_model.predict_candidate_pool_indices(eval_x, semantic_observations)
                if semantic_observations
                else h_model.predict_candidate_pool_indices(eval_x),
                dtype=np.int64,
            )
        except TypeError:
            pools = np.asarray(
                h_model.predict_candidate_pool_indices(eval_x), dtype=np.int64
            )
        if pools.ndim != 2 or len(pools) != len(eval_x):
            raise ValueError("hypergraph candidate pools are malformed")
        observations = list(semantic_observations or (() for _ in eval_x))
        if len(observations) != len(eval_x):
            raise ValueError("semantic observations and metric rows differ in length")
        cards = mechanism_cards or {}

        decisions: List[StructuredCandidateConflict] = []
        for query, pool, observed in zip(eval_x, pools, observations):
            local_index = int(pool[0])
            challenger_choice = self._challenger_choice(
                candidates, pool, query, observed, cards
            )
            challenger_index = None if challenger_choice is None else challenger_choice[0]
            status = "broad_pool_retained" if challenger_choice is None else challenger_choice[1]
            local_card = self._card(
                identifiers[local_index], candidates[local_index], query, observed, cards
            )
            challenger_card = (
                None
                if challenger_index is None
                else self._card(
                    identifiers[challenger_index],
                    candidates[challenger_index],
                    query,
                    observed,
                    cards,
                )
            )
            decisions.append(
                StructuredCandidateConflict(
                    local_candidate_id=identifiers[local_index],
                    challenger_candidate_id=(
                        None if challenger_index is None else identifiers[challenger_index]
                    ),
                    reviewable=challenger_index is not None,
                    status=status,
                    local_card=local_card,
                    challenger_card=challenger_card,
                )
            )
        return decisions

    def metadata(self) -> Dict[str, Any]:
        if self._profiles is None:
            raise RuntimeError("StructuredEvidenceJudge must be fit before metadata")
        return {
            "stage": "local_candidate_pool_contraction",
            "trained_second_diagnosis_model": False,
            "root_pair_failure_memory": False,
            "profile_source": "group-balanced outer-training metric contrasts",
            "routing_rule": (
                "retain the hypergraph top candidate as fallback and expose one broad-pool challenger "
                "when current direct evidence or training-profile alignment gives it a generic evidence advantage"
            ),
            "llm_handoff": "two compact evidence cards from the training-OOF-sized hypergraph pool",
            "uses_evaluation_labels": False,
        }
