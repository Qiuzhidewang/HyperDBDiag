"""Run the single six-row HyperDBDiag ablation on frozen DB-MAGS data.

The registered report has an ordinary binary graph, a higher-order hypergraph,
a local structured judge, direct hypergraph-to-LLM review, and the complete
HyperDBDiag pipeline. The LLM classifies ECSA candidate relations and emits a bounded modification
recommendation; it does not create labels or consume held-out identifiers, raw
SQL, plans, scores, distances, or evaluation labels. Anonymous query-shape
atoms are read from the integrity-bound frozen cohort. Invalid or unavailable
LLM output fails closed to the local hypergraph result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hyperdbdiag_pipeline import (
    CandidateBoundLLMReviewer,
    EvidenceItem,
    HyperDBDiagPipeline,
    LLMReviewEvidence,
)
from hyperdbdiag_llm import (
    BlindedRemediationQualityReviewer,
    PostDiagnosisLLMAdvisor,
    RemediationQualityResult,
    RemediationResult,
    llm_task_contract,
)
from hypergraph_core import (
    LightweightHDiffusionClassifier,
    OrdinaryBinaryGraphClassifier,
    _assert_candidate_coverage,
    _component_f1,
    _exact,
)
from metric_frozen_dataset import load_frozen_metric_dataset
from metric_frozen_schema import FEATURE_METRICS, RELATIVE_TIME_BINS_SECONDS
from sql_semantic_evidence import (
    SemanticObservation,
    load_frozen_case_observations,
    semantic_inventory,
)
from structured_evidence_judge import (
    CandidateEvidenceCard,
    StructuredCandidateConflict,
    StructuredEvidenceJudge,
)
DEFAULT_OUTPUT = Path("runs/dbmags-ablation/full_report.json")
DEFAULT_DBMAGS_ROOT = Path("data/dbmags_interaction_v10_metric_only")
DEFAULT_SEED = 20260802
ROOT_MECHANISM_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "root_mechanism_cards.json"
)
ROOT_PROFILE_ATOM_COUNT = 2
QUERY_ATOM_COUNT = 8
SEMANTIC_PROFILE_MIN_CONTRAST = 0.75
SEMANTIC_PROFILE_MIN_GROUP_STABILITY = 1.0


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return value


# The reviewer remains a bounded exception path. Evidence contraction now
# removes unsupported pairs before this budget is allocated.
LLM_REVIEW_BUDGET_PER_OUTER_SPLIT = _env_int(
    "HYPERDBDIAG_LLM_REVIEW_BUDGET_PER_OUTER_SPLIT", 12, minimum=0
)
LLM_REMEDIATION_BUDGET_PER_DATASET = _env_int(
    "HYPERDBDIAG_LLM_REMEDIATION_BUDGET_PER_DATASET", 24, minimum=0
)
LLM_MIN_OOF_DECISIONS = 8
LLM_MIN_OOF_ACTIVE_GROUPS = 3
LLM_MAX_OOF_SIGN_P_VALUE = 0.10
LLM_RELATION_MIN_OOF_DECISIONS = 4
LLM_RELATION_MIN_OOF_ACTIVE_GROUPS = 2
LLM_SELECTION_RELATIONS = (
    "COMPLEMENTARY",
    "COVERAGE",
    "REDUNDANT",
    "CONFLICT",
)
# The prompt is compact enough for high-reasoning review. One transport retry
# absorbs transient gateway failures without changing the decision protocol.
LLM_REQUEST_TIMEOUT_SECONDS = _env_float(
    "HYPERDBDIAG_LLM_REQUEST_TIMEOUT_SECONDS", 120.0, minimum=5.0
)
LLM_MAX_WORKERS = _env_int("HYPERDBDIAG_LLM_MAX_WORKERS", 2, minimum=1)
LLMClient = Callable[[Mapping[str, Any]], Union[Mapping[str, Any], str]]
_COMPLETED_LLM_REVIEW_STATUSES = frozenset(
    {
        "accepted_abstain",
        "accepted_candidate_selection",
        "audited_candidate_selection_without_override",
    }
)


@dataclass(frozen=True)
class AblationSplit:
    split_id: str
    train_ids: Tuple[str, ...]
    eval_ids: Tuple[str, ...]
    train_x: np.ndarray
    eval_x: np.ndarray
    train_labels: Tuple[Tuple[str, ...], ...]
    eval_labels: Tuple[Tuple[str, ...], ...]
    oof_groups: Tuple[int, ...]
    train_semantic: Tuple[Tuple[SemanticObservation, ...], ...] = ()
    eval_semantic: Tuple[Tuple[SemanticObservation, ...], ...] = ()

    def validate(self, feature_count: int, labels: Sequence[str]) -> None:
        if set(self.train_ids) & set(self.eval_ids):
            raise ValueError(f"{self.split_id} has overlapping train and evaluation IDs")
        if self.train_x.shape != (len(self.train_ids), feature_count):
            raise ValueError(f"{self.split_id} has an invalid training feature matrix")
        if self.eval_x.shape != (len(self.eval_ids), feature_count):
            raise ValueError(f"{self.split_id} has an invalid evaluation feature matrix")
        if len(self.train_labels) != len(self.train_ids) or len(self.eval_labels) != len(
            self.eval_ids
        ):
            raise ValueError(f"{self.split_id} labels and samples differ in length")
        if len(self.oof_groups) != len(self.train_ids) or len(set(self.oof_groups)) < 2:
            raise ValueError(f"{self.split_id} has invalid inner cross-fit groups")
        if self.train_semantic and len(self.train_semantic) != len(self.train_ids):
            raise ValueError(f"{self.split_id} semantic training rows differ in length")
        if self.eval_semantic and len(self.eval_semantic) != len(self.eval_ids):
            raise ValueError(f"{self.split_id} semantic evaluation rows differ in length")
        known = set(labels)
        if any(not row or not set(row) <= known for row in self.train_labels):
            raise ValueError(f"{self.split_id} has invalid training labels")
        if any(not row or not set(row) <= known for row in self.eval_labels):
            raise ValueError(f"{self.split_id} has invalid evaluation labels")


@dataclass(frozen=True)
class AblationDataset:
    name: str
    labels: Tuple[str, ...]
    feature_count: int
    splits: Tuple[AblationSplit, ...]
    activation_threshold: float
    decoder_criterion: str
    decoder_n_estimators: int
    metadata: Mapping[str, Any]
    training_selection: Mapping[str, Any]
    evidence_eligibility: Mapping[str, Any]
    feature_names: Tuple[str, ...] = ()
    mechanism_cards: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    epdg_path_edges: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    require_discriminating_semantic_for_llm: bool = False


@dataclass(frozen=True)
class _StructuredHandoff:
    """A broad H pool contracted to one fallback and at most one challenger."""

    hypergraph_indices: Tuple[int, ...]
    local_indices: Tuple[int, ...]
    alternative_indices: Tuple[Optional[int], ...]
    candidate_cards: Tuple[Tuple[CandidateEvidenceCard, ...], ...]
    conflicts: Tuple[StructuredCandidateConflict, ...]


def _limit_review_handoff(
    handoff: _StructuredHandoff,
    limit: int,
    candidate_probabilities: Optional[np.ndarray] = None,
) -> _StructuredHandoff:
    """Keep the most uncertain candidate pairs under a bounded call budget."""
    if int(limit) < 0:
        raise ValueError("review budget must be nonnegative")
    eligible = [
        index
        for index, candidate in enumerate(handoff.alternative_indices)
        if candidate is not None
    ]

    if candidate_probabilities is None:
        ordered = eligible
    else:
        probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(
            handoff.local_indices
        ):
            raise ValueError("candidate probabilities and handoff rows differ")
        if probabilities.shape[1] <= max(handoff.local_indices, default=-1):
            raise ValueError("candidate probabilities do not cover the handoff inventory")
        ordered = sorted(
            eligible,
            key=lambda index: (
                abs(
                    float(probabilities[index, handoff.local_indices[index]])
                    - float(probabilities[index, handoff.alternative_indices[index]])
                ),
                index,
            ),
        )
    selected = set(ordered[: int(limit)])
    if selected >= set(eligible):
        return handoff
    alternatives = tuple(
        value if index in selected else None
        for index, value in enumerate(handoff.alternative_indices)
    )
    cards = tuple(
        row if index in selected else ()
        for index, row in enumerate(handoff.candidate_cards)
    )
    return _StructuredHandoff(
        hypergraph_indices=handoff.hypergraph_indices,
        local_indices=handoff.local_indices,
        alternative_indices=alternatives,
        candidate_cards=cards,
        conflicts=handoff.conflicts,
    )


def _retain_semantic_challengers(
    dataset: AblationDataset,
    h_model: LightweightHDiffusionClassifier,
    handoff: _StructuredHandoff,
    semantic_observations: Sequence[Sequence[SemanticObservation]],
    train_labels: Sequence[Sequence[str]],
    train_groups: Sequence[Any],
    train_semantic_observations: Sequence[Sequence[SemanticObservation]],
) -> _StructuredHandoff:
    """Retain challengers with strong current evidence for a differing root."""

    if not dataset.require_discriminating_semantic_for_llm:
        return handoff
    observations = tuple(tuple(row) for row in semantic_observations)
    if len(observations) != len(handoff.local_indices):
        raise ValueError("semantic observations and structured handoff differ")
    associations = _strong_semantic_associations(
        train_labels,
        train_groups,
        dataset.labels,
        train_semantic_observations,
    )
    alternatives: List[Optional[int]] = []
    cards: List[Tuple[CandidateEvidenceCard, ...]] = []
    for row, local_index, alternative_index, candidate_cards in zip(
        observations,
        handoff.local_indices,
        handoff.alternative_indices,
        handoff.candidate_cards,
    ):
        if alternative_index is None:
            alternatives.append(None)
            cards.append(())
            continue
        local_roots = {
            dataset.labels[root] for root in h_model.candidates[local_index]
        }
        alternative_roots = {
            dataset.labels[root] for root in h_model.candidates[alternative_index]
        }
        distinguishing = local_roots ^ alternative_roots
        observed_atoms = {
            atom for observation in row for atom in observation.atoms
        }
        supported = {
            label
            for label in distinguishing
            if any(
                atom in observed_atoms
                for atom, _, _ in associations.get(label, ())
            )
        }
        supports_alternative = bool(supported & alternative_roots)
        supports_local = bool(supported & local_roots)
        retain = supports_alternative and not supports_local
        alternatives.append(alternative_index if retain else None)
        cards.append(candidate_cards if retain else ())
    return _StructuredHandoff(
        hypergraph_indices=handoff.hypergraph_indices,
        local_indices=handoff.local_indices,
        alternative_indices=tuple(alternatives),
        candidate_cards=tuple(cards),
        conflicts=handoff.conflicts,
    )


def _structured_handoff(
    h_model: LightweightHDiffusionClassifier,
    judge: StructuredEvidenceJudge,
    vectors: np.ndarray,
    candidate_ids: Sequence[str],
    hypergraph_indices: Optional[Sequence[int]] = None,
    semantic_observations: Optional[Sequence[Sequence[SemanticObservation]]] = None,
    mechanism_cards: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> _StructuredHandoff:
    """Translate a judge decision into the pipeline's local/alternative contract."""

    inventory_size = len(h_model.candidates)
    identifiers = tuple(str(value) for value in candidate_ids)
    if len(identifiers) != inventory_size or len(set(identifiers)) != inventory_size:
        raise ValueError("candidate ids must align to the hypergraph inventory")
    if hypergraph_indices is None:
        h_indices = tuple(
            int(index)
            for index in np.argmax(
                h_model.predict_proba(vectors, semantic_observations), axis=1
            )
        )
    else:
        h_indices = tuple(int(index) for index in hypergraph_indices)
    if len(h_indices) != len(vectors) or any(not 0 <= index < inventory_size for index in h_indices):
        raise ValueError("hypergraph prediction indices are malformed")
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(identifiers)}
    conflicts = tuple(
        judge.propose(
            h_model,
            vectors,
            identifiers,
            semantic_observations=semantic_observations,
            mechanism_cards=mechanism_cards,
        )
    )
    if len(conflicts) != len(vectors):
        raise RuntimeError("structured judge and metric observations differ")

    local_indices: List[int] = []
    alternative_indices: List[Optional[int]] = []
    cards: List[Tuple[CandidateEvidenceCard, ...]] = []
    for conflict in conflicts:
        local_indices.append(index_by_id[conflict.local_candidate_id])
        if conflict.reviewable and conflict.challenger_candidate_id is not None:
            if conflict.challenger_card is None:
                raise RuntimeError("reviewable local conflict has no challenger evidence card")
            alternative_indices.append(index_by_id[conflict.challenger_candidate_id])
            cards.append((conflict.local_card, conflict.challenger_card))
        else:
            alternative_indices.append(None)
            cards.append(())
    return _StructuredHandoff(
        hypergraph_indices=h_indices,
        local_indices=tuple(local_indices),
        alternative_indices=tuple(alternative_indices),
        candidate_cards=tuple(cards),
        conflicts=conflicts,
    )


def _direct_handoff(
    h_model: LightweightHDiffusionClassifier,
    vectors: np.ndarray,
    candidate_ids: Sequence[str],
    labels: Sequence[str],
    semantic_observations: Optional[Sequence[Sequence[SemanticObservation]]] = None,
) -> _StructuredHandoff:
    """Expose the top hypergraph candidate pair without local-judge decisions."""

    identifiers = tuple(str(value) for value in candidate_ids)
    inventory_size = len(h_model.candidates)
    if len(identifiers) != inventory_size or len(set(identifiers)) != inventory_size:
        raise ValueError("candidate ids must align to the hypergraph inventory")
    pools = np.asarray(
        h_model.predict_candidate_pool_indices(vectors, semantic_observations)
        if semantic_observations
        else h_model.predict_candidate_pool_indices(vectors),
        dtype=np.int64,
    )
    if pools.ndim != 2 or len(pools) != len(vectors) or not pools.shape[1]:
        raise ValueError("hypergraph candidate pools are malformed")
    if np.any(pools < 0) or np.any(pools >= inventory_size):
        raise ValueError("hypergraph candidate pools contain an invalid index")
    local_indices = tuple(int(row[0]) for row in pools)
    alternative_indices = tuple(
        int(row[1]) if pools.shape[1] > 1 else None for row in pools
    )
    cards: List[Tuple[CandidateEvidenceCard, ...]] = []
    for local_index, alternative_index in zip(local_indices, alternative_indices):
        row = [
            CandidateEvidenceCard(
                candidate_id=identifiers[index],
                root_labels=tuple(
                    labels[root] for root in h_model.candidates[index]
                ),
                supporting_atoms=(),
                counterevidence_atoms=(),
            )
            for index in (local_index, alternative_index)
            if index is not None
        ]
        cards.append(tuple(row))
    return _StructuredHandoff(
        hypergraph_indices=local_indices,
        local_indices=local_indices,
        alternative_indices=alternative_indices,
        candidate_cards=tuple(cards),
        conflicts=(),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / float(len(rows) or 1)


def _load_root_mechanism_cards(
    path: Path = ROOT_MECHANISM_CARD_PATH,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load the versioned generic mechanism registry used by the LLM only."""

    encoded = path.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    if set(payload) != {"schema_version", "cards"} or not isinstance(payload["cards"], list):
        raise ValueError("root mechanism card registry is malformed")
    cards: Dict[str, Dict[str, Any]] = {}
    for row in payload["cards"]:
        required_fields = {
            "root_label",
            "summary",
            "registered_metric_discriminability",
            "observables",
            "source_refs",
        }
        optional_fields = {"semantic_observables"}
        if set(row) - required_fields - optional_fields or not required_fields <= set(row):
            raise ValueError("a root mechanism card is malformed")
        label = str(row["root_label"])
        summary = str(row["summary"])
        discriminability = str(row["registered_metric_discriminability"])
        observables = row["observables"]
        source_refs = row["source_refs"]
        semantic_observables = row.get("semantic_observables") or ()
        if (
            not label
            or not summary
            or label in cards
            or discriminability not in {"partial", "none"}
            or not isinstance(observables, list)
            or not observables
            or not isinstance(source_refs, list)
            or not source_refs
            or not isinstance(semantic_observables, (list, tuple))
        ):
            raise ValueError("root mechanism cards must be unique and nonempty")
        normalized_observables = []
        for observable in observables:
            if set(observable) != {"metric_family", "role", "expected_direction"}:
                raise ValueError("a mechanism observable is malformed")
            family = str(observable["metric_family"])
            role = str(observable["role"])
            direction = str(observable["expected_direction"])
            if (
                family not in {"lock-wait", "lock-occupancy", "execution-pressure"}
                or role not in {"direct", "contextual", "compatible_only"}
                or direction not in {"elevated", "unspecified"}
            ):
                raise ValueError("a mechanism observable has an unsupported value")
            normalized_observables.append(
                {
                    "metric_family": family,
                    "role": role,
                    "expected_direction": direction,
                }
            )
        sources = tuple(str(value) for value in source_refs)
        if not all(sources):
            raise ValueError("mechanism card source references must be nonempty")
        normalized_semantic = []
        for observable in semantic_observables:
            if set(observable) != {"atom", "role"}:
                raise ValueError("a semantic mechanism observable is malformed")
            atom = str(observable["atom"])
            role = str(observable["role"])
            if not atom or role not in {"direct", "contextual"}:
                raise ValueError("a semantic mechanism observable has an unsupported value")
            normalized_semantic.append({"atom": atom, "role": role})
        cards[label] = {
            "summary": summary,
            "registered_metric_discriminability": discriminability,
            "observables": tuple(normalized_observables),
            "semantic_observables": tuple(normalized_semantic),
            "source_refs": sources,
        }
    if not cards:
        raise ValueError("root mechanism card registry is empty")
    return cards, {
        "schema_version": str(payload["schema_version"]),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "card_count": len(cards),
        "source": str(path.relative_to(path.parents[1])),
    }


def _cards_for_labels(
    cards: Mapping[str, Mapping[str, Any]], labels: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    missing = sorted(set(labels) - set(cards))
    if missing:
        raise ValueError(f"root mechanism cards are missing labels: {missing}")
    return {label: dict(cards[label]) for label in labels}


def _epdg_path_edges(
    feature_names: Sequence[str],
    labels: Sequence[str],
    mechanism_cards: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Materialize only registered root-to-metric EPDG edges available here."""

    family_to_features = {
        "lock-wait": {
            str(name): str(name).split("[", 1)[0]
            for name in feature_names
            if str(name).split("[", 1)[0]
            in {"mysql_data_lock_waits", "mysql_innodb_row_lock_current_waits"}
        },
        "lock-occupancy": {
            str(name): str(name).split("[", 1)[0]
            for name in feature_names
            if str(name).split("[", 1)[0] == "mysql_data_locks"
        },
    }
    edges: Dict[str, Dict[str, float]] = {}
    for label in labels:
        root_edges: Dict[str, float] = {}
        for observable in mechanism_cards.get(label, {}).get("observables", ()):
            family = str(observable.get("metric_family"))
            role = str(observable.get("role"))
            weight = 1.0 if role == "direct" else 0.25 if role == "contextual" else 0.0
            if weight <= 0.0:
                continue
            for feature in family_to_features.get(family, {}):
                root_edges[feature] = max(root_edges.get(feature, 0.0), weight)
        edges[label] = root_edges
    return edges


def _epdg_semantic_path_edges(
    labels: Sequence[str], mechanism_cards: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Tuple[str, ...]]:
    """Return registered direct root-to-anonymous-observation EPDG edges."""

    return {
        str(label): tuple(
            sorted(
                str(observable["atom"])
                for observable in mechanism_cards.get(label, {}).get(
                    "semantic_observables", ()
                )
                if observable.get("role") == "direct" and observable.get("atom")
            )
        )
        for label in labels
    }


def _dbmags_feature_names() -> Tuple[str, ...]:
    return tuple(
        f"{metric}[{start:g}-{end:g}s]"
        for start, end in RELATIVE_TIME_BINS_SECONDS
        for metric in FEATURE_METRICS
    )


_METRIC_SEMANTICS = {
    "cpu_user": "host CPU time spent in user processes; general execution pressure",
    "cpu_system": "host CPU time spent in kernel work; general system pressure",
    "cpu_idle": "idle CPU capacity; inverse context for aggregate CPU pressure",
    "cpu_io_wait": "CPU time waiting for I/O completion; storage-pressure context",
    "disk_read": "aggregate disk read activity; read-path pressure context",
    "disk_write": "aggregate disk write activity; write-path pressure context",
    "memory_used": "used host memory; broad capacity context",
    "memory_free": "free host memory; inverse broad capacity context",
    "memory_buffered": "buffer memory; filesystem-I/O context",
    "memory_cached": "cached host memory; filesystem-I/O context",
    "mysql_data_locks": (
        "current Performance Schema data-lock records; weak occupancy evidence, "
        "not proof of waiting or contention"
    ),
    "mysql_data_lock_waits": (
        "current data-lock wait relationships; direct contention/wait evidence"
    ),
    "mysql_innodb_row_lock_current_waits": (
        "current InnoDB row-lock waits; direct contention/wait evidence, correlated "
        "with data-lock wait relationships"
    ),
    "mysql_threads_running": (
        "currently executing non-sleeping MySQL threads; general workload/concurrency "
        "evidence, not proof of a particular root"
    ),
    "mysql_active_sessions": (
        "currently active non-sleeping sessions; general activity/concurrency evidence, "
        "not proof of a particular root"
    ),
}


def _metric_semantics(feature_names: Sequence[str]) -> str:
    """Return fixed metric meanings without adding a decision feature."""

    names = []
    for feature_name in feature_names:
        metric = str(feature_name).split("[", 1)[0]
        if metric in _METRIC_SEMANTICS and metric not in names:
            names.append(metric)
    if not names:
        return "No frozen metric dictionary is registered for these feature names."
    entries = "; ".join(f"{name}: {_METRIC_SEMANTICS[name]}" for name in names)
    return (
        "Frozen metric semantics (descriptive context only): "
        + entries
        + ". Standardized values are relative trajectories, not raw counts, scores, "
        "or probabilities. Correlated lock-wait metrics are one evidence family and "
        "must not be double-counted; missing a signal does not exclude a cause."
    )


def _strongest_atoms(values: np.ndarray, feature_names: Sequence[str], limit: int) -> List[str]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(feature_names),):
        raise ValueError("feature names and metric evidence width differ")
    if limit < 1:
        raise ValueError("metric evidence atom limit must be positive")
    order = np.argsort(-np.abs(values), kind="stable")[: min(limit, len(values))]
    return [f"{feature_names[index]}={values[index]:+.3f}" for index in order]


_TIME_BIN_FEATURE = re.compile(r"^(?P<metric>.+)\[(?P<start>[0-9.]+)-(?P<end>[0-9.]+)s\]$")


def _trajectory_summary(values: np.ndarray, feature_names: Sequence[str]) -> str:
    """Render all time-indexed KPI evidence without turning it into a score.

    Metric-only reviews need the shape of an anomaly, not just its largest
    individual deviations.  Feature names without the registered time-bin
    form retain the compact strongest-atom fallback.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(feature_names),):
        raise ValueError("feature names and metric evidence width differ")
    grouped: Dict[str, List[Tuple[float, float, float]]] = {}
    for feature_name, value in zip(feature_names, values):
        match = _TIME_BIN_FEATURE.fullmatch(str(feature_name))
        if match is None:
            return ", ".join(_strongest_atoms(values, feature_names, QUERY_ATOM_COUNT))
        grouped.setdefault(match.group("metric"), []).append(
            (float(match.group("start")), float(match.group("end")), float(value))
        )
    if not grouped or any(len(rows) < 2 for rows in grouped.values()):
        return ", ".join(_strongest_atoms(values, feature_names, QUERY_ATOM_COUNT))
    descriptions = []
    for metric, rows in grouped.items():
        bins = ", ".join(
            f"{start:g}-{end:g}s={value:+.3f}"
            for start, end, value in sorted(rows, key=lambda row: row[:2])
        )
        descriptions.append(f"{metric}: {bins}")
    return "; ".join(descriptions)


def _configured_llm_client() -> Tuple[Optional[LLMClient], Dict[str, Any]]:
    """Build an opt-in Responses client without storing credentials."""

    def configured_value(*names: str) -> str:
        return next((os.environ.get(name, "").strip() for name in names if os.environ.get(name, "").strip()), "")

    def normalized_base_url(value: str) -> str:
        """Prefer the OpenAI-compatible /v1 endpoint for host-only proxies."""

        text = str(value or "").strip().rstrip("/")
        if not text:
            return ""
        # The OpenAI SDK uses a caller-provided base URL literally. Most
        # OpenAI-compatible gateways expose Responses under /v1/responses; if a
        # host-only URL is supplied, route to /v1 by default while preserving
        # explicit paths supplied by the caller.
        if re.fullmatch(r"https?://[^/]+", text):
            return text + "/v1"
        return text

    api_key = configured_value("HYPERDBDIAG_LLM_API_KEY", "OPENAI_API_KEY")
    model = configured_value("HYPERDBDIAG_LLM_MODEL", "OPENAI_MODEL")
    raw_base_url = configured_value("HYPERDBDIAG_LLM_BASE_URL", "OPENAI_BASE_URL")
    base_url = normalized_base_url(raw_base_url)
    reasoning_effort = configured_value(
        "HYPERDBDIAG_LLM_REASONING_EFFORT", "OPENAI_REASONING_EFFORT"
    )
    metadata: Dict[str, Any] = {
        "configured": bool(api_key and model),
        "model": model or None,
        "base_url_configured": bool(base_url),
        "base_url_normalized_to_v1": bool(raw_base_url and raw_base_url.rstrip("/") != base_url),
        "api_style": "responses",
        "transport": "httpx_responses",
        "response_storage": False,
        "response_format": "strict_json_schema_validated_locally",
        "reasoning_effort": reasoning_effort or None,
        "request_timeout_seconds": LLM_REQUEST_TIMEOUT_SECONDS,
        "max_workers": LLM_MAX_WORKERS,
        "transport_attempts": 2,
        "configuration_source": "HYPERDBDIAG_LLM_* or OPENAI_* environment variables",
    }
    if not api_key or not model:
        return None, metadata
    try:
        import httpx
    except ImportError as exc:
        metadata["configuration_error"] = f"httpx_import_failed:{type(exc).__name__}"
        return None, metadata

    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/") + "/responses"
    timeout = httpx.Timeout(
        connect=min(10.0, float(LLM_REQUEST_TIMEOUT_SECONDS)),
        read=float(LLM_REQUEST_TIMEOUT_SECONDS),
        write=min(10.0, float(LLM_REQUEST_TIMEOUT_SECONDS)),
        pool=min(10.0, float(LLM_REQUEST_TIMEOUT_SECONDS)),
    )
    client = httpx.Client(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max(2, LLM_MAX_WORKERS * 2),
            max_keepalive_connections=max(1, LLM_MAX_WORKERS),
        ),
    )

    def response_output_text(response_payload: Mapping[str, Any]) -> str:
        direct = response_payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        fragments: List[str] = []
        for output in response_payload.get("output") or ():
            if not isinstance(output, Mapping):
                continue
            for content in output.get("content") or ():
                if not isinstance(content, Mapping):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text:
                    fragments.append(text)
        if fragments:
            return "".join(fragments)
        raise ValueError("LLM response has no text content")

    def invoke(payload: Mapping[str, Any]) -> str:
        contract = llm_task_contract(str(payload.get("task") or ""))
        request: Dict[str, Any] = {
            "model": model,
            "instructions": contract.instructions,
            "input": json.dumps(payload, ensure_ascii=True),
            "max_output_tokens": contract.max_output_tokens,
            "store": False,
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": contract.schema_name,
                    "strict": True,
                    "schema": contract.schema,
                },
            },
        }
        if reasoning_effort:
            request["reasoning"] = {"effort": reasoning_effort}
        for attempt in range(2):
            try:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request,
                )
                response.raise_for_status()
                return response_output_text(response.json())
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt:
                    raise
                time.sleep(0.5)
        raise RuntimeError("unreachable LLM transport state")

    return invoke, metadata


def _strong_semantic_associations(
    train_labels: Sequence[Sequence[str]],
    train_groups: Sequence[int],
    labels: Sequence[str],
    semantic_observations: Optional[Sequence[Sequence[SemanticObservation]]],
) -> Dict[str, Tuple[Tuple[str, float, float], ...]]:
    """Fit strong anonymous-atom associations using grouped training rows only."""

    semantic_rows = tuple(tuple(row) for row in (semantic_observations or ()))
    result = {str(label): tuple() for label in labels}
    if not semantic_rows:
        return result
    if len(semantic_rows) != len(train_labels) or len(semantic_rows) != len(train_groups):
        raise ValueError("training semantic observations and metric rows differ")
    semantic_atoms = semantic_inventory(
        observation for row in semantic_rows for observation in row
    )
    if not semantic_atoms:
        return result
    semantic_matrix = np.asarray(
        [
            [
                float(
                    atom
                    in {
                        current
                        for observation in row
                        for current in observation.atoms
                    }
                )
                for atom in semantic_atoms
            ]
            for row in semantic_rows
        ],
        dtype=np.float64,
    )
    label_sets = [set(row) for row in train_labels]
    for label in labels:
        semantic_deltas: List[np.ndarray] = []
        for group in sorted(set(train_groups)):
            present = np.asarray(
                [
                    current_group == group and label in row
                    for current_group, row in zip(train_groups, label_sets)
                ],
                dtype=bool,
            )
            absent = np.asarray(
                [
                    current_group == group and label not in row
                    for current_group, row in zip(train_groups, label_sets)
                ],
                dtype=bool,
            )
            if np.any(present) and np.any(absent):
                semantic_deltas.append(
                    np.mean(semantic_matrix[present], axis=0)
                    - np.mean(semantic_matrix[absent], axis=0)
                )
        if not semantic_deltas:
            continue
        values = np.vstack(semantic_deltas)
        average = np.mean(values, axis=0)
        stability = np.mean(values > 0.0, axis=0)
        strong = np.flatnonzero(
            (average >= SEMANTIC_PROFILE_MIN_CONTRAST)
            & (stability >= SEMANTIC_PROFILE_MIN_GROUP_STABILITY)
        )
        strong = strong[
            np.argsort(-(average[strong] * stability[strong]), kind="stable")[
                :ROOT_PROFILE_ATOM_COUNT
            ]
        ]
        result[str(label)] = tuple(
            (
                semantic_atoms[int(index)],
                float(average[int(index)]),
                float(stability[int(index)]),
            )
            for index in strong
        )
    return result


def _root_profile_items(
    train_x: np.ndarray,
    train_labels: Sequence[Sequence[str]],
    train_groups: Sequence[int],
    labels: Sequence[str],
    feature_names: Sequence[str],
    semantic_observations: Optional[
        Sequence[Sequence[SemanticObservation]]
    ] = None,
) -> Tuple[EvidenceItem, ...]:
    """Build group-balanced, training-only root profiles for the reviewer."""

    train_x = np.asarray(train_x, dtype=np.float64)
    if (
        train_x.ndim != 2
        or train_x.shape[0] != len(train_labels)
        or train_x.shape[0] != len(train_groups)
        or train_x.shape[1] != len(feature_names)
    ):
        raise ValueError("root profile inputs have incompatible dimensions")
    mean = np.mean(train_x, axis=0)
    scale = np.maximum(np.std(train_x, axis=0), 1e-12)
    standardized = (train_x - mean) / scale
    label_sets = [set(row) for row in train_labels]
    semantic_associations = _strong_semantic_associations(
        train_labels, train_groups, labels, semantic_observations
    )
    items: List[EvidenceItem] = []
    for label in labels:
        deltas: List[np.ndarray] = []
        for group in sorted(set(train_groups)):
            present = np.asarray(
                [current_group == group and label in row for current_group, row in zip(train_groups, label_sets)],
                dtype=bool,
            )
            absent = np.asarray(
                [current_group == group and label not in row for current_group, row in zip(train_groups, label_sets)],
                dtype=bool,
            )
            if np.any(present) and np.any(absent):
                deltas.append(
                    np.mean(standardized[present], axis=0) - np.mean(standardized[absent], axis=0)
                )
        if not deltas:
            summary = (
                f"Outer-training root profile for {label}: no within-group present-versus-absent "
                "contrast was available. This profile cannot support a revision."
            )
        else:
            matrix = np.vstack(deltas)
            average = np.mean(matrix, axis=0)
            signs = np.sign(average)
            stability = np.mean(np.sign(matrix) == signs[None, :], axis=0)
            atoms = []
            for feature_index in np.argsort(-np.abs(average), kind="stable")[:ROOT_PROFILE_ATOM_COUNT]:
                atoms.append(
                    f"{feature_names[int(feature_index)]}: "
                    f"present-minus-absent={average[int(feature_index)]:+.3f}, "
                    f"sign-stability={stability[int(feature_index)]:.2f}"
                )
            summary = (
                f"Outer-training group-balanced root profile for {label}; "
                f"{len(deltas)} groups contributed. These are descriptive associations, not a score "
                "or a selection instruction. "
                + "; ".join(atoms)
                + "."
            )
        if semantic_associations[str(label)]:
            semantic_summary = "; ".join(
                f"{atom}: present-minus-absent={contrast:+.3f}, "
                f"positive-group-stability={stability:.2f}"
                for atom, contrast, stability in semantic_associations[str(label)]
            )
            summary += (
                " Outer-training anonymous SQL/plan associations: "
                + semantic_summary
                + ". Only strong associations reproduced in every training group are shown; "
                "they are evidence context, not a decision rule."
            )
        items.append(
            EvidenceItem(
                evidence_id=f"training-profile:{label}",
                kind="training_root_profile",
                summary=summary,
                root_labels=(label,),
            )
        )
    return tuple(items)


def _mechanism_evidence_summary(label: str, card: Mapping[str, Any]) -> str:
    """Render the compact frozen mechanism facts needed for arbitration."""

    observables = tuple(card.get("observables") or ())
    if not observables:
        return f"{label}: {card['summary']}"
    roles = "; ".join(
        f"{item['metric_family']} is {item['role']} evidence, expected direction {item['expected_direction']}"
        for item in observables
    )
    semantic = tuple(card.get("semantic_observables") or ())
    semantic_text = (
        " Semantic query-shape roles: "
        + "; ".join(f"{item['atom']} is {item['role']}" for item in semantic)
        + "."
        if semantic
        else ""
    )
    return (
        f"{label}: {card['summary']} Metric roles: {roles}."
        + semantic_text
    )


def _semantic_evidence_items(
    observations: Sequence[SemanticObservation],
    candidate_cards: Sequence[CandidateEvidenceCard],
    mechanism_cards: Mapping[str, Mapping[str, Any]],
    training_associations: Mapping[str, Sequence[Tuple[str, float, float]]],
) -> Tuple[EvidenceItem, ...]:
    """Render anonymous query-shape atoms and mark training-stable matches."""

    candidate_roots = set(label for card in candidate_cards for label in card.root_labels)
    items: List[EvidenceItem] = []
    for index, observation in enumerate(observations, start=1):
        direct_roots = tuple(
            sorted(
                label
                for label in candidate_roots
                if any(
                    semantic.get("atom") in observation.atoms
                    and semantic.get("role") == "direct"
                    for semantic in mechanism_cards[label].get("semantic_observables") or ()
                )
            )
        )
        associated_roots = tuple(
            sorted(
                label
                for label in candidate_roots
                if any(
                    atom in observation.atoms
                    for atom, _, _ in training_associations.get(label, ())
                )
            )
        )
        supported_roots = tuple(sorted(set(direct_roots) | set(associated_roots)))
        if direct_roots:
            prefix = "Direct candidate-discriminating"
        elif associated_roots:
            prefix = "Strong cross-group training-associated candidate-discriminating"
        else:
            prefix = "Contextual/non-discriminating"
        items.append(
            EvidenceItem(
                evidence_id=f"semantic-observation:{index:02d}",
                kind="semantic_observation",
                summary=f"{prefix} {observation.summary()}",
                root_labels=supported_roots,
            )
        )
    return tuple(items)


def _llm_evidence(
    vectors: np.ndarray,
    train_x: np.ndarray,
    train_labels: Sequence[Sequence[str]],
    train_groups: Sequence[int],
    labels: Sequence[str],
    feature_names: Sequence[str],
    mechanism_cards: Mapping[str, Mapping[str, Any]],
    candidate_cards: Optional[Sequence[Sequence[CandidateEvidenceCard]]] = None,
    semantic_observations: Optional[Sequence[Sequence[SemanticObservation]]] = None,
    train_semantic_observations: Optional[
        Sequence[Sequence[SemanticObservation]]
    ] = None,
    require_direct_semantic: bool = False,
) -> List[LLMReviewEvidence]:
    """Create one evidence-grounded packet for every query and candidate pair."""

    vectors = np.asarray(vectors, dtype=np.float64)
    train_x = np.asarray(train_x, dtype=np.float64)
    if (
        train_x.ndim != 2
        or vectors.ndim != 2
        or train_x.shape[1] != vectors.shape[1]
        or vectors.shape[1] != len(feature_names)
        or set(labels) != set(mechanism_cards)
    ):
        raise ValueError("LLM evidence inputs are incompatible")
    if candidate_cards is None:
        cards_by_query: List[Tuple[CandidateEvidenceCard, ...]] = [tuple() for _ in vectors]
    else:
        cards_by_query = [tuple(row) for row in candidate_cards]
        if len(cards_by_query) != len(vectors):
            raise ValueError("candidate cards and metric observations differ")
        if any(
            not all(isinstance(card, CandidateEvidenceCard) for card in row)
            for row in cards_by_query
        ):
            raise ValueError("candidate cards must use CandidateEvidenceCard")
    if semantic_observations is None:
        semantic_by_query: List[Tuple[SemanticObservation, ...]] = [tuple() for _ in vectors]
    else:
        semantic_by_query = [tuple(row) for row in semantic_observations]
        if len(semantic_by_query) != len(vectors):
            raise ValueError("semantic observations and metric observations differ")
        if any(
            not all(isinstance(item, SemanticObservation) for item in row)
            for row in semantic_by_query
        ):
            raise ValueError("semantic observations have an invalid type")
    mean = np.mean(train_x, axis=0)
    scale = np.maximum(np.std(train_x, axis=0), 1e-12)
    standardized_query = (vectors - mean) / scale
    profiles_by_label = {
        item.root_labels[0]: item
        for item in _root_profile_items(
            train_x,
            train_labels,
            train_groups,
            labels,
            feature_names,
            train_semantic_observations,
        )
    }
    semantic_associations = _strong_semantic_associations(
        train_labels,
        train_groups,
        labels,
        train_semantic_observations,
    )
    mechanisms_by_label = {
        label: EvidenceItem(
            evidence_id=f"mechanism-card:{label}",
            kind="mechanism_card",
            summary=_mechanism_evidence_summary(label, mechanism_cards[label]),
            root_labels=(label,),
        )
        for label in labels
    }
    bundles: List[LLMReviewEvidence] = []
    for query, cards, observations in zip(
        standardized_query, cards_by_query, semantic_by_query
    ):
        relevant = {
            label
            for card in cards
            for label in card.root_labels
        }
        relevant_labels = tuple(label for label in labels if label in relevant) or tuple(labels)
        training_items = tuple(profiles_by_label[label] for label in relevant_labels)
        mechanism_items = tuple(mechanisms_by_label[label] for label in relevant_labels)
        trajectory = _trajectory_summary(query, feature_names)
        metric_context = _metric_semantics(feature_names)
        candidate_items = tuple(
            EvidenceItem(
                evidence_id=f"candidate-card:{card.candidate_id}",
                kind="candidate_evidence_card",
                summary=(
                    f"Candidate {', '.join(card.root_labels)}. Support: "
                    + (
                        "; ".join(
                            (
                                list(card.supporting_atoms)
                                + [
                                    f"{label}: current anonymous atom {atom} matches a strong cross-group training association"
                                    for label in card.root_labels
                                    for atom, _, _ in semantic_associations.get(label, ())
                                    if any(atom in observation.atoms for observation in observations)
                                ]
                            )[:4]
                        )
                        if card.supporting_atoms
                        or any(
                            atom in observation.atoms
                            for label in card.root_labels
                            for atom, _, _ in semantic_associations.get(label, ())
                            for observation in observations
                        )
                        else "none"
                    )
                    + ". Counterevidence: "
                    + ("; ".join(card.counterevidence_atoms) if card.counterevidence_atoms else "none")
                ),
                root_labels=card.root_labels,
            )
            for card in cards
        )
        semantic_items = _semantic_evidence_items(
            observations,
            cards,
            mechanism_cards,
            semantic_associations,
        )
        bundles.append(
            LLMReviewEvidence(
                query_items=(
                    EvidenceItem(
                        evidence_id="query-trajectory",
                        kind="query_metric_trajectory",
                        summary=(
                            f"{metric_context} Observed standardized KPI-time trajectory. "
                            "These are measurements, not candidate scores: "
                            f"{trajectory}."
                        ),
                    ),
                ),
                training_items=training_items,
                mechanism_items=mechanism_items,
                candidate_items=candidate_items,
                semantic_items=semantic_items,
                requires_direct_evidence=bool(require_direct_semantic),
            )
        )
    return bundles


def _select_remediation_jobs(
    jobs: Sequence[Tuple[str, Tuple[str, ...], LLMReviewEvidence]], budget: int
) -> Tuple[int, ...]:
    """Select a deterministic root-set-balanced audit sample without truth."""

    if int(budget) < 0:
        raise ValueError("remediation budget must be nonnegative")
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for index, (_, roots, _) in enumerate(jobs):
        groups.setdefault(tuple(roots), []).append(index)
    selected: List[int] = []
    depth = 0
    while len(selected) < min(int(budget), len(jobs)):
        progressed = False
        for roots in sorted(groups):
            if depth < len(groups[roots]):
                selected.append(groups[roots][depth])
                progressed = True
                if len(selected) >= min(int(budget), len(jobs)):
                    break
        if not progressed:
            break
        depth += 1
    return tuple(selected)


def _run_remediation_advice(
    client: LLMClient,
    jobs: Sequence[Tuple[str, Tuple[str, ...], LLMReviewEvidence]],
    budget: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Generate bounded post-diagnosis advice and its non-label audit."""

    from concurrent.futures import ThreadPoolExecutor

    if len({case_id for case_id, _, _ in jobs}) != len(jobs):
        raise ValueError("remediation jobs require unique case ids")
    selected_indices = _select_remediation_jobs(jobs, budget)

    def advice_inputs(
        index: int,
    ) -> Tuple[str, Tuple[str, ...], Tuple[EvidenceItem, ...], Tuple[EvidenceItem, ...]]:
        case_id, selected_roots, evidence = jobs[index]
        root_set = frozenset(selected_roots)
        mechanisms = tuple(
            item
            for item in evidence.mechanism_items
            if len(item.root_labels) == 1 and item.root_labels[0] in root_set
        )
        observations = tuple(
            item
            for item in evidence.query_items + evidence.semantic_items
            if set(item.root_labels) <= root_set
        )
        return case_id, selected_roots, mechanisms, observations

    def advise(index: int) -> Tuple[int, str, RemediationResult]:
        case_id, selected_roots, mechanisms, observations = advice_inputs(index)
        result = PostDiagnosisLLMAdvisor(client).advise(
            selected_roots, mechanisms, observations
        )
        return index, case_id, result

    if len(selected_indices) < 2 or LLM_MAX_WORKERS == 1:
        responses = [advise(index) for index in selected_indices]
    else:
        with ThreadPoolExecutor(
            max_workers=min(LLM_MAX_WORKERS, len(selected_indices))
        ) as executor:
            responses = list(executor.map(advise, selected_indices))

    by_case = {case_id: result.as_dict() for _, case_id, result in responses}
    accepted_responses = [
        (index, case_id, result)
        for index, case_id, result in responses
        if result.status == "accepted_recommendation"
    ]

    def review_quality(
        response: Tuple[int, str, RemediationResult]
    ) -> Tuple[str, RemediationQualityResult]:
        index, case_id, recommendation = response
        _, selected_roots, mechanisms, observations = advice_inputs(index)
        result = BlindedRemediationQualityReviewer(client).review(
            selected_roots, mechanisms, observations, recommendation
        )
        return case_id, result

    if len(accepted_responses) < 2 or LLM_MAX_WORKERS == 1:
        quality_responses = [review_quality(response) for response in accepted_responses]
    else:
        with ThreadPoolExecutor(
            max_workers=min(LLM_MAX_WORKERS, len(accepted_responses))
        ) as executor:
            quality_responses = list(executor.map(review_quality, accepted_responses))
    quality_by_case = {
        case_id: result.as_dict() for case_id, result in quality_responses
    }
    not_sampled = RemediationResult(
        status="not_sampled_fixed_evaluation_budget",
        called=False,
        response_count=0,
    ).as_dict()
    rows = {
        case_id: dict(by_case.get(case_id, not_sampled))
        for case_id, _, _ in jobs
    }
    quality_not_run = RemediationQualityResult(
        status="not_evaluated_advice_not_accepted",
        called=False,
        response_count=0,
    ).as_dict()
    for case_id, row in rows.items():
        row["quality_review"] = dict(
            quality_by_case.get(case_id, quality_not_run)
        )
    statuses = Counter(row["status"] for row in rows.values())
    action_types = Counter(
        row["action_type"]
        for row in rows.values()
        if row["status"] == "accepted_recommendation"
    )
    risk_levels = Counter(
        row["risk_level"]
        for row in rows.values()
        if row["status"] == "accepted_recommendation"
    )
    accepted = int(statuses["accepted_recommendation"])
    sampled = len(selected_indices)
    valid = accepted + int(statuses["accepted_no_action"])
    quality_rows = list(quality_by_case.values())
    quality_valid = [
        row
        for row in quality_rows
        if row["status"] == "accepted_blinded_quality_review"
    ]
    quality_statuses = Counter(row["status"] for row in quality_rows)
    issue_codes = Counter(
        issue for row in quality_valid for issue in row["issue_codes"]
    )
    confidence = Counter(row["confidence"] for row in quality_valid)

    def mean_score(field: str) -> Optional[float]:
        values = [int(row[field]) for row in quality_valid]
        return sum(values) / len(values) if values else None

    return rows, {
        "status": "evaluated",
        "authority": "post-diagnosis advice only; the root set is frozen",
        "selection": "deterministic round-robin by predicted root set; no truth labels",
        "eligible_case_count": len(jobs),
        "budget": int(budget),
        "sampled_case_count": sampled,
        "model_response_count": sum(
            int(row["response_count"]) for row in rows.values()
        ),
        "accepted_recommendation_count": accepted,
        "accepted_no_action_count": int(statuses["accepted_no_action"]),
        "actionable_recommendation_rate": accepted / sampled if sampled else None,
        "schema_and_citation_acceptance_rate": valid / sampled if sampled else None,
        "unsafe_recommendation_rejection_count": int(
            statuses["fallback_unsafe_recommendation"]
        ),
        "accepted_unsafe_action_count": 0,
        "diagnosis_mutation_count": 0,
        "statuses": dict(sorted(statuses.items())),
        "action_types": dict(sorted(action_types.items())),
        "risk_levels": dict(sorted(risk_levels.items())),
        "blinded_quality_review": {
            "status": "automated_blinded_second_pass_proxy_not_human_expert_review",
            "reviewed_recommendation_count": len(quality_rows),
            "valid_review_count": len(quality_valid),
            "model_response_count": sum(
                int(row["response_count"]) for row in quality_rows
            ),
            "quality_pass_count": sum(
                bool(row["quality_pass"]) for row in quality_valid
            ),
            "quality_pass_rate": (
                sum(bool(row["quality_pass"]) for row in quality_valid)
                / len(quality_valid)
                if quality_valid
                else None
            ),
            "mean_scores": {
                "evidence_grounding": mean_score("evidence_grounding_score"),
                "root_relevance": mean_score("root_relevance_score"),
                "actionability": mean_score("actionability_score"),
                "verification_quality": mean_score("verification_quality_score"),
            },
            "statuses": dict(sorted(quality_statuses.items())),
            "issue_codes": dict(sorted(issue_codes.items())),
            "confidence": dict(sorted(confidence.items())),
            "truth_labels_exposed": False,
            "dataset_method_and_case_identity_exposed": False,
            "reviewer_model_independence": "same configured model in a separate blinded call",
            "human_expert_status": "not_evaluated",
        },
    }


def _review_structured_pairs(
    dataset: AblationDataset,
    h_model: LightweightHDiffusionClassifier,
    vectors: np.ndarray,
    train_x: np.ndarray,
    train_labels: Sequence[Sequence[str]],
    train_groups: Sequence[int],
    handoff: _StructuredHandoff,
    client: LLMClient,
    *,
    allow_override: bool,
    allowed_override_relations: Optional[Sequence[str]] = None,
    semantic_observations: Optional[Sequence[Sequence[SemanticObservation]]] = None,
    train_semantic_observations: Optional[
        Sequence[Sequence[SemanticObservation]]
    ] = None,
    require_direct_semantic: bool = False,
) -> List[Dict[str, Any]]:
    """Run the one active, evidence-bounded reviewer for a prepared handoff."""

    evidence = _llm_evidence(
        vectors,
        train_x,
        train_labels,
        train_groups,
        dataset.labels,
        dataset.feature_names,
        dataset.mechanism_cards,
        handoff.candidate_cards,
        semantic_observations,
        train_semantic_observations,
        require_direct_semantic,
    )
    pipeline = HyperDBDiagPipeline(
        dataset.labels,
        metric_model=h_model,
        llm_reviewer=CandidateBoundLLMReviewer(
            client=client,
            allow_override=allow_override,
            allowed_override_relations=allowed_override_relations,
        ),
        llm_max_workers=LLM_MAX_WORKERS,
    )
    return pipeline.predict_details(
        vectors,
        evidence,
        handoff.local_indices,
        handoff.alternative_indices,
        semantic_observations,
    )


def _binomial_greater_p_value(successes: int, trials: int) -> float:
    """Exact one-sided sign-test tail without adding a scipy dependency."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid sign-test counts")
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2.0 ** trials)
    )


def _llm_policy_from_oof(
    group_outcomes: Sequence[Mapping[str, int]],
    *,
    minimum_decisions: int,
    minimum_active_groups: int,
    sign_test_alpha: float,
) -> Dict[str, Any]:
    """Gate overrides using training-only paired OOF outcomes."""

    corrected = int(sum(int(row.get("corrected", 0)) for row in group_outcomes))
    harmed = int(sum(int(row.get("harmed", 0)) for row in group_outcomes))
    changed = int(sum(int(row.get("changed", 0)) for row in group_outcomes))
    active = [
        row
        for row in group_outcomes
        if int(row.get("corrected", 0)) + int(row.get("harmed", 0)) > 0
    ]
    net = corrected - harmed
    decisions = corrected + harmed
    sign_p_value = (
        None
        if decisions == 0
        else _binomial_greater_p_value(corrected, decisions)
    )
    stable = bool(
        len(active) >= int(minimum_active_groups)
        and decisions >= int(minimum_decisions)
        and net > 0
        and sign_p_value is not None
        and sign_p_value <= float(sign_test_alpha)
    )
    return {
        "allow_override": stable,
        "corrected": corrected,
        "harmed": harmed,
        "changed": changed,
        "net_exact_gain_count": net,
        "active_group_count": len(active),
        "decision_count": decisions,
        "sign_test_p_value": sign_p_value,
        "minimum_decisions": int(minimum_decisions),
        "minimum_active_groups": int(minimum_active_groups),
        "sign_test_alpha": sign_test_alpha,
        "stability_rule": "paired exact sign test over OOF corrections versus harms, with minimum decision and group coverage",
        "status": "enabled_stable_positive_oof" if stable else "fail_closed_no_stable_positive_oof",
    }


def _relation_policies_from_oof(
    outcomes_by_relation: Mapping[str, Sequence[Mapping[str, int]]],
) -> Dict[str, Dict[str, Any]]:
    """Calibrate root-agnostic ECSA relation gates on outer-training OOF only."""

    policies: Dict[str, Dict[str, Any]] = {}
    for relation in LLM_SELECTION_RELATIONS:
        rows = tuple(outcomes_by_relation.get(relation, ()))
        policy = _llm_policy_from_oof(
            rows,
            minimum_decisions=LLM_RELATION_MIN_OOF_DECISIONS,
            minimum_active_groups=LLM_RELATION_MIN_OOF_ACTIVE_GROUPS,
            sign_test_alpha=LLM_MAX_OOF_SIGN_P_VALUE,
        )
        negative_groups = sum(
            int(row.get("harmed", 0)) > int(row.get("corrected", 0))
            for row in rows
        )
        relation_enabled = bool(policy["allow_override"] and negative_groups == 0)
        policies[relation] = {
            **policy,
            "allow_override": relation_enabled,
            "negative_net_group_count": int(negative_groups),
            "status": (
                "enabled_stable_positive_relation_oof"
                if relation_enabled
                else "fail_closed_no_stable_positive_relation_oof"
            ),
            "stratification": "ECSA relation only; root labels and evaluation outcomes are excluded",
        }
    return policies


def _summary(
    expected: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    labels: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    seconds: Sequence[float],
) -> Dict[str, Any]:
    by_root: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        indices = [index for index, row in enumerate(expected) if label in row]
        truth = [expected[index] for index in indices]
        guesses = [predicted[index] for index in indices]
        by_root[label] = {
            "sample_count": len(indices),
            "exact_set_accuracy": _exact(truth, guesses),
            "component_f1": _component_f1(truth, guesses, labels),
        }
    return {
        "overall": {
            "sample_count": len(expected),
            "exact_set_accuracy": _exact(expected, predicted),
            "component_f1": _component_f1(expected, predicted, labels),
            "mean_fit_plus_predict_seconds_per_outer_split": _mean(seconds),
        },
        "by_root": by_root,
        "results": list(rows),
    }


def _convergence_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize set coverage and contraction from the same prediction rows."""

    if not rows:
        raise ValueError("convergence rows must be nonempty")
    truth_sets = [frozenset(row["expected_labels"]) for row in rows]
    predicted_sets = [frozenset(row["predicted_labels"]) for row in rows]
    true_components = sum(len(row) for row in truth_sets)
    predicted_components = sum(len(row) for row in predicted_sets)
    true_positive_components = sum(
        len(truth & predicted) for truth, predicted in zip(truth_sets, predicted_sets)
    )
    return {
        "sample_count": len(rows),
        "mean_predicted_roots": _mean(len(row) for row in predicted_sets),
        "full_coverage": sum(
            truth <= predicted for truth, predicted in zip(truth_sets, predicted_sets)
        )
        / float(len(rows)),
        "component_redundancy": (
            float(predicted_components - true_positive_components)
            / float(predicted_components)
            if predicted_components
            else 0.0
        ),
        "root_count_accuracy": sum(
            len(truth) == len(predicted)
            for truth, predicted in zip(truth_sets, predicted_sets)
        )
        / float(len(rows)),
        "component_precision": (
            float(true_positive_components) / float(predicted_components)
            if predicted_components
            else 0.0
        ),
        "component_recall": (
            float(true_positive_components) / float(true_components)
            if true_components
            else 0.0
        ),
    }


def _revision_outcomes(
    expected: Sequence[Sequence[str]],
    hypergraph_prediction: Sequence[Sequence[str]],
    reviewed_prediction: Sequence[Sequence[str]],
) -> Dict[str, int]:
    """Post-prediction audit only; evaluation labels never enter the reviewer."""

    if not (len(expected) == len(hypergraph_prediction) == len(reviewed_prediction)):
        raise ValueError("revision outcome rows are misaligned")
    outcomes: Counter[str] = Counter()
    for truth, local, final in zip(expected, hypergraph_prediction, reviewed_prediction):
        if set(local) == set(final):
            outcomes["unchanged"] += 1
        elif set(final) == set(truth) and set(local) != set(truth):
            outcomes["corrected"] += 1
        elif set(local) == set(truth) and set(final) != set(truth):
            outcomes["harmed"] += 1
        else:
            outcomes["changed_but_still_incorrect"] += 1
    return {
        "corrected": int(outcomes["corrected"]),
        "harmed": int(outcomes["harmed"]),
        "changed_but_still_incorrect": int(outcomes["changed_but_still_incorrect"]),
        "unchanged": int(outcomes["unchanged"]),
    }


def _prediction_rows(
    split: AblationSplit,
    expected: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
) -> List[Dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "outer_split": split.split_id,
            "expected_labels": list(truth),
            "predicted_labels": list(guess),
        }
        for case_id, truth, guess in zip(split.eval_ids, expected, predicted)
    ]


def _new_h_model(
    dataset: AblationDataset, seed: int, *, enable_epdg: bool = True
) -> LightweightHDiffusionClassifier:
    semantic_edges = _epdg_semantic_path_edges(
        dataset.labels, dataset.mechanism_cards
    )
    epdg_available = bool(
        any(dataset.epdg_path_edges.get(label) for label in dataset.labels)
        or any(semantic_edges.get(label) for label in dataset.labels)
    )
    return LightweightHDiffusionClassifier(
        activation_threshold=dataset.activation_threshold,
        seed=int(seed),
        decoder_criterion=dataset.decoder_criterion,
        decoder_n_estimators=dataset.decoder_n_estimators,
        epdg_path_edges=dataset.epdg_path_edges,
        epdg_feature_names=dataset.feature_names,
        epdg_semantic_path_edges=semantic_edges,
        enable_epdg=bool(enable_epdg and epdg_available),
    )


def _allocate_llm_review_budget(capacities: Sequence[int], budget: int) -> Tuple[int, ...]:
    """Spread a bounded review budget across groups without wasting capacity."""

    values = tuple(int(value) for value in capacities)
    if int(budget) < 0 or any(value < 0 for value in values):
        raise ValueError("review capacities and budget must be nonnegative")
    allocation = [0] * len(values)
    remaining = int(budget)
    while remaining:
        progressed = False
        for index, capacity in enumerate(values):
            if remaining <= 0:
                break
            if allocation[index] >= capacity:
                continue
            allocation[index] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return tuple(allocation)


def _training_oof_pair_audit(
    expected: Sequence[Sequence[str]],
    labels: Sequence[str],
    h_model: LightweightHDiffusionClassifier,
    handoff: _StructuredHandoff,
) -> Dict[str, int]:
    """Measure candidate-pair coverage using held-out rows inside training only."""

    if len(expected) != len(handoff.local_indices):
        raise ValueError("training OOF truth and structured handoff differ in length")
    counts: Counter[str] = Counter()
    for truth, local_index, alternative_index in zip(
        expected,
        handoff.local_indices,
        handoff.alternative_indices,
    ):
        if alternative_index is None:
            continue
        truth_set = frozenset(truth)
        local_set = frozenset(labels[root] for root in h_model.candidates[local_index])
        alternative_set = frozenset(
            labels[root] for root in h_model.candidates[alternative_index]
        )
        local_correct = local_set == truth_set
        alternative_correct = alternative_set == truth_set
        counts["eligible_pair_count"] += 1
        counts["truth_in_pair_count"] += int(local_correct or alternative_correct)
        counts["recoverable_error_count"] += int(not local_correct and alternative_correct)
        counts["local_correct_count"] += int(local_correct)
        counts["neither_candidate_correct_count"] += int(
            not local_correct and not alternative_correct
        )
    return {
        key: int(counts[key])
        for key in (
            "eligible_pair_count",
            "truth_in_pair_count",
            "recoverable_error_count",
            "local_correct_count",
            "neither_candidate_correct_count",
        )
    }


def _calibrate_llm_on_training(
    dataset: AblationDataset,
    split: AblationSplit,
    seed: int,
    client: LLMClient,
    *,
    use_local_judge: bool = True,
) -> Dict[str, Any]:
    """Cross-fit the LLM handoff using only the current outer training fold."""

    groups = np.asarray(split.oof_groups, dtype=object)
    group_values = tuple(sorted(set(groups.tolist()), key=str))
    outcomes: List[Dict[str, int]] = []
    relation_outcomes: Dict[str, List[Dict[str, int]]] = {
        relation: [] for relation in LLM_SELECTION_RELATIONS
    }
    pair_audit: Counter[str] = Counter()
    review_requests = 0
    model_responses = 0
    review_statuses: Counter[str] = Counter()
    review_reason_codes: Counter[str] = Counter()
    review_relation_types: Counter[str] = Counter()
    review_recommendations: Counter[str] = Counter()
    eligible_pairs = 0
    deferred_pairs = 0
    skipped_groups = 0
    prepared_groups: List[Dict[str, Any]] = []
    for group in group_values:
        held = groups == group
        fit = ~held
        fit_groups = groups[fit]
        if len(set(fit_groups.tolist())) < 3:
            skipped_groups += 1
            continue
        held_labels = [row for row, keep in zip(split.train_labels, held) if keep]
        held_semantic = [row for row, keep in zip(split.train_semantic, held) if keep]
        h_model = _new_h_model(dataset, seed).fit(
            split.train_x[fit],
            [row for row, keep in zip(split.train_labels, fit) if keep],
            dataset.labels,
            fit_groups,
            [row for row, keep in zip(split.train_semantic, fit) if keep],
        )
        try:
            _assert_candidate_coverage(held_labels, h_model.candidates, dataset.labels)
        except ValueError:
            skipped_groups += 1
            continue
        identifiers = tuple(f"candidate-{index:03d}" for index in range(len(h_model.candidates)))
        if use_local_judge:
            judge = StructuredEvidenceJudge().fit(
                split.train_x[fit],
                [row for row, keep in zip(split.train_labels, fit) if keep],
                fit_groups,
                dataset.labels,
                dataset.feature_names,
            )
            handoff = _structured_handoff(
                h_model,
                judge,
                split.train_x[held],
                identifiers,
                semantic_observations=held_semantic or None,
                mechanism_cards=dataset.mechanism_cards,
            )
            handoff = _retain_semantic_challengers(
                dataset,
                h_model,
                handoff,
                held_semantic,
                [row for row, keep in zip(split.train_labels, fit) if keep],
                fit_groups,
                [row for row, keep in zip(split.train_semantic, fit) if keep],
            )
        else:
            handoff = _direct_handoff(
                h_model,
                split.train_x[held],
                identifiers,
                dataset.labels,
                held_semantic,
            )
        deferred = sum(index is not None for index in handoff.alternative_indices)
        eligible = deferred
        deferred_pairs += int(deferred)
        eligible_pairs += int(eligible)
        pair_audit.update(
            _training_oof_pair_audit(
                held_labels,
                dataset.labels,
                h_model,
                handoff,
            )
        )
        prepared_groups.append(
            {
                "held": held,
                "fit": fit,
                "fit_groups": fit_groups,
                "held_labels": held_labels,
                "fit_labels": [row for row, keep in zip(split.train_labels, fit) if keep],
                "held_semantic": held_semantic,
                "fit_semantic": [row for row, keep in zip(split.train_semantic, fit) if keep],
                "h_model": h_model,
                "handoff": handoff,
                "eligible": int(eligible),
            }
        )

    group_budgets = _allocate_llm_review_budget(
        [int(row["eligible"]) for row in prepared_groups],
        LLM_REVIEW_BUDGET_PER_OUTER_SPLIT,
    )
    for prepared, group_budget in zip(prepared_groups, group_budgets):
        if not prepared["eligible"] or not group_budget:
            outcomes.append({"corrected": 0, "harmed": 0, "changed": 0})
            for relation in LLM_SELECTION_RELATIONS:
                relation_outcomes[relation].append(
                    {"corrected": 0, "harmed": 0, "changed": 0}
                )
            continue
        held = np.asarray(prepared["held"], dtype=bool)
        fit = np.asarray(prepared["fit"], dtype=bool)
        fit_groups = np.asarray(prepared["fit_groups"], dtype=object)
        held_labels = list(prepared["held_labels"])
        fit_labels = list(prepared["fit_labels"])
        h_model = prepared["h_model"]
        handoff = prepared["handoff"]
        details = _review_structured_pairs(
            dataset,
            h_model,
            split.train_x[held],
            split.train_x[fit],
            fit_labels,
            fit_groups,
            _limit_review_handoff(
                handoff,
                group_budget,
                h_model.predict_proba(split.train_x[held], prepared["held_semantic"]),
            ),
            client,
            allow_override=True,
            semantic_observations=prepared.get("held_semantic") or None,
            train_semantic_observations=prepared.get("fit_semantic") or None,
            require_direct_semantic=dataset.require_discriminating_semantic_for_llm,
        )
        local_prediction = [
            [dataset.labels[root] for root in h_model.candidates[index]]
            for index in handoff.local_indices
        ]
        final_prediction = [list(row["predicted_labels"]) for row in details]
        outcome = _revision_outcomes(held_labels, local_prediction, final_prediction)
        changed = sum(
            set(local) != set(final)
            for local, final in zip(local_prediction, final_prediction)
        )
        outcomes.append(
            {
                "corrected": int(outcome["corrected"]),
                "harmed": int(outcome["harmed"]),
                "changed": int(changed),
            }
        )
        group_relation_outcomes = {
            relation: {"corrected": 0, "harmed": 0, "changed": 0}
            for relation in LLM_SELECTION_RELATIONS
        }
        for truth, local, final, detail in zip(
            held_labels, local_prediction, final_prediction, details
        ):
            relation = detail["llm_relation_type"]
            if relation not in group_relation_outcomes or set(local) == set(final):
                continue
            row = group_relation_outcomes[relation]
            row["changed"] += 1
            if set(final) == set(truth) and set(local) != set(truth):
                row["corrected"] += 1
            elif set(local) == set(truth) and set(final) != set(truth):
                row["harmed"] += 1
        for relation, row in group_relation_outcomes.items():
            relation_outcomes[relation].append(row)
        review_requests += sum(bool(row["llm_called"]) for row in details)
        model_responses += sum(int(row["llm_response_count"]) for row in details)
        review_statuses.update(str(row["llm_status"]) for row in details if row["llm_called"])
        review_reason_codes.update(
            str(row["llm_reason_code"])
            for row in details
            if row["llm_reason_code"] is not None
        )
        review_relation_types.update(
            str(row["llm_relation_type"])
            for row in details
            if row["llm_relation_type"] is not None
        )
        review_recommendations.update(
            str(row["llm_recommendation"])
            for row in details
            if row["llm_recommendation"] is not None
        )
    policy = _llm_policy_from_oof(
        outcomes,
        minimum_decisions=LLM_MIN_OOF_DECISIONS,
        minimum_active_groups=LLM_MIN_OOF_ACTIVE_GROUPS,
        sign_test_alpha=LLM_MAX_OOF_SIGN_P_VALUE,
    )
    relation_policies = _relation_policies_from_oof(relation_outcomes)
    allowed_override_relations = tuple(
        relation
        for relation in LLM_SELECTION_RELATIONS
        if relation_policies[relation]["allow_override"]
    )
    pair_count = int(pair_audit["eligible_pair_count"])
    pair_audit_counts = {
        key: int(pair_audit[key])
        for key in (
            "eligible_pair_count",
            "truth_in_pair_count",
            "recoverable_error_count",
            "local_correct_count",
            "neither_candidate_correct_count",
        )
    }
    return {
        "status": (
            "enabled_relation_specific_positive_oof"
            if allowed_override_relations
            else "fail_closed_no_stable_positive_relation_oof"
        ),
        "allow_override": bool(allowed_override_relations),
        "allowed_override_relations": list(allowed_override_relations),
        "group_count": len(group_values),
        "evaluated_group_count": len(outcomes),
        "skipped_group_count": skipped_groups,
        "eligible_pair_count": eligible_pairs,
        "deferred_pair_count": deferred_pairs,
        "review_requests": review_requests,
        "model_responses": model_responses,
        "review_budget_per_outer_split": LLM_REVIEW_BUDGET_PER_OUTER_SPLIT,
        "review_budget_allocation": "round-robin across inner groups, then smallest hypergraph candidate margin within each group",
        "review_statuses": dict(sorted(review_statuses.items())),
        "review_reason_codes": dict(sorted(review_reason_codes.items())),
        "review_relation_types": dict(sorted(review_relation_types.items())),
        "review_recommendations": dict(sorted(review_recommendations.items())),
        "oof_outcomes": outcomes,
        "global_policy_for_audit": policy,
        "relation_policies": relation_policies,
        "training_oof_candidate_pair_audit": {
            **pair_audit_counts,
            "truth_in_pair_rate": (
                float(pair_audit["truth_in_pair_count"]) / pair_count
                if pair_count
                else None
            ),
            "recoverable_error_rate": (
                float(pair_audit["recoverable_error_count"]) / pair_count
                if pair_count
                else None
            ),
            "scope": "held-out groups inside the current outer training fold only",
            "used_evaluation_labels": False,
        },
    }


def _run_dataset(dataset: AblationDataset, seed: int) -> Dict[str, Any]:
    llm_client, llm_configuration = _configured_llm_client()
    llm_enabled = llm_client is not None
    evaluated_method_ids = (
        "ordinary_binary_graph",
        "hypergraph_without_epdg",
        "hypergraph",
        "hypergraph_local_judge",
    ) + (("hypergraph_llm", "hyperdbdiag") if llm_enabled else ())
    collected: Dict[str, Dict[str, Any]] = {
        name: {"expected": [], "predicted": [], "rows": [], "seconds": []}
        for name in evaluated_method_ids
    }
    folds: List[Dict[str, Any]] = []
    local_statuses: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    review_reason_codes: Counter[str] = Counter()
    review_relation_types: Counter[str] = Counter()
    review_recommendations: Counter[str] = Counter()
    local_conflicts = 0
    local_changes = 0
    llm_review_requests = 0
    llm_model_responses = 0
    llm_calibration_requests = 0
    llm_calibration_responses = 0
    llm_calibration_enabled_splits = 0
    llm_completed_reviews = 0
    llm_candidate_selections = 0
    llm_overrides = 0
    llm_failed_reviews = 0
    llm_audit_only_skipped = 0
    direct_llm_review_requests = 0
    direct_llm_model_responses = 0
    direct_llm_calibration_requests = 0
    direct_llm_calibration_responses = 0
    direct_llm_calibration_enabled_splits = 0
    direct_llm_completed_reviews = 0
    direct_llm_candidate_selections = 0
    direct_llm_overrides = 0
    direct_llm_failed_reviews = 0
    direct_llm_audit_only_skipped = 0
    remediation_jobs: List[Tuple[str, Tuple[str, ...], LLMReviewEvidence]] = []

    for split in dataset.splits:
        split.validate(dataset.feature_count, dataset.labels)
        ordinary_started = time.perf_counter()
        ordinary_model = OrdinaryBinaryGraphClassifier(seed=int(seed)).fit(
            split.train_x, split.train_labels, dataset.labels, split.oof_groups
        )
        ordinary_prediction = ordinary_model.predict(split.eval_x, dataset.labels)
        ordinary_seconds = time.perf_counter() - ordinary_started

        h_without_epdg_started = time.perf_counter()
        h_without_epdg_model = _new_h_model(dataset, seed, enable_epdg=False).fit(
            split.train_x, split.train_labels, dataset.labels, split.oof_groups
        )
        h_without_epdg_prediction = h_without_epdg_model.predict(
            split.eval_x, dataset.labels
        )
        h_without_epdg_seconds = time.perf_counter() - h_without_epdg_started

        h_started = time.perf_counter()
        h_model = _new_h_model(dataset, seed).fit(
            split.train_x,
            split.train_labels,
            dataset.labels,
            split.oof_groups,
            split.train_semantic,
        )
        h_probabilities = h_model.predict_proba(
            split.eval_x, split.eval_semantic or None
        )
        h_indices = np.asarray(np.argmax(h_probabilities, axis=1), dtype=np.int64)
        h_prediction = [
            [dataset.labels[root] for root in h_model.candidates[index]]
            for index in h_indices
        ]
        h_seconds = time.perf_counter() - h_started
        expected = [list(row) for row in split.eval_labels]
        _assert_candidate_coverage(expected, h_model.candidates, dataset.labels)

        local_started = time.perf_counter()
        judge = StructuredEvidenceJudge().fit(
            split.train_x,
            split.train_labels,
            split.oof_groups,
            dataset.labels,
            dataset.feature_names,
        )
        candidate_ids = tuple(f"candidate-{index:03d}" for index in range(len(h_model.candidates)))
        handoff = _structured_handoff(
            h_model,
            judge,
            split.eval_x,
            candidate_ids,
            h_indices,
            semantic_observations=split.eval_semantic or None,
            mechanism_cards=dataset.mechanism_cards,
        )
        handoff = _retain_semantic_challengers(
            dataset,
            h_model,
            handoff,
            split.eval_semantic,
            split.train_labels,
            split.oof_groups,
            split.train_semantic,
        )
        conflicts = handoff.conflicts
        local_indices = list(handoff.local_indices)
        for conflict, h_index, local_index, alternative_index in zip(
            conflicts,
            h_indices,
            local_indices,
            handoff.alternative_indices,
        ):
            local_statuses.update([conflict.status])
            if local_index != int(h_index):
                local_changes += 1
            if alternative_index is not None:
                local_conflicts += 1
        local_prediction = [
            [dataset.labels[root] for root in h_model.candidates[index]] for index in local_indices
        ]
        local_seconds = h_seconds + (time.perf_counter() - local_started)

        direct_handoff = _direct_handoff(
            h_model,
            split.eval_x,
            candidate_ids,
            dataset.labels,
            split.eval_semantic,
        )
        direct_prediction: List[List[str]] = [list(row) for row in h_prediction]
        direct_seconds = h_seconds
        direct_details: List[Dict[str, Any]] = []
        direct_llm_calibration: Optional[Dict[str, Any]] = None
        if llm_enabled:
            direct_calibration_started = time.perf_counter()
            direct_llm_calibration = _calibrate_llm_on_training(
                dataset,
                split,
                seed,
                llm_client,
                use_local_judge=False,
            )
            direct_seconds += time.perf_counter() - direct_calibration_started
            direct_llm_calibration_requests += int(
                direct_llm_calibration["review_requests"]
            )
            direct_llm_calibration_responses += int(
                direct_llm_calibration["model_responses"]
            )
            if direct_llm_calibration["allow_override"]:
                direct_llm_calibration_enabled_splits += 1
                direct_review_started = time.perf_counter()
                direct_details = _review_structured_pairs(
                    dataset,
                    h_model,
                    split.eval_x,
                    split.train_x,
                    split.train_labels,
                    split.oof_groups,
                    _limit_review_handoff(
                        direct_handoff,
                        LLM_REVIEW_BUDGET_PER_OUTER_SPLIT,
                        h_probabilities,
                    ),
                    llm_client,
                    allow_override=True,
                    allowed_override_relations=direct_llm_calibration[
                        "allowed_override_relations"
                    ],
                    semantic_observations=split.eval_semantic or None,
                    train_semantic_observations=split.train_semantic or None,
                    require_direct_semantic=dataset.require_discriminating_semantic_for_llm,
                )
                direct_seconds += time.perf_counter() - direct_review_started
                direct_prediction = [
                    list(row["predicted_labels"]) for row in direct_details
                ]
                direct_llm_review_requests += sum(
                    bool(row["llm_called"]) for row in direct_details
                )
                direct_llm_model_responses += sum(
                    int(row["llm_response_count"]) for row in direct_details
                )
                direct_llm_completed_reviews += sum(
                    str(row["llm_status"]) in _COMPLETED_LLM_REVIEW_STATUSES
                    for row in direct_details
                )
                direct_llm_candidate_selections += sum(
                    str(row["llm_status"]) == "accepted_candidate_selection"
                    for row in direct_details
                )
                direct_llm_overrides += sum(
                    set(row["predicted_labels"]) != set(row["local_labels"])
                    for row in direct_details
                )
                direct_llm_failed_reviews += sum(
                    bool(row["llm_called"])
                    and str(row["llm_status"]).startswith("fallback_")
                    for row in direct_details
                )
            else:
                direct_llm_audit_only_skipped += sum(
                    alternative is not None
                    for alternative in direct_handoff.alternative_indices
                )

        reviewed_prediction: Optional[List[List[str]]] = None
        full_seconds = local_seconds
        details: List[Dict[str, Any]] = []
        llm_calibration: Optional[Dict[str, Any]] = None
        if llm_enabled:
            calibration_started = time.perf_counter()
            llm_calibration = _calibrate_llm_on_training(dataset, split, seed, llm_client)
            full_seconds += time.perf_counter() - calibration_started
            llm_calibration_requests += int(llm_calibration["review_requests"])
            llm_calibration_responses += int(llm_calibration["model_responses"])
            if llm_calibration["allow_override"]:
                llm_calibration_enabled_splits += 1
            if llm_calibration["allow_override"]:
                review_started = time.perf_counter()
                details = _review_structured_pairs(
                    dataset,
                    h_model,
                    split.eval_x,
                    split.train_x,
                    split.train_labels,
                    split.oof_groups,
                    _limit_review_handoff(
                        handoff,
                        LLM_REVIEW_BUDGET_PER_OUTER_SPLIT,
                        h_probabilities,
                    ),
                    llm_client,
                    allow_override=True,
                    allowed_override_relations=llm_calibration[
                        "allowed_override_relations"
                    ],
                    semantic_observations=split.eval_semantic or None,
                    train_semantic_observations=split.train_semantic or None,
                    require_direct_semantic=dataset.require_discriminating_semantic_for_llm,
                )
                full_seconds += time.perf_counter() - review_started
                reviewed_prediction = [list(row["predicted_labels"]) for row in details]
                llm_review_requests += sum(bool(row["llm_called"]) for row in details)
                llm_model_responses += sum(int(row["llm_response_count"]) for row in details)
                llm_completed_reviews += sum(
                    str(row["llm_status"]) in _COMPLETED_LLM_REVIEW_STATUSES for row in details
                )
                llm_candidate_selections += sum(
                    str(row["llm_status"]) == "accepted_candidate_selection" for row in details
                )
                llm_overrides += sum(
                    set(row["predicted_labels"]) != set(row["local_labels"])
                    for row in details
                )
                llm_failed_reviews += sum(
                    bool(row["llm_called"])
                    and str(row["llm_status"]).startswith("fallback_")
                    for row in details
                )
                review_statuses.update(str(row["llm_status"]) for row in details)
                review_reason_codes.update(
                    str(row["llm_reason_code"])
                    for row in details
                    if row["llm_reason_code"] is not None
                )
                review_relation_types.update(
                    str(row["llm_relation_type"])
                    for row in details
                    if row["llm_relation_type"] is not None
                )
                review_recommendations.update(
                    str(row["llm_recommendation"])
                    for row in details
                    if row["llm_recommendation"] is not None
                )
            else:
                # Calibration did not establish a stable benefit. Keep the
                # local prediction and skip evaluation-fold API calls.
                reviewed_prediction = local_prediction
                llm_audit_only_skipped += sum(
                    alternative is not None
                    for alternative in handoff.alternative_indices
                )

        predictions: Dict[str, Tuple[Sequence[Sequence[str]], float]] = {
            "ordinary_binary_graph": (ordinary_prediction, ordinary_seconds),
            "hypergraph_without_epdg": (
                h_without_epdg_prediction,
                h_without_epdg_seconds,
            ),
            "hypergraph": (h_prediction, h_seconds),
            "hypergraph_local_judge": (local_prediction, local_seconds),
        }
        if llm_enabled:
            predictions["hypergraph_llm"] = (direct_prediction, direct_seconds)
        if reviewed_prediction is not None:
            predictions["hyperdbdiag"] = (reviewed_prediction, full_seconds)
            remediation_evidence = _llm_evidence(
                split.eval_x,
                split.train_x,
                split.train_labels,
                split.oof_groups,
                dataset.labels,
                dataset.feature_names,
                dataset.mechanism_cards,
                handoff.candidate_cards,
                split.eval_semantic or None,
                split.train_semantic or None,
                dataset.require_discriminating_semantic_for_llm,
            )
            remediation_jobs.extend(
                (str(case_id), tuple(roots), evidence)
                for case_id, roots, evidence in zip(
                    split.eval_ids, reviewed_prediction, remediation_evidence
                )
            )
        for method_id, (prediction, seconds) in predictions.items():
            collected[method_id]["expected"].extend(expected)
            collected[method_id]["predicted"].extend(prediction)
            collected[method_id]["rows"].extend(_prediction_rows(split, expected, prediction))
            collected[method_id]["seconds"].append(float(seconds))
        folds.append(
            {
                "outer_split": split.split_id,
                "train_count": len(split.train_ids),
                "evaluation_count": len(split.eval_ids),
                "train_evaluation_id_overlap": [],
                "inner_oof_group_count": len(set(split.oof_groups)),
                "ordinary_binary_graph": ordinary_model.training_metadata,
                "hypergraph_without_epdg": h_without_epdg_model.training_metadata,
                "hypergraph": h_model.training_metadata,
                "epdg_paired_post_prediction_outcomes": _revision_outcomes(
                    expected, h_without_epdg_prediction, h_prediction
                ),
                "local_structured_judge": judge.metadata(),
                "direct_llm_training_oof_calibration": direct_llm_calibration,
                "llm_training_oof_calibration": llm_calibration,
                "direct_llm_reviewable_candidate_pair_count": sum(
                    row is not None for row in direct_handoff.alternative_indices
                ),
                "reviewable_structured_conflict_count": sum(
                    row is not None for row in handoff.alternative_indices
                ),
            }
        )

    remediation_audit: Dict[str, Any]
    if llm_enabled:
        remediation_rows, remediation_audit = _run_remediation_advice(
            llm_client,
            remediation_jobs,
            LLM_REMEDIATION_BUDGET_PER_DATASET,
        )
        for row in collected["hyperdbdiag"]["rows"]:
            row["remediation"] = dict(remediation_rows[str(row["case_id"])])
    else:
        remediation_audit = {
            "status": "not_run_missing_explicit_llm_configuration",
            "authority": "post-diagnosis advice only; the root set is frozen",
            "eligible_case_count": 0,
            "budget": int(LLM_REMEDIATION_BUDGET_PER_DATASET),
            "sampled_case_count": 0,
            "diagnosis_mutation_count": 0,
        }

    methods = {
        name: _summary(
            collected[name]["expected"],
            collected[name]["predicted"],
            dataset.labels,
            collected[name]["rows"],
            collected[name]["seconds"],
        )
        for name in evaluated_method_ids
    }
    binary_exact = methods["ordinary_binary_graph"]["overall"]["exact_set_accuracy"]
    binary_f1 = methods["ordinary_binary_graph"]["overall"]["component_f1"]
    h_without_epdg_exact = methods["hypergraph_without_epdg"]["overall"][
        "exact_set_accuracy"
    ]
    h_without_epdg_f1 = methods["hypergraph_without_epdg"]["overall"][
        "component_f1"
    ]
    h_exact = methods["hypergraph"]["overall"]["exact_set_accuracy"]
    h_f1 = methods["hypergraph"]["overall"]["component_f1"]
    local_exact = methods["hypergraph_local_judge"]["overall"]["exact_set_accuracy"]
    local_f1 = methods["hypergraph_local_judge"]["overall"]["component_f1"]
    methods["ordinary_binary_graph"]["stage"] = {
        "estimand_status": "evaluated",
        "structure": "fixed pairwise KPI graph with independent root-wise classifiers",
        "candidate_set_decoder": False,
        "shares_fitted_state_with_hypergraph": False,
    }
    methods["hypergraph_without_epdg"]["stage"] = {
        "estimand_status": "evaluated",
        "structure": "signed-incidence higher-order hypergraph without EPDG path-prior fusion",
        "ablation_target": "EPDG root-aware training incidence and direct anonymous SQL/plan path prior",
    }
    epdg_outcomes = _revision_outcomes(
        collected["hypergraph"]["expected"],
        collected["hypergraph_without_epdg"]["predicted"],
        collected["hypergraph"]["predicted"],
    )
    epdg_decisive_pairs = epdg_outcomes["corrected"] + epdg_outcomes["harmed"]
    methods["hypergraph"]["stage"] = {
        "estimand_status": "evaluated",
        "structure": "signed-incidence higher-order hypergraph with EPDG path-prior fusion and joint root-set decoding",
        "incremental_exact_vs_binary_graph": h_exact - binary_exact,
        "incremental_component_f1_vs_binary_graph": h_f1 - binary_f1,
        "incremental_exact_vs_without_epdg": h_exact - h_without_epdg_exact,
        "incremental_component_f1_vs_without_epdg": h_f1 - h_without_epdg_f1,
        "paired_post_prediction_outcomes_vs_without_epdg": epdg_outcomes,
        "one_sided_exact_sign_test_p_value": (
            _binomial_greater_p_value(epdg_outcomes["corrected"], epdg_decisive_pairs)
            if epdg_decisive_pairs
            else 1.0
        ),
    }
    methods["hypergraph_local_judge"]["stage"] = {
        "estimand_status": "evaluated",
        "structure": "training-OOF-sized hypergraph pool plus local structured-evidence contraction",
        "incremental_exact_vs_hypergraph": local_exact - h_exact,
        "incremental_component_f1_vs_hypergraph": local_f1 - h_f1,
        "reviewable_structured_conflict_count": local_conflicts,
    }
    if llm_enabled:
        direct_values = methods["hypergraph_llm"]
        direct_exact = direct_values["overall"]["exact_set_accuracy"]
        direct_f1 = direct_values["overall"]["component_f1"]
        methods["hypergraph_llm"]["stage"] = {
            "estimand_status": "evaluated",
            "structure": (
                "hypergraph top candidate-pair handoff with anonymous evidence and relation-aware LLM arbitration; "
                "no local structured-judge decision"
            ),
            "incremental_exact_vs_hypergraph": direct_exact - h_exact,
            "incremental_component_f1_vs_hypergraph": direct_f1 - h_f1,
            "training_oof_calibration_enabled_outer_split_count": direct_llm_calibration_enabled_splits,
            "training_oof_calibration_review_requests": direct_llm_calibration_requests,
            "training_oof_calibration_model_responses": direct_llm_calibration_responses,
            "completed_review_count": direct_llm_completed_reviews,
            "candidate_selection_count": direct_llm_candidate_selections,
            "override_count": direct_llm_overrides,
            "review_requests": direct_llm_review_requests,
            "model_responses": direct_llm_model_responses,
            "failed_review_count": direct_llm_failed_reviews,
            "audit_only_skipped_count": direct_llm_audit_only_skipped,
            "llm_configuration": llm_configuration,
        }
    local_outcomes = _revision_outcomes(
        collected["hypergraph_local_judge"]["expected"],
        collected["hypergraph"]["predicted"],
        collected["hypergraph_local_judge"]["predicted"],
    )
    if llm_enabled:
        llm_values = methods["hyperdbdiag"]
        llm_exact = llm_values["overall"]["exact_set_accuracy"]
        llm_f1 = llm_values["overall"]["component_f1"]
        llm_outcomes_vs_local = _revision_outcomes(
            collected["hyperdbdiag"]["expected"],
            collected["hypergraph_local_judge"]["predicted"],
            collected["hyperdbdiag"]["predicted"],
        )
        llm_outcomes_vs_hypergraph = _revision_outcomes(
            collected["hyperdbdiag"]["expected"],
            collected["hypergraph"]["predicted"],
            collected["hyperdbdiag"]["predicted"],
        )
        direct_llm_outcomes_vs_hypergraph = _revision_outcomes(
            collected["hypergraph_llm"]["expected"],
            collected["hypergraph"]["predicted"],
            collected["hypergraph_llm"]["predicted"],
        )
        llm_values["stage"] = {
            "estimand_status": "evaluated",
            "structure": (
                "broad hypergraph retrieval, local structured-evidence contraction, and symmetric relation-aware "
                "LLM arbitration enabled only for ECSA relations with stable positive training-fold OOF value"
            ),
            "incremental_exact_vs_hypergraph": llm_exact - h_exact,
            "incremental_component_f1_vs_hypergraph": llm_f1 - h_f1,
            "incremental_exact_vs_local_judge": llm_exact - local_exact,
            "incremental_component_f1_vs_local_judge": llm_f1 - local_f1,
            "intervention_status": (
                "candidate_override"
                if llm_overrides
                else "reviewed_without_override"
                if llm_completed_reviews
                else "audit_only_no_stable_oof"
                if llm_audit_only_skipped
                else "no_reviewable_structured_conflicts"
            ),
            "completed_review_count": llm_completed_reviews,
            "candidate_selection_count": llm_candidate_selections,
            "override_count": llm_overrides,
            "training_oof_calibration_enabled_outer_split_count": llm_calibration_enabled_splits,
            "training_oof_calibration_review_requests": llm_calibration_requests,
            "training_oof_calibration_model_responses": llm_calibration_responses,
            "relation_type_counts": dict(sorted(review_relation_types.items())),
            "recommendation_counts": dict(sorted(review_recommendations.items())),
            "post_diagnosis_remediation": remediation_audit,
            "llm_configuration": llm_configuration,
        }
    else:
        methods["hypergraph_llm"] = {
            "stage": {
                "estimand_status": "not_run_missing_explicit_llm_configuration",
                "fallback_behavior": "the EPDG-grounded hypergraph result remains the reproducible result",
                "incremental_exact_vs_hypergraph": None,
                "incremental_component_f1_vs_hypergraph": None,
                "llm_configuration": llm_configuration,
            }
        }
        methods["hyperdbdiag"] = {
            "stage": {
                "estimand_status": "not_run_missing_explicit_llm_configuration",
                "fallback_behavior": "the local structured-judge result remains the reproducible EPDG-grounded metric result",
                "incremental_exact_vs_hypergraph": None,
                "incremental_component_f1_vs_hypergraph": None,
                "incremental_exact_vs_local_judge": None,
                "incremental_component_f1_vs_local_judge": None,
                "post_diagnosis_remediation": remediation_audit,
                "llm_configuration": llm_configuration,
            }
        }
        llm_outcomes_vs_local = None
        llm_outcomes_vs_hypergraph = None
        direct_llm_outcomes_vs_hypergraph = None
    return {
        "dataset": {"name": dataset.name, **dict(dataset.metadata)},
        "protocol": {
            "feature_count": dataset.feature_count,
            "label_inventory": list(dataset.labels),
            "outer_split_count": len(dataset.splits),
            "hypergraph_training_selection": dict(dataset.training_selection),
            "epdg": {
                "root_metric_paths": "registered mechanism paths plus stable signed root-metric paths learned from the current outer training fold",
                "hypergraph_use": "registered root-metric paths weight training hyperedge incidence; held-out query incidence uses no labels",
                "cross_layer_paths": "only frozen direct root-to-anonymous SQL/plan-shape atoms; raw SQL, identifiers, plans, and evaluation labels are excluded",
                "fusion_selection": "joint grouped inner-OOF selection with at least 0.01 absolute Exact gain required for nonzero EPDG fusion",
                "unavailable_output": "no SQL-template or execution-operator binding is emitted",
            },
            "ordinary_graph_training_selection": "fixed shared baseline defaults inside each outer training split",
            "evidence_eligibility": dict(dataset.evidence_eligibility),
            "local_structured_support": (
                "training-OOF-sized H candidate pool, group-balanced training profiles, frozen mechanism cards, "
                "and current anonymous semantic evidence when available; no evaluation labels, raw SQL, plans, "
                "source IDs, scenarios, or provenance"
            ),
            "llm_structural_support": (
                "query KPI-time trajectory, anonymous SQL-shape atoms when available, outer-training root profiles, "
                "candidate-set topology, ECSA relation taxonomy, and frozen mechanism cards; local-judge runs may "
                "include two structured evidence cards, while the direct hypergraph-LLM run receives neutral "
                "candidate cards; no raw SQL, plans, source IDs, scenarios, case IDs, posterior values, "
                "candidate ranks, distances, or rarity"
            ),
            "ordinary_graph_hypergraph_boundary": (
                "same KPI rows and labels only; no shared candidate inventory, decoder, fitted "
                "parameters, or output threshold"
            ),
            "truth_access": "evaluation labels are not passed to fit, evidence construction, prediction, or LLM calibration",
        },
        "folds": folds,
        "methods": methods,
        "candidate_convergence": {
            name: _convergence_summary(values["rows"])
            for name, values in collected.items()
        },
        "intervention_audit": {
            "review_scope": "LLM review only for structured-evidence challengers retained from the training-OOF-sized candidate pool and enabled by a relation-specific positive training-OOF gate",
            "local_structured_conflicts": local_conflicts,
            "local_structured_changes": local_changes,
            "local_structured_statuses": dict(sorted(local_statuses.items())),
            "local_post_prediction_outcomes": local_outcomes,
            "llm_review_requests": llm_review_requests,
            "llm_model_responses": llm_model_responses,
            "llm_training_oof_calibration_requests": llm_calibration_requests,
            "llm_training_oof_calibration_responses": llm_calibration_responses,
            "llm_training_oof_calibration_enabled_outer_splits": llm_calibration_enabled_splits,
            "llm_completed_reviews": llm_completed_reviews,
            "llm_candidate_selections": llm_candidate_selections,
            "llm_overrides": llm_overrides,
            "llm_audit_only_skipped": llm_audit_only_skipped,
            "llm_failed_reviews": llm_failed_reviews,
            "llm_statuses": dict(sorted(review_statuses.items())),
            "llm_reason_codes": dict(sorted(review_reason_codes.items())),
            "llm_relation_types": dict(sorted(review_relation_types.items())),
            "llm_recommendations": dict(sorted(review_recommendations.items())),
            "llm_post_prediction_outcomes_vs_local_judge": llm_outcomes_vs_local,
            "llm_post_prediction_outcomes_vs_hypergraph": llm_outcomes_vs_hypergraph,
            "direct_llm_post_prediction_outcomes_vs_hypergraph": direct_llm_outcomes_vs_hypergraph,
            "direct_llm_review_requests": direct_llm_review_requests,
            "direct_llm_model_responses": direct_llm_model_responses,
            "direct_llm_training_oof_calibration_requests": direct_llm_calibration_requests,
            "direct_llm_training_oof_calibration_responses": direct_llm_calibration_responses,
            "direct_llm_training_oof_calibration_enabled_outer_splits": direct_llm_calibration_enabled_splits,
            "direct_llm_completed_reviews": direct_llm_completed_reviews,
            "direct_llm_candidate_selections": direct_llm_candidate_selections,
            "direct_llm_overrides": direct_llm_overrides,
            "direct_llm_audit_only_skipped": direct_llm_audit_only_skipped,
            "direct_llm_failed_reviews": direct_llm_failed_reviews,
            "post_diagnosis_remediation": remediation_audit,
            "llm_configuration": llm_configuration,
        },
    }


def _load_dbmags_semantic_rows(
    root: Path, case_ids: Sequence[str]
) -> Tuple[Dict[str, Tuple[SemanticObservation, ...]], Dict[str, Any]]:
    """Load integrity-bound anonymous SQL-shape observations."""

    rows = load_frozen_case_observations(root / "semantic_evidence.json", case_ids)
    return rows, {
        "status": "loaded_from_frozen_anonymous_semantic_evidence",
        "source": "semantic_evidence.json",
        "observation_atoms": list(
            semantic_inventory(item for rows_for_case in rows.values() for item in rows_for_case)
        ),
        "case_count": sum(bool(rows_for_case) for rows_for_case in rows.values()),
        "case_inventory_count": len(rows),
        "raw_sql_text_exposed_to_model": False,
        "source_metadata_exposed_to_model": False,
    }


def _load_dbmags(
    root: Path,
    mechanism_cards: Mapping[str, Mapping[str, Any]],
) -> AblationDataset:
    frozen = load_frozen_metric_dataset(root)
    semantic_rows, semantic_metadata = _load_dbmags_semantic_rows(root, frozen.case_ids)
    outer_groups = sorted(set(frozen.replicate_by_case.values()))

    splits: List[AblationSplit] = []
    for outer_group in outer_groups:
        train_ids = tuple(
            case_id
            for case_id in frozen.case_ids
            if frozen.replicate_by_case[case_id] != outer_group
        )
        eval_ids = tuple(
            case_id
            for case_id in frozen.case_ids
            if frozen.replicate_by_case[case_id] == outer_group
        )
        splits.append(
            AblationSplit(
                split_id=f"leave_replicate_index_out:{outer_group}",
                train_ids=train_ids,
                eval_ids=eval_ids,
                train_x=np.asarray([frozen.features[case_id] for case_id in train_ids]),
                eval_x=np.asarray([frozen.features[case_id] for case_id in eval_ids]),
                train_labels=tuple(frozen.labels_by_case[case_id] for case_id in train_ids),
                eval_labels=tuple(frozen.labels_by_case[case_id] for case_id in eval_ids),
                oof_groups=tuple(frozen.replicate_by_case[case_id] for case_id in train_ids),
                train_semantic=tuple(semantic_rows.get(case_id, ()) for case_id in train_ids),
                eval_semantic=tuple(semantic_rows.get(case_id, ()) for case_id in eval_ids),
            )
        )
    source_audit = json.loads((root / "source_audit.json").read_text(encoding="utf-8"))
    interaction = dict(source_audit.get("interaction_audit_summary") or {})
    return AblationDataset(
        name="dbmags_sql_interaction_subset",
        labels=tuple(frozen.labels),
        feature_count=frozen.feature_count,
        splits=tuple(splits),
        activation_threshold=1.0,
        decoder_criterion="gini",
        decoder_n_estimators=500,
        metadata={
            "sample_count": len(frozen.case_ids),
            "scenario_count": frozen.scenario_count,
            "physical_collection_block_count": frozen.physical_collection_block_count,
            "outer_split": "leave_one_replicate_index_out",
            "replicate_index_count": frozen.replicate_count,
            "covered_atomic_root_count": len(frozen.labels),
            "official_atomic_root_count": 18,
            "interaction_positive_pair_cases": interaction.get("interaction_positive"),
            "interaction_negative_or_incomplete_pair_cases": interaction.get(
                "interaction_negative_or_incomplete"
            ),
            "completeness_status": "complete_collected_SQL_interaction_cohort_but_not_full_DBMAGS",
            "semantic_evidence": semantic_metadata,
        },
        training_selection={
            "decoder_criterion": "gini",
            "decoder_n_estimators": 500,
            "activation_threshold": 1.0,
            "selection_policy": "registered shared core defaults; no outer evaluation result used",
        },
        evidence_eligibility={
            "epdg_predictor_support": "registered root-metric paths and direct anonymous runtime SQL-shape atoms",
            "llm_support": "KPI-time trajectory, training-fold root profiles, anonymous SQL-shape atoms, and frozen mechanism cards",
            "excluded": "raw SQL text, plans, source case IDs, scenarios, injection records, and evaluation labels",
            "reason": "EPDG may use only registered direct anonymous SQL-shape paths; local and LLM review use the same bounded atoms plus training-grounded evidence.",
            "semantic_observation_case_count": semantic_metadata.get("case_count", 0),
        },
        feature_names=_dbmags_feature_names(),
        mechanism_cards=_cards_for_labels(mechanism_cards, frozen.labels),
        epdg_path_edges=_epdg_path_edges(
            _dbmags_feature_names(), frozen.labels, mechanism_cards
        ),
    )


def run(
    dbmags_root: Path = DEFAULT_DBMAGS_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    mechanism_cards, mechanism_card_metadata = _load_root_mechanism_cards()
    requested = (_load_dbmags(Path(dbmags_root), mechanism_cards),)
    report = {
        "protocol": {
            "name": "hyperdbdiag_dbmags_retrieve_contract_arbitrate_v10_epdg",
            "seed": int(seed),
            "stage_order": [
                "ordinary_binary_graph",
                "hypergraph_without_epdg",
                "hypergraph",
                "hypergraph_local_judge",
                "hypergraph_llm",
                "hyperdbdiag",
            ],
            "no_posthoc_data_filtering": True,
            "llm_authority": (
                "may only choose between a hypergraph fallback and one candidate challenger; the direct ablation "
                "uses the training-OOF-sized top pair without local-judge decisions, while the complete method "
                "uses the locally retained structured-evidence pair; both require relation-specific stable "
                "positive grouped-OOF calibration and fail closed otherwise"
            ),
            "root_mechanism_card_registry": mechanism_card_metadata,
        },
        "datasets": {item.name: _run_dataset(item, seed) for item in requested},
    }
    _write_json(Path(output_path), report)
    return report


def _compact(report: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for dataset_name, dataset in report["datasets"].items():
        result[dataset_name] = {}
        for name, values in dataset["methods"].items():
            overall = values.get("overall")
            evaluated = values["stage"]["estimand_status"] == "evaluated"
            result[dataset_name][name] = {
                "status": values["stage"]["estimand_status"],
                "exact": None if overall is None or not evaluated else overall["exact_set_accuracy"],
                "component_f1": None if overall is None or not evaluated else overall["component_f1"],
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbmags-root", type=Path, default=DEFAULT_DBMAGS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(
        json.dumps(
            _compact(
                run(
                    dbmags_root=args.dbmags_root,
                    output_path=args.output,
                    seed=args.seed,
                )
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
