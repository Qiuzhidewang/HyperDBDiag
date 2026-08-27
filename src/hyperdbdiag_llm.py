"""Strict LLM task contracts and post-diagnosis remediation advice.

Diagnosis review and remediation share one transport client, but have separate
schemas and authority.  Remediation is observational only and cannot alter the
root set produced by HyperDBDiag.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from hyperdbdiag_pipeline import EvidenceItem, LLMClient


_CANDIDATE_REVIEW_TASKS = frozenset(
    {
        "candidate_bound_structured_evidence_review",
        "symmetric_candidate_evidence_review",
    }
)
REMEDIATION_TASK = "post_diagnosis_remediation_advice"
REMEDIATION_QUALITY_TASK = "blinded_remediation_quality_review"

_CANDIDATE_REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action",
        "candidate_id",
        "evidence_ids",
        "reason_code",
        "relation_type",
        "recommendation",
    ],
    "properties": {
        "action": {"type": "string", "enum": ["SELECT_CANDIDATE", "ABSTAIN"]},
        "candidate_id": {
            "anyOf": [
                {"type": "string", "enum": ["A", "B"]},
                {"type": "null"},
            ]
        },
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 7,
        },
        "reason_code": {
            "type": "string",
            "enum": [
                "evidence_closure",
                "candidate_contradicted",
                "insufficient_evidence",
            ],
        },
        "relation_type": {
            "type": "string",
            "enum": [
                "COMPLEMENTARY",
                "COVERAGE",
                "REDUNDANT",
                "CONFLICT",
                "UNRESOLVED",
            ],
        },
        "recommendation": {
            "type": "string",
            "enum": [
                "RETAIN_COMPLEMENTARY_ROOTS",
                "PRUNE_COVERED_ROOTS",
                "PRUNE_REDUNDANT_ROOTS",
                "REPLACE_CONFLICTING_ROOTS",
                "NO_CHANGE",
            ],
        },
    },
}

_REMEDIATION_ACTIONS = (
    "INSPECT_QUERY_PLAN",
    "REVIEW_INDEX_OR_PREDICATE",
    "REVIEW_LOCK_CHAIN",
    "REVIEW_QUERY_SHAPE",
    "NO_ACTIONABLE_RECOMMENDATION",
)
_REMEDIATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action_type",
        "steps",
        "preconditions",
        "verification",
        "rollback",
        "evidence_ids",
        "risk_level",
    ],
    "properties": {
        "action_type": {"type": "string", "enum": list(_REMEDIATION_ACTIONS)},
        "steps": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "preconditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "verification": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "rollback": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        },
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM"]},
    },
}

_QUALITY_ISSUES = (
    "NONE",
    "UNSUPPORTED_FACT",
    "ROOT_ACTION_MISMATCH",
    "GENERIC_OR_VAGUE",
    "UNVERIFIABLE_STEP",
    "ROLLBACK_INADEQUATE",
)
_REMEDIATION_QUALITY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_grounding_score",
        "root_relevance_score",
        "actionability_score",
        "verification_quality_score",
        "issue_codes",
        "confidence",
    ],
    "properties": {
        "evidence_grounding_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "root_relevance_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "actionability_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "verification_quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "issue_codes": {
            "type": "array",
            "items": {"type": "string", "enum": list(_QUALITY_ISSUES)},
            "maxItems": 5,
        },
        "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
    },
}

_CANDIDATE_REVIEW_INSTRUCTIONS = (
    "Act as the ECSA relation arbiter, never as a root generator. Classify candidate A versus B as COMPLEMENTARY, "
    "COVERAGE, REDUNDANT, CONFLICT, or UNRESOLVED, then select only A or B or abstain. Use the supplied set_relation "
    "as a hard topology constraint: COVERAGE or REDUNDANT requires one complete candidate root set to be a strict "
    "subset of the other; for PARTIAL_OVERLAP or DISJOINT candidates, never call either of those relations. In those "
    "non-subset cases, use CONFLICT only when current evidence supports one incompatible alternative, otherwise use "
    "UNRESOLVED and abstain. Distinct roots with independent current evidence are complementary only when one candidate "
    "is the strict superset containing both root sets. A root without independent evidence may be covered or redundant "
    "only under the strict-subset topology. Valid roots may coexist. For PARTIAL_OVERLAP or DISJOINT candidates, a "
    "direct candidate-discriminating semantic observation for a root present only in one candidate is sufficient current "
    "evidence for CONFLICT only when the competing candidate card lacks its registered direct atom; do not apply this "
    "to contextual atoms or roots with no registered direct observable. Treat anonymous query-shape atoms as current "
    "syntax facts, KPI trajectories as measurements, profiles as training associations, and mechanism cards as generic "
    "priors. Mechanisms being able to coexist is not evidence that they co-occur in the current observation. Do not "
    "confuse a strong cross-group training-associated semantic atom with causal proof; it may support a choice only "
    "when the cited root profile reproduces that association in every training group and the atom distinguishes the "
    "two current candidates. Never select a candidate that excludes the root supported by such a cited atom. Do not "
    "select a strict superset merely because it is compatible: every added root needs candidate-independent current "
    "evidence, or a stable root-specific training profile that matches the current query while the competing card has "
    "counterevidence. A generic mechanism card or one correlated KPI family alone cannot establish that closure. Never "
    "invent SQL, plans, identifiers, roots, scores, ranks, frequency, or rarity. A selection must cite "
    "the query, both candidate cards, one profile or mechanism card, and direct semantic evidence when required. "
    "UNRESOLVED means abstain. Return only the required JSON object."
)

_REMEDIATION_INSTRUCTIONS = (
    "Act as a low-authority post-diagnosis database remediation advisor. The selected root set is frozen: never "
    "add, remove, rename, rank, or reconsider roots. Use only the supplied current observations and generic "
    "mechanism cards. Recommend one non-mutating inspection or review action that can validate a later human "
    "decision. Do not invent SQL text, identifiers, execution-plan operators, lock owners, configuration values, "
    "or facts absent from the evidence. Do not emit executable commands or destructive, state-changing, restart, "
    "termination, or data-definition actions. An actionable answer must cite at least one current observation and "
    "one mechanism card and must state preconditions, verification, and rollback guidance. Abstain when the supplied "
    "facts do not support a bounded next step. Return only the required JSON object."
)

_REMEDIATION_QUALITY_INSTRUCTIONS = (
    "Act as a blinded database-advice quality reviewer. You are not judging whether the frozen root diagnosis is "
    "correct and you must not infer a hidden ground truth. Assess the recommendation only conditional on the supplied "
    "selected root set, current observations, and generic mechanism cards. Dataset name, method name, case identifier, "
    "and diagnosis labels from an evaluator are intentionally absent. Score four dimensions from 1 to 5: evidence "
    "grounding (5 means every material claim is supported, 3 means partly generic, 1 means contradicted or invented); "
    "root relevance (5 means the action directly addresses at least one selected root without contradicting the rest); "
    "actionability (5 means bounded, ordered, and usable by an operator, 3 means relevant but generic); and verification "
    "quality (5 means preconditions, success checks, and rollback are concrete and internally consistent). Flag every "
    "applicable registered issue. Use NONE only when no other issue applies. Do not reward verbosity and do not introduce "
    "new facts. Return only the required JSON object."
)


@dataclass(frozen=True)
class LLMTaskContract:
    schema_name: str
    schema: Mapping[str, Any]
    instructions: str
    max_output_tokens: int


def llm_task_contract(task: str) -> LLMTaskContract:
    """Return the registered transport contract for an LLM packet."""

    if task in _CANDIDATE_REVIEW_TASKS:
        return LLMTaskContract(
            "candidate_evidence_review",
            _CANDIDATE_REVIEW_SCHEMA,
            _CANDIDATE_REVIEW_INSTRUCTIONS,
            300,
        )
    if task == REMEDIATION_TASK:
        return LLMTaskContract(
            "post_diagnosis_remediation",
            _REMEDIATION_SCHEMA,
            _REMEDIATION_INSTRUCTIONS,
            700,
        )
    if task == REMEDIATION_QUALITY_TASK:
        return LLMTaskContract(
            "blinded_remediation_quality",
            _REMEDIATION_QUALITY_SCHEMA,
            _REMEDIATION_QUALITY_INSTRUCTIONS,
            400,
        )
    raise ValueError(f"unregistered LLM task: {task!r}")


@dataclass(frozen=True)
class RemediationResult:
    status: str
    called: bool
    response_count: int
    action_type: str = "NO_ACTIONABLE_RECOMMENDATION"
    steps: Tuple[str, ...] = ()
    preconditions: Tuple[str, ...] = ()
    verification: Tuple[str, ...] = ()
    rollback: Tuple[str, ...] = ()
    cited_evidence_ids: Tuple[str, ...] = ()
    risk_level: str = "LOW"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "called": self.called,
            "response_count": self.response_count,
            "action_type": self.action_type,
            "steps": list(self.steps),
            "preconditions": list(self.preconditions),
            "verification": list(self.verification),
            "rollback": list(self.rollback),
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "risk_level": self.risk_level,
        }


class PostDiagnosisLLMAdvisor:
    """Evidence-cited, non-mutating advice after diagnosis is complete."""

    _RESPONSE_KEYS = frozenset(_REMEDIATION_SCHEMA["required"])
    _ACTIONS = frozenset(_REMEDIATION_ACTIONS)
    _LIST_FIELDS = ("steps", "preconditions", "verification", "rollback", "evidence_ids")
    _MAX_ITEMS = {
        "steps": 3,
        "preconditions": 3,
        "verification": 3,
        "rollback": 2,
        "evidence_ids": 5,
    }
    _OBSERVATION_KINDS = frozenset({"query_metric_trajectory", "semantic_observation"})
    _UNSAFE = re.compile(
        r"(?:\bdrop\b|\bdelete\b|\btruncate\b|\bkill\b|\bshutdown\b|\breboot\b|"
        r"\brestart\b|\bterminate\b|\bpurge\b|\balter\b|\binsert\b|\bupdate\b|"
        r"\brm\s+-|\bsystemctl\b|\bservice\s+\S+\s+stop\b)",
        re.IGNORECASE,
    )
    _NEGATION = re.compile(r"\b(?:do not|don't|never|avoid)\b", re.IGNORECASE)
    _POSITIVE_PIVOT = re.compile(
        r"\b(?:but|however|instead|then|afterwards)\b", re.IGNORECASE
    )

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client

    @staticmethod
    def _parse_json(value: Mapping[str, Any] | str) -> Mapping[str, Any]:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1]).strip()
            value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("LLM remediation response must be a JSON object")
        return value

    @classmethod
    def _parse_response(cls, value: Mapping[str, Any] | str) -> Dict[str, Any]:
        result = dict(cls._parse_json(value))
        if set(result) != cls._RESPONSE_KEYS or result["action_type"] not in cls._ACTIONS:
            raise ValueError("LLM remediation response does not match the strict schema")
        if result["risk_level"] not in {"LOW", "MEDIUM"}:
            raise ValueError("invalid remediation risk level")
        for field in cls._LIST_FIELDS:
            values = result[field]
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item.strip() and len(item) <= 240 for item in values
            ):
                raise ValueError(f"{field} must contain short nonempty strings")
            if len(values) > cls._MAX_ITEMS[field]:
                raise ValueError(f"{field} exceeds its bounded item count")
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must not contain duplicates")
            result[field] = [item.strip() for item in values]
        return result

    @staticmethod
    def _validate_inputs(
        selected_roots: Sequence[str],
        mechanism_items: Sequence[EvidenceItem],
        observation_items: Sequence[EvidenceItem],
    ) -> Tuple[Tuple[str, ...], Tuple[EvidenceItem, ...], Tuple[EvidenceItem, ...]]:
        roots = tuple(selected_roots)
        mechanisms = tuple(mechanism_items)
        observations = tuple(observation_items)
        if not roots or any(not root for root in roots) or len(roots) != len(set(roots)):
            raise ValueError("selected roots must be nonempty and unique")
        if (
            len(mechanisms) != len(roots)
            or any(item.kind != "mechanism_card" or len(item.root_labels) != 1 for item in mechanisms)
            or {item.root_labels[0] for item in mechanisms} != set(roots)
        ):
            raise ValueError("remediation requires one mechanism card per selected root")
        if any(item.kind not in PostDiagnosisLLMAdvisor._OBSERVATION_KINDS for item in observations):
            raise ValueError("remediation observations must be current metric or semantic facts")
        if any(not set(item.root_labels) <= set(roots) for item in observations):
            raise ValueError("remediation observations cannot introduce an unselected root")
        ids = [item.evidence_id for item in mechanisms + observations]
        if len(ids) != len(set(ids)):
            raise ValueError("remediation evidence ids must be unique")
        return roots, mechanisms, observations

    @staticmethod
    def _packet(
        roots: Sequence[str],
        mechanisms: Sequence[EvidenceItem],
        observations: Sequence[EvidenceItem],
    ) -> Tuple[Dict[str, Any], frozenset[str], frozenset[str], frozenset[str]]:
        mechanism_records = [
            {
                **item.as_dict(),
                "evidence_id": f"mechanism-{index:02d}",
            }
            for index, item in enumerate(mechanisms, start=1)
        ]
        observation_records = [
            {
                **item.as_dict(),
                "evidence_id": f"observation-{index:02d}",
            }
            for index, item in enumerate(observations, start=1)
        ]
        mechanism_ids = frozenset(row["evidence_id"] for row in mechanism_records)
        observation_ids = frozenset(row["evidence_id"] for row in observation_records)
        payload = {
            "task": REMEDIATION_TASK,
            "selected_root_set": list(roots),
            "rules": {
                "diagnosis_is_frozen": True,
                "may_change_root_set": False,
                "may_execute_actions": False,
                "must_use_only_supplied_facts": True,
                "action_requires_mechanism_and_observation_citations": True,
                "destructive_or_state_changing_actions_forbidden": True,
                "may_abstain": True,
            },
            "evidence": {
                "mechanism_cards": mechanism_records,
                "current_observations": observation_records,
            },
        }
        return payload, mechanism_ids | observation_ids, mechanism_ids, observation_ids

    @classmethod
    def _unsafe_response(cls, response: Mapping[str, Any]) -> bool:
        text = "\n".join(
            item
            for field in ("steps", "preconditions", "verification", "rollback")
            for item in response[field]
        )
        # Safety prohibitions are expected in an advisory response. A clause
        # such as "do not create, drop, or alter indexes" is safe, while a
        # later positive clause remains independently subject to the filter.
        for clause in re.split(r"[.!?;\n]+", text):
            unsafe = cls._UNSAFE.search(clause)
            if unsafe is None:
                continue
            negation = cls._NEGATION.search(clause[: unsafe.start()])
            if negation is not None and cls._POSITIVE_PIVOT.search(
                clause[negation.end() :]
            ) is None:
                continue
            return True
        return False

    @staticmethod
    def _valid_actionable_response(
        response: Mapping[str, Any],
        valid_ids: frozenset[str],
        mechanism_ids: frozenset[str],
        observation_ids: frozenset[str],
    ) -> bool:
        cited = frozenset(response["evidence_ids"])
        return (
            1 <= len(response["steps"]) <= 3
            and 1 <= len(response["preconditions"]) <= 3
            and 1 <= len(response["verification"]) <= 3
            and 1 <= len(response["rollback"]) <= 2
            and bool(cited)
            and cited <= valid_ids
            and bool(cited & mechanism_ids)
            and bool(cited & observation_ids)
        )

    def advise(
        self,
        selected_roots: Sequence[str],
        mechanism_items: Sequence[EvidenceItem],
        observation_items: Sequence[EvidenceItem],
    ) -> RemediationResult:
        """Return bounded advice without changing or re-evaluating diagnosis."""

        roots, mechanisms, observations = self._validate_inputs(
            selected_roots, mechanism_items, observation_items
        )
        if self.client is None:
            return RemediationResult("disabled_no_client", False, 0)
        if not observations:
            return RemediationResult("skipped_no_current_observations", False, 0)
        payload, valid_ids, mechanism_ids, observation_ids = self._packet(
            roots, mechanisms, observations
        )
        try:
            response = self._parse_response(self.client(payload))
        except json.JSONDecodeError:
            return RemediationResult("fallback_invalid_json_recommendation", True, 1)
        except (KeyError, TypeError, ValueError):
            return RemediationResult("fallback_invalid_recommendation", True, 1)
        except Exception:
            return RemediationResult("fallback_transport_or_timeout", True, 0)

        if response["action_type"] == "NO_ACTIONABLE_RECOMMENDATION":
            empty = all(not response[field] for field in self._LIST_FIELDS)
            if not empty or response["risk_level"] != "LOW":
                return RemediationResult("fallback_invalid_recommendation", True, 1)
            return RemediationResult("accepted_no_action", True, 1)
        if self._unsafe_response(response):
            return RemediationResult("fallback_unsafe_recommendation", True, 1)
        if not self._valid_actionable_response(
            response, valid_ids, mechanism_ids, observation_ids
        ):
            return RemediationResult("fallback_invalid_recommendation", True, 1)
        return RemediationResult(
            status="accepted_recommendation",
            called=True,
            response_count=1,
            action_type=response["action_type"],
            steps=tuple(response["steps"]),
            preconditions=tuple(response["preconditions"]),
            verification=tuple(response["verification"]),
            rollback=tuple(response["rollback"]),
            cited_evidence_ids=tuple(sorted(response["evidence_ids"])),
            risk_level=response["risk_level"],
        )


@dataclass(frozen=True)
class RemediationQualityResult:
    status: str
    called: bool
    response_count: int
    quality_pass: bool = False
    evidence_grounding_score: Optional[int] = None
    root_relevance_score: Optional[int] = None
    actionability_score: Optional[int] = None
    verification_quality_score: Optional[int] = None
    issue_codes: Tuple[str, ...] = ()
    confidence: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        scores = (
            self.evidence_grounding_score,
            self.root_relevance_score,
            self.actionability_score,
            self.verification_quality_score,
        )
        return {
            "status": self.status,
            "called": self.called,
            "response_count": self.response_count,
            "quality_pass": self.quality_pass,
            "evidence_grounding_score": self.evidence_grounding_score,
            "root_relevance_score": self.root_relevance_score,
            "actionability_score": self.actionability_score,
            "verification_quality_score": self.verification_quality_score,
            "mean_score": (
                sum(int(score) for score in scores) / len(scores)
                if all(score is not None for score in scores)
                else None
            ),
            "issue_codes": list(self.issue_codes),
            "confidence": self.confidence,
        }


class BlindedRemediationQualityReviewer:
    """Score accepted advice without method identity, case ids, or truth."""

    _SCORE_FIELDS = (
        "evidence_grounding_score",
        "root_relevance_score",
        "actionability_score",
        "verification_quality_score",
    )
    _CRITICAL_ISSUES = frozenset({"UNSUPPORTED_FACT", "ROOT_ACTION_MISMATCH"})

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client

    @staticmethod
    def _packet(
        selected_roots: Sequence[str],
        mechanism_items: Sequence[EvidenceItem],
        observation_items: Sequence[EvidenceItem],
        recommendation: RemediationResult,
    ) -> Dict[str, Any]:
        if recommendation.status != "accepted_recommendation":
            raise ValueError("quality review requires an accepted recommendation")
        roots, mechanisms, observations = PostDiagnosisLLMAdvisor._validate_inputs(
            selected_roots, mechanism_items, observation_items
        )
        evidence_packet, _, _, _ = PostDiagnosisLLMAdvisor._packet(
            roots, mechanisms, observations
        )
        return {
            "task": REMEDIATION_QUALITY_TASK,
            "selected_root_set": list(roots),
            "evidence": evidence_packet["evidence"],
            "recommendation": {
                "action_type": recommendation.action_type,
                "steps": list(recommendation.steps),
                "preconditions": list(recommendation.preconditions),
                "verification": list(recommendation.verification),
                "rollback": list(recommendation.rollback),
                "evidence_ids": list(recommendation.cited_evidence_ids),
                "risk_level": recommendation.risk_level,
            },
        }

    @classmethod
    def _parse_response(cls, value: Mapping[str, Any] | str) -> Dict[str, Any]:
        result = dict(PostDiagnosisLLMAdvisor._parse_json(value))
        if set(result) != frozenset(_REMEDIATION_QUALITY_SCHEMA["required"]):
            raise ValueError("quality-review response does not match the strict schema")
        if any(
            not isinstance(result[field], int) or isinstance(result[field], bool)
            or not 1 <= result[field] <= 5
            for field in cls._SCORE_FIELDS
        ):
            raise ValueError("quality-review scores must be integers from one to five")
        issues = result["issue_codes"]
        if (
            not isinstance(issues, list)
            or not issues
            or len(issues) > 5
            or len(issues) != len(set(issues))
            or not set(issues) <= set(_QUALITY_ISSUES)
            or ("NONE" in issues and len(issues) != 1)
        ):
            raise ValueError("quality-review issue codes are invalid")
        if result["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("quality-review confidence is invalid")
        return result

    def review(
        self,
        selected_roots: Sequence[str],
        mechanism_items: Sequence[EvidenceItem],
        observation_items: Sequence[EvidenceItem],
        recommendation: RemediationResult,
    ) -> RemediationQualityResult:
        if self.client is None:
            return RemediationQualityResult("disabled_no_client", False, 0)
        payload = self._packet(
            selected_roots, mechanism_items, observation_items, recommendation
        )
        try:
            response = self._parse_response(self.client(payload))
        except json.JSONDecodeError:
            return RemediationQualityResult("fallback_invalid_json_quality_review", True, 1)
        except (KeyError, TypeError, ValueError):
            return RemediationQualityResult("fallback_invalid_quality_review", True, 1)
        except Exception:
            return RemediationQualityResult("fallback_quality_transport_or_timeout", True, 0)
        issues = tuple(response["issue_codes"])
        quality_pass = (
            response["evidence_grounding_score"] >= 4
            and response["root_relevance_score"] >= 4
            and response["actionability_score"] >= 3
            and response["verification_quality_score"] >= 3
            and not (set(issues) & self._CRITICAL_ISSUES)
        )
        return RemediationQualityResult(
            status="accepted_blinded_quality_review",
            called=True,
            response_count=1,
            quality_pass=quality_pass,
            evidence_grounding_score=response["evidence_grounding_score"],
            root_relevance_score=response["root_relevance_score"],
            actionability_score=response["actionability_score"],
            verification_quality_score=response["verification_quality_score"],
            issue_codes=issues,
            confidence=response["confidence"],
        )
