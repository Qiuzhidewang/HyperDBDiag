"""Fold-calibrated, variable-width handoff using the existing evidence rules.

No new diagnosis classifier and no root-specific rules. The historical judge
remains unchanged for reproducibility. This adapter changes two scalar decision
details: the minimum profile advantage for a local override, and the evidence
gap allowed when retaining candidates for arbitration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from structured_evidence_judge import StructuredEvidenceJudge


PROFILE_THRESHOLDS = (0.05, 0.10, 0.20, 0.40, 0.80, math.inf)
TARGET_POOL_RETENTION = 0.99
MIN_GROUP_RETENTION = 0.97


@dataclass(frozen=True)
class ConservativePoolDecision:
    upstream_candidate: int
    local_candidate: int
    retained_candidates: tuple[int, ...]
    status: str


def describe_pool(judge: StructuredEvidenceJudge, candidates, pool, query,
                  observations, mechanism_cards) -> dict[str, Any]:
    pool = [int(i) for i in pool]
    choice = judge._challenger_choice(candidates, pool, query, observations, mechanism_cards)
    scores = [judge._score(i, candidates[i], query, observations, mechanism_cards) for i in pool]
    by_index = {s.candidate_index: s for s in scores}
    challenger = None if choice is None else int(choice[0])
    margin = None if challenger is None else by_index[challenger].total - by_index[pool[0]].total
    return {'pool': pool, 'scores': {str(s.candidate_index): float(s.total) for s in scores},
            'challenger': challenger, 'reason': 'broad_pool_retained' if choice is None else choice[1],
            'profile_advantage': margin}


def local_choice(row: Mapping[str, Any], profile_threshold: float) -> int:
    challenger = row['challenger']
    if challenger is None:
        return int(row['pool'][0])
    if row['reason'] == 'direct_evidence_challenger':
        return int(challenger)
    if float(row['profile_advantage']) > profile_threshold:
        return int(challenger)
    return int(row['pool'][0])


def retain_pool(row: Mapping[str, Any], profile_threshold: float, evidence_gap: float) -> ConservativePoolDecision:
    pool = tuple(int(i) for i in row['pool'])
    local = local_choice(row, profile_threshold)
    if local == pool[0]:
        return ConservativePoolDecision(pool[0], local, pool, 'insufficient_advantage_keep_full_pool')
    scores = row['scores']
    best = max(float(scores[str(i)]) for i in pool)
    keep = {pool[0], local}
    keep.update(i for i in pool if best - float(scores[str(i)]) <= evidence_gap + 1e-12)
    retained = tuple(i for i in pool if i in keep)
    return ConservativePoolDecision(pool[0], local, retained,
        'calibrated_evidence_contraction' if len(retained) < len(pool) else 'ambiguous_evidence_keep_full_pool')


def calibrate(training_oof_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Only held-out predictions inside the current outer training fold.

    ``truth_candidate`` is permitted here, never in retain_pool or LLM packets.
    Select readout by mean OOF exact accuracy, ties favor fewer overrides. Pick
    the smallest evidence gap meeting empirical conditional candidate retention
    of 99% overall and 97% in every inner group. No formal coverage guarantee.
    """
    if not training_oof_rows:
        raise ValueError('empty training calibration rows')
    groups = sorted({str(r['group']) for r in training_oof_rows})
    if len(groups) < 3:
        raise ValueError('at least three independent training groups required')
    threshold_audit = []
    for threshold in PROFILE_THRESHOLDS:
        correct = sum(local_choice(r, threshold) == r['truth_candidate'] for r in training_oof_rows)
        changes = sum(local_choice(r, threshold) != r['pool'][0] for r in training_oof_rows)
        threshold_audit.append({'threshold': None if math.isinf(threshold) else threshold,
                                'correct': correct, 'n': len(training_oof_rows), 'overrides': changes})
    winner = max(range(len(PROFILE_THRESHOLDS)), key=lambda i:
                 (threshold_audit[i]['correct'], -threshold_audit[i]['overrides'], i))
    threshold = PROFILE_THRESHOLDS[winner]
    covered = [r for r in training_oof_rows if r['truth_candidate'] in r['pool']]
    if not covered:
        raise ValueError('no gold candidates in inner candidate pools')
    gaps = {0.0}
    for r in training_oof_rows:
        scores = r['scores']
        best = max(float(scores[str(i)]) for i in r['pool'])
        gaps.update(best - float(scores[str(i)]) for i in r['pool'])
    gap_audit = []
    selected_gap = math.inf
    for gap in sorted(gaps):
        kept = {g: [r['truth_candidate'] in retain_pool(r, threshold, gap).retained_candidates
                     for r in covered if str(r['group']) == g] for g in groups}
        overall = sum(sum(v) for v in kept.values()) / len(covered)
        group_rates = {g: float(np.mean(v)) for g, v in kept.items() if v}
        if overall + 1e-12 >= TARGET_POOL_RETENTION and all(v + 1e-12 >= MIN_GROUP_RETENTION for v in group_rates.values()):
            selected_gap = float(gap)
            gap_audit = {'conditional_retention': overall, 'by_group': group_rates,
                         'covered_cases': len(covered),
                         'mean_pool_size': float(np.mean([len(retain_pool(r,threshold,gap).retained_candidates)
                                                        for r in training_oof_rows]))}
            break
    return {'profile_threshold': None if math.isinf(threshold) else threshold,
            'evidence_gap': None if math.isinf(selected_gap) else selected_gap,
            'training_rows': len(training_oof_rows), 'training_groups': groups,
            'threshold_audit': threshold_audit, 'selected_pool_audit': gap_audit,
            'selection_uses_outer_test_labels': False}


class ConservativeEvidenceJudge(StructuredEvidenceJudge):
    """Opt-in local discriminator retaining every insufficiently separated candidate."""

    def __init__(self, calibration: Mapping[str, Any]):
        super().__init__()
        self.calibration = dict(calibration)
        self.profile_threshold = math.inf if calibration['profile_threshold'] is None else float(calibration['profile_threshold'])
        self.evidence_gap = math.inf if calibration['evidence_gap'] is None else float(calibration['evidence_gap'])

    def decide_pool(self, candidates, pool, query, observations=(), mechanism_cards=None):
        row = describe_pool(self, candidates, pool, query, observations, mechanism_cards or {})
        return retain_pool(row, self.profile_threshold, self.evidence_gap)

    def metadata(self):
        result = super().metadata()
        result.update(routing_rule='preserve the full pool without a calibrated evidence advantage; otherwise retain every candidate within the training-calibrated evidence gap',
                      llm_handoff='variable-width candidate pool with unchanged evidence cards',
                      decision_api='decide_pool', calibration=self.calibration)
        return result
