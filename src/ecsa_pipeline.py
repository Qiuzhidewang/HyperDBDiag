"""The calibrated local discriminator and variable-width ECSA inference path.

Prediction accepts observations and a response provider, never evaluation labels
or case identifiers. The frozen provider replays primary draws, including failures;
it cannot substitute a repeat draw or silently call a different model.
"""
from __future__ import annotations

from typing import Any, Mapping

from ecsa_arbitration import build_packets, digest, validate_response
from hyperdbdiag_ablation import _llm_evidence


def resolve_response(packet, response, fallback):
    status = response['status']
    if status in {'accepted_selection', 'accepted_abstain'}:
        decision = validate_response(response['decision'], packet)
        expected = 'SELECT_CANDIDATE' if status == 'accepted_selection' else 'ABSTAIN'
        if decision['action'] != expected:
            raise ValueError('response status and decision disagree')
        if status == 'accepted_selection':
            return next(list(c['root_labels']) for c in packet['candidate_records']
                        if c['candidate_id'] == decision['candidate_id'])
    elif status not in {'fallback_transport', 'fallback_invalid_response'}:
        raise ValueError(f'unsupported primary response status: {status}')
    if not any(set(fallback) == set(c['root_labels']) for c in packet['candidate_records']):
        raise ValueError('fallback is not in the supplied pool')
    return list(fallback)


class FrozenDecisionStore:
    """Exact-input primary-draw replay; missing inputs are errors, not fallbacks."""

    def __init__(self, jobs, decisions):
        self.responses = {}
        self.lookups = 0
        by_job = {r['job_id']: r for r in decisions}
        if len(by_job) != len(decisions):
            raise ValueError('duplicate saved decision ID')
        self.expected_packets = {}
        for job in jobs:
            if job['draw'] != 'primary' or job['arm'] not in {
                'llm_full_pool', 'conservative_filtered_pool'
            }:
                continue
            packet = job['packet']
            key = digest(packet)
            result = by_job[job['job_id']]
            if key != job['packet_sha256'] or key != result['packet_sha256']:
                raise ValueError('saved packet or response fingerprint differs')
            if result['draw'] != 'primary' or result['arm'] != job['arm']:
                raise ValueError('response belongs to another draw or arm')
            resolve_response(packet, result, job['fallback'])
            signature = {k: result.get(k) for k in ('status', 'decision', 'usage')}
            if key in self.responses:
                prior = {k: self.responses[key].get(k) for k in signature}
                if prior != signature:
                    raise ValueError('identical primary inputs have conflicting saved responses')
            else:
                self.responses[key] = result
                self.expected_packets[key] = packet
        if not self.responses:
            raise ValueError('no primary decisions available')

    def __call__(self, packet):
        key = digest(packet)
        if key not in self.responses or packet != self.expected_packets[key]:
            raise ValueError('no identical frozen primary request; new inference is required')
        self.lookups += 1
        return self.responses[key]


class ConservativeECSAPipeline:
    """One fitted inference path shared by main results and component ablations."""

    def __init__(self, model, judge, labels, feature_names, mechanism_cards,
                 train_x, train_labels, train_groups, train_semantic,
                 require_direct_evidence=False):
        self.model = model
        self.judge = judge
        self.labels = tuple(labels)
        self.feature_names = tuple(feature_names)
        self.cards = mechanism_cards
        self.training = (train_x, train_labels, train_groups, train_semantic)
        self.require_direct_evidence = require_direct_evidence

    def predict_details(self, vectors, observations, response_provider):
        pools = self.model.predict_candidate_pool_indices(vectors, observations)
        if len(vectors) != len(observations):
            raise ValueError('observation counts differ')
        candidates = self.model.candidates
        ids = tuple(f'candidate-{i:03d}' for i in range(len(candidates)))
        cards = [tuple(self.judge._card(ids[int(i)], candidates[int(i)], q, obs, self.cards)
                       for i in pool) for q, obs, pool in zip(vectors, observations, pools)]
        train_x, train_labels, groups, train_semantic = self.training
        bundles = _llm_evidence(vectors, train_x, train_labels, groups, self.labels,
                               self.feature_names, self.cards, cards, observations,
                               train_semantic, self.require_direct_evidence)
        details = []
        for q, obs, pool, bundle in zip(vectors, observations, pools, bundles):
            decision = self.judge.decide_pool(candidates, pool, q, obs, self.cards)
            labels_for = lambda i: [self.labels[j] for j in candidates[int(i)]]
            baseline = labels_for(pool[0])
            local = labels_for(decision.local_candidate)
            packets = build_packets(bundle, list(map(int, pool)),
                                    list(decision.retained_candidates), candidates, self.labels)
            full = packets['llm_full_pool']
            filtered = packets['ecsa_filtered_pool']
            full_response = response_provider(full)
            filtered_response = response_provider(filtered)
            row = {
                'baseline': baseline,
                'local_prediction': local,
                'llm_without_local_prediction': resolve_response(full, full_response, baseline),
                'predicted_labels': resolve_response(filtered, filtered_response, local),
                'filtered_common_fallback_prediction': resolve_response(filtered, filtered_response, baseline),
                'full_pool': [labels_for(i) for i in pool],
                'retained_pool': [labels_for(i) for i in decision.retained_candidates],
                'local_status': decision.status,
                'full_packet_sha256': digest(full),
                'filtered_packet_sha256': digest(filtered),
                'full_llm_status': full_response['status'],
                'filtered_llm_status': filtered_response['status'],
                'full_usage': full_response.get('usage', {}),
                'filtered_usage': filtered_response.get('usage', {}),
            }
            details.append(row)
        return details
