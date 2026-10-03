"""Protect sensitivity input reuse and outage handling from biased outcomes."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parameter_sensitivity as study
from test_ecsa_pipeline import ECSAPipelineTests


class SensitivityTests(unittest.TestCase):
    def test_fusion_endpoints_and_candidate_order(self):
        branches = {'raw': np.array([[.2, .6, .2]]),
                    'h_residual': np.array([[.7, .1, .2]]),
                    'epdg_prior': np.array([[.1, .2, .7]])}
        for alpha, beta, expected in [(1, 0, 'raw'), (0, 0, 'h_residual'), (.4, 1, 'epdg_prior')]:
            np.testing.assert_array_equal(study.fuse(branches, alpha, beta), branches[expected])
        np.testing.assert_array_equal(study.candidate_pools(branches, dict(alpha=1, beta=0, K=2)), [[1, 0]])

    def test_changed_input_requires_new_draw_but_failure_is_reused(self):
        job, result = ECSAPipelineTests().fixture('fallback_transport')
        store = study.FrozenDecisionStore([job], [result])
        packet = job['packet']
        changed = copy.deepcopy(packet)
        changed['evidence']['query_trajectory'][0]['summary'] = 'different evidence'
        requests = {study.digest(p): p for p in [packet, changed]}
        with tempfile.TemporaryDirectory() as directory:
            responses = study.saved_responses(Path(directory), requests, store)
        self.assertEqual(set(responses), {study.digest(packet)})
        self.assertEqual(responses[study.digest(packet)]['status'], 'fallback_transport')

    def test_same_abstention_applies_actual_configuration_local_fallback(self):
        job, response = ECSAPipelineTests().fixture()
        self.assertEqual(study.response_labels(job['packet'], response, ['a']), ['a'])
        self.assertEqual(study.response_labels(job['packet'], response, ['b']), ['b'])

    def test_singleton_shortcut_requires_one_candidate_and_matching_fallback(self):
        response = {'status': 'analytic_single_candidate'}
        packet = {'candidate_records': [{'root_labels': ['a']}]}
        self.assertEqual(study.response_labels(packet, response, ['a']), ['a'])
        with self.assertRaises(ValueError):
            study.response_labels(packet, response, ['b'])
        packet['candidate_records'].append({'root_labels': ['b']})
        with self.assertRaises(ValueError):
            study.response_labels(packet, response, ['a'])

    def test_outage_stops_scheduling_and_records_all_completed_jobs(self):
        recorded, called = [], []
        def fake(job, client):
            called.append(job)
            return {'status': 'fallback_transport', 'id': job}
        with patch.object(study, 'execute_job', fake):
            with self.assertRaisesRegex(RuntimeError, 'outage'):
                study.run_primary_jobs(list(range(100)), None, recorded.append, workers=3, outage_limit=2)
        self.assertEqual(set(called), {r['id'] for r in recorded})
        self.assertLessEqual(len(called), 4)

    def test_fatal_error_preserves_other_inflight_results(self):
        recorded, called = [], []
        def fake(job, client):
            called.append(job)
            if job == 0:
                raise PermissionError('test credential outage')
            return {'status': 'accepted_abstain', 'id': job}
        with patch.object(study, 'execute_job', fake):
            with self.assertRaises(PermissionError):
                study.run_primary_jobs(list(range(100)), None, recorded.append, workers=3)
        self.assertEqual(set(called) - {0}, {r['id'] for r in recorded})
        self.assertLessEqual(len(called), 3)


if __name__ == '__main__':
    unittest.main()
