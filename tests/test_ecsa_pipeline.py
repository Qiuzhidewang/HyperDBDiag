import copy
import unittest

import test_ecsa_arbitration as fixtures
from ecsa_arbitration import digest
from ecsa_pipeline import FrozenDecisionStore, resolve_response


def abstention():
    return dict(action='ABSTAIN', candidate_id=None, comparison_candidate_id=None,
                evidence_ids=[], reason_code='insufficient_evidence',
                relation_type='UNRESOLVED', recommendation='NO_CHANGE')


class ECSAPipelineTests(unittest.TestCase):
    def fixture(self, status='accepted_abstain', draw='primary'):
        packet = fixtures.IsolationTests().fixture()['ecsa_filtered_pool']
        key = digest(packet)
        job = dict(job_id='sample:filtered:' + draw, case_id='sample', fold='fold',
                   arm='conservative_filtered_pool', draw=draw, packet=packet,
                   packet_sha256=key, fallback=['a'])
        result = {k: v for k, v in job.items() if k not in {'packet', 'fallback'}}
        result.update(status=status, usage={}, decision=abstention())
        return job, result

    def test_complete_method_falls_back_to_local_not_upstream(self):
        job, result = self.fixture()
        self.assertEqual(resolve_response(job['packet'], result, ['b']), ['b'])
        self.assertEqual(resolve_response(job['packet'], result, ['a']), ['a'])

    def test_selection_is_mapped_from_validated_candidate_not_saved_labels(self):
        job, result = self.fixture('accepted_selection')
        result['decision'] = fixtures.IsolationTests().decision(job['packet'])
        result['predicted_labels'] = ['invented']
        self.assertEqual(resolve_response(job['packet'], result, ['a']), ['b'])

    def test_status_and_decision_must_agree(self):
        job, result = self.fixture('accepted_selection')
        with self.assertRaisesRegex(ValueError, 'disagree'):
            resolve_response(job['packet'], result, ['a'])

    def test_invalid_fallback_is_rejected(self):
        job, result = self.fixture()
        with self.assertRaisesRegex(ValueError, 'not in'):
            resolve_response(job['packet'], result, ['invented'])

    def test_repeat_is_never_substituted_for_failed_primary(self):
        primary, failed = self.fixture('fallback_transport')
        repeat, better = self.fixture('accepted_selection', 'repeat')
        better['decision'] = fixtures.IsolationTests().decision(repeat['packet'])
        store = FrozenDecisionStore([primary, repeat], [failed, better])
        self.assertEqual(store(primary['packet'])['status'], 'fallback_transport')
        with self.assertRaisesRegex(ValueError, 'no primary'):
            FrozenDecisionStore([repeat], [better])

    def test_changed_evidence_requires_new_inference(self):
        job, result = self.fixture()
        store = FrozenDecisionStore([job], [result])
        modified = copy.deepcopy(job['packet'])
        modified['evidence']['query_trajectory'][0]['summary'] = 'different observation'
        with self.assertRaisesRegex(ValueError, 'new inference'):
            store(modified)

    def test_conflicting_draws_for_same_primary_request_are_rejected(self):
        job, result = self.fixture()
        second_job = {**job, 'job_id': 'second:filtered:primary'}
        second_result = {**result, 'job_id': second_job['job_id'], 'status': 'fallback_transport'}
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            FrozenDecisionStore([job, second_job], [result, second_result])

    def test_duplicate_response_ids_are_rejected(self):
        job, result = self.fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            FrozenDecisionStore([job], [result, result])


if __name__ == '__main__':
    unittest.main()
