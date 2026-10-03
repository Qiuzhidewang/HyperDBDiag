import unittest


from ecsa_arbitration import build_packets, validate_response
from hyperdbdiag_pipeline import EvidenceItem, LLMReviewEvidence


class IsolationTests(unittest.TestCase):
    def fixture(self):
        candidates = [(0,), (1,), (0, 1)]
        labels = ['a', 'b']
        bundle = LLMReviewEvidence(
            query_items=(EvidenceItem('query', 'query_metric_trajectory', 'observed metrics'),),
            training_items=(EvidenceItem('profile-a', 'training_root_profile', 'training only', ('a',)),
                            EvidenceItem('profile-b', 'training_root_profile', 'training only', ('b',))),
            mechanism_items=(EvidenceItem('mechanism-a', 'mechanism_card', 'mechanism a', ('a',)),
                             EvidenceItem('mechanism-b', 'mechanism_card', 'mechanism b', ('b',))),
            candidate_items=tuple(EvidenceItem(f'candidate-card:candidate-{i:03d}', 'candidate_evidence_card',
                                               f'Candidate candidate-{i:03d} evidence', tuple(labels[j] for j in c))
                                  for i, c in enumerate(candidates)),
        )
        return build_packets(bundle, [0, 1, 2], [0, 1], candidates, labels)

    def test_full_pool_contains_third_candidate_and_shared_cards_are_identical(self):
        packets = self.fixture()
        full, filtered = packets['llm_full_pool'], packets['ecsa_filtered_pool']
        self.assertEqual(len(full['candidate_records']), 3)
        self.assertEqual(len(filtered['candidate_records']), 2)
        for family in ['query_trajectory', 'root_profiles', 'mechanism_cards', 'semantic_observations']:
            self.assertEqual(full['evidence'][family], filtered['evidence'][family])
        original = {r['evidence_id']: r for r in full['evidence']['candidate_evidence_cards']}
        for card in filtered['evidence']['candidate_evidence_cards']:
            self.assertEqual(card, original[card['evidence_id']])



    def decision(self, packet, chosen='C03', other='C01'):
        return dict(action='SELECT_CANDIDATE', candidate_id=chosen, comparison_candidate_id=other,
                    evidence_ids=['query', f'candidate-card-{chosen}', f'candidate-card-{other}', 'profile-b'],
                    reason_code='evidence_closure', relation_type='CONFLICT', recommendation='REPLACE_CONFLICTING_ROOTS')

    def test_rejects_out_of_pool_selection_and_unknown_citations(self):
        packet = self.fixture()['ecsa_filtered_pool']
        response = self.decision(packet, chosen='invented')
        with self.assertRaises(ValueError): validate_response(response, packet)
        response = self.decision(packet)
        response['evidence_ids'].append('test-ground-truth')
        with self.assertRaises(ValueError): validate_response(response, packet)

    def test_rejects_relation_unsupported_by_topology(self):
        packet = self.fixture()['ecsa_filtered_pool']
        response = self.decision(packet)
        response.update(relation_type='REDUNDANT', recommendation='PRUNE_REDUNDANT_ROOTS')
        with self.assertRaises(ValueError): validate_response(response, packet)

    def test_valid_decision_and_strict_abstention(self):
        packet = self.fixture()['ecsa_filtered_pool']
        self.assertEqual(validate_response(self.decision(packet), packet)['candidate_id'], 'C03')
        response = dict(action='ABSTAIN', candidate_id=None, comparison_candidate_id=None,
                        evidence_ids=[], reason_code='insufficient_evidence', relation_type='UNRESOLVED', recommendation='NO_CHANGE')
        self.assertEqual(validate_response(response, packet)['action'], 'ABSTAIN')
        response['candidate_id'] = 'C03'
        with self.assertRaises(ValueError): validate_response(response, packet)


if __name__ == '__main__':
    unittest.main()
