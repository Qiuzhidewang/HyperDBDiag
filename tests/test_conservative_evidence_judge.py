import unittest
from conservative_evidence_judge import calibrate, local_choice, retain_pool


def row(scores=(0.,1.,.8,-1.), challenger=1, reason='direct_evidence_challenger', truth=1, group='a'):
    return {'pool':[0,1,2,3], 'scores':{str(i):v for i,v in enumerate(scores)},
            'challenger':challenger, 'reason':reason,
            'profile_advantage':None if challenger is None else scores[challenger]-scores[0],
            'truth_candidate':truth,'group':group}


class ConservativeEvidenceJudgeTests(unittest.TestCase):
    def test_no_competitive_evidence_preserves_the_actual_full_pool(self):
        result=retain_pool(row(challenger=None),.05,0.)
        self.assertEqual(result.retained_candidates,(0,1,2,3))
        self.assertEqual(result.local_candidate,0)

    def test_ambiguous_third_candidate_survives_and_baseline_is_preserved(self):
        result=retain_pool(row(),.05,.25)
        self.assertEqual(result.retained_candidates,(0,1,2))
        self.assertEqual(result.local_candidate,1)

    def test_only_separated_candidates_are_removed(self):
        self.assertEqual(retain_pool(row(),.05,0.).retained_candidates,(0,1))
        self.assertEqual(retain_pool(row(),.05,3.).retained_candidates,(0,1,2,3))

    def test_profile_override_below_threshold_does_not_contract(self):
        r=row(scores=(0.,.1,.09,-1.),reason='profile_evidence_challenger')
        self.assertEqual(local_choice(r,.2),0)
        self.assertEqual(retain_pool(r,.2,0.).retained_candidates,(0,1,2,3))

    def test_prediction_does_not_consult_truth_fields(self):
        first=row(truth=1); second=row(truth=3)
        self.assertEqual(retain_pool(first,.05,.3),retain_pool(second,.05,.3))
        del second['truth_candidate'];del second['group']
        self.assertEqual(retain_pool(first,.05,.3),retain_pool(second,.05,.3))

    def test_training_calibration_protects_each_inner_group(self):
        rows=[row(truth=2,group=g) for g in ['a','b','c'] for _ in range(10)]
        result=calibrate(rows)
        self.assertAlmostEqual(result['evidence_gap'],.2)
        self.assertEqual(result['selected_pool_audit']['conditional_retention'],1.)
        self.assertTrue(all(v==1. for v in result['selected_pool_audit']['by_group'].values()))

    def test_training_selects_conservative_profile_readout_when_overrides_harm(self):
        rows=[row(scores=(0.,.1,.09,-1.),reason='profile_evidence_challenger',truth=0,group=g)
              for g in ['a','b','c'] for _ in range(5)]
        result=calibrate(rows)
        self.assertIsNone(result['profile_threshold'])
        self.assertEqual(result['selected_pool_audit']['mean_pool_size'],4.)


if __name__=='__main__': unittest.main()
