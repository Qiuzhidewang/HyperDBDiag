#!/usr/bin/env python3
"""Read-only independent checks of the sensitivity inputs and completed scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecsa_arbitration import digest, validate_response, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/parameter-sensitivity-20260930-v1')
    args = parser.parse_args()
    output = args.output
    protocol, prepared = read(output / 'protocol.json'), read(output / 'prepared.json')
    assert prepared['protocol_sha256'] == digest(protocol)
    assert hashlib.sha256((ROOT / 'scripts/run_parameter_sensitivity.py').read_bytes()).hexdigest() == protocol['script_sha256']
    for name, sha in protocol['protected_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    assert len(prepared['folds']) == 6 and len(prepared['rows']) == 14520
    requests = prepared['requests']
    for sha, packet in requests.items():
        assert sha == digest(packet)
        assert set(packet) == {'task', 'candidate_records', 'evidence', 'requires_direct_semantic_evidence'}
        records = packet['candidate_records']
        assert len({r['candidate_id'] for r in records}) == len(records)
        assert len({tuple(r['root_labels']) for r in records}) == len(records)
        assert not any(field in json.dumps(packet) for field in ['expected_labels', 'truth_candidate', 'case-00'])
    baseline = {r['case_id']: r for r in read(ROOT / 'runs/hyperdbdiag-unified-20260930/predictions.json')}
    all_eval = []
    configs = {c['id']: c for c in protocol['configurations']}
    for fold in prepared['folds']:
        train, evaluation = set(fold['train_ids']), set(fold['eval_ids'])
        assert len(train) == 550 and len(evaluation) == 110 and not (train & evaluation)
        all_eval.extend(fold['eval_ids'])
        arrays = {theta: np.load(output / 'branch_probabilities' /
                                 f'{fold["fold"].replace(":", "_")}_theta{theta:g}.npz')
                  for theta in protocol['grids']['theta']}
        sel = fold['training_selection']
        settings = dict(alpha=sel['selected_raw_weight'], beta=sel['selected_epdg_weight'], theta=1., K=sel['selected_candidate_pool_size'])
        label_sets = [[prepared['labels'][j] for j in candidate] for candidate in fold['candidates']]
        for config_id, config in configs.items():
            rows = [r for r in prepared['rows'] if r['fold'] == fold['fold'] and r['configuration'] == config_id]
            assert [r['case_id'] for r in rows] == fold['eval_ids']
            values = dict(settings)
            if config['parameter'] != 'reported':
                values[config['parameter']] = config['value']
            branches = arrays[values['theta']]
            scores = ((1-values['beta']) * (values['alpha'] * branches['raw'] +
                        (1-values['alpha']) * branches['h_residual']) + values['beta'] * branches['epdg_prior'])
            order = np.argsort(-scores, axis=1, kind='stable')[:, :int(values['K'])]
            for row, indices in zip(rows, order):
                assert row['parameters'] == values
                assert row['full_pool'] == [label_sets[i] for i in indices]
                assert row['baseline'] == row['full_pool'][0]
                assert row['baseline'] in row['retained_pool'] and row['local_prediction'] in row['retained_pool']
                assert all(c in row['full_pool'] for c in row['retained_pool'])
                assert sorted(map(sorted, row['retained_pool'])) == sorted(sorted(c['root_labels']) for c in requests[row['packet_sha256']]['candidate_records'])
                assert row['expected_labels'] == baseline[row['case_id']]['expected_labels']
                if config_id == 'reported':
                    assert row['packet_sha256'] == baseline[row['case_id']]['filtered_packet_sha256']
                    for name in ['baseline', 'local_prediction', 'full_pool', 'retained_pool']:
                        assert row[name] == baseline[row['case_id']][name]
    assert len(set(all_eval)) == len(all_eval) == 660
    audit = {'status': 'inputs_verified', 'configurations': 22, 'logical_cases': 14520,
             'unique_inputs': len(requests), 'all_fold_ids_disjoint': True,
             'all_candidate_pools_recomputed_from_saved_probabilities': True,
             'one_parameter_changes_only': True, 'reference_660_inputs_reproduced': True,
             'protected_files_unchanged': True}
    if (output / 'completion.json').exists():
        response_rows = [json.loads(line) for line in (output / 'new_primary_responses.jsonl').read_text().splitlines()]
        assert Counter(r['packet_sha256'] for r in response_rows) == Counter(prepared['new_request_order'])
        for response in response_rows:
            assert response['draw'] == 'primary'
            if response['status'].startswith('accepted_'):
                validate_response(response['decision'], requests[response['packet_sha256']])
        rows = read(output / 'predictions.json')
        results = read(output / 'results.json')['configurations']
        assert len(rows) == len(prepared['rows'])
        for result in results:
            group = [r for r in rows if r['configuration'] == result['id']]
            assert len({r['case_id'] for r in group}) == 660
            pairs = [(set(r['expected_labels']), set(r['predicted_labels'])) for r in group]
            exact = sum(a == b for a, b in pairs) / 660
            tp = sum(len(a & b) for a, b in pairs)
            errors = sum(len(a ^ b) for a, b in pairs)
            assert exact == result['overall']['Exact']
            assert 2*tp / (2*tp + errors) == result['overall']['F1']
            assert exact <= result['retained_coverage']
            for row in group:
                assert row['predicted_labels'] in row['retained_pool']
                if result['id'] == 'reported':
                    assert row['predicted_labels'] == baseline[row['case_id']]['predicted_labels']
        audit.update(status='complete_verified', new_calls=len(response_rows),
                     response_statuses=dict(Counter(r['status'] for r in response_rows)),
                     all_final_metrics_independently_recomputed=True,
                     reference_660_final_predictions_reproduced=True)
    write_json(output / 'independent_audit.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
