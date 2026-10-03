#!/usr/bin/env python3
"""Read-only verification of the registered paper results; no model requests."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecsa_arbitration import digest, validate_response
from ecsa_experiment import verify_frozen_inputs

MAIN = ROOT / 'runs/hyperdbdiag-unified-20260930'
SENSITIVITY = ROOT / 'runs/parameter-sensitivity-20260930-v1'
STRESS = ROOT / 'runs/ecsa-evidence-stress-full660-20260930-v1'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def equal(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f'{name}: {actual} != {expected}')

def unique_rows(rows, key='case_id', expected=660):
    indexed = {row[key]: row for row in rows}
    require(len(indexed) == len(rows) == expected, f'duplicate or missing {key}')
    return indexed

def scores(rows, field='predicted_labels'):
    require(bool(rows), 'cannot score an empty cohort')
    pairs = [(set(row['expected_labels']), set(row[field])) for row in rows]
    tp = sum(len(a & b) for a, b in pairs)
    fp = sum(len(b - a) for a, b in pairs)
    fn = sum(len(a - b) for a, b in pairs)
    return dict(n=len(rows), Exact=sum(a == b for a, b in pairs) / len(rows),
                Precision=tp / (tp + fp) if tp + fp else 0,
                Recall=tp / (tp + fn) if tp + fn else 0,
                F1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0)

def completion_hashes(folder):
    hashes = read(folder / 'completion.json')['files_sha256']
    for name, expected in hashes.items():
        require(sha(folder / name) == expected, f'recorded result changed: {folder.name}/{name}')
    return len(hashes)

def source_hashes(protocol):
    ports = read(ROOT / 'docs/source_provenance.json')['portable_changes']
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) == expected:
            continue
        port = ports.get(name, {})
        require(port.get('recorded_sha256') == expected, f'unregistered source change: {name}')
        require(sha(ROOT / port['original_copy']) == expected, f'original source changed: {name}')
        require(sha(ROOT / name) == port['current_sha256'], f'portable source changed: {name}')

def verify_main(folder):
    completion_hashes(folder)
    source_hashes(read(folder / 'protocol.json'))
    canonical = unique_rows(read(folder / 'predictions.json'))
    folds = Counter(row['fold'] for row in canonical.values())
    require(len(folds) == 6 and set(folds.values()) == {110}, 'invalid evaluation folds')
    main = read(folder / 'main_comparison.json')['methods']
    ablation = next(iter(read(folder / 'ablation.json')['datasets'].values()))['methods']
    fields = {'Exact': 'exact_set_accuracy', 'Precision': 'component_precision',
              'Recall': 'component_recall', 'F1': 'component_f1'}
    for label, methods in [('main', main), ('ablation', ablation)]:
        for name, method in methods.items():
            rows = unique_rows(method['results'])
            require(rows.keys() == canonical.keys(), f'{label}/{name}: different cases')
            for case, row in rows.items():
                require(set(row['expected_labels']) == set(canonical[case]['expected_labels']),
                        f'{label}/{name}: different ground truth')
            actual = scores(list(rows.values()))
            for metric, field in fields.items():
                equal(actual[metric], method['overall'][field], f'{label}/{name}/{metric}')
            for group_field, row_field in [('by_root_cardinality', None),
                                           ('by_occurrence_pattern', 'occurrence_pattern')]:
                for group, reported in method.get(group_field, {}).items():
                    selected = [r for r in rows.values() if
                                (str(len(r['expected_labels'])) if row_field is None else r[row_field]) == group]
                    result = scores(selected)
                    for metric in ['Exact', 'F1']:
                        equal(result[metric], reported[fields[metric]], f'{name}/{group}/{metric}')
            if name == 'hyperdbdiag':
                require(all(row['predicted_labels'] == canonical[case]['predicted_labels']
                            for case, row in rows.items()), 'main and ablation predictions differ')
    require(all(row['predicted_labels'] in row['retained_pool'] for row in canonical.values()),
            'final main prediction outside candidate pool')
    return canonical, {'cases': len(canonical), 'main': {k: scores(v['results']) for k, v in main.items()},
                       'ablation': {k: scores(v['results']) for k, v in ablation.items()}}

def verify_sensitivity(canonical):
    completion_hashes(SENSITIVITY)
    protocol = read(SENSITIVITY / 'protocol.json')
    results = read(SENSITIVITY / 'results.json')
    require(results['protocol_sha256'] == digest(protocol), 'sensitivity protocol mismatch')
    grouped = defaultdict(list)
    for row in read(SENSITIVITY / 'predictions.json'):
        grouped[row['configuration']].append(row)
    require(len(grouped) == len(results['configurations']) == 22, 'incomplete sensitivity sweep')
    for config in results['configurations']:
        rows = unique_rows(grouped[config['id']])
        require(rows.keys() == canonical.keys(), 'sensitivity cohort changed')
        for case, row in rows.items():
            require(row['expected_labels'] == canonical[case]['expected_labels'], 'sensitivity truth changed')
            require(row['predicted_labels'] in row['retained_pool'], 'sensitivity prediction outside pool')
            if config['id'] == 'reported':
                require(row['predicted_labels'] == canonical[case]['predicted_labels'], 'reference prediction changed')
        actual = scores(list(rows.values()))
        for metric in ['Exact', 'F1']:
            equal(actual[metric], config['overall'][metric], f'sensitivity/{config["id"]}/{metric}')
    return {'configurations': len(grouped), 'evaluations': sum(map(len, grouped.values())), 'reference_matches_main': True}

def verify_stress(canonical):
    completion_hashes(STRESS)
    protocol = read(STRESS / 'protocol.json')
    prepared = read(STRESS / 'prepared.json')
    require(prepared['protocol_sha256'] == digest(protocol), 'stress protocol mismatch')
    responses = [json.loads(line) for name in ['reused_primary_responses.jsonl', 'new_responses.jsonl']
                 for line in (STRESS / name).read_text().splitlines()]
    records = unique_rows(responses, 'job_id', len(prepared['jobs']))
    require(records.keys() == prepared['jobs'].keys(), 'stress response inventory mismatch')
    grouped = defaultdict(list)
    for row in read(STRESS / 'predictions.json'):
        grouped[row['arm']].append(row)
        response = records[row['job_id']]
        packet = prepared['requests'][row['packet_sha256']]
        require(digest(packet) == row['packet_sha256'] == response['packet_sha256'], 'stress request mismatch')
        labels = row['local_prediction']
        if 'raw_response' in response:
            raw = response['raw_response']
            require(digest(raw) == response['raw_response_sha256'], 'stress raw response changed')
            try:
                decision = validate_response(raw, packet)
                status = 'accepted_selection' if decision['action'] == 'SELECT_CANDIDATE' else 'accepted_abstain'
                if status == 'accepted_selection':
                    labels = next(c['root_labels'] for c in packet['candidate_records']
                                  if c['candidate_id'] == decision['candidate_id'])
            except (ValueError, TypeError, KeyError):
                status = 'fallback_invalid_response'
        else:
            status = 'fallback_transport'
        require(labels == row['predicted_labels'] == response['predicted_labels'], 'stress decision mismatch')
        require(status == row['status'] == response['status'], 'stress validation status mismatch')
        require(labels in [c['root_labels'] for c in packet['candidate_records']], 'stress final prediction outside pool')
    results = read(STRESS / 'results.json')
    require(set(grouped) == set(protocol['arms']) and len(grouped) == 7, 'stress arm mismatch')
    output = {}
    for arm in results['arms']:
        rows = unique_rows(grouped[arm['arm']])
        require(rows.keys() == canonical.keys(), 'stress cohort changed')
        require(all(r['expected_labels'] == canonical[c]['expected_labels'] for c, r in rows.items()), 'stress ground truth changed')
        actual = scores(list(rows.values()))
        for metric in ['Exact', 'F1']:
            equal(actual[metric], arm['metrics'][metric], f'stress/{arm["arm"]}/{metric}')
        require(arm['final_out_of_pool'] == 0, 'stress reported pool violations mismatch')
        output[arm['arm']] = actual
    return {'arms': output, 'response_records_checked': len(records),
            'raw_responses_revalidated': sum('raw_response' in r for r in records.values()),
            'final_out_of_pool': 0}

def verify_localization():
    report = read(ROOT / 'runs/dbmags-operator-bound-v4/full_report.json')
    for name, method in report['methods'].items():
        rows = method['results']
        require(len(rows) == len({(r['case_id'], r['root']) for r in rows}) == 576, f'{name}: incomplete localization cohort')
        for category, key, field in [('root_sql_pairs', 'hit_at_1', 'sql_hit_at_1'),
                                     ('root_sql_pairs', 'hit_at_2', 'sql_hit_at_2'),
                                     ('root_operator_pairs', 'hit_at_1', 'operator_hit_at_1'),
                                     ('joint_sql_operator', 'hit_at_1', 'joint_sql_operator_hit_at_1')]:
            equal(sum(r[field] for r in rows) / len(rows), method['metrics'][category][key], f'{name}/{field}')
        require(len({r['case_id'] for r in rows}) == method['metrics']['case_count'] == 296,
                f'{name}: localization case count mismatch')
        for category, field in [('root_sql_pairs', 'target_sql_rank'),
                                ('root_operator_pairs', 'target_operator_rank')]:
            equal(statistics.mean(r[field] for r in rows),
                  method['metrics'][category]['average_target_rank'], f'{name}/{field}')
    return {'cases': 296, 'root_chains': 576, 'conditional_on_root_and_target_plan': True}

def verify_structure():
    report = read(ROOT / 'runs/dbmags-structure-efficiency/full_report.json')
    folds, aggregate = report['folds'], report['aggregate']
    require(len(folds) == aggregate['fold_count'] == 6, 'structure fold count mismatch')
    for name in ['hypergraph_incidence_count', 'equivalent_pairwise_occurrence_count']:
        equal(statistics.mean(f[name] for f in folds), aggregate['mean_' + name], name)
    for family in ['hypergraph', 'pairwise']:
        for fold in folds:
            equal(statistics.median(fold['timing'][family + '_measurements_ms']),
                  fold['timing'][family + '_median_ms'], family + ' timing median')
        equal(statistics.mean(f['timing'][family + '_median_ms'] for f in folds),
              aggregate['mean_' + family + '_traversal_ms'], family + ' timing mean')
    equal(1 - aggregate['mean_hypergraph_incidence_count'] / aggregate['mean_equivalent_pairwise_occurrence_count'],
          aggregate['relation_item_reduction'], 'relation reduction')
    equal(1 - aggregate['mean_hypergraph_traversal_ms'] / aggregate['mean_pairwise_traversal_ms'],
          aggregate['relative_traversal_time_reduction'], 'traversal reduction')
    return aggregate

def verify_all(main=MAIN):
    old_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        frozen = verify_frozen_inputs()
        canonical, main_result = verify_main(main)
        return {'status': 'verified', 'frozen_inputs': frozen, 'main': main_result,
                'sensitivity': verify_sensitivity(canonical), 'evidence_perturbations': verify_stress(canonical),
                'localization': verify_localization(), 'structure': verify_structure(), 'new_model_calls': 0}
    finally:
        os.chdir(old_cwd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=MAIN)
    args = parser.parse_args()
    print(json.dumps(verify_all(args.main.resolve()), indent=2))
