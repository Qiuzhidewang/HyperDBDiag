#!/usr/bin/env python3
"""One-factor sensitivity of the frozen HyperDBDiag configuration.

Training and evaluation partitions, downstream local thresholds, and inference
instructions remain fixed. Changed theta values rebuild the graph and models;
changed probability weights reuse identical fitted branch probabilities. LLM
responses are reused only for byte-equivalent structured input (including old
failures), and every other non-singleton request receives a new primary draw.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import random
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import hyperdbdiag_ablation as components
from conservative_evidence_judge import ConservativeEvidenceJudge
from ecsa_arbitration import (INSTRUCTIONS, SCHEMA, build_packets,
                              configured_codex_client, digest, execute_job,
                              file_digest, write_json)
from ecsa_experiment import SEED, verify_frozen_inputs
from ecsa_pipeline import FrozenDecisionStore, resolve_response

OUTPUT = ROOT / 'runs/parameter-sensitivity-20260930-v1'
CANONICAL = ROOT / 'runs/hyperdbdiag-unified-20260930'
HISTORY = ROOT / 'runs/ecsa-full660-20260929'
GRIDS = {
    'alpha': [0., .2, .4, .6, .8, 1.],
    'beta': [0., .2, .4, .5, .6, .8, 1.],
    'theta': [.5, .75, 1., 1.25, 1.5],
    'K': [1, 2, 3, 4, 6, 8],
}
BASE = {'beta': .5, 'theta': 1., 'K': 4}


def load(path):
    return json.loads(Path(path).read_text())


def configurations():
    result = [{'id': 'reported', 'parameter': 'reported', 'value': None}]
    for parameter, grid in GRIDS.items():
        for value in grid:
            if parameter in BASE and value == BASE[parameter]:
                continue
            result.append({'id': f'{parameter}={value:g}',
                           'parameter': parameter, 'value': value})
    return result


def configured_values(selected, config):
    values = dict(alpha=float(selected['selected_raw_weight']),
                  beta=float(selected['selected_epdg_weight']),
                  theta=1., K=int(selected['selected_candidate_pool_size']))
    if config['parameter'] != 'reported':
        values[config['parameter']] = config['value']
    return values


def fuse(branches, alpha, beta):
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError('fusion weights must be in [0,1]')
    return (1. - beta) * (alpha * branches['raw'] +
                         (1. - alpha) * branches['h_residual']) + beta * branches['epdg_prior']


def candidate_pools(branches, values):
    if values['K'] < 1:
        raise ValueError('candidate width must be positive')
    scores = fuse(branches, values['alpha'], values['beta'])
    return np.argsort(-scores, axis=1, kind='stable')[:, :int(values['K'])]


def response_labels(packet, response, fallback):
    if response['status'] == 'analytic_single_candidate':
        records = packet['candidate_records']
        if len(records) != 1 or set(fallback) != set(records[0]['root_labels']):
            raise ValueError('singleton shortcut cannot apply to multiple candidates')
        return list(records[0]['root_labels'])
    return resolve_response(packet, response, fallback)


def historical_store():
    return FrozenDecisionStore(load(HISTORY / 'prepared.json')['jobs'],
                               load(HISTORY / 'decisions.json'))


def saved_responses(output, requests, old):
    responses = {}
    for key, packet in requests.items():
        if digest(packet) != key:
            raise ValueError('request fingerprint mismatch')
        if key in old.responses:
            if packet != old.expected_packets[key]:
                raise ValueError('historical input mismatch')
            responses[key] = {**old.responses[key], 'response_source': 'historical_exact_primary'}
        elif len(packet['candidate_records']) == 1:
            responses[key] = {'status': 'analytic_single_candidate', 'usage': {},
                              'response_source': 'deterministic_singleton'}
    path = output / 'new_primary_responses.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row['packet_sha256']
            if key not in requests or key in responses:
                raise ValueError('duplicate or out-of-protocol primary response')
            response_labels(requests[key], row, requests[key]['candidate_records'][0]['root_labels'])
            responses[key] = {**row, 'response_source': 'new_primary'}
    return responses


def check_preserved(output):
    for path, expected in load(output / 'protocol.json')['protected_sha256'].items():
        if file_digest(ROOT / path) != expected:
            raise ValueError(f'protected manuscript, source or result changed: {path}')


def initialize(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        p = load(output / 'protocol.json')
        if p['grids'] != GRIDS or p['script_sha256'] != file_digest(Path(__file__)):
            raise ValueError('protocol/source changed; use a new output directory')
        check_preserved(output)
        return
    integrity = verify_frozen_inputs()
    protected = [ROOT / 'main.tex', ROOT / 'HyperDBDiag2.docx',
                 ROOT / 'build/compile-input/main.tex', ROOT / 'build/compiled-latest/main.pdf']
    for directory in ['src', 'tests', 'figures', 'runs/hyperdbdiag-unified-20260930']:
        protected += [p for p in (ROOT / directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts]
    protocol = {
        'name': 'hyperdbdiag_one_factor_parameter_sensitivity_v1',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'grids_fixed_before_new_outcomes': True, 'seed': SEED,
        'grids': GRIDS, 'configurations': configurations(),
        'evaluation': 'same frozen 660 cases and six replicate folds; 550 train / 110 test',
        'estimand': 'one-factor perturbation around each fold of the reported trained configuration',
        'other_parameters': 'retain each fold baseline alpha/beta/theta/K and trained local thresholds; no retuning after a perturbation',
        'theta_training': 'rebuild training-only incidence, cross-fitted features and both classifiers; retain baseline fusion weights and K',
        'theta_invariant_branches': 'check raw/EPDG probabilities against baseline with atol=1e-12, rtol=0, then retain baseline values to avoid parallel forest summation roundoff',
        'pre_inference_engineering_note': 'initial preparation in runs/parameter-sensitivity-20260930 stopped at raw-probability equality check (max difference 1.11e-16); no new inference or sensitivity outcomes were produced; parameter grids and evaluation design unchanged',
        'alpha_beta_training': 'reuse exactly the same trained branch probabilities; change only the named fusion coefficient',
        'K_definition': 'number of upstream candidate root sets, before local filtering; not number of roots',
        'reference_configuration': 'training-selected alpha and beta, theta=1, training-selected K; exact per-case match to current main experiment required',
        'LLM': {'model': 'gpt-5.5', 'reasoning_effort': 'xhigh',
                'instructions_sha256': digest(INSTRUCTIONS), 'schema_sha256': digest(SCHEMA),
                'all_non_singleton_inputs': 'exact previous primary input reuse or one new primary inference; no repeat substitution',
                'singletons': 'analytical final label only: valid selection and abstention/local fallback yield the same sole candidate; no invented LLM response or usage',
                'fallback': 'same calibrated local prediction as complete HyperDBDiag',
                'new_request_order': 'fixed seeded shuffle; exact input deduplication across configurations',
                'workers': 8, 'application_attempts': 1},
        'metrics': ['complete-method Exact and micro component F1', 'upstream/local Exact and F1',
                    'candidate-set coverage before/after filtering', 'mean candidate width',
                    'fold scores and paired correct/incorrect changes from reported configuration'],
        'scope': 'parameter sensitivity on existing known scenarios; not unseen-scenario or data-noise robustness',
        'script_sha256': file_digest(Path(__file__)),
        'protected_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in protected},
        'frozen_input_verification': integrity,
    }
    write_json(output / 'protocol.json', protocol)


def make_fold_rows(dataset, split, judge, candidates, branches, config, values):
    pools = candidate_pools(branches, values)
    ids = tuple(f'candidate-{i:03d}' for i in range(len(candidates)))
    cards = [tuple(judge._card(ids[int(i)], candidates[int(i)], q, obs,
                              dataset.mechanism_cards) for i in pool)
             for q, obs, pool in zip(split.eval_x, split.eval_semantic, pools)]
    bundles = components._llm_evidence(
        split.eval_x, split.train_x, split.train_labels, split.oof_groups,
        dataset.labels, dataset.feature_names, dataset.mechanism_cards,
        cards, split.eval_semantic, split.train_semantic,
        dataset.require_discriminating_semantic_for_llm)
    result = []
    for index, (q, obs, pool, bundle) in enumerate(zip(split.eval_x, split.eval_semantic, pools, bundles)):
        decision = judge.decide_pool(candidates, pool, q, obs, dataset.mechanism_cards)
        labels_for = lambda i: [dataset.labels[j] for j in candidates[int(i)]]
        packets = build_packets(bundle, list(map(int, pool)),
                                list(decision.retained_candidates), candidates, dataset.labels)
        packet = packets['ecsa_filtered_pool']
        row = {'configuration': config['id'], 'case_id': split.eval_ids[index],
               'fold': split.split_id, 'parameters': values,
               'baseline': labels_for(pool[0]), 'local_prediction': labels_for(decision.local_candidate),
               'full_pool': [labels_for(i) for i in pool],
               'retained_pool': [labels_for(i) for i in decision.retained_candidates],
               'local_status': decision.status, 'packet_sha256': digest(packet),
               'full_packet_sha256': digest(packets['llm_full_pool'])}
        # Evaluation labels are attached only after constructing all inference input.
        row['expected_labels'] = list(split.eval_labels[index])
        result.append((row, packet))
    return result


def prepare(output):
    initialize(output)
    if (output / 'prepared.json').exists():
        return load(output / 'prepared.json')
    cards, _ = components._load_root_mechanism_cards()
    dataset = components._load_dbmags(components.DEFAULT_DBMAGS_ROOT, cards)
    reference = {r['case_id']: r for r in load(CANONICAL / 'predictions.json')}
    old_folds = load(ROOT / 'runs/dbmags-ablation/full_report_llm_xhigh.json')['datasets'][dataset.name]['folds']
    expected_settings = {f['outer_split']: f['hypergraph']['fusion_selection'] for f in old_folds}
    records, requests, folds, fits = [], {}, [], 0
    branch_dir = output / 'branch_probabilities'
    branch_dir.mkdir(exist_ok=True)
    for split in dataset.splits:
        split.validate(dataset.feature_count, dataset.labels)
        print(f'Preparing {split.split_id}', flush=True)
        model = components._new_h_model(dataset, SEED).fit(
            split.train_x, split.train_labels, dataset.labels, split.oof_groups, split.train_semantic)
        fits += 1
        selected = model.fusion_selection
        for key in ['selected_raw_weight', 'selected_epdg_weight', 'selected_candidate_pool_size', 'scores', 'candidate_pool_oof_recall']:
            if selected[key] != expected_settings[split.split_id][key]:
                raise ValueError(f'baseline training selection changed: {split.split_id}/{key}')
        if selected['selected_epdg_weight'] != .5 or selected['selected_candidate_pool_size'] != 4:
            raise ValueError('reference no longer matches preregistered design')
        calibration = load(CANONICAL / 'calibration' / (split.split_id.replace(':', '_') + '.json'))
        if set(calibration['outer_test_ids_excluded']) != set(split.eval_ids):
            raise ValueError('local calibration does not match evaluation fold')
        if set(split.eval_ids) & {r['case_id'] for r in calibration['training_oof_rows']}:
            raise ValueError('evaluation case leaked into local calibration')
        judge = ConservativeEvidenceJudge(calibration['selected']).fit(
            split.train_x, split.train_labels, split.oof_groups, dataset.labels, dataset.feature_names)
        baseline_branches = model.predict_branch_probabilities(split.eval_x, split.eval_semantic)
        branches = {}
        theta_metadata = {}
        for theta in GRIDS['theta']:
            if theta == 1.:
                fitted = model
            else:
                fitted = components._new_h_model(replace(dataset, activation_threshold=theta), SEED)
                fitted.adaptive_fusion = False  # downstream mixture is kept at the reference values
                fitted.fit(split.train_x, split.train_labels, dataset.labels, split.oof_groups, split.train_semantic)
                fits += 1
            if fitted.candidates != model.candidates:
                raise ValueError('candidate inventory changed within one-factor study')
            branches[theta] = baseline_branches if theta == 1. else fitted.predict_branch_probabilities(split.eval_x, split.eval_semantic)
            for key in ['raw', 'epdg_prior']:
                if not np.allclose(branches[theta][key], baseline_branches[key], atol=1e-12, rtol=0):
                    raise ValueError(f'theta changed the invariant {key} branch')
                branches[theta][key] = baseline_branches[key]
            theta_metadata[str(theta)] = fitted.encoder.metadata()
            np.savez_compressed(branch_dir / f'{split.split_id.replace(":", "_")}_theta{theta:g}.npz',
                                **{k: branches[theta][k] for k in ['raw', 'h_residual', 'epdg_prior']})
        fold_values = []
        for config in configurations():
            values = configured_values(selected, config)
            pairs = make_fold_rows(dataset, split, judge, model.candidates, branches[values['theta']], config, values)
            for row, packet in pairs:
                if config['id'] == 'reported':
                    saved = reference[row['case_id']]
                    for key in ['baseline', 'local_prediction', 'full_pool', 'retained_pool', 'local_status', 'full_packet_sha256']:
                        if row[key] != saved[key]:
                            raise ValueError(f'reported configuration differs: {row["case_id"]}/{key}')
                    if row['packet_sha256'] != saved['filtered_packet_sha256']:
                        raise ValueError('reported arbitration packet differs')
                key = row['packet_sha256']
                if key in requests and requests[key] != packet:
                    raise ValueError('input digest collision')
                requests[key] = packet
                records.append(row)
            fold_values.append({'configuration': config['id'], **values})
        folds.append({'fold': split.split_id, 'train_ids': list(split.train_ids),
                      'eval_ids': list(split.eval_ids), 'candidates': model.candidates,
                      'training_selection': selected, 'local_calibration': calibration['selected'],
                      'theta_encoders': theta_metadata, 'configurations': fold_values})
        print(f'  finished: {len(records)} logical cases, {len(requests)} distinct inputs', flush=True)
    old = historical_store()
    reuse = saved_responses(output, requests, old)
    keys = sorted(set(requests) - set(reuse))
    random.Random(SEED).shuffle(keys)
    prepared = {'protocol_sha256': digest(load(output / 'protocol.json')), 'labels': list(dataset.labels),
                'rows': records, 'requests': requests, 'new_request_order': keys, 'folds': folds,
                'model_fits': fits, 'historical_unique_inputs': sum(v['response_source'] == 'historical_exact_primary' for v in reuse.values()),
                'analytical_unique_singletons': sum(v['response_source'] == 'deterministic_singleton' for v in reuse.values())}
    if len(records) != len(configurations()) * 660:
        raise ValueError('incomplete configurations')
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'preparation_summary.json', {k: v for k, v in prepared.items() if k not in ['rows', 'requests', 'folds', 'new_request_order']}
               | {'logical_cases': len(records), 'unique_inputs': len(requests), 'new_model_calls_planned': len(keys)})
    check_preserved(output)
    print(f'Prepared {len(records)} logical cases; {len(keys)} new inference inputs.', flush=True)
    return prepared


def run_primary_jobs(jobs, client, record, workers=8, outage_limit=16):
    """Bound requests and persist every completed draw, even when stopping."""
    iterator = iter(jobs)
    failures, stop_error = 0, None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        def replenish():
            while stop_error is None and len(pending) < workers:
                job = next(iterator, None)
                if job is None:
                    break
                pending.add(pool.submit(execute_job, job, client))
        replenish()
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                try:
                    row = future.result()
                except Exception as exc:
                    stop_error = stop_error or exc
                    continue
                record(row)
                failures = failures + 1 if row['status'] == 'fallback_transport' else 0
                if failures >= outage_limit:
                    stop_error = stop_error or RuntimeError('sustained inference transport outage; results incomplete, inspect progress')
            replenish()
    if stop_error is not None:
        raise stop_error


def execute(output):
    initialize(output)
    prepared = load(output / 'prepared.json')
    if prepared['protocol_sha256'] != digest(load(output / 'protocol.json')):
        raise ValueError('prepared protocol changed')
    requests = prepared['requests']
    responses = saved_responses(output, requests, historical_store())
    pending = [k for k in prepared['new_request_order'] if k not in responses]
    if not pending:
        return
    client, metadata = configured_codex_client(output)
    metadata.update(workers=8, input_protocol_sha256=prepared['protocol_sha256'],
                    exact_input_cache=True, primary_only=True)
    write_json(output / 'llm_configuration.json', metadata)
    first = {}
    for row in prepared['rows']:
        first.setdefault(row['packet_sha256'], row)
    jobs = []
    for key in pending:
        row = first[key]
        jobs.append(dict(job_id=f'sensitivity-primary-{key}', case_id=row['case_id'], fold=row['fold'],
                         arm='parameter_sensitivity', packet_sha256=key, packet=requests[key],
                         fallback=row['local_prediction']))
    started = time.monotonic()
    completed = len(prepared['new_request_order']) - len(pending)
    with (output / 'new_primary_responses.jsonl').open('a', buffering=1) as sink:
        def record(row):
            nonlocal completed
            row.update(draw='primary', response_source='new_primary',
                       recorded_utc=datetime.now(timezone.utc).isoformat())
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            responses[row['packet_sha256']] = row
            completed += 1
            status_counts = Counter(v['status'] for v in responses.values() if v.get('response_source') == 'new_primary')
            progress = dict(completed_new_calls=completed, planned_new_calls=len(prepared['new_request_order']),
                            statuses=dict(status_counts), current_process_seconds=time.monotonic() - started)
            write_json(output / 'progress.json', progress)
            if completed % 20 == 0 or completed == len(prepared['new_request_order']):
                print(json.dumps(progress), flush=True)
        run_primary_jobs(jobs, client, record)
    check_preserved(output)


def metrics(rows, field='predicted_labels'):
    pairs = [(set(r['expected_labels']), set(r[field])) for r in rows]
    tp = sum(len(a & b) for a, b in pairs)
    fp = sum(len(b - a) for a, b in pairs)
    fn = sum(len(a - b) for a, b in pairs)
    return {'n': len(rows), 'correct': sum(a == b for a, b in pairs), 'tp': tp, 'fp': fp, 'fn': fn,
            'Exact': sum(a == b for a, b in pairs) / len(rows),
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def analyze(output):
    initialize(output)
    prepared = load(output / 'prepared.json')
    requests = prepared['requests']
    responses = saved_responses(output, requests, historical_store())
    missing = set(requests) - set(responses)
    if missing:
        raise ValueError(f'{len(missing)} required inputs lack inference; no final sensitivity scores emitted')
    rows = []
    canonical = {r['case_id']: r for r in load(CANONICAL / 'predictions.json')}
    for source in prepared['rows']:
        row = dict(source)
        response = responses[row['packet_sha256']]
        row['predicted_labels'] = response_labels(requests[row['packet_sha256']], response, row['local_prediction'])
        row['response_source'] = response['response_source']
        row['llm_status'] = response['status']
        if not any(set(row['predicted_labels']) == set(c) for c in row['retained_pool']):
            raise ValueError('final prediction escaped retained pool')
        if row['configuration'] == 'reported' and row['predicted_labels'] != canonical[row['case_id']]['predicted_labels']:
            raise ValueError('reported final diagnosis changed')
        rows.append(row)
    summaries = []
    for config in configurations():
        group = [r for r in rows if r['configuration'] == config['id']]
        if len(group) != 660 or len({r['case_id'] for r in group}) != 660:
            raise ValueError('incomplete per-configuration cohort')
        result = dict(config, overall=metrics(group), upstream=metrics(group, 'baseline'),
                      local=metrics(group, 'local_prediction'))
        for prefix, field in [('upstream', 'full_pool'), ('retained', 'retained_pool')]:
            result[f'{prefix}_coverage'] = sum(any(set(r['expected_labels']) == set(c) for c in r[field]) for r in group) / len(group)
            result[f'{prefix}_mean_width'] = float(np.mean([len(r[field]) for r in group]))
        result['by_fold'] = {fold['fold']: metrics([r for r in group if r['fold'] == fold['fold']]) for fold in prepared['folds']}
        good = lambda r: set(r['predicted_labels']) == set(r['expected_labels'])
        result['paired_to_reported'] = {
            'wins': sum(good(r) and not good(canonical[r['case_id']]) for r in group),
            'losses': sum(not good(r) and good(canonical[r['case_id']]) for r in group),
            'changed_predictions': sum(set(r['predicted_labels']) != set(canonical[r['case_id']]['predicted_labels']) for r in group)}
        result['response_sources'] = dict(Counter(r['response_source'] for r in group))
        result['statuses'] = dict(Counter(r['llm_status'] for r in group))
        summaries.append(result)
    write_json(output / 'predictions.json', rows)
    write_json(output / 'results.json', {'protocol_sha256': prepared['protocol_sha256'], 'configurations': summaries,
                                        'new_primary_calls': len(prepared['new_request_order']),
                                        'historical_unique_inputs': prepared['historical_unique_inputs'],
                                        'analytical_unique_singletons': prepared['analytical_unique_singletons']})
    columns = ['configuration', 'case_id', 'fold', 'expected_labels', 'predicted_labels',
               'baseline', 'local_prediction', 'packet_sha256', 'response_source', 'llm_status']
    with (output / 'predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '+'.join(row[k]) if isinstance(row[k], list) else row[k] for k in columns})
    with (output / 'sensitivity_summary.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['configuration', 'Exact_pct', 'F1_pct', 'upstream_Exact_pct', 'local_Exact_pct',
                         'candidate_coverage_pct', 'retained_coverage_pct', 'mean_retained_candidates'])
        for row in summaries:
            writer.writerow([row['id'], 100 * row['overall']['Exact'], 100 * row['overall']['F1'],
                             100 * row['upstream']['Exact'], 100 * row['local']['Exact'],
                             100 * row['upstream_coverage'], 100 * row['retained_coverage'], row['retained_mean_width']])
    plot(output, summaries)
    check_preserved(output)
    write_json(output / 'completion.json', {'status': 'complete', 'logical_cases': len(rows),
        'configurations': len(summaries), 'new_model_calls': len(prepared['new_request_order']),
        'reference_matches_all_660_cases': True, 'protected_files_unchanged': True,
        'files_sha256': {p.name: file_digest(p) for p in output.iterdir()
                         if p.is_file() and p.name != 'completion.json'}})
    print(json.dumps({r['id']: {k: round(v*100, 3) for k, v in r['overall'].items() if k in ['Exact','F1']} for r in summaries}, indent=2))


def plot(output, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    reference = summaries[0]['overall']
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8), sharey=True)
    for axis, (parameter, grid), label in zip(axes, GRIDS.items(), ['(a) α', '(b) β', '(c) θ', '(d) K']):
        selected = [next((r for r in summaries if r['parameter'] == parameter and r['value'] == v),
                         summaries[0] if BASE.get(parameter) == v else None) for v in grid]
        for key, color, marker in [('Exact', '#4c78a8', 'o'), ('F1', '#f58518', 's')]:
            axis.plot(grid, [100*r['overall'][key] for r in selected], color=color, marker=marker,
                      linewidth=1.8, markersize=4.5, label=key)
            axis.axhline(100*reference[key], color=color, linestyle='--', linewidth=.8, alpha=.65)
        axis.set_xlabel(label, fontsize=12)
        axis.set_xticks(grid)
        axis.tick_params(labelsize=9)
        axis.grid(axis='y', color='#d9e2eb', linewidth=.6)
        axis.spines[['top','right']].set_visible(False)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel('Diagnostic performance (%)', fontsize=10)
    handles = [Line2D([], [], color='#4c78a8', marker='o', label='Exact'),
               Line2D([], [], color='#f58518', marker='s', label='Component F1'),
               Line2D([], [], color='#777777', linestyle='--', label='Reported configuration')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .88])
    fig.savefig(output / 'parameter_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(output / 'parameter_sensitivity.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--phase', choices=['prepare', 'execute', 'analyze', 'all'], default='all')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase in ['prepare', 'all']:
        prepare(output)
    if args.phase in ['execute', 'all']:
        execute(output)
    if args.phase in ['analyze', 'all']:
        analyze(output)


if __name__ == '__main__':
    main()
