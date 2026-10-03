"""Rebuild the calibrated ECSA experiment and replay verified, frozen primary LLM draws.

All inner/outer deterministic models are fitted again. Exact request matching is
mandatory for replay. Previously failed requests remain failures; repeated draws
are excluded. This command never silently starts new model calls.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np

import hyperdbdiag_ablation as components
from conservative_evidence_judge import ConservativeEvidenceJudge, calibrate, describe_pool
from dbmags_experiment import _case_rows, _summary, compare_saved_predictions
from ecsa_arbitration import INSTRUCTIONS, SCHEMA, digest, file_digest, write_json
from ecsa_pipeline import ConservativeECSAPipeline, FrozenDecisionStore
from metric_frozen_dataset import load_frozen_metric_dataset
from structured_evidence_judge import StructuredEvidenceJudge

OUTPUT = Path('runs/hyperdbdiag-unified-20260930')
FROZEN = Path('runs/ecsa-full660-20260929')
CALIBRATION = Path('runs/ecsa-conservative-20260929/calibration')
OLD_MAIN = Path('runs/dbmags-main-comparison/full_report_llm_xhigh.json')
OLD_ABLATION = Path('runs/dbmags-ablation/full_report_llm_xhigh.json')
SEED = 20260802
METHOD_FIELDS = {
    'hypergraph_local_judge': 'local_prediction',
    'hypergraph_llm': 'llm_without_local_prediction',
    'hyperdbdiag': 'predicted_labels',
}


def load(path):
    return json.loads(Path(path).read_text())


def archived_source_member(recorded_path, members):
    """Resolve a recorded source name after relocating the checkout.

    Historical protocols can contain an absolute workstation path. Resolve only
    an unambiguous complete archive-member suffix; the caller still checks the
    original recorded SHA-256. Never rewrite the frozen protocol.
    """
    path = Path(recorded_path)
    if not path.is_absolute():
        if str(path) not in members:
            raise ValueError(f'archived source missing: {recorded_path}')
        return str(path)
    matches = [name for name in members if str(path).endswith('/' + name)]
    if len(matches) != 1:
        raise ValueError(f'archived source path is missing or ambiguous: {recorded_path}')
    return matches[0]


def verify_frozen_inputs():
    before = load(OUTPUT / 'before_manifest.json')
    archive = Path(before['archive'])
    if file_digest(archive) != before['archive_sha256']:
        raise ValueError('historical source archive changed')
    # Manuscripts may evolve; experimental inputs and recorded outcomes may not.
    checked = 0
    for name, expected in before['protected_sha256'].items():
        if not name.startswith(('runs/', 'data/')):
            continue
        if file_digest(Path(name)) != expected:
            raise ValueError(f'frozen experimental input changed: {name}')
        checked += 1
    completed = load(FROZEN / 'completion_manifest.json')
    for name, expected in completed['artifacts'].items():
        if file_digest(FROZEN / name) != expected:
            raise ValueError(f'completed inference artifact changed: {name}')
    protocol = load(FROZEN / 'protocol.json')
    prepared = load(FROZEN / 'prepared.json')
    if digest(protocol) != prepared['protocol_sha256']:
        raise ValueError('original inference protocol differs')
    for key, value in [('instructions_sha256', digest(INSTRUCTIONS)),
                       ('schema_sha256', digest(SCHEMA)),
                       ('model', 'gpt-5.5'), ('reasoning_effort', 'xhigh')]:
        if protocol[key] != value:
            raise ValueError(f'primary inference contract differs: {key}')
    with ZipFile(archive) as saved:
        for name, expected in protocol['source_sha256'].items():
            member = archived_source_member(name, saved.namelist())
            if hashlib.sha256(saved.read(member)).hexdigest() != expected:
                raise ValueError(f'archived historical source differs: {name}')
    return {'experimental_files_checked': checked,
            'completed_inference_files_checked': len(completed['artifacts']),
            'historical_source_archive_sha256': before['archive_sha256'],
            'frozen_protocol_sha256': file_digest(FROZEN / 'protocol.json')}


class Fitter:
    """Deduplicate equivalent fits within this run; never load historical pickles."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def fit(self, ids, x, labels, groups, semantic):
        key = digest({'ids': list(ids), 'x': hashlib.sha256(x.tobytes()).hexdigest(),
                      'labels': list(labels), 'groups': [int(g) for g in groups],
                      'semantic': repr(semantic), 'seed': SEED})
        if key not in self.cache:
            model = components._new_h_model(self.dataset, SEED).fit(
                x, labels, self.dataset.labels, groups, semantic)
            judge = StructuredEvidenceJudge().fit(
                x, labels, groups, self.dataset.labels, self.dataset.feature_names)
            self.cache[key] = (model, judge)
        return self.cache[key]


def calibrate_fold(dataset, split, fitter):
    groups = np.asarray(split.oof_groups)
    inner = []
    for group in sorted(set(split.oof_groups)):
        held = groups == group
        fit = ~held
        train_ids = [c for c, keep in zip(split.train_ids, fit) if keep]
        held_ids = [c for c, keep in zip(split.train_ids, held) if keep]
        if set(train_ids) & set(held_ids) or set(split.eval_ids) & (set(train_ids) | set(held_ids)):
            raise ValueError('inner calibration overlaps evaluation cases')
        train_labels = [v for v, keep in zip(split.train_labels, fit) if keep]
        held_labels = [v for v, keep in zip(split.train_labels, held) if keep]
        train_semantic = [v for v, keep in zip(split.train_semantic, fit) if keep]
        held_semantic = [v for v, keep in zip(split.train_semantic, held) if keep]
        model, judge = fitter.fit(train_ids, split.train_x[fit], train_labels, groups[fit], train_semantic)
        pools = model.predict_candidate_pool_indices(split.train_x[held], held_semantic)
        lookup = {frozenset(dataset.labels[j] for j in c): i for i, c in enumerate(model.candidates)}
        for case, q, obs, pool, truth in zip(held_ids, split.train_x[held], held_semantic, pools, held_labels):
            row = describe_pool(judge, model.candidates, pool, q, obs, dataset.mechanism_cards)
            row.update(case_id=case, group=str(group), truth_candidate=lookup.get(frozenset(truth)))
            inner.append(row)
    if {r['case_id'] for r in inner} != set(split.train_ids) or len(inner) != len(split.train_ids):
        raise ValueError('training OOF inventory differs')
    return {'selected': calibrate(inner), 'training_oof_rows': inner,
            'outer_test_ids_excluded': list(split.eval_ids)}


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'completion.json').exists():
        raise ValueError('completed output exists; use a separate output directory')
    integrity = verify_frozen_inputs()
    source_names = ['ecsa_experiment', 'ecsa_pipeline', 'ecsa_arbitration',
                    'conservative_evidence_judge', 'structured_evidence_judge',
                    'hyperdbdiag_ablation', 'hyperdbdiag_pipeline', 'hypergraph_core',
                    'dbmags_experiment', 'metric_frozen_dataset', 'metric_frozen_schema',
                    'sql_semantic_evidence', 'opdiag_baseline', 'dbaiops_baseline']
    protocol = {
        'name': 'hyperdbdiag_ecsa_v2', 'seed': SEED,
        'model': 'gpt-5.5', 'reasoning_effort': 'xhigh',
        'evaluation': 'six fixed replicate folds; 550 training and 110 evaluation cases per fold',
        'calibration': 'five training-only inner groups; refitted models; unchanged frozen selection rule',
        'arbitration': 'all evaluation cases; no test-call budget or selected-case gate',
        'full_method_fallback': 'calibrated local prediction',
        'llm_without_local_fallback': 'upstream top-1',
        'pool_effect_supplement': 'both arms use upstream top-1 fallback; kept separate from complete method',
        'llm_execution': 'exact frozen primary-input replay including original failures; no repeat substitution',
        'new_model_calls': 0,
        'limitations': ['same development dataset and known scenarios; not a new independent test set',
                        'changes from historical main include local policy, arbitration coverage and prompt/transport',
                        'replay duration is not end-to-end inference latency'],
        'instructions_sha256': digest(INSTRUCTIONS), 'schema_sha256': digest(SCHEMA),
        'source_sha256': {f'src/{name}.py': file_digest(Path('src', name + '.py')) for name in source_names},
        'frozen_inputs': integrity,
    }
    write_json(output / 'protocol.json', protocol)
    started = time.monotonic()
    old_prepared = load(FROZEN / 'prepared.json')
    old_rows = {r['case_id']: r for r in old_prepared['rows']}
    old_decisions = load(FROZEN / 'decisions.json')
    old_results = {(r['case_id'], r['arm']): r for r in old_decisions if r['draw'] == 'primary'}
    store = FrozenDecisionStore(old_prepared['jobs'], old_decisions)
    cards, _ = components._load_root_mechanism_cards()
    dataset = components._load_dbmags(components.DEFAULT_DBMAGS_ROOT, cards)
    frozen = load_frozen_metric_dataset(components.DEFAULT_DBMAGS_ROOT)
    fitter = Fitter(dataset)
    results, folds = [], []
    for split in dataset.splits:
        split.validate(dataset.feature_count, dataset.labels)
        print(f'Refitting and calibrating {split.split_id}', flush=True)
        calibration = calibrate_fold(dataset, split, fitter)
        previous = load(CALIBRATION / (split.split_id.replace(':', '_') + '.json'))
        if calibration != previous:
            raise ValueError(f'refitted calibration differs from the frozen policy: {split.split_id}')
        write_json(output / 'calibration' / (split.split_id.replace(':', '_') + '.json'), calibration)
        model, _ = fitter.fit(split.train_ids, split.train_x, split.train_labels,
                              split.oof_groups, split.train_semantic)
        judge = ConservativeEvidenceJudge(calibration['selected']).fit(
            split.train_x, split.train_labels, split.oof_groups, dataset.labels, dataset.feature_names)
        pipeline = ConservativeECSAPipeline(model, judge, dataset.labels, dataset.feature_names,
                                           dataset.mechanism_cards, split.train_x, split.train_labels,
                                           split.oof_groups, split.train_semantic,
                                           dataset.require_discriminating_semantic_for_llm)
        # Predictions are generated before labels/identifiers are attached for scoring.
        details = pipeline.predict_details(split.eval_x, split.eval_semantic, store)
        for case, truth, detail in zip(split.eval_ids, split.eval_labels, details):
            previous = old_rows[case]
            for actual, expected in [('baseline', 'baseline'), ('local_prediction', 'calibrated_local'),
                                     ('full_pool', 'full_pool'), ('retained_pool', 'conservative_pool')]:
                if detail[actual] != previous[expected]:
                    raise ValueError(f'new inference differs: {case}/{actual}')
            for arm, field, saved_field in [
                ('llm_full_pool', 'llm_without_local_prediction', 'predicted_labels'),
                ('conservative_filtered_pool', 'predicted_labels', 'local_fallback_prediction'),
                ('conservative_filtered_pool', 'filtered_common_fallback_prediction', 'predicted_labels'),
            ]:
                if detail[field] != old_results[(case, arm)][saved_field]:
                    raise ValueError(f'replayed prediction differs: {case}/{field}')
            if list(truth) != previous['expected_labels'] or split.split_id != previous['fold']:
                raise ValueError('frozen case/fold identity differs')
            results.append({'case_id': case, 'fold': split.split_id,
                            'expected_labels': list(truth), **detail})
        folds.append({'outer_split': split.split_id, 'training_count': len(split.train_ids),
                      'evaluation_count': len(split.eval_ids), 'train_evaluation_overlap': [],
                      'calibration': calibration['selected']})
        print(f'  verified {len(results)}/660 cases; all regenerated requests matched', flush=True)
    if len(results) != 660 or len({r['case_id'] for r in results}) != 660 or store.lookups != 1320:
        raise ValueError('incomplete experiment')
    history = load(OLD_ABLATION)['datasets'][dataset.name]
    methods = {}
    for name in ['ordinary_binary_graph', 'hypergraph_without_epdg', 'hypergraph']:
        source = history['methods'][name]
        by_id = {r['case_id']: r for r in source['results']}
        if set(by_id) != set(frozen.case_ids):
            raise ValueError('unchanged ablation cohort differs')
        rows = _case_rows(frozen, frozen.case_ids, [by_id[c]['predicted_labels'] for c in frozen.case_ids])
        methods[name] = _summary(rows, dataset.labels, {'source': str(OLD_ABLATION), 'unchanged_component': True})
        methods[name]['stage'] = copy.deepcopy(source['stage'])
        if name == 'hypergraph':
            if any(set(r['baseline']) != set(by_id[r['case_id']]['predicted_labels']) for r in results):
                raise ValueError('refitted upstream predictions differ from registered baseline')
    displays = {'hypergraph_local_judge': 'Local Structured Discriminator (w/o LLM)',
                'hypergraph_llm': 'LLM Arbitration (w/o Local Filtering)',
                'hyperdbdiag': 'HyperDBDiag'}
    for name, field in METHOD_FIELDS.items():
        rows = _case_rows(frozen, [r['case_id'] for r in results], [r[field] for r in results])
        methods[name] = _summary(rows, dataset.labels, {'display_name': displays[name]})
        methods[name]['stage'] = {'estimand_status': 'evaluated', 'display_name': displays[name],
                                  'protocol': protocol['name'], 'local_filter': name != 'hypergraph_llm',
                                  'llm': name != 'hypergraph_local_judge',
                                  'test_call_budget': None,
                                  'fallback': 'upstream_top1' if name == 'hypergraph_llm' else 'calibrated_local'}
    ablation = {'protocol': protocol,
                'datasets': {dataset.name: {'folds': folds, 'methods': methods, 'metadata': dataset.metadata}}}
    write_json(output / 'ablation.json', ablation)
    main = compare_saved_predictions(ablation, load(OLD_MAIN), frozen, SEED)
    write_json(output / 'main_comparison.json', main)
    write_json(output / 'predictions.json', results)
    with (output / 'predictions.csv').open('w', newline='') as stream:
        fields = ['case_id', 'fold', 'expected_labels', 'baseline', *METHOD_FIELDS.values(),
                  'filtered_common_fallback_prediction', 'full_llm_status', 'filtered_llm_status',
                  'full_pool_size', 'retained_pool_size']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fields if k in r}
            for k, value in row.items():
                if isinstance(value, list): row[k] = '+'.join(sorted(value))
            row.update(full_pool_size=len(r['full_pool']), retained_pool_size=len(r['retained_pool']))
            writer.writerow(row)
    verification = {'status': 'passed', 'cases': len(results), 'fresh_unique_model_fits': len(fitter.cache),
                    'calibration_folds_rebuilt_and_matched': len(folds),
                    'all_deterministic_predictions_and_pools_match': True,
                    'exact_primary_request_replays': store.lookups,
                    'unique_primary_requests_reused': len(store.responses),
                    'new_model_calls': 0, 'repeat_draws_used': 0,
                    'primary_statuses': {arm: dict(Counter(r[f'{arm}_llm_status'] for r in results))
                                         for arm in ['full', 'filtered']},
                    'execution_seconds_not_llm_latency': time.monotonic() - started,
                    'input_integrity': verify_frozen_inputs()}
    for name, expected in protocol['source_sha256'].items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f'source changed during execution: {name}')
    write_json(output / 'verification.json', verification)
    write_json(output / 'completion.json', {'status': 'experiment_complete',
               'files_sha256': {str(p.relative_to(output)): file_digest(p)
                               for p in sorted(output.rglob('*.json')) if p.name != 'completion.json'}})
    print(json.dumps({'complete': True, 'verification': verification,
                      'main': {k: v['overall'] for k, v in main['methods'].items()}}, indent=2), flush=True)
    return ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
