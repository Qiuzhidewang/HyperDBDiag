#!/usr/bin/env python3
"""Independently audit saved predictions and export the affected paper results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def scores(rows, field='predicted_labels'):
    pairs = [(set(r['expected_labels']), set(r[field])) for r in rows]
    tp = sum(len(t & p) for t, p in pairs)
    fp = sum(len(p - t) for t, p in pairs)
    fn = sum(len(t - p) for t, p in pairs)
    correct = sum(t == p for t, p in pairs)
    return {'sample_count': len(rows), 'correct': correct, 'tp': tp, 'fp': fp, 'fn': fn,
            'exact_set_accuracy': correct / len(rows),
            'component_precision': tp / (tp + fp) if tp + fp else 0.,
            'component_recall': tp / (tp + fn) if tp + fn else 0.,
            'component_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def audit(output):
    main = load(output / 'main_comparison.json')
    ablation = next(iter(load(output / 'ablation.json')['datasets'].values()))
    predictions = load(output / 'predictions.json')
    direct = {}
    for name, method in main['methods'].items():
        actual = scores(method['results'])
        for key in ('sample_count', 'exact_set_accuracy', 'component_precision', 'component_recall', 'component_f1'):
            assert abs(actual[key] - method['overall'][key]) < 1e-12, (name, key)
        for field, key in [('by_root_cardinality', 'cardinality'), ('by_occurrence_pattern', 'pattern')]:
            for group, summary in method[field].items():
                rows = [r for r in method['results'] if
                        (str(len(r['expected_labels'])) if key == 'cardinality' else r['occurrence_pattern']) == group]
                independent = scores(rows)
                for metric in ('sample_count', 'exact_set_accuracy', 'component_f1'):
                    assert abs(independent[metric] - summary[metric]) < 1e-12
        direct[name] = actual
    for name, method in ablation['methods'].items():
        actual = scores(method['results'])
        for key in ('exact_set_accuracy', 'component_f1'):
            assert abs(actual[key] - method['overall'][key]) < 1e-12
    old = load('runs/dbmags-main-comparison/full_report_llm_xhigh.json')
    for name in ('opdiag', 'dbaiops'):
        old_by_id = {r['case_id']: r for r in old['methods'][name]['results']}
        for row in main['methods'][name]['results']:
            assert row['predicted_labels'] == old_by_id[row['case_id']]['predicted_labels']
    main_by_id = {r['case_id']: r['predicted_labels'] for r in main['methods']['hyperdbdiag']['results']}
    ablation_by_id = {r['case_id']: r['predicted_labels'] for r in ablation['methods']['hyperdbdiag']['results']}
    assert main_by_id == ablation_by_id == {r['case_id']: r['predicted_labels'] for r in predictions}
    csv_rows = list(csv.DictReader((output / 'predictions.csv').open()))
    for r in csv_rows:
        for field in ('expected_labels', 'predicted_labels'):
            r[field] = r[field].split('+')
    csv_score = scores(csv_rows)
    assert csv_score == direct['hyperdbdiag']
    fallback_differences = [r['case_id'] for r in predictions
                           if set(r['predicted_labels']) != set(r['filtered_common_fallback_prediction'])]
    verification = load(output / 'verification.json')
    assert verification['new_model_calls'] == 0 and verification['repeat_draws_used'] == 0
    result = {'status': 'passed', 'independent_main_counts': direct,
              'main_and_ablation_predictions_identical_for_all_cases': True,
              'all_cardinality_and_temporal_groups_recounted': True,
              'csv_and_json_counts_agree': True, 'baseline_predictions_preserved': True,
              'complete_vs_common_fallback_changed_cases': fallback_differences,
              'new_model_calls': 0, 'repeat_draws_used': 0}
    write(output / 'independent_audit.json', result)
    return main, ablation, predictions, old


def case_audit(output, main, old):
    old_rows = {name: {r['case_id']: r for r in method['results']}
                for name, method in old['methods'].items()}
    new = {r['case_id']: r for r in main['methods']['hyperdbdiag']['results']}
    same_scenario = []
    for case, row in old_rows['hyperdbdiag'].items():
        if row['scenario'] != 'customer_missing_index_implicit_conversion' or len(row['expected_labels']) != 2:
            continue
        top1 = {}
        for method, field in [('opdiag', 'anomaly_probabilities'), ('dbaiops', 'root_scores')]:
            values = old_rows[method][case]['diagnostics'][field]
            top1[method] = max(values, key=values.get)
        same_scenario.append({'case_id': case, 'condition': row['collection_condition'],
                              'expected_labels': row['expected_labels'], 'old_prediction': row['predicted_labels'],
                              'new_prediction': new[case]['predicted_labels'], 'baseline_top1_from_saved_scores': top1,
                              'old_correct': set(row['predicted_labels']) == set(row['expected_labels']),
                              'new_correct': set(new[case]['predicted_labels']) == set(row['expected_labels'])})
    second = new['case-0017']
    operator = load('runs/dbmags-operator-bound-v4/full_report.json')
    sql_cases = []
    for case in ['obv4-variant-4-230-implicit_conversion_missing_index',
                 'obv4-variant-4-231-implicit_conversion_missing_index']:
        folder = Path('data/dbmags_operator_bound_v4/cases') / case
        truth = load(folder / 'ground_truth.json')
        candidates = {r['sql_id']: r for r in load(folder / 'blind_candidates.json')['candidates']}
        sql_cases.append({'case_id': case,
                          'target_sql': {root: candidates[sid]['sql'] for root, sid in truth['target_sql_ids'].items()},
                          'rankings': {name: [r for r in method['results'] if r['case_id'] == case]
                                       for name, method in operator['methods'].items()}})
    result = {
        'missed_diagnosis': {
            'status': 'historical scenario identified; unique original sample ID is not recorded',
            'history_source': 'rollout-2026-08-12T14-30-30-019ff4aa-5200-7301-ae6d-390b5a6d987c.jsonl:13458',
            'policy': 'audit every current mixed-root case in the named scenario; do not silently replace the original with a successful case',
            'current_scenario_cases': same_scenario,
            'warning': 'single-output baseline description differs from the adapted multi-root main comparison'},
        'misdiagnosis': {
            'status': 'original sample recovered and prediction verified', 'case_id': 'case-0017',
            'history_source': 'rollout-2026-08-12T14-30-30-019ff4aa-5200-7301-ae6d-390b5a6d987c.jsonl:27693',
            'expected_labels': second['expected_labels'],
            'old_hyperdbdiag': old_rows['hyperdbdiag']['case-0017']['predicted_labels'],
            'new_hyperdbdiag': second['predicted_labels'],
            'baselines': {name: old_rows[name]['case-0017']['predicted_labels'] for name in ('opdiag', 'dbaiops')},
            'condition': second['collection_condition'],
            'wording_note': 'the recorded case is sequential (b_then_a), not temporally overlapping'},
        'sql_operator': {'status': 'unchanged conditional localization stage; saved query and rank records checked',
                         'matching_records': sql_cases, 'new_llm_calls': 0},
    }
    write(output / 'case_study_audit.json', result)
    return result


def plot(output, main):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 1.5})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.16, top=.78, wspace=.16)
    exported = {}
    for ax, card, title in zip(axes, ['1', '2'], ['(a) Single-root', '(b) Mixed-root']):
        values = [main['methods'][name]['by_root_cardinality'][card]
                  for name in ['opdiag', 'dbaiops', 'hyperdbdiag']]
        exact = [100 * r['exact_set_accuracy'] for r in values]
        f1 = [100 * r['component_f1'] for r in values]
        positions = np.arange(3)
        first = ax.bar(positions - .115, exact, .23, color='#4e79a7', edgecolor='#34424a', linewidth=1.4, label='Exact')
        second = ax.bar(positions + .115, f1, .23, color='#f58510', edgecolor='#34424a', linewidth=1.4, label='Component F1')
        for bars in [first, second]:
            bars[-1].set_edgecolor('#165841'); bars[-1].set_linewidth(3)
        for x, a, b in zip(positions, exact, f1):
            ax.text(x, max(a, b) + 3, f'{a:.1f}/{b:.1f}', ha='center', va='bottom', fontsize=14, color='#263238')
        ax.set_title(title, fontsize=19, fontweight='bold', pad=15)
        ax.set_xticks(positions, ['OpDiag', 'DBAIOps', 'HyperDBDiag'])
        ax.set_ylim(0, 108); ax.set_yticks(np.arange(0, 101, 20)); ax.set_xlim(-.48, 2.62)
        ax.grid(axis='y', color='#d5dce0', linewidth=1); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        exported[card] = {'Exact': exact, 'Component F1': f1}
    axes[0].set_ylabel('Accuracy (%)', fontsize=18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2,
               frameon=False, fontsize=18, bbox_to_anchor=(.5, 1), handlelength=1)
    fig.savefig(output / 'figure5_updated.png', dpi=220, facecolor='white')
    fig.savefig(output / 'figure5_updated.pdf', facecolor='white')
    plt.close(fig)
    write(output / 'figure5_data.json', exported)


def report(output, main, ablation, predictions, old, cases):
    pct = lambda x: f'{100*x:.2f}'
    lines = ['# HyperDBDiag 新版统一实验结果', '',
             '采用训练侧校准的保守本地筛选、全量 LLM 仲裁，弃权或失败时回退到本地判断。', '',
             '本次重新拟合 21 组不同训练划分，重建六折内部校准、全部本地预测、候选池与证据请求。1,320 个逻辑请求逐字节对应已冻结的主调用输入，复用 855 个不同主请求的结果；新增 LLM 调用为 0。没有采用重复调用中更好的答案，历史失败记录保持原样。', '',
             '## 整体结果', '', '| 方法 | Exact (%) | Precision (%) | Recall (%) | F1 (%) |', '|---|---:|---:|---:|---:|']
    for name, method in main['methods'].items():
        o = method['overall']
        lines.append('| ' + name + ' | ' + ' | '.join(pct(o[k]) for k in ['exact_set_accuracy', 'component_precision', 'component_recall', 'component_f1']) + ' |')
    lines += ['', '## 受影响的分组实验', '', '| HyperDBDiag 子集 | 样本数 | 原 Exact (%) | 新 Exact (%) | 新 F1 (%) |', '|---|---:|---:|---:|---:|']
    for field, group, label in [('by_root_cardinality','1','单根因'),('by_root_cardinality','2','多根因'),
                                ('by_occurrence_pattern','sequential','先后发生'),('by_occurrence_pattern','overlap','重叠发生')]:
        r = main['methods']['hyperdbdiag'][field][group]
        previous = old['methods']['hyperdbdiag'][field][group]
        lines.append(f"| {label} | {r['sample_count']} | {pct(previous['exact_set_accuracy'])} | {pct(r['exact_set_accuracy'])} | {pct(r['component_f1'])} |")
    lines += ['', '## 消融与主实验一致性', '', '| 变体 | Exact (%) | F1 (%) |', '|---|---:|---:|']
    for name, m in ablation['methods'].items():
        lines.append(f"| {m['stage'].get('display_name', name)} | {pct(m['overall']['exact_set_accuracy'])} | {pct(m['overall']['component_f1'])} |")
    lines += ['', '完整方法在主对比与消融中使用逐例完全相同的预测。共同超图回退只用于候选筛选效应的辅助对照；它与完整方法总分相同但有 7 例预测不同，未混用。', '',
              '相对 OpDiag 的 Exact 提高 45.30 个百分点，相对 DBAIOps 提高 54.55 个百分点。按原方案的场景×采集重复块重采样，区间分别为 [40.30, 50.30] 与 [49.70, 59.09] 个百分点。这是同一开发数据集上的结果，不能称为全新独立测试或未知场景泛化。', '',
              '新旧完整方法的差异包含本地读出、筛选、LLM 覆盖与仲裁配置变化，不能将全部差额归因于本地判别器。', '',
              '## 案例核对', '',
              '- 误诊案例 `case-0017`：新版仍正确输出 `query_whole_table + record_lock`，两基线仍误报 `group_by`。该样本实际为先后发生，正文“同时”的时间描述需校正。',
              '- 漏诊案例：历史稿只绑定了场景，没有可恢复的唯一案例编号。已审计该场景全部 18 个双根因样本，保留成功与失败结果；不能宣称原案例已经逐例复现，也未擅自换成一个成功样本。另需区分案例中的单标签输出与主实验的多根因适配输出。',
              '- SQL／算子案例：原文两条 SQL 可对应 variant-4 的 230、231 两条记录；HyperDBDiag 两个根因的 SQL 与算子均排第一，OpDiag 在缺失索引分支的 SQL 排第三。该条件定位实验未受判别器影响。', '',
              '## 保留与待同步项目', '',
              'OpDiag、DBAIOps、消融前三项、SQL／算子定位和图规模／遍历效率结果保持原样。图 5 已按新统计重新生成，保留原有双面板、配色和标签结构，提供 PNG 与矢量 PDF。', '',
              '实验数据和代码已经更新；本次没有覆盖中文主稿、英文主稿或论文原图。后续写回涉及表 1 前四列、表 2、图 5、引言中的 Exact 提升数值、实验配置、消融配置说明，以及上述案例时间／样本说明。表 4 的已确认数值不变。', '',
              '## 代码清理', '',
              '删除活跃目录中的 7 个阶段性实验／分析脚本和 2 个对应旧测试文件，将公共请求构造、响应校验与推理路径合并。有效测试已迁移并补充。删除前的源文件、测试及 README 保存在 `archives/pre-unification-20260930.zip`，历史原始结果、协议和数据未删改。', '',
              '主入口为根目录 `reproduce.py`；`src/hyperdbdiag_ablation.py` 和 `src/dbmags_experiment.py` 默认指向新版统一结果。旧两候选流程中仍被共同证据构造、输出校验及回归测试使用的代码保留。', '']
    (output / '实验结果与核对说明.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/hyperdbdiag-unified-20260930'))
    args = parser.parse_args()
    main_report, ablation, predictions, old = audit(args.output)
    cases = case_audit(args.output, main_report, old)
    plot(args.output, main_report)
    report(args.output, main_report, ablation, predictions, old, cases)
    print('Independent metrics, all subgroups, case audit and Figure 5 exported.')


if __name__ == '__main__':
    main()
