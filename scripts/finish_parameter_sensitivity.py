#!/usr/bin/env python3
"""Finish analysis and audit after the already running inference process exits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def supplemental_outputs(output):
    """Readable export, descriptive error audit, and final immutable hashes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    results = json.loads((output / 'results.json').read_text())['configurations']
    protocol = json.loads((output / 'protocol.json').read_text())
    rows = json.loads((output / 'predictions.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True)
    base = {'beta': .5, 'theta': 1, 'K': 4}
    for axis, (parameter, grid), symbol in zip(axes, protocol['grids'].items(), ['α', 'β', 'θ', 'K']):
        points = [next((r for r in results if r['parameter'] == parameter and r['value'] == v),
                       results[0] if base.get(parameter) == v else None) for v in grid]
        for key, color, marker in [('Exact', '#4c78a8', 'o'), ('F1', '#f58518', 's')]:
            axis.plot(grid, [100*r['overall'][key] for r in points], color=color,
                      marker=marker, linewidth=2, markersize=5)
            axis.axhline(100*results[0]['overall'][key], color=color,
                         linestyle='--', linewidth=1, alpha=.7)
        axis.set_xlabel(f'({"abcd"[list(axes).index(axis)]}) {symbol}', fontsize=16)
        axis.set_xticks([v for v in grid if parameter != 'beta' or v != .5])
        axis.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axis.tick_params(labelsize=12)
        axis.set_ylim(50, 100)
        axis.set_yticks([50, 60, 70, 80, 90, 100])
        axis.grid(axis='y', color='#d9e2eb', linewidth=.6)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Diagnostic performance (%)', fontsize=14)
    fig.legend(handles=[Line2D([], [], color='#4c78a8', marker='o', label='Exact'),
                        Line2D([], [], color='#f58518', marker='s', label='Component F1'),
                        Line2D([], [], color='#777777', linestyle='--', label='Reported configuration')],
               loc='upper center', ncol=3, frameon=False, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .87], w_pad=1)
    fig.savefig(output / 'parameter_sensitivity_review.pdf', bbox_inches='tight')
    fig.savefig(output / 'parameter_sensitivity_review.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    errors = {}
    for config in protocol['configurations']:
        group = [r for r in rows if r['configuration'] == config['id']]
        counts, statuses = Counter(), Counter()
        for row in group:
            expected, predicted = set(row['expected_labels']), set(row['predicted_labels'])
            extra, missing = predicted - expected, expected - predicted
            kind = ('correct' if not extra and not missing else 'extra_only' if extra and not missing
                    else 'missing_only' if missing and not extra else 'substitution')
            counts[kind] += 1
            if expected != predicted:
                statuses[row['llm_status']] += 1
        errors[config['id']] = {'error_types': dict(counts), 'wrong_prediction_statuses': dict(statuses)}
    (output / 'error_breakdown.json').write_text(json.dumps({
        'analysis_type': 'post-run descriptive analysis; no parameter reselection',
        'definitions': {'missing_only': 'predicted roots are a strict subset of the true root set',
                        'extra_only': 'predicted roots are a strict superset of the true root set',
                        'substitution': 'both extra and missing roots'},
        'configurations': errors}, indent=2)+'\n')


def finalize_manifest(output):
    manifest = json.loads((output / 'completion.json').read_text())
    manifest['hash_scope'] = 'all immutable output files, recursively; excludes this manifest and append-only logs'
    manifest['hashes_finalized_after_postprocessing'] = True
    manifest['files_sha256'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'completion.json' and p.suffix != '.log'}
    (output / 'completion.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/parameter-sensitivity-20260930-v1')
    args = parser.parse_args()
    output = args.output
    if args.wait_pid:
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(20)
    for script, extra in [('run_parameter_sensitivity.py', ['--phase', 'analyze']),
                          ('audit_parameter_sensitivity.py', [])]:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--output', str(output), *extra],
                       cwd=ROOT, check=True)
    results = json.loads((output / 'results.json').read_text())
    audit = json.loads((output / 'independent_audit.json').read_text())
    assert audit['status'] == 'complete_verified'
    rows = results['configurations']
    reference = rows[0]['overall']
    text = f'''# HyperDBDiag 参数敏感性实验结果

已完成 22 组配置，每组使用相同的 660 个样本和六折划分，共 14,520 次样本评估。当前主配置的 660 个最终预测逐例复现。中文稿、英文稿及已有结果和图片均未修改。

## 实验设计

α 取 0、0.2、0.4、0.6、0.8、1；β 取 0、0.2、0.4、0.5、0.6、0.8、1；θ 取 0.5、0.75、1、1.25、1.5；K 取 1、2、3、4、6、8。每次仅改变一个参数，其余设置和本地判别阈值保持为对应外层折的原配置。θ 改变时重新构建训练超图并训练模型。

当前主配置中，α 由训练侧分组验证选定，六折依次为 0.3、0.4、0.4、0.4、0.5、0.3；β=0.5、K=4 也由训练侧选择，θ=1 为固定设置。本次 β>0.5 与 K>4 属于扩展敏感性测试，不会根据测试成绩回选主配置。具体选择规则与训练记录见 protocol.json、prepared.json 及 README_实验记录.md。

主配置 Exact 为 {100*reference['Exact']:.2f}%，component F1 为 {100*reference['F1']:.2f}%。下表所有数值均为完整方法结果；候选覆盖率另列用于解释候选遗漏。

## 完整结果

| 配置 | Exact (%) | F1 (%) | 上游候选覆盖 (%) | 本地筛选后覆盖 (%) |
|---|---:|---:|---:|---:|
'''
    for row in rows:
        text += f"| {row['id']} | {100*row['overall']['Exact']:.2f} | {100*row['overall']['F1']:.2f} | {100*row['upstream_coverage']:.2f} | {100*row['retained_coverage']:.2f} |\n"
    text += '\n## 变化范围\n\n'
    for name in ['alpha', 'beta', 'theta', 'K']:
        group = [r for r in rows if r['parameter'] == name]
        if name != 'alpha':
            group.append(rows[0])
        low = min(group, key=lambda r: r['overall']['Exact'])
        high = max(group, key=lambda r: r['overall']['Exact'])
        text += f"- {name}：Exact 为 {100*low['overall']['Exact']:.2f}%～{100*high['overall']['Exact']:.2f}%。\n"
    text += f'''
## 推断与审计

新运行 {results['new_primary_calls']} 次 GPT-5.5 / xhigh 仲裁，复用 {results['historical_unique_inputs']} 份完全相同输入的历史主调用，并对 {results['analytical_unique_singletons']} 份单候选输入直接确定最终标签。相同输入共用一次主调用，未用重复调用中的更好结果替换。单候选情形未伪造 LLM 输出或 token 量。首轮调用失败和拒答均按原有本地回退规则处理。

新增调用状态：{json.dumps(audit['response_statuses'], ensure_ascii=False)}。

全部候选池已从保存的分支概率独立重算，所有最终 Exact/F1 已独立核对；训练/测试划分、原配置预测及受保护文件校验全部通过。

图：parameter_sensitivity.pdf / parameter_sensitivity.png；汇总数据：sensitivity_summary.csv；逐例结果：predictions.csv / predictions.json；审计：independent_audit.json。

本实验检验现有场景下的参数敏感性，不能替代未知场景或数据噪声鲁棒性测试，也不能据单因素曲线断言不存在参数交互。应区分常用参数邻域的表现与去掉某一分支、只留一个候选等边界设置；测试曲线最高点不自动成为新的主配置。
'''
    (output / 'RESULTS_实验结果.md').write_text(text)
    (output / 'postprocessing_complete.json').write_text(json.dumps({
        'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'independent_audit': audit['status'], 'report': 'RESULTS_实验结果.md'}, indent=2)+'\n')
    supplemental_outputs(output)
    finalize_manifest(output)
    print('Analysis, independent audit, figures and report are complete.', flush=True)


if __name__ == '__main__':
    main()
