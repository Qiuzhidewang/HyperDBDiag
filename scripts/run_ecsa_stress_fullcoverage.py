#!/usr/bin/env python3
"""Extend the frozen ECSA evidence stress test to every case, with clean repeats.

The 120-case pilot remains immutable. Its exact first responses are retained,
including invalid responses; all missing primary inputs receive one new call.
The extra clean repeat is an independent second draw and never replaces a primary.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_ecsa_evidence_stress as pilot
from ecsa_pipeline import resolve_response

OUT=ROOT/'runs/ecsa-evidence-stress-full660-20260930-v1'
PILOT=pilot.OUT
WORKERS=24
ARMS=(*pilot.ARMS,'clean_repeat')
NAMES={**pilot.NAMES,'clean_repeat':'原始输入独立重复'}


def load(p): return json.loads(Path(p).read_text())
def write(p,v): pilot.write_json(p,v)


def preserved(out):
    p=load(out/'protocol.json')
    assert p['script_sha256']==pilot.file_digest(Path(__file__))
    for path,expected in p['protected_sha256'].items():
        assert pilot.file_digest(ROOT/path)==expected,path
    assert load(out/'prepared.json')['protocol_sha256']==pilot.digest(p)


def prepare(out):
    if (out/'prepared.json').exists():
        preserved(out);return
    assert not (out/'protocol.json').exists(),'do not overwrite a partial frozen protocol'
    old_protocol=load(PILOT/'protocol.json')
    assert old_protocol['script_sha256']==pilot.file_digest(ROOT/'scripts/run_ecsa_evidence_stress.py')
    for path,expected in load(PILOT/'completion.json')['files_sha256'].items():
        assert pilot.file_digest(PILOT/path)==expected,path
    canonical=load(pilot.CANONICAL/'predictions.json')
    assert len(canonical)==660 and len({r['case_id'] for r in canonical})==660
    source_jobs={j['case_id']:j for j in load(pilot.HISTORY/'prepared.json')['jobs']
                 if j['arm']=='conservative_filtered_pool' and j['draw']=='primary'}
    old_prepared=load(PILOT/'prepared.json')
    old_inputs=old_prepared['requests']
    old_responses={r['packet_sha256']:r for r in pilot.saved(PILOT,old_prepared).values()}
    assert len(old_responses)==713
    old_cases={r['case_id'] for r in old_prepared['rows']}
    rows=[];requests={};checks=[]
    for source in sorted(canonical,key=lambda r:(r['fold'],r['case_id'])):
        job=source_jobs[source['case_id']];packet=job['packet']
        assert pilot.digest(packet)==job['packet_sha256']==source['filtered_packet_sha256']
        assert job['local_fallback']==source['local_prediction']
        assert len(packet['candidate_records'])>=2
        checks.extend({'case_id':source['case_id'],**v} for v in pilot.guard_probes(packet,source['local_prediction']))
        for arm in ARMS:
            modified=pilot.perturb(packet,'clean' if arm=='clean_repeat' else arm,source['case_id'])
            pilot.verify_perturbation(packet,modified,'clean' if arm=='clean_repeat' else arm)
            key=pilot.digest(modified)
            requests[key]=modified
            draw='repeat' if arm=='clean_repeat' else 'primary'
            rows.append(dict(case_id=source['case_id'],fold=source['fold'],arm=arm,draw=draw,
                job_id=f'{draw}:{key}',packet_sha256=key,original_packet_sha256=pilot.digest(packet),
                expected_labels=source['expected_labels'],local_prediction=source['local_prediction'],
                historical_prediction=source['predicted_labels'],retained_pool=source['retained_pool'],
                input_changed=modified!=packet,cohort='pilot120' if source['case_id'] in old_cases else 'additional540'))
    assert len(rows)==4620
    first={}
    for row in rows:
        first.setdefault(row['job_id'],row)
        assert first[row['job_id']]['local_prediction']==row['local_prediction']
    reused=[]
    for key,response in old_responses.items():
        assert key in requests and requests[key]==old_inputs[key]
        resolve_response(requests[key],response,first[f'primary:{key}']['local_prediction'])
        reused.append({**response,'job_id':f'primary:{key}','response_source':'pilot_exact_primary'})
    reused_keys={r['job_id'] for r in reused}
    new_order=sorted(set(first)-reused_keys)
    random.Random(pilot.SEED+1).shuffle(new_order)
    protected={**old_protocol['protected_sha256']}
    protected.update({str(p.relative_to(ROOT)):pilot.file_digest(p) for p in PILOT.rglob('*') if p.is_file()})
    protected.update({str(p.relative_to(ROOT)):pilot.file_digest(p) for p in
                      [ROOT/'scripts/run_ecsa_evidence_stress.py',ROOT/'scripts/audit_ecsa_evidence_stress.py']})
    protocol=dict(name='ecsa_evidence_stress_full660_v1',registered_utc=pilot.utc(),seed=pilot.SEED,
        expansion_requested_after_pilot=True,pilot_results_already_known=True,
        new_outcomes_seen_before_freeze=False,selection='all 660 canonical cases, six folds of 110; no correctness or confidence filtering',
        per_fold_cardinality='44 single-root and 66 double-root cases',arms=list(ARMS),arm_names=NAMES,
        transformations=old_protocol['transformations'],lure_text=pilot.LURE,
        perturbation_stage=old_protocol['perturbation_stage'],
        inference={**old_protocol['inference'],'workers':WORKERS,'historical_inferences_reused':True,
                   'primary_only':False,'repeat_definition':'one separate new clean-input draw per case; same prompt/input/schema/model; never substitute for primary',
                   'transport_note':'pilot used 8 concurrent requests; expansion uses 24; no latency or throughput comparison'},
        reuse='reuse all 713 pilot exact primary responses, including failures; packets/prompt/schema/client version/model checked; do not reconstruct missing raw outputs from earlier main results',
        primary_metrics=old_protocol['metrics'],denominators='all 660 cases per arm for final metrics; received parseable objects for raw-output violations; failures and abstentions separate',
        clean_repeat_plan='compare clean_repeat with clean using paired correctness and prediction changes; contrast with order_only; one repeated draw does not fully identify all stochastic/order interactions',
        statistical_plan='report paired wins/losses, six-fold and pilot120/additional540 breakdowns; no retuning or cherry-picked repeated calls; no claim of external generalization',
        planned_logical_evaluations=len(rows),planned_unique_primary_inputs=sum(j.startswith('primary:') for j in first),
        planned_clean_repeat_calls=sum(j.startswith('repeat:') for j in first),
        reused_primary_calls=len(reused),planned_new_calls=len(new_order),synthetic_guard_checks=len(checks),
        outage_policy='stop scheduling after 8 consecutive transport failures; preserve recorded failures and never replace them with favorable draws',
        protected_sha256=protected,script_sha256=pilot.file_digest(Path(__file__)))
    out.mkdir(parents=True,exist_ok=True)
    write(out/'protocol.json',protocol)
    write(out/'prepared.json',dict(protocol_sha256=pilot.digest(protocol),requests=requests,rows=rows,
                                  jobs=first,new_request_order=new_order))
    (out/'reused_primary_responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reused))
    write(out/'synthetic_guard_checks.json',{'status':'passed','count':len(checks),'checks':checks})
    print(json.dumps({k:protocol[k] for k in ['planned_logical_evaluations','planned_unique_primary_inputs',
                     'planned_clean_repeat_calls','reused_primary_calls','planned_new_calls','synthetic_guard_checks']}),flush=True)


def saved(out,prepared):
    result={}
    for name in ['reused_primary_responses.jsonl','new_responses.jsonl']:
        path=out/name
        if not path.exists(): continue
        for line in path.read_text().splitlines():
            row=json.loads(line);job=prepared['jobs'][row['job_id']]
            assert row['job_id'] not in result
            assert row['packet_sha256']==job['packet_sha256'] and row['draw']==job['draw']
            if 'raw_response' in row: assert pilot.digest(row['raw_response'])==row['raw_response_sha256']
            result[row['job_id']]=row
    return result


def invoke(job,packet,client):
    row=pilot.invoke_one(job['packet_sha256'],packet,job['local_prediction'],client)
    row.update(job_id=job['job_id'],draw=job['draw'],response_source='new_'+job['draw'])
    return row


def execute(out):
    preserved(out)
    p=load(out/'prepared.json');responses=saved(out,p)
    keys=[k for k in p['new_request_order'] if k not in responses]
    if not keys: return
    client,meta=pilot.configured_codex_client(out)
    old_meta=load(PILOT/'llm_configuration.json')
    for k in ['model','reasoning_effort','transport','cli_version','instructions_sha256','schema_sha256',
              'timeout_seconds','ignore_user_provider_config','model_tool_use_permitted']:
        assert meta[k]==old_meta[k],f'client mismatch: {k}'
    meta.update(workers=WORKERS,raw_visible_responses_preserved=True,primary_and_repeat_kept_separate=True,
                input_protocol_sha256=p['protocol_sha256'])
    conf=out/'llm_configuration.json'
    if conf.exists(): assert load(conf)==meta
    else: write(conf,meta)
    start=time.monotonic();error=None;failures=0;queue=iter(keys)
    with (out/'new_responses.jsonl').open('a',buffering=1) as sink,ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={}
        def replenish():
            while error is None and len(futures)<WORKERS:
                key=next(queue,None)
                if key is None: break
                job=p['jobs'][key]
                futures[pool.submit(invoke,job,p['requests'][job['packet_sha256']],client)]=key
        replenish()
        while futures:
            finished,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in finished:
                key=futures.pop(future)
                try: row=future.result()
                except Exception as exc: error=error or exc;continue
                sink.write(json.dumps(row,ensure_ascii=False)+'\n');responses[key]=row
                failures=failures+1 if row['status']=='fallback_transport' else 0
                if failures>=8: error=error or RuntimeError('sustained transport outage; incomplete')
                new=[r for r in responses.values() if r['response_source']!='pilot_exact_primary']
                progress=dict(completed_new=len(new),planned_new=len(p['new_request_order']),
                    reused_primary=713,total_saved=len(responses),total_planned=len(p['jobs']),
                    new_statuses=dict(Counter(r['status'] for r in new)),
                    process_seconds=time.monotonic()-start,updated_utc=pilot.utc())
                write(out/'progress.json',progress)
                if len(new)%40==0 or len(new)==len(p['new_request_order']): print(json.dumps(progress),flush=True)
            replenish()
    if error: raise error
    preserved(out)


def paired(group,reference):
    return dict(wins=sum(set(r['predicted_labels'])==set(r['expected_labels']) and set(reference[r['case_id']]['predicted_labels'])!=set(r['expected_labels']) for r in group),
                losses=sum(set(r['predicted_labels'])!=set(r['expected_labels']) and set(reference[r['case_id']]['predicted_labels'])==set(r['expected_labels']) for r in group),
                prediction_flips=sum(set(r['predicted_labels'])!=set(reference[r['case_id']]['predicted_labels']) for r in group))


def summary(group,reference):
    s=dict(metrics=pilot.metrics(group),statuses=dict(Counter(r['status'] for r in group)),
           responses_received=sum(r['response_received'] for r in group),
           parseable_responses=sum(r['raw_audit'].get('parseable_object',False) for r in group),
           final_out_of_pool=sum(r['final_out_of_pool'] for r in group),paired_vs_clean=paired(group,reference))
    for k in ['proposed_candidate_outside_pool','comparison_candidate_outside_pool','unknown_citation','raw_abstention']:
        s[k]=sum(r['raw_audit'].get(k,False) for r in group)
    s['validation_errors']=dict(Counter(r['validation_error'] for r in group if r['validation_error']))
    return s


def analyze(out):
    preserved(out)
    p=load(out/'prepared.json');responses=saved(out,p)
    assert set(responses)==set(p['jobs']),'incomplete; do not issue final scores'
    rows=[]
    for source in p['rows']:
        response=responses[source['job_id']];packet=p['requests'][source['packet_sha256']]
        if 'raw_response' in response:
            replay=pilot.evaluate_visible(response['raw_response'],packet,source['local_prediction'])
            for k in ['status','raw_audit','predicted_labels']: assert replay[k]==response[k]
        labels=resolve_response(packet,response,source['local_prediction'])
        rows.append({**source,'predicted_labels':labels,'status':response['status'],
            'response_received':'raw_response' in response,'raw_audit':response.get('raw_audit',{}),
            'validation_error':response.get('validation_error'),'response_source':response['response_source'],
            'final_out_of_pool':not any(set(labels)==set(c['root_labels']) for c in packet['candidate_records'])})
    clean={r['case_id']:r for r in rows if r['arm']=='clean'}
    summaries=[]
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        assert len(group)==660 and len({r['case_id'] for r in group})==660
        s={'arm':arm,**summary(group,clean)}
        s['by_fold']={f:summary([r for r in group if r['fold']==f],clean) for f in sorted({r['fold'] for r in group})}
        s['by_cohort']={c:summary([r for r in group if r['cohort']==c],clean) for c in ['pilot120','additional540']}
        accepted=[r for r in group if r['status']=='accepted_selection']
        s['accepted_selection_metrics']=pilot.metrics(accepted) if accepted else None
        summaries.append(s)
    repeated={r['case_id']:r for r in rows if r['arm']=='clean_repeat'}
    local=[{**r,'predicted_labels':r['local_prediction']} for r in clean.values()]
    results=dict(status='complete',cases=660,arms=summaries,local_metrics=pilot.metrics(local),
        unique_primary_inputs=sum(k.startswith('primary:') for k in responses),
        clean_repeat_calls=sum(k.startswith('repeat:') for k in responses),
        reused_primary_calls=713,new_calls=len(responses)-713,
        order_vs_clean_repeat=paired([r for r in rows if r['arm']=='order_only'],repeated),
        protocol_sha256=p['protocol_sha256'])
    write(out/'predictions.json',rows);write(out/'results.json',results)
    with (out/'summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['arm','n','correct','Exact_pct','F1_pct','raw_out_of_pool','final_out_of_pool','abstain','invalid','transport','wins','losses','flips'])
        for s in summaries:
            m=s['metrics'];st=s['statuses'];pa=s['paired_vs_clean']
            w.writerow([s['arm'],m['n'],m['correct'],100*m['Exact'],100*m['F1'],s['proposed_candidate_outside_pool'],s['final_out_of_pool'],
                        st.get('accepted_abstain',0),st.get('fallback_invalid_response',0),st.get('fallback_transport',0),pa['wins'],pa['losses'],pa['prediction_flips']])
    text='''# ECSA 全部 660 例证据扰动实验（未写回论文）

六种主条件覆盖全部 660 例、六个原有划分，不按正确率挑选样本。沿用试验阶段的证据变换、随机种子、模型、提示词、输出格式、校验器与本地回退；原有 120 例的首次原始响应全部保留。仅补齐缺失输入，没有用更好的重复调用替换旧答案。

另对全部 660 例的原始输入各做一次独立重复调用，单独报告，用于观察模型调用波动。重复结果不替换主结果。该扩展是在已知 120 例结果后按用户要求进行的全量评估，并非新数据集或全盲外部验证。

| 输入条件 | Exact (%) | F1 (%) | 原始候选越界 | 最终结果越界 | 弃权 | 非法输出 | 接口失败 | 相对原始对照改对/改错 | 预测变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for s in summaries:
        m=s['metrics'];st=s['statuses'];pa=s['paired_vs_clean']
        text+=f"| {NAMES[s['arm']]} | {100*m['Exact']:.2f} | {100*m['F1']:.2f} | {s['proposed_candidate_outside_pool']}/{s['parseable_responses']} | {s['final_out_of_pool']}/660 | {st.get('accepted_abstain',0)} | {st.get('fallback_invalid_response',0)} | {st.get('fallback_transport',0)} | {pa['wins']}/{pa['losses']} | {pa['prediction_flips']} |\n"
    text+='''
原始越界计数使用接口结构化输出之后、本地校验之前的可见 JSON 响应，不是完全无约束的自由文本生成。最终准确率包含本地回退；弃权、校验失败和接口失败分别报告。零越界不表示候选内一定选对，也不表示证据引用的语义完全正确。

本次在本地筛选后改变证据卡；候选池、本地输出、指标轨迹、独立语义观察、训练模式与机制卡保持不变。因此它检验候选卡受损时的约束与表现，不能直接证明本地判别器或正反证据各自的贡献。候选外诱导使用试验阶段同一条固定文本，不代表覆盖了所有可能攻击。

原始重复组提供调用波动的参照；每例只有一次额外重复，不能把所有顺序和随机因素彻底分离。分折与原 120 例／新增 540 例的结果均保留在 results.json，不将不同批次隐藏为完全同时采样。所有实际运行失败保留，不能把接口失败当成模型理解证据不足。

代码与输入校验、全部人工非法输出检查、原始响应、逐例结果及文件指纹均保存。主方法代码、论文和已有图表未修改。
'''
    (out/'RESULTS_实验结果.md').write_text(text)
    write(out/'completion.json',dict(status='analyzed_pending_independent_audit',completed_utc=pilot.utc(),
         files_sha256={str(f.relative_to(out)):pilot.file_digest(f) for f in out.rglob('*') if f.is_file() and f.name!='completion.json' and f.suffix!='.log'}))
    print(json.dumps({'phase':'analyzed','new_calls':results['new_calls'],'arms':[{k:s[k] for k in ['arm','metrics','statuses','final_out_of_pool']} for s in summaries]}),flush=True)


def main():
    c=argparse.ArgumentParser(description=__doc__);c.add_argument('--phase',choices=['prepare','run','analyze','all'],default='all');c.add_argument('--output',type=Path,default=OUT)
    a=c.parse_args()
    if a.phase in ('prepare','all'): prepare(a.output)
    if a.phase in ('run','all'): execute(a.output)
    if a.phase in ('analyze','all'): analyze(a.output)


if __name__=='__main__': main()
