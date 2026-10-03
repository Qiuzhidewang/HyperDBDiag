#!/usr/bin/env python3
"""Independent input, raw-output and scoring audit for the frozen ECSA stress run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecsa_arbitration import validate_response


def read(p):
    return json.loads(p.read_text())


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def body(card):
    prefix,remaining=card['summary'].split('. Support: ',1)
    support,counter=remaining.rsplit('. Counterevidence: ',1)
    return prefix,support,counter


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,default=ROOT/'runs/ecsa-evidence-stress-20260930-v1')
    cli.add_argument('--inputs-only',action='store_true')
    args=cli.parse_args();out=args.output
    protocol,prepared=read(out/'protocol.json'),read(out/'prepared.json')
    assert prepared['protocol_sha256']==digest(protocol)
    assert sha(ROOT/'scripts/run_ecsa_evidence_stress.py')==protocol['script_sha256']
    for p,s in protocol['protected_sha256'].items(): assert sha(ROOT/p)==s,p
    canonical=read(ROOT/'runs/hyperdbdiag-unified-20260930/predictions.json')
    truth={r['case_id']:r for r in canonical}
    expected_ids=[]
    for f in sorted({r['fold'] for r in canonical}):
        for n,count in [(1,8),(2,12)]:
            subset=[r for r in canonical if r['fold']==f and len(r['expected_labels'])==n]
            expected_ids += [r['case_id'] for r in sorted(subset,key=lambda r:digest([protocol['seed'],'sample',f,n,r['case_id']]))[:count]]
    assert len(expected_ids)==len(set(expected_ids))==120
    rows=prepared['rows'];requests=prepared['requests']
    assert len(rows)==720
    assert set(prepared['request_order'])==set(requests) and len(prepared['request_order'])==len(requests)
    lookup={(r['case_id'],r['arm']):r for r in rows}
    assert len(lookup)==720
    original={j['case_id']:j['packet'] for j in read(ROOT/'runs/ecsa-full660-20260929/prepared.json')['jobs']
              if j['draw']=='primary' and j['arm']=='conservative_filtered_pool'}
    for arm in protocol['arms']:
        assert {r['case_id'] for r in rows if r['arm']==arm}==set(expected_ids)
    for key,p in requests.items():
        assert digest(p)==key
        assert set(p)=={'task','candidate_records','evidence','requires_direct_semantic_evidence'}
        assert not any(s in json.dumps(p) for s in ['expected_labels','historical_prediction','case-00','local_prediction'])
    for row in rows:
        p=requests[row['packet_sha256']];base=original[row['case_id']];arm=row['arm']
        assert row['expected_labels']==truth[row['case_id']]['expected_labels']
        assert row['fold']==truth[row['case_id']]['fold']
        assert row['local_prediction']==truth[row['case_id']]['local_prediction']
        assert row['original_packet_sha256']==digest(base)==truth[row['case_id']]['filtered_packet_sha256']
        assert row['input_changed']==(p!=base)
        assert {c['candidate_id']:c['root_labels'] for c in p['candidate_records']}=={c['candidate_id']:c['root_labels'] for c in base['candidate_records']}
        assert len(p['candidate_records'])>=2
        assert row['local_prediction'] in [c['root_labels'] for c in p['candidate_records']]
        assert p['requires_direct_semantic_evidence']==base['requires_direct_semantic_evidence']
        sort=lambda xs:sorted(xs,key=lambda c:c['evidence_id'])
        for family,items in p['evidence'].items():
            if family!='candidate_evidence_cards': assert sort(items)==sort(base['evidence'][family])
        cards=p['evidence']['candidate_evidence_cards'];old=base['evidence']['candidate_evidence_cards']
        for c,d in zip(sort(cards),sort(old)):
            assert {k:v for k,v in c.items() if k!='summary'}=={k:v for k,v in d.items() if k!='summary'}
            assert body(c)[0]==body(d)[0]
        if arm=='clean': assert p==base
        elif arm=='identical_support':
            rng=random.Random(int(digest([protocol['seed'],row['case_id'],arm]),16))
            donor=body(old[rng.randrange(len(old))])[1]
            assert all(body(c)[1]==donor for c in cards)
            assert [body(c)[2] for c in cards]==[body(c)[2] for c in old]
        elif arm=='empty_counterevidence':
            assert all(body(c)[2]=='none' for c in cards)
            assert [body(c)[1] for c in cards]==[body(c)[1] for c in old]
        elif arm=='shuffled_evidence':
            a=[s for c in old for s in body(c)[1:]]
            b=[s for c in cards for s in body(c)[1:]]
            assert Counter(a)==Counter(b)
            rng=random.Random(int(digest([protocol['seed'],row['case_id'],arm]),16))
            order=list(range(len(a)))
            while True:
                rng.shuffle(order)
                if all(i!=j for i,j in enumerate(order)): break
            assert b==[a[i] for i in order]
        elif arm=='order_only':
            assert p['candidate_records']==base['candidate_records'][::-1]
            assert all(p['evidence'][k]==v[::-1] for k,v in base['evidence'].items())
        elif arm=='out_of_pool_lure':
            assert cards[1:]==old[1:]
            assert body(cards[0])[1]==body(old[0])[1]+'; '+protocol['lure_text']
            assert body(cards[0])[2]==body(old[0])[2]
            assert 'C99' not in {c['candidate_id'] for c in p['candidate_records']}
    audit={'status':'input_verified','sample_cases':120,'six_fold_single_double_strata':'8 + 12 per fold',
           'logical_evaluations':len(rows),'unique_requests':len(requests),
           'selection_recomputed_without_correctness_filter':True,'all_transformations_independently_checked':True,
           'candidate_inventories_and_other_evidence_unchanged':True,'no_labels_sent_to_model':True,
           'protected_files_unchanged':True}
    if not args.inputs_only:
        responses=[json.loads(s) for s in (out/'primary_responses.jsonl').read_text().splitlines()]
        by_sha={r['packet_sha256']:r for r in responses}
        assert len(by_sha)==len(responses)==len(requests) and set(by_sha)==set(requests)
        predictions=read(out/'predictions.json');summary=read(out/'results.json')
        assert len(predictions)==720
        computed=[]
        for row in rows:
            resp=by_sha[row['packet_sha256']];packet=requests[row['packet_sha256']]
            labels=row['local_prediction'];received='raw_response' in resp;parseable=False;outside=False;unknown=False
            if received:
                assert digest(resp['raw_response'])==resp['raw_response_sha256']
                try:
                    raw=json.loads(resp['raw_response']);parseable=isinstance(raw,dict)
                except ValueError: raw=None
                if parseable:
                    ids={c['candidate_id'] for c in packet['candidate_records']}
                    outside=isinstance(raw.get('candidate_id'),str) and raw['candidate_id'] not in ids
                    evidence={e['evidence_id'] for group in packet['evidence'].values() for e in group}
                    unknown=isinstance(raw.get('evidence_ids'),list) and any(isinstance(s,str) and s not in evidence for s in raw['evidence_ids'])
                try:
                    decision=validate_response(resp['raw_response'],packet)
                    status='accepted_selection' if decision['action']=='SELECT_CANDIDATE' else 'accepted_abstain'
                    if status=='accepted_selection': labels=next(c['root_labels'] for c in packet['candidate_records'] if c['candidate_id']==decision['candidate_id'])
                except (ValueError,KeyError,TypeError): status='fallback_invalid_response'
            else: status='fallback_transport'
            assert status==resp['status'] and labels==resp['predicted_labels']
            assert any(set(labels)==set(c['root_labels']) for c in packet['candidate_records'])
            published=next(r for r in predictions if r['case_id']==row['case_id'] and r['arm']==row['arm'])
            assert labels==published['predicted_labels'] and status==published['status']
            computed.append({**row,'predicted_labels':labels,'status':status,'received':received,'parseable':parseable,'outside':outside,'unknown':unknown})
        for report in summary['arms']:
            group=[r for r in computed if r['arm']==report['arm']]
            pairs=[(set(r['expected_labels']),set(r['predicted_labels'])) for r in group]
            correct=sum(a==b for a,b in pairs);tp=sum(len(a&b) for a,b in pairs)
            errors=sum(len(a^b) for a,b in pairs)
            assert report['metrics']['correct']==correct and report['metrics']['Exact']==correct/120
            assert report['metrics']['F1']==2*tp/(2*tp+errors)
            assert report['statuses']==dict(Counter(r['status'] for r in group))
            for key,name in [('responses_received','received'),('parseable_responses','parseable'),
                             ('proposed_candidate_outside_pool','outside'),('unknown_citation','unknown')]:
                assert report[key]==sum(r[name] for r in group)
        audit.update(status='complete_verified',unique_calls=len(responses),
                     raw_outputs_checked_before_local_validation=True,all_final_metrics_independently_recounted=True,
                     final_pool_violations=0,primary_response_duplicates=0)
    (out/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    if not args.inputs_only:
        complete=read(out/'completion.json')
        complete['independent_audit']=audit['status']
        complete['files_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
                                  if p.is_file() and p.name!='completion.json' and p.suffix!='.log'}
        (out/'completion.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
