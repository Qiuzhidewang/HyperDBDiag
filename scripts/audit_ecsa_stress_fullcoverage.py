#!/usr/bin/env python3
"""Independent audit of the full-corpus stress expansion and separate repeats."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_ecsa_evidence_stress import body,digest,read,sha
from ecsa_arbitration import validate_response


def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--output',type=Path,default=ROOT/'runs/ecsa-evidence-stress-full660-20260930-v1')
    c.add_argument('--inputs-only',action='store_true');args=c.parse_args();out=args.output
    prot,p=read(out/'protocol.json'),read(out/'prepared.json')
    assert p['protocol_sha256']==digest(prot)
    assert sha(ROOT/'scripts/run_ecsa_stress_fullcoverage.py')==prot['script_sha256']
    for path,h in prot['protected_sha256'].items(): assert sha(ROOT/path)==h,path
    canonical=read(ROOT/'runs/hyperdbdiag-unified-20260930/predictions.json')
    expected={r['case_id']:r for r in canonical}
    assert len(expected)==660
    counts=Counter((r['fold'],len(r['expected_labels'])) for r in canonical)
    assert len(counts)==12 and sorted(counts.values())==[44]*6+[66]*6
    old_dir=ROOT/'runs/ecsa-evidence-stress-20260930-v1'
    old={r['packet_sha256']:r for r in map(json.loads,(old_dir/'primary_responses.jsonl').read_text().splitlines())}
    reused=list(map(json.loads,(out/'reused_primary_responses.jsonl').read_text().splitlines()))
    assert len(reused)==713
    for r in reused:
        assert r=={**old[r['packet_sha256']],'job_id':'primary:'+r['packet_sha256'],'response_source':'pilot_exact_primary'}
    base={j['case_id']:j['packet'] for j in read(ROOT/'runs/ecsa-full660-20260929/prepared.json')['jobs']
          if j['draw']=='primary' and j['arm']=='conservative_filtered_pool'}
    rows,requests,jobs=p['rows'],p['requests'],p['jobs']
    assert len(rows)==4620 and len({(r['case_id'],r['arm']) for r in rows})==4620
    by_pair={(r['case_id'],r['arm']):r for r in rows}
    for arm in prot['arms']: assert {r['case_id'] for r in rows if r['arm']==arm}==set(expected)
    assert set(p['new_request_order'])==set(jobs)-{r['job_id'] for r in reused}
    assert len(p['new_request_order'])==len(set(p['new_request_order']))==prot['planned_new_calls']
    for key,packet in requests.items():
        assert digest(packet)==key
        assert set(packet)=={'task','candidate_records','evidence','requires_direct_semantic_evidence'}
        assert not any(s in json.dumps(packet) for s in ['expected_labels','historical_prediction','local_prediction','case-00'])
    for r in rows:
        packet=requests[r['packet_sha256']];original=base[r['case_id']];arm=r['arm']
        assert r['expected_labels']==expected[r['case_id']]['expected_labels']
        assert r['fold']==expected[r['case_id']]['fold']
        assert r['local_prediction']==expected[r['case_id']]['local_prediction']
        assert r['original_packet_sha256']==digest(original)==expected[r['case_id']]['filtered_packet_sha256']
        assert r['job_id']==r['draw']+':'+r['packet_sha256']
        assert r['draw']==('repeat' if arm=='clean_repeat' else 'primary')
        assert r['local_prediction']==jobs[r['job_id']]['local_prediction']
        assert r['input_changed']==(packet!=original)
        assert packet['requires_direct_semantic_evidence']==original['requires_direct_semantic_evidence']
        assert {x['candidate_id']:x['root_labels'] for x in packet['candidate_records']}=={x['candidate_id']:x['root_labels'] for x in original['candidate_records']}
        assert r['local_prediction'] in [x['root_labels'] for x in packet['candidate_records']]
        sort=lambda items:sorted(items,key=lambda x:x['evidence_id'])
        for family,items in packet['evidence'].items():
            if family!='candidate_evidence_cards': assert sort(items)==sort(original['evidence'][family])
        cards=packet['evidence']['candidate_evidence_cards'];oldcards=original['evidence']['candidate_evidence_cards']
        for x,y in zip(sort(cards),sort(oldcards)):
            assert {k:v for k,v in x.items() if k!='summary'}=={k:v for k,v in y.items() if k!='summary'}
            assert body(x)[0]==body(y)[0]
        if arm in ['clean','clean_repeat']:
            assert packet==original
            if arm=='clean_repeat':
                assert r['packet_sha256']==by_pair[(r['case_id'],'clean')]['packet_sha256']
                assert r['job_id']!=by_pair[(r['case_id'],'clean')]['job_id']
        elif arm=='identical_support':
            rng=random.Random(int(digest([prot['seed'],r['case_id'],arm]),16))
            donor=body(oldcards[rng.randrange(len(oldcards))])[1]
            assert all(body(x)[1]==donor for x in cards)
            assert [body(x)[2] for x in cards]==[body(x)[2] for x in oldcards]
        elif arm=='empty_counterevidence':
            assert all(body(x)[2]=='none' for x in cards)
            assert [body(x)[1] for x in cards]==[body(x)[1] for x in oldcards]
        elif arm=='shuffled_evidence':
            before=[s for x in oldcards for s in body(x)[1:]];after=[s for x in cards for s in body(x)[1:]]
            assert Counter(before)==Counter(after)
            rng=random.Random(int(digest([prot['seed'],r['case_id'],arm]),16));order=list(range(len(before)))
            while True:
                rng.shuffle(order)
                if all(i!=j for i,j in enumerate(order)):break
            assert after==[before[i] for i in order]
        elif arm=='order_only':
            assert packet['candidate_records']==original['candidate_records'][::-1]
            assert all(packet['evidence'][k]==v[::-1] for k,v in original['evidence'].items())
        elif arm=='out_of_pool_lure':
            assert cards[1:]==oldcards[1:]
            assert body(cards[0])[1]==body(oldcards[0])[1]+'; '+prot['lure_text']
            assert body(cards[0])[2]==body(oldcards[0])[2]
            assert 'C99' not in {x['candidate_id'] for x in packet['candidate_records']}
    report=dict(status='input_verified',cases_per_arm=660,arms=len(prot['arms']),logical_evaluations=4620,
                all_cases_covered=True,all_perturbations_checked=True,no_test_labels_in_model_inputs=True,
                all_pilot_primary_responses_preserved=True,repeats_never_substituted=True,protected_files_unchanged=True,
                unique_primary_inputs=prot['planned_unique_primary_inputs'],repeat_calls=prot['planned_clean_repeat_calls'])
    if not args.inputs_only:
        records=reused+list(map(json.loads,(out/'new_responses.jsonl').read_text().splitlines()))
        responses={r['job_id']:r for r in records}
        assert len(responses)==len(records)==len(jobs) and set(responses)==set(jobs)
        predictions=read(out/'predictions.json');published={(r['case_id'],r['arm']):r for r in predictions}
        assert len(published)==4620
        independent=[]
        for row in rows:
            response=responses[row['job_id']];packet=requests[row['packet_sha256']]
            assert response['draw']==row['draw'] and response['packet_sha256']==row['packet_sha256']
            labels=row['local_prediction'];received='raw_response' in response;parseable=False;outside=False
            if received:
                assert digest(response['raw_response'])==response['raw_response_sha256']
                try: raw=json.loads(response['raw_response']);parseable=isinstance(raw,dict)
                except ValueError: raw=None
                if parseable: outside=isinstance(raw.get('candidate_id'),str) and raw['candidate_id'] not in {x['candidate_id'] for x in packet['candidate_records']}
                try:
                    decision=validate_response(response['raw_response'],packet)
                    status='accepted_selection' if decision['action']=='SELECT_CANDIDATE' else 'accepted_abstain'
                    if status=='accepted_selection': labels=next(x['root_labels'] for x in packet['candidate_records'] if x['candidate_id']==decision['candidate_id'])
                except (ValueError,TypeError,KeyError):status='fallback_invalid_response'
            else: status='fallback_transport'
            assert labels==response['predicted_labels'] and status==response['status']
            assert labels==published[(row['case_id'],row['arm'])]['predicted_labels']
            assert status==published[(row['case_id'],row['arm'])]['status']
            assert any(set(labels)==set(x['root_labels']) for x in packet['candidate_records'])
            independent.append({**row,'predicted_labels':labels,'status':status,'received':received,'parseable':parseable,'outside':outside})
        reference={r['case_id']:r for r in independent if r['arm']=='clean'}
        results=read(out/'results.json')
        for s in results['arms']:
            group=[r for r in independent if r['arm']==s['arm']]
            pairs=[(set(r['expected_labels']),set(r['predicted_labels'])) for r in group]
            correct=sum(a==b for a,b in pairs);tp=sum(len(a&b) for a,b in pairs);err=sum(len(a^b) for a,b in pairs)
            assert s['metrics']['correct']==correct and s['metrics']['Exact']==correct/660
            assert s['metrics']['F1']==2*tp/(2*tp+err)
            assert s['statuses']==dict(Counter(r['status'] for r in group))
            assert s['proposed_candidate_outside_pool']==sum(r['outside'] for r in group)
            wins=sum(set(r['expected_labels'])==set(r['predicted_labels']) and set(r['expected_labels'])!=set(reference[r['case_id']]['predicted_labels']) for r in group)
            losses=sum(set(r['expected_labels'])!=set(r['predicted_labels']) and set(r['expected_labels'])==set(reference[r['case_id']]['predicted_labels']) for r in group)
            assert s['paired_vs_clean']['wins']==wins and s['paired_vs_clean']['losses']==losses
        old_predictions=read(old_dir/'predictions.json')
        for r in old_predictions:
            now=published[(r['case_id'],r['arm'])]
            for k in ['packet_sha256','predicted_labels','status','raw_audit']:assert now[k]==r[k]
        report.update(status='complete_verified',all_original_pilot_predictions_identical=True,
                      unique_calls=len(records),new_calls=len(records)-len(reused),
                      all_raw_responses_revalidated=True,all_metrics_independently_recomputed=True,
                      final_pool_violations=0)
    (out/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not args.inputs_only:
        complete=read(out/'completion.json');complete['status']='complete_verified';complete['independent_audit']=report['status']
        complete['files_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='completion.json' and f.suffix!='.log'}
        (out/'completion.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
