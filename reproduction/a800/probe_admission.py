"""Admit actual initial G1 calibration and a working frozen two-head endpoint.

HTTP smoke responses verify finite readout plumbing, not task accuracy or AUC.
The latter comes only from intervention calibration and grouped CV evidence.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from reproduction.a800.prepare_run import digest
from search_g1.signals import validate_probe_reply


def atomic(path, payload):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(payload,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def calibrated_version(run, live):
    assert live['run_id']==run['run_id'] and live['round']==0
    assert live['encoder']==run['initial_model']
    assert live['version'].startswith(run['run_id']+':0:')
    report=live['calibration_report'];proof=live['label_proof']
    assert report['unique_question_count']==proof['question_count']==300
    assert report['question_grouped_cv'] is True and proof['group_size']==5
    assert proof['closed_book_label']=='exact_match_no_retrieval'
    assert proof['reliance_label']=='normalized_answer_change_after_evidence_deletion'
    assert proof['model_authored_answer_spans'] is True and proof['boundary_all_reserved_questions'] is True
    assert len(set(proof['question_ids']))==300
    assert proof['searched_valid_trajectory_count']>0
    for key in ('commit','boundary'):
        assert len(report[key+'_class_counts'])==2 and min(report[key+'_class_counts'])>0
        assert math.isfinite(report[key+'_cv_auc']) and 0<=report[key+'_cv_auc']<=1
    for name,value in run['initial_model_sha256'].items():
        assert digest(Path(run['initial_model'])/name)==value
    for name,value in live['encoder_sha256'].items():
        assert value==run['initial_model_sha256'][name]
        assert digest(Path(live['encoder'])/name)==value
    for name,key in [('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')]:
        assert digest(Path(live[key]))==live['head_sha256'][name]
    return live['version']


def verify_training_gate(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    accepted=json.loads((base/'logs/initial-probes-accepted.json').read_text())
    live=json.loads((base/'artifacts/live/manifest.json').read_text())
    version=calibrated_version(run,live)
    assert accepted['accepted'] is True and accepted['run_id']==run['run_id']
    assert accepted['version']==run['initial_probe_version']==version
    assert accepted['live_manifest_sha256']==digest(base/'artifacts/live/manifest.json')
    for name,expected in run['input_sha256'].items():assert digest(base/'data'/name)==expected
    for name,expected in run['source_sha256'].items():assert digest(base/'repo'/name)==expected, 'Source changed since prelaunch admission: '+name
    import requests
    session=requests.Session();session.trust_env=False
    response=session.get('http://127.0.0.1:18012/health',timeout=60);response.raise_for_status();health=response.json()
    assert health.get('ok') is True and health.get('round')==0 and health.get('version')==version and health.get('encoder')==run['initial_model']
    return run


def admit(base):
    import requests
    run_path=base/'logs/run-manifest.json';run=json.loads(run_path.read_text())
    target=base/'logs/initial-probes-accepted.json'
    if target.exists():
        verify_training_gate(base);return
    live_path=base/'artifacts/live/manifest.json';live=json.loads(live_path.read_text())
    version=calibrated_version(run,live)
    assert run['initial_probe_version'] in (None,version)
    audit=json.loads((base/'data/audit.json').read_text())
    assert set(live['label_proof']['question_ids'])==set(audit['calibration_question_ids'])
    for name,expected in run['input_sha256'].items():assert digest(base/'data'/name)==expected
    for name,expected in run['source_sha256'].items():assert digest(base/'repo'/name)==expected
    session=requests.Session();session.trust_env=False
    health=session.get('http://127.0.0.1:18012/health',timeout=60);health.raise_for_status();health=health.json()
    assert health['ok'] is True and health['round']==0 and health['encoder']==live['encoder'] and health['version']==version
    response='<answer>Paris</answer>'
    items=[dict(id='admission-direct',prompt='Question: What is the capital of France?\n',response=response,model_ranges=[[0,len(response)]]),
           dict(id='admission-invalid',prompt='Question: What is the capital of France?\n',response='',model_ranges=[])]
    answer=session.post('http://127.0.0.1:18012/judge_probe',json={'items':items},timeout=120);answer.raise_for_status()
    rows,actual=validate_probe_reply(answer.json(),items);assert actual==version
    evidence=dict(run_id=run['run_id'],accepted=True,version=version,live_manifest_sha256=digest(live_path),
                  grouped_calibration_cv=live['calibration_report'],label_proof=live['label_proof'],
                  actual_http_readouts=rows,http_smoke_semantics='finite identity-aligned readout plumbing, not accuracy',checked_unix=time.time())
    run['initial_probe_version']=version;atomic(run_path,run);atomic(target,evidence)
    verify_training_gate(base)
    print('SEARCH_G1_INITIAL_PROBES_ACCEPTED='+version,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'))
    if args.verify:verify_training_gate(base)
    else:admit(base)
