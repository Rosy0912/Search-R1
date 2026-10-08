"""New two-gap regressions only; no replay of old82/native/GPU/remote checks."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest
import deployment_contract_v5 as contract
from deployment_contract_v5 import SOURCE,SCOPE,MANIFEST,CHECKPOINT,CONTROLS,sha
from freeze_ready_v5 import freeze
import freeze_ready_v5 as freezer

# Read-only reuse of CPU fixture builders, not execution of the old82 test suite.
V4=Path(__file__).resolve().parent.with_name('safety-calibration-deployment-v4')
sys.path.append(str(V4))
import test_deployment_v4 as f4


def raw(path,value):
    data=json.dumps(value,sort_keys=True).encode()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),content_base64=base64.b64encode(data).decode())


def native_fixture5(root=None):
    a,r,v,b,g,t,p=copy.deepcopy(f4.native_fixture(root or contract.ROOT))
    r['restore_step']=25
    b.update(restore_step=25,restore_checkpoint=CHECKPOINT,restore_manifest_sha256=MANIFEST,
             fresh_model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1')
    br=raw(a['native_binding_path'],b);a['native_binding_sha256']=r['binding_sha256']=br['sha256']
    rr=raw(a['result_path'],r);a['result_sha256']=v['result_sha256']=rr['sha256']
    p.update(result=copy.deepcopy(r),originals=[rr,raw(a['terminal_path'],t),br],
             physical=[{k:rr[k] for k in ('path','bytes','sha256')}])
    return a,r,v,b,g,t,p


def actual_native(tmp,monkeypatch):
    monkeypatch.setattr(contract,'ROOT',tmp)
    a,r,v,b,g,t,p=native_fixture5(tmp)
    for record in p['originals']:
        path=Path(record['path']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(base64.b64decode(record['content_base64']))
    physical=Path(a['physical_payload_path']);physical.write_bytes(gzip.compress(json.dumps(p).encode(),mtime=0))
    a['physical_payload_sha256']=v['physical_payload_sha256']=sha(physical)
    review=tmp/'logs/native-review.json';review.write_text(json.dumps(v))
    a.update(review_path=str(review),review_sha256=sha(review))
    return a,r,v,b,g,t,p


def freeze_fixture(tmp,monkeypatch):
    a,r,v,b,g,t,p=actual_native(tmp,monkeypatch)
    c=f4.isolated_candidate(g);c.update(controls=str(CONTROLS),repo=str(tmp/'repo-policy-safety-52befe9'))
    inherited=tmp/'inherited-evidence.json';inherited.write_bytes(b'{"v":1}')
    c['evidence']=[dict(path=str(inherited),bytes=7,sha256=sha(inherited))]
    contract.validate_candidate(c)
    physical=dict(candidate_sha256=None,code_commit=SOURCE,
        status='PHYSICAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',
        metadata=copy.deepcopy(c['files']),evidence=copy.deepcopy(c['evidence']),pad_token_id=151643)
    return a,c,physical


def call_freeze(tmp,a,c,physical):
    cp=tmp/'candidate.json';cp.write_text(json.dumps(c));physical['candidate_sha256']=sha(cp)
    pp=tmp/'physical.json';pp.write_text(json.dumps(physical))
    review=dict(status='PASS',scope=SCOPE,candidate_sha256=sha(cp),physical_sha256=sha(pp),
        freeze_helper_sha256=sha(freezer.__file__),native_result_sha256=a['result_sha256'],native_review_sha256=a['review_sha256'],
        native_terminal_sha256=a['terminal_sha256'],native_physical_payload_sha256=a['physical_payload_sha256'])
    rp=tmp/'deployment-review.json';rp.write_text(json.dumps(review))
    spec=tmp/'native-spec.json';spec.write_text(json.dumps(a))
    target=tmp/'ready.json'
    response=freeze(str(cp),sha(cp),str(pp),sha(pp),str(rp),sha(rp),str(spec),str(target))
    return response,json.loads(target.read_text())


@pytest.mark.parametrize('mutation',['evidence_same_size_different_sha','evidence_size','evidence_sha_missing',
    'evidence_duplicate','evidence_missing','evidence_extra','file_same_size_different_sha','file_size','file_sha_missing',
    'file_duplicate','file_missing','file_extra'])
def test_physical_cannot_replace_duplicate_or_reduce_immutable_closure(tmp_path,monkeypatch,mutation):
    a,c,p=freeze_fixture(tmp_path,monkeypatch)
    if mutation=='evidence_same_size_different_sha':
        inherited=Path(c['evidence'][0]['path']);inherited.write_bytes(b'{"v":2}')
        p['evidence'][0]['sha256']=sha(inherited)
    elif mutation=='evidence_size':p['evidence'][0]['bytes']+=1
    elif mutation=='evidence_sha_missing':p['evidence'][0].pop('sha256')
    elif mutation=='evidence_duplicate':p['evidence']+=copy.deepcopy(p['evidence'])
    elif mutation=='evidence_missing':p['evidence']=[]
    elif mutation=='evidence_extra':p['evidence'].append(dict(path='/extra',bytes=1,sha256='f'*64))
    elif mutation=='file_same_size_different_sha':p['metadata'][0]['sha256']='f'*64
    elif mutation=='file_size':p['metadata'][0]['bytes']+=1
    elif mutation=='file_sha_missing':p['metadata'][0].pop('sha256')
    elif mutation=='file_duplicate':p['metadata'].append(copy.deepcopy(p['metadata'][0]))
    elif mutation=='file_missing':p['metadata'].pop()
    elif mutation=='file_extra':p['metadata'].append(dict(path='/extra',bytes=1,sha256='f'*64))
    with pytest.raises((ValueError,KeyError)):call_freeze(tmp_path,a,c,p)
    assert not (tmp_path/'ready.json').exists()


def test_exact_bound_closure_retains_candidate_evidence_sha(tmp_path,monkeypatch):
    a,c,p=freeze_fixture(tmp_path,monkeypatch)
    p['metadata'][0]['inode']=123 # Provenance metadata allowed, identity projection remains exact.
    response,ready=call_freeze(tmp_path,a,c,p)
    assert response['ready']
    assert ready['files']==c['files']
    assert next(r for r in ready['evidence'] if r['path']==c['evidence'][0]['path'])==c['evidence'][0]
    contract.validate_ready(ready)


@pytest.mark.parametrize('mutation',['result_step_missing','result_step999','result_manifest','binding_step',
    'binding_manifest','binding_checkpoint','binding_revision','phase_before','phase_after','phase_revision'])
def test_summary_binding_and_official_phase_identity_cannot_contradict_healthy25(mutation):
    a,r,v,b,g,t,p=native_fixture5()
    if mutation=='result_step_missing':r.pop('restore_step')
    elif mutation=='result_step999':r['restore_step']=999
    elif mutation=='result_manifest':r['restore_manifest_sha256']='f'*64
    elif mutation=='binding_step':b['restore_step']=999
    elif mutation=='binding_manifest':b['restore_manifest_sha256']='f'*64
    elif mutation=='binding_checkpoint':b['restore_checkpoint']=CHECKPOINT.replace('25','100')
    elif mutation=='binding_revision':b['fresh_model_revision']='base'
    elif mutation=='phase_before':r['phases'][0][0]['optimizer_steps_before']=250
    elif mutation=='phase_after':r['phases'][1][0]['optimizer_steps_after']=511
    elif mutation=='phase_revision':r['phases'][1][0]['official_revision']='base'
    # Keep raw identities internally self-consistent, so rejection is the new
    # semantic gate, not stale hashes from editing the synthetic summary.
    br=raw(a['native_binding_path'],b);a['native_binding_sha256']=r['binding_sha256']=br['sha256']
    rr=raw(a['result_path'],r);a['result_sha256']=v['result_sha256']=rr['sha256']
    p.update(result=copy.deepcopy(r),originals=[rr,raw(a['terminal_path'],t),br],physical=[{k:rr[k] for k in ('path','bytes','sha256')}])
    with pytest.raises((ValueError,KeyError)):contract.validate_native_payload(a,r,v,b,g,t,p)


def test_complete_native_identity_has_explicit_healthy25_summary():
    contract.validate_native_payload(*native_fixture5())
