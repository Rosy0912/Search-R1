"""CPU fixtures only: no SSH, task private key, signing or GPU execution."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import types

import pytest
import publish_first5_release as pub


def fixture():
    run=pub.ROOT+'/runs/formal-3261'
    args=types.SimpleNamespace(job_id='3261',driver_pid=1533312,driver_start='1829085316',
        source=pub.SOURCE,tools_sha256=pub.TOOLS_SHA,manifest_sha256='1'*64,
        unsigned_payload_sha256='2'*64,unsigned_payload_remote_path=run+'/review/unsigned-step5-payload.json',
        evidence_spec_sha256='3'*64,evidence_spec_remote_path=run+'/review/spec.json',
        root_review_sha256='4'*64,independent_review_sha256='5'*64)
    rec=lambda name,pin:{'path':run+'/'+name,'sha256':pin*64}
    manifest=dict(version=1,method='pure_opsd',run_root=run,source_commit=pub.SOURCE,
        config_sha256=pub.CONFIG_SHA,global_step=5,actual_outer_updates=5,next_global_step=6,
        horizon=200,warmup_steps=0,world_size=8,driver_pid=1533312,uid=0,job_id='3261',
        pause_nonce='6'*64,files={'TEST_FIXTURE_ONLY':rec('checkpoint.json','7')})
    spec={role:rec('review/'+role+'.json','8') for role in pub.ROLES}
    rootrec=rec('review/root.json','4');indrec=rec('review/independent.json','5')
    payload={k:manifest[k] for k in pub.IDENTITY}
    payload.update(status='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW',checkpoint_manifest_sha256='1'*64,
        checkpoint_files=manifest['files'],independent_review=indrec,
        evidence={**spec,'root_review':rootrec,'preparation_audit':rec('review/audit.json','9')},
        train_log_prefix={'live_path':run+'/train.log','snapshot_path':run+'/review/prefix.out',
                          'sha256':'a'*64,'bytes':123})
    def review(role):
        return dict(version=1,role=role,status='PASS_ACTUAL_FIRST5_EVIDENCE',source_commit=pub.SOURCE,
            job_id='3261',driver_pid=1533312,pause_nonce=manifest['pause_nonce'],
            checkpoint_manifest_sha256='1'*64,evidence=spec)
    return args,manifest,payload,spec,review('ROOT_AUDITOR'),review('INDEPENDENT_REVIEWER')


def test_exact_author200_input_binding_positive_without_signing():
    args,m,p,s,r,i=fixture()
    run,records=pub.check_inputs(args,m,p,s,r,i)
    assert run==pub.ROOT+'/runs/formal-3261'
    assert {'path':args.unsigned_payload_remote_path,'sha256':'2'*64} in records
    assert {'path':run+'/full_state/global_step_5/manifest.json','sha256':'1'*64} in records


@pytest.mark.parametrize('kind',['job','driver','source','tools','horizon','warmup','nextstep',
    'world','boolclock','config','nonce','payload_identity','payload_extra','manifest_pin',
    'checkpoint_files','root_role','independent_status','review_nonce','review_pin',
    'review_spec','spec_roles','payload_evidence','live_prefix','live_equals_snapshot',
    'prefix_bytes_bool','record_escape','record_foreign','record_badpin','record_extra'])
def test_actual_release_binding_drift_refused(kind):
    a,m,p,s,r,i=fixture()
    if kind=='job':a.job_id='3210'
    elif kind=='driver':a.driver_pid=42
    elif kind=='source':a.source='a0968c2d24beeaa0400f60e4e34677fe6deef9c9'
    elif kind=='tools':a.tools_sha256='0'*64
    elif kind=='horizon':m['horizon']=300
    elif kind=='warmup':m['warmup_steps']=57
    elif kind=='nextstep':m['next_global_step']=7
    elif kind=='world':m['world_size']=1
    elif kind=='boolclock':m['global_step']=True
    elif kind=='config':m['config_sha256']='0'*64
    elif kind=='nonce':m['pause_nonce']='guess'
    elif kind=='payload_identity':p['driver_pid']=42
    elif kind=='payload_extra':p['unguarded']=True
    elif kind=='manifest_pin':p['checkpoint_manifest_sha256']='0'*64
    elif kind=='checkpoint_files':p['checkpoint_files']={}
    elif kind=='root_role':r['role']='INDEPENDENT_REVIEWER'
    elif kind=='independent_status':i['status']='CPU_TEST_PASS'
    elif kind=='review_nonce':i['pause_nonce']='0'*64
    elif kind=='review_pin':a.independent_review_sha256='0'*64
    elif kind=='review_spec':i['evidence']={}
    elif kind=='spec_roles':s['hypothetical']=s['resources']
    elif kind=='payload_evidence':p['evidence']=dict(p['evidence'],invented=s['resources'])
    elif kind=='live_prefix':p['train_log_prefix']['live_path']=pub.ROOT+'/runs/formal-3210/train.log'
    elif kind=='live_equals_snapshot':p['train_log_prefix']['snapshot_path']=p['train_log_prefix']['live_path']
    elif kind=='prefix_bytes_bool':p['train_log_prefix']['bytes']=True
    elif kind=='record_escape':m['files']['TEST_FIXTURE_ONLY']['path']=pub.ROOT+'/runs/formal-3261/../outside.json'
    elif kind=='record_foreign':m['files']['TEST_FIXTURE_ONLY']['path']=pub.ROOT+'/runs/formal-3210/outside.json'
    elif kind=='record_badpin':m['files']['TEST_FIXTURE_ONLY']['sha256']='notSHA'
    elif kind=='record_extra':m['files']['TEST_FIXTURE_ONLY']['guessed']=True
    with pytest.raises(ValueError):pub.check_inputs(a,m,p,s,r,i)


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e9999}'])
def test_nonfinite_duplicate_json_refused(raw):
    with pytest.raises(ValueError):pub.unique(raw)


def test_original_local_pin_physical_owner_and_hash(tmp_path):
    p=tmp_path/'original.json';p.write_bytes(b'exact original\r\n')
    assert pub.pinned_local(p,hashlib.sha256(p.read_bytes()).hexdigest())==p.read_bytes()
    with pytest.raises(ValueError):pub.pinned_local(p,'0'*64)
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):pub.pinned_local(link,hashlib.sha256(p.read_bytes()).hexdigest())


def exact_remote_function(name,namespace):
    tree=ast.parse(pub.LIVE_BODY);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'EXACT_ROOT_FUNCTION_CPU_FIXTURE','exec'),namespace)
    return namespace[name]


@pytest.mark.parametrize('kind',['valid','wrong_start','wrong_uid','only_one_uid','foreign_job','prefix_job','zombie'])
def test_exact_live_process_identity_guard(tmp_path,kind):
    root=tmp_path/'proc';p=root/'123';p.mkdir(parents=True)
    fields=['S']+['0']*22;fields[19]='1829085316';(p/'stat').write_text('123 (ray::main_task) '+' '.join(fields)+'\n')
    uids='0\t0\t0\t0' if kind not in ('wrong_uid','only_one_uid') else ('0\t1\t0\t0' if kind=='wrong_uid' else '0')
    state='Z' if kind=='zombie' else 'S';(p/'status').write_text('State:\t'+state+' (fixture)\nUid:\t'+uids+'\n')
    job='3210' if kind=='foreign_job' else ('32610' if kind=='prefix_job' else '3261')
    (p/'cgroup').write_text('1:freezer:/slurm_h100-10/uid_0/job_'+job+'/step_0\n')
    namespace={'pathlib':types.SimpleNamespace(Path=lambda value:root if value=='/proc' else Path(value)),
               're':re,'D':{'job_id':'3261'},'need':pub.need}
    fn=exact_remote_function('process',namespace)
    expected='999999' if kind=='wrong_start' else '1829085316'
    if kind=='valid':
        value=fn(123,expected);assert value['start']==expected and value['uids']==[0,0,0,0]
    else:
        with pytest.raises(ValueError):fn(123,expected)


class RootOwnedPath(type(Path())):
    """Local path fixture simulates only remote stat uid=0, preserving actual IO."""
    def stat(self,*args,**kwargs):
        result=super().stat(*args,**kwargs)
        return types.SimpleNamespace(st_uid=0,st_mode=result.st_mode)


class ForeignOwnedPath(type(Path())):
    def stat(self,*args,**kwargs):
        result=super().stat(*args,**kwargs)
        return types.SimpleNamespace(st_uid=1,st_mode=result.st_mode)


@pytest.mark.parametrize('kind',['fresh','exists','symlink','wrong_parent_owner'])
def test_exact_exclusive_publication_preserves_existing_bytes_and_fsync(tmp_path,kind,monkeypatch):
    target=tmp_path/'acceptance-5-receipt.json';original=tmp_path/'original';original.write_bytes(b'KEEP ORIGINAL')
    if kind=='exists':target.write_bytes(b'KEEP ORIGINAL')
    elif kind=='symlink':target.symlink_to(original)
    namespace={'pathlib':types.SimpleNamespace(Path=ForeignOwnedPath if kind=='wrong_parent_owner' else RootOwnedPath),
               'os':os,'secrets':secrets,'need':pub.need}
    calls=[];fsync=os.fsync
    monkeypatch.setattr(os,'fsync',lambda fd:(calls.append(fd),fsync(fd))[1])
    fn=exact_remote_function('exclusive',namespace);raw=b'CPU FIXTURE ONLY NO SIGNATURE\r\n'
    if kind=='fresh':
        fn(target,raw);assert target.read_bytes()==raw and len(calls)==2
        assert target.stat().st_mode&0o777==0o600
    elif kind=='wrong_parent_owner':
        with pytest.raises(ValueError):fn(target,raw)
        assert not target.exists()
    else:
        with pytest.raises(FileExistsError):fn(target,raw)
        assert target.read_bytes()==b'KEEP ORIGINAL'
    assert original.read_bytes()==b'KEEP ORIGINAL'
    assert not list(tmp_path.glob('.root-first5-*.tmp'))


def test_complete_remote_program_compiles_and_no_gpu_or_job_mutation():
    compile(pub.LIVE_BODY,'PREPARED_ONLY_NOT_EXECUTED','exec')
    for forbidden in ('import torch','sbatch','scancel','os.kill','-sign','private.pem','eval('):
        assert forbidden not in pub.LIVE_BODY
    assert "gate.validate_outer(row['metrics'],profile='author200')" in pub.LIVE_BODY
    assert 'before=validate_live()' in pub.LIVE_BODY
    assert 'after=validate_live(require_paused=False)' in pub.LIVE_BODY
    assert 'nativeinspectormanifestchanged' in pub.LIVE_BODY
    assert 'formaltoolbyteschanged' in pub.LIVE_BODY
    assert 'nativeclosurebyteschanged' in pub.LIVE_BODY
    assert pub.PRIVATE==pub.HERE.parent/'e1-300/review-signing/private.pem'
    assert pub.SOURCE=='188f4e5e7b63cf479290d0888a4a075499886750'
    assert pub.local_gate().REVIEW_LOG_MARKERS
