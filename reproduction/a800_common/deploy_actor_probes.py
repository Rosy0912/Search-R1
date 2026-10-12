"""Prepare remote canonical inputs, independently review, then explicitly launch.

The local handle contains small identity/hash metadata only. Neither preparation
nor observation transfers model, QA, state or environment bytes through the Mac.
An uncertain action is never automatically repeated; observe its original handle.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from reproduction.a800_common import deploy
from reproduction.a800_common.actor_probe_production_review import MODE,PREPARED
from reproduction.a800_common.prepare_actor_probe_production import TARGETS,validate_intent
from reproduction.a800_common.predecessor_admission import validate_binding
from reproduction.common8.verify_data import digest

MODULE='reproduction/a800_common/deploy_actor_probes.py'
PREPARER='reproduction/a800_common/prepare_actor_probe_production.py'

# This bootstrap checks the published file before importing it. All heavy CPU
# configuration/model-file verification runs with the existing remote runtime.
REMOTE=r'''
import json,hashlib,os,sys,subprocess
from pathlib import Path
payload=json.load(sys.stdin);intent=payload['intent'];action=payload['action']
assert action in ('prepare','inspect-prepared','launch')
repo=Path(intent['source_staging_base'] if action=='prepare' else intent['base'])/'repo'
assert repo==repo.resolve() and repo.is_dir() and repo.stat().st_uid==os.getuid()
name='reproduction/a800_common/prepare_actor_probe_production.py' if action=='prepare' else 'reproduction/a800_common/deploy_actor_probes.py'
assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==intent['source_sha256'][name]
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==intent['source_commit']
sys.path.insert(0,str(repo))
from reproduction.a800_common.prepare_actor_probe_production import validate_intent,environment,layout
validate_intent(intent)
if action=='prepare':
 result=layout(intent)
 print('DEPLOYMENT_JSON:'+json.dumps(result),flush=True)
else:
 base=Path(intent['base']);run=json.loads((base/'logs/run-manifest.json').read_bytes())
 env=environment(base,Path(run['ray_tmp_dir']))
 result=subprocess.run(['/data/search-g1/env/bin/python','-m','reproduction.a800_common.deploy_actor_probes',
  '--remote-action',action],cwd=repo,env=env,input=json.dumps(payload).encode(),timeout=600 if action=='inspect-prepared' else 180)
 sys.exit(result.returncode)
'''


def create_intent(parent,stage,binding,repo,commit):
    """Read only already-retained small deployment metadata, never local QA."""
    binding=validate_binding(binding)
    assert all(parent[key]==binding[key] for key in ('base','source_commit'))
    assert parent['role']=='fresh_a800_common_deployment_intent' and parent['host']=='baai-a800'
    assert stage['role']=='a800_published_source_staging_only' and stage['source_commit']==commit
    assert stage['exact_published_source_verified'] is True and stage['git_tree_clean'] is True
    assert stage['production_admission'] is False and stage['services_started'] is False and stage['trainer_started'] is False
    source=deploy.published_source(repo,commit);assert source==stage['source_sha256']
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/actor-probes-ten'],text=True).strip()
    assert remote.split()==[commit,'refs/heads/actor-probes-ten'], 'Latest source has not been published'
    rows=[row for row in parent['inputs'] if row['target'] in TARGETS]
    assert len(rows)==len(TARGETS) and {row['target'] for row in rows}==TARGETS
    inputs=[dict(remote_source=str(Path(binding['base'])/row['target']),
        **{key:row[key] for key in ('target','bytes','sha256')}) for row in rows]
    # Staging writes this exact canonical byte serialization remotely. The
    # actual prepare gate rehashes the original remote file before trusting it.
    stage_sha=hashlib.sha256((json.dumps(stage,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()
    identifier=uuid.uuid4().hex
    return validate_intent(dict(schema_version=1,role='fresh_a800_common_deployment_intent',
        deployment_id=identifier,created_unix=time.time(),host='baai-a800',
        base='/data/shaoshitong/Search-G1-a800-fresh200-'+identifier,
        source_commit=commit,source_sha256=source,source_staging_base=stage['base'],
        source_staging_receipt_sha256=stage_sha,preparation_mode=MODE,
        reward_schedule='g1_through100_then_binary',predecessor_completion=binding,inputs=inputs,
        training_world_size=4,training_physical_indices=[0,1,2,3],auxiliary_physical_indices=[5,6],
        target_updates=200,same_process_after_five=True,
        execution_policy='remote-only prepare; independent actual deployment review; one launch; no automatic retry'))


def launch_payload(intent,repo):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==intent['source_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    assert deploy.published_source(repo,intent['source_commit'])==intent['source_sha256']
    assert digest(repo/MODULE)==intent['source_sha256'][MODULE]
    assert digest(repo/'reproduction/a800_common/deploy.py')==intent['source_sha256']['reproduction/a800_common/deploy.py']
    return dict(launcher_source_commit=intent['source_commit'],
        launcher_source_sha256=intent['source_sha256']['reproduction/a800_common/deploy.py'],
        embedded_launcher_sha256=hashlib.sha256(deploy.REMOTE_LAUNCH_PREPARED.encode()).hexdigest(),
        preclaim_retry=False)


def remote_action(payload):
    """Existing remote environment entry; review must precede the Popen claim."""
    from reproduction.a800_common.actor_probe_production_review import actual_preparation,read,verify
    intent=validate_intent(payload['intent']);base=Path(intent['base']);repo=base/'repo'
    assert Path(__file__).resolve()==repo/MODULE
    assert digest(Path(__file__))==intent['source_sha256'][MODULE]
    run,_=read(base,'logs/run-manifest.json')
    assert run['source_commit']==intent['source_commit'] and run['source_sha256']==intent['source_sha256']
    assert read(base,'preparation/deployment-intent.json')[0]==intent
    if payload['action']=='inspect-prepared':
        prepared,file=read(base,PREPARED)
        assert prepared['preparation']==actual_preparation(base,run)
        print('DEPLOYMENT_JSON:'+json.dumps(dict(role='actual_actor_probe_prepared_review_records',
            prepared=prepared,prepared_receipt_sha256=file['sha256'],source_commit=run['source_commit'],
            source_sha256=run['source_sha256'],native_algorithm_accepted=False)),flush=True)
        return
    assert payload['action']=='launch'
    verify(base,run)  # Real independent acceptance, never manufacture a review.
    expected=launch_payload(intent,repo)
    assert all(payload[key]==value for key,value in expected.items())
    launch=dict(intent=intent,**expected)
    # The existing launcher rejects every process whose cwd is the prepared
    # repo. This read-only reviewer process must leave it before that inventory;
    # the trainer receives the repository cwd only inside the exclusive Popen.
    os.chdir(base)
    original=sys.stdin
    try:
        sys.stdin=io.StringIO(json.dumps(launch))
        exec(deploy.REMOTE_LAUNCH_PREPARED,{'__name__':'__main__'})
    finally:sys.stdin=original


def perform(handle,action):
    handle=Path(handle).resolve();intent=validate_intent(json.loads(handle.read_text()))
    if action=='observe':
        return deploy.perform(SimpleNamespace(handle=handle,execute=False,launch_prepared=False,retry_unclaimed=False))
    assert action in ('prepare','inspect-prepared','launch')
    payload=dict(intent=intent,action=action)
    if action=='launch':payload.update(launch_payload(intent,Path(__file__).resolve().parents[2]))
    if action in ('prepare','launch'):
        deploy.write_once(handle.with_name(handle.stem+'.'+action+'-once.json'),
            dict(deployment_id=intent['deployment_id'],claimed_unix=time.time(),no_automatic_retry=True))
    stderr=handle.with_name(handle.stem+'.'+action+'-'+str(time.time_ns())+'.stderr')
    output=b'';exit_code=None;failure_type=None
    with stderr.open('xb') as errors:
        try:
            result=subprocess.run(deploy.ssh_command(REMOTE),input=json.dumps(payload,allow_nan=False).encode(),
                stdout=subprocess.PIPE,stderr=errors,timeout=1000 if action=='prepare' else 650 if action=='inspect-prepared' else 240)
            output=result.stdout;exit_code=result.returncode
        except (subprocess.TimeoutExpired,OSError) as failure:
            failure_type=type(failure).__name__;output=getattr(failure,'stdout',None) or b''
        errors.flush();os.fsync(errors.fileno())
    assert len(output)<=16*1024**2, 'Unexpected non-metadata output; preserve the exact handle'
    rows=[json.loads(line.split(':',1)[1]) for line in output.decode(errors='replace').splitlines()
        if line.startswith('DEPLOYMENT_JSON:')]
    receipt=handle.with_name(handle.stem+'.'+action+'-'+str(time.time_ns())+'.json')
    deploy.write_once(receipt,dict(action=action,deployment_id=intent['deployment_id'],exit_code=exit_code,
        failure_type=failure_type,observations=rows,transport_stderr=str(stderr),checked_unix=time.time()))
    print(json.dumps(dict(handle=str(handle),receipt=str(receipt),exit_code=exit_code,failure_type=failure_type,
        observation=rows[-1] if rows else None,next_action='inspect the same handle; no automatic retry')))
    return exit_code if exit_code is not None else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handle',type=Path)
    a=p.add_mutually_exclusive_group();a.add_argument('--prepare',action='store_true');a.add_argument('--launch',action='store_true')
    a.add_argument('--observe',action='store_true');a.add_argument('--inspect-prepared',action='store_true')
    p.add_argument('--predecessor-handle',type=Path);p.add_argument('--source-staging-complete',type=Path)
    p.add_argument('--predecessor-binding',type=Path);p.add_argument('--source-commit')
    p.add_argument('--remote-action',choices=['inspect-prepared','launch'],help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.remote_action:
        payload=json.load(sys.stdin);assert payload['action']==args.remote_action;remote_action(payload);return 0
    if args.prepare or args.launch or args.observe or args.inspect_prepared:
        assert args.handle is not None
        assert not any((args.predecessor_handle,args.source_staging_complete,args.predecessor_binding,args.source_commit))
        action='prepare' if args.prepare else 'launch' if args.launch else 'observe' if args.observe else 'inspect-prepared'
        return perform(args.handle,action)
    assert all((args.handle,args.predecessor_handle,args.source_staging_complete,args.predecessor_binding,args.source_commit))
    intent=create_intent(json.loads(args.predecessor_handle.read_text()),json.loads(args.source_staging_complete.read_text()),
        json.loads(args.predecessor_binding.read_text()),Path(__file__).resolve().parents[2],args.source_commit)
    deploy.write_once(args.handle.resolve(),intent);print(json.dumps(dict(handle=str(args.handle.resolve()),base=intent['base'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
