#!/usr/bin/env python3
"""Owned hardware raw capture only. No GPU calls, signals, submissions or PASS receipt.

Default prints the planned collection. --collect requires an explicit job ID and
captures terminal scheduler evidence before archiving immutable regular files.
The native-state author must not sign their own mathematical acceptance.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = '/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-20261004'
SOURCE = '2175549691d6856e2b393b2834a32febe41ee56e'
REQUIRED_RUN = ['source-commit.txt', 'slurm-terminal.txt', 'own-process-cleanup.json',
    'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY', 'hardware.log',
    'retrieval-comparison.json', 'resource-watch.jsonl', 'model-gpu.json',
    'retrieval-cpu.json', 'retrieval-gpu.json', 'gpu-embeddings.npz',
    'parity-cpu.log', 'parity-gpu.log', 'parity-cpu-time.log',
    'parity-gpu-time.log', 'launch-approval.json', 'tools-manifest.json', 'cpu-gate.json']
EXCLUDED_PARTS = {'cache', 'runtime', '__pycache__', '.cache'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def safe_extract(archive, destination, inventory):
    """No links/special files/traversal, and every regular file must be inventoried."""
    destination = Path(destination)
    if destination.exists(): raise ValueError('Extraction destination already exists')
    expected = {x['archive_path']: x for x in inventory['files']}
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers(); seen = set()
        for member in members:
            p = Path(member.name)
            if p.is_absolute() or '..' in p.parts or not member.isfile():
                raise ValueError('Unsafe/nonregular archive member ' + member.name)
            if member.name in seen or member.name not in expected:
                raise ValueError('Duplicate/uninventoried member ' + member.name)
            if member.size != expected[member.name]['bytes']:
                raise ValueError('Member size mismatch')
            seen.add(member.name)
        if seen != set(expected): raise ValueError('Missing archive members')
        destination.mkdir(parents=True)
        for member in members:
            out = destination / member.name; out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, out.open('xb') as dst:
                while True:
                    block = src.read(1048576)
                    if not block: break
                    dst.write(block)
            if sha(out) != expected[member.name]['sha256']:
                raise ValueError('Raw file hash mismatch: ' + member.name)


TERMINAL_STATES = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL',
    'OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def validate_terminal(raw, job):
    text = raw.decode('utf-8', errors='strict')
    def field(name):
        values = re.findall(r'(?:^|\s)' + re.escape(name) + r'=(\S+)', text)
        if len(values) != 1: raise ValueError('Missing/duplicate terminal field ' + name)
        return values[0]
    if field('JobId') != str(job): raise ValueError('Wrong scheduler job ID')
    if field('JobName') != 'bapo-instruct300-hardware': raise ValueError('Wrong job name')
    if field('NodeList') != 'h100-5': raise ValueError('Wrong hardware node')
    if field('JobState') not in TERMINAL_STATES: raise ValueError('Not an actual terminal state')
    return text


def scheduler_terminal(job, runner):
    """Fallback only for a nonzero live query, never malformed active output."""
    live = runner('scontrol show job -dd ' + str(job))
    if live.returncode == 0:
        validate_terminal(live.stdout, job)
        return live.stdout, dict(origin='LIVE_SCONTROL', live_returncode=0), live.stderr
    path = ROOT + '/scheduler-head/hardware-' + str(job) + '-terminal.txt'
    durable = ROOT + '/scheduler-head/hardware-job-id.txt'
    code = ("import pathlib,sys;p=pathlib.Path(%r);d=pathlib.Path(%r);"
            "assert d.is_file() and not d.is_symlink() and d.read_text().strip()==%r;"
            "assert p.is_file() and not p.is_symlink();sys.stdout.buffer.write(p.read_bytes())"
            % (path, durable, str(job)))
    fallback = runner('python3 -c ' + shlex.quote(code))
    if fallback.returncode != 0:
        raise ValueError('Live scheduler and preserved terminal observations failed; no terminal inferred')
    validate_terminal(fallback.stdout, job)
    return fallback.stdout, dict(origin='HEAD_WATCHER_PRESERVED_ORIGINAL',
        path=path, durable_job_id_path=durable, live_returncode=live.returncode,
        live_stderr=live.stderr.decode('utf-8', errors='replace')), live.stderr


def worker_code(job, terminal, observation_origin=None):
    # All mutations are new owned evidence records; never source/tools/checkpoints.
    return '''import pathlib,json,hashlib,tarfile,os,subprocess,time,re
ROOT=pathlib.Path(%r);SOURCE=%r;job=%d;terminal=%r;observation_origin=%r
run=ROOT/'runs'/('bapo-instruct300-hardware-%%d'%%job)
assert run.is_dir() and not run.is_symlink()
assert 'JobId='+str(job)+' ' in terminal and 'JobState=' in terminal
assert not any('JobState='+s in terminal for s in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED'))
path=run/'slurm-terminal.txt'
if path.exists():
 existing=path.read_text();assert 'JobId='+str(job)+' ' in existing
 assert re.findall(r'(?:^|\s)JobName=(\S+)',existing)==['bapo-instruct300-hardware']
 assert re.findall(r'(?:^|\s)NodeList=(\S+)',existing)==['h100-5']
 states=re.findall(r'(?:^|\s)JobState=(\S+)',existing)
 assert len(states)==1 and states[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED')
 assert not any('JobState='+s in existing for s in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED'))
else:
 with path.open('xb')as f:f.write(terminal.encode('utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
actual=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
manifest=json.loads((ROOT/'logs/tools-manifest.json').read_text())
required=[run/name for name in %r]
required += [run/k/('rank-%%d.json'%%r) for k in ('native-actor','native-state','numeric-fixture','flash-native') for r in range(8)]
required += [run/'native-state'/profile/directory/('rank_%%d.%%s'%%(rank,suffix)) for profile in ('baseline-fp32','production-nested-bf16-manual') for directory in ('initial-step5','roundtrip-step5','no-roundtrip-step6','reloaded-step6') for rank in range(8) for suffix in ('pt','json')]
required += [ROOT/'source-expected.txt',ROOT/'common-data/manifest.json',ROOT/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json']
required += [ROOT/'logs'/name for name in ('launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate.log')]
required += [ROOT/'tools'/name for name in manifest['files']]
required += [p for p in (ROOT/'tools').iterdir() if p.is_file()]
for name in ('cpu-gate-source.txt','isolation-setup.json'):
 p=ROOT/'logs'/name
 if p.is_file():required.append(p)
files=set(required)
for p in run.rglob('*'):
 if p.is_file() and not set(p.relative_to(run).parts)&set(%r):files.add(p)
# Direct actual source identities used by fixtures, not an indirect manifest alone.
for name in ('verl/workers/actor/dp_actor.py','search_r1/baselines/objectives.py','search_r1/baselines/gradient_norm.py','verl/trainer/ppo/core_algos.py','verl/utils/fsdp_utils.py','verl/workers/fsdp_workers.py','verl/protocol.py','verl/utils/torch_functional.py','search_r1/baselines/distillation.py','search_r1/baselines/full_state.py','search_r1/baselines/checkpoint_gate.py'):
 files.add(ROOT/'repo'/name)
missing=[str(p)for p in required if not p.is_file()]
records=[]
for p in sorted(files):
 if not p.is_file():continue
 assert not p.is_symlink() and ROOT in p.resolve().parents
 st=p.stat();records.append(dict(path=str(p),archive_path=str(p.relative_to(ROOT)),bytes=st.st_size,sha256=sha(p)))
stamp=time.strftime('%%Y%%m%%dT%%H%%M%%SZ',time.gmtime())
base=ROOT/'logs'/('hardware-%%d-raw-capture-%%s-%%d'%%(job,stamp,os.getpid()))
archive=base.with_suffix('.tar.gz');inventory=base.with_suffix('.json')
with tarfile.open(archive,'x:gz')as tf:
 for item in records:
  p=pathlib.Path(item['path']);assert sha(p)==item['sha256']
  tf.add(p,arcname=item['archive_path'],recursive=False)
# Detect concurrent changes after archive as well; do not bless mutable evidence.
for item in records:assert sha(pathlib.Path(item['path']))==item['sha256']
tool_issues=[name for name,d in manifest['files'].items() if sha(ROOT/'tools'/name)!=d]
actual_tools={p.name for p in (ROOT/'tools').iterdir() if p.is_file()}
tool_issues += sorted(actual_tools-set(manifest['files']))
assert len(manifest['files'])==20
out=dict(status='RAW_CAPTURE_ONLY_REQUIRES_NONAUTHOR_INDEPENDENT_REVIEW',root=str(ROOT),run_root=str(run),job_id=job,expected_source=SOURCE,actual_source=actual,terminal_observation=observation_origin,source_clean=not bool(dirty),source_matches=actual==SOURCE,terminal_completed='JobState=COMPLETED' in terminal and 'ExitCode=0:0' in terminal,missing_required_files=missing,frozen_tool_hash_mismatches=tool_issues,files=records,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),author_conflict='collector authored native-state fixture; cannot sign its mathematics')
with inventory.open('x')as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out,allow_nan=False))
''' % (ROOT, SOURCE, job, terminal, observation_origin, REQUIRED_RUN, sorted(EXCLUDED_PARTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    if args.job_id <= 0 or not 1 <= args.timeout <= 60: parser.error('Positive job ID; timeout 1..60 seconds')
    transport = Path(__file__).resolve().parents[1] / 'h100_exec.py'
    def command(node, cmd, stdin_file=None):
        result = [sys.executable, str(transport), '--socks', '--node', node, '--timeout', str(args.timeout)]
        if stdin_file: result += ['--stdin-file', str(stdin_file)]
        return result + [cmd]
    if not args.collect:
        print(json.dumps(dict(status='PLAN_ONLY_NO_REMOTE_ACTION',job_id=args.job_id,root=ROOT,
            scheduler_node='h100-16',worker_node='h100-5',output=str(args.output),
            required_run_files=REQUIRED_RUN,expected_source=SOURCE,signs_receipt=False),indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    def scheduler_runner(cmd):
        return subprocess.run(command('h100-16',cmd),capture_output=True,check=False)
    terminal_raw, origin, stderr = scheduler_terminal(args.job_id, scheduler_runner)
    terminal_path = args.output / 'scheduler-terminal-original.txt'
    with terminal_path.open('xb') as handle: handle.write(terminal_raw)
    with (args.output/'scheduler-read.stderr').open('xb') as handle: handle.write(stderr)
    with (args.output/'scheduler-observation.json').open('x') as handle:
        json.dump(origin,handle,indent=2)
    terminal = terminal_raw.decode('utf-8')
    with tempfile.TemporaryDirectory(prefix='e1-hardware-capture-') as tmp:
        script = Path(tmp)/'capture.py';script.write_text(worker_code(args.job_id,terminal,origin))
        local_inventory=args.output/'inventory.json'
        with local_inventory.open('xb') as stdout, (args.output/'worker-capture.stderr').open('xb') as stderr:
            subprocess.run(command('h100-5','python3 -',script),stdout=stdout,stderr=stderr,check=True)
    inventory=json.loads(local_inventory.read_text())
    remote_archive=inventory['archive']['path']
    if not remote_archive.startswith(ROOT+'/logs/') or not remote_archive.endswith('.tar.gz'):
        raise ValueError('Archive outside owned logs')
    archive=args.output/'raw-evidence.tar.gz'
    with archive.open('xb') as stdout,(args.output/'download.stderr').open('xb') as stderr:
        subprocess.run(command('h100-5','cat -- '+shlex.quote(remote_archive)),stdout=stdout,stderr=stderr,check=True)
    if archive.stat().st_size!=inventory['archive']['bytes'] or sha(archive)!=inventory['archive']['sha256']:
        raise ValueError('Archive size/SHA mismatch; partial kept, not extracted')
    safe_extract(archive,args.output/'raw',inventory)
    result=dict(status='RAW_SHA_VERIFIED_SAFE_EXTRACT_ONLY',inventory_sha256=sha(local_inventory),
        archive_sha256=sha(archive),regular_files=len(inventory['files']),missing=inventory['missing_required_files'],
        tool_mismatches=inventory['frozen_tool_hash_mismatches'],signs_receipt=False,
        requires_nonauthor_native_state_review=True)
    (args.output/'capture-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
