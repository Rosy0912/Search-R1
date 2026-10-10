"""Readonly historical G1 replay preflight. No model/Torch import or Node writes.

Streams the exact old probe manifest files for fresh SHA. Queries only the
explicit candidate GPU, constrained to this run's auxiliary whitelist and never
its production probe / excluded / training GPUs. Reports sharing, not permission
to allocate. No GPU replay is started by this module.
"""
import argparse, csv, datetime, hashlib, io, json, os, socket, subprocess, sys
from pathlib import Path

def digest(path, chunk=4*1024*1024):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(chunk),b''):value.update(block)
    return value.hexdigest()

def proc_identity(pid):
    p=Path('/proc')/str(pid);text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split()
    raw=(p/'cmdline').read_bytes()
    return dict(pid=int(pid),start_ticks=int(tail[19]),sid=int(tail[3]),uid=p.stat().st_uid,
                state=tail[0],command_sha256=hashlib.sha256(raw.replace(b'\0',b' ').decode(errors='replace').encode()).hexdigest())

def same_process(actual, recorded):
    return all(actual[k]==recorded[k] for k in ['pid','start_ticks','sid','uid']) and actual['state']!='Z'

def candidate_allowed(run, candidate):
    return candidate in run['auxiliary_physical_indices'] and candidate not in run['training_physical_indices'] and candidate not in run['excluded_physical_indices'] and candidate!=run['probe_physical_index']

def parse_csv(text):return [r for r in csv.reader(io.StringIO(text)) if r]

def resource_snapshot(base, run, candidate):
    assert candidate_allowed(run,candidate),'Candidate not an allowed separate auxiliary GPU'
    gp=base/'logs/process-ledger.json';raw=gp.read_bytes();ledger=json.loads(raw);assert ledger['run_id']==run['run_id']
    # Only the recorded retrieval session is allowed to share this auxiliary GPU.
    job=ledger['jobs']['retriever'];records={int(k):v for k,v in job['ledger'].items()}
    cmd=['nvidia-smi','-i',str(candidate),'--query-gpu=index,uuid,memory.total,memory.used,memory.free,name','--format=csv,noheader,nounits']
    observed=subprocess.run(cmd,capture_output=True,text=True,check=True)
    rows=parse_csv(observed.stdout);assert len(rows)==1 and int(rows[0][0])==candidate
    gpu=dict(index=int(rows[0][0]),uuid=rows[0][1].strip(),total_mib=int(rows[0][2]),used_mib=int(rows[0][3]),actual_free_mib=int(rows[0][4]),name=rows[0][5].strip())
    cmd=['nvidia-smi','-i',str(candidate),'--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']
    observed=subprocess.run(cmd,capture_output=True,text=True,check=True)
    processes=[];unknown=[]
    for r in parse_csv(observed.stdout):
        if r[0].strip()!=gpu['uuid']:continue
        pid=int(r[1]);actual=proc_identity(pid);record=records.get(pid)
        known=record is not None and same_process(actual,record)
        actual.update(memory_mib=int(r[2]),recorded_role='retriever' if known else 'UNKNOWN',matched_recorded_pid_tick_sid_uid=known)
        if record is not None:actual['recorded_command_sha256']=hashlib.sha256(record['command'].encode()).hexdigest()
        processes.append(actual)
        if not known:unknown.append(pid)
    return dict(candidate_gpu=gpu,compute_processes=processes,foreign_or_unbound_pids=unknown,
                only_original_retriever_shares_candidate=not unknown,
                dedicated_gpu=False if processes else True,
                sharing_existing_retriever=bool(processes),
                run_owned_auxiliary_candidate=True,resource_authorization_to_allocate=False,
                interpretation='A resource snapshot and ownership evidence; free memory is not an allocation authorization.',
                process_ledger_path=str(gp),process_ledger_fresh_sha256=hashlib.sha256(raw).hexdigest())

def run_audit(base, candidate, max_step):
    manifest_path=base/'logs/run-manifest.json';raw=manifest_path.read_bytes();run=json.loads(raw)
    assert run['run_id']=='db474b6abf654ad9892b02ae67ab6f86' and max_step<=150
    assert socket.gethostname()==run['execution_host'] and manifest_path.stat().st_uid==os.getuid()
    out=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),run_id=run['run_id'],source_commit=run['source_commit'],run_manifest_fresh_sha256=hashlib.sha256(raw).hexdigest(),readonly=True,no_model_inference=True,no_node_write=True,scope_steps=[1,max_step],versions=[],sources=[])
    for rel in ['probe_server.py','verl/trainer/main_ppo_eco.py','search_g1/signals.py','search_r1/llm_agent/generation.py','verl/trainer/ppo/ray_trainer.py','verl/utils/seqlen_balancing.py','reproduction/a800/owned_processes.py']:
        p=base/'repo'/rel;actual=digest(p);expected=run['source_sha256'][rel];assert actual==expected
        out['sources'].append(dict(path=str(p),fresh_sha256=actual,run_bound_sha256=expected))
    updates=[]
    for line in (base/'logs/actual-updates.jsonl').read_bytes().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if r.get('run_id')==run['run_id'] and 1<=r.get('step',0)<=max_step:updates.append(r)
    versions={r['probe_version'] for r in updates};out['completed_steps']=[r['step'] for r in updates];out['completed_trajectory_slots']=len(updates)*2560
    out['uncompleted_requested_steps']=[x for x in range(1,max_step+1) if x not in out['completed_steps']]
    for directory in sorted((base/'artifacts').glob('round_*')):
        mp=directory/'version-manifest.json'
        if not mp.is_file():continue
        m=json.loads(mp.read_bytes())
        if m['version'] not in versions:continue
        row=dict(version=m['version'],round=m['round'],manifest_path=str(mp),manifest_fresh_sha256=digest(mp),encoder=m['encoder'],files=[],head_files=[])
        for name,expected in sorted(m['encoder_sha256'].items()):
            p=Path(m['encoder'])/name;assert p.is_file()
            print('PREFLIGHT_WEIGHT_SHA '+str(m['round'])+' '+name,file=sys.stderr,flush=True)
            actual=digest(p);assert actual==expected,'Historical encoder file changed: '+name
            row['files'].append(dict(path=str(p),bytes=p.stat().st_size,fresh_sha256=actual,manifest_sha256=expected,equal=True))
        for label in ['commit','boundary']:
            p=Path(m[label]);actual=digest(p);expected=m['head_sha256'][label+'_probe.pkl'];assert actual==expected
            row['head_files'].append(dict(role=label,path=str(p),bytes=p.stat().st_size,fresh_sha256=actual,manifest_sha256=expected,equal=True))
        row['all_manifest_files_fresh_equal']=True;out['versions'].append(row)
    assert {r['version'] for r in out['versions']}==versions
    out['total_fresh_verified_encoder_member_bytes']=sum(f['bytes'] for r in out['versions'] for f in r['files'])
    out['resource_snapshot']=resource_snapshot(base,run,candidate)
    out['readiness']='HOLD_GPU_ALLOCATION_AUTHORIZATION_AND_INDEPENDENT_REPLAY_GATE'
    out['torch_imported']='torch' in sys.modules;assert not out['torch_imported'];out['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES');out['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--candidate-gpu',required=True,type=int);p.add_argument('--max-step',type=int,default=150);a=p.parse_args()
    print(json.dumps(run_audit(Path(a.base),a.candidate_gpu,a.max_step),ensure_ascii=False))
