"""SOURCE080 formal3359 first5 CPU evidence; never a release signer."""
import math
import argparse
import datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
SOURCE='0fe642bd0c71f0b5f9813cbbad7bdd94e0625244'
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
COMMON_SHA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
BORROWED_MODEL=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from aligned_first5_math import require, digest, safetensor_map, resolve_qwen2_tied_alias, tokenizer_proof


def cpu_only():
    # The caller must hide GPUs before Python starts, including unpickling DTensor.
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit empty CUDA_VISIBLE_DEVICES required')
    import torch
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized(),'CPU-only process required')


def official_model_path(logical):
    require(logical==BORROWED_MODEL and logical.resolve(strict=True)==BORROWED_MODEL and logical.is_dir(),'fixed actual shared official model path differs')
    return logical

FORMAL_DEPLOY=ROOT/'tools/deployment/cohost-node11-v11'
FORMAL_TOOL_NAMES={'formal.sbatch','formal_worker.sh','formal_inner.sh','formal_driver.py','formal_service_proof.py','bind_retrieval_proof.py','ray_cli_compat.py','bootstrap.py','formal_admission.py','submit_formal.py','tools/resource_watch.py'}
JOB='3359'
TOKEN='16c570e111f947e2b0026983df2f7938'
def formal_tool_path(name):
    require(name in FORMAL_TOOL_NAMES,'foreign formal tool name')
    return ROOT/name if name=='tools/resource_watch.py' else FORMAL_DEPLOY/name


def regular(path, root=None):
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and not p.is_symlink(), 'noncanonical/symlink evidence')
    if root is not None:require(root in p.parents, 'foreign evidence root')
    require(stat.S_ISREG(p.stat().st_mode),'ordinary evidence file required')
    return p



def validated_original_closure(manifest, gate, run, driver_pid=None):
    """Cross-ROOT originals come exclusively from the actual source lineage gate."""
    members={item['path']:item['sha256']for item in manifest['files']}
    require(len(members)==len(manifest['files']),'duplicate original members')
    lineage=gate.complete5_lineage(run,SOURCE,members,manifest['rank_roundtrips'],driver_pid)
    require(lineage is not None,'actual3359 requires original3357 COMPLETE5 with nested3356 lineage')
    return lineage['originals']


def verify_only_summaries(text):
    require(not marked(text,'FRESH200_ATTEMPT_JSON=')and not marked(text,'BAPO_TRAIN_METRICS_JSON='),'verify-only recovery must not repeat training summaries')


def complete5_timeline(manifest,gate,run,driver_pid=None):
    members={item['path']:item['sha256']for item in manifest['files']}
    require(len(members)==len(manifest['files']),'duplicate original members')
    lineage=gate.complete5_lineage(run,SOURCE,members,manifest['rank_roundtrips'],driver_pid)
    require(lineage is not None,'actual COMPLETE5 source lineage required')
    log=regular(lineage['old_run']/'train.log')
    require(members.get(str(log))==digest(log)and lineage['originals'].get(str(log))==members[str(log)],'original COMPLETE5 timeline missing/unbound')
    raw=log.read_bytes();require(raw.endswith(b'\n'),'incomplete original COMPLETE5 timeline')
    text=raw.decode('utf-8')
    attempts=marked(text,'FRESH200_ATTEMPT_JSON=');events=marked(text,'BAPO_TRAIN_METRICS_JSON=')
    require(len(attempts)==len(events)==5,'exact five original COMPLETE5 summaries required')
    return lineage,attempts,events


def regular_member(path, root, originals):
    p=regular(path)
    if root not in p.parents:
        require(str(p)in originals and digest(p)==originals[str(p)],'outside file not in source-validated original closure')
    return p

def binding(path):
    p=regular(path)
    before=p.stat();value=digest(p);after=p.stat()
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(fields(before)==fields(after),'evidence changed while hashing')
    return dict(path=str(p),sha256=value,bytes=after.st_size)


def verify_original_bindings(records):
    for record in records:
        require(binding(record['path'])==record,'original inputs changed during CPU extraction')


def write_new(path,value):
    with path.open('x')as handle:
        json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def identity(receipt, root=ROOT):
    require(receipt.get('source')==SOURCE and receipt.get('root')==str(root), 'foreign source/root receipt')
    require(receipt.get('scope')=='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION','actual formal submit receipt required')
    require(receipt.get('job')==JOB and receipt.get('token')==TOKEN and receipt.get('host')=='h100-11','actual unique node11 formal3359 required')
    require(receipt.get('deployment_directory')==str(FORMAL_DEPLOY) and set(receipt.get('files',{}))==FORMAL_TOOL_NAMES,'actual node11 launch inventory required')
    job,token=receipt.get('job'),receipt.get('token')
    require(isinstance(job,str)and re.fullmatch(r'[1-9][0-9]*',job)and job not in ('3315','3345'),'new actual formal job required')
    require(isinstance(token,str)and re.fullmatch(r'[0-9a-f]{32}',token),'actual new formal token required')
    run=root/'runs'/('formal-aligned-em200-'+token+'-'+job)
    require(receipt.get('run')==str(run)and receipt.get('comment')=='BAPO_ALIGNED_EM200:formal:'+SOURCE+':'+token,'actual formal run/comment differs')
    return dict(job=job,token=token,run=run)


def bind_source(root=ROOT):
    require(root==ROOT and sys.platform=='linux' and os.uname().nodename=='h100-11' and Path(sys.prefix).resolve()==root/'env','actual fixed node11 isolated Linux environment required')
    repo=root/'repo'
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE,'source080 HEAD differs')
    require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'source080 dirty')
    sys.path.insert(0,str(repo))
    gate=importlib.import_module('search_r1.baselines.fresh200_gate')
    require(Path(gate.__file__).resolve()==repo/'search_r1/baselines/fresh200_gate.py','foreign gate imported')
    provenance=json.loads((HERE/'aligned-first5-math-provenance.json').read_text())
    require(digest(HERE/'aligned_first5_math.py')==provenance['extracted_sha256'],'generic math helper changed')
    return gate


def marked(text,marker):
    return [json.loads(line.split(marker,1)[1])for line in text.splitlines()if marker in line]


def process_identity(pid,run,job):
    require(type(pid)is int and pid>1,'positive original PID required')
    p=Path('/proc')/str(pid)
    before=(p/'stat').read_text().rsplit(')',1)[1].split()
    require(before[0]not in ('Z','X'),'original process not live')
    env=dict(part.split(b'=',1)for part in(p/'environ').read_bytes().split(b'\0')if b'='in part)
    require(env.get(b'BAPO_RUN_ID')==str(run).encode() and env.get(b'SLURM_JOB_ID')==job.encode(),'original process run/job differs')
    uids=[int(v)for line in(p/'status').read_text().splitlines()if line.startswith('Uid:')for v in line.split()[1:]]
    require(len(uids)==4 and all(uid==os.getuid()for uid in uids),'original process belongs to another UID')
    executable=(p/'exe').resolve(strict=True)
    require(executable==Path(sys.executable).resolve(strict=True),'original process uses another executable')
    command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    after=(p/'stat').read_text().rsplit(')',1)[1].split()
    require(before[19]==after[19]and after[0]not in('Z','X'),'original PID identity changed')
    return dict(pid=pid,start_ticks=int(before[19]),state=after[0],command=command,uids=uids,executable=str(executable),
                selected_env={key.decode():env[key].decode()for key in (b'BAPO_RUN_ID',b'SLURM_JOB_ID',b'CUDA_VISIBLE_DEVICES')if key in env})


def first5_evidence(manifest_path, submission, output, gate, root=ROOT, processes=True):
    import torch
    from search_r1.baselines.full_state import state_digest
    who=identity(submission,root);run=who['run']
    p=regular(manifest_path,run);require(p==run/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json','exact new fifth manifest path required')
    manifest_binding=binding(p)
    manifest_raw=p.read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest()==manifest_binding['sha256'] and len(manifest_raw)==manifest_binding['bytes'],'manifest changed during original read')
    verify_original_bindings([manifest_binding])
    manifest=json.loads(manifest_raw)
    write_new(output/'original-step5-manifest.json',manifest)
    expected=dict(version=3,protocol='BAPO_BASELINE_ALIGNED_FRESH200_V1',baseline_aligned=True,
        source_commit=SOURCE,run_root=str(run),global_step=5,actual_outer_updates=5,horizon=200,warmup_steps=57,
        mini_batch_switch_outer=None,update_semantics='COMPLETE20_NO_SKIP_ACCEPTED_OUTERS_V1',
        optimizer_opportunities_before_switch=20,optimizer_opportunities_after_switch=20)
    for key,value in expected.items():require(type(manifest.get(key))is type(value)and manifest.get(key)==value,'aligned manifest identity differs: '+key)
    require(isinstance(manifest.get('pause_nonce'),str)and re.fullmatch(r'[0-9a-f]{64}',manifest['pause_nonce']),'actual pause nonce required')
    originals=validated_original_closure(manifest,gate,run)
    members={}
    hashes=[]
    for item in manifest['files']:
        path=regular_member(item['path'],run,originals);require(str(path)not in members,'duplicate manifest member')
        actual=binding(path);require(actual['sha256']==item['sha256']and actual['bytes']==item['bytes'],'original manifest member changed')
        members[str(path)]=item['sha256'];hashes.append(actual)
    def member(name):
        found=[Path(path)for path in members if Path(path).name==name and run in Path(path).parents]
        require(len(found)==1,'unique original member required: '+name);return found[0]
    require((run/'source-commit.txt').read_text().strip()==SOURCE and str(run/'source-commit.txt')in members,'original run source marker unbound')
    write_new(output/'original-member-bindings.json',hashes)
    config=json.loads(member('resolved-config.json').read_text())
    require(members[str(member('resolved-config.json'))]==manifest['config_sha256'],'config digest differs')
    require(gate.validate_gate_config(config)==(200,57),'production200/57 configuration required')
    require(config['baselines']['algorithm_source_commit']==SOURCE and config['baselines']['alignment']['common_manifest_sha256']==COMMON_SHA,'source080/common0b binding differs')
    cpu_only()
    state=torch.load(member('driver.pt'),map_location='cpu',weights_only=False)
    cpu_only()
    driver=json.loads(member('driver-reload.json').read_text());pid=driver['pid']
    require(driver.get('status')=='EXACT_DRIVER_RELOAD_OK'and driver.get('step')==5
        and driver.get('path')==str(member('driver.pt'))and driver.get('sha256')==members[str(member('driver.pt'))]
        and driver.get('state_digest')==state_digest(state)and state.get('driver_pid')==pid
        and state.get('global_step')==5 and state.get('next_global_step')==6
        and state.get('actual_outer_updates')==5 and state.get('source_commit')==SOURCE
        and state.get('progress')==manifest['attempt_accounting'],'original driver reload/state identity differs')
    log=regular(run/'train.log',run);raw=log.read_bytes();require(raw.endswith(b'\n'),'partial live terminal log line')
    snapshot=output/'train-step5-prefix.log'
    with snapshot.open('xb')as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    prefix=dict(live_path=str(log),snapshot_path=str(snapshot),bytes=len(raw),sha256=digest(snapshot))
    text=raw.decode('utf-8');paused=marked(text,'FRESH200_STEP5_PAUSED_JSON=')
    require(len(paused)==1 and paused[0].get('pid')==pid and paused[0].get('manifest_path')==str(p)
        and paused[0].get('manifest_sha256')==manifest_binding['sha256'],'actual same-PID fifth pause required')
    require(not marked(text,'FRESH200_STEP5_RELEASED_JSON='),'helper only collects original paused fifth state')
    verify_only_summaries(text)
    lineage,attempts,events=complete5_timeline(manifest,gate,run,pid)
    gate.validate_first_five_attempts(attempts,manifest['attempt_accounting'],lineage['old_driver_pid'],baseline_aligned=True)
    require(len(events)==len(attempts)==5,'exact five original accepted summaries required')
    for outer,(attempt,event)in enumerate(zip(attempts,events),1):
        require(event.get('step')==outer and event.get('attempt_id')==attempt['attempt_id']
            and event.get('accepted_complete_outer')is True and event['metrics'].get('actor/optimizer_steps')==20
            and event['metrics'].get('bapo/actual_outer_updates')==outer,'train metrics differ from five complete outers')
        require(attempt.get('source_commit')==lineage['old_source'],'original attempt source differs')
        ledger=regular_member(attempt['ledger_path'],run,originals)
        require(str(ledger)in members and json.loads(ledger.read_text())==attempt,'original all-rank attempt ledger missing')
    reports=manifest['rank_roundtrips']
    require(len(reports)==8 and {r.get('rank')for r in reports}==set(range(8)),'eight original native ranks required')
    for report in reports:
        path=regular(report['path'],run);side=path.with_suffix('.json')
        require(path.name==f"rank_{report['rank']}.pt"and members.get(str(path))==report['sha256']
            and str(side)in members and json.loads(side.read_text())==report,'native rank original bytes/report differ')
        require(report.get('status')=='EXACT_FULL_STATE_RELOAD_OK'and report.get('native_state_version')==3
            and report.get('roundtrip')is True and report.get('step')==5 and report.get('world_size')==8
            and report.get('scheduler_last_epoch')==5 and report.get('model_state_dict_type')=='SHARDED_STATE_DICT'
            and type(report.get('model_dtensor_entries'))is int and report['model_dtensor_entries']>0
            and report.get('optimizer_state_entries',0)>0
            and 'aligned_pending_outer'in report and report['aligned_pending_outer']is None
            and re.fullmatch(r'[0-9a-f]{64}',report.get('state_digest',''))
            and report.get('reloaded_state_digest')==report['state_digest']
            and re.fullmatch(r'[0-9a-f]{64}',report.get('generation_rng_before_digest',''))
            and report.get('generation_rng_after_digest')==report['generation_rng_before_digest']
            and report.get('latest_rank_event')==attempts[-1]['rank_write_ledger'][report['rank']], 'native GPU-produced exact-roundtrip report missing')
    require(state.get('rank_write_ledger')==attempts[-1]['rank_write_ledger'],'driver fifth final ledger differs')
    gate.validate_complete5_first_five(members,config,SOURCE,run,pid,lineage,attempts,events)
    process=[]
    if processes:
        process=[process_identity(pid,run,who['job'])]+[process_identity(r['pid'],run,who['job'])for r in reports]
    verify_original_bindings([manifest_binding])
    return dict(status='CPU_ORIGINAL_ALIGNED_FIRST5_BINDINGS_CHECKED_NOT_RELEASE',manifest=manifest_binding,
        pause_nonce=manifest['pause_nonce'],driver_pid=pid,source_commit=SOURCE,run_root=str(run),job=who['job'],token=who['token'],
        train_log_prefix=prefix,original_member_bindings=hashes,processes=process,
        native_roundtrip_reports=reports,native_roundtrip_role='ORIGINAL_GPU_PRODUCER_REPORTS_HASH_BOUND_NOT_CPU_RELOADED',
        config=config,attempt_accounting=manifest['attempt_accounting'])


def complete_weight_scan(base,checkpoint):
    import torch
    from safetensors import safe_open
    before,after=safetensor_map(base),safetensor_map(checkpoint)
    alias=resolve_qwen2_tied_alias(base,checkpoint,before,after)
    total=changed=0;maximum=0.;l1=0.;dtypes=set()
    for key in sorted(before):
        storage=alias['base_storage_key']if alias and key==alias['base_logical_key']else key
        with safe_open(str(before[key]),framework='pt',device='cpu')as left,safe_open(str(after[key]),framework='pt',device='cpu')as right:
            shape=left.get_slice(storage).get_shape();require(shape==right.get_slice(key).get_shape(),'weight shape differs: '+key)
            for start in range(0,shape[0],1024)if shape else [0]:
                a=left.get_slice(storage)[start:start+1024]if shape else left.get_tensor(storage)
                b=right.get_slice(key)[start:start+1024]if shape else right.get_tensor(key)
                require(torch.isfinite(a).all()and torch.isfinite(b).all(),'nonfinite official/export weight: '+key)
                aligned=a.to(b.dtype);require(torch.isfinite(aligned).all(),'nonfinite base precision cast')
                delta=(aligned.double()-b.double()).abs();count=int(delta.count_nonzero())
                require(torch.isfinite(delta).all(),'nonfinite dtype-aligned difference')
                total+=b.numel();changed+=count;maximum=max(maximum,float(delta.max())if delta.numel()else 0.)
                l1+=float(delta.sum());dtypes.add((str(a.dtype),str(b.dtype)))
    require(changed>0,'no dtype-aligned trained weight change')
    return dict(tensors_checked=len(before),logical_elements_checked=total,changed_elements=changed,max_abs_delta=maximum,
        l1_delta=l1,all_official_and_export_tensors_finite=True,comparison='base_cast_to_export_dtype_then_float64',
        dtype_pairs=sorted(dtypes),tied_weight_resolution=alias)


def cpu_hf(base,checkpoint,official=True):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    cpu_only()
    start=time.monotonic();cpu=time.process_time()
    left=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    right=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    tokenizer=tokenizer_proof(left,right);delta=complete_weight_scan(base,checkpoint)
    model,loading=AutoModelForCausalLM.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False,
        torch_dtype='auto',attn_implementation='eager',output_loading_info=True)
    require(all(not loading.get(k)for k in('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'actual HF loading mismatch')
    require(all(p.device.type=='cpu'and torch.isfinite(p).all()for p in model.parameters()),'all actual HF parameters must be finite CPU')
    count=sum(p.numel()for p in model.parameters());tied=model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    require(model.config.tie_word_embeddings is True and tied,'actual HF tied storage not preserved')
    if official:require(model.config.model_type=='qwen2'and model.config.vocab_size==151936 and model.config.hidden_size==2048 and count==3085938688,'official Qwen2.5-3B dimensions/parameter count differs')
    model.eval();inputs=right('The capital of France is',return_tensors='pt',truncation=True,max_length=16,return_token_type_ids=False)
    with torch.inference_mode():output=model(**inputs,use_cache=False)
    require(output.logits.device.type=='cpu'and torch.isfinite(output.logits).all(),'actual CPU forward nonfinite')
    result=dict(status='ACTUAL_CPU_HF_CHECKED_NOT_RELEASE',scope='official'if official else'CPU_FIXTURE_ONLY',tokenizer=tokenizer,
        parameter_delta=delta,loading_info=loading,parameters=count,tied_input_output_storage=tied,all_parameters_finite=True,
        forward=dict(device='cpu',logits_shape=list(output.logits.shape),next_token_id=int(output.logits[0,-1].argmax()),finite=True),
        cost=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del model,output;gc.collect();cpu_only()
    return result


def full_model_files(model,expected=12):
    p=regular(model/'VERIFIED-MODEL-MANIFEST.json');require(digest(p)==MODEL_SHA,'pinned official manifest changed')
    doc=json.loads(p.read_text());require(len(doc['files'])==expected,'complete official12-file manifest required')
    records=[]
    for item in doc['files']:
        path=regular(model/item['file'],model);actual=binding(path)
        require(actual['bytes']==item['bytes']and actual['sha256']==item['sha256'],'official asset bytes changed')
        records.append(actual)
    return dict(manifest=binding(p),files=records)


def native_cpu_inspection(reports):
    import torch
    results=[]
    for report in reports:
        start=time.monotonic()
        row=dict(rank=report['rank'],path=report['path'],sha256=report['sha256'],reloaded=False,
            scope='CPU_SERIALIZATION_INSPECTION_ONLY_NOT_GPU_ROUNDTRIP')
        cpu_only()
        try:
            payload=torch.load(report['path'],map_location='cpu',weights_only=False)
            require(payload.get('version')==3 and payload.get('rank')==report['rank']and payload.get('step')==5,'native payload metadata differs')
            require(payload.get('latest_rank_event')==report['latest_rank_event'] and payload.get('aligned_pending_outer','MISSING')is None,'native payload final event/pending differs')
            row.update(status='CPU_DESERIALIZED_NOT_RELOADED',payload_version=payload['version']);del payload;gc.collect()
        except Exception as error:
            row.update(status='CPU_NATIVE_INSPECTION_UNAVAILABLE_NOT_RELOADED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        # Outside exception handling: CUDA leakage invalidates the whole extraction.
        cpu_only()
        row['wall_seconds']=time.monotonic()-start;results.append(row)
    return results


def cohost_watch_prefix(who,output):
    """Factual same-job cohost prefix; foreign occupancy is recorded, never idle-gated."""
    run=who['run'];bootstrap=regular(run/'cohost-lease-bootstrap.json',run)
    boot=json.loads(bootstrap.read_bytes())
    require(boot.get('run')==str(run)and boot.get('job')==who['job']and boot.get('policy')=='COHOST_NODE11_INITIAL_FREE_NOT_IDLE','actual node11 lease bootstrap differs')
    uuids={r['uuid']for r in boot['leases']}
    require(len(boot['leases'])==len(uuids)==8 and all(isinstance(u,str)and u.startswith('GPU-')for u in uuids),'actual eight cohost UUIDs required')
    live=regular(run/'resource-watch.jsonl',run);raw=live.read_bytes();end=raw.rfind(b'\n')+1
    require(end>0,'actual complete cohost samples absent')
    prefix=raw[:end];rows=[json.loads(line)for line in prefix.decode('utf-8').splitlines()]
    for row in rows:
        require(row.get('run')==str(run)and row.get('slurm_job_id')==who['job']and row.get('policy')=='COHOST_FREE_RESERVE_NO_FOREIGN_SM_OR_MEMORY_PERCENT_GATE','foreign/old watcher schema')
        require(row.get('pause')is False and row.get('error')is None and row.get('reason')=='within_free_reserve'and row.get('reserve_free_mib')==1024,'cohost reserve fault/unknown sample')
        require(len(row['gpus'])==8 and {r['uuid']for r in row['gpus']}==uuids,'cohost UUIDs changed')
        require(all(type(r['free_mib'])in(int,float)and math.isfinite(r['free_mib'])and r['free_mib']>=1024 for r in row['gpus']),'cohost free reserve evidence differs')
    snapshot=output/'resource-step5-prefix.jsonl'
    with snapshot.open('xb')as f:f.write(prefix);f.flush();os.fsync(f.fileno())
    return dict(scope='ACTUAL_COHOST_RESERVE_PREFIX_NOT_INDEPENDENT_ACCEPTANCE',bootstrap=binding(bootstrap),snapshot=binding(snapshot),live_path=str(live),prefix_bytes=end,prefix_sha256=hashlib.sha256(prefix).hexdigest(),incomplete_tail_bytes=len(raw)-end,incomplete_tail_sha256=hashlib.sha256(raw[end:]).hexdigest(),rows=rows)


def verify_watch_prefix(proof):
    verify_original_bindings([proof['bootstrap'],proof['snapshot']])
    raw=Path(proof['live_path']).read_bytes()
    require(hashlib.sha256(raw[:proof['prefix_bytes']]).hexdigest()==proof['prefix_sha256'],'original cohost prefix changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='new exclusive directory under fixed ROOT/evidence')
    args=parser.parse_args();output=args.output
    require(output.is_absolute()and output.parent==ROOT/'evidence'and output.parent.resolve()==output.parent and not output.is_symlink(),'own canonical evidence output required')
    output.mkdir()
    start=time.monotonic();cpu=time.process_time()
    evidence=dict(schema='aligned-first5-cpu-evidence-candidate-v1',candidate=True,acceptance=False,release_receipt=False,
        source_commit=SOURCE,root=str(ROOT),helper_sha256=digest(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='CANDIDATE_NOT_EXECUTED_OR_NOT_BOUND_NOT_RELEASE',issues=[])
    try:
        cpu_only()
        gate=bind_source()
        import torch,transformers,safetensors
        evidence['runtime']=dict(python=sys.version,executable=sys.executable,platform=sys.platform,torch=torch.__version__,transformers=transformers.__version__,safetensors=safetensors.__version__,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_available=False,cuda_initialized=False)
        evidence['helper_files']=[binding(HERE/name)for name in('aligned_first5_cpu_evidence.py','aligned_first5_math.py','aligned-first5-math-provenance.json')]
        receipt=regular(ROOT/'scheduler-head/formal-submit-receipt.json',ROOT)
        submit=json.loads(receipt.read_text());who=identity(submit)
        evidence['actual_submission']=binding(receipt)
        require(isinstance(submit.get('files'),dict)and bool(submit['files']),'actual formal tools binding required')
        evidence['formal_tool_bindings']=[]
        for name,expected in submit['files'].items():
            record=binding(regular(formal_tool_path(name),ROOT))
            require(record['sha256']==expected,'actual formal tool changed: '+name)
            evidence['formal_tool_bindings'].append(record)
        write_new(output/'original-formal-submit-receipt.json',submit)
        first=first5_evidence(who['run']/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json',submit,output,gate)
        evidence['first5']=first
        evidence['cohost_watch']=cohost_watch_prefix(who,output)
        logical_base=Path(first['config']['actor_rollout_ref']['model']['path'])
        base=official_model_path(logical_base)
        evidence['official_model_location']=dict(logical_path=str(logical_base),resolved_path=str(base),reviewed_borrowed_target=str(BORROWED_MODEL))
        evidence['official_model']=full_model_files(base)
        checkpoint=who['run']/'checkpoints/actor/global_step_5'
        regular(checkpoint/'config.json',who['run'])
        paths=sorted(p for p in checkpoint.rglob('*')if p.is_file())
        require(len(paths)==12,'complete fifth HF12-file export required')
        evidence['hf_files']=[binding(regular(p,who['run']))for p in paths]
        evidence['cpu_hf']=cpu_hf(base,checkpoint)
        evidence['native_cpu_inspection']=native_cpu_inspection(first['native_roundtrip_reports'])
        cpu_only()
        # Rehash all checkpoint inputs after costly CPU work, including originals.
        verify_original_bindings([first['manifest']]+evidence['helper_files']+first['original_member_bindings']
            +evidence['hf_files']+evidence['official_model']['files']+evidence['formal_tool_bindings'])
        require(binding(receipt)==evidence['actual_submission'],'original formal receipt changed')
        require(official_model_path(logical_base)==base,'borrowed model link changed during extraction')
        require(binding(base/'VERIFIED-MODEL-MANIFEST.json')==evidence['official_model']['manifest'],'official manifest changed during extraction')
        live=Path(first['train_log_prefix']['live_path']).read_bytes()
        require(hashlib.sha256(live[:first['train_log_prefix']['bytes']]).hexdigest()==first['train_log_prefix']['sha256']
            and not marked(live.decode('utf-8'),'FRESH200_STEP5_RELEASED_JSON='),'actual fifth live prefix/pause changed during extraction')
        for process in first['processes']:
            after=process_identity(process['pid'],who['run'],who['job'])
            require(after['start_ticks']==process['start_ticks'],'original fifth PID reused during CPU extraction')
        verify_watch_prefix(evidence['cohost_watch'])
        evidence['status']='CPU_EVIDENCE_COLLECTED_NOT_INDEPENDENTLY_REVIEWED_NOT_RELEASE'
    except Exception as error:
        evidence['status']='CPU_EVIDENCE_INCOMPLETE_OR_UNAVAILABLE_NOT_RELEASE'
        evidence['issues'].append(dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
    evidence['cost']=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        monetary_price_assumed=False,scope='CPU extraction cost only; no invented allocated GPU-hours')
    write_new(output/'cpu-evidence.json',evidence)
    print(json.dumps(dict(status=evidence['status'],acceptance=False,release_receipt=False,path=str(output/'cpu-evidence.json'),sha256=digest(output/'cpu-evidence.json'))))

if __name__=='__main__':main()
