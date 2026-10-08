#!/usr/bin/env python3
"""Read-only actual behavior V2 fifth-outer collection; never creates a release receipt."""
import argparse,importlib.util,json,os,subprocess,time
from pathlib import Path
from first5_helpers import *

def collect(output):
    bind_source();output=owned(output);require(RUN/'logs' in output.parents and not output.exists(),'Fresh run/logs evidence directory required')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    output.mkdir();raw=output/'raw';raw.mkdir();started=time.monotonic()
    result=dict(status='ACTUAL_FIRST5_COLLECTION_REQUIRES_INDEPENDENT_REVIEW',source_commit=SOURCE,run_root=str(RUN),job_id=JOB,
        tool=binding(Path(__file__).resolve()),helper=binding(Path(__file__).with_name('first5_helpers.py').resolve()),native_training_accepted=False)
    try:
        from search_r1.baselines import fresh200_gate as gate
        from search_r1.baselines.full_state import state_digest
        from native_process_strict import verify_native_process, ORIGINAL_SOURCE_SHA, conservation_proof
        result['native_process_adapter']=binding(Path(__file__).with_name('native_process_strict.py').resolve())
        result['original_native_verifier']=binding(ROOT/'repo/search_r1/baselines/fresh200_native_evidence.py')
        require(result['original_native_verifier']['sha256']==ORIGINAL_SOURCE_SHA,'Frozen original verifier bytes changed')
        proof=conservation_proof(Path(result['original_native_verifier']['path']))
        proof_path=raw/'native-process-conservation.json';write_new(proof_path,proof)
        result['native_process_conservation']=binding(proof_path)
        result['native_process_adapter_scope']='Only collector allocation-format call replaced; raw initial process/Slurm bytes unchanged; original remaining statements conserved'
        import torch
        require(not torch.cuda.is_initialized(),'CPU evidence collector initialized CUDA')
        prefix=immutable_prefix(RUN/'train.log',raw/'step5-prefix.log');result['train_log_prefix']=prefix
        text=(raw/'step5-prefix.log').read_text();pauses=marked(text,'FRESH200_STEP5_PAUSED_JSON=')
        require(len(pauses)==1 and not marked(text,'FRESH200_STEP5_RELEASED_JSON='),'Exactly one unreleased first5 pause required')
        pause=pauses[0];mp=RUN/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json'
        require(pause.get('manifest_path')==str(mp)and pause.get('manifest_sha256')==digest(mp)and type(pause.get('pid'))is int,'Actual pause manifest/PID changed')
        m=json.loads(mp.read_text());require(m['source_commit']==SOURCE and m['run_root']==str(RUN)and m['actual_outer_updates']==5,'Exact new source/run/fifth outer required')
        require(m['version']==2 and m['protocol']=='BAPO_FRESH200_V1'and m['update_semantics']=='SKIP_CONTINUE_ACCEPTED_OUTERS_V1','Fresh skip protocol required')
        members={x['path']:x['sha256']for x in m['files']};require(len(members)==len(m['files']),'Duplicate member')
        for item in m['files']:
            path=owned(item['path']);require(RUN in path.parents and path.is_file()and path.stat().st_size==item['bytes']and digest(path)==item['sha256'],'Missing/changed native/HF member '+str(path))
        def member(name):
            matches=[Path(p)for p in members if Path(p).name==name];require(len(matches)==1,'Unique '+name+' required');return matches[0]
        cfg=json.loads(member('resolved-config.json').read_text());require(gate.validate_gate_config(cfg)==(200,57),'Frozen formal200 config differs')
        require(cfg['trainer']['fresh200_gate_receipt']==str(RUN/'logs/fresh200-step5-independent-release.json'),'Exact new release location')
        require(not Path(cfg['trainer']['fresh200_gate_receipt']).exists(),'Already released process cannot be retrospectively admitted')
        attempts=marked(text,'FRESH200_ATTEMPT_JSON=');accepted=gate.validate_first_five_attempts(attempts,m['attempt_accounting'],pause['pid'])
        metrics=marked(text,'BAPO_TRAIN_METRICS_JSON=');require(len(metrics)==len(attempts),'Each attempt needs metrics')
        require([x['step']for x in metrics if x.get('accepted_complete_outer')is True]==[1,2,3,4,5],'Exact accepted outer sequence required')
        for a,e in zip(attempts,metrics):
            require(e.get('attempt_id')==a['attempt_id']and e.get('accepted_complete_outer')is a['accepted_complete_outer']and e.get('metrics',{}).get('actor/optimizer_steps')==a['rank_write_ledger'][0]['optimizer_steps'],'Metrics versus actual writes differs')
            require(a.get('source_commit')==SOURCE and members.get(a['ledger_path'])==digest(a['ledger_path'])and json.loads(Path(a['ledger_path']).read_text())==a,'Attempt raw ledger changed')
        reports=m['rank_roundtrips'];require(len(reports)==8 and {r['rank']for r in reports}==set(range(8))and len({r['pid']for r in reports})==8,'Eight distinct actual workers required')
        process=process_proof(pause['pid'],RUN);require(process['selected_env'].get('SLURM_JOB_ID')==str(JOB),'Driver job changed')
        result['driver_process']=process;native=[]
        for r in reports:
            require(r.get('status')=='EXACT_FULL_STATE_RELOAD_OK'and r.get('scheduler_last_epoch')==5 and r.get('step')==5 and r.get('world_size')==8 and r.get('roundtrip')is True,'Native5 roundtrip/scheduler invalid')
            require(r.get('state_digest')==r.get('reloaded_state_digest')and len(r['state_digest'])==64 and r.get('generation_rng_before_digest')==r.get('generation_rng_after_digest'),'Native model/Adam/scheduler/RNG roundtrip differs')
            require(r.get('model_state_dict_type')=='SHARDED_STATE_DICT'and r.get('model_dtensor_entries',0)>0 and r.get('optimizer_state_entries',0)>0,'Actual native fullstate missing')
            require(members.get(r['path'])==r['sha256']and json.loads(Path(r['path']).with_suffix('.json').read_text())==r,'Actual native report differs')
            initial=json.loads(member('native-initial-rank_'+str(r['rank'])+'.json').read_text());p=initial['native_process'];verify_native_process(p,SOURCE)
            live=process_proof(r['pid'],RUN);require(str(live['start_ticks'])==p['proc_start_ticks']and p['job_id']==str(JOB)and p['run_root']==str(RUN),'Worker PIDbirth/job/run differs')
            native.append(dict(rank=r['rank'],initial_process=p,current_process=live,roundtrip=r))
        require(len({x['initial_process']['gpu_uuid']for x in native})==8,'Actual8 workers need distinct allocated UUIDs')
        proof=behavior_proof(gate.validate_first_sampling_evidence(members,cfg,SOURCE,RUN,pause['pid'],reports))
        require(any(a['attempt_id']==proof['attempt_id']and a['candidate_outer']==proof['actual_outer']for a in accepted),'Raw proof was not an actual accepted attempt')
        state=torch.load(member('driver.pt'),map_location='cpu',weights_only=False);reload=json.loads(member('driver-reload.json').read_text())
        require(reload['state_digest']==state_digest(state)and reload['pid']==pause['pid']and reload['status']=='EXACT_DRIVER_RELOAD_OK','Actual driver reload differs')
        require(state['global_step']==5 and state['next_global_step']==6 and state['actual_outer_updates']==5 and state['progress']==m['attempt_accounting'],'Driver accepted counter differs')
        require(all(key in state for key in ('sampler','rng','bapo','reward_metrics','attempt_id')),'Native driver sampler/RNG/BAPO/reward fields missing')
        result.update(manifest=binding(mp),pause=pause,attempt_accounting=m['attempt_accounting'],attempts=attempts,sampling_proof=proof,native_roundtrip=native,
            driver_state=dict(state_digest=state_digest(state),keys=sorted(state),reload=reload),offline_rank_pt_unpickle_performed=False,
            native_state_scope='ACTUAL_SAME_RANK_FULL_STATE_RELOAD_REPORTS_AND_COMPLETE_FILE_HASHES_NO_FABRICATED_CPU_PROCESS_GROUP',official_manifest=official_manifest())
        # Read only actual current allocation and frozen guardian. No submission/RPC/flock mutations.
        current=subprocess.check_output(['scontrol','show','job',str(JOB)],text=True)
        (raw/'slurm-current.txt').write_text(current);result['allocation']=scheduler_proof(raw/'slurm-current.txt',JOB)
        journal=RUN/'resource-watch.jsonl'
        journal_prefix=immutable_prefix(journal,raw/'resource-watch.jsonl')
        from guard_snapshot import replay_snapshot
        result['resources']=replay_snapshot(Path(journal_prefix['snapshot_path']),JOB,str(RUN),SOURCE,ROOT/'tools/resource_watch.py',journal_prefix)
        require(journal_prefix['sha256']==result['resources']['raw_sha256'],'Snapshot replay bytes differ')
        for name in ('cpu-process-monitor.jsonl','process-monitor.csv'):
            immutable_prefix(RUN/name,raw/name)
        result['retriever']=live_retriever_proof(raw,RUN,result['resources']['allocated_uuids'])
        result['leases']=lease_proof(process,result['resources']['allocated_uuids'],RUN)
        result['snapshot_adapter']=binding(Path(__file__).with_name('guard_snapshot.py').resolve())
        result['evidence']=m['files']+[binding(mp),result['snapshot_adapter'],result['helper'],result['tool'],result['native_process_adapter'],result['original_native_verifier']]+[binding(p)for p in sorted(raw.iterdir())]
        result['elapsed_collection_seconds']=time.monotonic()-started
        write_new(output/'collection.json',result);return result
    except BaseException as exc:
        result.update(status='FAILED_ACTUAL_COLLECTION_NOT_RELEASE',error=repr(exc),elapsed_collection_seconds=time.monotonic()-started)
        write_new(output/'collection.failed.json',result);raise
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();collect(args.output_dir)
