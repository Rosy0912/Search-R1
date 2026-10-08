#!/usr/bin/env python3
"""Actual read-only full CPU HF/tokenizer verification of behavior V2 first5 export."""
import argparse,importlib.util,json,os,resource,subprocess,time
from pathlib import Path
from first5_helpers import *
def verify(collection_path):
    bind_source();collection_path=owned(collection_path);collection=json.loads(collection_path.read_text())
    require(collection['status']=='ACTUAL_FIRST5_COLLECTION_REQUIRES_INDEPENDENT_REVIEW'and collection['source_commit']==SOURCE and collection['run_root']==str(RUN)and collection['job_id']==JOB,'Actual new collection required')
    out=collection_path.parent/'hf.json';require(not out.exists(),'HF report must be immutable')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    start=time.monotonic();cpu=time.process_time();value=dict(status='ACTUAL_CPU_HF_REQUIRES_INDEPENDENT_REVIEW',source_commit=SOURCE,run_root=str(RUN),job_id=JOB,collection=binding(collection_path),tool=binding(Path(__file__).resolve()),helper=binding(Path(__file__).with_name('first5_helpers.py').resolve()),native_training_accepted=False)
    try:
        import torch
        torch.set_num_threads(8)
        value['official_manifest']=official_manifest()
        for item in collection['evidence']:
            require(digest(item['path'])==item['sha256'],'Collected evidence changed')
        from search_r1.baselines.fresh200_native_evidence import compare_official_parameters,tensor_digest
        members={Path(x['path']).name:Path(x['path'])for x in collection['evidence']}
        official=torch.load(members['native-initial-official-hf.pt'],map_location='cpu',weights_only=False)
        alignment=json.loads(members['native-initial-official-alignment.json'].read_text())
        hashes=compare_official_parameters(official['model_parameters'],str(MODEL))
        require(hashes==alignment['tensor_sha256']and official['official_model_root']==str(MODEL.resolve()),'Actual initial official full tensor replay differs')
        value['initial_official_tensor_replay']=dict(parameters=len(hashes),tensor_sha256=hashes,official_root=official['official_model_root'])
        del official
        initials=[]
        for rank in range(8):
            initial=torch.load(members['native-initial-rank_'+str(rank)+'.pt'],map_location='cpu',weights_only=False)
            metadata=json.loads(members['native-initial-rank_'+str(rank)+'.json'].read_text())
            observed={name:tensor_digest(tensor)for name,tensor in initial['model_parameters'].items()}
            require(observed==metadata['model_tensor_sha256']and not initial['optimizer']['state']and initial['scheduler']['last_epoch']==0,'Actual native initial parameters/Adam0/scheduler0 differs')
            require(all(g['lr']==1e-6/57 for g in initial['optimizer']['param_groups'])and initial['native_process']==metadata['native_process']and initial['source_sha256']==metadata['source_sha256'],'Raw initial native bindings differ')
            require(all(bool(torch.isfinite(t).all())for t in initial['model_parameters'].values()),'Initial local parameters nonfinite')
            initials.append(dict(rank=rank,parameters=len(observed),adam_state_entries=0,scheduler_last_epoch=0,lr=[1e-6/57]))
            del initial
        value['initial_raw_state_replay']=initials
        value['before_process']=process_proof(collection['pause']['pid'],RUN)
        checkpoint=RUN/'checkpoints/actor/global_step_5'
        value['proof']=hf_proof(MODEL,checkpoint)
        require(not torch.cuda.is_initialized(),'CPU HF unexpectedly initialized CUDA')
        value['after_process']=process_proof(collection['pause']['pid'],RUN)
        require(value['before_process']['start_ticks']==value['after_process']['start_ticks']==collection['driver_process']['start_ticks'],'Paused driver PID changed during CPUHF')
        slurm=collection_path.parent/'raw/slurm-after-hf.txt'
        slurm.write_text(subprocess.check_output(['scontrol','show','job',str(JOB)],text=True))
        value['allocation_after_cpu_hf']=scheduler_proof(slurm,JOB);value['slurm_after_hf']=binding(slurm)
        journal=RUN/'resource-watch.jsonl';snapshot=collection_path.parent/'raw/resource-watch-after-hf.jsonl'
        saved=immutable_prefix(journal,snapshot)
        from guard_snapshot import replay_snapshot
        value['resources_after_cpu_hf']=replay_snapshot(snapshot,JOB,str(RUN),SOURCE,ROOT/'tools/resource_watch.py',saved)
        require(saved['sha256']==value['resources_after_cpu_hf']['raw_sha256'],'PostHF snapshot replay bytes differ')
        value['resource_journal_after_cpu_hf']=binding(snapshot)
        value['snapshot_adapter']=binding(Path(__file__).with_name('guard_snapshot.py').resolve())
        value.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,monetary_price_assumed=False)
        write_new(out,value);return value
    except BaseException as exc:
        value.update(status='FAILED_ACTUAL_CPU_HF_NOT_RELEASE',error=repr(exc),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu)
        write_new(collection_path.parent/'hf.failed.json',value);raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);a=p.parse_args();verify(a.collection)
