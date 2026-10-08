"""Future same-node saved real DataProto mu/Tree bindings; no forwards/writes.

Prepared locally as source only. Never execute on Mac production tensor assets.
"""
import argparse
import json
import os
from pathlib import Path
import time

from calibration_contract import held_tensor,expected_mu_provenance,sha
from production_contract import read_bound,require


def verify(run,package,config_hash,last,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Behavior-mu evidence checker must be CPU-only')
    import numpy as np
    import torch
    from verl import DataProto
    from search_r1.baselines.tree_behavior_mu import validate_behavior_mu_batch
    torch.set_num_threads(8); start=time.perf_counter();cpu=time.process_time()
    expected=dict(job_id=Path(run).name.rsplit('-',1)[1],run_id=Path(run).name,code_commit=package['code_commit'],config_sha256=config_hash,
                  data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    records=[]
    from attempt_accounting import attempt_results
    observed=attempt_results(run,package,config_hash,last)
    for event in sorted(observed['credited']+observed['all_skipped'],key=lambda x:x['attempt']):
        outer=event.get('outer',event.get('candidate_outer'));attempt=event['attempt']
        directory=Path(run)/'output/trajectories'/f'outer-{outer:06d}-attempt-{attempt:06d}'
        manifest,msha,_=read_bound(directory/'manifest.json')
        require(manifest['status']=='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED' and manifest['rows']==2560,'Full five-leaf real actor input evidence missing')
        full=held_tensor(directory/'actor-inputs.pt',manifest['payloads']['actor-inputs.pt'])
        identity=dict(outer_step=outer,attempt_id=attempt,job_id=expected['job_id'],source=expected['code_commit'],config=config_hash,data=expected['data_identity'])
        require(full['rows']==2560 and full['identity']['outer_step']==outer
                and manifest['identity']==full['identity']==identity
                and sha(directory/'trajectories.jsonl.gz')==manifest['payloads']['trajectories.jsonl.gz'], 'Actual Tree trace/input identity differs')
        metadata={}
        for key,rows in full['non_tensor_batch'].items():
            require(len(rows)==2560,'Saved real non-tensor row count differs')
            array=np.empty(2560,dtype=object); array[:]=rows;metadata[key]=array
        data=DataProto.from_dict(tensors=full['tensors'],non_tensors=metadata,meta_info=full['meta_info'])
        receipt=validate_behavior_mu_batch(data,expected_provenance=expected_mu_provenance(expected),required=True)
        require(data.meta_info['tree_behavior_mu_corrected_profile'] is True and torch.equal(data.batch['old_log_probs'],data.batch['sampled_old_log_probs']), 'Actual full training denominator is not fixed sampled mu')
        width=data.batch['responses'].shape[1];valid=data.batch['attention_mask'][:,-width:].bool() & data.batch['loss_mask'].bool(); beta=data.batch['actor_old_log_probs']
        require(beta.shape==data.batch['sampled_old_log_probs'].shape and beta.is_floating_point() and torch.isfinite(beta[valid]).all(), 'Actual independent full beta evidence invalid')
        records.append(dict(candidate_outer=outer,attempt_id=attempt,credited_outer=('outer' in event),manifest_sha256=msha,actor_input_sha256=manifest['payloads']['actor-inputs.pt'],actual_rows=2560,
                            supervised_tokens=int(valid.sum()),mu_binding_receipt=receipt,independent_full_beta_finite=True))
        del data,full
    result=dict(status='FRESH200_REAL_BEHAVIOR_MU_CPU_BINDING_VERIFIED',identity=expected,completed_outers_checked=last,
                outer_records=records,real_DataProto=True,optimizer_writes=0,model_forwards=0,cuda_initialized=torch.cuda.is_initialized(),
                training_attempts_checked=observed['training_attempts'],
                original_sampler_actor_gate_accepted=False,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu)
    require(result['cuda_initialized'] is False,'CPU evidence checker initialized CUDA')
    with Path(output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--package',required=True)
    parser.add_argument('--config-sha',required=True);parser.add_argument('--outer',required=True,type=int);parser.add_argument('--output',required=True)
    args=parser.parse_args();package,_,_=read_bound(Path(args.package)/'manifest.json')
    print(json.dumps(verify(args.run,package,args.config_sha,args.outer,args.output),sort_keys=True))
