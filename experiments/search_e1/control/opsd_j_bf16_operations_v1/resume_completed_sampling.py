"""Reuse only fully completed first sampling with original actual wait evidence; resume same config/run."""
import os,sys,json,time
from pathlib import Path
from types import SimpleNamespace
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010');ref=B/'receipts/training-config.json';cfg=json.loads(ref.read_text());sys.path.insert(0,cfg['paths']['helper_root'])
from controller import *
class Recovery(Controller):
    def verify_merge_identity(self,identity,index,previous,updates):
        require(identity['checkpoint']==previous and identity['optimizer_updates']==updates
                and identity['source_model_identity']==self.config['source_model_manifest']['sha256']
                and identity['common_initial_adapter']==self.config['common_initial_adapter_manifest']
                and identity['variant']==self.config['variant'] and identity['precision']==self.config['precision'],
                'Merged sampling identity differs from current outer-start raw adapter')
        expected=(checked_json(previous)[0]['all_rank_adapter_sha256'][0] if previous else self.config['common_initial_adapter_tensor_sha256'])
        require(identity['adapter_tensor_sha256']==expected,'Merged raw adapter digest mismatch')
        proof=identity['device_proof']
        require(all(proof[k]==dict(cuda_initialized=False,visible_cuda_devices=0,cuda_visible_devices='') for k in ('before','after')),
                'CPU merge CUDA-free proof missing')
        require(proof['peft_parameters']['all_cpu'] is True and proof['merged_parameters']['all_cpu'] is True
                and proof['merge_calls']==1 and proof['cast_after_merge_count']==(1 if self.config['precision']['mode']=='fp32_master_fp16' else 0),
                'CPU merge actual parameter device/method differs')
        bind_model_files(self.config,identity)

    def sample(self,outer,questions,identity):
        if outer.name!='outer-00000':return super().sample(outer,questions,identity)
        attempt=outer/'sampling-attempt-00000'
        identity,model_ref=checked_json(fileproof(attempt/'sampling-model.json'))
        terminals=[p for p in (self.output/'phase-exits').glob('*.json') if json.loads(p.read_text()).get('phase')=='sampling-running']
        require(len(terminals)==1,'Exactly one preserved sampler terminal needed')
        terminal,terminal_ref=checked_json(fileproof(terminals[0]));evidence=terminal['child_exit_evidence']
        require(len(evidence)==8 and all(e['actual_wait_returncode']==0 for e in evidence),'Original actual wait must be all zero')
        jobs=[(SimpleNamespace(pid=e['identity']['pid']),None,e['identity']) for e in evidence]
        require(all(process_identity(e['identity']['pid']) is None for e in evidence),'Old sampler still present')
        release=self.wait_release([])
        exit_proof=dict(accepted=True,child_exit_evidence=evidence,gpu_release=release,actual_terminal=terminal_ref)
        rows=[];worker_proofs=[];http_proofs=[]
        for rank in range(8):
            ref=fileproof(attempt/f'rollouts-{rank}.jsonl.worker-complete.json');value,proof=checked_json(ref)
            require(value['accepted'] is True and value['rank']==rank and value['rows']==320 and value['run_id']==self.run_id,'Wrong/missing worker completion')
            require(value['config_sha256']==self.config_ref['sha256'] and value['variant']==self.config['variant']
                    and value['precision']==self.config['precision'] and value['pid']==jobs[rank][0].pid
                    and value['uid']==self.owner['uid'],'Worker completion group/process differs')
            require(value['actual_llm_load']['actual_dtype']=='torch.'+self.config['sampling']['dtype'],
                    'Worker did not load the authorized actual inference dtype')
            require(value['sampling_model_identity']==identity['sha256'],'Worker sampled different weight identity')
            checked_file(value['rollouts']);checked_file(value['generation_trace'])
            http_proof=verify_http_trace(value['http_trace'],rank,identity['sha256'],value['http_requests'])
            require(Path(http_proof['path']).resolve()==(attempt/f'http-requests-{rank}.jsonl').resolve(),'Worker HTTP trace path mismatch')
            rows.extend(read_rows(value['rollouts']['path']));worker_proofs.append(proof);http_proofs.append(http_proof)
        sys.path.insert(0,self.config['paths']['runtime'])
        scoring_proof=verify_scoring(self.config)
        from search_e1 import core
        pairs,stats=verify_rollouts(rows,questions,core,self.config['variant'],int(outer.name.split('-')[-1]))
        rollout_ref=write_rows(attempt/'rollouts.jsonl',rows);pair_ref=write_rows(attempt/'pairs.jsonl',pairs)
        receipt=dict(schema='opsd-online-sampling-complete-v1',accepted=True,run_id=self.run_id,
            sampling_model_identity=identity,workers=worker_proofs,exit_proof=exit_proof,
            private_http_traces=http_proofs,scoring_identity=scoring_proof,
            rollouts=rollout_ref,pairs=pair_ref,stats=stats)
        return receipt,exclusive_json(outer/'sampling-complete.json',receipt)

if __name__=='__main__':
    os.environ['E1_RUN_ID']=cfg['run_id']
    x=Recovery(fileproof(ref),cfg['run_id'],resume=True);x.run()
