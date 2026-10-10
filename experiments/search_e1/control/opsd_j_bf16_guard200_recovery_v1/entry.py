"""Same config/output/CP100 state; retain successful original outer11 generation, fix only 0..9 guards."""
import os,sys,json,time,argparse
from pathlib import Path
from types import SimpleNamespace
from bootstrap import B,H,R,cfg,manifest
from controller import *
class Recovery(Controller):
    def spawn(self,command,log,phase,overrides=None):
        target=str(H/'trainer.py')
        if target in command:
            command=list(command);i=command.index(target);command[i:i+1]=[str(R/'child_entry.py'),target]
        return super().spawn(command,log,phase,overrides)
    def sample(self,outer,questions,identity):
        if outer.name!='outer-00010':return super().sample(outer,questions,identity)
        attempt=outer/'sampling-attempt-00000'
        old_identity,model_ref=checked_json(fileproof(attempt/'sampling-model.json'))
        require(identity==old_identity,'Retained sampler used different CP100 weights')
        candidates=[]
        for path in (self.output/'phase-exits').glob('*.json'):
            value=json.loads(path.read_text())
            if value.get('phase')=='sampling-running' and len(value['child_exit_evidence'])==8 and str(attempt/'rollouts-0.jsonl') in value['child_exit_evidence'][0]['identity']['argv']:candidates.append((path,value))
        require(len(candidates)==1,'Unique original actual sampling terminal required')
        path,terminal=candidates[0];require(terminal['run_id']==self.run_id and terminal['config_sha256']==self.config_ref['sha256'],'Original sampler config/run differs')
        evidence=terminal['child_exit_evidence'];require(all(e['actual_wait_returncode']==0 for e in evidence),'Original samplers failed')
        require(all(process_identity(e['identity']['pid']) is None for e in evidence),'Old sampler remains live')
        jobs=[(SimpleNamespace(pid=e['identity']['pid']),None,e['identity']) for e in evidence]
        release=self.wait_release([]);exit_proof=dict(accepted=True,child_exit_evidence=evidence,gpu_release=release,actual_terminal=fileproof(path))
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
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args();pub=json.loads(Path(a.publication).read_text());require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha'],'Repair source not published')
    require(os.environ['E1_RUN_ID']==cfg['run_id'],'Wrong actual launch environment')
    status=json.loads((B/'runs/status.json').read_text());require(status['phase']=='FAILED_OWNED_RELEASE_CONFIRMED' and status['error']=='Explicit seven-group variant and outer0..9 required','Wrong failure scope')
    require(not Path('/proc/'+str(status['owner']['pid'])).exists(),'Previous controller still exists')
    proof=dict(accepted=True,previous_failure=status,repair_publication=pub,repair_manifest=fileproof(R/'source-manifest.json'),same_config=fileproof(B/'receipts/training-config.json'),only_guard_changed=True,no_resampling=True,no_Adam_reset=True,no512_evaluation=True,observed_unix=time.time());exclusive_json(B/'receipts/guard200-recovery-launch.json',proof)
    x=Recovery(fileproof(B/'receipts/training-config.json'),cfg['run_id'],resume=True)
    try:x.run()
    except Exception as exc:
        if x.owns_output:x.contain_failure(exc)
        raise
