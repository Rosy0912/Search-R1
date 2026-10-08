"""Synthetic runtime contract refusals; no Torch, assets, GPUs or remote calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-safety110-a632ba9-v2')
OUT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-safety110-review')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def main():
    contract = load('calibration_contract', ROOT/'calibration_contract.py')
    builder = load('builder', ROOT.parent/'build_tree_safety110_runtime.py')
    observer = load('observer', ROOT/'validation_cost_hooks_instruct300.py')
    parent = json.loads((ROOT/'HEALTHY110_PARENT.json').read_bytes())
    old = json.loads((ROOT.parent/'tree-recovery-runtime/manifest.json').read_bytes())
    model = old['model_identity']
    package = dict(parent_manifest={key: parent[key] for key in ('path','sha256','identity')},
                   healthy110_metadata=parent, model_identity=model,
                   common_data_identity=parent['identity']['data_identity'],
                   code_commit='0123456789abcdef0123456789abcdef01234567',
                   probability_calibration={'ready_status': 'CALIBRATION_OBSERVED_ZERO_WRITES'})
    restoration = dict(status='COLD_PARENT_STATE_LOADED_PENDING_EVALUATION',
        workers=[dict(rank=row['identity']['rank'], parent_identity=row['identity'],
                      scheduler=110, adam_steps=[2500], parent_sha256=row['sha256'])
                 for row in parent['worker_states']],
        sampler_migration=dict(parent_batch_size=640,batch_size=640,consumed_cursor=57600,
                               epoch=0,permutation_sha256=contract.PERMUTATION,exact_executor_resume=False),
        provenance=dict(parent_identity=parent['identity'],parent_completed_step=110,
                        parent_cursor=57600,parent_epoch=0,new_batch_size=640,exact_executor_resume=False))
    ready = dict(status='CALIBRATION_OBSERVED_ZERO_WRITES',code_commit=package['code_commit'],
                 config_sha256='a'*64,job_id='999999',data_identity=package['common_data_identity'],
                 model_identity=model['sha256'],parent_completed_outer=110,observed_outer=111,
                 new_completed_outer=0,optimizer_committed=0,scheduler_advanced=False,
                 sampler_cursor_before_outer=57600,sampler_cursor_after_observation=58240,
                 workers=[dict(rank=rank,world_size=8,parent_completed_outer=110,observed_outer=111,
                               optimizer_committed=0,scheduler_advanced=False,restored_optimizer_steps=[2500],
                               scheduler_step=110,diagnostic_records=[]) for rank in range(8)])
    checks=[]
    def refused(name, function):
        try:
            function()
        except (ValueError, AssertionError):
            checks.append(name+'_refused')
        else:
            raise AssertionError('Invalid fixture was admitted: '+name)
    with tempfile.TemporaryDirectory(prefix='contract-test-',dir=OUT) as temporary:
        run=Path(temporary)/'tree-safety300-999999'
        (run/'output/probability-calibration').mkdir(parents=True)
        sampling=run/'output/sampling-consistency';sampling.mkdir()
        for rank in range(8):
            identity=dict(job_id='999999',run_id=run.name,code_commit=package['code_commit'],
                          config_sha256='a'*64,data_identity=package['common_data_identity'],model_identity=model['sha256'])
            snapshot=sampling/f'rank-{rank:02d}-call-00001.pt';snapshot.write_bytes(b'synthetic test fixture, not a model state')
            diagnostic=sampling/f'rank-{rank:02d}-call-00001.json'
            diagnostic.write_text(json.dumps(dict(rank=rank,world_size=8,scheduler_step=110,
                optimizer_committed=0,**identity,enforced=False,snapshot_path=str(snapshot),
                generation_call=1,local_rows=2,saved_rows=2,decision={'allowed':False,'reasons':['trial_observation']},
                snapshot_sha256=contract.sha(snapshot))))
            ready['workers'][rank]['diagnostic_records']=[str(diagnostic)]
            monitor=run/'output/probability-calibration'/f'rank-{rank:02d}-monitor.pt';monitor.write_bytes(b'synthetic monitor fixture')
            ready['workers'][rank].update(**identity,decision={'allowed':False,'reasons':['trial_observation']},
                monitor_path=str(monitor),monitor_sha256=contract.sha(monitor),diagnostic_entries=[
                    dict(path=str(diagnostic),sha256=contract.sha(diagnostic),snapshot_path=str(snapshot),snapshot_sha256=contract.sha(snapshot))])
        restoration_path=run/'output/recovery-loaded.json'
        ready_path=run/'output/probability-calibration/ready.json'
        restoration_path.write_text(json.dumps(restoration));ready_path.write_text(json.dumps(ready))
        contract.validate_restored110(run,package)
        contract.validate_calibration_ready(ready_path,run,package,'a'*64)
        checks.append('valid_schema_and_diagnostic_snapshot_sha')
        for key,bad in [('optimizer_committed',1),('scheduler_advanced',True),('observed_outer',112),
                        ('sampler_cursor_after_observation',58241),('code_commit','b'*40)]:
            altered=copy.deepcopy(ready);altered[key]=bad;ready_path.write_text(json.dumps(altered))
            refused('ready_'+key,lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64))
        for key,bad in [('restored_optimizer_steps',[2501]),('scheduler_step',111),('world_size',7),
                        ('diagnostic_records',[])]:
            altered=copy.deepcopy(ready);altered['workers'][7][key]=bad;ready_path.write_text(json.dumps(altered))
            refused('rank_'+key,lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64))
        ready_path.write_text(json.dumps(ready))
        snapshot=sampling/'rank-07-call-00001.pt';saved=snapshot.read_bytes();snapshot.write_bytes(b'changed')
        refused('raw_snapshot_changed',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64))
        snapshot.write_bytes(saved)
        diagnostic=sampling/'rank-07-call-00001.json';saved=diagnostic.read_bytes()
        altered=json.loads(saved);altered['scheduler_step']=111;diagnostic.write_text(json.dumps(altered))
        refused('raw_diagnostic_scheduler',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64))
        diagnostic.write_bytes(saved)
        monitor=run/'output/probability-calibration/rank-07-monitor.pt';saved_monitor=monitor.read_bytes();monitor.write_bytes(b'changed')
        refused('calibration_monitor_changed',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64));monitor.write_bytes(saved_monitor)
        for name,fields in [('raw_saved_rows_incomplete',{'saved_rows':1}),
                            ('raw_sample_count_differs',{'saved_rows':3,'local_rows':3}),
                            ('raw_distributed_decision_differs',{'decision':{'allowed':True,'reasons':[]}}),
                            ('raw_actor_model_identity',{'model_identity':'b'*64})]:
            altered=json.loads(saved);altered.update(fields);diagnostic.write_text(json.dumps(altered))
            altered_ready=copy.deepcopy(ready);altered_ready['workers'][7]['diagnostic_entries'][0]['sha256']=contract.sha(diagnostic)
            ready_path.write_text(json.dumps(altered_ready))
            refused(name,lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64))
        diagnostic.write_bytes(saved);ready_path.write_text(json.dumps(ready))
        altered_ready=copy.deepcopy(ready);altered_ready['workers'][7]['job_id']='999998';ready_path.write_text(json.dumps(altered_ready))
        refused('calibration_worker_job_identity',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64))
        ready_path.write_text(json.dumps(ready))
        audit=run/'output/tree-rank-audit';(audit/'ratio-tail').mkdir(parents=True)
        ratio=audit/'ratio-tail/optimizer0.json';ratio.write_text('{}')
        refused('ratio_optimizer_file',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64));ratio.unlink()
        (audit/'policy-safety').mkdir();event=audit/'policy-safety/event.json'
        event.write_text(json.dumps(dict(committed_optimizer_calls=1,scheduler_advanced=False)))
        refused('safety_committed_event',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64));event.unlink()
        checkpoints=run/'output/checkpoints';checkpoints.mkdir();unexpected=checkpoints/'checkpoint.pt';unexpected.write_bytes(b'fake')
        refused('unexpected_new_checkpoint',lambda:contract.validate_calibration_ready(ready_path,run,package,'a'*64));unexpected.unlink()
        altered=copy.deepcopy(restoration);altered['sampler_migration']['permutation_sha256']='b'*64
        restoration_path.write_text(json.dumps(altered))
        refused('restore_question_order',lambda:contract.validate_restored110(run,package))
        altered=copy.deepcopy(restoration);altered['workers'][7]['adam_steps']=[2501]
        restoration_path.write_text(json.dumps(altered))
        refused('restore_rank_adam',lambda:contract.validate_restored110(run,package))
        with patch.dict(os.environ,{'TREE_PPO_CALIBRATION_ONLY':'0'}):
            refused('ray_driver_missing_zero_write_flag',observer.install)
            # The validation observer's finally publishes only that actual call.
        # A normal calibration return must not invoke final51713 preparation.
        class Scalar:
            def __init__(self,value):self.value=value
            def item(self):return self.value
        class Tensor:
            shape=(1,1)
            def sum(self):return Scalar(1)
            def __ne__(self,other):return self
        class Worker:
            def generate_sequences(self,batch):return SimpleNamespace(batch={'responses':Tensor()})
        class Manager:
            def __init__(self,worker):self.actor_rollout_wg=worker;self.config=SimpleNamespace(num_gpus=8)
            def run_llm_loop(self,batch,initial):return self._generate_with_gpu_padding(batch)
            def _generate_with_gpu_padding(self,batch):return self.actor_rollout_wg.generate_sequences(batch)
            def _batch_search(self,queries):return {'result':[]}
        class Trainer:
            global_steps=110;config=SimpleNamespace(do_search=True);val_dataset=[1]
            tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=1)
            def __init__(self):self.actor_rollout_wg=Worker()
            def _validate(self):
                Manager(self.actor_rollout_wg).run_llm_loop(SimpleNamespace(batch={'input_ids':Tensor(),'attention_mask':Tensor()}),None)
                return {'val/evaluated_count/nq':1}
            def _prepare_final_validation(self):raise AssertionError('Calibration may not request final51713')
        with patch.dict(os.environ,{'TREE_RECOVERY_PARENT_IDENTITY':json.dumps(parent['identity'])}):
            observer.wrap_trainer(Trainer,Manager,run/'output/validation-cost',final_expected_rows=51713)
            Trainer()._validate()
        reports=list((run/'output/validation-cost').glob('*.json'))
        assert len(reports)==1
        report=json.loads(reports[0].read_bytes())
        assert report['status']=='VALIDATION_COST_COMPLETE' and report['phase']=='restored_initial' and report['evaluated_rows']==1
        checks.append('calibration_validation_return_does_not_require_final51713')
    proof=dict(status='DRAFT_CONTRACT_CPU_SCHEMA_REFUSAL_PASS',checks=checks,checks_count=len(checks),
               contract_sha256=contract.sha(ROOT/'calibration_contract.py'),
               test_script_sha256=contract.sha(__file__),actual_3b_calibration_executed=False,submitted=False,
               scope='Synthetic schema/identity/error-path and cost-observer return tests; no source-hook/GPU/3B acceptance')
    (OUT/'CONTRACT_CPU_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))

if __name__=='__main__':
    main()
