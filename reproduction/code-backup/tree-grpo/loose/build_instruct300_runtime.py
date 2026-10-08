from pathlib import Path
old=Path('work/tree-fixed-formal-296217a/supervisor.py').read_text();target=Path('work/tree-instruct300-runtime')
header='''"""Owned approved Instruct300, one allocation and live step5 acceptance gate."""
import fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from tree_shared_owned_runtime_fixed import ROOT,require,sha,cgroup_scope,cgroup_pids,OwnedProcesses,process_sample,gpu_sample,available
TOOLS=Path(__file__).resolve().parent
PACKAGE=json.loads((TOOLS/'manifest.json').read_text())
COMMIT=PACKAGE['code_commit']
CAND=TOOLS/'source'
CONTENT='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
class CandidateOwnedProcesses(OwnedProcesses):
    def spawn(self,command,log,env=None,pass_fds=()):
        return super().spawn(['bash','-c','cd "$SEARCH_R1_REPO" && exec "$@"','candidate-cwd',*command],log,env,pass_fds)

def interrupted(sig,frame): raise RuntimeError('Own Instruct300 supervisor interrupted '+str(sig))

def configuration(run,stage):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import prepare_common_holdout
    with initialize_config_dir(config_dir=str(CAND/'repo/verl/trainer/config'),version_base=None):
        value=compose(config_name='ppo_tree_instruct300')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'input-config.yaml')
    split=prepare_common_holdout(value)
    require(split['content_identity']['sha256']==CONTENT and split['train_rows']==169028 and split['holdout_rows']==512,'Fixed split changed')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'resolved.yaml')
    require(value.trainer.total_training_steps==301 and value.trainer.actual_update_limit==300 and value.actor_rollout_ref.actor.optim.lr_warmup_steps==57,'300/57 budget changed')
    require(value.data.train_batch_size==512 and value.actor_rollout_ref.actor.ppo_mini_batch_size==128 and value.actor_rollout_ref.actor.ppo_micro_batch_size==16,'Approved batch changed')
    require(value.actor_rollout_ref.rollout.log_prob_micro_batch_size==32 and value.actor_rollout_ref.ref.log_prob_micro_batch_size==32 and value.data.val_batch_size==32,'Scoring/development batch changed')
    require(value.baselines.tree.roots==2 and value.baselines.tree.rounds==1 and value.baselines.tree.expansions==2 and value.baselines.tree.leaves==3,'Tree changed')
    require(value.max_turns==4 and value.force_final_answer is False and value.retriever.topk==3,'Search changed')
    return value,300

def main():
    a=SimpleNamespace(stage='instruct300')
    for file,expected in PACKAGE['files'].items(): require(sha(TOOLS/file)==expected,'Frozen package changed: '+file)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'repo',text=True).strip()=='4d133efeb1f675139d2558640f1e21d50780f133','Old production source changed')
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit() and socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('SLURM_JOB_NODELIST')=='h100-9','Wrong allocation')
    require(len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8 and os.environ.get('SLURM_JOB_GPUS'),'Eight reserved GPUs required')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    from search_r1.baselines.select_teacher import model_manifest
    require(model_manifest(MODEL)==PACKAGE['model_identity'],'Official model changed')
'''
body=old[old.index("    lock=(ROOT/'shared-gpu-job.lock')"):]
body=body.replace("f'tree-fixed-{a.stage}-{job}'","f'tree-instruct300-{job}'").replace("f'sst-tree-fixed-{job}'","f'sst-tree-instruct300-{job}'")
body=body.replace("    os.environ['TREE_VALIDATION_TRAINER_SHA256']='b3c9d740a3a4d7963eb48922551f0e6ff643325b58bea0c408392af478c01bf5'", "    os.environ.update(TREE_VALIDATION_TRAINER_SHA256=PACKAGE['files']['source/repo/verl/trainer/ppo/ray_trainer.py'],TREE_INSTRUCT_TOOLS=str(TOOLS),TREE_INSTRUCT_MODEL=str(MODEL),TREE_COMMON_DATA=str(ROOT/'instruct300/common-data'),SEARCH_R1_MODEL_IDENTITY=PACKAGE['model_identity']['sha256'])")
body=body.replace("str(TOOLS/'run_tree_with_validation_cost_fixed_short.sh')","str(TOOLS/'run_tree.sh')")
needle="        result['gpu_kl_proof_passed']=True\n"
body=body.replace(needle,needle+'''        os.environ['TREE_NATIVE_STATE_OUTPUT']=str(run/'native-fullstate')
        native=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=26382',str(TOOLS/'native_fullstate_probe.py')],run/'native-fullstate.log')
        deadline=time.monotonic()+600
        while native.poll() is None:
            require(guard.poll() is None and time.monotonic()<deadline,'Native state guard failed or timeout');time.sleep(1)
        require(native.returncode==0,'Native fullstate replay failed before any3B training')
        for rank in range(8):
            record=json.loads((run/f'native-fullstate/rank-{rank:02d}.json').read_text())
            require(record['status']=='PASS' and record['same_world_sixth_outer_replay_equal'] is True,'Native state rank coverage failed')
        result['native_fullstate_replay_passed']=True
''')
body=body.replace("            with (run/'process-resources.jsonl').open('a') as stream:","            hf5=None\n            with (run/'process-resources.jsonl').open('a') as stream:")
needle="                while training.poll() is None:\n"
body=body.replace(needle,needle+'''                    gate=run/'output/acceptance-step5'
                    if hf5 is None and (gate/'ready.json').is_file():
                        ready=json.loads((gate/'ready.json').read_text())
                        hf5=owned.spawn([sys.executable,str(ROOT/'tools/check_real_checkpoint.py'),'--base',str(MODEL),'--checkpoint',ready['hf_checkpoint'],'--output',str(gate/'hf5-full-reload.json')],gate/'hf5-full-reload.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                    if hf5 is not None and hf5.poll() is not None:
                        require(hf5.returncode==0,'Actual HF5 reload failed while paused; no6 permitted')
''')
body=body.replace("    if a.stage!='gate': (TOOLS/f'{a.stage}-stage-completed.run').write_text(str(run)+'\\n')", "    print('INSTRUCT300_COMPLETED_RUN='+str(run),flush=True)")
(target/'supervisor.py').write_text(header+body)
