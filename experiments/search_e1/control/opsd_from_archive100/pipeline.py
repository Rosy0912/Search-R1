from pathlib import Path
import os,json,time,subprocess,hashlib,uuid,fcntl
b=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007');old=Path('/data/shaoshitong/search-e1-config-study-r1-aligned-8gpu-20261005')
lock=(b/'experiment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def phase(s):
 (b/'receipts/pipeline-stage.json').write_text(json.dumps(dict(phase=s,time=time.time(),controller_pid=os.getpid()),indent=2));print('PHASE',s,flush=True)
def run(args,log):
 with (b/'logs'/log).open('a') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
env=dict(os.environ,E1_BASE=str(b),E1_PROFILE='opsd100',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=str(b/'runtime/opsd/upstream/Search-R1')+':'+str(b/'runtime/opsd'),HF_HOME=str(b/'hf-cache'),HF_DATASETS_CACHE=str(b/'hf-datasets'),XDG_CACHE_HOME=str(b/'cache'),TMPDIR=str(b/'tmp'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',E1_INFERENCE_UTIL='.85',WANDB_MODE='offline',E1_RUN_ID=uuid.uuid4().hex)
(b/'receipts/new-run.json').write_text(json.dumps(dict(run_id=env['E1_RUN_ID'],pid=os.getpid(),model=str(b/'assets/archive100'),stage='OPSD_FROM_ARCHIVE100',trajectory_batch=32,world=8,accumulation=4,epochs=1,lr=1e-5,rank=16,alpha=32,clip=10,chunk=32,seed=42,reset_grpo_optimizer=True,additional_grpo=False),indent=2))
py=str(b/'env/bin/python')
try:
 phase('WAITING_COMPLETE_OLD_STATE_AND_ARCHIVE100_TRANSFER')
 full=old/'preserved/replacement-state-b8b93546fb314915b3e72f355f22f291/full-complete.json'
 while not full.exists() or not (b/'receipts/archive100-transfer-accepted.json').exists():time.sleep(10)
 phase('STOPPING_ONLY_PRESERVED_OLD_ALIGNED8')
 run([py,str(b/'source/stop_preserved_aligned.py')],'stop-old.log')
 phase('WAITING_OLD_OWNED_GPU_RELEASE')
 stop=json.loads((b/'receipts/aligned-user-replacement-stop.json').read_text());owned=set(stop['owned_pids'])
 for attempt in range(120):
  lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines();gpu_pids={int(x.strip()) for x in lines if x.strip().isdigit()}
  if not gpu_pids & owned:break
  time.sleep(5)
 else:raise RuntimeError('Old owned GPU workers not released; no foreign process will be stopped')
 assert not gpu_pids,'Unexpected GPU processes on HKUST1; do not stop them'
 phase('VERIFYING_NEW_FROZEN_RUNTIME')
 m=json.loads((b/'receipts/opsd-runtime.sha256.json').read_text());root=b/'runtime/opsd'
 assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in m.items())
 phase('OPSD100_FULL_FRESH_TRAJECTORIES')
 run([py,'-m','search_e1.rollout','--model',str(b/'assets/archive100'),'--data',str(b/'data/train.jsonl'),'--output',str(b/'runs/rollouts.jsonl'),'--k','5','--backend','vllm','--workers','8','--retriever-url','http://10.120.17.101:18001/retrieve'],'rollouts.log')
 phase('OPSD100_MINE_FULL_PAIRS')
 run([py,'-m','search_e1.mine','--rollouts',str(b/'runs/rollouts.jsonl'),'--output',str(b/'runs/pairs.jsonl')],'mine.log')
 phase('OPSD100_TRAIN_AND_BASELINE_VALIDATION')
 run([py,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','search_e1.opsd','--model',str(b/'assets/archive100'),'--pairs',str(b/'runs/pairs.jsonl'),'--validation',str(b/'data/validation.jsonl'),'--accumulation','4','--output',str(b/'runs/opsd'),'--retriever-url','http://10.120.17.101:18001/retrieve'],'opsd.log')
 phase('OPSD100_FINAL_SEVEN_BENCHMARK_EVALUATION')
 run([py,'-m','search_e1.rollout','--model',str(b/'runs/opsd/merged'),'--data',str(b/'data/test.jsonl'),'--output',str(b/'runs/final-test.jsonl'),'--k','1','--backend','vllm','--workers','8','--temperature','0','--retriever-url','http://10.120.17.101:18001/retrieve'],'final-test.log')
 phase('COMPLETED_OPSD_AND_EVALUATION')
except Exception as e:
 phase('FAILED_'+repr(e));raise
