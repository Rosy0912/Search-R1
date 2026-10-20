"""Only transport/runtime orchestration; original native restoration and training unchanged."""
from pathlib import Path
import json,os,socket,subprocess,sys,threading,time
from .fresh_contract import BASE,SCIENCE,verify,project_config,sha
BIND='RECURSIVE_ABLATION_7B_LUMIA_RESUME_BINDING'
def worker_setup():
 assert not os.environ.get('RECURSIVE_ABLATION_RESUME_MANIFEST')
 from release.recursive_bran_a.ablation150_v1 import bootstrap
 bootstrap.install()
def train(planfile):
 from omegaconf import OmegaConf
 from release.recursive_bran_a.ablation150_v1 import PROTOCOL,EVALUATION_STEPS
 p=json.loads(Path(planfile).read_text());root=verify(p);host=socket.gethostname().split('.')[0]
 old=OmegaConf.to_container(OmegaConf.load(BASE/'parent-small'/(p['group']+'-resolved.yaml')),resolve=True)
 cfg=project_config(old,p['group'],host);path=root/'receipts/resolved.yaml';path.open('x').write(OmegaConf.to_yaml(OmegaConf.create(cfg),resolve=True))
 output=root/'output'/('attempt-'+p['entry_run_id']);output.mkdir(parents=True,exist_ok=False)
 record=dict(protocol=PROTOCOL,group=p['group'],run_id=p['run_id'],config=str(path),config_sha256=sha(path),source_commit=SCIENCE,
  model=p['model'],data=p['data_root'],data_identity=p['data_identity'],base=str(root),output=str(output),fresh_authorized=True,
  parent_checkpoint=None,parent_manifest_sha256=None,entry_source_commit=p['entry_source_commit'],
  execution_backend='slurm_lumia',allocation=os.environ['SLURM_JOB_ID'],scheduler_node=os.environ['SLURM_JOB_NODELIST'],host=host,condition=p['condition'],actual_update_limit=150,
  evaluation_steps=list(EVALUATION_STEPS),scheduler_scale=200,warmup_rounds=57,model_size='7B',model_revision=p['model_revision'],retriever_receipt=p['retriever_receipt'],retriever_receipt_sha256=p['retriever_receipt_sha256'])
 receipt=root/'receipts/configuration.json';receipt.open('x').write(json.dumps(record,indent=2))
 from reproduction.recursive_lumia.ablation_entry import runtime_environment
 env=runtime_environment(record,output,receipt);env.update(SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-7B-Instruct@'+p['model_revision'])
 assert not any(k.startswith(('RECURSIVE_ABLATION_RESUME','TREE_RECOVERY_','TREE_CURRENT_CHAIN_RESUME_','RECURSIVE_A_CONTINUATION_')) for k in os.environ)
 os.environ.update(env);worker_setup()
 from search_r1.baselines.preflight import validate_gpu_lease
 config=OmegaConf.create(cfg);validate_gpu_lease(config)
 entry=dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),tick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),host=host,allocation=os.environ['SLURM_JOB_ID'],run_id=p['run_id'],group=p['group'],output=str(output),config_sha256=record['config_sha256'],source_commit=SCIENCE,execution_backend='slurm_lumia',parent_uid=p['parent_uid'],parent_manifest_sha256=p['parent_manifest_sha256'])
 (root/'receipts/current-entry.json').open('x').write(json.dumps(entry,indent=2))
 from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
 prepare_click();import ray
 inherited={k:v for k,v in os.environ.items() if k.startswith(('SEARCH_R1_','RECURSIVE_A_','RECURSIVE_ABLATION_','RECURSIVE_LUMIA_','TREE_','SLURM_'))}
 for k in ('CPATH','C_INCLUDE_PATH','TRITON_LIBCUDA_PATH','LIBRARY_PATH','TRITON_PTXAS_PATH'): 
  if k in os.environ:inherited[k]=os.environ[k]
 inherited.update(PYTHONPATH=p['source'],TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN')
 error=None
 try:
  context=ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,object_store_memory=8*1024**3,_node_ip_address='127.0.0.1',_temp_dir='/tmp/abl-'+os.environ['SLURM_JOB_ID'],runtime_env={'env_vars':inherited,'worker_process_setup_hook':worker_setup})
  os.environ['RAY_ADDRESS']=context.address_info['gcs_address']
  from search_r1.baselines.main import main
  main.__wrapped__(config)
 except BaseException as e:error=repr(e);raise
 finally:
  ray.shutdown();(output/'entry-exit.json').open('x').write(json.dumps(dict(at=time.time(),rc=0 if error is None else 1,error=error)))
def launch(planfile):
 p=json.loads(Path(planfile).read_text());root=verify(p);jid=os.environ['SLURM_JOB_ID']
 sub=json.loads((root/'submission.json').read_text());assert sub['job_id']==jid and sub['job']==p['job'] and sub['plan_sha256']==sha(planfile)
 row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
 assert 'JobName='+p['job']+' ' in row and 'UserId=jmzhang(1066)' in row and 'gres/gpu=8' in row and 'mem=900G' in row
 assert os.environ['SLURM_JOB_NODELIST'] in ('node8','node9','node10','node11')
 inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
 assert len(inv)==8 and len({r.split(',')[1].strip() for r in inv})==8 and all(float(r.split(',')[-1])>=44000 for r in inv)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 assert not (root/'checkpoints').exists() and not (root/'output').exists()
 (root/'entry-once.json').open('x').write(json.dumps(dict(at=time.time(),job_id=jid,gpus=inv,plan_sha256=sha(planfile)),indent=2))
 os.environ['CUDA_VISIBLE_DEVICES']=','.join(r.split(',')[0].strip() for r in inv)
 compiler='/home/jmzhang/miniconda3/envs/searchr1/lib/python3.9/site-packages/triton/backends/nvidia/bin/ptxas'
 os.environ['TRITON_PTXAS_PATH']=compiler
 smoke="from reproduction.sevenb import ablation_smoke as s;s.COMPILER=dict(path="+repr(compiler)+",sha256='94dda54a1673940d56d3072c64ef7aacb125a1e64394586e10f77a7a5ff0ff87');s.run("+repr(str(root/'compiler-smoke.json'))+")"
 subprocess.run([sys.executable,'-c',smoke],check=True,timeout=180)
 from http.server import ThreadingHTTPServer
 from reproduction.recursive_lumia.proxy import Handler
 ready=json.loads(Path(p['retriever_receipt']).read_text());server=ThreadingHTTPServer(('127.0.0.1',29146),Handler);server.remote=ready['url'];threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  rc=subprocess.call([sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,'-m','reproduction.sevenb_ablation_lumia_resume_v1.fresh_entry','train',str(planfile)])
  (root/'actual-parent-wait.json').open('x').write(json.dumps(dict(at=time.time(),rc=rc,job_id=jid)));raise SystemExit(rc)
 finally:server.shutdown();server.server_close()
if __name__=='__main__':(train if sys.argv[1]=='train' else launch)(sys.argv[2])
