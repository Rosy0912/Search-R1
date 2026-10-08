"""NEW blocked local corrected eval job orchestration; invokes unchanged native tools."""
import argparse,json,os,re,socket,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'strict-tools'))
from training_evidence import OWN,digest
from hf_launcher import stop_own_child

def run(config,stage,stage_sha):
 from eval_supervisor import run as evaluate,validate_config
 from omegaconf import OmegaConf
 from search_r1.baselines import corrected_evaluation as ce
 job=os.environ.get('SLURM_JOB_ID','')
 if not re.fullmatch('[0-9]+',job):raise ValueError('Actual own Slurm job required')
 # All actual training/HF/source/input/config CPU gates precede Ray/model/index work.
 if digest(Path(stage))!=stage_sha or os.environ.get('OPSD_CORRECTED_EVAL_STAGE_SHA256')!=stage_sha:raise ValueError('External stage SHA differs before any runtime startup')
 cfg=OmegaConf.load(config);provenance=ce.load_stage(cfg.baselines.corrected_evaluation);validate_config(cfg,provenance)
 run=OWN/'runs'/('opsd-corrected-eval-'+job);runtime=Path('/tmp')/('oce-'+job)
 if run.exists() or runtime.exists():raise ValueError('New immutable own evaluation run required')
 run.mkdir();runtime.mkdir();(run/'evidence').mkdir();(runtime/'tmp').mkdir();(runtime/'ray').mkdir()
 evidence=run/'evidence';repo=Path(os.environ['SEARCH_R1_REPO']);assets=OWN/'assets-root/assets'
 os.environ.update(OPSD_EVIDENCE=str(evidence),SEARCH_R1_BASELINE_OUTPUT=str(run/'output'),SEARCH_R1_BASELINE_RUN_ROOT=str(runtime),TMPDIR=str(runtime/'tmp'),RAY_TMPDIR=str(runtime/'ray'),HF_HOME=str(OWN/'cache/hf'),XDG_CACHE_HOME=str(runtime/'cache'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',TOKENIZERS_PARALLELISM='false',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',OMP_NUM_THREADS='8',RAY_DEDUP_LOGS='0',RAY_USAGE_STATS_ENABLED='0',VLLM_NO_USAGE_STATS='1')
 for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy'):os.environ.pop(key,None)
 if len(os.fsencode(str(runtime/'ray')+'/session_2026-10-03_05-22-02_615594_9999999999/sockets/plasma_store'))>107:raise ValueError('Native Ray socket path too long')
 for port in (18084,16384):
  with socket.socket() as probe:probe.bind(('127.0.0.1',port))
 # Child environment receives only existing own open file descriptions.
 fds=tuple({9,*[int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])]})
 def child(command,env=None,**kwargs):return subprocess.run(command,cwd=repo,env=dict(os.environ) if env is None else env,check=True,pass_fds=fds,**kwargs)
 (evidence/'slurm-start.txt').write_text(subprocess.check_output(['scontrol','show','job',job],text=True))
 sampler=subprocess.Popen([sys.executable,'-u',str(HERE/'upstream-pilot/resource_sample.py'),str(evidence/'resources.ndjson'),str(os.getpid())],pass_fds=fds)
 retriever=None
 try:
  child(['ray','start','--head','--port=16384','--num-gpus=8','--num-cpus=80','--object-store-memory=8589934592','--include-dashboard=false','--min-worker-port=30000','--max-worker-port=30999','--temp-dir='+str(runtime/'ray')])
  os.environ['RAY_ADDRESS']='127.0.0.1:16384'
  gpu_env=dict(os.environ,PYTHONPATH=str(OWN/'retriever-gpu-lib')+':'+str(repo))
  child([sys.executable,'-u',str(HERE/'upstream-pilot/retrieval_compare.py')],env=gpu_env)
  command=[sys.executable,'-u',str(HERE/'retrieval_audit_entry.py'),'--target',str(repo/'reproduction/h20/retriever_local.py'),'--frame-output',str(evidence/'retriever-frames.json'),'--index_path',str(assets/'wiki/e5_Flat.index'),'--corpus_path',str(assets/'wiki/wiki-18.jsonl'),'--retriever_model',str(assets/'e5-base-v2'),'--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18084','--threads','8']
  with (run/'retriever.log').open('xb') as log:
   retriever=subprocess.Popen(command,cwd=repo,env=gpu_env,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds)
  child([sys.executable,'-u',str(HERE/'upstream-pilot/http_retrieval_check.py')])
  child([sys.executable,'-u',str(HERE/'upstream-pilot/gpu_retrieval_bulk.py')])
  capture=json.loads((evidence/'retriever-frames.json').read_text());capture['listeners']=subprocess.check_output(['ss','-ltnp','sport = :18084'],text=True)
  if f'pid={retriever.pid},' not in capture['listeners']:raise ValueError('Actual local GPU service process/port differs')
  capture.update(http_sha256=digest(evidence/'retrieval-http.json'),bulk_sha256=digest(evidence/'gpu-retrieval-bulk24.json'),comparison_sha256=digest(evidence/'retrieval-comparison.json'))
  (evidence/'retriever-capture.json').write_text(json.dumps(capture,sort_keys=True))
  evaluate(config,stage,stage_sha,run/'evaluation')
 finally:
  if retriever is not None:stop_own_child(retriever)
  stop_own_child(sampler)
 # Slurm final terminal + afterjob PID/port cleanup are collected/reviewed externally,
 # never created by a still-running job or substituted with local "passed" flags.

if __name__=='__main__':
 print('BLOCKED corrected eval job preview: strict actual stage/receipt activation UNFILLED',file=sys.stderr)
 raise SystemExit(78)
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',required=True);parser.add_argument('--stage-sha256',required=True)
 args=parser.parse_args();run(args.config,args.stage,args.stage_sha256)
