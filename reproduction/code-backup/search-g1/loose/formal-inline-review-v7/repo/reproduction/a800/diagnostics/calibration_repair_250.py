import os,sys,json,time,fcntl,hashlib,subprocess,uuid,traceback
from pathlib import Path
base=Path("/data/search-g1-calibration-repair-250");repo=base/"repo"
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import OwnedProcesses,identity,publish
import requests
out=base/"artifacts"/("round250-"+uuid.uuid4().hex);out.mkdir(parents=True)
publish(base/"latest-job.json",dict(directory=str(out),leader=identity(os.getpid()),purpose="actual300/G5 calibration only; no trainer",source_commit="ddab367"))
lock=open("/data/search-g1/gpu-job.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
owned=OwnedProcesses(out/"owned-ledger.json",out.name)
model=Path("/data/search-g1-continuation-200/checkpoints/search-g1-a800/actor/global_step_250")
data=Path("/data/search-g1-continuation-200/data/calibration-300.parquet")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""):h.update(x)
 return h.hexdigest()
model_sha={str(p.relative_to(model)):sha(p) for p in sorted(model.rglob("*")) if p.is_file()}
source_sha={str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and p.suffix in (".py",".sh",".yaml") and "__pycache__" not in p.parts}
publish(out/"input-binding.json",dict(model=str(model),model_sha256=model_sha,data_sha256=sha(data),source_sha256=source_sha,source_commit="ddab367",generation_randomness="historical sampling RNG defaults; no label-selected seed",question_count=300,G=5))
env=dict(os.environ,SEARCH_G1_BASE=str(base),PYTHONPATH="/data/search-r1/retriever-gpu-lib:"+str(repo),
 HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="8",NCCL_NVLS_ENABLE="0",
 VLLM_ATTENTION_BACKEND="XFORMERS",RETRIEVER_URL="http://127.0.0.1:18010/retrieve",NO_PROXY="localhost,127.0.0.1",
 C_INCLUDE_PATH="/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include")
py="/data/search-g1/env/bin/python"
stage="PREPARING";error=None
def status(s,**kw):
 global stage
 stage=s;publish(out/"status.json",dict(stage=s,unix=time.time(),**kw))
def wait(p,name):
 while p.poll() is None:
  owned.observe(name);owned.observe("retriever")
  sample=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
  with (out/"gpu-samples.jsonl").open("a") as f:f.write(json.dumps(dict(unix=time.time(),csv=sample.stdout))+"\n")
  time.sleep(2)
 assert p.returncode==0,name+" exit "+str(p.returncode)
try:
 before=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
 for row in before.stdout.strip().splitlines():
  idx,mem=[int(x.strip()) for x in row.split(",")]
  if idx in (0,1,2,3,5,6):assert mem<1024,"Owned allowed resource not idle"
 r=owned.start("retriever",[py,"reproduction/a800/retriever_local.py","--index_path","/data/search-r1/assets/wiki/e5_Flat.index","--corpus_path","/data/search-r1/assets/wiki/wiki-18.jsonl","--topk","3","--retriever_name","e5","--retriever_model","/data/search-r1/assets/e5-base-v2","--faiss_gpu"],dict(env,CUDA_VISIBLE_DEVICES="5,6"),out/"retriever.log",repo)
 status("RETRIEVER_LOADING")
 session=requests.Session();session.trust_env=False;started=time.time()
 while True:
  owned.observe("retriever");assert r.poll() is None
  try:
   z=session.post(env["RETRIEVER_URL"],json={"queries":["capital of France"],"topk":3},timeout=10);z.raise_for_status();assert len(z.json()["result"][0])==3;break
  except requests.RequestException:
   assert time.time()-started<1500;time.sleep(2)
 status("GENUINE_300_G5_INTERVENTIONS")
 dumps=out/"dumps"
 p=owned.start("dump",[py,"ts_probe_dump.py","--model",str(model),"--data",str(data),"--tag","nq-hotpot-common","--cap","300","--G","5","--layers","21,24,27","--max_turns","4","--rollout_max_new","500","--correction_feedback","--gpu_mem","0.25","--out",str(dumps)],dict(env,CUDA_VISIBLE_DEVICES="6"),out/"dump.log",repo)
 wait(p,"dump");owned.stop("dump")
 import numpy as np,joblib
 proof=json.loads((dumps/"calibration-label-proof.json").read_text())
 audit=json.loads(Path("/data/search-g1-continuation-200/data/audit.json").read_text())
 assert proof["question_count"]==300 and proof["group_size"]==5
 assert len(proof["question_ids"])==300 and set(proof["question_ids"])==set(audit["calibration_question_ids"])
 for zf in dumps.glob("*.npz"):
  with np.load(zf,allow_pickle=False) as z:
   for key in z.files:
    if z[key].dtype.kind in "fc":assert np.isfinite(z[key]).all(),key
 status("STRICT_TWO_FOLD_FIT",searched_valid=proof["searched_valid_trajectory_count"],reliance_counts=proof["reliance_class_counts"])
 heads=out/"heads"
 p=owned.start("fit",[py,"ts_probe_train.py","--dumps",str(dumps),"--layers","21,24,27","--commit-position","Xa","--commit-label","flip","--require-all-cv-folds","--save",str(heads)],dict(env,CUDA_VISIBLE_DEVICES=""),out/"fit.log",repo)
 wait(p,"fit");owned.stop("fit")
 report=json.loads((heads/"calibration-report.json").read_text())
 assert report["require_all_cv_folds"] and report["configured_cv_folds"]==2
 for name in ("commit_probe.pkl","boundary_probe.pkl"):
  obj=joblib.load(heads/name);assert list(obj["clf"].classes_)==[0,1] and np.isfinite(obj["auc"])
  for value in vars(obj["clf"]).values():
   if isinstance(value,np.ndarray) and value.dtype.kind in "fc":assert np.isfinite(value).all()
 assert model_sha=={str(p.relative_to(model)):sha(p) for p in sorted(model.rglob("*")) if p.is_file()}
 publish(out/"calibration-accepted.json",dict(accepted=True,report=report,source_commit="ddab367",model_sha256=model_sha,
  dump_sha256={p.name:sha(p) for p in dumps.iterdir() if p.is_file()},head_sha256={p.name:sha(p) for p in heads.iterdir() if p.is_file()},finished_unix=time.time(),semantics="real300/G5 new correction500 calibration; no GRPO and not HTTP admission"))
 status("CALIBRATION_FITTED_NOT_TRAINING")
except BaseException as e:
 error=repr(e);status("FAILED",error=error);traceback.print_exc()
finally:
 cleanup={}
 for name in reversed(list(owned.jobs)):
  try:owned.stop(name);cleanup[name]=owned.observe(name)
  except BaseException as e:cleanup[name]={"error":repr(e)}
 publish(out/"cleanup.json",dict(components=cleanup,unix=time.time()))
 if error:sys.exit(1)
