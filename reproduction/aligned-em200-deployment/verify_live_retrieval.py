"""Validate this hardware run's local service against independently captured golden."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib, json, os, subprocess, sys
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
run=Path(os.environ['BAPO_RUN_ID']);job=os.environ['SLURM_JOB_ID']
require(run.parent==ROOT/'runs' and run.resolve()==run and run.name.startswith('hardware-aligned-em200-') and run.name.endswith('-'+job),'same new hardware run required')
require(sys.prefix==str(ROOT/'env') and os.uname().nodename=='h100-5','new local environment required')
require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'source differs')
require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'dirty fixed source')
sys.path.insert(0,str(ROOT/'repo'))
from search_r1.baselines import preflight, baseline_alignment_retrieval
require(Path(preflight.__file__).resolve()==ROOT/'repo/search_r1/baselines/preflight.py','foreign preflight import')
require(Path(baseline_alignment_retrieval.__file__).resolve()==ROOT/'repo/search_r1/baselines/baseline_alignment_retrieval.py','foreign validator import')
leases=preflight.validate_inherited_gpu_leases(os.environ['CUDA_VISIBLE_DEVICES'].split(','),require_guard=True)
manifest=run/'retrieval-golden.json';require(manifest.is_file() and manifest.resolve()==manifest,'canonical actual golden required')
golden=json.loads(manifest.read_text());claim_path=run/'GPU-GOLDEN-CAPTURE-CLAIM.json';claim=json.loads(claim_path.read_text())
completed=json.loads((run/'capture-completed-before-live.json').read_text())
require(completed['source']==SOURCE and completed['run']==str(run) and completed['job']==job and completed['scope']=='ARCHIVE_CAPTURE_COMPLETED_BEFORE_LIVE_SERVICE_START','capture completion identity differs')
require(completed['manifest_sha256']==sha(manifest) and completed['capture_claim_sha256']==sha(claim_path) and completed['capture_log_sha256']==sha(run/'retrieval-golden.log'),'archive capture bytes changed after completion')
require(completed['collector_sha256']==golden['provenance']['collector_sha256']==sha(Path(__file__).parent/'capture_retrieval_golden.py'),'capture collector identity differs')
require(claim['scope']=='INDEPENDENT_ARCHIVE_GPU_GOLDEN_CAPTURE' and golden['provenance']['scope']=='ACTUAL_ARCHIVE_GPU_GOLDEN_NOT_TRAINING_OR_LIVE_SERVICE_ACCEPTANCE','independent archive scope differs')
require(len(leases)==8 and sorted(v['uuid']for v in leases)==sorted(v['uuid']for v in golden['provenance']['leases']),'capture and live GPU UUID leases differ')
archive=json.loads((Path(__file__).parent/'reference/retrieval-provenance.json').read_text())
require(golden['provenance']['archive']==archive and archive['sha256']=='e85ab42b9032d66a35239a36e8cd73c6f3e3efde929541150fe7c5bfdb4936d9','archive provenance differs')
require(golden['provenance']['capture_claim']==claim and claim['run']==str(run) and claim['source']==SOURCE and claim['job']==job,'actual independent capture identity differs')
require(golden['provenance']['query_batch_size']==1 and len(golden['probes'])==12,'golden probe scope differs')
import requests
transport=requests.Session();transport.trust_env=False
response=transport.get('http://127.0.0.1:18083/identity',timeout=(5,30));response.raise_for_status();identity=response.json()
import bind_retrieval_proof as binding
require(Path(binding.__file__).resolve()==Path(__file__).parent/'bind_retrieval_proof.py','foreign birth binding helper')
birth=binding.birth
proof=json.loads((run/'own-retriever-birth.json').read_text())
require(proof['source']==SOURCE and proof['run']==str(run) and proof['job']==job and proof['scope']=='OWN_GPU_RETRIEVER_BIRTH_BEFORE_READY','server birth provenance differs')
pid=int(os.environ['BAPO_LOCAL_RETRIEVER_PID']);require(proof['identity']==birth(pid,run,job),'server birth identity changed')
require(identity['pid']==pid,'loopback service is not the launched own retriever')
markers=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
require(('BAPO_RUN_ID='+str(run)).encode() in markers and ('SLURM_JOB_ID='+job).encode() in markers,'live retriever ownership differs')
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725'),local_files_only=True)
config=NS(retriever=NS(url='http://127.0.0.1:18083/retrieve',identity_manifest=str(manifest),identity_manifest_sha256=completed['manifest_sha256'],topk=3),data=NS(max_obs_length=500))
result=baseline_alignment_retrieval.validate_retriever_identity(config,tokenizer,transport=transport)
require(proof['identity']==birth(pid,run,job),'server birth changed during live validator')
require(completed['manifest_sha256']==sha(manifest),'golden changed during live validation')
require(result['manifest_sha256']==completed['manifest_sha256'],'consumer used a different manifest')
require(result['live_pid']==pid and result['verified_probes']==12 and result['gpu_identity_verified'],'live acceptance scope differs')
output=dict(source=SOURCE,run=str(run),job=job,host=os.uname().nodename,leases=leases,manifest_sha256=sha(manifest),identity=identity,result=result,scope='ACTUAL_LOCAL_CUDA_E5_GPU_FAISS_ARCHIVE_GOLDEN_PARITY_NOT_TRAINING_ACCEPTANCE')
with (run/'live-retrieval-acceptance.json').open('x') as f:json.dump(output,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(manifest_sha256=sha(manifest),verified_probes=12,scope=output['scope'])),flush=True)
