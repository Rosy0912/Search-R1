"""Node11 cohost launch bindings; actual node11 proof, never node5 hardware acceptance."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,subprocess,sys,time
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v5-20261009')
OLDROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47'
DEPLOY=ROOT/'tools/deployment/cohost-node11-v5'
MODEL=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725')
COMMON=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data')
RETRIEVAL=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets')
COMMON_SHA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
def require(v,m):
    if not v:raise ValueError(m)
def raw_file(path):
    p=Path(path);require(p.is_file()and not p.is_symlink()and p.resolve()==p,'canonical file required: '+str(p));require(p.stat().st_size<=64*1024*1024,'oversized small binding');return p.read_bytes()
def sha(path):return hashlib.sha256(raw_file(path)).hexdigest()
def bindings():
    b=json.loads(raw_file(ROOT/'inputs/node11-bindings.json'))
    require(b.get('source')==SOURCE and b.get('root')==str(ROOT)and b.get('host')=='h100-11' and b.get('scope')=='SHARED_ASSET_PATH_BINDINGS_NOT_GPU_ACCEPTANCE','node11 shared bindings differ')
    require(b.get('model')==str(MODEL)and b.get('common')==str(COMMON)and b.get('retrieval_assets')==str(RETRIEVAL),'fixed shared assets differ')
    for name in ('data','order','retrieval'):
        p=ROOT/'inputs'/('baseline-aligned-'+name+'.json');require(b[name]['path']==str(p)and sha(p)==b[name]['sha256'],'node11 input pin changed: '+name)
    return b

def validate_admission():
    require(sys.platform=='linux'and os.uname().nodename=='h100-11'and sys.prefix==str(ROOT/'env'),'node11 lexical isolated env required')
    require(ROOT.resolve()==ROOT,'canonical own root required')
    for n in ('repo','env','runs','logs','scheduler-head','inputs','tools','tools/deployment/cohost-node11-v5'):
        require((ROOT/n).is_dir()and(ROOT/n).resolve()==ROOT/n,'canonical own directory required: '+n)
    require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'source080 required')
    require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'dirty production source')
    b=bindings();require(MODEL.is_dir()and COMMON.is_dir()and RETRIEVAL.is_dir(),'shared assets absent')
    require(sha(COMMON/'manifest.json')==COMMON_SHA,'common169028 identity changed')
    names={'formal.sbatch','formal_worker.sh','formal_inner.sh','formal_driver.py','formal_service_proof.py','bind_retrieval_proof.py','ray_cli_compat.py','bootstrap.py','formal_admission.py','submit_formal.py'}
    require(set(b['tools'])==names|{'tools/resource_watch.py'},'exact node11 launch tool inventory required')
    for n,h in b['tools'].items():require(sha(ROOT/n if n=='tools/resource_watch.py'else DEPLOY/n)==h,'node11 tool changed: '+n)
    # Unknown queue outcomes fail closed; never duplicate original3350 or another formal.
    queue=subprocess.check_output(['squeue','-h','-o','%i|%k|%T'],text=True)
    current=os.environ.get('SLURM_JOB_ID')
    for line in queue.splitlines():
        if 'BAPO_ALIGNED_EM200:formal:'in line:require(line.split('|',1)[0]==current,'another formal allocation exists; no duplicate')
    return b

def verify_submission(run,timeout=60):
    p=ROOT/'scheduler-head/formal-submit-receipt.json';deadline=time.monotonic()+timeout
    while not p.exists():
        require(time.monotonic()<deadline,'actual submit receipt unavailable');time.sleep(.2)
    d=json.loads(raw_file(p));job=os.environ['SLURM_JOB_ID']
    require(d.get('source')==SOURCE and d.get('root')==str(ROOT)and d.get('job')==job and d.get('host')=='h100-11'and d.get('run')==str(run),'wrong actual node11 submission')
    require(run.name=='formal-aligned-em200-'+d['token']+'-'+job and d.get('scope')=='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION','submission token/scope differs')
    require(d.get('bindings_sha256')==sha(ROOT/'inputs/node11-bindings.json'),'submit bindings changed')

def claim(run):
    validate_admission();run=Path(run);require(run.parent==ROOT/'runs'and run.resolve()==run,'canonical own new RUN');verify_submission(run)
    sys.path.insert(0,str(ROOT/'repo'))
    from search_r1.baselines import aligned_startup
    require(Path(aligned_startup.__file__).resolve()==ROOT/'repo/search_r1/baselines/aligned_startup.py','foreign claim helper')
    aligned_startup.claim_new_run(run,SOURCE)
    with(run/'source-commit.txt').open('x')as f:f.write(SOURCE+'\n');f.flush();os.fsync(f.fileno())

def environment(run):
    b=validate_admission();run=Path(run);verify_submission(run,timeout=0)
    require(json.loads(raw_file(run/'.aligned-startup-claim.json'))['source_commit']==SOURCE,'fresh claim required')
    return dict(VLLM_ATTENTION_BACKEND='XFORMERS',CUDA_LAUNCH_BLOCKING='1',BAPO_FRESH200_MODEL=str(MODEL),BAPO_FRESH200_DATA=str(COMMON),BAPO_FRESH200_DATA_MANIFEST_SHA256=COMMON_SHA,BAPO_FRESH200_RUN=str(run),BAPO_ALIGNED_DATA_MANIFEST=b['data']['path'],BAPO_ALIGNED_DATA_SHA256=b['data']['sha256'],BAPO_ALIGNED_ORDER_MANIFEST=b['order']['path'],BAPO_ALIGNED_ORDER_SHA256=b['order']['sha256'],BAPO_ALIGNED_RETRIEVAL_MANIFEST=b['retrieval']['path'],BAPO_ALIGNED_RETRIEVAL_SHA256=b['retrieval']['sha256'],BAPO_FRESH200_RETRIEVER_URL='http://127.0.0.1:18083/retrieve')
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','claim','environment']);p.add_argument('--run');a=p.parse_args()
    if a.action=='verify':validate_admission();print('NODE11_ASSET_TOOL_ADMISSION_NOT_GPU_ACCEPTANCE')
    elif a.action=='claim':claim(a.run)
    else:
        for k,v in environment(a.run).items():print('export '+k+'='+shlex.quote(v))
if __name__=='__main__':main()
