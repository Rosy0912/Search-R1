#!/usr/bin/env python3
"""Native CPU evidence only; never load a checkpoint or initialize CUDA."""
import argparse,hashlib,json,os,pathlib,re,socket,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha256',required=True);a=ap.parse_args()
    manifest=HERE/'files.json'
    if sha(manifest)!=a.manifest_sha256:raise ValueError('frozen manifest bytes changed')
    if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11' or not HERE.is_relative_to(ROOT):raise ValueError('own native CPU Slurm location required')
    inventory=json.loads(manifest.read_text())
    for name,record in inventory['files'].items():
        p=HERE/name
        if p.is_symlink() or p.stat().st_size!=record['size'] or sha(p)!=record['sha256']:raise ValueError('frozen source mismatch '+name)
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU parent has CUDA context')
    directory=ROOT/'logs'/('g1-staged-calibration-cpu-'+os.environ['SLURM_JOB_ID']);directory.mkdir(exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    began=time.monotonic();log=directory/'pytest.log'
    with open(log,'wb') as f:result=subprocess.run([sys.executable,'-m','pytest','-q','--basetemp',str(directory/'pytest-tmp'),'-o','cache_dir='+str(directory/'pytest-cache'),str(HERE/'test_producer.py'),str(HERE/'test_phase_control.py'),str(HERE/'test_stage_cleanup.py'),str(HERE/'test_stage_baseline.py'),str(HERE/'test_first_fault.py')],cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
    text=log.read_text();match=re.search(r'(\d+) passed',text);passed=int(match.group(1)) if match else 0
    receipt={'scope':'CPU_ORIGINAL_DUMP_STAGED_SCHEMA_ONLY_NO_GPU_NO_TRAINING','status':'PASS' if result.returncode==0 and passed==76 else 'FAIL','returncode':result.returncode,'passed':passed,'expected':76,'elapsed_seconds':time.monotonic()-began,'manifest_sha256':a.manifest_sha256,'log_sha256':sha(log),'host':socket.gethostname(),'job':os.environ['SLURM_JOB_ID'],'pid':os.getpid(),'ppid':os.getppid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'python':sys.version,'torch':torch.__version__,'parent_cuda_initialized':torch.cuda.is_initialized(),'weights_loaded':False,'real_heads_fitting_tested':False}
    (directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True));print(json.dumps(receipt))
    if receipt['status']!='PASS' or receipt['parent_cuda_initialized']:raise SystemExit(1)
if __name__=='__main__':main()
