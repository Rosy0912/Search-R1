#!/usr/bin/env python3
"""Exact installed official overlay binding, native CPU fixture fitting only."""
import argparse,hashlib,importlib.metadata,json,os,pathlib,socket,subprocess,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');HERE=pathlib.Path(__file__).resolve().parent
OVERLAY_FILES='9d6135dedad5647d8d2e73b04cc727f78dff3e99c832ba34b44a01e5f019f317'
FITTING_SOURCE='e9281af4f0c67490e990fec0bb14a83c2032b98ba367fff4fbedec2b445f9df7'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verify_overlay(directory,expected):
 directory=pathlib.Path(directory);receipt=directory/'install-receipt.json'
 if receipt.is_symlink() or sha(receipt)!=expected:raise ValueError('actual install receipt bytes mismatch')
 r=json.loads(receipt.read_text())
 if r.get('status')!='COMPLETE_CPU_OVERLAY_ONLY' or r.get('files_sha256')!=OVERLAY_FILES or r.get('fixed_env_before')!=r.get('fixed_env_after'):raise ValueError('incomplete or changed fixed environment overlay')
 overlay=directory/'overlay'
 if overlay.is_symlink() or str(overlay)!=r['overlay']:raise ValueError('overlay path identity changed')
 for name,digest in r['overlay_files'].items():
  p=overlay/name
  if p.is_symlink() or not p.resolve().is_relative_to(overlay.resolve()) or sha(p)!=digest:raise ValueError('installed official overlay bytes changed')
 return r,overlay

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--files-sha256',required=True);ap.add_argument('--install-receipt-sha256',required=True);a=ap.parse_args();os.umask(0o077)
 if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('SLURM_JOB_GPUS') or not HERE.is_relative_to(ROOT/'tools'):raise ValueError('own CPU-only Slurm required')
 f=HERE/'files.json'
 if sha(f)!=a.files_sha256:raise ValueError('head fixture files manifest drift')
 for name,r in json.loads(f.read_text())['files'].items():
  p=HERE/name
  if p.is_symlink() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:raise ValueError('head fixture source drift')
 if sha(HERE/'head_fidelity.py')!=FITTING_SOURCE:raise ValueError('audited fitting source drift')
 installed,overlay=verify_overlay(ROOT/'tools/g1-head-dependencies-v2',a.install_receipt_sha256)
 for name,record in installed['fixed_env_after'].items():
  d=importlib.metadata.distribution(name)
  if d.version!=record['version'] or sha(d._path/'METADATA')!=record['metadata_sha256']:raise ValueError('fixed env drift since install')
 directory=ROOT/'logs'/('g1-head-fidelity-native-'+os.environ['SLURM_JOB_ID']);directory.mkdir(exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(overlay)+os.pathsep+str(ROOT/'test-tools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 command=[sys.executable,str(HERE/'head_fidelity.py'),'--producer',str(ROOT/'tools/g1-staged-calibration-producer-v2'),'--directory',str(directory/'fidelity')]
 with open(directory/'head-fidelity.log','wb') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=360)
 receipt={'scope':'CPU_FIXTURE_ORIGINAL_HEAD_FIDELITY_NO_CALIBRATION_NO_TRAINING','returncode':result.returncode,'source_sha256':FITTING_SOURCE,'files_sha256':a.files_sha256,'install_receipt_sha256':a.install_receipt_sha256,'log_sha256':sha(directory/'head-fidelity.log'),'job':os.environ['SLURM_JOB_ID'],'pid':os.getpid(),'ppid':os.getppid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'command':command}
 if (directory/'fidelity/receipt.json').exists():receipt['raw_fidelity_receipt_sha256']=sha(directory/'fidelity/receipt.json')
 (directory/'execution.json').write_text(json.dumps(receipt,sort_keys=True));print(json.dumps(receipt))
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
