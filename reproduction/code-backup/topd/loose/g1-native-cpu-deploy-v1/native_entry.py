"""CPU suites only; incomplete proof until ROOT captures actual Slurm terminal."""
import argparse,hashlib,json,os,pathlib,re,socket,subprocess,sys,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
PIPELINE_SHA='10e5d3dd3220825859656483dd9ac08e05f0e7f4a33b8d2415f1b8189c04ef23'
KNOWN_SHA='e14c95bd26ccf5c748f849e6f2f2b0e60904e1f91e24a92174462cbd4202157b'
CANDIDATE_SHA='8f6582c4cb31ccbdbef62cfbe602981d1cf7032b69a01655604908d64fb783ff'
PRODUCER_SHA='b78f064b492af20c7092202c8f2f7268712aa40a74a66347fc55028c31fe6cbd'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verify(d,relative,expected):
 if d.is_symlink():raise ValueError('Own source directory alias')
 m=d/relative
 if sha(m)!=expected:raise ValueError('Source manifest differs')
 records=dict(json.loads(m.read_text())['files']);records[relative]={'size':m.stat().st_size,'sha256':expected};actual=set()
 for p in d.rglob('*'):
  if p.is_symlink():raise ValueError('Source alias')
  if p.is_file():actual.add(str(p.relative_to(d)))
 if actual!=set(records):raise ValueError('Source exact file set differs')
 for name,rec in records.items():
  p=d/name
  if p.stat().st_size!=rec['size'] or sha(p)!=rec['sha256']:raise ValueError('Source bytes differ')
 return {'directory':str(d),'manifest_sha256':expected,'files':len(records)}
def atomic(path,value):
 p=pathlib.Path(path);tmp=p.with_name(p.name+'.private')
 with tmp.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Own CPU-only native allocation required')
 out=pathlib.Path(a.output)
 if ROOT/'logs' not in out.resolve().parents or out.is_symlink():raise ValueError('Own logs required')
 here=pathlib.Path(__file__).resolve().parent
 verified=[verify(here,'files.json',a.manifest_sha256)]
 for name,relative,digest in [('g1-production-pipeline-v10','files.json',PIPELINE_SHA),('g1-known-run-tools-v2','files.json',KNOWN_SHA),('g1-32b-fallback-candidate-v3','reproduction/conditional-teacher100/files.json',CANDIDATE_SHA),('g1-staged-calibration-producer-v4','files.json',PRODUCER_SHA)]:verified.append(verify(ROOT/'tools'/name,relative,digest))
 os.environ.update(PIPELINE_ROOT=str(ROOT/'tools/g1-production-pipeline-v10'),CANDIDATE_ROOT=str(ROOT/'tools/g1-32b-fallback-candidate-v3'),PIPELINE_MANIFEST_SHA256=PIPELINE_SHA,KNOWN_TOOLS_ROOT=str(ROOT/'tools/g1-known-run-tools-v2'),KNOWN_TOOLS_MANIFEST_SHA256=KNOWN_SHA)
 logs={}
 for name,entry,count in [('pipeline','g1-production-pipeline-v10',176),('known_tools','g1-known-run-tools-v2',14)]:
  log=out/(name+'-pytest.log');started=time.time()
  with log.open('xb') as f:r=subprocess.run(['bash',str(ROOT/'tools'/entry/'native-cpu.sbatch')],env=dict(os.environ),stdout=f,stderr=subprocess.STDOUT)
  text=log.read_text();ok=r.returncode==0 and re.search(r'(?m)^'+str(count)+r' passed in [0-9.]+s$',text) is not None
  logs[name]={'path':str(log),'sha256':sha(log),'passed':count if ok else None,'returncode':r.returncode,'started_epoch':started,'completed_epoch':time.time()}
  if not ok:
   atomic(out/'receipt.json',{'scope':'ACTUAL_NATIVE_CPU_SOURCE_NO_GPU_NO_TRAINING','status':'FAIL','job_id':int(os.environ['SLURM_JOB_ID']),'source':verified,'logs':logs});raise RuntimeError('Actual native zero-skip CPU suite failed '+name)
 pending={'directory':str(ROOT/'tools/g1-production-pipeline-v10'),'manifest_sha256':PIPELINE_SHA,'native_expected':176,'native_cpu':{'job_id':int(os.environ['SLURM_JOB_ID']),'passed':176,'log':{'path':logs['pipeline']['path'],'sha256':logs['pipeline']['sha256']},'terminal':None},'scope':'CPU_TESTS_COMPLETE_TERMINAL_PENDING_NOT_USABLE_SOURCE_PROOF'}
 atomic(out/'pipeline-source-binding.pending.json',pending)
 receipt={'scope':'ACTUAL_NATIVE_CPU_SOURCE_NO_GPU_NO_TRAINING','status':'CPU_TESTS_COMPLETE_TERMINAL_PENDING','job_id':int(os.environ['SLURM_JOB_ID']),'source':verified,'logs':logs,'pid':os.getpid(),'ppid':os.getppid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'parent_torch_imported':'torch' in sys.modules,'submitted_gpu_jobs':False}
 atomic(out/'receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
