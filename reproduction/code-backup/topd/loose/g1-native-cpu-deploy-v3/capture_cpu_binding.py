"""ROOT invokes after native job completion; preserve real raw scontrol before TTL."""
import argparse,json,os,pathlib,re,socket,subprocess,time
from native_entry import ROOT,PIPELINE_SHA,KNOWN_SHA,sha,atomic

def finalize(out,*,runner=subprocess.run):
 out=pathlib.Path(out)
 if ROOT/'logs' not in out.resolve().parents or out.is_symlink():raise ValueError('Own source output required')
 pending_path=out/'pipeline-source-binding.pending.json';p=json.loads(pending_path.read_text());receipt=json.loads((out/'receipt.json').read_text());job=p['native_cpu']['job_id']
 if receipt['status']!='CPU_TESTS_COMPLETE_TERMINAL_PENDING' or receipt['job_id']!=job or p['manifest_sha256']!=PIPELINE_SHA or p['native_expected']!=184:raise ValueError('Actual source CPU receipt mismatch')
 for name,count in [('pipeline',184),('known_tools',14)]:
  log=receipt['logs'][name]
  if log['passed']!=count or log['returncode']!=0 or sha(log['path'])!=log['sha256']:raise ValueError('Actual native source logs drift')
  if not re.search(r'(?m)^'+str(count)+r' passed in [0-9.]+s$',pathlib.Path(log['path']).read_text()):raise ValueError('Actual complete zero-skip suite absent')
 binding_path=out/'pipeline-source-binding.json'
 if binding_path.exists():
  b=json.loads(binding_path.read_text());terminal=b['native_cpu']['terminal']
  if b['manifest_sha256']!=PIPELINE_SHA or b['directory']!=p['directory'] or b['native_cpu']['job_id']!=job or b['native_cpu']['log']!=p['native_cpu']['log'] or b['native_expected']!=184 or sha(terminal['path'])!=terminal['sha256']:raise ValueError('Existing actual binding drift')
  return b
 r=runner(['scontrol','show','job','-dd',str(job)],text=True,capture_output=True)
 text=r.stdout;complete=r.returncode==0 and all(re.search(r'(?:^|\s)'+re.escape(v)+r'(?:\s|$)',text) for v in ['JobId='+str(job),'JobState=COMPLETED','ExitCode=0:0','NodeList=h100-11'])
 raw={'job_id':job,'scontrol_returncode':r.returncode,'scontrol_stdout':r.stdout,'scontrol_stderr':r.stderr,'captured_epoch':time.time(),'final_state_observed':bool(complete)}
 raw_path=out/('scontrol-captured-'+str(time.time_ns())+'.json');atomic(raw_path,raw)
 if not complete:raise RuntimeError('Actual successful native terminal still absent; raw preserved')
 p['native_cpu']['terminal']={'path':str(raw_path),'sha256':sha(raw_path)};p['scope']='ACTUAL_NATIVE_CPU_SOURCE_IDENTITY_NOT_GPU_OR_TRAINING_APPROVAL'
 atomic(binding_path,p);return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 if socket.gethostname()!='h100-11':raise ValueError('Own h11 only')
 print(json.dumps(finalize(a.output)))
if __name__=='__main__':main()
