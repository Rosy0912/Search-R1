"""External native CPU historical transport replay; no acceptance/signing/GPU."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess,sys
from pathlib import Path
SOURCE='89beac273f783d8e48d7d3e34da040f753514d90'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--candidate',required=True);p.add_argument('--candidate-sha256',required=True);p.add_argument('--log',required=True);p.add_argument('--log-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-hidden visibility required')
 import torch
 require(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'Native CPU Torch2.4 only')
 repo=Path(a.repo);candidate=Path(a.candidate);log=Path(a.log)
 for q in (repo,candidate,log):require(q.is_absolute() and q.resolve(strict=True)==q and q.stat().st_uid==os.getuid(),'Actual owned physical inputs required')
 require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Old actual source changed')
 require(sha(candidate)==a.candidate_sha256 and sha(log)==a.log_sha256,'External candidate/log content pin differs')
 module=type(sys)('native_marker_candidate');module.__file__=str(candidate);exec(compile(candidate.read_bytes(),str(candidate),'exec'),module.__dict__)
 marker='OPSD_E1_OUTER_JSON=';workers=(470179,470515,470516,470517,470518,470519,470520,470522);rows=[];noise=None;line4=None
 for line in log.read_text().splitlines():
  if marker not in line:continue
  row=module.parse_log_marker(line,marker,driver_pid=469564,job_id='3198',worker_pids=workers);rows.append(row)
  if row['actual_outer_updates']==4:line4=line
 require([r['actual_outer_updates'] for r in rows]==[1,2,3,4] and line4 is not None,'Historical completed four-row scope differs')
 raw=line4.split(marker,1)[1];decoded,end=json.JSONDecoder().raw_decode(raw);require(rows[-1]==decoded,'Parsed object differs')
 try:json.loads(raw)
 except json.JSONDecodeError as error:noise=str(error)
 require(noise is not None,'Actual old full-suffix failure missing')
 # Exact bad transport and body negatives against the same default actual Torch path.
 negatives=[line4+'{}',line4+' extra',line4.replace('WorkerDict pid=470520','WorkerDict pid=1'),line4.replace('1399: FutureWarning','1400: FutureWarning'),line4.replace('job_id": "3198','job_id": "3196'),line4+marker+'{}',marker+'{"x":1e309}',marker+'{"x":{"y":-1e309}}',marker+'{"x":1,"x":2}']
 for line in negatives:
  try:module.parse_log_marker(line,marker,driver_pid=469564,job_id='3198',worker_pids=workers)
  except ValueError:pass
  else:raise ValueError('Malicious transport/body accepted')
 require(not torch.cuda.is_initialized() and sha(candidate)==a.candidate_sha256 and sha(log)==a.log_sha256,'Inputs drifted or CUDA initialized')
 require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Old actual source drifted')
 result={'scope':'NATIVE_CPU_HISTORICAL3198_TRANSPORT_REPLAY_NOT_FIRST5_ACCEPTANCE','source_commit':SOURCE,'node':socket.gethostname(),'uid':os.getuid(),'pid':os.getpid(),'torch_version':torch.__version__,'candidate_sha256':a.candidate_sha256,'input_log_sha256':a.log_sha256,'observed_completed_rows':[1,2,3,4],'old_json_loads_error':noise,'actual_row4_object_exact':True,'malicious_cases_rejected':len(negatives),'cuda_is_initialized':False,'passed_parse_replay_only':True}
 out=Path(a.output);require(out.is_absolute() and out.parent.resolve(strict=True)==out.parent and not out.is_symlink(),'New physical output required')
 with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
