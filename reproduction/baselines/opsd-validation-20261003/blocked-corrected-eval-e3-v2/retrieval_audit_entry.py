"""Capture executed immutable retrieval frames without changing encoder/index/math."""
import argparse,hashlib,json,os,runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'strict-tools'))
from runtime_capture import process_identity
from training_evidence import digest

def run(target,frame_output,args):
 from runtime_binding import load_runtime
 bundle,binding=load_runtime(__file__)
 import uvicorn
 original=uvicorn.run;target=Path(target).resolve();repo=Path(os.environ['SEARCH_R1_REPO']).resolve()
 if target!=repo/'reproduction/h20/retriever_local.py':raise ValueError('Exact source-pinned native local retriever required')
 def capture(app,**kwargs):
  # Native local_run already forces localhost; this wrapper does not change kwargs.
  if kwargs.get('host')!='127.0.0.1':raise ValueError('Actual native retrieval host is not local')
  frame=sys._getframe(1);found={}
  while frame:
   path=Path(frame.f_code.co_filename).resolve()
   for module,relative in [('search_r1.search.retrieval_server','search_r1/search/retrieval_server.py'),('reproduction.h20.retriever_local','reproduction/h20/retriever_local.py')]:
    if path==repo/relative:found[module]={'path':str(path),'sha256':digest(path)}
   frame=frame.f_back
  if len(found)!=2:raise ValueError('Actual executed native retrieval frames missing')
  record={'job_id':os.environ['SLURM_JOB_ID'],'node':__import__('socket').gethostname().split('.')[0],'url':'http://127.0.0.1:18084/retrieve','process':process_identity(os.getpid()),'source_frames':found,'runtime_binding':binding}
  with Path(frame_output).open('x') as handle:json.dump(record,handle,sort_keys=True)
  return original(app,**kwargs)
 uvicorn.run=capture;sys.argv=[str(target)]+args
 runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__':
 print('BLOCKED retrieval audit entry preview',file=sys.stderr)
 raise SystemExit(78)
 parser=argparse.ArgumentParser();parser.add_argument('--target',required=True);parser.add_argument('--frame-output',required=True)
 args,remaining=parser.parse_known_args();run(args.target,args.frame_output,remaining)
