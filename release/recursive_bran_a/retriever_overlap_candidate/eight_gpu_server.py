"""Independent eight-shard HTTP service; original oracle is never modified."""
import argparse, hashlib, importlib.util, json, os, sys, threading, time
from pathlib import Path
os.environ['NVIDIA_TF32_OVERRIDE']='0'
os.environ['E1_ENCODER_DEVICE']='cpu'

def main():
 p=argparse.ArgumentParser();p.add_argument('--oracle-source',required=True);p.add_argument('--base',required=True)
 p.add_argument('--assets',required=True);p.add_argument('--asset-manifest',required=True)
 p.add_argument('--host',default='10.120.17.101');p.add_argument('--port',type=int,default=18006)
 p.add_argument('--collection-seconds',type=float,default=.025);p.add_argument('--max-pending',type=int,default=128)
 a=p.parse_args();sys.path.insert(0,a.oracle_source)
 spec=importlib.util.spec_from_file_location('frozen_oracle_server',Path(a.oracle_source)/'server.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 from gpu_phase_proxy import PhaseGpuScanner
 class EightCandidate(old.Candidate):
  def health(self):
   result=super().health();result.update(index='full FP32 eight-GPU tiled FlatIP',gpu_shards=8,
    partition='round-robin whole original 32768-row tiles',resident_index=False)
   result.update(gpu_worker_pid=self.scan.worker_pid if self.scan is not None else None,
    gpu_worker_active=self.scan.worker_pid is not None if self.scan is not None else False)
   result['last_scan_profile']=self.scan.last_scan_profile if self.scan is not None else None
   return result
  def admitted(self):
   c=self.read_bound_control(self.admission)
   if not self.gpu_ready or not c or c.get('accepted') is not True:return False
   try:
    f=Path(c['gate_path']);g=json.loads(f.read_text())
    return (old.digest(f)==c['gate_sha256'] and g.get('accepted') is True and
     g['candidate_pid']==os.getpid() and g['source_identity_sha256']==self.identity_sha256 and
     g['scope']=='eight_gpu_full_scan_and_http_oracle_equality')
   except (OSError,ValueError,KeyError):return False
  def activate_and_run(self):
   try:
    while self.read_bound_control(self.activation) is None:time.sleep(.5)
    if old.torch.cuda.device_count()!=8:raise RuntimeError('Expose exactly the eight authorized GPUs')
    original=old.SegmentedStreamingFlatIP(self.retriever.index,'cuda:7',vector_rows=32768,query_micro=64)
    self.scan=PhaseGpuScanner(original, index_path=str(Path(a.assets)/'index/e5_Flat.index'), oracle_source=a.oracle_source)
    self.gpu_ready=True;self.record_event(dict(event='eight_gpu_probe_enabled',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    while True:
     batch=self.queue.next_batch()
     if not batch:return
     self.process_batch(batch)
   except BaseException as error:
    self.activation_error=repr(error);self.gpu_ready=False;self.queue.close(str(error))
    self.record_event(dict(event='eight_gpu_scanner_failed',error=repr(error),traceback=old.traceback.format_exc()))
  def retrieve(self,request,probe_token=None,compare_original=False):
   if compare_original:raise old.HTTPException(400,'Use external immutable oracle equality gate, not single-thread operator recorder')
   return super().retrieve(request,probe_token=probe_token,compare_original=False)
 candidate=EightCandidate(a)
 # No enable-gpu file exists at startup. Bind the implementation before activation.
 implementation={str(f.resolve()):old.digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))}
 candidate.identity['eight_gpu_implementation']=implementation
 candidate.identity_sha256=hashlib.sha256(json.dumps(candidate.identity,sort_keys=True).encode()).hexdigest()
 old.atomic_json(candidate.base/'receipts/source-identity.json',dict(pid=os.getpid(),source_identity_sha256=candidate.identity_sha256,
  identity=candidate.identity,cuda_initialized=False,cpu_loaded_unadmitted=True))
 app=old.FastAPI()
 @app.get('/health')
 def health():return candidate.health()
 @app.post('/retrieve')
 def retrieve(request:old.original_server.QueryRequest):return candidate.retrieve(request)
 @app.post('/release-gpu')
 def release_gpu(x_e1_probe_token:str=old.Header(default='')):
  if not old.secrets.compare_digest(x_e1_probe_token,candidate.probe_token):raise old.HTTPException(403,'Invalid private control token')
  if candidate.scan is None:return dict(released=True,gpu_worker_pid=None,previous_gpu_worker_pid=None)
  return candidate.scan.release_gpu()
 @app.post('/probe/retrieve')
 def probe(request:old.original_server.QueryRequest,x_e1_probe_token:str=old.Header(default='')):
  return candidate.retrieve(request,probe_token=x_e1_probe_token)
 old.uvicorn.run(app,host=a.host,port=a.port)
if __name__=='__main__':main()
