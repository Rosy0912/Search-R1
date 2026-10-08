"""Only approved explicit invocation performs original full CPU weight scan."""
import argparse,sys,os,pathlib,json,hashlib,importlib.util,time
sys.dont_write_bytecode=True
HERE=pathlib.Path(__file__).resolve().parent
assert hashlib.sha256((HERE/'files.json').read_bytes()).hexdigest()==os.environ['R1_HF_OPERATION_MANIFEST']
m=json.loads((HERE/'files.json').read_bytes())
for n,r in m['files'].items():
 p=HERE/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
sys.path.insert(0,str(HERE));from admission import check_request,ROOT,MANIFEST,own,sha
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='2' and sys.dont_write_bytecode
request=own(a.request);q=json.loads(request.read_bytes());check_request(q)
tool=ROOT/'tools/r1-32b-hf-semantic-review-v4';assert sha(tool/'files.json')==MANIFEST
for n,r in json.loads((tool/'files.json').read_bytes())['files'].items():
 f=tool/n;assert f.is_file() and not f.is_symlink() and f.stat().st_size==r['size'] and sha(f)==r['sha256']
output=own(a.output);assert ROOT/'logs' in output.resolve().parents and not output.exists()
s=importlib.util.spec_from_file_location('pinned_full_hf_semantic_v4',tool/'review.py');review=importlib.util.module_from_spec(s);s.loader.exec_module(review)
# Original function validates all actual metadata and full files before/after comparison.
review.binding_records(q);started=time.time()
try:result=review.run(q)
except Exception as exc:review.publish(output,dict(status='FAIL',error=repr(exc),request={'path':str(request),'sha256':sha(request)},operation_started_epoch=started));raise
review.publish(output,result)
assert not __import__('torch').cuda.is_initialized()
print(json.dumps({'scope':'CPU_FULL_WEIGHT_SCAN_UNREVIEWED_NOT_TEACHER_APPROVAL','output':str(output),'sha256':sha(output),'job_id':os.environ['SLURM_JOB_ID'],'started_epoch':started,'finished_epoch':time.time()}),flush=True)
