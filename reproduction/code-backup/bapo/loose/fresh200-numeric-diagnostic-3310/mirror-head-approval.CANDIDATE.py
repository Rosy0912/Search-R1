"""Mirror exact existing worker approval stdin bytes once; never republishes/submits."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/numeric-diagnostic-3310-20261006')
SOURCE='90755a561abd239fbb23882939410e643bb913ae'
MANIFEST='8a79b017d526de2cb9ff78f3e701e2572a5da0029545956d985394b72d492830'
def require(ok,message):
 if not ok:raise ValueError(message)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--expected-approval-sha',required=True);p.add_argument('--expected-nonce',required=True);args=p.parse_args()
 require(not sys.flags.optimize and sys.platform=='linux'and os.uname().nodename=='h100-16','Actual own scheduler head required')
 raw=sys.stdin.buffer.read();require(hashlib.sha256(raw).hexdigest()==args.expected_approval_sha,'Trusted actual worker approval bytes required');a=json.loads(raw)
 require(ROOT.resolve()==ROOT and not ROOT.is_symlink()and (ROOT/'logs').resolve()==ROOT/'logs','Canonical own head root required')
 require(hashlib.sha256((ROOT/'logs/tools-manifest.json').read_bytes()).hexdigest()==MANIFEST and (ROOT/'source-expected.txt').read_text().strip()==SOURCE,'Exact own head metadata required')
 require(a.get('root')==str(ROOT)and a.get('source_commit')==SOURCE and a.get('tools_manifest_sha256')==MANIFEST and a.get('nonce')==args.expected_nonce and a.get('protocol')=='BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1'and a.get('status')=='ALLOW_ZERO_ADAM_NUMERIC_REPLAY','Exact new diagnostic identity required')
 for key in ['optimizer_writes','formal_outer_updates','scheduler_updates']:require(type(a.get(key))is int and a[key]==0,'Strict zero update scope required')
 require(a.get('native_acceptance')is False and a.get('production_release')is False and type(a.get('allocation_gpu_count'))is int and a['allocation_gpu_count']==8 and type(a.get('probe_gpu_count'))is int and a['probe_gpu_count']==1,'Only diagnostic 8allocated1compute required')
 require(not list((ROOT/'scheduler-head').glob('*-intent.json'))and not list((ROOT/'scheduler-head').glob('*-receipt.json')),'Existing or unknown submission must not be retried')
 target=ROOT/'logs/diagnostic-approval.json'
 with target.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 require(hashlib.sha256(target.read_bytes()).hexdigest()==args.expected_approval_sha,'Head mirror differs')
 print(json.dumps({'scope':'EXACT_EXISTING_WORKER_APPROVAL_HEAD_MIRROR_ONLY','approval_sha256':args.expected_approval_sha,'submitted':False}))
