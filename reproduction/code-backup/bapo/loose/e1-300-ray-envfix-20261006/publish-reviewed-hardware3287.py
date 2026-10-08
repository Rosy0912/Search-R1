"""Exclusive hardware evidence publication and frozen gate; never submit training."""
import base64,hashlib,json,pathlib,re,subprocess,sys,time,os
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006')
SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876'
def need(v,msg):
 if not v:raise ValueError(msg)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def decode(row):
 raw=base64.b64decode(row['base64'],validate=True);need(type(row['bytes'])is int and len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Original publication bytes/SHA differ');return raw

def validate_payload(payload):
 need(set(payload)=={'receipt','report'},'Exact two original publication files')
 receipt=decode(payload['receipt']);report=decode(payload['report']);r=json.loads(receipt)
 need(r['publication_ready']is True and r['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and r['source_commit']==SOURCE and r['root']==str(ROOT)and type(r['job_id'])is int and r['job_id']==3287 and r['run_root']==str(ROOT/'runs/bapo-instruct300-hardware-3287'),'Only fresh signed actual3287 hardware review may publish')
 need(r['actual_gpu_accepted']is True and r['production_training_permission_signed']is False,'Hardware only, never production permission')
 p=pathlib.Path(r['independent_review']['path']);need(p.parent==ROOT/'logs'and p.name.startswith('HARDWARE3287-')and p.suffix=='.md','Exact final report newROOT path')
 need(r['independent_review']['sha256']==hashlib.sha256(report).hexdigest()and r['independent_review']['bytes']==len(report),'Signed final report bytes/SHA differ')
 need(payload['report']['path']==str(p)and payload['receipt']['path']==str(ROOT/'logs/hardware-acceptance.json'),'Exact report/receipt publication destinations')
 evidence=r['evidence'];need(type(evidence)is list and evidence and len({x['path']for x in evidence})==len(evidence),'Unique original receipt evidence')
 for x in evidence:
  q=pathlib.Path(x['path']);need(q.is_absolute()and ROOT in q.parents and '..'not in q.parts and type(x['bytes'])is int and x['bytes']>=0 and re.fullmatch('[a-f0-9]{64}',x['sha256']),'Canonical complete evidence binding')
 return r,receipt,report,p

def publish(payload):
 start=time.monotonic();r,receipt,report,p=validate_payload(payload);logs=ROOT/'logs';target=logs/'hardware-acceptance.json'
 need(logs.is_dir()and logs.resolve()==logs and not logs.is_symlink()and not p.exists()and not p.is_symlink()and not target.exists()and not target.is_symlink(),'Exclusive fresh ordinary publication only')
 need((ROOT/'source-expected.txt').read_text().strip()==SOURCE,'Exact fixed source marker')
 manifestp=logs/'tools-manifest.json';need(digest(manifestp)=='8bd3fccd459f4b58a6dd048c624328b6fe4f067f8c26333aaef81d560e32ba46','Frozen36 exact original manifest')
 manifest=json.loads(manifestp.read_text());need(manifest['source_commit']==SOURCE and len(manifest['files'])==36,'Exact source/tools count')
 gate=ROOT/'tools/verify_hardware_receipt.py';need(gate.is_file()and not gate.is_symlink()and digest(gate)==manifest['files'][gate.name],'Only original frozen full gate may execute')
 for x in r['evidence']:
  q=pathlib.Path(x['path']);need(q.is_file()and not q.is_symlink()and q.resolve()==q and q.stat().st_size==x['bytes']and digest(q)==x['sha256'],'Actual original evidence changed before publication '+str(q))
 need(digest(gate)==manifest['files'][gate.name],'Gate changed during preflight')
 with p.open('xb')as f:f.write(report)
 need(p.read_bytes()==report,'Published original report differs')
 with target.open('xb')as f:f.write(receipt)
 need(target.read_bytes()==receipt,'Published original receipt differs')
 command=[str(ROOT/'env/bin/python'),'-B',str(gate)];result=subprocess.run(command,cwd=ROOT/'repo',env=dict(os.environ,PYTHONPATH=str(ROOT/'repo'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
 record=dict(status='ACTUAL_FROZEN_HARDWARE_GATE_EXECUTION_NOT_TRAINING_PERMISSION',source_commit=SOURCE,job_id=3287,receipt_path=str(target),receipt_sha256=hashlib.sha256(receipt).hexdigest(),report_path=str(p),report_sha256=hashlib.sha256(report).hexdigest(),gate_path=str(gate),gate_sha256=digest(gate),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-start,training_submitted=False)
 out=logs/'hardware3287-final-frozen-gate-execution.json';need(not out.exists(),'Fresh gate execution record only')
 with out.open('x')as f:f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(json.dumps(record),flush=True);return result.returncode
if __name__=='__main__':raise SystemExit(publish(json.loads(sys.stdin.buffer.read())))
