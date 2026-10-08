"""Prepared once-only CPU scan operation; no remote actions without --execute."""
import sys,pathlib,json,hashlib,tarfile,subprocess,fcntl
ROOT='/dataset/shaoshitong/search-r1-rl/topd'
ARCHIVES={'r1-32b-hf-semantic-review-v4.tar.gz':'70ece242398ded12ecc63756601c7b205e3a8eb68d09659b16ad373aa1a9e715','r1-hf-semantic-v4-native-operation.tar.gz':'e36be963913666cca8990bba846ec6efd481930f371dab87822169ef57a5ff60'}
REMOTE=r'''
import os,time,socket,importlib.util,re
root=pathlib.Path(ROOT);assert socket.gethostname()=='h100-11'
blobs={}
for name,h in ARCHIVES.items():
 p=root/'tools'/name;b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==h
 with tarfile.open(fileobj=__import__('io').BytesIO(b),mode='r:gz') as tf:
  for m in tf.getmembers():
   p=pathlib.PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.name not in blobs;blobs[m.name]=tf.extractfile(m).read()
for folder,pin in [('r1-32b-hf-semantic-review-v4','1b4bae153472c5778c5bfd8ab00905491a5b58820dbec62e742a3a377bab17db'),('r1-hf-semantic-v4-native-operation','55dea29c6886bebc56a21c52148b1ecca9aa1c646fcb0d7d7ec67448f5aa948b')]:
 prefix=folder+'/';manifest=blobs[prefix+'files.json'];assert hashlib.sha256(manifest).hexdigest()==pin
 members={n[len(prefix):]:b for n,b in blobs.items() if n.startswith(prefix)}
 spec=json.loads(manifest)['files'];assert set(members)==set(spec)|{'files.json'}
 for n,r in spec.items():assert len(members[n])==r['size'] and hashlib.sha256(members[n]).hexdigest()==r['sha256']
 target=root/'tools'/folder;assert not target.is_symlink()
 if target.exists():
  files={str(p.relative_to(target)):p for p in target.rglob('*') if p.is_file()};assert set(files)==set(members)
  for n,b in members.items():assert not files[n].is_symlink() and files[n].read_bytes()==b
 else:
  stage=target.with_name(folder+'.scan-staging');assert not stage.exists();stage.mkdir(mode=0o700)
  for n,b in members.items():
   p=stage/n;p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  fd=os.open(stage,os.O_RDONLY);os.fsync(fd);os.close(fd);os.rename(stage,target);fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
module=root/'tools/r1-hf-semantic-v4-native-operation/admission.py';spec=importlib.util.spec_from_file_location('pinned_scan_admission',module);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.check_request(REQUEST);job=a.job_id(REQUEST['job_id']);output=a.own(OUTPUT);assert root/'logs' in output.resolve().parents
request_sha=hashlib.sha256(json.dumps(REQUEST,sort_keys=True,separators=(',',':')).encode()).hexdigest()
record=root/'logs'/('r1-hf-semantic-v4-train'+str(job)+'-submission.json')
def durable(p,d):
 t=p.with_name(p.name+'.private');assert not t.exists() and not p.is_symlink()
 with t.open('x') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['request_canonical_sha256']==request_sha and d['output']==str(output) and d['archives']==ARCHIVES,'UNKNOWN/mismatch: inspect recorded job; never repeat'
 else:
  assert not output.exists(),'Original result cannot be overwritten'
  queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True,timeout=10);assert not any(x.split('|')[-1]=='r1-hf-semantic-v4-cpu' for x in queue.splitlines())
  request=root/'logs'/('r1-hf-semantic-v4-train'+str(job)+'-request.json');assert not request.exists();durable(request,REQUEST)
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',scope='CPU_FULL_HF_WEIGHT_EVIDENCE_NOT_TEACHER_APPROVAL',training_job_id=job,request_canonical_sha256=request_sha,request_path=str(request),request_sha256=a.sha(request),output=str(output),archives=ARCHIVES,epoch=time.time());durable(record,d)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',R1_HF_REQUEST=str(request),R1_HF_OUTPUT=str(output),R1_HF_OPERATION_MANIFEST='55dea29c6886bebc56a21c52148b1ecca9aa1c646fcb0d7d7ec67448f5aa948b')
  entry=root/'tools/r1-hf-semantic-v4-native-operation/native-cpu.sbatch';subprocess.run(['bash','-n',str(entry)],check=True,timeout=5)
  run=subprocess.run(['sbatch','--parsable','--chdir='+str(root),str(entry)],capture_output=True,text=True,env=env,timeout=30)
  d.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr);durable(record,d);assert run.returncode==0
  jid=run.stdout.strip().split(';')[0];assert jid.isdigit();d.update(status='SUBMITTED',job_id=int(jid));durable(record,d)
 jid=str(d['job_id']);watch=root/'logs'/('r1-hf-semantic-v4-'+jid+'-control.jsonl');deadline=time.monotonic()+10860
 while time.monotonic()<deadline:
  try:ctl=subprocess.run(['scontrol','show','job','-dd',jid],text=True,capture_output=True,timeout=min(5,max(.1,deadline-time.monotonic())))
  except (subprocess.TimeoutExpired,OSError) as exc:
   with watch.open('a') as f:f.write(json.dumps(dict(epoch=time.time(),job_id=int(jid),observer_error=repr(exc)))+'\n');f.flush();os.fsync(f.fileno())
   raise
  snap=dict(epoch=time.time(),job_id=int(jid),returncode=ctl.returncode,stdout=ctl.stdout,stderr=ctl.stderr)
  with watch.open('a') as f:f.write(json.dumps(snap)+'\n');f.flush();os.fsync(f.fileno())
  assert ctl.returncode==0,'Actual terminal unavailable: do not repeat scan'
  assert re.findall(r'\bJobId=([0-9]+)',ctl.stdout)==[jid]
  if any('JobState='+x in ctl.stdout for x in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']):
   final=root/'logs'/('r1-hf-semantic-v4-'+jid+'-actual-final.json');assert not final.exists();durable(final,snap);d.update(actual_final={'path':str(final),'sha256':a.sha(final)});durable(record,d)
   if output.exists():d['semantic_output']={'path':str(output),'sha256':a.sha(output)}
   print(json.dumps(d),flush=True);break
  time.sleep(2)
 else:raise RuntimeError('Bounded observer timed out; retain record, do not retry scan')
'''
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--request',type=pathlib.Path);p.add_argument('--output');p.add_argument('--execute',action='store_true');args=p.parse_args();base=pathlib.Path(__file__).resolve().parent.parent
 for n,h in ARCHIVES.items():assert hashlib.sha256((base/'work'/n).read_bytes()).hexdigest()==h
 compile(REMOTE,'cpu-scan-operation','exec')
 if not args.execute:print('PREPARED_STATIC_ONLY_NOT_SUBMITTED_NOT_SCANNED');return
 assert args.request and args.output
 REQUEST=json.loads(args.request.read_text());OUTPUT=args.output
 with (base/'outputs/r1-hf-semantic-v4-native-operation-local.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for n in ARCHIVES:subprocess.run([sys.executable,str(base/'work/transfer_file.py'),str(base/'work'/n),ROOT+'/tools/'+n],check=True,cwd=base)
  prefix=pathlib.Path(__file__).read_text().split('REMOTE=',1)[0]
  script="python3 -B - <<'HF_SCAN_CPU_ONCE'\n"+prefix+'\nREQUEST='+repr(REQUEST)+'\nOUTPUT='+repr(OUTPUT)+'\n'+REMOTE+'\nHF_SCAN_CPU_ONCE\n'
  with (base/'outputs/r1-hf-semantic-v4-cpu-operation-invocation.log').open('a') as out:
   r=subprocess.run([sys.executable,str(base/'work/remote_topd.py')],input=script,text=True,stdout=out,stderr=subprocess.STDOUT,cwd=base);out.flush();__import__('os').fsync(out.fileno())
  assert r.returncode==0,'Inspect record/job; no automatic repeated submission'
if __name__=='__main__':main()
