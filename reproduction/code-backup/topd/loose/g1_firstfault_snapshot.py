"""Read-only capture of actual primary failures and registration artifacts."""
import argparse,inspect,json,os,pathlib,subprocess,sys

def collect(root,job):
 import hashlib,json,pathlib,subprocess,time
 r=pathlib.Path(root);prefix='/dataset/shaoshitong/search-r1-rl/topd/logs/'
 if not str(r).startswith(prefix) or '..' in r.parts or r.is_symlink():raise ValueError('Not an own logs root')
 result={'scope':'READ_ONLY_ACTUAL_FIRST_FAULT_CAPTURE_NOT_ACCEPTANCE','epoch':time.time(),'job_id':job,'root':str(r),'files':[],'errors':[]}
 for command,key in [(['squeue','-h','-j',str(job),'-o','%i|%T|%M'],'queue'),(['scontrol','show','job',str(job),'-o'],'scontrol')]:
  try:
   q=subprocess.run(command,capture_output=True,text=True,timeout=15);result[key]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
  except Exception as exc:result[key]={'error':repr(exc)}
 if not r.exists():result['root_exists']=False;return result
 result['root_exists']=True
 for p in sorted(r.rglob('*.json')):
  if p.is_symlink() or not p.is_file():continue
  is_primary=p.name.endswith('-primary-error.json')
  is_registration=p.name.endswith(('-registration-before-cleanup.json','-registration-after-cleanup.json'))
  if not (is_primary or is_registration):continue
  try:
   # Atomic producer publications are complete final files. Preserve bytes/hash;
   # avoid collecting generated text, full token traces or environment values.
   st=p.stat();h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   item={'path':str(p),'relative_path':str(p.relative_to(r)),'size':st.st_size,'mtime_epoch':st.st_mtime,'sha256':h.hexdigest()}
   if st.st_size<=8*1024*1024:
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise RuntimeError('Artifact changed during capture')
    data=json.loads(raw)
    if is_primary:item['primary_error']=data
    else:
     events=data.get('events') or [];ledger=data.get('ledger') or {};roots=data.get('roots') or {}
     item['registration']={'scope':data.get('scope'),'sealed':data.get('sealed'),'roots':roots,'ledger':ledger,'open_pidfd_registered_pids':data.get('open_pidfd_registered_pids'),'events_count':len(events),'event_scope_counts':{name:sum(e.get('scope')==name for e in events) for name in sorted({e.get('scope') for e in events if e.get('scope')})},'last_events':events[-8:]}
   else:item['parse_status']='TOO_LARGE_NOT_PARSED_FULL_SHA_ONLY'
   result['files'].append(item)
  except Exception as exc:result['errors'].append({'path':str(p),'error':repr(exc)})
 result['primary_error_artifact_count']=sum('primary_error' in f for f in result['files'])
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('job',type=int);a.add_argument('root');a.add_argument('--output');args=a.parse_args()
 if args.job<=0:raise ValueError('Invalid actual job')
 body=inspect.getsource(collect)+'\nimport json\nprint(json.dumps(collect('+repr(args.root)+','+repr(args.job)+'),sort_keys=True))\n'
 command="python3 - <<'READ_ONLY_CAPTURE'\n"+body+'\nREAD_ONLY_CAPTURE\n'
 route=pathlib.Path(__file__).with_name('remote_topd.py')
 completed=subprocess.run([sys.executable,str(route)],input=command,text=True,capture_output=True,timeout=180)
 if completed.returncode:
  print(completed.stderr,file=sys.stderr);raise SystemExit(completed.returncode)
 record=json.loads(completed.stdout.strip().splitlines()[-1]);record['reader_sha256']=__import__('hashlib').sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
 output=pathlib.Path(args.output) if args.output else pathlib.Path(__file__).resolve().parents[1]/'outputs/remote-evidence'/('g1-firstfault-'+str(args.job)+'.jsonl')
 with output.open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(record,sort_keys=True))
if __name__=='__main__':main()
