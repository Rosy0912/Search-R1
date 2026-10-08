import pathlib,tempfile,json,hashlib,subprocess,sys,os
root=pathlib.Path(__file__).resolve().parents[2];rows=[]
with tempfile.TemporaryDirectory() as tmp:
 t=pathlib.Path(tmp);package=t/'urllib';package.mkdir();(package/'__init__.py').write_text('import sysconfig,os\n__path__.append(os.path.join(sysconfig.get_path("stdlib"),"urllib"))\n');(package/'request.py').write_text('''import io,json
counts={}
class Request:
 def __init__(self,url,data,**kw): self.data=data
def urlopen(req,timeout):
 if isinstance(req,str): return io.BytesIO(json.dumps({'pid':1234}).encode())
 p=json.loads(req.data);k=p['topk'];key=(len(p['queries']),k);n=counts.get(key,0);counts[key]=n+1
 data=[[{'document':{'id':str(j),'contents':'doc'+str(j)},'score':float(j)+n*0.0001} for j in range(k)] for q in p['queries']]
 return io.BytesIO(json.dumps({'result':data}).encode())
''')
 qs=['q'+str(i) for i in range(14)];base={'queries':qs,'queries_sha256':'a'*64,'results':[{str(k):{'ids':list(range(k)),'scores':list(map(float,range(k))),'contents_sha256':[hashlib.sha256(('doc'+str(j)).encode()).hexdigest() for j in range(k)]} for k in (3,50)} for i in qs]};reference=t/'base.json';reference.write_text(json.dumps(base));output=t/'http.json'
 p=subprocess.run([sys.executable,str(root/'work/probe_aux_http.py'),'--reference',str(reference),'--output',str(output)],env=dict(os.environ,PYTHONPATH=str(t)),capture_output=True,text=True);assert p.returncode==0,p.stderr
 report=json.loads(output.read_text());assert report['status']=='DIFFERENCE_REPORT_ONLY' and all(not r['repeat_equal'] and len(r['repeats'])==2 for r in report['rows']);rows.append({'case':'unstable_repeats_preserved','exit_code':p.returncode,'row_counts':[r['query_count'] for r in report['rows']],'status':report['status']})
 text=(root/'work/train_tree_stage.sh').read_text();loop=text[text.index('while kill -0 "$TRAIN_PID"'):text.index('printf \'%s\\n\' "$RUN"')]
 for backend,ret,guard,check,name in [('cpu',0,0,0,'cpu_alive_no_aux'),('cpu',1,0,0,'cpu_dead'),('hybrid',0,0,0,'aux_alive'),('hybrid',0,0,1,'aux_failure'),('hybrid',0,1,0,'aux_guard_dead')]:
  log=t/(name+'.log');script='''set -euo pipefail
TRAIN_PID=99; RETRIEVER_PID=77; GUARD_PID=88; SLURM_JOB_ID=42; ROOT=/fixture; RUN=/fixture/run; AUX_ACCEPTANCE_SHA256=fixtureSHA; STAGE_PROVENANCE_SHA256=fixtureSTAGEsha; calls=0
kill() { if [ "$1" = -TERM ]; then echo "TERM:$2" >> "$LOG"; return 0; fi; case "$2" in 99) calls=$((calls+1)); [ "$calls" -eq 1 ];; 77) return "$RET";; 88) return "$GUARD";; *) return 1;; esac; }
python() { echo "CHECK:$*" >> "$LOG"; return "$CHECK"; }
timeout() { echo "TIMEOUT:$1 $2 $3" >> "$LOG"; [ "$1" = --signal=TERM ] && [ "$2" = --kill-after=5 ] && [ "$3" = 190 ]; shift 3; "$@"; }
scancel() { echo "SCANCEL:$*" >> "$LOG"; }
sleep() { :; }
wait() { :; }
'''+loop
  p=subprocess.run(['bash','-c',script],env=dict(os.environ,BACKEND=backend,RET=str(ret),GUARD=str(guard),CHECK=str(check),LOG=str(log)),capture_output=True,text=True);actions=log.read_text().splitlines() if log.exists() else []
  assert p.returncode==(1 if ret or guard or check else 0),(name,p.stderr)
  if backend=='cpu':assert not any(x.startswith(('CHECK','TIMEOUT')) for x in actions)
  else:assert any('--expected-acceptance-sha256 fixtureSHA' in x for x in actions) and any(x=='TIMEOUT:--signal=TERM --kill-after=5 190' for x in actions) and any('--expected-stage-provenance-sha256 fixtureSTAGEsha' in x for x in actions)
  if p.returncode:assert 'SCANCEL:--signal=TERM 42' in actions and 'TERM:99' in actions
  rows.append({'case':name,'exit_code':p.returncode,'actions':actions})
(root/'work/aux-gate-review/http-monitor-fixed-results.json').write_text(json.dumps(rows,indent=2)+'\n');print('6 HTTP/monitor fixtures passed')
