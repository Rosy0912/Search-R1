import pathlib,tempfile,json,hashlib,subprocess,sys,re
root=pathlib.Path(__file__).resolve().parents[2];checker=root/'work/check_aux_retriever.py';launcher=root/'work/train_tree_stage.sh'
rows=[]
with tempfile.TemporaryDirectory() as tmp:
 t=pathlib.Path(tmp);payload=t/'real-encoder-difference.json';payload.write_text('{"status":"DIFFERENCE_REPORT_ONLY"}\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 prov={'gpu_uuid':'GPU-c63282c6-1d05-224e-234b-2a8c794b1f04','pid':1234,'run':'/data/shaoshitong/search-r1-rl/tree-grpo-retrieval/runs/fixture','encoder_device':'cpu','encoder_fp16':True,'index_float16':False,'difference_report_sha256':sha(payload),'source_sha256':'a'*64}
 acceptance={'status':'ACCEPTED_AFTER_REAL_ENCODER_HTTP_DIFFERENCE_REVIEW','url':'http://127.0.0.1:18082/retrieve','evidence_file_sha256':{str(payload):sha(payload)},'provenance':prov}
 (t/'requests.py').write_text('''import os,json
class R:
 def raise_for_status(self): pass
 def json(self): return json.loads(os.environ['FAKE_PROVENANCE'])
def get(url,timeout):
 assert url=='http://127.0.0.1:18082/provenance' and 0<timeout<=180
 if os.environ.get('FAKE_TIMEOUT')=='1': raise TimeoutError('fixture timeout')
 return R()
''')
 def case(name,record,actual=prov,timeout=False):
  import os
  p=t/(name+'.json');p.write_text(json.dumps(record));env=dict(os.environ,PYTHONPATH=str(t),FAKE_PROVENANCE=json.dumps(actual),FAKE_TIMEOUT=str(int(timeout)))
  run=subprocess.run([sys.executable,str(checker),'--acceptance',str(p),'--timeout','0.001'],env=env,text=True,capture_output=True)
  rows.append({'case':name,'exit_code':run.returncode,'accepted':run.returncode==0,'stderr_tail':run.stderr[-300:]})
 case('positive',acceptance)
 bad=dict(acceptance,status='DIFFERENCE_REPORT_ONLY');case('unaccepted_status',bad)
 case('instance_changed',acceptance,dict(prov,pid=9999))
 case('timeout',acceptance,timeout=True)
 payload.write_text('changed');case('evidence_changed',acceptance);payload.write_text('{"status":"DIFFERENCE_REPORT_ONLY"}\n')
 case('empty_evidence',dict(acceptance,evidence_file_sha256={}))
 minimal={k:prov[k] for k in ('encoder_device','encoder_fp16','index_float16')};case('missing_uuid_pid_run_source_binding',dict(acceptance,provenance=minimal),minimal)
 wrong=dict(prov,gpu_uuid='OTHER-GPU',run='/foreign/run',pid=-1,source_sha256='invalid',difference_report_sha256='b'*64);case('wrong_fixed_identity',dict(acceptance,provenance=wrong),wrong)
 text=launcher.read_text();loop=text[text.index('while kill -0 "$TRAIN_PID"'):text.index('printf \'%s\\n\' "$RUN"')]
 for backend,ret,guard,check,name in [('cpu',0,0,0,'cpu_alive_no_aux_checker'),('cpu',1,0,0,'cpu_dead_cancel_own'),('aux_gpu',0,0,1,'aux_unhealthy_cancel_own'),('aux_gpu',0,1,0,'aux_guard_dead_cancel_own')]:
  log=t/(name+'.log')
  script='''set -euo pipefail
TRAIN_PID=99; RETRIEVER_PID=77; GUARD_PID=88; SLURM_JOB_ID=42; ROOT=/fixture; calls=0
kill() { if [ "$1" = -TERM ]; then echo "TERM:$2" >> "$LOG"; return 0; fi; case "$2" in 99) calls=$((calls+1)); [ "$calls" -eq 1 ];; 77) return "$RET";; 88) return "$GUARD";; *) return 1;; esac; }
python() { echo "CHECK:$*" >> "$LOG"; return "$CHECK"; }
scancel() { echo "SCANCEL:$*" >> "$LOG"; }
sleep() { :; }
wait() { :; }
'''+loop
  import os
  out=subprocess.run(['bash','-c',script],env=dict(os.environ,BACKEND=backend,RET=str(ret),GUARD=str(guard),CHECK=str(check),LOG=str(log)),text=True,capture_output=True)
  rows.append({'case':name,'exit_code':out.returncode,'actions':log.read_text().splitlines() if log.exists() else [],'stdout':out.stdout.strip()})
(root/'work/aux-gate-review/fixtures-results.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
