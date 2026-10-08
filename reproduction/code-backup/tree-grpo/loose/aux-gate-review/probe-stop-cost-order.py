import pathlib,tempfile,subprocess,os,json,hashlib
r=pathlib.Path(__file__).resolve().parents[2];source=r/'work/train_tree_stage.sh';text=source.read_text();loop=text[text.index('while kill -0 "$TRAIN_PID"'):text.index('printf \'%s\\n\' "$RUN"')];finish=text[text.index('finish() {'):text.index('trap finish EXIT')];rows=[]
with tempfile.TemporaryDirectory() as tmp:
 run=pathlib.Path(tmp);env=dict(os.environ,RUN=tmp,LOG=str(run/'actions'))
 stub='''set -euo pipefail
BACKEND=hybrid; TRAIN_PID=99; RETRIEVER_PID=77; GUARD_PID=88; SLURM_JOB_ID=42; ROOT=/fixture; STAGE=two; start=1; AUX_ACCEPTANCE_SHA256=fixtureSHA; STAGE_PROVENANCE_SHA256=fixtureSTAGEsha; calls=0
kill() { if [ "$1" = -TERM ] || [ "$1" = -KILL ]; then echo "$1:$2" >> "$LOG"; return 0; fi; case "$2" in 99) calls=$((calls+1)); [ "$calls" -eq 1 ];; 77|88) return 0;; *) return 1;; esac; }
python() { echo 'fixture checker failure' >&2; return 7; }
timeout() { shift 3; "$@"; }
scancel() { test -s "$RUN/service-stop.json"; echo 'SCANCEL_SEES_STOP_RECORD' >> "$LOG"; }
sleep() { :; }
wait() { :; }
'''
 p=subprocess.run(['bash','-c',stub+loop],env=env,capture_output=True,text=True);stop=json.loads((run/'service-stop.json').read_text());assert p.returncode==1 and stop['checker_exit_code']==7 and stop['service_ok'] is False and stop['allocated_gpus']==8 and 'fixture checker failure' in (run/'hybrid-health-check.log').read_text() and (run/'actions').read_text().splitlines()==['SCANCEL_SEES_STOP_RECORD','-TERM:99']
 rows.append({'case':'caller_record_before_scancel','exit_code':p.returncode,'stop':stop,'actions':(run/'actions').read_text().splitlines(),'stderr_preserved':True})
 # Execute the actual finish function; simulated cleanup requires the termination file first.
 script='''set -euo pipefail
SLURM_JOB_ID=42; STAGE=two; start=1; RETRIEVER_PID=77; GUARD_PID=88; CPU_SAMPLER_PID=66
kill() { test -s "$RUN/termination-cost.json"; echo "cleanup:$*" >> "$LOG"; if [ "$1" = -0 ]; then return 1; fi; }
wait() { :; }
sleep() { :; }
'''+finish+'\nfinish\n'
 p=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True);termination=json.loads((run/'termination-cost.json').read_text());cost=json.loads((run/'cost.json').read_text());assert p.returncode==0 and termination['allocated_gpus']==cost['allocated_gpus']==8 and termination['exit_code']==cost['exit_code']==0
 rows.append({'case':'finish_termination_record_before_cleanup','exit_code':p.returncode,'termination':termination,'final_cost':cost})
(r/'work/aux-gate-review/stop-cost-order-fixtures.json').write_text(json.dumps({'scope':'actual extracted launcher monitor/finish shell with stub own PIDs/scancel; no actual scheduler/GPU/process termination','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n');print('2 stop/cost order fixtures passed')
