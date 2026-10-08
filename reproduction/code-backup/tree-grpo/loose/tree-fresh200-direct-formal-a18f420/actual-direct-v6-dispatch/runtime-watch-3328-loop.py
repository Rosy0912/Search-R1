import datetime,json,pathlib,subprocess,sys,time
base=pathlib.Path(__file__).resolve().parent
number=2
last=datetime.datetime.fromisoformat(json.loads((base/'runtime-watch-3328-observation-01.json').read_text())['observed_at']).timestamp()
previous=None
terminal={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def summary(o):
    files=o.get('files',{});logs=o.get('anchored_log_metadata',{});stage=logs.get('stage.log',{})
    counts={k:v.get('count') for k,v in files.items() if k not in {'resource-samples.jsonl','output/resource-samples.jsonl'}}
    phases={k:v.get('exists') for k,v in logs.items()}
    recent={k:[r.get('metadata') for r in v.get('recent',[])] for k,v in files.items() if v.get('count') and k!='output/fresh-initialization.json'}
    policy=o.get('policy_safety_events',{}).get('latest_by_rank',{})
    roll=o.get('rolling_completed_native_metadata',{})
    roll_json=[{'name':r['name'],'metadata':r.get('metadata'),'sha256':r.get('sha256')} for r in roll.get('json_metadata',[])]
    resources=files.get('resource-samples.jsonl',{}).get('recent',[])[-1:]
    guard=[{k:r.get(k) for k in ['epoch','gpu_count','foreign_process_count','max_foreign_memory_fraction','max_foreign_sm_upper']} for r in resources]
    return {'observed_at':o['observed_at'],'observation':number,'slurm':{k:v for k,v in o.get('slurm_identity',{}).get('fields',{}).items() if k in {'JobState','ExitCode','RunTime','Reason'}},'startup_phases':phases,'published_counts':counts,'progress_numeric':stage.get('recent_progress_numeric',{}),'epoch_step_count':stage.get('epoch_step_count'),'exception_classes':{k:v.get('exception_classes') for k,v in logs.items() if v.get('exception_classes')},'native_training_metadata':recent,'policy_latest_by_rank':policy,'rolling_json':roll_json,'rolling_file_count':len(roll.get('files',[])),'background_all_preserved':all(r.get('starttime_preserved') for r in o.get('original_eight_pid_identities',[])),'guard_latest':guard}
def signature(s):
    value={k:v for k,v in s.items() if k not in {'observed_at','observation','guard_latest'}}
    value['slurm']={k:v for k,v in value['slurm'].items() if k!='RunTime'}
    value['policy_latest_by_rank']={k:{kk:vv for kk,vv in v.items() if kk not in {'mtime','name'}} for k,v in value['policy_latest_by_rank'].items()}
    return json.dumps(value,sort_keys=True)
while True:
    remaining=last+61-time.time()
    if remaining>0:time.sleep(remaining)
    response=subprocess.run([sys.executable,str(base/'runtime-watch-3328-poll.py'),str(number)],capture_output=True,text=True,timeout=70)
    (base/('runtime-watch-3328-poll-%02d.stdout'%number)).write_text(response.stdout)
    (base/('runtime-watch-3328-poll-%02d.stderr'%number)).write_text(response.stderr)
    path=base/('runtime-watch-3328-observation-%02d.json'%number)
    if response.returncode or not path.exists():
        last=time.time()
        print(json.dumps({'observation':number,'connection_or_observation_status':'UNKNOWN','returncode':response.returncode,'diagnostic_metadata_saved':True}),flush=True)
        number+=1
        continue
    value=json.loads(path.read_text());last=datetime.datetime.fromisoformat(value['observed_at']).timestamp()
    current=summary(value);key=signature(current)
    if key!=previous:
        print(json.dumps(current,sort_keys=True),flush=True);previous=key
    status=current['slurm'].get('JobState')
    run_results=value.get('files',{}).get('run-result.json',{}).get('recent',[])
    failed=any(r.get('metadata',{}).get('status')=='FAILED' for r in run_results)
    if status in terminal or failed:
        print(json.dumps({'watch_completed':'ACTUAL_FAULT','observation':number,'positive_slurm_captured':bool(value.get('positive_terminal_slurm_raw_stdout'))}),flush=True)
        break
    latest=[r for r in value.get('rolling_completed_native_metadata',{}).get('json_metadata',[]) if r['name'].endswith('/latest.json')]
    if any(r.get('metadata',{}).get('status')=='COMPLETE_NATIVE_ROLLING_STATE_SAVED' and r.get('metadata',{}).get('completed_outer_updates',0)>=1 for r in latest):
        print(json.dumps({'watch_completed':'FIRST_NATIVE_COMPLETED_OUTER_AND_ROLLING_SAVED','observation':number}),flush=True)
        break
    number+=1
