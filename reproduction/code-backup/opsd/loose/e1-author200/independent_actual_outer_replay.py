"""Replay exact actual3261 small originals. Does not issue first5 release."""
import argparse,ast,base64,hashlib,importlib.util,json,math,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
DRIVER=1533312
WORKERS=(1533910,1534229,1534230,1534231,1534232,1534233,1534234,1534235)
def need(ok,msg):
    if not ok:raise ValueError(msg)
def h(raw):return hashlib.sha256(raw).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True,type=Path);parser.add_argument('--count',required=True,type=int);parser.add_argument('--output',required=True,type=Path);a=parser.parse_args()
need(1<=a.count<=5,'Actual first1..5 only')
o=json.loads((a.snapshot/'observed-remote-original.json').read_text());s=json.loads((a.snapshot/'observation-summary.json').read_text())
need(h((a.snapshot/'observed-remote-original.json').read_bytes())==s['transport_original_sha256'],'Transport original hash changed')
files=a.snapshot/'original-files'
for name,rec in o['files'].items():
    raw=base64.b64decode(rec['base64'],validate=True);need(h(raw)==rec['sha256'] and len(raw)==rec['bytes'] and raw==(files/name).read_bytes(),'Exact copied bytes changed '+name)
def load(name):
    rec=o['files'][name];need(rec['before']['uid']==rec['after']['uid']==0 and rec['same_inode'] and rec['complete_original_at_start'] and rec['source_offset']==0,'Own full-prefix original required '+name)
    return (files/name).read_text()
gate_path=HERE.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py';need(h(gate_path.read_bytes())=='c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb','Frozen gate changed')
sp=importlib.util.spec_from_file_location('independent_frozen_gate',gate_path);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
cfg=json.loads(load('resolved-profile.json'));bind=json.loads(load('bindings/manifest.json'));cl=json.loads(load('bindings/import_closure.json'))
cfg['actor_rollout_ref']['actor']['optim']['total_training_steps']=200;cfg['critic']['optim']['total_training_steps']=200;rec=cl['modules']['reproduction.baselines.e1_import_bootstrap']
cfg['actor_rollout_ref']['actor']['e1_import_checker']={'checker_path':rec['path'],'checker_sha256':rec['sha256'],'closure_path':bind['files']['import_closure']['path'],'closure_sha256':bind['files']['import_closure']['sha256']}
config_sha=h(g.canonical(cfg));need(config_sha=='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60','Exact runtime config differs')
streams={};events={};evaluations={}
for name in ('train.log','ray-stdout-pid-1533312.out'):
    text=load(name);rows=[];evals=[]
    for line in text.splitlines():
        if 'OPSD_E1_OUTER_JSON=' in line:
            d=g.parse_log_marker(line,'OPSD_E1_OUTER_JSON=',driver_pid=DRIVER,job_id='3261',worker_pids=WORKERS)
            need(d['source_commit']==SOURCE and d['config_sha256']==config_sha and d['driver_pid']==DRIVER and d['job_id']=='3261' and d['actual_outer_updates']==d['global_step'],'Outer exact binding differs')
            g.validate_outer(d['metrics'],world=8,profile='author200');m=d['metrics'];need(type(m['actor/lr']) in (int,float) and math.isfinite(m['actor/lr']) and m['actor/lr']==5e-6,'Actual learning rate differs')
            need(m['rollout/trajectories']==512 and m['rollout/context_mismatches']==0,'Actual trajectory count/context differs')
            for rank in range(8):
                for key,expect in [('teacher/update_device_is_cuda_per_rank',1),('teacher/cuda_update_entries_cumulative_per_rank',d['global_step'])]:need(m[key+'/rank_'+str(rank)]==expect,'Actual teacher update device/count differs')
            rows.append(d)
        if 'BASELINE_EVALUATION_METRICS_JSON=' in line:
            d=g.parse_log_marker(line,'BASELINE_EVALUATION_METRICS_JSON=',driver_pid=DRIVER,job_id='3261',worker_pids=WORKERS);g.validate_evaluation(d,int(d['val/qa_count/all']));evals.append(d)
    need([d['global_step'] for d in rows]==list(range(1,a.count+1)),'Actual contiguous completed rows differ '+name)
    events[name]=rows;evaluations[name]=evals;streams[name]={'original_sha256':o['files'][name]['sha256'],'bytes':o['files'][name]['bytes'],'unchanged_during_read':o['files'][name]['unchanged_during_read']}
need(events['train.log']==events['ray-stdout-pid-1533312.out'],'Forwarded/direct actual outer objects differ')
need(evaluations['train.log']==evaluations['ray-stdout-pid-1533312.out'],'Forwarded/direct actual evaluation objects differ')
need(evaluations['train.log'][0]['val/qa_count/all']==512 and evaluations['train.log'][0]['val/evaluated_count/nq']==evaluations['train.log'][0]['val/evaluated_count/hotpotqa']==256,'Actual initialized512 counts differ')
worker_proofs=[]
for rank,pid in enumerate(WORKERS):
    name='ray-stdout-pid-'+str(pid)+'.out';text=load(name);process=o['processes'][str(pid)]
    need(process['uids']==[0,0,0,0] and process['in_observed_job'] and process['state'] in ('R','S','D','I'),'Actual GPU worker foreign/stale')
    need(text.count('Total steps: 200, num_warmup_steps: 0')==1,'Actual worker schedule differs')
    kwargs=[ast.literal_eval(l.split('kwargs: ',1)[1]) for l in text.splitlines() if l.startswith('kwargs: ')]
    need(len(kwargs)==1 and kwargs[0]['n']==1 and kwargs[0]['temperature']==1.1 and kwargs[0]['top_p']==.95 and kwargs[0]['top_k']==20 and kwargs[0]['max_tokens']==500,'Actual worker SamplingParams differ')
    apps=[r for r in o['unfiltered_application_csv_rows'] if int(r[1])==pid];need(len(apps)==1,'One observed GPU per model worker required')
    worker_proofs.append({'pid':pid,'start_ticks':process['start_ticks'],'observed_gpu_uuid':apps[0][0].strip(),'stdout_sha256':o['files'][name]['sha256'],'rank_mapping_pending_native5':True})
need(len({d['observed_gpu_uuid'] for d in worker_proofs})==8,'Actual eight model GPUs not distinct')
rows=[]
for e in events['train.log']:
    m=e['metrics'];rows.append({'outer':e['global_step'],'actual_optimizer_calls_per_rank':[m['actor/optimizer_steps/rank_'+str(i)] for i in range(8)],'actual_samples_per_rank':[m['actor/samples/rank_'+str(i)] for i in range(8)],'reward':m['critic/rewards/mean'],'finish':m['env/finish_ratio'],'valid_actions':m['env/ratio_of_valid_action'],'clipped_loss':m['actor/opsd_loss'],'pre_clip_global_grad_norm':m['actor/grad_norm'],'lr':m['actor/lr'],'step_seconds':m['timing_s/step'],'generation_seconds':m['timing_s/gen'],'update_seconds':m['timing_s/update_actor']})
out={'status':'PASS_ACTUAL_OUTER_LOG_CONFIG_AND_WORKER_REPLAY_ONLY','source':SOURCE,'job_id':'3261','driver_pid':DRIVER,'config_sha256':config_sha,'snapshot_original_sha256':s['transport_original_sha256'],'observed_finished_utc':s['finished_utc'],'actual_completed_outer_iterations':a.count,'streams':streams,'rows':rows,'actual_evaluations':evaluations['train.log'],'actual_worker_proofs':worker_proofs,'first5_implementation_acceptance':False,'first5_effect_improvement_proven':False,'scope':'Strict exact frozen-source log/config/worker firstN replay only. Native8rank mapping/Adam80/full-state reload/HF/source/import/resource/state/behavior review required before real first5 acceptance. Different training batches do not provide same-question effect comparison. Negative clipped surrogate is not trueKL. No invented reward-rise threshold.','observer_errors_retained':o['errors']}
with a.output.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'status':out['status'],'actual_completed_outer_iterations':a.count,'config_sha256':config_sha,'report_sha256':h(a.output.read_bytes()),'rows':rows},sort_keys=True))
