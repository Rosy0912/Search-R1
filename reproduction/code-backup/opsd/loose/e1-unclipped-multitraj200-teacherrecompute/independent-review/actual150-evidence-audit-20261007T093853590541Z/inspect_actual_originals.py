"""Offline actual150 evidence audit; no main/remote/model/Slurm execution."""
from pathlib import Path
import base64, hashlib, json, math, re, types

H = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute')
D = H / 'checkpoint150-collection-once-v1-original'
R = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
S = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
C = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
M = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
A = 'd1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
PIN = '56d096390abc92cfb1f58b06f49f2acb2eaafb75b5f0bad41fcf39edbd5c7e1f'

def sha(b): return hashlib.sha256(b).hexdigest()
def parse(b):
    def pairs(rows):
        out = {}
        for k,v in rows:
            assert k not in out, ('duplicate',k)
            out[k] = v
        return out
    out = json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    json.dumps(out,allow_nan=False)
    return out
def load(name): return parse((D/name).read_bytes())
def canonical(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def eq(a,b): assert canonical(a)==canonical(b)
def decode(record):
    raw = base64.b64decode(record['base64'],validate=True)
    assert len(raw)==record['bytes'] and sha(raw)==record['sha256']
    return raw
def public_record(record,installed=False):
    assert all(type(record[k])is int and record[k]>=0 for k in ('bytes','device','inode','uid','mode','nlink','mtime_ns','ctime_ns'))
    assert record['uid']==0 and record['nlink']==(2 if installed else 1)
    # record() stores permission-only S_IMODE; regular-file provenance is the
    # actual approved collector's opened/S_ISREG guard, not a fabricated mode bit.
    assert 0<=record['mode']<=0o7777 and re.fullmatch('[0-9a-f]{64}',record['sha256'])
    assert Path(record['physical_path']).is_absolute() and '..'not in Path(record['physical_path']).parts
    if 'base64'in record: decode(record)

raw=(D/'stdout.original').read_bytes()
assert len(raw)==41129553 and sha(raw)=='8d32f191b6b9aeca727e9510fc451346fc1b9b38aba414e10bd5808ba6fd6e03'
v=parse(raw);wait=load('wait.original.json');intent=load('intent.original.json');sent=load('stdin.original.json')
assert wait['waited']is True and type(wait['returncode'])is int and wait['returncode']==0 and wait['stdin_complete']is True and wait['error']is None
assert not (D/'stderr.original').read_bytes()
assert sent['complete']is True and sent['sha256']==PIN and sent['bytes']==28148
code=(D/'collector-source.original.py').read_bytes();assert len(code)==28148 and sha(code)==PIN
assert sha((D/'caller-source.original.py').read_bytes())=='415a9afd7115f736f294e5278d6865c71a3b4f94c965f2d3c2a424495d13479e'
assert intent['collector_sha256']==PIN and intent['requested_steps']==[150] and '--steps' in intent['argv'][-1] and PIN in intent['argv'][-1]
assert sha((D/'software-peer.original.json').read_bytes())=='21caf17b7b8a787ecefbcbee7c8e18d2cc14658daff844cfe78f228e2e4e06ed'
assert v['requested_steps']==[150] and v['job_id']=='3304' and v['source_commit']==S and set(v['snapshots'])=={'150'}
assert v['status']=='READONLY_ACTUAL_SAVED_CHECKPOINT_LOG_AND_FILE_SNAPSHOTS_REQUIRES_INDEPENDENT_REVIEW' and 'error'not in v
mod=types.ModuleType('independent_actual150_pure_readers');mod.__file__=str(D/'collector-source.original.py')
exec(compile(code,mod.__file__,'exec',dont_inherit=True,optimize=0),mod.__dict__)
assert len(v['source_files'])==19 and set(v['source_files'])==set(mod.SOURCE_FILES)
for relative,r in v['source_files'].items():
    public_record(r);assert r['physical_path']==str(mod.REPO/relative) and r['sha256']==mod.SOURCE_FILES[relative]
    assert decode(r)==(H.parent/'opsd-unclipped-multitraj200-teacherrecompute'/relative).read_bytes()
assert len(v['files'])==37
for key,r in v['files'].items(): public_record(r,installed=(key==str(mod.FSDP_PATH)))
cfg=parse(decode(v['files'][R+'/full_state/global_step_5/resolved-config.json']))
manifest=parse(decode(v['files'][R+'/bindings/data_manifest.json']))
assert v['files'][R+'/full_state/global_step_5/resolved-config.json']['sha256']==C
assert v['files'][R+'/bindings/data_manifest.json']['sha256']==M
assert manifest['splits']['validation']['rows']==512 and manifest['splits']['validation']['source_counts']=={'nq':256,'hotpotqa':256}
assert manifest['splits']['stage']['rows']==64 and manifest['splits']['stage']['source_counts']=={'nq':32,'hotpotqa':32}
assert cfg['data']['val_files']==cfg['baselines']['holdout']['data_root']+'/validation.parquet'
assert cfg['baselines']['opsd_e1']['stage_files']==cfg['baselines']['holdout']['data_root']+'/stage.parquet'
assert cfg['baselines']['holdout']['manifest_sha256']==M and cfg['baselines']['opsd_e1']['profile']=='multitraj200'
actor=cfg['actor_rollout_ref']['actor'];roll=cfg['actor_rollout_ref']['rollout'];dist=cfg['baselines']['distillation']
assert cfg['data']['train_batch_size']==512 and roll['n_agent']==5 and actor['ppo_mini_batch_size']==256 and actor['ppo_micro_batch_size']==32 and actor['ppo_epochs']==1
assert roll['log_prob_micro_batch_size']==64 and roll['temperature']==1 and roll['top_p']==1 and roll['top_k']==-1 and roll['n']==1
assert dist['component_clip']is None and dist['beta']==0 and actor['kl_loss_coef']==0 and actor['use_kl_loss']is False and cfg['baselines']['method']=='opsd'
assert actor['optim']['lr_warmup_steps']==57 and actor['optim']['total_training_steps']==200 and actor['optim']['lr']==1e-6
for name,cmd in v['commands'].items():
    assert cmd['waited']is True and type(cmd['returncode'])is int and cmd['returncode']==0 and cmd['stderr']==''
    if 'git_head'in name: assert cmd['stdout'].strip()==S
    if 'git_clean'in name: assert cmd['stdout']==''
assert all(t in v['commands']['slurm']['stdout']for t in ('JobId=3304 ','UserId=root(0)','JobState=RUNNING','Restarts=0','NodeList=h100-10','gres:gpu:h100:8'))
for phase in ('before','after'):
    assert v['driver_'+phase]['pid']==mod.DRIVER[0] and v['driver_'+phase]['start_ticks']==mod.DRIVER[1]
    assert set(v['workers_'+phase])=={str(i)for i in range(8)}
    for row,pid,start in [(v['driver_'+phase],*mod.DRIVER)]+[(v['workers_'+phase][str(i)],*mod.WORKERS[i])for i in range(8)]:
        assert type(row['pid'])is int and row['pid']==pid and row['start_ticks']==start and row['uid']==0
        assert row['cgroup']=='1:freezer:/slurm_h100-10/uid_0/job_3304/step_0\n0::/system.slice/slurmd.service\n'
        assert row['stat_original'].startswith(str(pid)+' (') and row['stat_original'].rsplit(')',1)[1].split()[19]==str(start)
gate=mod.load_gate(decode(v['source_files']['search_r1/baselines/checkpoint_gate.py']))
warn=decode(v['files'][str(mod.FSDP_PATH)])
driver=decode(v['files']['driver-current-prefix.out']);aggregate=decode(v['files']['train.log'])
events,partial=mod.parse_events(driver,gate,warn);te,tp=mod.parse_events(aggregate,gate,warn)
progress,rows=mod.progress(events,gate,150)
assert len(rows)==150 and progress['optimizer_calls_per_rank']==[1500.0]*8
agg=[e for e in te if e['marker']=='OPSD_E1_OUTER_JSON='];assert len(agg)==150
eq([e['value']for e in rows],[e['value']for e in agg]);eq(progress,v['snapshots']['150']['metrics'])
ev=mod.evaluations_and_console(driver,events,rows,gate,150);eq(ev,v['snapshots']['150']['evaluations'])
metrics_events=[e for e in events if e['marker']=='BASELINE_EVALUATION_METRICS_JSON='];agg_eval=[e for e in te if e['marker']=='BASELINE_EVALUATION_METRICS_JSON=']
assert len(metrics_events)==len(agg_eval)==8;eq([e['value']for e in metrics_events],[e['value']for e in agg_eval])
fixed=[]
for i,e in enumerate(metrics_events):
    metrics=e['value'];count=512 if i%2==0 else 64;step=[0,5,50,50,100,100,150,150][i]
    assert metrics['val/qa_count/all']==count
    source_count=count//2
    for source in ('nq','hotpotqa'): assert metrics['val/evaluated_count/'+source]==metrics['val/qa_count/'+source]==source_count
    arithmetic={}
    for name in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):
        weighted=sum(metrics['val/'+name+'/'+s]*source_count for s in ('nq','hotpotqa'))/count
        assert math.isclose(weighted,metrics['val/'+name+'/all'],rel_tol=1e-12,abs_tol=1e-12)
        if name!='f1':
            for source,n in [('nq',source_count),('hotpotqa',source_count),('all',count)]:
                value=metrics['val/'+name+'/'+source]*n
                assert math.isclose(value,round(value),rel_tol=1e-10,abs_tol=1e-7)
        arithmetic[name]={'weighted_all':weighted,'sum_or_equivalent':weighted*count}
    for s in ('nq','hotpotqa'):assert metrics['val/test_score/'+s]==metrics['val/em/'+s]
    # Frozen source only prints the full precision JSON each validation; plain
    # console step0/5/50/100/150 prints rounded metrics with explicit stage prefix.
    expected_prefix='' if count==512 else 'opsd_stage/'
    for log in (driver,aggregate):
        lines=[line for line in log.splitlines() if line.startswith(('step:'+str(step)+' - ').encode())]
        assert len(lines)==1
        displayed={}
        for field in lines[0].decode().split(' - ')[1:]:
            key,sep,value=field.rpartition(':');assert sep and key not in displayed;displayed[key]=value
        for key,value in metrics.items(): assert displayed.get(expected_prefix+key)==format(float(value),'.3f')
    fixed.append(dict(cohort='dev512'if count==512 else 'stage64',step=step,count=count,em_correct=int(round(metrics['val/em/all']*count)),metrics=metrics,arithmetic=arithmetic,event_sha256=e['sha256']))
store=v['snapshots']['150']['stored'];assert store['member_count']==31 and store['native_reload']is None and store['whole_hf_load']is None
actorroot=R+'/checkpoints/actor/global_step_150';driverroot=R+'/full_state/global_step_150';algpath=R+'/checkpoints/algorithm/global_step_150/state.json'
index=parse(decode(v['files'][actorroot+'/model.safetensors.index.json']));assert len(index['weight_map'])==435 and len(set(index['weight_map'].values()))==3
paths={actorroot+'/'+n for n in mod.HF_SMALL|set(index['weight_map'].values())}|{actorroot+'/training_state/rank_'+str(i)+ext for i in range(8)for ext in('.json','.pt')}|{driverroot+'/driver.json',driverroot+'/driver.pt',algpath}
assert len(paths)==31 and len({(v['files'][p]['device'],v['files'][p]['inode'])for p in paths})==31
for i in range(8):
    p=actorroot+'/training_state/rank_'+str(i)+'.json';r=parse(decode(v['files'][p]));eq(r,store['rank_reports'][i]);assert r['status']=='FULL_STATE_SERIALIZED_OK' and r['step']==150 and r['rank']==i and r['world_size']==8 and r['pid']==mod.WORKERS[i][0]
    assert r['model_dtensor_entries']==435 and r['optimizer_state_entries']==37 and r['scheduler_last_epoch']==150 and r['teacher']['counters']['cuda_update_entries']==150
    assert r['path']==actorroot+'/training_state/rank_'+str(i)+'.pt' and r['sha256']==r['state_sha256']==v['files'][r['path']]['sha256']
    prior=parse((H/'job3304-first5-complete-originals-v3/raw/checkpoints/actor/global_step_5/training_state'/('rank_'+str(i)+'.json')).read_bytes())
    eq(r['object_ids'],prior['object_ids'])
    for key in ('asset_directory','asset_files','model_content_digest','buffer_content_digest','parameter_count','model_bytes','model_config'): eq(r['teacher'][key],prior['teacher'][key])
eq(parse(decode(v['files'][driverroot+'/driver.json'])),store['driver_report']);dr=store['driver_report'];assert dr['status']=='DRIVER_STATE_SERIALIZED_OK' and dr['step']==150 and dr['pid']==mod.DRIVER[0] and dr['job_id']=='3304' and dr['state_sha256']==v['files'][driverroot+'/driver.pt']['sha256']
eq(parse(decode(v['files'][algpath])),store['algorithm']);assert store['algorithm']['method']=='pure_opsd' and store['algorithm']['outcome_rewards_train_policy']is False and store['algorithm']['trajectory_archive']['completed_outer']==150
for x in store['actual_observed_rank_console_records']:
    e=x['event'];eq(e['value'],store['rank_reports'][e['value']['rank']]);rlog=driver if x['log_origin']=='driver'else aggregate;b=base64.b64decode(e['raw_base64'],validate=True);assert rlog[e['offset']:e['offset']+e['bytes']]==b and sha(b)==e['sha256']
receipt_raw=decode(v['files'][R+'/acceptance-5-receipt.json']);assert sha(receipt_raw)==A and receipt_raw==(H/'job3304-release-live-originals-v1/acceptance-5-receipt.json').read_bytes();receipt=parse(receipt_raw)
public_raw=(H/'job3304-pre-gpu-actual-originals/bindings/public.json').read_bytes();assert sha(public_raw)=='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c';gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],parse(public_raw))
rp=receipt['payload'];assert rp['source_commit']==S and rp['job_id']=='3304' and rp['config_sha256']==C and rp['actual_outer_updates']==rp['global_step']==5 and rp['next_global_step']==6 and rp['horizon']==200 and rp['warmup_steps']==57
eq(cfg['actor_rollout_ref']['actor']['distillation'],store['algorithm']['distillation'])
priorreview=parse((H/'ACTUAL3304_CHECKPOINT50_100_STORED_SERIALIZED_INDEPENDENT_REVIEW.json').read_bytes())
for s,event_index in [('50',2),('100',4)]:
    assert metrics_events[event_index]['value']['val/em/all']==priorreview['snapshots'][s]['dev']['EM'] and metrics_events[event_index+1]['value']['val/em/all']==priorreview['snapshots'][s]['stage']['EM']
print(json.dumps({'status':'ACTUAL150_OFFLINE_EVIDENCE_AUDIT_FINISHED_NO_NATIVE_RELOAD','saved_records31':{p:{k:v['files'][p][k]for k in v['files'][p]if k!='base64'}for p in sorted(paths)},'saved_bytes':sum(v['files'][p]['bytes']for p in paths),'source19':{p:{k:r[k]for k in r if k!='base64'}for p,r in v['source_files'].items()},'file_records':len(v['files']),'complete_prefix':{'driver':len(rows),'aggregate':len(agg),'not_closed_writers':True},'optimizer_calls_per_rank':progress['optimizer_calls_per_rank'],'fixed_evaluations8':fixed,'checkpoint150_training_batch_metrics':{'reward':rows[149]['value']['metrics']['critic/score/mean'],'loss':rows[149]['value']['metrics']['actor/opsd_loss'],'gradient_norm':rows[149]['value']['metrics']['actor/grad_norm']},'rank_json_count':8,'visible_rank_console_count':len(store['actual_observed_rank_console_records']),'signed5_sha256':A,'pure_opsd_config_verified':True,'RSA5_verified':True,'object_ids_and_fixed_teacher_report_join5_to150':True,'native_reload':None,'whole_hf_load':None,'rng_contents':None,'sampler_contents':None,'final200':None,'resource_release':None},sort_keys=True,allow_nan=False))
