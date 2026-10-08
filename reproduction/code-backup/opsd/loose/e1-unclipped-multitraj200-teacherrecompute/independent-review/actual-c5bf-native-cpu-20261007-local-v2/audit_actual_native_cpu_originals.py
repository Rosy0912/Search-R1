import ast,base64,datetime,hashlib,json,re,types
from pathlib import Path
H=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute')
D=H/'c5bf-native-cpu-actual-originals-v1';L=H/'c5bf-native-cpu-launch-execution-v1-original';B=H/'final200-eval-native-cpu-ee31da655703'
def record(p):
 raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def j(p):return json.loads(p.read_bytes())
def need(x,s):
 if not x:raise ValueError(s)
def zero(r):need(type(r['returncode']) is int and r['returncode']==0 and r['waited'] is True,'True waited0 required')
def literal(p,n):
 t=ast.parse(p.read_bytes())
 return next(ast.literal_eval(a.value) for a in t.body if isinstance(a,ast.Assign) and any(isinstance(x,ast.Name) and x.id==n for x in a.targets))
root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
wire=j(D/'transport.stdout.original');zero(j(D/'transport.wait.original.json'))
need(wire['node']=='h100-10' and wire['uid']==0 and len(wire['files'])==20,'Actual wire identity/count')
allraw={};inodes=set();total=0
for n,r in wire['files'].items():
 raw=(D/'raw'/n).read_bytes();need(raw==base64.b64decode(r['base64'],validate=True),'Wire/local exact bytes '+n)
 need(hashlib.sha256(raw).hexdigest()==r['sha256'] and len(raw)==r['bytes'],'Wire SHA/length '+n)
 need(r['physical_path']==root+'/runs/formal-3304/evidence/'+n,'Actual fixed physical path '+n)
 need(r['uid']==0 and r['nlink']==1 and (r['device'],r['inode']) not in inodes,'Physical ownership/unique inode '+n)
 inodes.add((r['device'],r['inode']));total+=len(raw);allraw[n]={k:v for k,v in r.items() if k!='base64'}
need(total==wire['total_member_bytes']==479115,'All original bytes total')
launch=D/'raw/c5bf-native-cpu-launch-once-v1';native=D/'raw/c5bf-native-cpu-preparation-once-v1'
report=j(native/'native-cpu-preparation.json');comp=j(native/'controller-complete.original.json');intent=j(native/'intent.original.json');wi=j(native/'child-wait.original.json');cs=j(native/'child-start.original.json');pi=j(native/'child-popen.original.json');ci=j(native/'child-invocation.original.json')
zero(wi);zero(j(launch/'srun-wait.original.json'));zero(j(L/'exit.original.json'))
need(report['passed'] is True and comp['cpu_preparation_qualified'] is True,'Actual limited report/completion not pass')
need(report['eval_source_commit']=='c5bf4c8db33562a259fa227a940dc69543901b98' and report['stored_training_source_commit']=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6','Source exact commits')
for r in [report,comp,intent]:need(r['actual_eval_job_id'] is None,'No actual evaluation job')
for k in ['actual_input_manifest','actual_model_directory','actual_output_directory','actual_stage_manifest']:need(report[k] is None,'Future field must remain null')
for k in ['evaluation_started','gpu_execution_verified','model_loaded']:need(report[k] is False,'No model/GPU/evaluation qualification')
need(report['evaluation_eligibility']==comp['evaluation_eligibility']=='BLOCKED_ACTUAL200_AND_REVIEWED_STAGE_REQUIRED','Blocked eligibility')
execs=[intent['execution'],comp['execution'],report['execution']]
for e in execs:
 need(e['execution_job_id']=='3304' and e['execution_step_id']=='2' and e['stored_training_job_id']=='3304' and e['node']=='h100-10' and e['uid']==0 and e['uids']==[0]*4,'Actual native job2/UID/node')
 need(type(e['pid']) is int and type(e['start_ticks']) is int and re.search(r'(?:^|/)job_3304/step_2(?:/|$)',e['cgroup'],re.M),'Actual same path job/step')
need(intent['execution']==comp['execution'],'Controller identity stable')
need(intent['execution']['pid']==pi['parent_pid']==cs['parent_pid']==wi['parent_pid']==2752194,'Controller parent identity')
need(report['execution']['pid']==pi['pid']==cs['pid']==wi['pid']==2752196 and report['execution']['start_ticks']==cs['start_ticks']==1836550772 and report['execution']['cgroup']==cs['cgroup'] and cs['uids']==[0]*4,'Checker child identity')
need(comp['child_wait']==wi and wi['process_identity_error'] is None,'Actual child waited join')
need(ci['argv']==pi['argv']==wi['argv']==report['execution']['actual_process_argv'],'Actual invocation/Popen/wait/report argv join')
need(ci['CUDA_VISIBLE_DEVICES']=='' and ci['PYTHONDONTWRITEBYTECODE']=='1','Actual CPU only child env')
need(wi['argv'][wi['argv'].index('-c')+1]==literal(B/'run_eval_native_cpu_once.py','CHILD_BOOTSTRAP'),'Exact child bootstrap')
need(report['runtime']['cuda_is_initialized'] is False and report['runtime']['torch_version']=='2.4.0+cu121' and report['runtime']['python_version'].startswith('3.10.12'),'Actual native package/CUDA state')
need(report['runtime']['python']==report['execution']['python'],'Interpreter consistency')
for role,p in [('report',native/'native-cpu-preparation.json')]:need(comp[role]['sha256']==record(p)['sha256'] and comp[role]['bytes']==p.stat().st_size,'Completion file SHA')
need(j(native/'child.stdout.original')['report']==comp['report'],'Actual checker stdout report join')
controller_stdout=j(launch/'srun.stdout.original');need(controller_stdout['sha256']==record(native/'controller-complete.original.json')['sha256'] and controller_stdout['bytes']==(native/'controller-complete.original.json').stat().st_size,'Controller stdout completion SHA join')
for role,filename in [('import_closure','native-cpu-preparation.json.closure.json'),('partial_config','native-cpu-preparation.json.partial-config.json'),('source_physical_records','native-cpu-preparation.json.source-physical-records.json')]:
 r=report[role];p=native/filename;need(r['sha256']==record(p)['sha256'] and r['bytes']==p.stat().st_size,'Native referenced artifact exact '+role)
 need(r['physical_path']==root+'/runs/formal-3304/evidence/c5bf-native-cpu-preparation-once-v1/'+filename and r['uid']==0,'Native artifact physical '+role)
sp=H/'final200-eval-source-publication-v1-execution-original';payload=j(sp/'payload.original.json');inv={k:payload['inventory'][k] for k in ['version','kind','source_commit','parent_commit','member_count','files']};inv['remote_repository']=root+'/repo-final200-eval-c5bf4c8db335';rawinv=json.dumps(inv,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n';invsha=hashlib.sha256(rawinv).hexdigest()
need(invsha=='fd8fa25d8321a4cd11b9faa5d8fdbd5f1692172a9e51ea5eba447c3565b8cc0d' and report['source_inventory']['sha256']==intent['source_inventory']['sha256']==invsha,'Exact7-key inventory reconstruction from original publication bytes')
need(inv['source_commit']==report['eval_source_commit'] and inv['parent_commit']==report['stored_training_source_commit'] and inv['member_count']==857 and len(inv['files'])==857,'Full inventory source parent/count')
physical=j(native/'native-cpu-preparation.json.source-physical-records.json');need(set(physical)==set(inv['files']) and report['source_tree_members']==857,'857 physical records exact member set');ids=set()
for n,r in physical.items():
 need(r['physical_path']==inv['remote_repository']+'/'+n and r['uid']==0,'Source physical path/owner')
 need(r['sha256']==inv['files'][n]['sha256'] and r['bytes']==inv['files'][n]['bytes'],'Source fullSHA/bytes '+n)
 need((r['device'],r['inode']) not in ids,'Source unique sampled inode');ids.add((r['device'],r['inode']))
closure=j(native/'native-cpu-preparation.json.closure.json')['modules'];need(len(closure)==report['module_count']==94 and report['verified_live_function_code_count']==837,'Actual emitted94/837 counts');need(set(report['required_modules'])<=set(closure),'Required actual loaded module coverage');project_modules={}
for n,r in closure.items():
 p=r['path']
 if p.startswith(inv['remote_repository']+'/'):
  rel=p[len(inv['remote_repository'])+1:];need(rel in inv['files'] and r['sha256']==inv['files'][rel]['sha256'],'Project imported source exact inventory '+n);project_modules[n]=r
 elif p.startswith(root+'/repo-'):raise ValueError('Wrong project source '+n)
need(closure['search_r1.baselines.final200_evaluation']['sha256']=='2d927c8bb87b55ea140d3d08931528c5b5fcf604902ccf67902c9830f1b4fd52','Actual optin final200 module source')
cfg=j(native/'native-cpu-preparation.json.partial-config.json');config=report['configuration'];future=config['future_interpolations'];need(len(future)==5 and set(future)==set(config['future_values']) and all(v is None for v in config['future_values'].values()),'Five actual future env slots null')
def get(path):
 r=cfg
 for k in path.split('.'):r=r[k]
 return r
for path,val in config['checked_fields'].items():need(get(path)==val,'Partial checked config '+path)
for path,val in future.items():need(get(path)==val and isinstance(val,str) and val.startswith('${oc.env:'),'Unresolved future config remains unchanged '+path)
need(config['runtime_resolved_config'] is False and config['retrieval_service_verified'] is False and cfg['trainer']['val_only'] is True and cfg['trainer']['total_training_steps']==0 and cfg['baselines']['opsd_e1']['profile'] is None and cfg['baselines']['opsd_e1']['enabled'] is False,'Partial only evalmode and disabledtrainprofile')
need(cfg['actor_rollout_ref']['actor']['entropy_coeff']==0 and cfg['actor_rollout_ref']['actor']['use_kl_loss'] is False and cfg['actor_rollout_ref']['actor']['kl_loss_coef']==0 and cfg['algorithm']['kl_ctrl']['kl_coef']==0,'No PG/referenceKL candidate route declarations')
# Literal dictionaries are authored config only; no claim of an evaluated model or live retrieval budget.
launchint=j(launch/'intent.original.json');after=j(launch/'after.original.json');sw=j(launch/'srun-wait.original.json');srunpi=j(launch/'srun-popen.original.json');lr=j(launch/'result.original.json')
need(launchint['argv']==sw['argv']==srunpi['argv'],'Actual srun command same graph');need(sw['pid']==srunpi['pid']==j(launch/'srun-start.original.json')['pid']==2752181 and sw['parent_pid']==srunpi['parent_pid']==2752176,'Actual srun PID/parent');need(sw['start_record_error'] is None,'srun identity observation succeeded')
need(launchint['argv'][launchint['argv'].index('-c')+1]==literal(H/'c5bf_native_cpu_launch_once_candidate.py','CPU_BOOTSTRAP'),'Exact original launcher CPU bootstrap')
need(launchint['argv'][:8]==['srun','--jobid=3304','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=none','--nodelist=h100-10'],'Exact existingCPU-only allocation command')
need(lr==j(L/'stdout.original') and lr['srun_waited'] is True and type(lr['srun_returncode']) is int and lr['srun_returncode']==0 and lr['actual_evaluation_job_id'] is None,'Actual remote/local result exact')
localpi=j(L/'local-popen.original.json');localintent=j(L/'intent.original.json');localexit=j(L/'exit.original.json');need(localpi['argv']==localintent['argv'] and localpi['pid']==localexit['pid'],'True local transport waited-Popen join')
need(record(L/'launcher-source.original.py')['sha256']==localintent['launcher_sha256']=='7731a24dcadf631b899003ba66b8b7f2cfb0e7fcc39b34e1cfab8869174b0196','Executed launcher pin')
for context in [launchint['before'],after]:
 e=context['driver'];need(e['pid']==2548319 and e['start_ticks']=='1834784786' and e['uids']==[0]*4 and re.search(r'(?:^|/)job_3304/step_0(?:/|$)',e['cgroup'],re.M),'Original train driver identity unchanged')
 for k in ['source_head','source_clean','slurm']:zero(context[k])
 need(context['source_head']['stdout'].strip()==report['stored_training_source_commit'] and context['source_clean']['stdout']=='','Original frozen fcec HEAD and clean')
 need(context['tool_manifest']['sha256']=='ee31da65570318227ad6b037187cffa14e4dc489bac2a504d2bc54010c95da05' and context['source_inventory']['sha256']==invsha,'Original current tool/inventory pins')
 m=j(B/'manifest.json');need(set(context['tool_records'])==set(m['tools']),'Actual22 tool context exact')
 for n,r in context['tool_records'].items():need(r['sha256']==m['tools'][n] and r['uid']==0 and r['mode']==0o444 and r['nlink']==1,'Actual22 member pin/physical '+n)
 fields=dict(x.split('=',1) for x in context['slurm']['stdout'].split() if '=' in x)
 for k,v in {'JobId':'3304','JobState':'RUNNING','Restarts':'0','NodeList':'h100-10','UserId':'root(0)','TresPerNode':'gres:gpu:h100:8'}.items():need(fields[k]==v,'Original training allocation')
for r in wire['source_git_originals']:zero(r)
need(wire['source_git_originals'][0]['stdout'].strip()==report['stored_training_source_commit'] and wire['source_git_originals'][2]['stdout'].strip()==report['eval_source_commit'] and not wire['source_git_originals'][1]['stdout'] and not wire['source_git_originals'][3]['stdout'],'Post collection Git sources still clean/exact')
for r in wire['former_CPU_processes']:need(r['current_present'] is False and r['same_process_identity'] is False,'Original checker/controller exited at collection')
need(set(x['pid'] for x in wire['former_CPU_processes'])=={2752194,2752196},'Only inspected own CPU PIDs gone, no training release claim')
result={'version':1,'status':'PASS_ACTUAL_CPU_IMPORT_PARTIAL_CONFIG_ORIGINALS_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_original_members':allraw,'actual_total_bytes':total,'source_inventory_sha256':invsha,'source_tree_members':857,'module_count':94,'verified_live_function_code_count':837,'project_module_count':len(project_modules),'controller_execution':intent['execution'],'checker_execution':report['execution'],'runtime':report['runtime'],'future_interpolations':future,'future_values':config['future_values'],'report':record(native/'native-cpu-preparation.json'),'closure':record(native/'native-cpu-preparation.json.closure.json'),'partial_config':record(native/'native-cpu-preparation.json.partial-config.json'),'physical_records':record(native/'native-cpu-preparation.json.source-physical-records.json'),'all_waited_originals':[record(L/'exit.original.json'),record(launch/'srun-wait.original.json'),record(native/'child-wait.original.json'),record(native/'controller-complete.original.json'),record(D/'transport.wait.original.json')],'child_stderr_original':record(native/'child.stderr.original'),'reviewer_called_production_or_remote_or_helpers':False}
with (Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute/independent-review/actual-c5bf-native-cpu-20261007-local-v2')/'actual-byte-graph-audit.original.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['actual_original_members','checker_execution','controller_execution','runtime']}))
