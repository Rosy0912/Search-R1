"""Read-only signed layered eligibility; no signing, GPU, training, or submission.

A report is qualified by its exact executed source, original execution witnesses,
full physical input records and a separately pinned independent review. The
root signature authenticates that composition; it never makes first5 into 200.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import socket
import sys

SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
RECEIPT5 = 'd1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
RUN = ROOT / 'runs/formal-3304'
BASE = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725')
HF200 = RUN / 'checkpoints/actor/global_step_200'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
COMMON = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
FINAL = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
IDS = '4e9df91a75beba40c3316576f6da049851e736df4a1e5c873f4f0cdb883fd4b1'
MODEL = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
COUNTS = {'nq':3610,'triviaqa':11313,'popqa':14267,'hotpotqa':7405,'2wikimultihopqa':12576,'musique':2417,'bamboogle':125}
PINS = {
 'common.py':'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581',
 'observe_final200.py':'9ac3015f60dc98012678203049985b0fb575cd44ca2e8a287e1b61cc54c6ee31',
 'inspect_final200_native.py':'03dcdc1e0cd84eb77780d1e52193c7b904e1e084467e8fb130e184ce008777d9',
 'load_final200_hf.py':'9fbe4ae0eaca5bcbbf75611ae47b2f0c8e94745c123138e6fee004558adb87cb',
 'check_final200_archives.py':'7fc3e7b96465a681390aae1a59ca9d3aae381b95ef5ea80914425f13ada8f4ed',
 'consume_final200_logs_terminal.py':'509df9332effd1196c6cb07a6a9d67612d9efdd75498f39a2bdb36726dcd7306',
}
ROLES = {'inventory','native','hf','archives','logs_terminal'}
SCOPES = {
 'native':'NATIVE_CPU_FINAL200_STORED_CHECKPOINT_CONTENT_INSPECTION_NOT_LIVE_RESTORE',
 'hf':'HF200_WHOLE_CPU_MODEL_LOAD_NOT_TRAINING_RESTORE',
 'archives':'ALL200_RAW_PAIRED_TRAINING_ARCHIVES_AND_DATA_CURSOR_CONTENT_INSPECTION_NOT_FINAL_EVALUATION',
}
REQUIRED_RAW = {
 'Stored200': {'native_wait'}, 'WholeHF200': {'hf_wait'}, 'All200Archives': {'archives_wait'},
 'Closed200LogWriterLifetime': {'writer_processes','writer_fd_closure','driver_complete','aggregate_complete','resources_complete'},
 'LiteralTerminalStates': {'child_exit','wrapper_exit','scheduler_original','whole_resources'},
 'PostTerminalSourceBindings': {'postterminal_source_bindings'},
 'IndependentResourceRelease': {'release_proc','release_nvml','release_ports','release_cgroups','release_leases_locks'},
}
QUALIFICATIONS = {'Stored200','WholeHF200','All200Archives','Closed200LogWriterLifetime',
                  'LiteralTerminalStates','PostTerminalSourceBindings','IndependentResourceRelease'}
BUDGET = dict(prompt=4096,start=2048,action=500,observation=500,max_turns=4,topk=3,
              tensor_parallel=1,n=1,world_size=8,val_batch_size=64,force_final_answer=False,
              do_search=True,do_sample=False,temperature=0.0,top_p=1.0,top_k=-1,min_p=0.0,best_of=1,
              retriever_transport='SAME_H10010_GPU_E5_GPU_FAISS_LOOPBACK')
KIND = 'LAYERED_TRAINING3304_ACTUAL200_HF200_EVIDENCE_RECEIPT_NOT_COMPLETION'
ELIGIBLE = 'QUALIFIED_STORED200_FOR_SEPARATE_PAIRED_EVAL_NOT_OVERALL_COMPLETION'
REVIEW_KIND = 'INDEPENDENT_ACTUAL3304_FINAL200_LAYERED_ARTIFACT_QUALIFICATION'
PROOF_KIND = 'STRICT_ACTUAL_TRAINING3304_HF200_PREREQUISITES'
LOG_BUNDLE_SHA = 'f11580b6a7bc52c04f51e7e6332e58e01611dae0a4f3ddda0e916a75757787ce'
LOG_BUNDLE = ROOT/'tools'/('final200-'+LOG_BUNDLE_SHA[:12])


def require(ok,message):
    if not ok: raise ValueError(message)


def exact(value,target,message):
    require(type(value) is type(target) and value == target,message)


def same(value,target,c,message):
    require(c.canonical(value)==c.canonical(target),message)


def header(payload,c):
    require(type(payload) is dict and set(payload)=={'version','kind','status','bindings','tools','reports',
        'independent_qualification','literal_terminal','cpu_executions','eligibility','effect_completion'},'Layered receipt schema differs')
    for key,value in dict(version=1,kind=KIND,status=ELIGIBLE,effect_completion=None).items():exact(payload.get(key),value,'Layered receipt is blocked or has wrong scope: '+key)
    bindings=dict(training_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id='3304',run_root=str(RUN),
        actual_outer_updates=200,hf_step=200,horizon=200,optimizer_steps_per_rank=2000,world_size=8,
        accepted5_receipt_sha256=RECEIPT5,common_manifest_sha256=COMMON,final_parquet_sha256=FINAL,
        final_question_ids_sha256=IDS,final_questions=51713,final_source_counts=COUNTS,official_revision=REVISION,
        official_model_manifest_sha256=MODEL,official_base=str(BASE),actual_hf200=str(HF200),evaluation_budget=BUDGET,
        evaluation_budget_sha256=hashlib.sha256(c.canonical(BUDGET)).hexdigest())
    same(payload['bindings'],bindings,c,'Exact training/model/cohort/budget bindings differ')
    require(set(payload['reports'])==ROLES,'Five actual200 reports required, not model-only qualification')
    same(payload['eligibility'],dict(stored_checkpoint=True,paired_raw_eval_may_launch=True,
         evaluation_allocation='REQUIRES_NEW_OWNED_SAME_H10010_ALLOCATION',same_receipt_for_both_roles=True),c,'Eligibility cannot claim allocation or final effect completion')
    require(set(payload['tools'])==set(PINS)|{'verify_actual200_prerequisites.py'},'Exact executed tool closure differs')
    return payload


def record_shape(record,c):
    require(type(record) is dict and set(record)==c.RECORD_KEYS,'Full physical record required')
    require(type(record['physical_path']) is str and Path(record['physical_path']).is_absolute(),'Physical record path missing')
    c.valid_sha(record['sha256'])
    require(all(type(record[k]) is int and record[k]>=0 for k in c.RECORD_KEYS-{'physical_path','sha256'}),'Typed physical metadata required')


def merge(records,record,c):
    record_shape(record,c);path=record['physical_path']
    if path in records:same(record,records[path],c,'Same physical input has conflicting observed identity')
    records[path]=record


def execution(value,c,context):
    require(type(value) is dict and set(value)=={'node','pid','uid','uids','start_ticks','cgroup','execution_job_id',
        'execution_step_id','stored_training_job_id','python','python_version'},'Actual CPU inspection identity schema differs')
    job=value['execution_job_id'];step=value['execution_step_id']
    require(type(job) is str and re.fullmatch('[1-9][0-9]*',job),'CPU execution job must come from actual scheduler')
    require(type(step) is str and re.fullmatch(r'(?:[0-9]+|batch|extern)',step),'Actual CPU step missing')
    exact(value['stored_training_job_id'],'3304','Stored/execution jobs conflated')
    require(type(value['node']) is str and value['node'].split('.')[0]=='h100-10','CPU inspection executed on wrong node')
    require(type(value['uid']) is int and value['uid']==0 and type(value['pid']) is int and value['pid']>0 and
        type(value['start_ticks']) is int and value['start_ticks']>0 and type(value['uids']) is list and
        all(type(x) is int for x in value['uids']) and value['uids']==[0]*4,'Actual CPU PID/start/UID types differ')
    paired=r'(?:^|/)job[_-]'+re.escape(job)+r'(?:\.scope)?/step[_-]'+re.escape(step)+r'(?:\.scope)?(?:/|$)'
    require(type(value['cgroup']) is str and re.search(paired,value['cgroup'],re.M),'CPU cgroup job and step must be in same path')
    require(type(value['python']) is str and Path(value['python']).is_absolute() and type(value['python_version']) is str,'Executed CPU interpreter provenance missing')
    historical=[context['driver_identity'],context['service_identity']]+[w['historical_process'] for w in context['workers']]
    require(not any(value['pid']==p['pid'] and value['start_ticks']==int(p['start']) for p in historical),'Historical training process impersonates CPU inspector')
    return value


def validate_contents(payload,reports,c,context,hf_validator):
    inventory=reports['inventory'];c.validate_inventory(inventory)
    same(inventory['accepted5'],context['accepted5'],c,'200 inventory does not join historical signed5')
    require('passed' not in inventory,'Observed inventory is not training acceptance manifest')
    invsha=payload['reports']['inventory']['sha256'];native=reports['native']
    hf_validator(native,inventory,invsha,PINS['inspect_final200_native.py'],c)
    same(native['accepted5'],context['accepted5'],c,'Native200 signed5 join differs')
    for role in ('native','hf','archives'):
        r=reports[role]
        for key,target in dict(version=1,scope=SCOPES[role],passed=True,source_commit=SOURCE,config_sha256=CONFIG,
          job_id='3304',run_root=str(RUN),checkpoint_step=200,inventory_sha256=invsha,cuda_is_initialized=False,
          common_source_sha256=PINS['common.py'],inspector_source_sha256=PINS[{'native':'inspect_final200_native.py','hf':'load_final200_hf.py','archives':'check_final200_archives.py'}[role]]).items():exact(r.get(key),target,'Actual200 '+role+' header differs: '+key)
        same(r['accepted5'],context['accepted5'],c,'Content report historical anchor differs')
        same(payload['cpu_executions'][role],execution(r['execution_identity'],c,context),c,'Signed CPU execution differs from qualified artifact')
        if role!='native':
            exact(r.get('native_report_sha256'),payload['reports']['native']['sha256'],'HF/archive report inspected different native proof')
            exact(r.get('native_inspector_source_sha256'),PINS['inspect_final200_native.py'],'Native source pin differs')
        else:exact(r.get('math_helper_sha256'),c.MATH_HELPER,'Native mathematics helper differs')
        if role in ('native','hf'):
            require(type(r.get('cpu_runtime')) is dict and type(r['cpu_runtime'].get('torch_version')) is str and
              r['cpu_runtime']['torch_version'].split('+')[0]=='2.4.0','Actual nativeTorch2.4 content proof required')
    require(set(payload['cpu_executions'])=={'native','hf','archives'},'All three real CPU execution witnesses required')
    hf=reports['hf']
    for key,target in dict(checkpoint=str(HF200),official_base=str(BASE),hf_helper_sha256=c.HF_HELPER,
       actual_unique_parameters=c.NUMEL,all_loaded_parameters_finite=True,actual_cpu_forward_finite=True,tied_embeddings=True).items():exact(hf.get(key),target,'Whole actual HF200 qualification differs: '+key)
    require(type(hf['hf_map']) is dict and len(hf['hf_map'])==435 and type(hf['canonical_state_keys']) is list and
            len(set(hf['canonical_state_keys']))==434,'Whole435/434 loaded HF keys missing')
    same(hf['base_named_parameter_digest'],native['base_named_parameter_digest'],c,'Official base identity differs')
    same(hf['hf_files'],native['hf_files'],c,'HF and native8slice member pins differ')
    a=reports['archives']
    for key,target in dict(outer_count=200,member_count=4000,student_count=512000,teacher_count=102400,
       unique_common_question_count=102400,archived_questions_exact_stored200_sampler_and_common_parquet=True,
       official_tokenizer_replayed=True).items():exact(a.get(key),target,'All200 raw archive qualification differs: '+key)
    require(type(a['outers']) is list and len(a['outers'])==200,'All200 individual raw archive summaries missing')
    for outer,row in enumerate(a['outers'],1):
        for key,target in dict(outer=outer,student_count=2560,teacher_count=512,actual_worker_bindings_verified=True).items():exact(row.get(key),target,'All200 raw archive group differs')
        exact(row['paired_manifest_sha256'],inventory['archive_files'][f'trajectory_outer_{outer:06d}/paired-manifest.json']['sha256'],'Raw archive summary inspected different manifest')
    logs=reports['logs_terminal']
    for key,target in dict(version=1,kind='FINAL200_LOG_TERMINAL_LAYER_REPORT_NOT_COMPLETION_RECEIPT',
      status='VERIFIED_LOGS_WITH_INDEPENDENT_TERMINAL_LAYERS_FINAL_ACCEPTANCE_BLOCKED',source_commit=SOURCE,
      config_sha256=CONFIG,stored_training_job_id='3304',actual_final_completion=None).items():exact(logs.get(key),target,'Original log layer scope differs: '+key)
    exact(logs['consumer_source']['sha256'],PINS['consume_final200_logs_terminal.py'],'Logs consumer source pin differs')
    exact(logs['common_source']['sha256'],PINS['common.py'],'Logs common source pin differs')
    same(logs['accepted5'],context['accepted5'],c,'Log stream signed5 anchor differs')
    for role in ('driver','aggregate'):
        layer=logs['log_layers'][role]
        for key,target in dict(status='VERIFIED_LOG_EVENTS_ONLY_NOT_FINAL_ACCEPTANCE',actual_outer_updates=200,
          adam_calls_per_rank=2000,no_executed_outer201_in_read_originals=True,
          log_writer_lifetime_qualification='BLOCKED_REQUIRE_INDEPENDENT_WRITER_EXIT_AND_FD_CLOSURE').items():exact(layer.get(key),target,'Full200 log scope differs: '+key)
        require(type(layer['outer']) is list and len(layer['outer'])==200 and type(layer['evaluations']) is list and len(layer['evaluations'])==11,'Actual200/evaluation console+JSON records missing')
    d=logs['dataset']['final']
    for key,target in dict(rows=51713,question_ids_sha256=IDS).items():exact(d.get(key),target,'Final cohort differs')
    same(d['source_counts'],COUNTS,c,'Final seven source counts differ');exact(d['file']['sha256'],FINAL,'Common final parquet differs')
    for split,total in [('validation',512),('stage',64)]:exact(logs['dataset'][split]['rows'],total,'Fixed validation/stage cohort differs')
    term=logs['terminal_layers']
    for name in ('training_child_exit','wrapper_exit'):
        exact(term[name]['status'],'OBSERVED_LITERAL_EXIT','Terminal original absent')
        require(type(term[name]['returncode']) is int,'Literal terminal exit type differs')
    exact(term['scheduler']['status'],'OBSERVED_LITERAL_SCHEDULER_TERMINAL','Actual scheduler terminal original missing')
    require(term['scheduler']['state'] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'} and
            type(term['scheduler']['exit_code']) is str and re.fullmatch('[0-9]+:[0-9]+',term['scheduler']['exit_code']),'Literal scheduler terminal type differs')
    same(payload['literal_terminal'],dict(training_child_exit=term['training_child_exit'],wrapper_exit=term['wrapper_exit'],
         scheduler=term['scheduler'],whole_resource_layer=logs['whole_resource_layer']),c,'Receipt changed actual terminal states or whole bad66')
    whole=logs['whole_resource_layer']
    exact(whole['status'],'FAILED_WHOLE_RESOURCE_OWNERSHIP_NOT_UPGRADED_BY_TRAINING_PHASE','Whole bad66 failure must remain literal')
    require(whole.get('bad66') is not None and type(whole.get('unresolved_or_bad_rows')) is list and whole['unresolved_or_bad_rows'],'Whole resource bad originals missing')


def qualify_review(review,payload,c):
    require(type(review) is dict and set(review)=={'version','kind','status','source_commit','config_sha256',
       'stored_training_job_id','report_sha256','qualifications','evidence_records','cpu_execution_witnesses',
       'literal_terminal','effect_completion'},'Independent layered qualification schema differs')
    for key,target in dict(version=1,kind=REVIEW_KIND,status='QUALIFIED_ARTIFACTS_AND_EVAL_ELIGIBILITY_NOT_EFFECT_COMPLETION',
       source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id='3304',effect_completion=None).items():exact(review.get(key),target,'Independent qualification blocked or has wrong scope')
    same(review['report_sha256'],{k:v['sha256'] for k,v in payload['reports'].items()},c,'Independent review bound different full reports')
    same(review['literal_terminal'],payload['literal_terminal'],c,'Independent review did not retain actual failure states')
    require(set(review['qualifications'])==QUALIFICATIONS,'Missing independent source/release/writer-lifetime qualification')
    for scope,v in review['qualifications'].items():
        require(type(v) is dict and set(v)=={'status','evidence_roles'},'Layered independent review scope schema differs')
        exact(v['status'],'QUALIFIED_INDEPENDENT_RAW_EVIDENCE_REVIEW','Unreviewed raw qualification is BLOCKED')
        require(type(v['evidence_roles']) is list and v['evidence_roles'] and len(set(v['evidence_roles']))==len(v['evidence_roles']) and
          all(type(x) is str and x in review['evidence_records'] for x in v['evidence_roles']),'Independent qualification raw evidence missing')
    for scope,roles in REQUIRED_RAW.items():
        require(roles <= set(review['qualifications'][scope]['evidence_roles']),'Scope-specific original evidence is missing: '+scope)
    require(set(review['cpu_execution_witnesses'])=={'native','hf','archives'},'Real CPU wait evidence missing')
    for role,entry in review['cpu_execution_witnesses'].items():
        require(type(entry) is dict and set(entry)=={'execution_identity','command_record','started_record','wait_record','stdout_record','stderr_record'},'Actual CPU execution original roles differ')
        same(entry['execution_identity'],payload['cpu_executions'][role],c,'Independent CPU identity did not join actual content inspector')
    require(type(review['evidence_records']) is dict and review['evidence_records'],'Raw independently reviewed evidence cannot be empty')


def qualify_original_bindings(review,reports,c):
    spec,_=c.read_json(reports['logs_terminal']['input_spec']['physical_path'],reports['logs_terminal']['input_spec']['sha256'])
    evidence=review['evidence_records']
    for name,role in [('child_exit','training_child_exit'),('wrapper_exit','wrapper_exit'),('scheduler_original','scheduler_original')]:
        same(evidence[name],spec['terminal'][role],c,'Independent terminal qualification referenced a different original')
    for name,role in [('driver_complete','driver'),('aggregate_complete','aggregate')]:
        same(evidence[name],spec['logs'][role]['file'],c,'Closed writer review referenced a different full log')
    for name in ('whole_resources','resources_complete'):same(evidence[name],spec['resources'],c,'Whole original resources changed in qualification')
    for role,entry in review['cpu_execution_witnesses'].items():same(evidence[role+'_wait'],entry['wait_record'],c,'Independent CPU wait qualification did not join actual wait')


def collect_inputs(payload,reports,review,context,c):
    records=dict(context['records'])
    for rec in list(payload['tools'].values())+list(payload['reports'].values())+[payload['independent_qualification']]:merge(records,rec,c)
    inventory=reports['inventory']
    seen=set()
    for category,expected in [('checkpoint_files',c.checkpoint_paths()),('archive_files',c.archive_paths())]:
        require(set(inventory[category])==set(expected),'Complete actual200 physical member inventory differs')
        for role,rec in inventory[category].items():
            record_shape(rec,c);exact(rec['physical_path'],str(expected[role]),'Actual200 member path differs')
            require((rec['device'],rec['inode']) not in seen,'Actual200 members alias inode');seen.add((rec['device'],rec['inode']));merge(records,rec,c)
    for role in ('native','hf','archives'):
        require(type(reports[role]['verified_input_records']) is dict,'Full verified input record map missing')
        for path,rec in reports[role]['verified_input_records'].items():exact(path,rec['physical_path'],'Input map key differs');merge(records,rec,c)
        for rec in inventory['checkpoint_files'].values():same(reports[role]['verified_input_records'].get(rec['physical_path']),rec,c,'Content report did not inspect all31 current checkpoint files')
    for rec in inventory['archive_files'].values():same(reports['archives']['verified_input_records'].get(rec['physical_path']),rec,c,'All200 archives report did not inspect all4000 current members')
    for rec in reports['logs_terminal']['input_records'].values():merge(records,rec,c)
    for rec in review['evidence_records'].values():merge(records,rec,c)
    for entry in review['cpu_execution_witnesses'].values():
        for key in ('command_record','started_record','wait_record','stdout_record','stderr_record'):merge(records,entry[key],c)
    return records


def validate_wait_originals(review,payload,c):
    for role,entry in review['cpu_execution_witnesses'].items():
        identity=entry['execution_identity']
        command,_=c.read_json(entry['command_record']['physical_path'],entry['command_record']['sha256'])
        started,_=c.read_json(entry['started_record']['physical_path'],entry['started_record']['sha256'])
        waited,_=c.read_json(entry['wait_record']['physical_path'],entry['wait_record']['sha256'])
        require(type(command.get('argv')) is list and command['argv'] and all(type(v) is str for v in command['argv']),'Actual command argv original missing')
        same(started['argv'],command['argv'],c,'Spawn and command argv differ')
        exact(started['child_pid'],identity['pid'],'Spawned inspector differs from report execution PID')
        child=started['child'];require(type(child) is dict,'Actual spawned inspector process identity missing')
        exact(child['pid'],identity['pid'],'Spawn child PID differs');exact(child['start_ticks'],identity['start_ticks'],'Spawn child start differs')
        exact(waited.get('returncode'),0,'Actual content inspector failed');exact(waited.get('waited'),True,'True subprocess wait missing')
        exact(waited.get('child_pid'),identity['pid'],'Waited child differs');exact(waited.get('capture_errors'),[],'Execution capture failed or unknown')
        require(type(waited.get('waited_by_pid')) is int and waited['waited_by_pid']>0,'Actual wait parent identity missing')
        inspector={'native':'inspect_final200_native.py','hf':'load_final200_hf.py','archives':'check_final200_archives.py'}[role]
        require(command['argv'][0]==identity['python'] and command['argv'].count(payload['tools'][inspector]['physical_path'])==1,'Actual CPU argv executed different source/interpreter')
        require(started.get('capture_errors')==[] and type(started.get('parent')) is dict and type(started['parent'].get('pid')) is int and type(started.get('parent_pid')) is int and started['parent']['pid']==waited['waited_by_pid']==started['parent_pid'],'Original wait parent does not match original spawn')
        for k in ('uid','uids','cgroup','node'):
            same(child[k],identity[k],c,'Actual spawned CPU identity differs: '+k)
        dependencies={'--common-sha256':PINS['common.py'],'--inventory':payload['reports']['inventory']['physical_path'],
          '--inventory-sha256':payload['reports']['inventory']['sha256'],'--accepted5-receipt-sha256':RECEIPT5,
          '--output':payload['reports'][role]['physical_path']}
        if role!='native':dependencies.update({'--native-report':payload['reports']['native']['physical_path'],
          '--native-report-sha256':payload['reports']['native']['sha256'],'--native-inspector':payload['tools']['inspect_final200_native.py']['physical_path'],
          '--native-inspector-sha256':PINS['inspect_final200_native.py']})
        for flag,val in dependencies.items():
            require(command['argv'].count(flag)==1 and command['argv'][command['argv'].index(flag)+1]==val,'Actual CPU argv source/input/output dependency differs: '+flag)
        stdout,_=c.read_json(entry['stdout_record']['physical_path'],entry['stdout_record']['sha256'])
        same(stdout,dict(scope=SCOPES[role],passed=True,output=payload['reports'][role],stored_training_job_id='3304',execution_identity=identity),c,'Actual stdout did not publish the qualified content report')
        for flag,key in [('--execution-job-id','execution_job_id'),('--execution-step-id','execution_step_id')]:
            require(command['argv'].count(flag)==1 and command['argv'][command['argv'].index(flag)+1]==identity[key],'Actual CPU argv execution identity differs')


def load_common():
    """Bootstrap only the explicitly pinned sibling source; no pyc reuse."""
    import stat,types
    p=Path(__file__).parent/'common.py';require(p.is_absolute() and p.resolve(strict=True)==p and p.stat().st_uid==os.getuid(),'Physical owned common source required')
    before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    fields=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    require(stat.S_ISREG(opened.st_mode) and all(tuple(getattr(x,k) for k in fields)==tuple(getattr(before,k) for k in fields) for x in (opened,after,p.stat())) and hashlib.sha256(raw).hexdigest()==PINS['common.py'],'Pinned common source changed')
    m=types.ModuleType('prerequisite_exact_common');m.__file__=str(p);exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),m.__dict__);return m


def recorded_log_common(old,c):
    """Replay with the physically bound original common, retaining caller identity."""
    import stat
    bundle=LOG_BUNDLE
    require(bundle.is_absolute() and bundle.resolve(strict=True)==bundle and
      bundle.stat().st_uid==os.getuid() and stat.S_ISDIR(bundle.stat().st_mode) and
      stat.S_IMODE(bundle.stat().st_mode)==0o555,'Original log bundle physical identity or mode differs')
    manifest,mrec=c.read_json(bundle/'manifest.json',LOG_BUNDLE_SHA,bundle)
    require(type(manifest) is dict and set(manifest)=={'version','kind','source_commit','tools'},'Original log bundle schema differs')
    for key,target in dict(version=1,kind='FINAL200_READ_ONLY_RUNTIME_BUNDLE',source_commit=SOURCE).items():exact(manifest[key],target,'Original log bundle field differs: '+key)
    require(type(manifest['tools']) is dict and set(p.name for p in bundle.iterdir())==set(manifest['tools'])|{'manifest.json'},'Original log bundle membership differs')
    require(stat.S_IMODE((bundle/'manifest.json').stat().st_mode)==0o444,'Original log manifest mode differs')
    records={mrec['physical_path']:mrec}
    for name,pin in manifest['tools'].items():
        require(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name),'Unsafe original log bundle member')
        _,rec=c.read_file(bundle/name,pin,bundle);merge(records,rec,c)
        require(stat.S_IMODE((bundle/name).stat().st_mode)==0o444,'Original log bundle member mode differs')
    for name,pin in PINS.items():exact(manifest['tools'].get(name),pin,'Original log helper pin differs: '+name)
    path=str(bundle/'common.py');record=records[path]
    same(old['common_source'],record,c,'Original log report common physical identity differs')
    same(old['input_records'].get(path),record,c,'Original log report omitted its actual common source')
    same(old['consumer_source'],records[str(bundle/'consume_final200_logs_terminal.py')],c,'Original log report consumer physical identity differs')
    original,rec=c.exact_module(path,PINS['common.py'],'prerequisite_original_log_common')
    same(rec,record,c,'Compiled original log common changed');c.recheck(records)
    return original,records


def verify(a,c):
    own=c.read_file(Path(__file__).absolute())[1]
    envelope,rrec=c.read_json(a.receipt,a.receipt_sha256)
    require(type(envelope) is dict and set(envelope)=={'payload','signature_hex'},'Signed layered eligibility envelope required')
    context=c.accepted_context(RECEIPT5)
    public,_=c.read_json(context['accepted5']['public_key_path'],c.PUBLIC)
    context['gate'].verify_rsa_signature(envelope['payload'],envelope['signature_hex'],public)
    payload=header(envelope['payload'],c)
    exact(payload['tools']['verify_actual200_prerequisites.py'],own,'Actually executed verifier byte identity differs')
    exact(payload['tools']['common.py'],c.read_file(Path(c.__file__))[1],'Actually compiled common physical identity differs')
    for name,pin in PINS.items():exact(payload['tools'][name]['sha256'],pin,'Executed helper source differs')
    reports={role:c.read_json(rec['physical_path'],rec['sha256'])[0] for role,rec in payload['reports'].items()}
    hfmod,_=c.exact_module(payload['tools']['load_final200_hf.py']['physical_path'],PINS['load_final200_hf.py'],'prerequisite_exact_hf_validator')
    validate_contents(payload,reports,c,context,lambda *args:hfmod.validate_native(*args))
    review,_=c.read_json(payload['independent_qualification']['physical_path'],payload['independent_qualification']['sha256'])
    qualify_review(review,payload,c);qualify_original_bindings(review,reports,c);validate_wait_originals(review,payload,c)
    # Re-run only the sealed read-only log consumer: no training, Tensor/HF load or evaluation.
    logsmod,_=c.exact_module(payload['tools']['consume_final200_logs_terminal.py']['physical_path'],PINS['consume_final200_logs_terminal.py'],'prerequisite_exact_logs')
    from types import SimpleNamespace
    old=reports['logs_terminal'];original_log_common,replay_records=recorded_log_common(old,c)
    fresh=logsmod.consume(SimpleNamespace(common_sha256=PINS['common.py'],accepted5_receipt_sha256=RECEIPT5,
      evidence_spec=old['input_spec']['physical_path'],evidence_spec_sha256=old['input_spec']['sha256']),original_log_common)
    same({k:v for k,v in fresh.items() if k!='execution'},{k:v for k,v in old.items() if k!='execution'},c,'Original final200 logs/resource/terminal content no longer joins qualified report')
    records=collect_inputs(payload,reports,review,context,c);merge(records,rrec,c);merge(records,own,c)
    for rec in replay_records.values():merge(records,rec,c)
    # Full official11 assets are present in the qualified HF report; do not infer from its label.
    official,omrec=c.read_json(BASE/'VERIFIED-MODEL-MANIFEST.json',MODEL);merge(records,omrec,c)
    exact(official['repo'],'Qwen/Qwen2.5-3B-Instruct','Official model repo differs');exact(official['revision'],REVISION,'Official revision differs')
    require(type(official['files']) is list and len(official['files'])==11 and len({v['file'] for v in official['files']})==11,'Official11 assets missing')
    for item in official['files']:
        require(type(item['file']) is str and Path(item['file']).name==item['file'],'Official asset path differs')
        path=str(BASE/item['file']);rec=reports['hf']['verified_input_records'].get(path);require(rec is not None,'Whole HF qualification omitted official asset')
        exact(rec['sha256'],item['sha256'],'Official asset fullSHA differs');exact(rec['bytes'],item['bytes'],'Official asset size differs')
    require(a.model_role in {'official_base','actual_hf200'},'Only official_base or actual_hf200 roles allowed')
    exact(a.model_directory,str(BASE if a.model_role=='official_base' else HF200),'Actual eval model path differs')
    c.recheck(records)
    return dict(kind=PROOF_KIND,training_commit=SOURCE,training_job_id='3304',actual_outer_updates=200,hf_step=200,
       final_questions=51713,receipt_sha256=rrec['sha256'],model_role=a.model_role,model_directory=a.model_directory)


def invocation_identity(c):
    """Read-only caller may run inside the new GPU eval job; never initializes CUDA."""
    job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID')
    require(sys.platform=='linux' and socket.gethostname().split('.')[0]=='h100-10' and os.getuid()==0,'Eval verifier must run in owned same-node allocation')
    require(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job!='3304','Separate actual eval allocation required')
    require(type(step) is str and re.fullmatch(r'(?:[0-9]+|batch|extern)',step),'Actual evaluation step missing')
    status=Path('/proc/self/status').read_text();uids=re.search(r'^Uid:\s+(.*)$',status,re.M)
    require(uids is not None and [int(x) for x in uids.group(1).split()]==[0]*4,'Actual evaluation process four UIDs differ')
    cgroup=Path('/proc/self/cgroup').read_text();paired=r'(?:^|/)job[_-]'+re.escape(job)+r'(?:\.scope)?/step[_-]'+re.escape(step)+r'(?:\.scope)?(?:/|$)'
    require(re.search(paired,cgroup,re.M),'Actual eval job/step cgroup differs')
    # No Torch import, no GPU call, no training dispatch; executed source is also receipt-bound.


def main():
    p=argparse.ArgumentParser()
    for name in ('receipt','receipt-sha256','model-role','model-directory'):p.add_argument('--'+name,required=True)
    a=p.parse_args();c=load_common();invocation_identity(c);proof=verify(a,c);print(c.canonical(proof).decode())


if __name__=='__main__':main()
