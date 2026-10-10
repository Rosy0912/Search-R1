"""Host-only migration overlay; compile exact frozen scientific modules.

Original config/CP/rank/optimizer/RNG bytes and same-run identity are retained.
The effective allocation is explicitly K5 0--3, never the historical K4 owner.
"""
import argparse,copy,hashlib,importlib,json,os,socket,subprocess,sys,time,uuid,fcntl,threading
from pathlib import Path
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
NATIVE=S/'source/training-00fec71/opsd_signal_weight_v1'
UUIDS=['GPU-15061c70-49d6-3014-da87-915b370c1e89','GPU-0c0aeed1-c7b9-c9ad-4b1f-7396b6c19f4c','GPU-d2609c94-e30a-a106-e1cc-6d4ae3e044c2','GPU-d6d81990-8737-f64c-aaef-d0ceabb5123a']
SHAS={'H_POS':'934f3e1ab8f70d7a1faa14b9806e9ddaecdbaea6afc97668ca3389a5c7237612','H_NEG':'ddb7d18dfcd5826f6aede00bcb507cebe553d8af5a28b9ef3d299c137b92c8bc'}
MANIFEST_SHA='af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314'
def require(value,reason):
    if not value:raise ValueError(reason)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(ref):
    require(sha(ref['path'])==ref['sha256'],'Pinned input changed: '+ref['path']);return json.loads(Path(ref['path']).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def effective(config):
    require(config['variant'] in SHAS and config['owner']['uid']==1001,'Only migrated H groups')
    config['owner']=dict(uid=1001,physical_gpus=[0,1,2,3],gpu_uuids=UUIDS[:]);return config

def install(binding):
    require(os.uname().nodename=='xie-lab-4' and os.getuid()==1001,'Wrong target host/UID')
    require(Path('/weka_3').resolve()==Path('/weka_3'),'Migration namespace absent')
    require(binding['human_authorization']=='可以的，迁移过去，去HKUST5跑。' and binding['source_stop']['all_original_targets_absent'],'Source stop/authorization missing')
    pub=read(binding['publication']);require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha'],'Migration not published')
    own=read(binding['helper_manifest']);root=Path(__file__).parent
    for name,digest in own['files'].items():require(sha(root/name)==digest,'Migration source changed')
    sys.path.insert(0,str(S/'source/port-resume-0146350/opsd_signal_weight_port_resume_v1'))
    from recovery_common import install_original,tree
    ref=dict(path=str(NATIVE/'source-manifest.json'),sha256=MANIFEST_SHA);native=tree(NATIVE,ref);install_original(NATIVE,native)
    cp=importlib.import_module('checkpoint');original_json=cp.checked_json
    def json_with_allocation(ref):
        value,proof=original_json(ref)
        if value.get('schema')=='opsd-signal-weight-config-v1' and value.get('variant') in SHAS:
            require(proof['sha256']==SHAS[value['variant']],'Original immutable config changed');effective(value)
        return value,proof
    cp.checked_json=json_with_allocation
    prepare=importlib.import_module('prepare');native_validate=prepare.validate_config
    def validate(config,launch=False):
        original=copy.deepcopy(config);original['owner']=prepare.template(original['paths']['base'],original['variant'],original['study_id'],original['paths']['study_base'])['owner']
        result=native_validate(original,launch);require(config['owner']==dict(uid=1001,physical_gpus=[0,1,2,3],gpu_uuids=UUIDS),'Effective target allocation differs');return result
    prepare.validate_config=validate
    gates=importlib.import_module('gates');gates.inventory=inventory
    sampling=importlib.import_module('sampling');native_metadata=sampling.verify_model_metadata;validated={}
    def migrated_metadata(identity):
        try:return native_metadata(identity)
        except ValueError:
            # Only historical metadata from K4 is portable. Byte identity is mandatory.
            for ref in identity['files']:
                key=(ref['path'],ref['sha256']);now=cp.stat_identity(ref['path'])
                if validated.get(key)!=now:cp.checked_file(ref);validated[key]=now
            return True
    sampling.verify_model_metadata=migrated_metadata
    return cp,prepare,gates

def inventory():
    # GPU4 is outside our allocation and has an NVML N/A PID. Never inspect/use it.
    prefix=['nvidia-smi','-i','0,1,2,3'];query=lambda name:subprocess.check_output(prefix+['--query-'+name,'--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in query('gpu=index,uuid,memory.free,memory.total').splitlines() if r.strip()]
    devices=[dict(index=int(r[0]),uuid=r[1].strip(),actual_free_mib=int(r[2]),total_mib=int(r[3])) for r in rows]
    require([d['index'] for d in devices]==list(range(4)) and [d['uuid'] for d in devices]==UUIDS,'Actual target GPU mapping changed')
    apps=[]
    for line in query('compute-apps=gpu_uuid,pid,used_gpu_memory').splitlines():
        if not line.strip():continue
        r=[v.strip() for v in line.split(',')];require(r[1].isdigit() and r[2].isdigit(),'Unknown process on allocated GPU')
        apps.append(dict(gpu_uuid=r[0],pid=int(r[1]),used_mib=int(r[2])))
    return devices,apps

def oracle(config,gates):
    from urllib.request import urlopen
    receipt=gates.bound(config['gates']['retriever_admission']);require(receipt['pid']==1749337 and receipt['starttick']==641133475,'Original remote oracle changed')
    full=gates.bound(dict(path=receipt['full_gpu_gate_path'],sha256=receipt['full_gpu_gate_sha256']));numeric=gates.verify_existing_joined_gate(full)
    require(config['paths']['retriever_url']=='http://10.123.4.11:18003/retrieve','Remote route changed')
    with urlopen('http://10.123.4.11:18003/health',timeout=10) as f:health=json.load(f)
    require(health['ready'] and health['ntotal']==21015324 and health['d']==768 and health['tf32'] is False and health['query_micro']==64 and health['acceptance_sha256']==receipt['full_gpu_gate_sha256'] and health['source_manifest_sha256']==full['source_manifest_sha256'],'Remote oracle health/source/numeric identity changed')
    return dict(host='HKUST4',health=health,numeric=numeric,remote_pid_binding=receipt,local_pid_check_not_applicable=True)

def release(config,run_id,owner,known_children=None):
    import gates
    known_children=known_children or [];blockers=[]
    for expected in known_children:
        try:actual=gates.process(expected['pid'])
        except (FileNotFoundError,ProcessLookupError):continue
        if actual['starttick']==expected['starttick']:blockers.append(actual)
    # Fresh scan of marked same-run descendants, including unregistered GPU workers.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==owner['pid']:continue
        try:
            if p.stat().st_uid!=owner['uid']:continue
            env=dict(x.split('=',1) for x in (p/'environ').read_text().split('\0') if '=' in x)
            if env.get('E1_RUN_ID')==run_id:
                actual=gates.process(int(p.name));blockers.append(actual)
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            # CPU infrastructure outside this run is not a GPU ownership exemption.
            argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            require(str(config['paths']['output_root']) not in argv,'Unreadable potential experiment process')
    _,apps=inventory();require(not apps,'Allocated GPU processes remain after actual wait')
    require(not blockers,'Same-run/known child still present after wait: '+str(blockers))
    return dict(accepted=True,checked_unix=time.time(),known_children=known_children,marked_same_run_scan=True,allocated_gpu_processes=apps,no_exit_code_inferred=True)

def admit_factory(cp,gates,binding,ref):
    def admit(config,phase,run_id,owner,known_children=None):
        require(sha(ref['path'])==SHAS[config['variant']],'Immutable config changed')
        require(gates.same_process(gates.process(owner['pid']),owner),'Controller identity changed')
        retrieval=oracle(config,gates);devices,apps=inventory();owned=[];owned_uuids=set()
        for app in apps:
            actual=gates.owned_gpu_process(app['pid'],run_id,owner,known_children or []);owned.append(actual);owned_uuids.add(app['gpu_uuid'])
        gates.resource_limits(config,phase,devices,owned,owned_uuids)
        require(__import__('shutil').disk_usage(config['paths']['base']).free>=config['resources']['minimum_disk_bytes'],'Target disk full')
        return dict(accepted=True,phase=phase,checked_unix=time.time(),devices=devices,owned_gpu_processes=owned,retriever=retrieval,migration= binding['publication'],effective_owner=config['owner'],historical_config_owner='HKUST4; unchanged immutable config',no_signals_sent=True)
    admit.require_owned_release=release;return admit

def dispatch_context(binding_ref):
    original=subprocess.Popen
    def dispatch(command,*args,**kwargs):
        if isinstance(command,(list,tuple)):
            command=list(command);role=None;args0=None;launcher=None
            for name in ['sampling','teacher_diag']:
                if len(command)>1 and command[1]==str(NATIVE/(name+'.py')):role=name;args0=command[2:]
            if len(command)>5 and command[1:5]==['-m','torch.distributed.run','--standalone','--nproc_per_node=4'] and command[5]==str(NATIVE/'trainer.py'):
                role='trainer';args0=command[6:]
                with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
                launcher=[command[0],'-m','torch.distributed.run','--rdzv_backend=static','--master_addr=127.0.0.1','--master_port='+str(port),'--rdzv_id=signal-k5-'+uuid.uuid4().hex,'--nproc_per_node=4']
            if role:
                replacement=[command[0],str(Path(__file__).resolve()),'--binding',binding_ref['path'],'--binding-sha256',binding_ref['sha256'],'--mode',role,'--']+args0
                command=(launcher+replacement[1:]) if launcher else replacement
                env=kwargs.get('env');require(env is not None,'Native child env missing')
                if env.get('CUDA_VISIBLE_DEVICES')=='4,5,6,7':env=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3');kwargs['env']=env
        return original(command,*args,**kwargs)
    class Context:
        def __enter__(self):subprocess.Popen=dispatch
        def __exit__(self,*_):subprocess.Popen=original
    return Context()

def controller(binding,binding_ref,args,cp,prepare,gates):
    # Skip only the two already fully accepted K4 outers. Replaying their historical
    # inode/ctime provenance on another filesystem would be invalid. Their immutable
    # summaries and full state are preserved and checked separately in the binding.
    import types
    source=(NATIVE/'controller.py').read_text()
    marker="previous=None;updates=0;cursor=dict(cycle=0,offset=0)"
    require(source.count(marker)==1,'Original controller cursor setup changed')
    source=source.replace(marker,marker+"\n        if self.config['variant']=='H_POS':\n            previous=_migration_binding['resume_checkpoint'];updates=21;cursor=_migration_binding['resume_cursor']")
    loop="for index in range(self.config['scientific']['outer_iterations']):"
    require(source.count(loop)==1,'Original outer loop changed')
    source=source.replace(loop,"for index in range(2 if self.config['variant']=='H_POS' else 0,self.config['scientific']['outer_iterations']):")
    module=types.ModuleType('_k5_native_controller_cursor_overlay');module.__file__=str(NATIVE/'controller.py');module.__dict__['_migration_binding']=binding
    exec(compile(source,str(NATIVE/'controller.py'),'exec'),module.__dict__)
    module.logical_gpu=lambda config,rank:config['owner']['physical_gpus'][config['sampling']['logical_rank_to_slot'][rank]]
    # Defer test inference without touching the original science/evaluation contract.
    def deferred(self,n,checkpoint_ref=None):
        if n==0:return dict(evaluation_pending=True,not_executed=True)
        value=dict(accepted=False,evaluation_pending=True,completed_outer=n,raw_checkpoint=checkpoint_ref,run_id=self.run_id,config_sha256=self.config_ref['sha256'],migration=binding_ref)
        path=self.output/'evaluations'/('deferred-outer-%02d.json'%n)
        if path.exists():require(json.loads(path.read_text())['raw_checkpoint']==checkpoint_ref,'Saved endpoint changed')
        else:atomic(path,value)
        return value
    module.Controller.evaluate_boundary=deferred
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--config-sha256');parser.add_argument('--run-id');parser.add_argument('--resume',action='store_true');a=parser.parse_args(args)
    ref=dict(path=a.config,sha256=a.config_sha256)
    require(a.config_sha256==SHAS[read(ref)['variant']],'Only original H config')
    if read(ref)['variant']=='H_POS':
        verified,_=cp.check_checkpoint(binding['resume_checkpoint'],dict(read(ref),_config_sha256=ref['sha256']),a.run_id,21,1)
        require(verified['cursor']['next_pair_cursor']==verified['cursor']['usable_pair_count']==309 and verified['cumulative_pair_exposure']==636,'Migrated original completed state differs')
        for item in binding['previous_hpos_summaries']:
            summary=read(item);require(summary['train_denominator']==2560 and summary['accepted'],'Previous full TRAIN EM missing')
    c=module.Controller(ref,a.run_id,a.resume,admit_factory(cp,gates,binding,ref));c.admit.require_owned_release=release
    atomic(S/'receipts/migration-live-controller.json',dict(identity=c.owner,config=ref,binding=binding_ref,effective_owner=c.config['owner']))
    stop=threading.Event();errors=[]
    def memory():
        count=0;minimum={i:None for i in range(4)};below=0
        path=Path(c.config['paths']['receipts'])/'migration-memory-samples.jsonl'
        with path.open('a') as f:
            while not stop.is_set():
                try:
                    devices,apps=inventory();count+=1;good=all(d['actual_free_mib']>=c.config['resources']['training_minimum_free_mib'] for d in devices);below+=int(not good)
                    for d in devices:minimum[d['index']]=d['actual_free_mib'] if minimum[d['index']] is None else min(minimum[d['index']],d['actual_free_mib'])
                    row=dict(sample_unix=time.time(),devices=devices,compute_applications=apps,phase=c.phase,actual_free_limit_met=good)
                    f.write(json.dumps(row)+'\n');f.flush()
                    atomic(path.with_name('migration-memory-live.json'),dict(run_id=c.run_id,config_sha256=ref['sha256'],binding=binding_ref,controller_identity=c.owner,effective_owner=c.config['owner'],current=row,samples=count,samples_below_limit=below,minimum_actual_free_mib=minimum,measurement='actual nvidia-smi memory.free'))
                except Exception as exc:errors.append(str(exc));return
                stop.wait(2)
    thread=threading.Thread(target=memory,daemon=True);thread.start()
    native_gate=c.gate
    def checked_gate(*args,**kwargs):
        require(not errors,'Memory monitor failed: '+str(errors));return native_gate(*args,**kwargs)
    c.gate=checked_gate
    try:
        with dispatch_context(binding_ref):return c.run()
    except Exception as exc:
        if c.owns_output:c.contain_failure(exc)
        raise
    finally:
        stop.set();thread.join(timeout=30)

def trainer(args,binding,cp):
    module=importlib.import_module('trainer');import peft
    sys.path.insert(0,str(S/'source/port-resume-0146350/opsd_signal_weight_port_resume_v1'));from trainer_launcher import repaired_setter
    i=args.index('--outer-manifest');ref=dict(path=args[i+1],sha256=args[args.index('--outer-manifest-sha256')+1]);outer=read(ref)
    old=read(outer['outer_old_adapter']);expected=old['all_rank_adapter_sha256'][0] if 'adapter' in old else old['adapter_tensor_sha256']
    def record(fields):atomic(Path(outer['outer_dir'])/('migration-outer-old-rank-'+os.environ['RANK']+'.json'),dict(**fields,rank=int(os.environ['RANK']),pid=os.getpid(),migration_publication=binding['publication']))
    native_state=module.verify_rank_state
    def verified_restore(state,manifest,rank,config,optimizer_state=None):
        result=native_state(state,manifest,rank,config,optimizer_state)
        if optimizer_state is not None:
            atomic(Path(outer['outer_dir'])/('migration-restored-rank-'+str(rank)+'.json'),dict(accepted=True,rank=rank,pid=os.getpid(),run_id=config['run_id'],restored_optimizer_updates=manifest['optimizer_updates'],restored_adapter_digest=manifest['all_rank_adapter_sha256'][rank],actual_optimizer_digest=module.optimizer_tensor_digest(optimizer_state),saved_optimizer_digest=state['optimizer_tensor_sha256'],source_checkpoint=outer['previous_complete_checkpoint'],source_rng_restoration='Unmodified native restore_rng follows this actual Adam verification',migration_publication=binding['publication']))
        return result
    module.verify_rank_state=verified_restore
    peft.set_peft_model_state_dict=repaired_setter(peft.set_peft_model_state_dict,module.tensor_digest,expected,record)
    sys.argv=[str(NATIVE/'trainer.py')]+args;return module.main()

def driver(binding,binding_ref):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'CPU-only driver required')
    sequence=(S/'study-sequence.lock').open('a+');fcntl.flock(sequence,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target_lease=Path('/home/shaoshitong/search-e1-config-study-20261004/experiment.lock').open('a+');fcntl.flock(target_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # G_POS/G_NEG stay on K4: existing actual25 terminal/ledger proofs bind this lineage.
    for item in binding['completed_groups']:
        value=read(item['ledger']);require(value['accepted'] and value['outer_iterations']==25 and len(value['rows'])==25,'Completed G ledger differs')
    for variant in ['H_POS','H_NEG']:
        path=S/('group-'+variant)/'receipts/config.json';ref=dict(path=str(path),sha256=SHAS[variant]);config=read(ref);output=Path(config['paths']['output_root'])
        if (output/'training-em-25-complete.json').exists():continue
        command=[config['paths']['python'],str(Path(__file__).resolve()),'--binding',binding_ref['path'],'--binding-sha256',binding_ref['sha256'],'--mode','controller','--','--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',config['run_id']]
        if (output/'run-identity.json').exists():command+=['--resume']
        env=dict(os.environ,E1_RUN_ID=config['run_id'],CUDA_VISIBLE_DEVICES='')
        log=S/'logs'/('hkust5-'+variant+'-'+str(int(time.time()))+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('x') as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT);atomic(S/'receipts/migration-active-child.json',dict(pid=child.pid,variant=variant,command=command,log=str(log)));rc=child.wait()
        atomic(S/'receipts'/('migration-'+variant+'-terminal.json'),dict(actual_wait_returncode=rc,variant=variant,child_pid=child.pid,binding=binding_ref,ended=time.time()))
        require(rc==0,'Migration group child failed: '+variant)
    atomic(S/'receipts/migration-training-complete.json',dict(accepted=True,groups=['G_POS','G_NEG','H_POS','H_NEG'],training_outer_total=100,evaluations_pending=True,binding=binding_ref,ended=time.time()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha256',required=True);parser.add_argument('--mode',choices=['driver','controller','sampling','teacher_diag','trainer'],required=True);parser.add_argument('args',nargs=argparse.REMAINDER);a=parser.parse_args();ref=dict(path=a.binding,sha256=a.binding_sha256);binding=read(ref);cp,prepare,gates=install(binding)
    require(a.args[:1]==['--'] or a.mode=='driver','Native args separator required');args=a.args[1:]
    if a.mode=='driver':return driver(binding,ref)
    if a.mode=='controller':return controller(binding,ref,args,cp,prepare,gates)
    if a.mode=='trainer':return trainer(args,binding,cp)
    module=importlib.import_module(a.mode);sys.argv=[str(NATIVE/(a.mode+'.py'))]+args;return module.main()
if __name__=='__main__':main()
