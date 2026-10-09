"""Read-only admission and explicit eight leases; no process signalling."""
import fcntl
from pathlib import Path
import os
import re
import subprocess
import time
try:
    from .utils import *
except ImportError:
    from utils import *

def same(a,b):return all(a.get(k)==b.get(k) for k in ('pid','uid','starttick'))

def native_admission(config):
    train,_=original_config(config);inv,_=checked_json(train['helper_source_manifest'])
    return fresh_module('results_original_cpu_admission',Path(train['paths']['helper_root'])/'gates.py',inv['files']['gates.py']['sha256'])

def process(config,pid):return native_admission(config).process(pid)

def publication(config):
    manifest,mp=checked_json(config['helper_source_manifest']);root=Path(config['paths']['helper_root']).resolve()
    require(Path(mp['path']).parent==root,'Evaluator manifest outside independent source')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='source-manifest.json'}
    require(actual==set(manifest['files']),'Evaluator source inventory changed')
    proofs=[]
    for name,row in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source path')
        fp=fileproof(root/name,row['sha256']);require(fp['bytes']==row['bytes'],'Source byte count changed');proofs.append(dict(relative_path=name,**fp))
    pub,pp=checked_json(config['gates']['publication']);commit=config['source_commit']
    require(re.fullmatch('[0-9a-f]{40}',commit or '') and pub.get('accepted') is True and pub.get('branch')=='search-e1'
       and pub.get('source_commit')==pub.get('remote_readback_sha')==commit
       and pub.get('run_id')==config['run_id'] and pub.get('source_manifest_sha256')==mp['sha256'],'Independent publication/readback binding missing')
    for fp in proofs+[dict(relative_path='source-manifest.json',**mp)]:
        matches=[r for r in pub['files'] if r['path']=='experiments/search_e1/control/opsd_pr100_periodic_recovery_v1/'+fp['relative_path']]
        require(len(matches)==1 and matches[0].get('equal') is True and matches[0]['sha256']==fp['sha256'] and matches[0]['bytes']==fp['bytes'],'Source not remotely readback equal')
    return dict(manifest=mp,publication=pp,source_commit=commit,files=proofs)

def oracle(config):
    train,_=original_config(config);receipt,_=checked_json(config['gates']['retriever_admission'])
    value=native_admission(config).retriever_admission(train,receipt)
    for key in ('uid','starttick'):
        if key in value['health']:require(value['health'][key]==receipt[key],'Oracle health identity differs')
    # Native health does not expose UID/tick in all versions; those were read
    # from the actual /proc process by retriever_admission, never inferred.
    gate,gp=checked_json(dict(path=receipt['full_gpu_gate_path'],sha256=receipt['full_gpu_gate_sha256']))
    recorded=gate['source_identity'];identity=recorded['identity']
    require(recorded['pid']==receipt['pid'] and recorded['source_identity_sha256']==receipt['source_identity_sha256']
        and hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==receipt['source_identity_sha256'],
        'Oracle full source-identity digest differs')
    sources=[fileproof(path,sha) for path,sha in identity['sources'].items()]
    for key in ('original_scan','original_encoder','assets_manifest'):
        sources.append(fileproof(identity[key]['path'],identity[key]['sha256']))
    for key in ('index_file','corpus_file'):
        old=identity[key];current=metadata(old['path'])
        require(current['path']==old['realpath'] and all(current[a]==old[b] for a,b in
            [('bytes','size'),('mtime_ns','mtime_ns'),('inode','inode'),('device','device')]),'Original oracle asset metadata changed')
    value.update(full_gate=gp,current_source_files=sources,asset_scope='Full accepted historical asset SHA plus current inode/size/mtime continuity; no repeated64GiB index SHA')
    return value

def observer(config,now=None):
    train,training_ref=original_config(config)
    binding,bp=checked_json(config['gates']['memory_observer']);identity=binding['observer_identity']
    observer_config,ocp=checked_json(config['gates']['memory_observer_config'])
    source=config['gates']['memory_observer_source'];sp=fileproof(source['path'],source['sha256'])
    actual=process(config,identity['pid'])
    require(binding['accepted'] is True and binding['config_sha256']==ocp['sha256']==training_ref['sha256']
        and binding['experiment_run_id']==observer_config['run_id']==train['run_id']
        and Path(bp['path']).resolve()==Path(train['gates']['memory_binding_path']).resolve()
        and Path(sp['path']).resolve()==Path(train['paths']['helper_root']).resolve()/'memory_observer.py'
        and same(actual,identity) and actual['run_id']==identity['run_id'] and identity['run_id']!=config['run_id']
        and identity['source_sha256']==sp['sha256'] and sp['path'] in actual['argv']
        and actual['argv'].count('--config')==1 and actual['argv'][actual['argv'].index('--config')+1]==ocp['path'],
        'Explicit independent observer identity/config/source differs')
    live,lp=checked_json(fileproof(binding['live_report_path']));age=(time.time() if now is None else now)-live['current']['sample_unix']
    require(0<=age<=30 and live['observer_identity']==identity and live['run_id']==observer_config['run_id'] and live['config_sha256']==ocp['sha256']
        and live['updated_unix']>=live['current']['sample_unix']
        and [d['uuid'] for d in live['current']['devices']]==config['owner']['gpu_uuids'],'Observer stale/foreign/UUID differs')
    require(live['current']['actual_free_limit_met'] is True,'Observer current actualfree floor failed')
    return dict(binding=bp,observer_config=ocp,source=sp,report=lp,age_seconds=age,current=live['current'],
        historical_samples_below_limit=live['samples_below_limit'],scope='Only fresh current directfree; historical violations are retained')

def storage(config):
    receipt,fp=checked_json(config['gates']['storage'])
    require(receipt.get('run_id')==config['run_id'],'Output admission bound to another run')
    # Reuse independently frozen CPU storage algorithm, not an old evaluation gate.
    adapted=dict(paths={'output_root':config['paths']['output_root']},resources=config['resources'])
    result=native_admission(config).storage_admission(adapted,receipt)
    return dict(receipt=fp,actual=result)

def locks(config):
    return list(config['resources']['shared_gpu_lease_paths'])+[config['resources']['group_lock'],config['resources']['study_lock'],str(Path(config['paths']['base'])/'results.lock')]

class Leases:
    def __init__(self,config):self.paths=locks(config);self.handles=[]
    def __enter__(self):
        require(len(self.paths)==len(set(self.paths))==8,'Require five shared+group+study+results leases')
        try:
            for path in self.paths:
                # Original locks must already exist. Only new results lock may be created.
                p=Path(path)
                if path!=self.paths[-1]:require(p.is_file(),'Missing original GPU lease')
                p.parent.mkdir(parents=True,exist_ok=True);h=p.open('a+');self.handles.append(h)
                fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.__exit__(None,None,None);raise
        return self
    def __exit__(self,*args):
        for h in reversed(self.handles):h.close()
        self.handles=[]

def training_release(config,owner):
    train,_=original_config(config);terminal,_=checked_json(config['training']['terminal'])
    require(terminal.get('accepted') is True and terminal.get('actual_exit_code')==0
        and terminal['run_id']==train['run_id'] and terminal['config_sha256']==config['training']['config']['sha256'],
        'Real successful group terminal not accepted')
    history,_=rows(fileproof(Path(train['paths']['output_root'])/'child-history.jsonl'))
    known=[x['identity'] for x in history]+[terminal['controller']]
    known=[{k:x[k] for k in ('pid','uid','starttick')} for x in known]
    roles=native_admission(config).scoped_cpu_processes(train)
    scan=scanner(config);result=scan.audit_same_uid(train['run_id'],owner['uid'],terminal['controller']['starttick'],known,
        [str(Path(train['paths']['helper_root'])/n) for n in ('sampling.py','trainer.py')]+[train['paths']['output_root']],
        exclusions=scan.load_exclusions(config['gates'].get('process_exclusions',[])),ignored_exact=[owner]+roles['identities'],ancestry_anchors=known)
    require(result['accepted_no_owned_or_opaque'],'Group training descendant/opaque remains; no new evaluation work')
    return result

def owned_release(config,owner,known):
    train,_=original_config(config);roles=native_admission(config).scoped_cpu_processes(train);scan=scanner(config)
    def environment(pid):
        value=scan.proc_environment(pid)
        if pid==owner['pid']:
            require(same(scan.proc_snapshot(pid),owner),'Evaluator controller PID reused');value=dict(value);value.pop('E1_RUN_ID',None)
        return value
    if owner['run_id']==config['run_id']:anchors=[owner]
    else:
        drivers=[x for x in roles['identities'] if same(x,owner)]
        require(len(drivers)==1,'External release owner is not the exact source-bound study driver')
        anchors=known
    result=scan.audit_same_uid(config['run_id'],owner['uid'],owner['starttick'],known,
        [str(Path(config['paths']['helper_root'])/name) for name in ('worker.py','cpu_merge.py')]+[config['paths']['output_root']],
        exclusions=scan.load_exclusions(config['gates'].get('process_exclusions',[])),ignored_exact=[owner]+roles['identities'],
        environment_reader=environment,ancestry_anchors=anchors)
    require(result['accepted_no_owned_or_opaque'],'Evaluator descendant/opaque process remains')
    return result

def resource(config,owner,children,phase,inventory=None,reader=None):
    require(phase in ('start','running','released'),'Unknown resource lifecycle')
    admit=native_admission(config);reader=reader or admit.process
    devices,apps=inventory if inventory is not None else admit.inventory()
    require([d['index'] for d in devices]==list(range(8)) and [d['uuid'] for d in devices]==config['owner']['gpu_uuids'],'GPU UUID allocation differs')
    bound,_=checked_json(config['gates']['retriever_admission']);own_cards=set()
    for app in apps:
        require(app['pid']!=owner['pid'],'CPU-only evaluator controller has a CUDA context')
        if app['pid']==bound['pid']:
            require(same(reader(app['pid']),bound) and app['gpu_uuid']==devices[7]['uuid'] and app['used_mib']<=1024,'Oracle PID/card/memory differs')
        else:
            require(phase=='running','GPU work not released before phase')
            admit.owned_gpu_process(app['pid'],config['run_id'],owner,children,reader=reader);own_cards.add(app['gpu_uuid'])
    for d in devices:
        floor=config['resources']['running_minimum_free_mib'] if phase=='running' and d['uuid'] in own_cards else config['resources']['startup_minimum_free_mib']
        require(d['actual_free_mib']>=floor,'Card-specific actual memory.free below gate')
    return dict(accepted=True,sample_unix=time.time(),devices=devices,apps=apps,phase=phase)

def require_launch(config,owner,children,phase):
    require(os.getuid()==config['owner']['uid'] and same(process(config,owner['pid']),owner),'Controller identity changed')
    pub=publication(config);st=storage(config);health=oracle(config);memory=observer(config)
    return dict(accepted=True,publication=pub,storage=st,oracle=health,memory_observer=memory,resource=resource(config,owner,children,phase))
