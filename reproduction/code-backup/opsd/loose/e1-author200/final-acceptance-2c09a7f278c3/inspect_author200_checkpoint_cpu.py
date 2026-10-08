"""Read actual saved author200 state on native CPU; never restore a training object.

The old inspection module is executed unchanged from verified source bytes.
Only its mathematical/storage helpers are reused, never its fixed-step entrypoint.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import types

SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
MODEL = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CLOSURE = '3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
FORMAL = 'bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c'
ACCEPTED_MANIFEST = '8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3'
ACCEPTED_RECEIPT = '812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b'
PUBLIC = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
HELPER = 'e949ebab98a49b12ac67fd6531089ef34513e276f169d9fd240c6aaccf623db4'
FULL_STATE = 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6'
GATE = 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb'
NUMEL = 3085938688
STEPS = (25,50,75,100,125,150,175,200)
KIND = 'OBSERVED_AUTHOR200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST'
SCOPE = 'NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION'
RUN_ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261'
DRIVER_PID = 1533312
DRIVER_START = 1829085316


def require(ok, message):
    if not ok: raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key,value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def number(raw):
        value = float(raw)
        require(math.isfinite(value), 'Nonfinite/overflowed JSON float')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def physical(path, root=None):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=True)==p and p.is_file() and p.stat().st_uid==os.getuid(), 'Physical owned regular file required')
    if root is not None: require(p.is_relative_to(root), 'File outside owned run')
    return p


def stable_sha(path):
    p = physical(path); before = p.stat(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
    after = p.stat()
    fields = ('st_dev','st_ino','st_size','st_uid','st_mtime_ns','st_ctime_ns')
    require(tuple(getattr(before,k) for k in fields)==tuple(getattr(after,k) for k in fields), 'Input file changed during read')
    return h.hexdigest()


def pinned(path, sha, pins, root=None):
    require(type(sha) is str and re.fullmatch('[0-9a-f]{64}',sha), 'Invalid external SHA')
    p=physical(path,root); require(stable_sha(p)==sha, 'Pinned file bytes differ: '+str(p))
    require(str(p) not in pins or pins[str(p)]==sha, 'Conflicting file pins')
    pins[str(p)]=sha
    return p


def read_pinned_json(path,pins):
    """Hash and parse the same stable opened bytes, not a second path read."""
    p=physical(path);expected=pins[str(p)];before=p.stat();fields=('st_dev','st_ino','st_size','st_uid','st_mtime_ns','st_ctime_ns')
    with p.open('rb') as stream:
        opened=os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode) and tuple(getattr(before,k) for k in fields)==tuple(getattr(opened,k) for k in fields), 'JSON opened file identity changed')
        require(opened.st_size<=32*1024**2, 'JSON exceeds inspection size limit')
        raw=stream.read();finished=os.fstat(stream.fileno())
    after=p.stat()
    require(tuple(getattr(before,k) for k in fields)==tuple(getattr(finished,k) for k in fields)==tuple(getattr(after,k) for k in fields), 'JSON changed while reading')
    require(len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==expected, 'Actual parsed JSON bytes differ from external pin')
    return strict_json(raw)


def exact_module(path, sha, name, pins):
    """Execute the unchanged bytes just hashed, avoiding import caches/pyc."""
    p=pinned(path,sha,pins); raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==sha, 'Source drift between hash and compilation')
    module=types.ModuleType(name); module.__file__=str(p); module.__package__=name.rpartition('.')[0]
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=sys.flags.optimize),module.__dict__)
    require(stable_sha(p)==sha, 'Helper source drifted during execution')
    return module


def clock(step):
    require(type(step) is int and step in STEPS, 'Only actually saved author200 steps are supported')
    return dict(global_step=step,next_global_step=step+1,adam_steps=16*step,
                scheduler_last_epoch=step,scheduler_step_count=step+1,
                teacher_cuda_updates=step,cursor=512*step,epoch=0,lr=5e-6)


def validate_inventory(value,step):
    c=clock(step)
    require(type(value) is dict and value.get('kind')==KIND, 'Observed inventory scope differs')
    require(type(value.get('version')) is int and value['version']==1, 'Observed inventory version differs')
    for key,expected in dict(source_commit=SOURCE,job_id='3261',run_root=RUN_ROOT,
        checkpoint_step=step,config_sha256=CONFIG,profile='author200',horizon=200,world_size=8,
        native_import_closure_sha256=CLOSURE,formal_manifest_sha256=FORMAL).items():
        require(type(value.get(key)) is type(expected) and value[key]==expected, 'Inventory identity differs: '+key)
    a=value['accepted5']
    for key,expected in dict(manifest_sha256=ACCEPTED_MANIFEST,signed_receipt_sha256=ACCEPTED_RECEIPT,
        public_key_sha256=PUBLIC,driver_pid=DRIVER_PID,driver_start_ticks=DRIVER_START).items():
        require(type(a.get(key)) is type(expected) and a[key]==expected, 'Accepted5 identity differs: '+key)
    workers=a['workers']; require(type(workers) is list and len(workers)==8, 'Accepted5 worker list incomplete')
    require([w['rank'] for w in workers]==list(range(8)) and all(type(w['rank']) is int for w in workers), 'Accepted5 rank identity differs')
    pids=[]
    for w in workers:
        require(type(w['pid']) is int and w['pid']>0, 'Accepted5 worker PID invalid');pids.append(w['pid'])
        validate_object_ids(w['object_ids'])
    require(len(set(pids))==8 and DRIVER_PID not in pids, 'Accepted5 PID aliases')
    require(type(value['rank_reports']) is list and len(value['rank_reports'])==8, 'Current rank reports incomplete')
    require(all(type(r) is dict and type(r.get('rank')) is int for r in value['rank_reports']) and
            [r['rank'] for r in value['rank_reports']]==list(range(8)), 'Current reports rank order/type differs')
    statuses=value['raw_statuses']
    require(type(statuses) is dict and set(statuses)=={'ranks','driver'} and type(statuses['ranks']) is list and
            statuses['ranks']==['FULL_STATE_SERIALIZED_OK']*8 and statuses['driver']=='DRIVER_STATE_SERIALIZED_OK', 'Original serialized statuses must be preserved')
    require(value['driver_report']['status']=='DRIVER_STATE_SERIALIZED_OK', 'Driver was not serialized at this step')
    return c


def validate_object_ids(value):
    require(type(value) is dict and set(value)=={'model','optimizer','scheduler','teacher'} and
            all(type(n) is int and n>0 for n in value.values()), 'Saved object IDs invalid')


def validate_rank_report(report,rank,step,pid):
    require(type(report) is dict, 'Rank report must be a mapping')
    expected=dict(version=3,rank=rank,world_size=8,step=step,pid=pid,scheduler_last_epoch=step,
                  optimizer_state_entries=37,model_dtensor_entries=435)
    for key,target in expected.items():
        require(type(report.get(key)) is int and report[key]==target, 'Rank report numeric identity/type differs: '+key)
    require(report['status']=='FULL_STATE_SERIALIZED_OK' and report['algorithm']=='pure_opsd' and
            report['model_state_dict_type']=='SHARDED_STATE_DICT', 'Original serialized rank scope differs')
    validate_object_ids(report['object_ids'])


def validate_driver_report(report,step):
    require(type(report) is dict and report.get('status')=='DRIVER_STATE_SERIALIZED_OK', 'Original serialized driver scope differs')
    for key,target in dict(pid=DRIVER_PID,step=step,uid=os.getuid()).items():
        require(type(report.get(key)) is int and report[key]==target, 'Driver report numeric identity/type differs: '+key)
    require(type(report.get('job_id')) is str and report['job_id']=='3261', 'Driver report actual stored job type differs')


def scheduler(value,step):
    c=clock(step)
    require(type(value) is dict, 'Scheduler state must be a mapping')
    for key,target in [('last_epoch',step),('_step_count',step+1)]:
        require(type(value[key]) is int and value[key]==target, 'Scheduler clock differs: '+key)
    require(value['base_lrs']==[5e-6] and value['_last_lr']==[5e-6] and
            all(type(n) is float and math.isfinite(n) for n in value['base_lrs']+value['_last_lr']), 'Constant scheduler LR differs')
    require(value['lr_lambdas']==[None], 'Scheduler lambda state differs')
    return dict(last_epoch=step,step_count=c['scheduler_step_count'],base_lrs=value['base_lrs'],last_lr=value['_last_lr'])


def validate_sampler(current,old,step,torch):
    expected={'size','shuffle','seed','epoch','cursor','consumed_cursor','order','generator'}
    require(type(current) is dict and set(current)==expected and type(old) is dict and set(old)==expected, 'Sampler component schema differs')
    for key,target in dict(size=169028,shuffle=True,seed=42,epoch=0,cursor=512*step,consumed_cursor=512*step).items():
        require(type(current[key]) is type(target) and current[key]==target, 'Sampler position/config differs: '+key)
    require(old['size']==169028 and old['epoch']==0 and old['cursor']==old['consumed_cursor']==2560 and type(old['cursor']) is int, 'Accepted5 sampler position differs')
    order=current['order']; require(type(order) is list and len(order)==169028 and all(type(x) is int for x in order) and set(order)==set(range(169028)), 'Sampler permutation differs')
    for key in ('size','shuffle','seed','epoch','order'):
        require(type(current[key]) is type(old[key]) and current[key]==old[key], 'Sampler drift versus accepted5: '+key)
    for state in (current['generator'],old['generator']):
        require(torch.is_tensor(state) and state.device.type=='cpu' and state.dtype==torch.uint8 and state.ndim==1, 'Sampler generator state invalid')
        torch.Generator().set_state(state)
    require(torch.equal(current['generator'],old['generator']), 'Sampler generator changed within epoch0')
    return dict(size=169028,epoch=0,sampler_cursor=512*step,consumed_cursor=512*step,
                unchanged_epoch0_order=True,unchanged_epoch0_generator=True,seed=42,shuffle=True)


def validate_algorithm(value,config,step):
    expected=dict(version=1,method='pure_opsd',global_step=step,distillation=config['baselines']['distillation'],
        resume_scope='same_live_objects_same_topology',outcome_rewards_train_policy=False)
    require(canonical(value)==canonical(expected), 'Pure OPSD saved algorithm/configuration differs')


def execution_identity(job,step):
    require(sys.platform=='linux' and socket.gethostname().split('.')[0]=='h100-10', 'Actual CPU execution must be on h100-10')
    require(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and os.environ.get('SLURM_JOB_ID')==job, 'Actual execution job differs')
    require(type(step) is str and re.fullmatch(r'(?:[0-9]+|batch|extern)',step) and os.environ.get('SLURM_STEP_ID')==step, 'Actual execution Slurm step differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CUDA must be explicitly hidden')
    stat=Path('/proc/self/stat').read_text(); fields=stat[stat.rfind(')')+2:].split();start=int(fields[19])
    status=Path('/proc/self/status').read_text();uid=re.search(r'^Uid:\s+(.*)$',status,re.M)
    require(uid is not None, 'Actual CPU UID identity missing');uids=[int(v) for v in uid.group(1).split()]
    require(len(uids)==4 and uids==[os.getuid()]*4 and start>0, 'Actual CPU identity differs')
    cgroup=Path('/proc/self/cgroup').read_text()
    require(re.search(r'(?:^|/)job[_-]'+re.escape(job)+r'(?:\.scope)?(?:/|$)',cgroup,re.M), 'CPU process is outside actual own job cgroup')
    require(re.search(r'(?:^|/)step[_-]'+re.escape(step)+r'(?:\.scope)?(?:/|$)',cgroup,re.M), 'CPU process is outside actual claimed Slurm step cgroup')
    return dict(node=socket.gethostname(),uid=os.getuid(),uids=uids,pid=os.getpid(),start_ticks=start,
                execution_job_id=job,execution_step_id=step,cgroup=cgroup,
                stored_training_job_id='3261',python_executable=sys.executable,python_version=sys.version)


def observed_files(inventory,pins):
    root=Path(inventory['run_root']); step=inventory['checkpoint_step'];paths={};identities=set()
    expected_hf={'added_tokens.json','config.json','generation_config.json','merges.txt','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}|{f'model-{n:05d}-of-00003.safetensors' for n in (1,2,3)}
    expected={f'actor/{name}' for name in expected_hf}|{f'actor/training_state/rank_{r}.{suffix}' for r in range(8) for suffix in ('pt','json')}|{'driver/driver.pt','driver/driver.json','algorithm'}
    require(set(inventory['files'])==expected, 'Observed checkpoint member inventory differs')
    for role,rec in inventory['files'].items():
        require(type(rec) is dict and set(rec)=={'physical_path','sha256','bytes','device','inode','uid'}, 'Observed file record schema differs')
        require(all(type(rec[k]) is int and rec[k]>=0 for k in ('bytes','device','inode','uid')), 'Observed file metadata type differs')
        if role.startswith('actor/'):
            target=root/'checkpoints/actor'/f'global_step_{step}'/role[len('actor/'):]
        elif role.startswith('driver/'):
            target=root/'full_state'/f'global_step_{step}'/role[len('driver/'):]
        else:target=root/'checkpoints/algorithm'/f'global_step_{step}'/'state.json'
        require(rec['physical_path']==str(target), 'Checkpoint physical member path differs')
        p=pinned(rec['physical_path'],rec['sha256'],pins,root);st=p.stat()
        require((st.st_size,st.st_dev,st.st_ino,st.st_uid)==tuple(rec[k] for k in ('bytes','device','inode','uid')), 'Observed file metadata differs')
        identity=(st.st_dev,st.st_ino);require(identity not in identities, 'New checkpoint members alias an inode');identities.add(identity);paths[role]=p
    return paths


def source_modules(repo,inventory,pins):
    require(repo.is_absolute() and repo.resolve(strict=True)==repo and str(repo)==inventory['source_root'], 'Exact physical source root differs')
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and
        not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(), 'Frozen repository must be clean source188')
    path=pinned(inventory['native_import_closure_path'],CLOSURE,pins,Path(inventory['run_root']));closure=read_pinned_json(path,pins)
    require(set(closure)=={'modules'} and len(closure['modules'])==210, 'Native source closure inventory differs')
    for name,rec in closure['modules'].items():
        require(set(rec)=={'path','sha256'}, 'Native source closure record differs');pinned(rec['path'],rec['sha256'],pins)
        if name.startswith(('search_r1','verl','reproduction.baselines')):
            stem=repo.joinpath(*name.split('.'));expected=stem.with_suffix('.py')
            if not expected.is_file():expected=stem/'__init__.py'
            require(rec['path']==str(expected), 'Native source module path differs')
    sys.path[:]=[str(repo)]+[v for v in sys.path if v and Path(v).resolve()!=repo]
    # Parent packages have no executable training entrypoint; verify their code
    # and every dependency origin using the pinned bootstrap after loading.
    importlib.import_module('search_r1.baselines')
    for name in ('holdout','sharedholdout','e1_profile','full_state','checkpoint_gate'):
        qualified='search_r1.baselines.'+name;rec=closure['modules'][qualified]
        if name=='full_state':require(rec['sha256']==FULL_STATE, 'Frozen state helper pin differs')
        if name=='checkpoint_gate':require(rec['sha256']==GATE, 'Frozen gate pin differs')
        module=exact_module(rec['path'],rec['sha256'],qualified,pins);sys.modules[qualified]=module
        setattr(sys.modules['search_r1.baselines'],name,module)
    rec=closure['modules']['reproduction.baselines.e1_import_bootstrap']
    checker=exact_module(rec['path'],rec['sha256'],'reproduction.baselines.e1_import_bootstrap',pins)
    sys.modules[checker.__name__]=checker
    _,count=checker.verify_imports(repo,closure,required=())
    require(count>0, 'No actual pinned native functions audited')
    expected={str(p.relative_to(repo)):stable_sha(p) for namespace in ('search_r1','verl') for p in sorted((repo/namespace).rglob('*.py'))}
    return sys.modules['search_r1.baselines.full_state'],sys.modules['search_r1.baselines.e1_profile'],sys.modules['search_r1.baselines.checkpoint_gate'],expected,count


def accepted_anchor(rec,pins,root):
    require(type(rec) is dict and set(rec) in ({'path','sha256'},{'physical_path','sha256'}), 'Accepted anchor record schema differs')
    return pinned(rec.get('path',rec.get('physical_path')),rec['sha256'],pins,root)


def accepted5(inventory,gate,pins):
    a=inventory['accepted5'];root=Path(inventory['run_root'])
    mp=pinned(a['manifest_path'],ACCEPTED_MANIFEST,pins,root);manifest=read_pinned_json(mp,pins)
    rp=pinned(a['signed_receipt_path'],ACCEPTED_RECEIPT,pins,root);receipt=read_pinned_json(rp,pins)
    pk=pinned(a['public_key_path'],PUBLIC,pins,root);public=read_pinned_json(pk,pins)
    require(set(receipt)=={'payload','signature_hex'}, 'Accepted signed envelope schema differs')
    gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],public)
    payload=receipt['payload']
    require(payload['checkpoint_manifest_sha256']==ACCEPTED_MANIFEST and payload['checkpoint_files']==manifest['files'] and
        payload['source_commit']==SOURCE and payload['config_sha256']==CONFIG and payload['job_id']=='3261' and payload['driver_pid']==DRIVER_PID and
        payload['global_step']==payload['actual_outer_updates']==5 and payload['next_global_step']==6 and payload['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW', 'Accepted5 signed scope differs')
    anchors=inventory['anchor_files']
    require(type(anchors) is dict and all(type(rec) is dict and set(rec)=={'physical_path','sha256'} for rec in anchors.values()), 'Accepted5 observed anchor schema differs')
    require({role:dict(path=rec['physical_path'],sha256=rec['sha256']) for role,rec in anchors.items()}==manifest['files'], 'Accepted5 original anchor map differs')
    config_path=accepted_anchor(anchors['resolved_config'],pins,root);config=read_pinned_json(config_path,pins)
    require(hashlib.sha256(canonical(config)).hexdigest()==CONFIG, 'Resolved configuration differs')
    audit_path=accepted_anchor(payload['evidence']['rank_audit'],pins,root);audit=read_pinned_json(audit_path,pins)
    require(audit['passed'] is True and audit['source_commit']==SOURCE and audit['job_id']=='3261' and audit['checkpoint_manifest_sha256']==ACCEPTED_MANIFEST and
        audit['scope']=='NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE' and len(audit['ranks'])==8, 'Accepted5 native CPU witness differs')
    for rank,w in enumerate(a['workers']):
        rec=anchors[f'actor/training_state/rank_{rank}.json'];p=accepted_anchor(rec,pins,root)
        require(w['report_path']==str(p) and w['report_sha256']==rec['sha256'], 'Accepted5 worker report binding differs')
        report=read_pinned_json(p,pins);prior=audit['ranks'][rank]
        require(report['status']=='EXACT_FULL_STATE_RELOAD_OK' and report['rank']==rank and report['pid']==w['pid'] and report['object_ids']==w['object_ids'] and
            prior['rank']==rank and prior['pid']==w['pid'] and prior['object_ids']==w['object_ids'], 'Accepted5 original worker identity differs')
    return manifest,payload,config,audit


def teacher_check(teacher,report,base,assets,base_digest,step,helper,digest):
    require(type(teacher) is dict and helper.teacher_report_exact({k:v for k,v in teacher.items() if k!='buffers'},report), 'Raw teacher/JSON projection differs')
    require(type(teacher['version']) is int and teacher['version']==1 and teacher['parameter_count']==NUMEL and type(teacher['parameter_count']) is int and
        teacher['storage_device']=='cpu' and type(teacher['execution_device']) is str and re.fullmatch(r'cuda(?::[0-9]+)?',teacher['execution_device']) and
        teacher['model_content_digest']==base_digest, 'Teacher fixed FP32 identity differs')
    require(Path(teacher['asset_directory'])==base.directory and teacher['asset_files']==assets, 'Teacher official full assets differ')
    counters=teacher['counters'];require(set(counters)==set(helper.COUNTERS) and
        all(type(n) in (int,float) and math.isfinite(n) and n>=0 for n in counters.values()), 'Teacher counters invalid')
    require(type(counters['cuda_update_entries']) is int and counters['cuda_update_entries']==step and type(counters['scored_tokens']) is int and counters['scored_tokens']>0, 'Teacher cumulative real updates differ')
    require(all(teacher['model_config'][key]==base.config[key] and type(teacher['model_config'][key]) is type(base.config[key]) for key in
        ('model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','tie_word_embeddings')), 'Teacher architecture differs')
    caches=teacher['caches'];require(type(caches) is dict, 'Teacher cache metadata missing')
    for name,values in caches.items():
        require(type(name) is str and type(values) is dict and bool(values) and set(values)<={'max_seq_len_cached','original_max_seq_len'} and
            all(type(n) is int and n>=0 for n in values.values()), 'Teacher history cache metadata invalid')
    require(type(teacher['buffers']) is dict and all(type(k) is str for k in teacher['buffers']), 'Teacher buffers schema differs')
    for tensor in teacher['buffers'].values():helper.finite(tensor)
    require(digest(teacher['buffers'])==teacher['buffer_content_digest'], 'Saved teacher buffer bytes differ')
    require(type(teacher['model_bytes']) is int and teacher['model_bytes']==NUMEL*4+sum(t.numel()*t.element_size() for t in teacher['buffers'].values()), 'Teacher model/buffer byte count differs')
    return dict(content_digest=base_digest,cuda_update_entries=step,counters=counters,buffer_content_digest=teacher['buffer_content_digest'],history_caches=caches,
        direct_saved_teacher_parameter_arrays=False,fixed_official_assets_and_content_digest=True)


def inspect(a):
    execution=execution_identity(a.execution_job_id,a.execution_step_id);pins={}
    self_path=physical(Path(__file__).resolve());self_sha=stable_sha(self_path)
    ip=pinned(a.inventory,a.inventory_sha256,pins);inventory=read_pinned_json(ip,pins);c=validate_inventory(inventory,a.step)
    repo=Path(a.repo);state,profile,gate,source_files,code_count=source_modules(repo,inventory,pins)
    import torch
    import numpy as np
    require(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(), 'Native Torch2.4 CPU required')
    helper=exact_module(a.helper,HELPER,'author200_immutable_math_helper',pins)
    paths=observed_files(inventory,pins)
    formal_path=pinned(inventory['formal_manifest_path'],FORMAL,pins);formal=read_pinned_json(formal_path,pins)
    require(formal['source_commit']==SOURCE, 'Original formal tool/source binding differs')
    manifest,payload,config,prior=accepted5(inventory,gate,pins)
    from omegaconf import OmegaConf
    require(profile.validate_profile(OmegaConf.create(config))==config, 'Actual pure author200 configuration differs')
    require(config['baselines']['opsd_e1']['profile']=='author200' and config['actor_rollout_ref']['actor']['use_kl_loss'] is False and
        config['actor_rollout_ref']['actor']['kl_loss_coef']==config['algorithm']['kl_ctrl']['kl_coef']==0, 'PG/reference KL configuration differs')
    bp=pinned(a.base_manifest,MODEL,pins);require(a.base_manifest_sha256==MODEL, 'Official model external pin differs')
    official=read_pinned_json(bp,pins);base=helper.Base(Path(a.base_model),official,pins)
    require(str(base.directory)==config['actor_rollout_ref']['model']['path']==config['baselines']['distillation']['teacher_path'], 'Actual original model path differs')
    base_digest=base.named_parameter_digest();assets=helper.base_asset_inventory(base.directory,{r['file'] for r in official['files']},bp,pins)
    hf=helper.Export(paths,set(base.headers)|{'lm_head.weight'});require(len(hf.shards)==3 and len(hf.headers)==435, 'Complete HF export geometry differs')
    hf_files={str(p):pins[str(p)] for role,p in paths.items() if role.startswith('actor/') and '/training_state/' not in role}
    ranks=[];configs=set()
    for rank in range(8):
        pt=paths[f'actor/training_state/rank_{rank}.pt'];report=read_pinned_json(paths[f'actor/training_state/rank_{rank}.json'],pins);w=inventory['accepted5']['workers'][rank]
        require(canonical(report)==canonical(inventory['rank_reports'][rank]), 'Observer/original rank report differs')
        validate_rank_report(report,rank,a.step,w['pid'])
        require(report['status']=='FULL_STATE_SERIALIZED_OK' and type(report['step']) is int and report['step']==a.step and report['rank']==rank and
            report['world_size']==8 and report['version']==3 and report['algorithm']=='pure_opsd' and report['model_state_dict_type']=='SHARDED_STATE_DICT', 'Current serialized rank scope differs')
        require(report['path']==str(pt) and report['sha256']==report['state_sha256']==pins[str(pt)], 'Raw rank full-file SHA binding differs')
        raw=torch.load(pt,map_location='cpu',weights_only=False)
        require(state.state_digest(raw)==report['state_digest'], 'Raw full checkpoint content digest differs')
        for key,target in dict(version=3,algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',rank=rank,world_size=8,step=a.step,pid=w['pid']).items():
            require(type(raw[key]) is type(target) and raw[key]==target, 'Raw saved rank identity differs: '+key)
        validate_object_ids(raw['object_ids']);require(raw['object_ids']==report['object_ids']==w['object_ids'] and raw['pid']==report['pid'], 'Stored training object/PID continuity differs')
        identity=raw['recovery_identity'];require(identity['source_closure']==dict(version=1,git_source_commit=SOURCE,files=source_files) and identity['worker_source_sha256']==stable_sha(repo/'verl/workers/fsdp_workers.py'), 'Raw complete recovery source closure differs')
        require(identity['actor_config_digest']==prior['ranks'][rank]['actor_config_digest'] and re.fullmatch('[0-9a-f]{64}',identity['actor_config_digest']), 'Actor configuration drifted since accepted5');configs.add(identity['actor_config_digest'])
        model=raw['model'];require(set(model)==set(base.headers)|{'lm_head.weight'}, 'Complete435 native model keys differ')
        layouts={};changes={};numel=0
        for key,tensor in model.items():
            layouts[key]=helper.layout(tensor,rank);local=tensor.to_local();target=base.local(key,rank)
            original='model.embed_tokens.weight' if key=='lm_head.weight' else key
            require(tuple(tensor.shape)==tuple(base.headers[original]['shape']) and hf.headers[key]['shape']==list(tensor.shape) and local.shape==target.shape, 'Stored/native/HF model geometry differs')
            hf.equal_slice(key,rank,local);changes[key]=int(torch.count_nonzero(local!=target))
            if key!='lm_head.weight':numel+=local.numel()
        require(numel==NUMEL//8 and torch.equal(model['lm_head.weight'].to_local(),model['model.embed_tokens.weight'].to_local()) and
            sum(n for key,n in changes.items() if key!='lm_head.weight')>0, 'Student tied/fullparameter/delta evidence differs')
        opt=helper.adam(raw['optimizer'],NUMEL//8,expected_steps=c['adam_steps']);require(len(opt['entries'])==37, 'Full37 native Adam parameter states required')
        require([(v['parameter_id'],v['shape'],v['dtype']) for v in opt['entries']]==[(v['parameter_id'],v['shape'],v['dtype']) for v in prior['ranks'][rank]['optimizer_moments']], 'Native Adam ID/flatlayout changed since accepted5')
        sched=scheduler(raw['scheduler'],a.step);rng=helper.rng(raw['rng'],True)
        require([v['device'] for v in rng['cuda']]==[v['device'] for v in prior['ranks'][rank]['rng_content']['cuda']], 'Stored rank-local RNG device changed since accepted5')
        require(type(report['scheduler_last_epoch']) is int and report['scheduler_last_epoch']==a.step and report['optimizer_state_entries']==37 and report['model_dtensor_entries']==435, 'Original rank component counts differ')
        teacher=teacher_check(raw['teacher'],report['teacher'],base,assets,base_digest,a.step,helper,state.state_digest)
        original_report=read_pinned_json(w['report_path'],pins)
        require(helper.teacher_report_exact(raw['teacher']['model_config'],original_report['teacher']['model_config']) and
            raw['teacher']['execution_device']==original_report['teacher']['execution_device'], 'Fixed teacher configuration/execution topology changed since accepted5')
        ranks.append(dict(rank=rank,pid=raw['pid'],object_ids=raw['object_ids'],raw_saved_status=report['status'],state_sha256=pins[str(pt)],state_digest=report['state_digest'],
            actor_config_digest=identity['actor_config_digest'],component_digests={k:state.state_digest(raw[k]) for k in ('model','optimizer','scheduler','rng','teacher')},
            model_dtensor_entries=435,model_hf_slice_exact=True,model_hf_slice_keys=sorted(model),model_layouts=layouts,unique_local_model_numel=numel,
            changed_elements_vs_original_fp32=changes,optimizer_state_entries=37,optimizer_step_values=opt['steps'],optimizer_moments=opt['entries'],scheduler=sched,rng_content=rng,teacher=teacher))
        del raw,model
    require(len(configs)==1 and len({r['pid'] for r in ranks})==8, 'Stored worker configuration/PID aliases')
    driver_pt=paths['driver/driver.pt'];report=read_pinned_json(paths['driver/driver.json'],pins);validate_driver_report(report,a.step);driver=torch.load(driver_pt,map_location='cpu',weights_only=False)
    require(canonical(report)==canonical(inventory['driver_report']) and report['status']=='DRIVER_STATE_SERIALIZED_OK' and report['pid']==DRIVER_PID and
        report['step']==a.step and report['uid']==os.getuid() and str(report['job_id'])=='3261' and report['state_sha256']==pins[str(driver_pt)] and report['digest']==state.state_digest(driver), 'Actual serialized driver digest/identity differs')
    for key,target in dict(version=1,method='pure_opsd',global_step=a.step,next_global_step=a.step+1,actual_outer_updates=a.step,epoch=0,
        pid=DRIVER_PID,uid=os.getuid(),job_id='3261',source_commit=SOURCE,config_sha256=CONFIG).items():
        require(type(driver[key]) is type(target) and driver[key]==target, 'Raw driver identity differs: '+key)
    old_pt=accepted_anchor(inventory['anchor_files']['driver/driver.pt'],pins,Path(inventory['run_root']));old=torch.load(old_pt,map_location='cpu',weights_only=False)
    old_report=prior['driver'];require(state.state_digest(old)==old_report['state_digest'] and old_report['state_sha256']==pins[str(old_pt)], 'Actual accepted5 driver full PT digest differs')
    sampler=validate_sampler(driver['sampler'],old['sampler'],a.step,torch);rng=helper.rng(driver['rng'],False)
    algorithm=read_pinned_json(paths['algorithm'],pins);require(canonical(driver['algorithm'])==canonical(algorithm), 'Driver/controller original saved state differs');validate_algorithm(algorithm,config,a.step)
    require(not torch.cuda.is_initialized(), 'Inspection initialized CUDA')
    for path,pin in pins.items():require(stable_sha(path)==pin, 'Input changed during inspection')
    require(source_files=={str(p.relative_to(repo)):stable_sha(p) for namespace in ('search_r1','verl') for p in sorted((repo/namespace).rglob('*.py'))}, 'Frozen source changed during inspection')
    require(stable_sha(self_path)==self_sha, 'Inspector bytes changed')
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(), 'Frozen source changed')
    return dict(version=1,scope=SCOPE,passed=True,checkpoint_step=a.step,source_commit=SOURCE,config_sha256=CONFIG,job_id='3261',run_root=RUN_ROOT,
        inventory_path=str(ip),inventory_sha256=a.inventory_sha256,inspector_source_sha256=self_sha,immutable_math_helper_sha256=HELPER,
        execution_identity=execution,cpu_runtime=dict(torch_version=torch.__version__,torch_file=str(Path(torch.__file__).resolve()),numpy_file=str(Path(np.__file__).resolve())),
        native_import_closure_sha256=CLOSURE,native_closure_module_count=210,actual_import_function_count=code_count,formal_manifest_sha256=FORMAL,
        accepted5_manifest_sha256=ACCEPTED_MANIFEST,accepted5_signed_receipt_sha256=ACCEPTED_RECEIPT,original_raw_statuses=inventory['raw_statuses'],
        clock=c,ranks=ranks,driver=dict(pid=DRIVER_PID,state_sha256=pins[str(driver_pt)],state_digest=report['digest'],raw_saved_status=report['status'],
            sampler=sampler,sampler_digest=state.state_digest(driver['sampler']),rng_content=rng,algorithm_digest=state.state_digest(algorithm)),
        base_named_parameter_digest=base_digest,hf_all_rank_slices_exact=True,hf_files=hf_files,all_files=pins,cuda_is_initialized=False,
        limitations=['Stored serialized content only; no live or new-process restore is performed or claimed',
                    'Object IDs and PIDs are stored training identities, not live witnesses of this CPU process',
                    'Named-to-flat FSDP handle reconstruction is not performed',
                    'Fixed teacher content is anchored to original assets and digest; raw payload stores history buffers rather than all teacher parameter arrays',
                    'This mechanical inspection does not show improved reward or final evaluation completion'])


def write_exclusive(path,value):
    p=Path(path);require(p.is_absolute() and p.parent.resolve(strict=True)==p.parent and not p.is_symlink(), 'Physical output parent required')
    with p.open('xb') as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    return stable_sha(p)


def main():
    p=argparse.ArgumentParser()
    for name in ('inventory','inventory-sha256','repo','helper','base-model','base-manifest','base-manifest-sha256','execution-job-id','execution-step-id','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--step',type=int,choices=STEPS,required=True);a=p.parse_args()
    result=inspect(a);sha=write_exclusive(a.output,result)
    print(json.dumps(dict(scope=SCOPE,checkpoint_step=a.step,stored_training_job_id='3261',execution_job_id=a.execution_job_id,output=a.output,output_sha256=sha,passed=True,cuda_is_initialized=False)))


if __name__=='__main__':main()
