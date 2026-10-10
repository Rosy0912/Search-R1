"""Published external recovery dispatch; original inputs remain byte-identical.

Only the exact controller and torchrun trainer commands are redirected. The
original study/controller execute from their original filename via runpy. No
CUDA/model import occurs in either CPU role. Every overlay role records its
actual entry argv as well as the original code executed; never fake /proc argv.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid


def require(ok, message):
    if not ok: raise ValueError(message)


def read_file(path):
    path = Path(path).resolve()
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Recovery input changed during read')
    return raw,dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def file_ref(path):return read_file(path)[1]


def checked_json(reference):
    raw,proof = read_file(reference['path'])
    require(proof['sha256'] == reference['sha256'], 'Pinned recovery input SHA differs')
    return json.loads(raw), proof


def actual_self_identity():
    """Linux kernel identity, including python argv; not sys.argv shorthand."""
    root=Path('/proc')/str(os.getpid())
    before=(root/'stat').read_text();tail=before[before.rfind(')')+2:].split()
    argv=[v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v]
    environ=dict(v.decode().split('=',1) for v in (root/'environ').read_bytes().split(b'\0') if b'=' in v)
    after=(root/'stat').read_text();after_tail=after[after.rfind(')')+2:].split()
    require(tail[19]==after_tail[19] and int(before.split(' ',1)[0])==os.getpid(),'Entry kernel identity changed')
    return dict(pid=os.getpid(),uid=os.getuid(),starttick=int(tail[19]),ppid=int(tail[1]),pgid=int(tail[2]),
        argv=argv,run_id=environ.get('E1_RUN_ID'))


def parse_original(mode, argv):
    parser = argparse.ArgumentParser(add_help=False)
    if mode == 'study':
        parser.add_argument('--study-manifest', required=True)
        parser.add_argument('--study-manifest-sha256', required=True)
        parser.add_argument('--resume', action='store_true')
    else:
        parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
        if mode == 'controller':
            parser.add_argument('--run-id', required=True); parser.add_argument('--resume', action='store_true')
        elif mode == 'evaluation':
            parser.add_argument('--completed-outer', required=True, type=int, choices=(15,25))
            parser.add_argument('--checkpoint', required=True); parser.add_argument('--checkpoint-sha256', required=True)
            parser.add_argument('--output-dir', required=True); parser.add_argument('--receipt', required=True)
        elif mode == 'trainer':
            parser.add_argument('--outer-manifest', required=True); parser.add_argument('--outer-manifest-sha256', required=True)
            parser.add_argument('--resume-checkpoint'); parser.add_argument('--resume-checkpoint-sha256')
            parser.add_argument('--retry-capsule-manifest'); parser.add_argument('--retry-capsule-sha256')
        else: raise ValueError('Unknown recovery execution role')
    # Each option must occur once; argparse normally accepts repeated options.
    options = [item for item in argv if item.startswith('--')]
    require(len(options) == len(set(options)), 'Duplicate original entry option')
    parsed = parser.parse_args(argv)
    if mode == 'study': require(parsed.resume, 'Recovery supervisor must explicitly resume the same study')
    return parsed


def overlay_command(python, entry, mode, recovery_ref, original_arguments, original_entry=None):
    command = [python, str(entry), '--mode', mode, '--recovery-ref', recovery_ref['path'],
               '--recovery-sha256', recovery_ref['sha256']]
    if mode == 'evaluation':
        require(original_entry is not None, 'Evaluator must expose actual original source path in /proc argv')
        command += ['--original-entry', str(original_entry)]
    else:
        require(original_entry is None, 'Original-entry option only belongs to evaluator')
    return command + ['--'] + list(original_arguments)


def rewrite_command(command, configs, original_root, entry, recovery_ref):
    """Return (new argv, original role) only for two exact whitelisted entries."""
    if not isinstance(command, (list, tuple)) or not all(isinstance(x, str) for x in command):
        return command, None
    command = list(command); root = Path(original_root)
    if len(command) >= 2 and command[1] == str(root / 'controller.py'):
        parsed = parse_original('controller', command[2:])
        reference = dict(path=parsed.config, sha256=parsed.config_sha256)
        require(reference['path'] in configs, 'Controller dispatch is outside four pinned configs')
        config, expected = configs[reference['path']]
        require(reference == expected and command[0] == config['paths']['python']
                and parsed.run_id == config['run_id'], 'Original controller config/python/run changed')
        return overlay_command(command[0], entry, 'controller', recovery_ref, command[2:]), 'controller'
    prefix = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    if len(command) >= 6 and command[1:5] == prefix and command[5] == str(root / 'trainer.py'):
        parsed = parse_original('trainer', command[6:])
        reference = dict(path=parsed.config, sha256=parsed.config_sha256)
        require(reference['path'] in configs, 'Trainer dispatch is outside four pinned configs')
        config, expected = configs[reference['path']]
        require(reference == expected and command[0] == config['paths']['python'], 'Original trainer config/python changed')
        replacement = overlay_command(command[0], entry, 'trainer', recovery_ref, command[6:])
        return command[:5] + replacement[1:], 'trainer'
    if len(command) >= 2 and command[1] == str(root / 'evaluation.py'):
        from evaluation_retry import evaluation_dispatch
        arguments = command[2:]; parsed = parse_original('evaluation', arguments)
        evaluation_dispatch(parsed, command[0], configs, original_root)
        return overlay_command(command[0], entry, 'evaluation', recovery_ref, arguments,
                               root / 'evaluation.py'), 'evaluation'
    return command, None


@contextlib.contextmanager
def exact_dispatch(configs, original_root, recovery_ref, before_spawn, receipt):
    """Keep actual Popen handle/wait/env unchanged; never redirect foreign jobs."""
    original = subprocess.Popen; entry = Path(__file__).resolve()
    def dispatch(command, *args, **kwargs):
        replacement, role = rewrite_command(command, configs, original_root, entry, recovery_ref)
        if role is not None:
            require(kwargs.get('shell', False) is False, 'Recovery cannot wrap a shell command')
            before_spawn()
            receipt(dict(original_role=role, original_argv=list(command), actual_dispatch_argv=replacement,
                         env_is_inherited_unchanged=True, actual_returncode_not_yet_known=True))
        return original(replacement, *args, **kwargs)
    subprocess.Popen = dispatch
    try: yield
    finally: subprocess.Popen = original


def evaluation_recovery_gate(repair, configs, gates):
    """This failure is15 completed outers/150 Adam; never the obsolete zero guard."""
    from failure_gate import validate_recovery_start
    positive, ref = next(value for value in configs.values() if value[0]['variant'] == 'G_POS')
    return validate_recovery_start(repair, positive, ref, gates, configs)


def checked_json_ref_file(ref):
    proof=file_ref(ref['path']);require(proof['sha256']==ref['sha256'],'Pinned original artifact changed');return proof


def scoped_roles(config, gates, repair_ref, repair, study_ref, actual_entry_argv,observer_optional=False):
    """Native CPU role contract plus explicit, fully verified overlay driver."""
    from recovery_common import bind
    bind(repair_ref,dict(path=config['_config_ref_path'],sha256=config['_config_sha256']),config['run_id'],
        cpu_supervisor=os.environ.get('E1_RUN_ID')==repair['supervisor_run_id'])
    path=Path(config['gates']['scoped_cpu_processes_path']).resolve()
    require(path==Path(config['paths']['receipts']).resolve()/'scoped-cpu-processes.json','CPU role registry path changed')
    receipt=gates.bound(dict(path=str(path),sha256=gates.sha(path)))
    require(receipt.get('schema')=='opsd-ablation-scoped-cpu-processes-v1' and receipt.get('accepted') is True
        and receipt.get('run_id')==config['run_id'] and receipt.get('config_sha256')==config['_config_sha256'],
        'CPU registry changed original config/run')
    roles=receipt['roles'];names=[role['role'] for role in roles]
    require(len(names)==len(set(names)) and (set(names)=={'study_driver','memory_observer'}
        or observer_optional and set(names)=={'study_driver'}),
            'Recovery needs exactly one explicit supervisor and optional original observer')
    devices,apps=gates.inventory();gpu_pids={a['pid'] for a in apps};identities=[];proofs=[]
    for role in roles:
        identity=role['identity'];actual=gates.process(identity['pid'])
        require(all(actual.get(k)==identity.get(k) for k in ('pid','uid','starttick','ppid','pgid','argv','run_id'))
            and actual['uid']==config['owner']['uid'] and actual.get('run_id') and actual['run_id']!=config['run_id']
            and role.get('no_gpu_or_training') is True and actual['pid'] not in gpu_pids,'CPU role actual identity/GPU differs')
        argv_digest=hashlib.sha256(json.dumps(actual['argv'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        require(role['argv_sha256']==argv_digest,'CPU role actual argv SHA differs')
        input_value=gates.bound(role['input_config'])
        if role['role']=='study_driver':
            require(role['input_config']['path']==study_ref['path'] and role['input_config']['sha256']==study_ref['sha256']
                and input_value['study_id']==config['study_id'],'Driver original study differs')
            helper=Path(input_value['helper_root']).resolve();name='study_runner.py'
            manifest_ref=input_value['helper_source_manifest'];publication_ref=input_value['publication'];commit=input_value['source_commit']
            overlay=role.get('execution_overlay')
            require(overlay is not None and overlay['repair_manifest']==repair_ref
                and overlay['source']==file_ref(Path(__file__)) and overlay['publication']==repair['publication']
                and overlay['helper_source_manifest']==repair['source_manifest']
                and overlay['actual_entry_argv']==actual['argv']
                and actual['run_id']==repair['supervisor_run_id'], 'Overlay driver source/publication/actual argv missing')
            expected=overlay_command(config['paths']['python'],Path(__file__).resolve(),'study',repair_ref,
                ['--study-manifest',study_ref['path'],'--study-manifest-sha256',study_ref['sha256'],'--resume'])
            require(actual['argv']==expected,'Driver did not actually execute the exact external recovery entry')
        else:
            require(input_value=={k:v for k,v in config.items() if not k.startswith('_')},'Observer original immutable config differs')
            helper=Path(config['paths']['helper_root']).resolve();name='memory_observer.py'
            manifest_ref=config['helper_source_manifest'];publication_ref=config['gates']['publication'];commit=config['source_commit']
            require(actual['argv'][:1]==[config['paths']['python']]
                and (actual['argv'][1:2]==[str(helper/name)] or actual['argv'][1:3]==['-B',str(helper/name)])
                and actual['argv'].count('--config')==1
                and actual['argv'][actual['argv'].index('--config')+1]==role['input_config']['path'],
                'Original observer actual source/input argv differs')
        require(role['helper_source_manifest']==manifest_ref and role['publication']==publication_ref,'CPU original source bindings differ')
        publication=gates.bound(publication_ref);manifest=gates.bound(manifest_ref);source=helper/name
        require(publication.get('accepted') is True and publication.get('branch')=='search-e1'
            and publication['source_commit']==publication['remote_readback_sha']==commit
            and publication['source_manifest_sha256']==manifest_ref['sha256'],'Original CPU role not published')
        entry=manifest['files'][name];digest=entry['sha256'] if isinstance(entry,dict) else entry
        require(Path(role['source']['path']).resolve()==source and role['source']['sha256']==digest
            and gates.sha(source)==digest and source.stat().st_size==role['source']['bytes'],'Original CPU source bytes differ')
        identities.append(actual);proofs.append(dict(role=role['role'],identity=actual,source=role['source'],input_config=role['input_config'],
            execution_overlay=role.get('execution_overlay')))
    require(len({identity['run_id'] for identity in identities})==len(identities),'CPU roles must have distinct markers')
    return dict(accepted=True,identities=identities,roles=proofs,receipt=dict(path=str(path),sha256=gates.sha(path)),
        observer_optional=observer_optional,direct_gpu_inventory=devices,checked_unix=time.time())


def execute(mode, repair_ref, original_arguments, original_entry=None):
    from recovery_common import bind, checked_json as bound_json, install_original, receipt
    parsed=parse_original(mode,original_arguments);repair=bound_json(repair_ref)
    study_ref=repair['study_manifest'];study=bound_json(study_ref)
    require([g['config'] for g in study['groups']]==repair['original_configs'],'Original four study configs changed')
    configs={};native_root=None;native_manifest=None
    for reference in repair['original_configs']:
        config=bound_json(reference)
        if mode=='study':
            _,selected,root,manifest,_=bind(repair_ref,reference,config['run_id'],cpu_supervisor=True)
        else:
            # Verify all four inputs/source trees once; the selected controller
            # identity is checked without changing any process environment.
            if reference['path']!=parsed.config:
                configs[reference['path']]=(config,dict(path=reference['path'],sha256=reference['sha256']));continue
            _,selected,root,manifest,_=bind(repair_ref,reference,config['run_id'])
        configs[reference['path']]=(config,dict(path=reference['path'],sha256=reference['sha256']))
        native_root=root;native_manifest=manifest
    require(native_root is not None,'Selected original config outside pinned four inputs')
    if mode=='study':
        require(dict(path=parsed.study_manifest,sha256=parsed.study_manifest_sha256)==
            dict(path=study_ref['path'],sha256=study_ref['sha256']),'Original study CLI changed')
        selected=next(value[0] for value in configs.values() if value[0]['variant']=='G_POS')
    else:
        selected,reference=configs[parsed.config]
        require(parsed.config_sha256==reference['sha256'],'Original config CLI SHA changed')
        if mode=='controller':require(parsed.run_id==selected['run_id'],'Original group run CLI changed')
    if mode=='trainer':
        import trainer_launcher
        actual=actual_self_identity()
        require(actual['run_id']==selected['run_id'] and actual['uid']==1001,'Actual trainer entry run/UID differs')
        receipt(selected,repair_ref,'trainer-entry-rank-'+os.environ.get('RANK','unknown'),
            dict(actual_entry_identity=actual,actual_entry_source=file_ref(Path(__file__)),
                original_trainer=file_ref(native_root/'trainer.py'),config=dict(path=parsed.config,sha256=parsed.config_sha256),
                outer_manifest=dict(path=parsed.outer_manifest,sha256=parsed.outer_manifest_sha256),
                repair_source_manifest=repair['source_manifest'],publication=repair['publication'],
                original_arguments=original_arguments,actual_returncode_not_yet_known=True))
        saved=sys.argv;sys.argv=[str(native_root/'trainer.py')]+original_arguments
        try:
            return trainer_launcher.run(repair_ref,dict(path=parsed.config,sha256=parsed.config_sha256),
                dict(path=parsed.outer_manifest,sha256=parsed.outer_manifest_sha256),selected['run_id'],
                int(os.environ.get('RANK','-1')),int(os.environ.get('WORLD_SIZE','0')))
        finally:sys.argv=saved
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,
            'External study/controller must remain CPU-only')
    install_original(native_root,native_manifest)
    gates=importlib.import_module('gates')
    actual_entry_identity=gates.process(os.getpid());original_actual_argv=actual_entry_identity['argv']
    if mode=='study':
        recovery_gate=evaluation_recovery_gate(repair,configs,gates)
        native_evaluation=importlib.import_module('evaluation')
        native_evaluation.verify_common_zero(selected,repair['common_zero'])
    else:recovery_gate=None
    execution=dict(mode=mode,actual_entry_argv=original_actual_argv,actual_entry_identity=actual_entry_identity,
        actual_entry_source=file_ref(Path(__file__)),
        original_code=file_ref(native_root/('study_runner.py' if mode=='study' else 'evaluation.py' if mode=='evaluation' else 'controller.py')),
        original_arguments=original_arguments,original_config_references=repair['original_configs'],
        repair_source_manifest=repair['source_manifest'],publication=repair['publication'],
        original_study=study_ref,recovery_gate=recovery_gate,torch_initialized=False)
    receipt(selected,repair_ref,mode+'-entry',execution)
    original_scoped=gates.scoped_cpu_processes
    def scoped(config,observer_optional=False):
        # Config SHA is the unchanged original. Keep the config filepath only
        # in an internal key, never mutate its scientific/public dictionary.
        matching=[ref for cfg,ref in configs.values() if cfg['run_id']==config['run_id']]
        require(len(matching)==1,'CPU gate outside bound four original configs')
        config=dict(config,_config_ref_path=matching[0]['path'])
        return scoped_roles(config,gates,repair_ref,repair,study_ref,original_actual_argv,observer_optional)
    gates.scoped_cpu_processes=scoped
    def before_spawn():
        reference=next(ref for cfg,ref in configs.values() if cfg['variant']=='G_POS') if mode=='study' else dict(path=parsed.config,sha256=parsed.config_sha256)
        config=configs[reference['path']][0]
        bind(repair_ref,reference,config['run_id'],cpu_supervisor=mode=='study')
    def dispatched(fields):receipt(selected,repair_ref,'dispatch-'+uuid.uuid4().hex,
        dict(repair_source_manifest=repair['source_manifest'],publication=repair['publication'],**fields))
    saved_argv=sys.argv
    try:
        if mode == 'evaluation':
            from evaluation_retry import evaluation_dispatch, validate_actual_evaluation_entry, retry_hold
            evaluation_dispatch(parsed, selected['paths']['python'], configs, native_root)
            validate_actual_evaluation_entry(original_entry, native_root, original_actual_argv,
                                             native_manifest, file_ref)
            path = native_root / 'evaluation.py'; sys.argv=[str(path)]+original_arguments
            loaded=runpy.run_path(str(path),run_name='_signal_recovery_original_evaluation')
            namespace=loaded['run'].__globals__
            namespace['boundary']=retry_hold(loaded['boundary'], gates.Hold)
            return loaded['main']()
        require(original_entry is None, 'Only evaluation role may carry original-entry')
        path=native_root/('study_runner.py' if mode=='study' else 'controller.py')
        sys.argv=[str(path)]+original_arguments
        with exact_dispatch(configs,native_root,repair_ref,before_spawn,dispatched):
            loaded=runpy.run_path(str(path),run_name='_signal_recovery_original_'+mode)
            if mode=='study':
                namespace=loaded['run'].__globals__;original_role=namespace['cpu_role']
                def role(name,*args,**kwargs):
                    value=original_role(name,*args,**kwargs)
                    if name=='study_driver':value['execution_overlay']=dict(repair_manifest=repair_ref,
                        source=file_ref(Path(__file__)),helper_source_manifest=repair['source_manifest'],
                        publication=repair['publication'],actual_entry_argv=original_actual_argv)
                    return value
                namespace['cpu_role']=role
                return loaded['run'](study_ref,resume=True)
            return loaded['main']()
    finally:
        sys.argv=saved_argv;gates.scoped_cpu_processes=original_scoped


def main():
    # Execution integration is deliberately completed against recovery_common's
    # separately audited binding API, not implicit environment/source defaults.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('study', 'controller', 'trainer', 'evaluation'), required=True)
    parser.add_argument('--recovery-ref', required=True); parser.add_argument('--recovery-sha256', required=True)
    parser.add_argument('--original-entry')
    parser.add_argument('original_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original = args.original_arguments
    require(original[:1] == ['--'], 'Original argv must follow a literal --')
    execute(args.mode, dict(path=args.recovery_ref, sha256=args.recovery_sha256), original[1:], args.original_entry)


if __name__ == '__main__': main()
