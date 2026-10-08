"""Mac-only external recovery of compact V1 empty-directory reclamation."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
PREFIX = 'experiments/search_e1/control/opsd_ablation_compact_cleanup_recovery_v1/'
ORIGINAL_MANIFEST = '63e1c9aa3d6fd50a8c80e23143e4375360da658bfa5170069855ede1e5f2f13c'
ORIGINAL_COMMIT = '4d2065e81e3350abc69a6a5861b33be64803a458'
ORIGINAL_CORE_SHA = 'a8b6f8e42202c595578cf8b9e71d65ab2681ce3e529967b9bf930acaeff6ca19'


def empty_tree(b, extraction, planned_files):
    """Validate the entire existing empty tree before removing any directory."""
    b.require(not extraction.is_symlink(), 'Extraction symlink refused')
    if not extraction.exists(): return
    b.checked_path(extraction, directory=True)
    allowed={extraction}
    for name,proof in planned_files.items():
        member=extraction/b.safe_name(name)
        b.require(Path(proof['path'])==member,'Planned member escapes extraction')
        for parent in member.parents:
            allowed.add(parent)
            if parent==extraction:break
    directories = []
    for parent, children, files in os.walk(extraction, topdown=True, followlinks=False):
        current = b.checked_path(Path(parent), directory=True)
        b.require(current in allowed,'Foreign empty directory refused; never remove it')
        b.require(not files, 'Foreign residual file refused; never remove it')
        for name in children:
            b.require(current/name in allowed,'Foreign empty directory refused; never remove it')
            b.checked_path(current / name, directory=True)
        directories.append(current)
    for current in reversed(directories):
        b.checked_path(current, directory=True)
        b.require(not any(current.iterdir()), 'New/foreign residual entry refused')
        current.rmdir(); b.fsync_dir(current.parent)


def patched_reclaim(core):
    """Compile original bytes, replacing only its final empty-directory block."""
    raw=(Path(core.__file__)).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_CORE_SHA: raise RuntimeError('Original core bytes differ')
    tree = ast.parse(raw, core.__file__)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reclaim_legacy']
    if len(functions) != 1: raise RuntimeError('Original reclamation function differs')
    function = functions[0]
    matches = [n for n in function.body if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Attribute) and isinstance(n.test.func.value, ast.Name)
        and n.test.func.value.id == 'extract' and n.test.func.attr == 'exists']
    if len(matches) != 1: raise RuntimeError('Original empty-directory block differs')
    position = function.body.index(matches[0])
    function.body[position] = ast.copy_location(ast.Expr(value=ast.Call(func=ast.Name(id='_empty_tree',ctx=ast.Load()),
        args=[ast.Name(id='b',ctx=ast.Load()),ast.Name(id='extract',ctx=ast.Load()),
            ast.Subscript(value=ast.Name(id='old',ctx=ast.Load()),slice=ast.Constant(value='files'),ctx=ast.Load())],keywords=[])), matches[0])
    module = ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace = dict(core.__dict__, _empty_tree=empty_tree)
    exec(compile(module, core.__file__ + ':external-empty-tree-recovery', 'exec'), namespace)
    return namespace['reclaim_legacy']


def own_publication(b, args, run):
    manifest, mp = b.checked_json(HERE/'source-manifest.json', args.recovery_manifest_sha256)
    b.require({p.name for p in HERE.iterdir()} == set(manifest['files'])|{'source-manifest.json'}, 'Recovery source inventory differs')
    sources = [b.fileproof(HERE/n,dict(path=str(HERE/n),**r)) for n,r in manifest['files'].items()]
    pub, pp = b.checked_json(args.recovery_publication,args.recovery_publication_sha256)
    b.require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
        and pub.get('source_commit') == pub.get('remote_readback_sha') == args.recovery_source_commit
        and re.fullmatch('[0-9a-f]{40}',args.recovery_source_commit)
        and pub.get('source_manifest_sha256') == mp['sha256']
        and (pub.get('run_id') == run or run in pub.get('run_ids',{}).values()), 'Recovery publication differs')
    for fp in sources+[mp]:
        hits=[r for r in pub['files'] if r['path']==PREFIX+Path(fp['path']).name]
        b.require(len(hits)==1 and hits[0]['equal'] is True and hits[0]['sha256']==fp['sha256']
            and hits[0]['bytes']==fp['bytes'],'Recovery byte readback differs')
    return dict(manifest=mp,publication=pp,source_commit=args.recovery_source_commit,sources=sources)


def option(arguments,name):
    flag='--'+name.replace('_','-'); hits=[i for i,v in enumerate(arguments) if v==flag]
    if len(hits)!=1 or hits[0]+1==len(arguments): raise RuntimeError('Missing/duplicate original option: '+flag)
    return arguments[hits[0]+1]


def install(original, launch_args, forwarded):
    prior_loader=original.load_source
    def load(expected):
        if expected != ORIGINAL_MANIFEST: raise RuntimeError('Only the published V1 compact source is supported')
        core=prior_loader(expected)
        b, _=core.legacy_modules(option(forwarded,'legacy_helper_root'),option(forwarded,'source_manifest_sha256'))
        fields=('legacy_helper_root','source_manifest_sha256','publication','publication_sha256','source_commit',
            'compact_manifest_sha256','compact_publication','compact_publication_sha256','compact_source_commit')
        opts=types.SimpleNamespace(**{n:option(forwarded,n) for n in fields})
        b.require(opts.compact_source_commit==ORIGINAL_COMMIT,'Original compact commit differs')
        goal,_=b.checked_json(option(forwarded,'goal_identity'),option(forwarded,'goal_identity_sha256'))
        groups=b.validate_goal(goal); old, compact=core.publication(b,opts,groups[0]['run_id'])
        own=own_publication(b,launch_args,groups[0]['run_id'])
        terminal,tp=b.checked_json(launch_args.previous_queue_terminal,launch_args.previous_queue_terminal_sha256)
        b.require(terminal.get('schema')=='opsd-ablation-compact-queue-terminal-v1' and terminal.get('accepted') is True
            and terminal['pid']==45060 and terminal['uid']==os.getuid()==os.geteuid()==501
            and type(terminal['actual_exit_code']) is int and terminal['source_commit']==ORIGINAL_COMMIT
            and terminal['goal_sha256']==option(forwarded,'goal_identity_sha256')
            and terminal['no_live_transfer'] is True and terminal['owned_children_terminal'] is True,
            'Original queue terminal proof differs; absence is not an exit code')
        status,_=b.checked_json(terminal['queue_status']['path'],terminal['queue_status']['sha256'])
        identity=dict(legacy=dict(goal_sha256=option(forwarded,'goal_identity_sha256'),
            source_manifest_sha256=opts.source_manifest_sha256,publication_sha256=opts.publication_sha256,source_commit=opts.source_commit),
            compact_manifest_sha256=expected,compact_publication_sha256=opts.compact_publication_sha256,
            compact_source_commit=ORIGINAL_COMMIT,legacy_queue_root=option(forwarded,'legacy_queue_root'))
        b.require(terminal['queue_identity']==status['identity']==identity,'Terminal queue identity differs')
        previous_launch,_=b.checked_json(terminal['launch']['path'],terminal['launch']['sha256'])
        destination=b.checked_path(Path(option(forwarded,'destination')).absolute(),directory=True)
        b.require(previous_launch['source_commit']==ORIGINAL_COMMIT and
            Path(previous_launch['queue_path'])==destination/'queue-status.json' and previous_launch['status']['identity']==identity,
            'Destination differs from the actual original queue launch')
        actual_status,_=b.checked_json(destination/'queue-status.json',b.fileproof(destination/'queue-status.json')['sha256'])
        b.require(actual_status==status,'Current queue differs from preserved terminal status; do not create/recompress a new queue')
        live=subprocess.run(['ps','-p','45060','-o','pid='],capture_output=True,check=False)
        b.require(live.returncode==1 and not live.stdout.strip(),'Old PID alive/reused or absence uncertain')
        b.require(forwarded[0]=='queue' and '--reclaim-legacy' in forwarded,'Recovery resumes the original queue with reclamation')
        target=Path(launch_args.recovery_launch_receipt).absolute(); b.checked_path(target.parent,directory=True)
        b.require(not target.exists(),'Exclusive recovery launch receipt required')
        lp=b.write_json(target,dict(schema='opsd-ablation-compact-cleanup-recovery-launch-v1',accepted=True,
            scope='Source/PID preflight only; original queue lease remains mandatory',pid=os.getpid(),uid=os.getuid(),
            observed_unix=time.time(),argv=launch_args.actual_launch_argv,original_source=dict(legacy=old,compact=compact),
            recovery_source=own,previous_queue_terminal=tp,queue_identity=identity,no_signal_sent=True))
        fixed=patched_reclaim(core)
        def reclaim(b,reference,job,args):
            for fp in own['sources']+[own['manifest'],own['publication'],lp,tp]: b.stable(fp)
            result=fixed(b,reference,job,args)
            record=Path(reference['path']).parent/('cleanup-recovery-'+launch_args.recovery_source_commit+'.json')
            body=dict(schema='opsd-ablation-compact-cleanup-recovery-complete-v1',accepted=True,
                compact_receipt=reference,legacy_reclaimed=result,recovery_launch=lp,recovery_source=own,
                original_compact_commit=ORIGINAL_COMMIT,original_compact_manifest=ORIGINAL_MANIFEST,
                previous_queue_terminal=tp,scope='Empty-directory cleanup only; exact original tar/member identity unchanged')
            if record.exists():
                saved,_=b.checked_json(record,b.fileproof(record)['sha256'])
                prior,_=b.checked_json(saved['recovery_launch']['path'],saved['recovery_launch']['sha256'])
                b.require(prior.get('accepted') is True and prior['recovery_source']==own,'Prior recovery launch differs')
                b.require({k:v for k,v in saved.items() if k!='recovery_launch'}==
                    {k:v for k,v in body.items() if k!='recovery_launch'},'Existing recovery provenance differs')
            else: b.write_json(record,body)
            return result
        core.reclaim_legacy=reclaim
        return core
    original.load_source=load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original-helper-root','recovery-manifest-sha256','recovery-publication','recovery-publication-sha256',
        'recovery-source-commit','previous-queue-terminal','previous-queue-terminal-sha256','recovery-launch-receipt'):
        p.add_argument('--'+name,required=True)
    args,remaining=p.parse_known_args(); args.actual_launch_argv=list(sys.argv)
    if remaining and remaining[0]=='--': remaining=remaining[1:]
    if sys.platform!='darwin': raise RuntimeError('Recovery launcher is Mac-only')
    root=Path(args.original_helper_root).absolute(); raw=(root/'source-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_MANIFEST: raise RuntimeError('Original manifest differs')
    row=json.loads(raw)['files']['queue.py']; source=(root/'queue.py').read_bytes()
    if len(source)!=row['bytes'] or hashlib.sha256(source).hexdigest()!=row['sha256']: raise RuntimeError('Original queue bytes differ')
    original=types.ModuleType('original_compact_queue'); original.__file__=str(root/'queue.py')
    exec(compile(source,original.__file__,'exec'),original.__dict__)
    install(original,args,remaining)
    original_argv=sys.argv
    try:
        sys.argv=[original.__file__]+remaining
        original.main()
    finally: sys.argv=original_argv


if __name__=='__main__':
    try: main()
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(1)
