"""Mac-only direct SSH transport recovery, retaining USTAR bound and empty-tree recovery."""
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
PREFIX = 'experiments/search_e1/control/opsd_ablation_compact_direct_bound_recovery_v1/'
PREVIOUS_CLEANUP_COMMIT = '23bd6fa8367c59dabe5444215fe2c486d56fa546'
PREVIOUS_CLEANUP_MANIFEST = '9b4b9ff3be6df57e80bef7660bfa60bfe066dd271c6f43f62b0e61a3b85a5a39'
PREVIOUS_LAUNCH = '/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008/goal-continuation-20261009-18/compact-bound-recovery-actual-launch.json'
PREVIOUS_LAUNCH_SHA = '6f60f5b6d4e8fa22553826c57fc77926c8c79de789b597a911737b564349cc45'
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


def raw_upper(inventory):
    """Original whole USTAR bytes: two EOF blocks, then 20-block record padding."""
    members = sum(512 + ((row['bytes'] + 511) // 512) * 512 for row in inventory['files'])
    return ((members + 1024 + 10239) // 10240) * 10240


def absent_pid(b, pid):
    b.require(type(pid) is int and pid > 0, 'Explicit prior owned PID required')
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid='], capture_output=True, check=False)
    b.require(result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip(),
              'Prior owned PID alive/reused or absence uncertain')


DIRECT_SSH = ['/usr/bin/ssh','-B','en0','-oProxyCommand=none','-oProxyJump=none',
              '-oControlMaster=no','-S','none','-oBatchMode=yes','-oConnectTimeout=10','hkust1']


def direct_command(command):
    # Keep the original single shlex-quoted remote argv/env string byte-for-byte.
    if not (isinstance(command,list) and len(command)==4 and command[:3]==['ssh','-oConnectTimeout=15','hkust1']
            and isinstance(command[3],str)):
        raise RuntimeError('Published remote command shape differs; never guess a route/remote command')
    return list(DIRECT_SSH)+command[3:]


def direct_legacy_loader(prior):
    def load(root,manifest_sha):
        b,q=prior(root,manifest_sha)
        remote=b.remote_command;discovery=q.discovery_command
        def remote_command(args,mode):return direct_command(remote(args,mode))
        def discovery_command(args):return direct_command(discovery(args))
        b.remote_command=remote_command;q.discovery_command=discovery_command
        return b,q
    return load


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
        core.legacy_modules=direct_legacy_loader(core.legacy_modules)
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
            and terminal['pid']==42299 and terminal['uid']==os.getuid()==os.geteuid()==501
            and type(terminal['actual_exit_code']) is int and terminal['source_commit']==PREVIOUS_CLEANUP_COMMIT
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
        b.require(terminal['launch']['path']==PREVIOUS_LAUNCH and terminal['launch']['sha256']==PREVIOUS_LAUNCH_SHA,
            'Previous launch must be the actual published cleanup recovery launch')
        b.require(previous_launch.get('schema')=='opsd-ablation-compact-bound-recovery-launch-v1'
            and previous_launch.get('accepted') is True and previous_launch['pid']==42299 and previous_launch['uid']==501
            and previous_launch['recovery_source']['source_commit']==PREVIOUS_CLEANUP_COMMIT
            and previous_launch['recovery_source']['manifest']['sha256']==PREVIOUS_CLEANUP_MANIFEST
            and previous_launch['queue_identity']==identity
            and Path(option(previous_launch['argv'],'destination'))==destination,
            'Destination/source/identity differs from the actual cleanup queue launch')
        prior_recovery=previous_launch['recovery_source']
        for fp in prior_recovery['sources']+[prior_recovery['manifest'],prior_recovery['publication']]:
            b.fileproof(fp['path'],fp)
        prior_manifest,_=b.checked_json(prior_recovery['manifest']['path'],PREVIOUS_CLEANUP_MANIFEST)
        prior_pub,_=b.checked_json(prior_recovery['publication']['path'],prior_recovery['publication']['sha256'])
        b.require(prior_pub.get('accepted') is True and prior_pub['source_commit']==prior_pub['remote_readback_sha']==PREVIOUS_CLEANUP_COMMIT
            and prior_pub['source_manifest_sha256']==PREVIOUS_CLEANUP_MANIFEST,
            'Actual previous cleanup publication differs')
        b.require({Path(fp['path']).name for fp in prior_recovery['sources']}==set(prior_manifest['files']),
            'Previous cleanup source inventory differs')
        actual_status,_=b.checked_json(destination/'queue-status.json',b.fileproof(destination/'queue-status.json')['sha256'])
        b.require(actual_status==status,'Current queue differs from preserved terminal status; do not create/recompress a new queue')
        known=terminal['known_owned_processes']
        b.require(isinstance(known,list) and known and all(type(p['pid']) is int and p['uid']==501 for p in known)
            and 42299 in [p['pid'] for p in known] and len({p['pid'] for p in known})==len(known),
            'Prior controller and all captured owned descendants must be explicitly bound')
        for process in known: absent_pid(b,process['pid'])
        b.require(forwarded[0]=='queue' and '--reclaim-legacy' in forwarded,'Recovery resumes the original queue with reclamation')
        target=Path(launch_args.recovery_launch_receipt).absolute(); b.checked_path(target.parent,directory=True)
        b.require(not target.exists(),'Exclusive recovery launch receipt required')
        lp=b.write_json(target,dict(schema='opsd-ablation-compact-direct-bound-recovery-launch-v1',accepted=True,
            scope='Source/PID preflight only; original queue lease remains mandatory',pid=os.getpid(),uid=os.getuid(),
            observed_unix=time.time(),argv=launch_args.actual_launch_argv,original_source=dict(legacy=old,compact=compact),
            recovery_source=own,previous_queue_terminal=tp,queue_identity=identity,no_signal_sent=True))
        fixed=patched_reclaim(core)
        def reclaim(b,reference,job,args):
            for fp in own['sources']+[own['manifest'],own['publication'],lp,tp]: b.stable(fp)
            result=fixed(b,reference,job,args)
            record=Path(reference['path']).parent/('direct-bound-recovery-'+launch_args.recovery_source_commit+'.json')
            body=dict(schema='opsd-ablation-compact-direct-bound-recovery-complete-v1',accepted=True,
                compact_receipt=reference,legacy_reclaimed=result,recovery_launch=lp,recovery_source=own,
                original_compact_commit=ORIGINAL_COMMIT,original_compact_manifest=ORIGINAL_MANIFEST,
                previous_queue_terminal=tp,scope='Direct scoped SSH, exact USTAR EOF/record bound and preserved empty-tree cleanup; whole tar/member identity unchanged')
            if record.exists():
                saved,_=b.checked_json(record,b.fileproof(record)['sha256'])
                prior,_=b.checked_json(saved['recovery_launch']['path'],saved['recovery_launch']['sha256'])
                b.require(prior.get('accepted') is True and prior['recovery_source']==own,'Prior recovery launch differs')
                b.require({k:v for k,v in saved.items() if k!='recovery_launch'}==
                    {k:v for k,v in body.items() if k!='recovery_launch'},'Existing recovery provenance differs')
            else: b.write_json(record,body)
            return result
        core.reclaim_legacy=reclaim
        core.raw_upper=raw_upper
        prior_save_complete=core.save_complete
        def save_complete(b,destination,inventory,ip,job,raw,verification,publications,legacy=None):
            for fp in own['sources']+[own['manifest'],own['publication'],lp,tp]: b.stable(fp)
            b.require('direct_bound_recovery' not in publications,'Do not overwrite a prior bound provenance')
            enriched=dict(publications,direct_bound_recovery=dict(source=own,launch=lp,previous_queue_terminal=tp,
                scope='Direct scoped SSH transport and exact tar EOF/record bound; original remote command and whole USTAR bytes/SHA unchanged'))
            return prior_save_complete(b,destination,inventory,ip,job,raw,verification,enriched,legacy)
        core.save_complete=save_complete
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
