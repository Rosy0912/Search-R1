"""Independent read-only live -> actual same-pane exit0 capture for the final driver.

Writes only a fresh audit output; never imports producer code, models or CUDA,
changes tmux settings, takes locks, sends signals or launches a scientific job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
HANDOFF = BASE / 'receipts/stats-v2-handoff-8b641cb51b4c478684f596f75c603851/inputs'
OUTPUT = BASE / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs'
STUDY_SHA = '4965dee13af70a02ef6a486d4b9dfb6518a8a2b1183a4c26591fb0fa19379aae'
RUN = '697c6bdf9cad41458cfed8acc78617e9'
OWNER = dict(pid=1894801, uid=1005, starttick=927990515)
SERVER = dict(pid=2373025, uid=1005, starttick=884304691)
SESSION = 'e1-reverse-seven-stats-697c6bdf-20261009'
LABELS = {'C0','C1','M','R','W','P','J','bf16_zero','fp16_zero','P_fp16','J_fp16'}
PINS = {
    'driver_start': (OUTPUT/'driver-start.json', '0d85b42dd0c68d2a3e7c90e656182868951a1d1badb4613961177916d99aa038'),
    'launch': (HANDOFF/'handoff-launch.json', 'eb129b4eeae74cbb2682d9f1c025f2142b57f673423fcf3c9001db878bdf9636'),
    'before_exec': (HANDOFF/'new-window-before-exec.json', '13b5eebb77e7c430cb6dfb9742929dc70fbb35b93e30822e5723c0ad71960f12'),
    'session_created': (HANDOFF/'new-session-created.json', 'e410f8029892f3482994504bd860262dc84109fd1f407b3fcd9abc71f2667854'),
}


class Hold(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Hold(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def read(path, expected=None):
    path = Path(path)
    require(not path.is_symlink(), 'Symlink input is not a pinned original file')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'Input is not a regular file')
    with path.open('rb') as handle:
        raw = handle.read()
        end = os.fstat(handle.fileno())
    after = path.stat()
    keys = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    require(all(getattr(before,k)==getattr(end,k)==getattr(after,k) for k in keys), 'Input changed during read')
    sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or sha == expected, 'Pinned file SHA differs: '+str(path))
    return raw, dict(path=str(path.resolve()), sha256=sha, bytes=len(raw))


def document(path, sha=None):
    raw, ref = read(path, sha)
    return json.loads(raw), ref


def bound(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and len(ref.get('sha256',''))==64, 'Missing pinned file reference')
    return document(ref['path'], ref['sha256'])


def same(actual, expected):
    return all(actual.get(k)==expected.get(k) for k in ('pid','uid','starttick'))


def process(pid):
    root = Path('/proc')/str(pid)
    try:
        raw = (root/'stat').read_text()
        tail = raw[raw.rindex(')')+2:].split()
        lines = (root/'status').read_text().splitlines()
        uids = next(s for s in lines if s.startswith('Uid:')).split()
        argv = [v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v]
        env = dict(v.split('=',1) for v in (root/'environ').read_bytes().decode().split('\0') if '=' in v)
    except (FileNotFoundError, ProcessLookupError):
        # Missing one component of a still existing PID is uncertainty, not absence.
        require(Path('/proc/self/stat').is_file() and not root.exists(), 'Process read raced or became opaque; not ABSENT')
        return dict(pid=int(pid), status='ABSENT', absent_not_opaque=True)
    except PermissionError as error:
        raise Hold('Process identity is opaque, not absent: '+str(pid)) from error
    return dict(pid=int(pid), uid=int(uids[1]), euid=int(uids[2]), starttick=int(tail[19]),
                ppid=int(tail[1]), pgid=int(tail[2]), state=tail[0], argv=argv,
                run_id=env.get('E1_RUN_ID'), cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
                dont_write_bytecode=env.get('PYTHONDONTWRITEBYTECODE'), status='PRESENT')


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    require(result.returncode==0, 'Read-only command did not succeed: '+argv[0])
    return result.stdout.rstrip('\n')


def tmux_snapshot():
    lines = command(['tmux','list-panes','-t',SESSION+':','-F',
                     '#{session_id}\t#{pane_id}\t#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}\t#{pane_start_command}']).splitlines()
    require(len(lines)==1, 'Original session no longer has its one retained pane')
    parts = lines[0].split('\t')
    require(len(parts)==6, 'Opaque tmux pane record')
    sid, pane, pid, dead, status, start_command = parts
    require(sid=='$74' and pane=='%74' and int(pid)==OWNER['pid'], 'Original session/pane/PID reused or changed')
    server_pid = int(command(['tmux','display-message','-p','-t',pane,'#{pid}']))
    server = process(server_pid)
    require(same(server,SERVER), 'Original tmux server identity changed')
    socket = Path(command(['tmux','display-message','-p','-t',pane,'#{socket_path}']))
    st = socket.stat()
    require(stat.S_ISSOCK(st.st_mode) and st.st_dev==64768 and st.st_ino==63045636 and st.st_uid==1005,
            'Original tmux socket identity changed')
    return dict(session=SESSION,session_id=sid,pane_id=pane,pane_pid=int(pid),pane_dead=int(dead),
                pane_dead_status=status,pane_start_command=start_command,server=server,
                socket=dict(path=str(socket.resolve()),device=st.st_dev,inode=st.st_ino,uid=st.st_uid,mode=st.st_mode),
                actual_pane_record=lines[0])


def originals():
    study, sf = document(HANDOFF/'study-manifest.json',STUDY_SHA)
    require(study['run_id']==RUN and Path(study['output_root']).resolve()==OUTPUT.resolve()
            and study['source_commit']=='72e1e1276a3e5b21be108c346337f63b13368d66', 'Wrong final study identity/source')
    values = {}; refs = {}
    for key,(path,sha) in PINS.items():
        values[key],refs[key] = document(path,sha)
    start = values['driver_start']; launch = values['launch']; before = values['before_exec']; created = values['session_created']
    require(same(start['owner'],OWNER) and start['owner']['run_id']==RUN and start['input_manifest']['path']==sf['path']
            and start['input_manifest']['sha256']==sf['sha256'] and start['no_training_in_parent'] is True,
            'Original driver start/input identity changed')
    require(launch['accepted'] is True and launch['actual_tmux_returncode']==0 and launch['actual_exec_command_submitted'] is True
            and launch['run_id']==RUN and launch['session']==SESSION and launch['input_manifest']['sha256']==STUDY_SHA,
            'Original actual exec launch is not this final study')
    require(before['session_id']=='$74' and before['pane_id']=='%74' and before['pane_pid']==OWNER['pid']
            and same(before['placeholder'],OWNER) and created['actual_tmux_returncode']==0
            and created['stdout']=='$74 @74 %74 1894801', 'Original placeholder/session birth differs')
    inv, mp = bound(study['helper_source_manifest'])
    root = Path(study['helper_root']).resolve()
    require(Path(mp['path'])==root/'source-manifest.json' and mp['sha256']=='2e55c809d91bd3b4654d0ec0c4b0532e3ee8c413aac69a19cfd8f28bf042c842',
            'Original final evaluator source manifest differs')
    src = read(root/'study_runner.py',inv['files']['study_runner.py']['sha256'])[1]
    require(src['sha256']=='063c4c328dd5b0bb925544649cdd0ad66e674314ae702d00267dc42e9ee4f40c', 'Original final driver source changed')
    pub, pp = bound(study['publication'])
    require(pub['accepted'] is True and pub['branch']=='search-e1' and pub['source_commit']==study['source_commit']
            and pub['remote_readback_sha']==study['source_commit'] and pub['source_manifest_sha256']==mp['sha256'],
            'Original final evaluator publication is not exact remote readback')
    first, _ = bound(study['groups'][0]['training_config'])
    expected_argv = [first['paths']['python'],'-B',str(root/'study_runner.py'),'--study-manifest',sf['path'],
                     '--study-manifest-sha256',STUDY_SHA]
    require(start['owner']['argv']==expected_argv, 'Native original driver argv differs from its pinned input/source')
    return study, dict(study=sf,driver_start=refs['driver_start'],handoff=refs,source=src,source_manifest=mp,
                       publication=pp,expected_argv=expected_argv), start


def live(study, orig, start):
    pane = tmux_snapshot(); owner = process(OWNER['pid'])
    require(pane['pane_dead']==0 and pane['pane_dead_status']=='', 'Live binding requires the original driver still live')
    require(owner['status']=='PRESENT' and same(owner,OWNER) and owner['argv']==orig['expected_argv']
            and owner['run_id']==RUN and owner['ppid']==SERVER['pid'] and owner['pgid']==OWNER['pid']
            and owner['uid']==owner['euid']==1005 and owner['cuda_visible_devices']=='' and owner['dont_write_bytecode']=='1',
            'Literal bash pane has no exact same-PID Python exec identity')
    require(pane['pane_start_command']=='bash', 'Unexpected literal pane start command')
    foreground = os.readlink('/proc/'+str(OWNER['pid'])+'/fd/1')
    require(Path(foreground).is_absolute() and Path(foreground).is_file() and str(Path(foreground).resolve()).startswith(str(BASE)+'/'),
            'Foreground stdout is not the original HKUST study log')
    return dict(schema='opsd-ablation-final-driver-live-binding-v1',accepted=True,mode='live',run_id=RUN,
                observed_unix=time.time(),originals=orig,owner=owner,tmux=pane,foreground_log=str(Path(foreground).resolve()),
                exec_same_pid=True,no_signals=True,no_tmux_mutation=True,no_training_or_gpu_launch=True)


def all_controller_waits(driver_summary):
    """Read actual eight evaluator waits and the seven bound training waits."""
    evals=[]; trains={}
    for name in ['00-C0','common-zero','01-C1','02-M','03-R','04-W','05-P','06-J']:
        scope=OUTPUT/name/'evaluation'
        result, rr=document(scope/'result-accepted.json')
        config, cf=bound(result['config'])
        actual, ar=document(scope/'actual-child-exit.json'); child, ch=document(scope/'child-start.json')
        require(result['accepted'] is True and actual['actual_wait_returncode']==0
                and actual['signal_termination_not_exit0'] is False and actual['run_id']==config['run_id']
                and actual['identity']==child['identity'] and actual['command']==child['command'],
                'An original evaluator actual native wait0 is missing')
        done, dr=bound(result['controller_completion']); summary, sr=bound(result['summary'])
        require(done['accepted'] is True and done['run_id']==config['run_id']
                and done['config']['sha256']==cf['sha256'] and done['summary']['sha256']==sr['sha256']
                and summary['accepted'] is True and summary['primary_slots']==512*len(config['scientific']['versions']),
                'An original evaluator accepted completion differs')
        labels={'bf16_zero':'bf16_zero','fp16_zero':'fp16_zero'} if name=='common-zero' else {
            v:config['variant']+('_fp16' if v=='raw10_fp16' else '') for v in config['scientific']['versions']}
        for version,label in labels.items():
            require(summary['models'][version]==driver_summary['models'][label], 'Model changed between native evaluator and driver summary')
        evals.append(dict(scope=name,result=rr,actual_child_exit=ar,child_start=ch,controller_complete=dr,summary=sr))
        terminal,tr=bound(config['training']['terminal']); train,tf=bound(config['training']['config'])
        wait,wr=bound(terminal['actual_child_exit'])
        require(terminal['accepted'] is True and terminal['actual_exit_code']==0
                and terminal['signal_termination_not_exit0'] is False and terminal['run_id']==train['run_id']
                and terminal['config_sha256']==tf['sha256'] and wait['actual_wait_returncode']==0
                and wait['identity']==terminal['controller'] and wait['run_id']==train['run_id'],
                'An original bound training actual wait0 is missing')
        record=dict(terminal=tr,actual_child_exit=wr,config=tf)
        if train['variant'] in trains: require(trains[train['variant']]==record, 'Duplicate C0 terminal binding differs')
        trains[train['variant']]=record
    require(set(trains)=={'C0','C1','M','R','W','P','J'}, 'Seven original successful training terminal bindings incomplete')
    return dict(accepted=True,evaluator_controllers=evals,training_controllers=trains,
                actual_evaluator_wait0_count=8,actual_training_wait0_count=7,no_exit_code_inferred_from_absence=True)


def last_evaluator(study, driver_summary):
    scope = OUTPUT/'06-J/evaluation'
    accepted, ar = document(scope/'result-accepted.json')
    config, cf = bound(accepted['config']); complete, cp = bound(accepted['controller_completion']); summary, sp = bound(accepted['summary'])
    actual, ap = document(scope/'actual-child-exit.json'); child, ch = document(scope/'child-start.json')
    release, rp = document(scope/'actual-child-release.json')
    require(accepted['accepted'] is True and config['variant']=='J' and config['scientific']['versions']==['raw10_bf16','raw10_fp16']
            and complete['accepted'] is True and complete['run_id']==config['run_id'] and complete['config']['sha256']==cf['sha256']
            and complete['summary']['sha256']==sp['sha256'] and summary['accepted'] is True and summary['primary_slots']==1024,
            'Final J evaluator native1024 completion is not accepted')
    require(actual['actual_wait_returncode']==0 and actual['signal_termination_not_exit0'] is False
            and actual['run_id']==config['run_id'] and actual['identity']==child['identity'] and actual['command']==child['command']
            and actual['identity']['run_id']==config['run_id'] and actual['identity']['uid']==1005,
            'Final J evaluator actual original wait0 identity missing')
    require(release['accepted'] is True and release['owned']['accepted_no_owned_or_opaque'] is True
            and release['owned']['blockers']==[] and release['owned']['run_id']==config['run_id']
            and release['current']['accepted'] is True and release['current']['resource']['accepted'] is True
            and release['current']['resource']['phase']=='released', 'Final evaluator actual owned release not accepted')
    require(process(actual['identity']['pid'])['status']=='ABSENT', 'Final evaluator controller still present or PID reused')
    final_worker_checks=[]
    for version in ('raw10_bf16','raw10_fp16'):
        phase,pr=document(Path(config['paths']['output_root'])/version/'phase-complete.json')
        require(phase['accepted'] is True and len(phase['child_exit_evidence'])==8,
                'Final J phase actual eight-worker terminal missing')
        identities=[]
        for row in phase['child_exit_evidence']:
            require(row['actual_wait_returncode']==0 and row['identity']['run_id']==config['run_id']
                    and type(row['identity'].get('starttick')) is int, 'Final J worker actual wait0 identity missing')
            current=process(row['identity']['pid'])
            require(current['status']=='ABSENT', 'Final J worker still present, reused or opaque')
            identities.append(dict(identity=row['identity'],actual_wait_returncode=0,pid_observation=current))
        final_worker_checks.append(dict(version=version,actual_phase=pr,workers=identities))
    for version,label in [('raw10_bf16','J'),('raw10_fp16','J_fp16')]:
        require(summary['models'][version]==driver_summary['models'][label], 'Final J model summary differs from eleven-model summary')
    # Current compute observation is independent of, and never replaces, actual child wait/release evidence.
    devices = []
    for line in command(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits']).splitlines():
        idx,uuid,free = [x.strip() for x in line.split(',')]
        devices.append(dict(index=int(idx),uuid=uuid,actual_free_mib=int(free)))
    require([d['index'] for d in devices]==list(range(8)) and [d['uuid'] for d in devices]==config['owner']['gpu_uuids'],
            'Current released GPU UUID inventory differs')
    oracle, op = bound(config['gates']['retriever_admission']); apps=[]
    for line in command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits']).splitlines():
        if not line.strip(): continue
        pid,uuid,memory = [x.strip() for x in line.split(',')]
        app=dict(pid=int(pid),gpu_uuid=uuid,used_mib=int(memory));apps.append(app)
        identity=process(app['pid'])
        require(app['pid']==oracle['pid'] and same(identity,oracle) and app['gpu_uuid']==devices[7]['uuid'] and app['used_mib']<=1024,
                'Current compute process is not the exact approved GPU7 oracle; terminal resources are not currently closed')
    return dict(config=cf,result_accepted=ar,controller_complete=cp,summary=sp,actual_child_exit=ap,child_start=ch,
                actual_child_release=rp,actual_wait_returncode=0,owned_release_accepted=True,current_resource=dict(devices=devices,apps=apps),
                retriever_admission=op,final_worker_checks=final_worker_checks,scope='Final J actual native wait/release and current compute inventory; other eight controller waits are in native driver closure, not independently rerun here')


def terminal(study, orig, start, binding_ref):
    binding, bf = bound(binding_ref)
    require(binding['schema']=='opsd-ablation-final-driver-live-binding-v1' and binding['accepted'] is True
            and binding['run_id']==RUN and binding['exec_same_pid'] is True and binding['originals']==orig,
            'Terminal lacks exact prior live binding / original file continuity')
    pane = tmux_snapshot(); previous=binding['tmux']
    for key in ('session','session_id','pane_id','pane_pid','pane_start_command','socket'):
        require(pane[key]==previous[key], 'Original tmux retained pane/socket changed after live binding')
    require(same(pane['server'],previous['server']) and pane['pane_dead']==1 and pane['pane_dead_status']=='0',
            'Same original pane has not recorded a true successful foreground exit0')
    gone=process(OWNER['pid'])
    require(gone['status']=='ABSENT' and gone['absent_not_opaque'] is True, 'Original driver PID is present, reused or opaque; absence alone is not success')
    complete, cr = document(OUTPUT/'driver-complete.json'); summary, sr = bound(complete['summary'])
    require(complete['accepted'] is True and complete['model_count']==11 and complete['primary_slots']==5632
            and complete['actual_group_train_and_eval_exit0'] is True and complete['no_signals'] is True
            and complete['input_manifest']['path']==orig['study']['path'] and complete['input_manifest']['sha256']==STUDY_SHA,
            'Native final driver eleven-model5632 completion missing')
    require(summary['schema']=='opsd-ablation-eleven-fixed-model-summary-v1' and summary['accepted'] is True
            and summary['primary_slots']==5632 and set(summary['models'])==LABELS, 'Final native summary is not eleven accepted fixed512 models')
    for label,value in summary['models'].items():
        require(value['overall']['questions']==512 and value['datasets']['nq']['questions']==256
                and value['datasets']['hotpotqa']['questions']==256, 'Final accepted model/question denominator differs')
    waits=all_controller_waits(summary)
    final_release=last_evaluator(study,summary)
    return dict(schema='opsd-ablation-final-driver-actual-terminal-v1',accepted=True,mode='terminal',run_id=RUN,
                actual_exit_code=0,signal_termination_not_exit0=False,owner=start['owner'],input_manifest=orig['study'],
                prior_live_binding=bf,originals=orig,_children_waits=waits,
                observed_unix=time.time(),same_retained_tmux=pane,original_driver_pid_observation=gone,
                driver_complete=cr,eleven_native_summary=sr,model_count=11,primary_slots=5632,final_evaluator=final_release,
                no_exit_code_inferred_from_absence=True,no_signals=True,no_tmux_mutation=True,no_training_or_gpu_launch=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('live','terminal'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--binding');parser.add_argument('--binding-sha256')
    args=parser.parse_args(); out=Path(args.output)
    require(out.is_absolute() and str(out.resolve()).startswith(str(BASE)+'/analysis/') and not out.exists(),
            'Require a fresh independent HKUST analysis output directory')
    out.mkdir(parents=True,mode=0o700)
    try:
        study,orig,start=originals()
        if args.mode=='live':
            require(args.binding is None and args.binding_sha256 is None, 'Live binding does not reuse another binding')
            result=live(study,orig,start); name='live-binding.json'
        else:
            require(args.binding is not None and args.binding_sha256 is not None, 'Terminal requires explicit pinned prior live binding')
            result=terminal(study,orig,start,dict(path=args.binding,sha256=args.binding_sha256));name='terminal-accepted.json'
        result['helper_source']=read(__file__)[1]
        if args.mode=='terminal':
            child_waits=result.pop('_children_waits')
            wait_value=dict(schema='opsd-ablation-final-driver-actual-wait-v1',accepted=True,
                identity=start['owner'],actual_wait_returncode=0,exit_status_source='retained_same_pid_foreground_pane',
                run_id=RUN,prior_live_binding=result['prior_live_binding'],same_retained_tmux=result['same_retained_tmux'],
                original_driver_pid_observation=result['original_driver_pid_observation'],no_exit_code_inferred_from_absence=True,no_signals=True)
            owned_value=dict(schema='opsd-ablation-final-driver-owned-release-v1',accepted=True,run_id=RUN,
                accepted_no_owned_or_opaque=True,
                final_native_release=result['final_evaluator']['actual_child_release'],original_controller_actual_waits=child_waits,
                final_evaluator=result['final_evaluator'],driver_has_no_GPU_context=True,
                no_exit_code_inferred_from_empty_compute_inventory=True,no_signals=True)
            for filename,value,key in [('actual-wait.json',wait_value,'actual_wait_evidence'),
                                       ('full-owned-release.json',owned_value,'full_owned_release')]:
                target=out/filename
                with target.open('xb') as handle:
                    os.chmod(target,0o600);handle.write(canonical(value)+b'\n');handle.flush();os.fsync(handle.fileno())
                result[key]=read(target)[1]
        destination=out/name
        with destination.open('xb') as handle:
            os.chmod(destination,0o600);handle.write(canonical(result)+b'\n');handle.flush();os.fsync(handle.fileno())
        proof=read(destination)[1]
        print(json.dumps(dict(accepted=True,status='LIVE_BOUND' if args.mode=='live' else 'ACTUAL_PARENT_EXIT0_11_MODELS_5632_RELEASED',
                              mode=args.mode,run_id=RUN,proof=proof,no_signals=True)),flush=True)
        return 0
    except Exception as error:
        failure=dict(accepted=False,status='HOLD_UNKNOWN_OR_NOT_READY',mode=args.mode,run_id=RUN,observed_unix=time.time(),
                     error_type=type(error).__name__,reason=str(error),no_signals=True,no_restart_or_tmux_mutation=True)
        destination=out/'hold.json'
        with destination.open('xb') as handle:handle.write(canonical(failure)+b'\n')
        print(json.dumps(dict(accepted=False,status=failure['status'],proof=read(destination)[1],reason=str(error))),flush=True)
        return 3


if __name__=='__main__':
    raise SystemExit(main())
