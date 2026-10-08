"""Original coevolve.refit unchanged; only dump resource-command routing changes."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,sys,types
from original_blocks import DUMP_SHA
from stage_cleanup import owned_stage
from first_fault import before_cleanup,after_cleanup
from stage_baseline import protected_stage_baseline,post_cleanup_remaining,continuous_baseline_missing
COEVOLVE_SHA=''
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load_original_coevolve():
    path=pathlib.Path(__file__).parent/'original/ts_coevolve.py'
    expected=json.loads(pathlib.Path(__file__).with_name('original-source-bindings.json').read_text())['ts_coevolve.py']
    if sha(path)!=expected:raise ValueError('original coevolve bytes mismatch')
    spec=importlib.util.spec_from_file_location('original_coevolve',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def owned_command(command,env,context,pass_fds=()):
    """Synchronous direct child and all new own-cgroup descendants, no CUDA parent."""
    import time
    from owner import support,sha,atomic,owned_gpu_remaining,gpu_baseline,baseline_gpu_missing
    tools=support(pathlib.Path(__file__).with_name('owner_support.py'),context['support_sha256'])
    config=context['config'];scope,relative=tools.cgroup_scope(str(config['job']))
    if relative!=config['cgroup'] or os.environ.get('SLURM_JOB_ID')!=str(config['job']):raise ValueError('own refit allocation mismatch')
    owned=owned_stage(tools,scope,relative,expected_tools_sha256=context['support_sha256'])
    global_gpu_baseline=gpu_baseline(tools,tools.gpu_sample())
    baseline=protected_stage_baseline(tools,scope,relative,global_gpu_baseline)
    evidence=pathlib.Path(context['evidence_root']);evidence.mkdir(parents=True,exist_ok=True)
    label='dump-owner' if '--dump-only' in command else 'heads'
    receipt_path=evidence/(label+'-'+str(time.time_ns())+'.json')
    child=None;identities={};error=None
    try:
        child=owned.spawn(command,receipt_path.with_suffix('.log'),env=env,pass_fds=pass_fds)
        began=time.monotonic()
        while child.poll() is None:
            for pid in owned.stage_scope.pids(scope)-owned.baseline:identities[pid]=tools.process_identity(pid)
            if time.monotonic()-began>int(config['stage_timeout_seconds']):raise TimeoutError('original refit child timeout')
            time.sleep(1)
    except BaseException as exc:error=exc
    finally:
        cleanup_census_error=None
        try:
            for pid in owned.stage_scope.pids(scope)-owned.baseline:identities[pid]=tools.process_identity(pid)
        except BaseException as exc:
            cleanup_census_error={'error_type':type(exc).__name__,'error':str(exc)}
            if error is None:error=exc
        first_fault=before_cleanup(evidence,'diagnostic-'+receipt_path.stem,error,owned,hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest())
        if first_fault['publication_errors'] and error is None:error=RuntimeError('Refit evidence publication failed before cleanup')
        try:owned.cleanup()
        except BaseException as exc:
            if error is None:error=exc
        registration_after=after_cleanup(evidence,'diagnostic-'+receipt_path.stem,owned)
        if registration_after['publication_errors'] and error is None:error=RuntimeError('Refit evidence publication failed after cleanup')
        remaining,post_cleanup=post_cleanup_remaining(owned,tools,scope)
        if post_cleanup.get('error') and error is None:error=RuntimeError(post_cleanup['error'])
        missing=continuous_baseline_missing(tools,baseline)
        gpu=tools.gpu_sample();gpu_remaining=owned_gpu_remaining(gpu,set(identities));global_missing=baseline_gpu_missing(tools,gpu,global_gpu_baseline)
        receipt={'scope':'ORIGINAL_REFIT_COMMAND_ACTUAL_EXIT_UNREVIEWED','command':command,'returncode':child.returncode if child else None,'identities':list(identities.values()),'own_processes_remaining':remaining,'post_cleanup_read_only_census':post_cleanup,'own_gpu_records_remaining':gpu_remaining,'baseline_identity_missing':missing,'baseline_preserved':not missing,'original_global_gpu_baseline':global_gpu_baseline,'original_global_gpu_baseline_missing':global_missing,'protected_stage_baseline':list(baseline.values()),'cleanup':getattr(owned,'cleanup_record',None),'error':str(error) if error else None,'cleanup_census_error':cleanup_census_error,'first_fault_evidence':first_fault,'registration_after_cleanup':registration_after}
        atomic(receipt_path,receipt)
        if remaining or missing or gpu_remaining or global_missing:
            if error is None:error=RuntimeError('refit command owned release failed')
    if error:raise error
    return types.SimpleNamespace(returncode=child.returncode,receipt_path=str(receipt_path),receipt_sha256=sha(receipt_path))

def refit_with_owner(exp,step,checkpoint,args,environment,owner_context,runner=subprocess.run):
    original=load_original_coevolve();original.PY=sys.executable
    calls=[]
    def route(command,env):
        if command[1]=='ts_probe_dump.py':
            parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--data');parser.add_argument('--tag');parser.add_argument('--cap',type=int);parser.add_argument('--offset',type=int);parser.add_argument('--G',type=int);parser.add_argument('--layers');parser.add_argument('--gpu_mem',type=float);parser.add_argument('--out');parser.add_argument('--max_turns',type=int,default=4)
            parsed=parser.parse_args(command[2:]);requested=pathlib.Path(parsed.out);directory=pathlib.Path(owner_context['evidence_root'])/f'outer-{step}'
            config=dict(owner_context['config']);config['dump_args']=vars(parsed);config['outer_step']=step
            directory.parent.mkdir(parents=True,exist_ok=True);configfile=directory.parent/f'outer-{step}-owner-config.json'
            with open(configfile,'x') as f:os.chmod(configfile,0o600);json.dump(config,f,sort_keys=True)
            cmd=[sys.executable,str(pathlib.Path(__file__).with_name('owner.py')),'--config',str(configfile),'--directory',str(directory),'--support-sha256',owner_context['support_sha256'],'--dump-only']
            fds=tuple(int(x['fd']) for x in config['leases'])
            result=owned_command(cmd,env,owner_context,fds) if runner is subprocess.run else runner(cmd,env=env,pass_fds=fds)
            calls.append({'original_command':command,'owned_command':cmd,'returncode':result.returncode})
            if result.returncode:return result.returncode
            # Only complete, validated original arrays are published for original fit.
            from publication import complete_dump
            _,receipt,npz=complete_dump(directory)
            requested.mkdir(parents=True,exist_ok=True)
            for source in (npz,npz.with_suffix('.jsonl')):
                destination=requested/source.name
                if destination.exists():raise FileExistsError('do not replace prior refit dump')
                os.link(source,destination)
            return 0
        if command[1]=='ts_probe_train.py':
            command=list(command);command[1]=str(pathlib.Path(__file__).parent/'original/ts_probe_train.py')
            from publication import TRAIN_SHA
            if sha(command[1])!=TRAIN_SHA:raise ValueError('original head source changed')
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
        result=owned_command(command,cpu_env,owner_context) if runner is subprocess.run else runner(command,env=cpu_env)
        calls.append({'original_command':command,'returncode':result.returncode,'actual_exit_receipt_path':getattr(result,'receipt_path',None),'actual_exit_receipt_sha256':getattr(result,'receipt_sha256',None)})
        if result.returncode==0 and command[1].endswith('ts_probe_train.py'):
            heads=pathlib.Path(command[command.index('--save')+1])
            if any(not (heads/name).is_file() or (heads/name).is_symlink() for name in ('commit_probe.pkl','boundary_probe.pkl')):raise ValueError('missing original fitted heads: no refit publication')
        return result.returncode
    original.run=route
    manifest=pathlib.Path(args.manifest_root)/('live_'+exp)/'manifest.json'
    if manifest.is_symlink():raise ValueError('live manifest symlink rejected')
    old=manifest.read_bytes() if manifest.exists() else None
    try:complete=original.refit(exp,step,str(checkpoint),args,environment)
    except BaseException:
        if old is None:
            if manifest.exists():manifest.unlink()
        else:
            rollback=manifest.with_suffix('.owner-rollback.tmp')
            with open(rollback,'xb') as f:os.chmod(rollback,0o600);f.write(old);f.flush();os.fsync(f.fileno())
            os.replace(rollback,manifest)
        raise
    return complete,calls

def main():
    """One synchronous refresh, for the actor's existing outer-step barrier."""
    ap=argparse.ArgumentParser()
    ap.add_argument('--request',required=True)
    a=ap.parse_args();request=json.loads(pathlib.Path(a.request).read_text())
    # All paths and allocation identities are checked again by the GPU owner.
    args=types.SimpleNamespace(**request['args'])
    environment=dict(os.environ)
    for key in list(environment):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):environment.pop(key)
    environment.update(RETRIEVER_URL=request['retriever_url'],NO_PROXY='127.0.0.1,localhost')
    complete,calls=refit_with_owner(request['exp'],request['outer_step'],request['checkpoint'],args,environment,request['owner_context'])
    print(json.dumps({'scope':'ORIGINAL_G1_REFIT_ROUTING_UNREVIEWED_NO_TRAINING','complete':complete,'calls':calls}))
    if not complete:raise SystemExit(1)
if __name__=='__main__':main()
