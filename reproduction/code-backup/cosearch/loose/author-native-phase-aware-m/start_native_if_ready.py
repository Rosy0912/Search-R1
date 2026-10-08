"""Advance once after N's full acceptance and resource release; never stop N."""
import fcntl, hashlib, json, subprocess, os, re
from pathlib import Path

BASE=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
ROOT=BASE/'author-native'
N=BASE/'runs/cosearch-local-gpu-five-20261003-n'

def queue(name):
    result=subprocess.run(['squeue','-h','--name='+name,'-o','%i %T %N'],capture_output=True,text=True,check=True)
    return result.stdout.strip()

def main():
    lock=(ROOT/'launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_id=os.environ.get('COSEARCH_NATIVE_NEXT_RUN_ID','author-native-raw-five')
    assert re.fullmatch(r'author-native-[a-z0-9-]+',run_id)
    claim=ROOT/('native-launch-claim.json' if run_id=='author-native-raw-five' else run_id+'-launch-claim.json')
    if claim.exists():
        print(json.dumps(dict(status='NATIVE_ALREADY_CLAIMED',claim=json.loads(claim.read_text()),
            queue=queue('cosearch-author-five'))));return
    if (N/'acceptance.json').is_file():
        accepted=json.loads((N/'acceptance.json').read_text())
        assert accepted['status']=='five_update_validation_save_reload_passed'
        handoff='N_STRICT_ACCEPTED'
    elif (N/'strict-rejection.json').is_file():
        rejected=json.loads((N/'strict-rejection.json').read_text())
        assert rejected['status']=='strict_rejected_actor_optimizer_not_saved'
        assert rejected['run']==N.name and rejected['job']==3002
        assert rejected['source_commit']=='e2e401c10ea0a20131eb9b0ed39f4c0464df7bc3'
        assert rejected['exact_outer_steps']==[1,2,3,4,5] and rejected['all_finite']
        assert rejected['validation_count']==64 and not rejected['accepted']
        assert rejected['missing_actor_optimizer'] is True
        assert Path(rejected['actor_checkpoint']).is_relative_to(N)
        assert rejected['no_resume']
        for name,expected in rejected['actor_files_sha256'].items():
            h=hashlib.sha256()
            with (Path(rejected['actor_checkpoint'])/name).open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==expected,name
        handoff='N_STRICT_REJECTED_AND_ARCHIVED_NOT_ACCEPTED'
    else:
        print(json.dumps(dict(status='WAITING_N_STRICT_ACCEPTANCE',queue=queue('cosearch-local-five'))));return
    assert json.loads((N/'driver-exit.json').read_text())['returncode']==0
    if queue('cosearch-local-five') or queue('cosearch-five-reload'):
        print(json.dumps(dict(status='WAITING_N_RESOURCE_RELEASE')));return
    assert not queue('cosearch-author-five'),'Existing native allocation requires inspection'
    assert not (ROOT/'runs'/run_id).exists()
    assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
    assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
    ready=json.loads((ROOT/'tools/preparation-ready.json').read_text())
    assert ready['cpu_preparation_passed'] and not ready['actual_gpu_resources_tested']
    for name,expected in ready['tools_sha256'].items():
        assert hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()==expected,name
    command=['srun','--immediate=60','-p','gpu','-w','h100-16','-N1','--ntasks=1',
        '--gres=gpu:h100:8','--exclusive','--cpus-per-task=128','--mem=768G',
        '--job-name=cosearch-author-five','env','COSEARCH_NATIVE_RUN_ID='+run_id,'bash',str(ROOT/'tools/start_native_entry.sh')]
    if os.environ.get('COSEARCH_NATIVE_GPU_PROBE_ONLY')=='1':
        command.insert(command.index('COSEARCH_NATIVE_RUN_ID='+run_id)+1,'COSEARCH_NATIVE_GPU_PROBE_ONLY=1')
    # Claim and record the command before detached execution, never re-submit blindly.
    with claim.open('x') as stream:json.dump(dict(command=command,status='claimed_not_accepted',previous_run_handoff=handoff),stream)
    import shlex
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    shell=shlex.join(command)+' > '+shlex.quote(str(logs/(run_id+'.log')))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',run_id,shell],check=True)
    print(json.dumps(dict(status='NATIVE_SUBMITTED',command=command,accepted=False,previous_run_handoff=handoff)))

if __name__=='__main__':main()
