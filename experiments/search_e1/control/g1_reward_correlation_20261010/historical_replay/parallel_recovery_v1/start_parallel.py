"""Root-invoked launch of explicitly selected independent shared-GPU workers.

No original process is signalled or modified. Worker itself performs the fresh
ownership/resource/source gates before importing Torch. Started is not accepted.
"""
import argparse,datetime,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpus',required=True);a=parser.parse_args()
    requested=[int(x) for x in a.gpus.split(',')];assert requested==sorted(set(requested)) and set(requested)<=set([0,1,2,3,5,6])
    sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
    co=json.loads((HERE/'cohort-config.json').read_bytes());d=json.loads((HERE/'deployment.stdout.json').read_bytes())
    assert d['CPU_fixture_rc']==0 and d['GPU_started'] is False and d['source']==co['remote_source']
    numeric=json.loads((HERE/'numeric-env-proof.json').read_bytes());assert numeric['rc']==0
    numeric=json.loads(numeric['stdout'])['numeric_environment_whitelist'];assert numeric['OMP_NUM_THREADS']=='8'
    unset=['MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','PYTHONHASHSEED']
    assert all(numeric[k] is None for k in unset)
    manifest_sha=hashlib.sha256((HERE/'source-manifest.json').read_bytes()).hexdigest();jobs=[]
    for entry in co['entries']:
        if entry['gpu'] not in requested:continue
        raw=pathlib.Path(entry['local_config']).read_bytes();c=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
        assert c['gpu_index']==entry['gpu'] and c['selected_steps']==entry['steps']
        args=['env']
        for key in unset+['USE_TORCH','USE_TF','USE_FLAX']:args+=['-u',key]
        args+=['CUDA_VISIBLE_DEVICES='+str(c['gpu_index']),'OMP_NUM_THREADS=8','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','setsid','--wait',c['base']+'/env/bin/python','-B',co['remote_source']+'/replay_gpu.py','--config',co['remote_source']+'/configs/gpu'+str(c['gpu_index'])+'.json','--config-sha256',sha,'--source-manifest',co['remote_source']+'/source-manifest.json','--source-manifest-sha256',manifest_sha]
        log=co['remote_parent']+'/logs/gpu'+str(c['gpu_index'])
        fg='tmux set-window-option -t "$TMUX_PANE" remain-on-exit on && exec '+shlex.join(args)+' > '+shlex.quote(log+'.stdout.jsonl')+' 2> '+shlex.quote(log+'.stderr.log')
        jobs.append(dict(gpu=c['gpu_index'],steps=c['selected_steps'],session=entry['session'],output=c['output'],foreground=fg,config_sha256=sha))
    assert len(jobs)==len(requested)
    remote=r'''
import datetime,hashlib,json,os,pathlib,socket,subprocess,sys
j=json.load(sys.stdin);co=j['cohort'];assert socket.gethostname()==co['expected_node']
# Preflight every launch name/output before starting any child; no destructive recovery.
for x in j['jobs']:
    assert not pathlib.Path(x['output']).exists()
    assert subprocess.run(['tmux','has-session','-t',x['session']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
result=[]
for x in j['jobs']:
    subprocess.run(['tmux','new-session','-d','-s',x['session'],x['foreground']],check=True)
    pane=subprocess.run(['tmux','list-panes','-t',x['session'],'-F','#{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,check=True)
    result.append(dict(gpu=x['gpu'],selected_steps=x['steps'],session=x['session'],pane=pane.stdout.strip(),output=x['output'],foreground_sha256=hashlib.sha256(x['foreground'].encode()).hexdigest(),config_sha256=x['config_sha256'],started=True,accepted=False))
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),uid=os.getuid(),launches=result,cohort_complete=False,original_training_unmodified=True)))
'''
    r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),input=json.dumps(dict(cohort=co,jobs=jobs)).encode(),capture_output=True,timeout=60)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=HERE/('launch-'+stamp)
    p.with_suffix('.stdout.json').write_bytes(r.stdout);p.with_suffix('.stderr.txt').write_bytes(r.stderr)
    p.with_suffix('.execution.json').write_text(json.dumps(dict(rc=r.returncode,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_gpus=requested),indent=2)+'\n')
    assert r.returncode==0,'inspect actual started sessions; never repeat launch blindly'
    print(r.stdout.decode())

if __name__=='__main__':main()
