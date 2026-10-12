"""Explicit recovery from the complete8 boundary of the stopped repaired B run."""
import hashlib,json,os,socket
from pathlib import Path

PROTOCOL='recursive_b_complete8_native_rng_resume_v1'
ROOT=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
PARENT_RUN='a177507c37c94e898616a194ed6b2a45'
PARENT_COMMIT='bc666df3a3fee9da7bb5d09f8da649cfa7165131'
PARENT_CONFIG_SHA='ab2a57d51ea3d0d864b5a02285599a267a3a471e3ecc7f95b830019b28e22e3b'

def require(ok,message):
    if not ok:raise ValueError('Complete8 B recovery: '+message)

def validate_binding(b):
    require(socket.gethostname().split('.')[0]=='xie-lab-4' and os.getuid()==1001,'host/UID')
    require(b['protocol']==PROTOCOL and b['parent_completed_step']==8,'explicit completed8 only')
    require((b['parent_run_id'],b['parent_commit'],b['parent_config_sha256'])==
        (PARENT_RUN,PARENT_COMMIT,PARENT_CONFIG_SHA),'current parent chain')
    require(Path(b['parent_config'])==ROOT/'receipts'/('speed-resume3-'+PARENT_RUN)/'resolved.yaml','parent config path')
    require(Path(b['parent_manifest'])==ROOT/('output-speed-resume3-'+PARENT_RUN)/
        'checkpoints/training_state/global_step_8/manifest.json','parent manifest path')
    require(b['parent_optimizer_calls']==160 and b['sampling_stream_status']=='COMPLETE_NATIVE_SAMPLING_STREAMS',
        'actual counters and full sampling streams required')
    require(len(b['sampling_streams'])==4 and {r['rank'] for r in b['sampling_streams']}==set(range(4)),
        'sampling rank coverage')
    terminal_path=Path(b['current_parent_terminal'])
    require(terminal_path==ROOT/'receipts/token-stop9-a177507c-terminal.json','terminal receipt path')
    require(hashlib.sha256(terminal_path.read_bytes()).hexdigest()==b['current_parent_terminal_sha256'],
        'terminal receipt changed')
    validate_terminal(json.loads(terminal_path.read_text()))
    return b

def validate_terminal(t):
    require(t['protocol']=='recursive-b-token-stop-parent-terminal-v1' and t['run_id']==PARENT_RUN
        and t['source_commit']==PARENT_COMMIT and t['status']=='PROGRAMMING_FAILURE_ALL_OWNED_KERNEL_EXITED',
        'own actual programming terminal')
    require(t['failed_outer']==9 and t['completed_native_outer']==8 and t['outer9_optimizer_calls']==0,
        'no completed update may be replayed')
    require(t['controller_pid']==1024017 and t['main_pid']==1026405 and
        t['worker_pids']==[1026891,1027302,1027303,1027304],'original process identities')
    for pid in [t['controller_pid'],t['main_pid'],*t['worker_pids']]:
        p=Path('/proc')/str(pid)
        if p.exists():
            state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
            require(state=='Z' and not list((p/'fd').glob('*')),'parent still live or holds descriptors')
    return t
