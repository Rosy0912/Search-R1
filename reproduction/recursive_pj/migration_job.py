"""One PT allocation: read-only full-state probe, explicit cutover gate, continuation."""
import json,os,subprocess,sys,time
from .resume_transport import STAGE,TARGET_JOB,GROUP,load_inputs,validate_gate

def storage_admission(path=STAGE, minimum_bytes=400_000_000_000):
    stat=os.statvfs(path)
    available=stat.f_bavail*stat.f_frsize
    if available < minimum_bytes:
        raise RuntimeError('Insufficient checkpoint headroom: '+str(available))
    return available

def main():
    job=os.environ['SEARCH_R1_JOB_ID']
    if job!=TARGET_JOB or os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Require the explicit unique PT continuation allocation')
    storage_admission()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
    if len(rows)!=8:raise ValueError('Exactly eight allocated GPUs required')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(x.strip() for x in rows)
    binding,_,_=load_inputs(verify_members=True)
    result=STAGE/('result-'+job+'.json')
    if result.exists():raise ValueError('Existing probe requires inspection; do not repeat a submitted job')
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
        '-m','reproduction.recursive_pj.restore_probe'],check=True)
    probe=json.loads(result.read_text())
    print('PJ_MIGRATION_PROBE_COMPLETE='+json.dumps(dict(job=job,result=str(result))),flush=True)
    deadline=time.monotonic()+3600
    while not (STAGE/'cutover.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('No authorized terminal-parent cutover within one hour; no training started')
        time.sleep(5)
    validate_gate(json.loads((STAGE/'cutover.json').read_text()),binding,probe)
    storage_admission()
    os.environ['RECURSIVE_PJ_NATIVE_CONTINUATION']='1'
    from .entry import launch
    launch(GROUP)
if __name__=='__main__':main()
