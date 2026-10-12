"""Freeze only the exact own node11 formal retriever birth."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v8-20261009')
SOURCE='e82ef0231a23eedfd68ebba1644966a5759a7575'
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run_path():
    run=Path(os.environ['BAPO_RUN_ID']);job=os.environ['SLURM_JOB_ID']
    require(run.parent==ROOT/'runs' and run.resolve()==run and run.is_dir() and run.name.startswith('formal-aligned-em200-') and run.name.endswith('-'+job),'canonical same-job formal run required')
    require(sys.prefix==str(ROOT/'env') and os.uname().nodename=='h100-11','new local environment required')
    return run,job
def write(path,value):
    require(path.parent.resolve()==path.parent and not path.is_symlink(),'canonical new evidence path required')
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def birth(pid,run,job):
    proc=Path('/proc')/str(pid)
    def state():
        f=(proc/'stat').read_text().rsplit(')',1)[1].split()
        require(f[0] not in ('Z','X'),'retriever is not alive')
        return dict(start_ticks=int(f[19]),ppid=int(f[1]))
    before=state();cmd=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    env=(proc/'environ').read_bytes().split(b'\0');exe=(proc/'exe').resolve();st=exe.stat()
    require(before==state(),'retriever birth changed during observation')
    require(proc.stat().st_uid==os.getuid(),'retriever UID differs')
    require(('BAPO_RUN_ID='+str(run)).encode() in env and ('SLURM_JOB_ID='+job).encode() in env,'retriever ownership differs')
    expected=[str(ROOT/'env/bin/python'),'-B','reproduction/h20/retriever_local.py',
        '--index_path',str(Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/wiki/e5_Flat.index')),
        '--corpus_path',str(Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/wiki/wiki-18.jsonl')),
        '--retriever_model',str(Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/e5-base-v2')),
        '--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18083','--threads','8']
    require(cmd==expected and exe==(ROOT/'env/bin/python').resolve(),'exact launched GPU retriever command differs')
    require(before['ppid']==int(os.environ['BAPO_RETRIEVER_PARENT_PID']),'retriever parent differs')
    return dict(pid=pid,**before,uid=os.getuid(),command=cmd,executable=dict(path=str(exe),device=st.st_dev,inode=st.st_ino),environment_marker=dict(run=str(run),job=job))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['process']);parser.parse_args()
    run,job=run_path();pid=int(os.environ['BAPO_LOCAL_RETRIEVER_PID']);deadline=time.monotonic()+10
    while True:
        try:value=birth(pid,run,job);break
        except (OSError,ValueError):
            if time.monotonic()>=deadline:raise
            time.sleep(.05)
    write(run/'own-retriever-birth.json',dict(source=SOURCE,run=str(run),job=job,identity=value,scope='OWN_NODE11_GPU_RETRIEVER_BIRTH_BEFORE_READY_NOT_HARDWARE_ACCEPTANCE'))
if __name__=='__main__':main()
