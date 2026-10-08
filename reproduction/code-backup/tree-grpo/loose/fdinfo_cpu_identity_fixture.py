"""Tiny same-node Linux CPU-only flock experiment, own temporary file only."""
import fcntl,json,os,socket,subprocess,sys,tempfile
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
with tempfile.TemporaryDirectory(prefix='tree-fdinfo-fixture-') as tmp:
    p=Path(tmp)/'synthetic.lock'
    with p.open('w') as parent:
        fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=Path('/proc/self/fdinfo',str(parent.fileno())).read_text()
        st=os.fstat(parent.fileno())
        fcntl.flock(parent,fcntl.LOCK_UN)
        child=subprocess.Popen([sys.executable,'-c',
            'import fcntl,sys; f=open(sys.argv[1]); fcntl.flock(f,fcntl.LOCK_EX); print("READY",flush=True); sys.stdin.readline()',str(p)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            assert child.stdout.readline().strip()=='READY'
            after=Path('/proc/self/fdinfo',str(parent.fileno())).read_text()
            with p.open() as probe:
                try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:someone=True
                else:someone=False
            assert someone and 'FLOCK' in before and 'FLOCK' not in after
            print(json.dumps(dict(status='ACTUAL_H1009_CPU_FDINFO_OFD_OWNER_DISTINGUISHED',
                before=before,after=after,device_major=os.major(st.st_dev),device_minor=os.minor(st.st_dev),inode=st.st_ino,
                somebody_holds_lock_after=True,parent_same_inode_fd_has_no_flock=True,
                cuda_imported=False,gpu_calls=0,model_forwards=0,optimizer_writes=0,
                scope='One own synthetic temporary file; no real GPU leases, background PIDs, tensor assets or Slurm jobs'),sort_keys=True))
        finally:
            child.communicate('\n',timeout=5)
            assert child.returncode==0
