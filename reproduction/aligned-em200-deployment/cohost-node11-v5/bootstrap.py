"""Acquire true inherited eight-UUID OFD leases; start canonical cohost guard."""
from pathlib import Path
import argparse,fcntl,json,os,stat,sys,time
import resource_watch
from resource_watch import ROOT,identity,require,snapshot,admission

def acquire(records,directory):
 directory.mkdir(mode=0o700,exist_ok=True)
 require(directory.is_dir() and not directory.is_symlink() and directory.stat().st_uid==os.getuid(),'own lease directory required')
 leases=[]
 for record in sorted(records,key=lambda r:r['uuid']):
  p=directory/(record['uuid']+'.lock');fd=os.open(p,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);s=os.fstat(fd)
  require(stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1,'ordinary own UUID lock required')
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(fd,True);leases.append(dict(uuid=record['uuid'],fd=fd))
 return leases

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--minimum-start-free-mib',type=float,default=30720);parser.add_argument('--reserve-free-mib',type=float,default=1024);parser.add_argument('command',nargs=argparse.REMAINDER);a=parser.parse_args();command=a.command[1:]if a.command[:1]==['--']else a.command
 require(Path(resource_watch.__file__).resolve()==ROOT/'tools/resource_watch.py','canonical bound watcher import required')
 identity(a.run);require(command,'command required');admission([],a.reserve_free_mib)
 selected=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')if x];before=snapshot(selected);require(admission(before,a.minimum_start_free_mib),'each GPU needs >=30GiB initial free')
 leases=acquire(before,Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}');time.sleep(.2);after=snapshot(selected)
 require({x['uuid']for x in before}=={x['uuid']for x in after} and admission(after,a.minimum_start_free_mib),'GPU identity/free memory changed after locking')
 with (a.run/'cohost-lease-bootstrap.json').open('x') as f:
  json.dump(dict(policy='COHOST_NODE11_INITIAL_FREE_NOT_IDLE',before=before,after=after,leases=leases,minimum_start_free_mib=a.minimum_start_free_mib,run=str(a.run),job=os.environ['SLURM_JOB_ID']),f);f.flush();os.fsync(f.fileno())
 os.environ['SEARCH_R1_GPU_PREFLIGHT']='1';os.environ['SEARCH_R1_GPU_LEASES']=json.dumps(leases)
 python=str(ROOT/'env/bin/python');watch=ROOT/'tools/resource_watch.py';require(watch.is_file() and watch.resolve()==watch,'canonical cohost watcher required')
 os.execv(python,[python,str(watch),'--run',str(a.run),'--reserve-free-mib',str(a.reserve_free_mib),'--']+command)
if __name__=='__main__':main()
