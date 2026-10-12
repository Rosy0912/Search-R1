"""Candidate only: externally authorized h100-5 DOWN/check/RESUME, never training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
OLD = Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
OLD_RUN = OLD/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
SOURCE = '080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
OLD_SOURCE = '60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
TOKEN = '13e07285c97044fd97974d94f53fea1b'
OLD_COMMENT = 'BAPO_FRESH200:formal200:'+OLD_SOURCE+':bab8981d9fd14afc98923eb4ebb89e33'
NEW_COMMENT = 'BAPO_ALIGNED_EM200:hardware:'+SOURCE+':'+TOKEN
DOWN = ['scontrol','update','NodeName=h100-5','State=DOWN','Reason=stopped3315_completion_stale']
RESUME = ['scontrol','update','NodeName=h100-5','State=RESUME']
EXE_SHA = 'be7770948fcfd24b074745d8add09005f77c1ae0ddb380b86abceb71d2bfdd6a'
BOOT = '9471cbc1-9ae1-4be5-8edb-1e84e850e2cc'
RISK = 'DOWN is node-global and can terminate allocations; RESUME can schedule other queued jobs and does not prove ghost cleanup.'
NODE_RECONCILIATION_ADMISSION = False  # New actual evidence: old h100-16 completion RPC/DNS failure is unresolved.

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def regular(path):
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and not p.is_symlink() and p.is_file(),'canonical regular original required')
    return p

def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def execute(argv):
    start=time.monotonic()
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        return dict(argv=argv,exit=p.returncode,stdout=p.stdout,stderr=p.stderr,elapsed=time.monotonic()-start)
    except Exception as e:
        return dict(argv=argv,exit=None,error=type(e).__name__+': '+str(e),elapsed=time.monotonic()-start,unknown_outcome=True)

def fields(raw): return dict(re.findall(r'(\w+)=([^\s]*)',raw))

def proc_identity():
    p=Path('/proc/755855');before=(p/'stat').read_text().rsplit(')',1)[1].split()
    uid=[int(v)for line in(p/'status').read_text().splitlines()if line.startswith('Uid:')for v in line.split()[1:]]
    argv=(p/'cmdline').read_bytes().rstrip(b'\0').replace(b'\0',b' ').decode().strip()
    exe=(p/'exe').resolve(strict=True);st=exe.stat()
    children={str(t.name):(t/'children').read_text().strip()for t in(p/'task').iterdir()}
    after=(p/'stat').read_text().rsplit(')',1)[1].split()
    require(before[19]==after[19]=='1433986891' and before[0]not in('D','Z','X')and after[0]not in('D','Z','X'),'ghost daemon birth/state changed')
    require(argv=='slurmstepd: [3315.0]' and before[1]=='1' and uid==[0]*4,'ghost daemon identity changed')
    require(str(exe)=='/usr/sbin/slurmstepd-wlm' and digest(exe)==EXE_SHA and all(not v for v in children.values()),'ghost executable/children changed')
    return dict(pid=755855,start_ticks=int(after[19]),ppid=1,uid=uid,argv=argv,exe=str(exe),exe_sha256=EXE_SHA,
                exe_inode=st.st_ino,exe_device=st.st_dev,children=children,cgroup=(p/'cgroup').read_text(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

QUERIES = dict(node=['scontrol','show','node','h100-5','-o'],old=['scontrol','show','job','3315','-o'],
    new=['scontrol','show','job','3345','-o'],node_queue=['squeue','-h','-w','h100-5','-o','%i|%T|%N|%j|%k'],
    queue=['squeue','-h','-o','%i|%T|%N|%j|%k'],pids=['scontrol','listpids','3315'],
    steps=['scontrol','show','step','3315'],gpu=['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],
    version=['scontrol','--version'],source=['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],
    old_source=['git','-C',str(OLD/'repo'),'rev-parse','HEAD'],processes=['ps','-eo','pid,ppid,args'])

def check_snapshot(rows, daemon, down=False):
    require(all(v.get('exit')==0 for v in rows.values()),'fresh observation failed/unknown')
    node,old,new=[fields(rows[k]['stdout'])for k in ('node','old','new')]
    require(node.get('NodeName')=='h100-5'and node.get('Version')=='21.08.5'and node.get('CPUAlloc')=='0'
        and node.get('AllocMem')=='0'and node.get('AllocTRES')==''and node.get('Gres')=='gpu:h100:8','node has unknown/allocated resources')
    states=set(node.get('State','').split('+'))
    require(('DOWN'in states)if down else('IDLE'in states and 'COMPLETING'in states and not states-{'IDLE','COMPLETING','PLANNED'}),'unexpected node state')
    require(old.get('JobId')=='3315'and old.get('Comment')==OLD_COMMENT and old.get('WorkDir')==str(OLD)
        and old.get('Command')==str(OLD/'tools/formal200.sbatch')and old.get('NodeList')=='h100-5'
        and old.get('Requeue')=='0'and old.get('Restarts')=='0','foreign/changed old3315')
    require(old.get('JobState')in({'COMPLETING','CANCELLED','FAILED','NODE_FAIL'}if down else{'COMPLETING'}),'old job state changed')
    require(new.get('JobId')=='3345'and new.get('Comment')==NEW_COMMENT and new.get('JobName')=='bapo-aligned-em200-hardware'
        and new.get('WorkDir')==str(ROOT)and new.get('ReqNodeList')=='h100-5'and new.get('NodeList')=='(null)'
        and new.get('JobState')=='PENDING'and new.get('Restarts')=='0'and new.get('StartTime','')not in('','Unknown')
        and new.get('RunTime')=='00:00:00','original3345 changed or started')
    command=re.search(r'(?:^|\s)Command=(.*?)(?=\s+\w+=|$)',rows['new']['stdout'])
    require(command and command.group(1).strip()==str(ROOT/'tools/deployment/hardware.sbatch')+' '+TOKEN,'hardware entry changed')
    own=[line for line in rows['queue']['stdout'].splitlines()if 'BAPO_ALIGNED_EM200:hardware:'+SOURCE+':'in line]
    require(len(own)==1 and own[0].split('|')[0]=='3345','duplicate/changed current hardware submission')
    for line in rows['node_queue']['stdout'].splitlines():
        item=line.split('|');require(len(item)==5,'unknown node queue schema')
        require(item[1]=='PENDING' or(item[0]=='3315'and item[1]=='COMPLETING'and item[4]==OLD_COMMENT),'foreign allocated Slurm job')
    require(not rows['gpu']['stdout'].strip(),'GPU compute exists')
    require(rows['pids']['stdout'].split()==['PID','JOBID','STEPID','LOCALID','GLOBALID'],'old3315 still has tasks/unknown listpids schema')
    steps=set(re.findall(r'StepId=([^\s]+)',rows['steps']['stdout']))
    require(steps<={'3315.batch','3315.0'} and (down or steps=={'3315.batch','3315.0'}),'unexpected old step identity')
    require(rows['version']['stdout'].strip()=='slurm-wlm 21.08.5'and rows['source']['stdout'].strip()==SOURCE
        and rows['old_source']['stdout'].strip()==OLD_SOURCE,'installed/source version changed')
    ghosts=[line.strip()for line in rows['processes']['stdout'].splitlines()if re.search(r'slurmstepd: \[',line)]
    missing=down and daemon.get('absent')is True
    require((not ghosts)if missing else(len(ghosts)==1 and re.fullmatch(r'755855\s+1\s+slurmstepd: \[3315\.0\]',ghosts[0])),'foreign/unexpected Slurm step daemon')
    require(not any(str(OLD_RUN)in line or(str(OLD/'env')in line)for line in rows['processes']['stdout'].splitlines()),'old training/runtime still alive')
    require(daemon.get('boot_id')==BOOT and (missing or(daemon.get('pid')==755855 and daemon.get('start_ticks')==1433986891
        and daemon.get('exe_sha256')==EXE_SHA)),'fresh exact ghost process proof missing')
    require(not(ROOT/'runs'/('hardware-aligned-em200-'+TOKEN+'-3345')).exists(),'3345 RUN exists: execution already started')

def snapshot(folder,name,runner=execute,process=proc_identity,down=False):
    rows={}
    for key,argv in QUERIES.items():
        value=runner(argv);save(folder/(name+'-'+key+'.json'),value);rows[key]=value
    try:daemon=process()
    except FileNotFoundError:
        require(down and not Path('/proc/755855').exists(),'ghost proof unavailable')
        daemon=dict(absent=True,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(folder/(name+'-ghost.json'),daemon)
    check_snapshot(rows,daemon,down);return dict(commands=rows,daemon=daemon)

def authorization(path,sha,plan_sha):
    path=regular(path);require(ROOT/'evidence'in path.parents,'authorization must be in own evidence')
    require(digest(path)==sha,'external authorization bytes changed')
    value=json.loads(path.read_text())
    require(value.get('status')=='EXPLICIT_HUMAN_AUTHORIZATION_NODE_GLOBAL_DOWN_RESUME'and value.get('node')=='h100-5'
        and value.get('commands')==[DOWN,RESUME]and value.get('plan_sha256')==plan_sha and value.get('risk_acknowledgement')==RISK
        and value.get('human_instruction_reference')and value.get('approved_utc'),'explicit additional human node-global authorization missing')
    return dict(path=str(path),sha256=sha,value=value)

def apply(plan_path,plan_sha,auth_path,auth_sha,output,runner=execute,process=proc_identity):
    require(NODE_RECONCILIATION_ADMISSION,'BLOCKED: old-controller completion RPC/DNS cause unresolved; DOWN/RESUME is not an established remedy')
    plan_path=regular(plan_path);require(digest(plan_path)==plan_sha,'reviewed plan changed')
    plan=json.loads(plan_path.read_text());require(plan.get('commands')==[DOWN,RESUME]and plan.get('source')==SOURCE
        and plan.get('root')==str(ROOT)and plan.get('scope')=='CANDIDATE_NODE_GLOBAL_REPAIR_REQUIRES_NEW_HUMAN_AUTHORIZATION','foreign reviewed plan')
    require(plan.get('helper_sha256')==digest(__file__),'reviewed helper changed')
    auth=authorization(auth_path,auth_sha,plan_sha)
    output=Path(output);require(output.parent==ROOT/'evidence'and output.parent.resolve()==output.parent,'own exclusive output required')
    output.mkdir();save(output/'authorization.json',auth);save(output/'reviewed-plan.json',plan)
    snapshot(output,'initial',runner,process)
    archive_old(output)
    snapshot(output,'immediate-before-down',runner,process)
    for index,(name,command)in enumerate((('down',DOWN),('resume',RESUME)),1):
        require(digest(plan_path)==plan_sha,'plan changed before mutation')
        authorization(auth_path,auth_sha,plan_sha)
        save(output/(name+'-intent.json'),dict(argv=command,authorization_sha256=auth_sha,plan_sha256=plan_sha,ordinal=index,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unknown_outcome_if_no_receipt=True))
        receipt=runner(command);save(output/(name+'-receipt.json'),receipt)
        require(receipt.get('exit')==0,'mutation failed/unknown; stop, no retries or escalation')
        if name=='down':snapshot(output,'after-down-before-resume',runner,process,down=True)
        else:
            for key in ('node','old','new','node_queue','queue','gpu'):
                save(output/('after-resume-'+key+'.json'),runner(QUERIES[key]))
    save(output/'result.json',dict(status='TWO_NODE_COMMANDS_RETURNED_ZERO_NOT_CLEANUP_OR_HARDWARE_ACCEPTANCE',
        training_permission=False,cleanup_verified=False,risk=RISK))

def archive_old(folder):
    paths=[OLD/'logs/slurm-formal200-3315.out',OLD/'logs/slurm-formal200-3315.err',OLD_RUN/'train.log',
           OLD_RUN/'source-commit.txt',OLD/'tools/formal200.sbatch']
    records=[]
    for index,path in enumerate(paths):
        path=regular(path);before=path.stat();target=folder/f'old-original-{index}.raw'
        with path.open('rb')as src,target.open('xb')as dst:
            for chunk in iter(lambda:src.read(1048576),b''):dst.write(chunk)
            dst.flush();os.fsync(dst.fileno())
        after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'old original changed during archive')
        records.append(dict(path=str(path),archive=str(target),bytes=after.st_size,sha256=digest(target)))
    require((OLD_RUN/'source-commit.txt').read_text().strip()==OLD_SOURCE,'old run source marker changed')
    save(folder/'old-originals.json',records)

def plan(output):
    value=dict(scope='CANDIDATE_NODE_GLOBAL_REPAIR_REQUIRES_NEW_HUMAN_AUTHORIZATION',source=SOURCE,root=str(ROOT),node='h100-5',
        commands=[DOWN,RESUME],helper_sha256=digest(__file__),risk=RISK,authorization_present=False,
        execution_admission=False,diagnosis_status='BLOCKED_OLD_H10016_CONTROLLER_COMPLETION_RPC_DNS_UNRESOLVED',
        mutation_executed=False,old_job='3315',never_resume_old=True,preserve_existing_new_job='3345',no_second_submission=True,
        failure='Stop on any unknown/nonzero/preflight mismatch; no automatic retry/escalation/daemon operation.')
    save(output,value);return value

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('plan');a.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('apply')
    for name in ('plan','authorization','output'):b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--plan-sha256',required=True);b.add_argument('--authorization-sha256',required=True)
    args=p.parse_args()
    if args.action=='plan':print(json.dumps(plan(args.output)));return
    require(sys.platform=='linux'and os.uname().nodename=='h100-5'and Path(sys.prefix).resolve()==ROOT/'env','actual new isolated h100-5 Linux runtime required')
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'unoptimized interpreter required')
    apply(args.plan,args.plan_sha256,args.authorization,args.authorization_sha256,args.output)

if __name__=='__main__':main()
