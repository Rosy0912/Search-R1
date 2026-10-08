"""Separate first5 collector adapter only. Does not replace frozen training-source verifier."""
import os,re
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006'
NONCE='bab8981d9fd14afc98923eb4ebb89e33';RUN=ROOT+'/runs/formal200-'+NONCE+'-3315'
ORIGINAL_SOURCE_SHA='258d6e15bbf6f60b7ca5beb2fd878838d42039dab021bbc1676429c520070f63'
def require(ok,message):
    if not ok:raise ValueError('Fresh200 native evidence: '+message)
def strict_allocation(raw,job,hostname,run,source,expected_uid):
    require(type(raw)is str and type(job)is str and job=='3315'and source==SOURCE and run==RUN,'Exact new actual3315 source/run/job required')
    require(type(expected_uid)is int and expected_uid>=0,'Actual target numeric UID required')
    fields={}
    for match in re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/]*)=([^\s]*)',raw):
        k,v=match.groups();require(k not in fields,'Duplicate Slurm field '+k);fields[k]=v
    expected=dict(JobId='3315',JobState='RUNNING',NodeList='h100-5',BatchHost='h100-5',NumNodes='1',Restarts='0',Requeue='0',Comment='BAPO_FRESH200:formal200:'+SOURCE+':'+NONCE,Command=ROOT+'/tools/formal200.sbatch',WorkDir=ROOT)
    require(all(fields.get(k)==v for k,v in expected.items()),'Actual Slurm identity/node/source/restart mismatch')
    require(hostname=='h100-5'and hostname==fields['NodeList']==fields['BatchHost'],'Actual process hostname/allocation mismatch')
    owner=re.fullmatch(r'[^()\s]+\((\d+)\)',fields.get('UserId',''))
    require(owner is not None and int(owner.group(1))==expected_uid,'Allocation owner differs from current target UID')
    require(not('TresPerNode'in fields and 'TRESPerNode'in fields),'Duplicate GPU per-node field aliases')
    claims=[]
    for key in ('TRES','AllocTRES','TresPerNode','TRESPerNode'):
        if key not in fields:continue
        seen=set();types=[]
        for item in fields[key].split(','):
            if not(item.startswith('gres/gpu')or item.startswith('gres:gpu')):continue
            pattern=(r'gres/gpu(?::([A-Za-z0-9_-]+))?[=:](\d+)'if item.startswith('gres/gpu')else r'gres:gpu(?::([A-Za-z0-9_-]+))?:(\d+)')
            match=re.fullmatch(pattern,item);require(match is not None,'Unsupported/ambiguous GPU count field')
            gpu_type,count=match.groups();require(gpu_type in (None,'h100'),'Explicit foreign GPU type rejected');require(gpu_type not in seen,'Duplicate GPU type/count claim');seen.add(gpu_type);types.append(gpu_type)
            require(int(count)==8 and count=='8','Exactly eight GPUs required, not substring80/18')
            claims.append((key,gpu_type,int(count)))
        require(len(types)<=2 and (len(types)<2 or None in types),'Ambiguous multiple typed GPU populations')
    require(claims,'Missing explicit GPU count; unrelated :8 not evidence')
    typed={t for _,t,_ in claims if t is not None};require(len(typed)<=1,'Contradictory GPU types across allocation fields')
    return dict(job_id=3315,hostname=hostname,allocation_uid=expected_uid,gpu_count=8,raw_gpu_count_fields=claims,scope='STRICT_UNMODIFIED_SLURM_FORMAT_REPLAY_NOT_NATIVE_ACCEPTANCE')

def verify_native_process(process,source,expected_uid=None):
    require(process.get('source_head')==source,'actual startup source HEAD mismatch')
    stat=process.get('proc_stat_raw','');require(stat.startswith(str(process.get('pid'))+' ('),'native PID raw stat mismatch')
    fields=stat[stat.rfind(')')+2:].split()
    require(len(fields)>19 and fields[19]==process.get('proc_start_ticks'),'native PID birth mismatch')
    maps=process.get('proc_maps_raw','')
    require('libcuda' in maps and ('libtorch_cuda' in maps or 'libcudart' in maps),'native CUDA runtime mappings missing')
    leases=process.get('gpu_leases',[]);guard=process.get('resource_guard',{});guard_stat=process.get('resource_guard_stat_raw','')
    require(len(leases)==8 and len({x.get('uuid') for x in leases})==8 and process.get('gpu_uuid') in {x.get('uuid') for x in leases},'native UUID not held in eight-card guardian lease inventory')
    guard_fields=guard_stat[guard_stat.rfind(')')+2:].split()
    require(guard_stat.startswith(str(guard.get('pid'))+' (') and len(guard_fields)>19 and guard_fields[19]==str(guard.get('start_ticks'))
            and guard.get('run')==process.get('run_root') and 'resource_watch.py' in process.get('resource_guard_cmdline','')
            and process.get('run_root') in process.get('resource_guard_cmdline',''),'actual guardian PID birth/run binding invalid')
    job=process.get('job_id');allocation=process.get('allocation_raw','')
    strict_allocation(allocation,job,process.get('hostname'),process.get('run_root'),source,os.getuid()if expected_uid is None else expected_uid)
    require(str(job) in process.get('proc_cgroup_raw',''),'native process not bound to actual Slurm job cgroup')
    import csv
    inventory=[list(map(str.strip,row)) for row in csv.reader(process.get('nvidia_smi_raw','').splitlines())]
    visible=process.get('cuda_visible_devices',[]);logical=int(process.get('device','-1'))
    require(1<=len(visible)<=8 and 0<=logical<len(visible),'native logical GPU mapping invalid')
    token=visible[logical];matches=[row for row in inventory if row[0]==token or row[1]==token]
    require(len(matches)==1 and matches[0][1]==process.get('gpu_uuid') and matches[0][3]==process.get('gpu_pci_bus_id'),'native scheduler-visible UUID/PCI mapping mismatch')
    require(process.get('gpu_uuid') in process.get('nvidia_smi_raw','') and len(process.get('nvidia_smi_raw','').splitlines())>=8,'actual GPU UUID inventory mismatch')

def conservation_proof(original_path):
    """Live CPU AST witness for copied guards; exact original source must match frozen SHA."""
    import ast,hashlib
    from pathlib import Path
    raw=Path(original_path).read_bytes();require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SOURCE_SHA,'Original verifier bytes changed')
    original=ast.parse(raw);adapter_raw=Path(__file__).read_bytes();adapter=ast.parse(adapter_raw)
    a=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name=='verify_native_process').body
    b=next(n for n in adapter.body if isinstance(n,ast.FunctionDef)and n.name=='verify_native_process').body
    index=next(i for i,n in enumerate(a)if isinstance(n,ast.Expr)and 'actual running Slurm allocation mismatch'in ast.unparse(n))
    require('actual eight-GPU allocation not established'in ast.unparse(a[index+1])and isinstance(b[index],ast.Expr)and isinstance(b[index].value,ast.Call)and isinstance(b[index].value.func,ast.Name)and b[index].value.func.id=='strict_allocation','Exact allocation-only replacement required')
    dumps=lambda nodes:[ast.dump(n,include_attributes=False)for n in nodes]
    left=dumps(a[:index]+a[index+2:]);right=dumps(b[:index]+b[index+1:]);require(left==right,'Original nonallocation native checks changed')
    digest=lambda x:hashlib.sha256(__import__('json').dumps(x,sort_keys=True).encode()).hexdigest()
    return dict(scope='EXACT_ORIGINAL_NONALLOCATION_AST_CONSERVED_ALLOCATION_REQUIRE_PAIR_REPLACED_ONLY',original_source_sha256=ORIGINAL_SOURCE_SHA,adapter_sha256=hashlib.sha256(adapter_raw).hexdigest(),original_conserved_ast_sha256=digest(left),adapter_conserved_ast_sha256=digest(right),conserved_statements=len(left),replaced_original_allocation_require_ast_sha256=digest(dumps(a[index:index+2])),native_acceptance=False)
