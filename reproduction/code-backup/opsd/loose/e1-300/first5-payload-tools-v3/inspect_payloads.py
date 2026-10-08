"""Read-only native CPU stored-state inspection. Never restores a live actor or signs."""
import argparse, hashlib, importlib.util, json, math, os, pathlib, random, re, struct, subprocess, sys
import types
import numpy as np
import torch
from torch.distributed._tensor import DTensor
P=pathlib.Path
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
NUMEL=3085938688
COUNTERS=('cpu_seconds','wall_seconds','transfer_wall_seconds','cuda_update_entries','cuda_seconds','scored_tokens','load_cpu_seconds','load_wall_seconds')

def require(ok, message):
    if not ok: raise ValueError(message)

def unique(raw):
    def pairs(rows):
        out={}
        for k,v in rows:
            require(k not in out,'duplicate JSON key'); out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def sha(path):
    path=P(path); require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),'nonphysical file')
    before=path.stat(); h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    after=path.stat()
    require(tuple(getattr(before,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))==tuple(getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'file mutated')
    return h.hexdigest()

def record(record, root, pins):
    require(set(record)=={'path','sha256'},'invalid file record'); p=P(record['path'])
    require(p.is_relative_to(root) and p.stat().st_uid==os.getuid(),'file outside owned run')
    digest=sha(p); require(digest==record['sha256'],'file SHA mismatch'); pins[str(p)]=digest
    return p

def finite(t):
    require(torch.is_tensor(t) and t.device.type=='cpu','nonCPU tensor')
    flat=t.detach().reshape(-1)
    for part in flat.split(1024**2):
        require(bool(torch.isfinite(part).all()),'nonfinite tensor content')

def tensor_hash(t):
    finite(t); t=t.detach().contiguous(); h=hashlib.sha256()
    for part in t.reshape(-1).split(1024**2): h.update(memoryview(part.view(torch.uint8).numpy()).cast('B'))
    return h.hexdigest()

def layout(t, rank, world=8):
    require(isinstance(t,DTensor),'model value is not true DTensor')
    mesh=t.device_mesh; local=t.to_local()
    require(mesh.device_type=='cuda' and list(mesh.mesh.shape)==[world] and mesh.mesh.flatten().tolist()==list(range(world)),'stored mesh topology differs')
    require(tuple(mesh.mesh_dim_names)==('fsdp',) and list(mesh.get_coordinate())==[rank],'stored rank mesh coordinate differs')
    require(len(t.placements)==1 and type(t.placements[0]).__name__=='Shard' and t.placements[0].dim==0,'stored placement differs')
    require(t.dtype==local.dtype==torch.float32 and local.device.type=='cpu','master shard dtype/device differs')
    shape=list(t.shape); require(shape and shape[0]%world==0 and list(local.shape)==[shape[0]//world]+shape[1:],'stored local/global shape differs')
    finite(local)
    return dict(global_shape=shape,global_stride=list(t.stride()),global_dtype=str(t.dtype),placements=[str(v) for v in t.placements],mesh_device_type=mesh.device_type,mesh_shape=list(mesh.mesh.shape),mesh_ranks=mesh.mesh.flatten().tolist(),mesh_dim_names=list(mesh.mesh_dim_names),mesh_coordinate=list(mesh.get_coordinate()),local_shape=list(local.shape),local_stride=list(local.stride()),local_dtype=str(local.dtype),local_bytes_sha256=tensor_hash(local))

def adam(state, expected_local_numel, expected_steps=100):
    require(set(state)=={'state','param_groups'} and state['state'] and state['param_groups'],'Adam schema missing')
    parameters=[]; lrs=[]
    for group in state['param_groups']:
        require(group['betas']==(0.9,0.999) and group['eps']==1e-8 and group['weight_decay']==.01,'Adam hyperparameters differ')
        require(group.get('amsgrad') is False and group.get('maximize') is False,'unsupported Adam mode')
        require(math.isfinite(group['lr']) and abs(group['lr']-1e-6*5/57)<1e-18,'Adam saved LR differs')
        parameters.extend(group['params']); lrs.append(group['lr'])
    require(all(type(v) is int for v in parameters) and len(set(parameters))==len(parameters) and set(parameters)==set(state['state']),'Adam parameter/state binding differs')
    entries=[]; numel=0
    for param in parameters:
        v=state['state'][param];require(set(v)=={'step','exp_avg','exp_avg_sq'},'Adam moment schema differs')
        step=v['step'];require(torch.is_tensor(step) and step.device.type=='cpu' and step.numel()==1 and math.isfinite(float(step)) and float(step)==expected_steps,'Adam actual step differs')
        a,b=v['exp_avg'],v['exp_avg_sq']; require(a.dtype==b.dtype==torch.float32 and a.shape==b.shape and a.ndim==1 and a.numel()>0,'Adam moment layout differs')
        finite(a);finite(b)
        for part in b.split(1024**2):require(bool((part>=0).all()),'negative Adam second moment')
        numel+=a.numel(); entries.append(dict(parameter_id=param,step=int(float(step)),shape=list(a.shape),dtype=str(a.dtype),exp_avg_sha256=tensor_hash(a),exp_avg_sq_sha256=tensor_hash(b)))
    require(numel==expected_local_numel,'Adam complete rank-local flattened parameter numel differs')
    return dict(entries=entries,local_parameter_numel=numel,lrs=lrs,steps=sorted({v['step'] for v in entries}))

def scheduler(s):
    require(s['last_epoch']==5 and s['_step_count']==6 and s['base_lrs']==[1e-6],'scheduler actual clock differs')
    require(len(s['_last_lr'])==1 and abs(s['_last_lr'][0]-1e-6*5/57)<1e-18,'scheduler warmup LR differs')
    require(s['lr_lambdas']==[None],'scheduler lambda state differs')
    return dict(last_epoch=s['last_epoch'],step_count=s['_step_count'],base_lrs=s['base_lrs'],last_lr=s['_last_lr'])

def rng(r, rank_cuda):
    require(set(r)=={'python','numpy','torch','cuda'},'RNG component missing')
    py=r['python']; npstate=r['numpy']
    require(type(py) is tuple and len(py)==3 and (py[2] is None or (type(py[2]) in (int,float) and math.isfinite(py[2]))),'nonfinite Python RNG metadata')
    require(type(npstate) is tuple and len(npstate)==5 and npstate[0]=='MT19937' and isinstance(npstate[1],np.ndarray) and npstate[1].dtype==np.uint32 and npstate[1].shape==(624,) and type(npstate[2]) is int and 0<=npstate[2]<=624 and type(npstate[3]) is int and npstate[3] in (0,1) and math.isfinite(float(npstate[4])),'invalid/nonfinite numpy RNG metadata')
    random.Random().setstate(py); np.random.RandomState().set_state(npstate)
    cpu=r['torch'];require(torch.is_tensor(cpu) and cpu.device.type=='cpu' and cpu.dtype==torch.uint8 and cpu.ndim==1 and cpu.numel()>100,'CPU RNG bytes missing');torch.Generator().set_state(cpu)
    require(len(r['cuda'])==(1 if rank_cuda else 0),'rank-local CUDA RNG entry differs')
    for entry in r['cuda']:
        require(set(entry)=={'device','state'} and type(entry['device']) is int and entry['device']>=0,'rank-local RNG device missing')
        t=entry['state']; require(torch.is_tensor(t) and t.device.type=='cpu' and t.dtype==torch.uint8 and t.ndim==1 and t.numel()==16,'CUDA Philox stored RNG bytes differ')
    return dict(python_state_valid=True,numpy_algorithm=r['numpy'][0],numpy_state_sha256=hashlib.sha256(r['numpy'][1].tobytes()).hexdigest(),torch_cpu_state_sha256=tensor_hash(cpu),cuda=[dict(device=v['device'],bytes=v['state'].numel(),sha256=tensor_hash(v['state'])) for v in r['cuda']])

class Base:
    def __init__(self,directory,manifest,pins):
        self.directory=P(directory); require(self.directory.resolve()==self.directory,'base directory alias')
        require(manifest['repo']=='Qwen/Qwen2.5-3B-Instruct' and manifest['revision']==REVISION and len(manifest['files'])==11,'official base identity differs')
        for r in manifest['files']:
            p=self.directory/r['file']; require(P(r['file']).name==r['file'] and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'official base content differs');pins[str(p)]=r['sha256']
        self.config=unique((self.directory/'config.json').read_bytes())
        self.index=unique((self.directory/'model.safetensors.index.json').read_bytes())['weight_map'];self.headers={};self.starts={}
        for name in sorted(set(self.index.values())):
            require(P(name).name==name,'unsafe shard path')
            with (self.directory/name).open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0];require(0<n<64*1024**2,'invalid header');header=unique(f.read(n));self.starts[name]=8+n
            for k,v in header.items():
                if k=='__metadata__':continue
                require(k not in self.headers and self.index[k]==name and v['dtype']=='BF16','official header inventory differs');self.headers[k]=v
        require(set(self.headers)==set(self.index) and len(self.headers)==434 and sum(math.prod(v['shape']) for v in self.headers.values())==NUMEL,'official unique parameter inventory differs')
    def local(self,key,rank,world=8):
        source='model.embed_tokens.weight' if key=='lm_head.weight' else key
        v=self.headers[source];shape=v['shape'];require(shape[0]%world==0,'base shape not divisible')
        row_bytes=math.prod(shape[1:])*2; count=shape[0]//world; start=v['data_offsets'][0]+rank*count*row_bytes;size=count*row_bytes
        with (self.directory/self.index[source]).open('rb') as f:f.seek(self.starts[self.index[source]]+start);raw=f.read(size)
        require(len(raw)==size,'base shard read truncated')
        return torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape([count]+shape[1:]).to(torch.float32)
    def named_parameter_digest(self):
        # Exact frozen full_state.state_digest(dict(named_parameters)) tensor format,
        # streamed in BF16->FP32 chunks, with no model load or full 12GiB dict.
        h=hashlib.sha256();h.update(('DICT:'+str(len(self.headers))+':').encode())
        for key in sorted(self.headers):
            v=self.headers[key];shape=tuple(v['shape']);stride=[];value=1
            for dim in reversed(shape):stride.insert(0,value);value*=dim
            h.update(str(('str',repr(key))).encode());h.update(str(('torch.float32',shape,tuple(stride))).encode())
            left=v['data_offsets'][1]-v['data_offsets'][0]
            with (self.directory/self.index[key]).open('rb') as f:
                f.seek(self.starts[self.index[key]]+v['data_offsets'][0])
                while left:
                    raw=f.read(min(left,8*1024**2));require(raw and len(raw)%2==0,'base tensor stream truncated');left-=len(raw)
                    fp32=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).to(torch.float32);h.update(memoryview(fp32.view(torch.uint8).numpy()).cast('B'))
        return h.hexdigest()

class Export:
    """Read only actual externally manifest-pinned HF tensor slices, not a model."""
    def __init__(self,paths,expected_keys):
        self.index_path=paths['actor/model.safetensors.index.json']
        index=unique(self.index_path.read_bytes())
        self.index=index['weight_map'];require(isinstance(self.index,dict) and set(self.index)==set(expected_keys),'HF index complete key inventory differs')
        self.headers={};self.starts={};self.shards={}
        for name in sorted(set(self.index.values())):
            require(isinstance(name,str) and P(name).name==name and name.endswith('.safetensors'),'HF unsafe shard path')
            path=paths['actor/'+name];require(path.parent==self.index_path.parent,'HF shard directory differs');self.shards[name]=path
            with path.open('rb') as f:
                length=f.read(8);require(len(length)==8,'HF header length missing');n=struct.unpack('<Q',length)[0]
                require(0<n<64*1024**2 and n+8<=path.stat().st_size,'HF header invalid');header=unique(f.read(n));self.starts[name]=8+n
            for key,v in header.items():
                if key=='__metadata__':continue
                require(key not in self.headers and self.index.get(key)==name,'HF duplicate or wrongly assigned tensor')
                require(isinstance(v,dict) and set(v)=={'dtype','shape','data_offsets'} and v['dtype']=='F32','HF native master dtype/schema differs')
                shape=v['shape'];offset=v['data_offsets']
                require(isinstance(shape,list) and shape and all(type(x) is int and x>0 for x in shape),'HF shape invalid')
                require(isinstance(offset,list) and len(offset)==2 and all(type(x) is int for x in offset) and 0<=offset[0]<=offset[1] and offset[1]-offset[0]==math.prod(shape)*4 and self.starts[name]+offset[1]<=path.stat().st_size,'HF tensor byte extent differs')
                self.headers[key]=v
        require(set(self.headers)==set(expected_keys),'HF header complete key inventory differs')
    def local(self,key,rank,world=8):
        v=self.headers[key];shape=v['shape'];require(shape[0]%world==0 and type(rank) is int and 0<=rank<world,'HF slice topology differs')
        count=shape[0]//world;row_bytes=math.prod(shape[1:])*4;start=v['data_offsets'][0]+rank*count*row_bytes;size=count*row_bytes
        with self.shards[self.index[key]].open('rb') as f:
            f.seek(self.starts[self.index[key]]+start);raw=f.read(size)
        require(len(raw)==size,'HF local slice truncated')
        return torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape([count]+shape[1:])
    def equal_slice(self,key,rank,local):
        exported=self.local(key,rank)
        require(exported.shape==local.shape and exported.dtype==local.dtype and torch.equal(exported,local),'native model/HF exact slice differs: '+key)
        return True

def base_asset_inventory(directory, expected_names, manifest_path, pins):
    """Known metadata remains hashed identity, never a marker acceptance signal."""
    directory=P(directory);manifest_path=P(manifest_path)
    allowed=set(expected_names)|{'VERIFIED'}
    if manifest_path.parent==directory:
        require(manifest_path.name=='VERIFIED-MODEL-MANIFEST.json' and sha(manifest_path)==MODEL_SHA,'metadata manifest identity differs')
        allowed.add(manifest_path.name)
    files={}
    for path in sorted(directory.iterdir()):
        require(not path.is_symlink(),'base inventory symlink')
        if path.is_file():
            require(path.name in allowed,'unknown base asset or metadata')
            files[path.name]=sha(path);pins[str(path)]=files[path.name]
    require(set(expected_names)<=set(files),'original model asset missing')
    return files

def inspect(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only visibility required')
    inspector_source_sha=sha(P(__file__).resolve())
    require(re.fullmatch('[0-9a-f]{40}',args.expected_source),'external exact source missing')
    repo=P(args.repo);require(repo.is_absolute() and repo.resolve()==repo,'repo alias')
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==args.expected_source and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'actual clean source differs')
    mp=P(args.manifest);require(sha(mp)==args.manifest_sha256,'external checkpoint manifest SHA differs');m=unique(mp.read_bytes());root=P(m['run_root'])
    require(mp.is_relative_to(root) and root.resolve()==root and root.is_absolute(),'manifest run path differs')
    require(m['source_commit']==args.expected_source and m['method']=='pure_opsd' and m['global_step']==m['actual_outer_updates']==5 and m['next_global_step']==6 and m['horizon']==300 and m['warmup_steps']==57 and m['world_size']==8,'actual first5 manifest counters differ')
    pins={str(mp):args.manifest_sha256};paths={name:record(v,root,pins) for name,v in m['files'].items()}
    resolved=unique(paths['resolved_config'].read_bytes());actual_config_digest=hashlib.sha256(json.dumps(resolved,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
    require(actual_config_digest==m['config_sha256'],'actual resolved global config digest differs')
    base_manifest=P(args.base_manifest);require(args.base_manifest_sha256==MODEL_SHA and sha(base_manifest)==MODEL_SHA,'external official manifest differs');pins[str(base_manifest)]=MODEL_SHA
    base=Base(P(args.base_model),unique(base_manifest.read_bytes()),pins);base_digest=base.named_parameter_digest();base_assets=base_asset_inventory(base.directory,{r['file'] for r in unique(base_manifest.read_bytes())['files']},base_manifest,pins)
    helper=repo/'search_r1/baselines/full_state.py';module=types.ModuleType('independent_actual_full_state');module.__file__=str(helper);exec(compile(helper.read_bytes(),str(helper),'exec'),module.__dict__);digest=module.state_digest
    expected_source_files={str(p.relative_to(repo)):sha(p) for namespace in ('search_r1','verl') for p in sorted((repo/namespace).rglob('*.py'))}
    hf=Export(paths,set(base.headers)|{'lm_head.weight'});hf_files={str(path):pins[str(path)] for path in [hf.index_path,*hf.shards.values()]}
    ranks=[];actor_configs=set(); teacher_digests=set()
    for rank in range(8):
        pt=paths[f'actor/training_state/rank_{rank}.pt'];report=unique(paths[f'actor/training_state/rank_{rank}.json'].read_bytes())
        require(report['status']=='EXACT_FULL_STATE_RELOAD_OK' and report['version']==3 and report['model_state_dict_type']=='SHARDED_STATE_DICT' and report['algorithm']=='pure_opsd' and report['rank']==rank and report['world_size']==8 and report['step']==5,'rank report identity/status differs')
        require(report['path']==str(pt) and report['sha256']==report['state_sha256']==pins[str(pt)],'native full-file SHA differs')
        raw=torch.load(pt,map_location='cpu',weights_only=False); require(digest(raw)==report['state_digest'],'raw full payload digest differs')
        require(raw['version']==3 and raw['algorithm']=='pure_opsd' and raw['model_state_dict_type']=='SHARDED_STATE_DICT' and raw['rank']==rank and raw['world_size']==8 and raw['step']==5 and raw['pid']==report['pid'] and raw['object_ids']==report['object_ids'],'raw native same-object identity differs')
        require(set(raw['object_ids'])=={'model','optimizer','scheduler','teacher'} and all(type(v) is int and v>0 for v in raw['object_ids'].values()),'saved object IDs missing')
        identity=raw['recovery_identity']; require(identity['worker_source_sha256']==sha(repo/'verl/workers/fsdp_workers.py') and identity['source_closure']==dict(version=1,git_source_commit=args.expected_source,files=expected_source_files),'raw complete source inventory differs')
        require(re.fullmatch('[0-9a-f]{64}',identity['actor_config_digest']),'actor config digest invalid');actor_configs.add(identity['actor_config_digest'])
        model=raw['model'];require(set(model)==(set(base.headers)|{'lm_head.weight'}),'full model tensor inventory differs')
        layouts={};changes={};unique_local_numel=0
        for key,t in model.items():
            layouts[key]=layout(t,rank);target=base.local(key,rank);local=t.to_local();require(local.shape==target.shape,'base/local tensor shape mismatch')
            require(tuple(t.shape)==tuple(base.headers['model.embed_tokens.weight' if key=='lm_head.weight' else key]['shape']),'global tensor shape mismatch')
            require(hf.headers[key]['shape']==list(t.shape),'HF/global native tensor shape differs')
            hf.equal_slice(key,rank,local)
            diff=int(torch.count_nonzero(local!=target));changes[key]=diff
            if key!='lm_head.weight':unique_local_numel+=local.numel()
        require(torch.equal(model['lm_head.weight'].to_local(),model['model.embed_tokens.weight'].to_local()),'tied weights no longer equal')
        require(unique_local_numel==NUMEL//8 and sum(v for k,v in changes.items() if k!='lm_head.weight')>0,'student has no actual same-dtype parameter change')
        opt=adam(raw['optimizer'],NUMEL//8);sched=scheduler(raw['scheduler']);rng_report=rng(raw['rng'],True)
        require(report['scheduler_last_epoch']==5 and report['optimizer_state_entries']==len(opt['entries']) and report['model_dtensor_entries']==len(model),'rank report component count differs')
        teacher=raw['teacher'];require(report['teacher']=={k:v for k,v in teacher.items() if k!='buffers'},'teacher raw/report differs')
        require(teacher['version']==1 and teacher['parameter_count']==NUMEL and teacher['storage_device']=='cpu' and re.fullmatch(r'cuda(?::[0-9]+)?',teacher['execution_device']) and teacher['model_content_digest']==base_digest,'teacher fixed actual content/role differs')
        require(P(teacher['asset_directory'])==base.directory and teacher['asset_files']==base_assets,'teacher full asset identity differs')
        require(set(teacher['counters'])==set(COUNTERS) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in teacher['counters'].values()) and teacher['counters']['cuda_update_entries']==5 and teacher['counters']['scored_tokens']>0,'teacher actual counters differ')
        require(all(teacher['model_config'][key]==base.config[key] for key in ('model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','tie_word_embeddings')),'teacher actual architecture differs')
        require(isinstance(teacher['caches'],dict) and all(isinstance(v,dict) and all(type(n) is int and n>=0 for n in v.values()) for v in teacher['caches'].values()),'invalid teacher history cache metadata')
        for t in teacher['buffers'].values():finite(t)
        require(digest(teacher['buffers'])==teacher['buffer_content_digest'],'teacher buffer raw bytes differ')
        require(teacher['model_bytes']==NUMEL*4+sum(t.numel()*t.element_size() for t in teacher['buffers'].values()),'teacher actual model+buffer byte count differs')
        teacher_digests.add(teacher['model_content_digest'])
        ranks.append(dict(rank=rank,pid=raw['pid'],object_ids=raw['object_ids'],state_sha256=pins[str(pt)],state_digest=report['state_digest'],actor_config_digest=identity['actor_config_digest'],component_digests={k:digest(raw[k]) for k in ('model','optimizer','scheduler','rng','teacher')},model_dtensor_entries=len(model),model_hf_slice_exact=True,model_hf_slice_keys=sorted(model),model_layouts=layouts,unique_local_model_numel=unique_local_numel,changed_elements_vs_original_fp32=changes,optimizer_state_entries=len(opt['entries']),optimizer_step_values=opt['steps'],optimizer_moments=opt['entries'],scheduler_last_epoch=5,scheduler_last_lr=sched['last_lr'],rng_verified=True,rng_content=rng_report,teacher_content_digest=base_digest,teacher_cuda_update_entries=5,teacher_counters=teacher['counters'],teacher_buffer_digest=teacher['buffer_content_digest']))
        del raw,model
    require(len({r['pid'] for r in ranks})==8 and len(actor_configs)==len(teacher_digests)==1,'rank processes/configs/teacher alias or drift')
    driver_path=paths['driver/driver.pt'];dr=unique(paths['driver/driver.json'].read_bytes());driver=torch.load(driver_path,map_location='cpu',weights_only=False)
    require(dr['status']=='EXACT_DRIVER_STATE_RELOAD_OK' and dr['pid']==m['driver_pid'] and dr['step']==5 and dr['state_sha256']==pins[str(driver_path)] and dr['digest']==digest(driver),'driver actual digest/report differs')
    require(driver['method']=='pure_opsd' and driver['global_step']==driver['actual_outer_updates']==5 and driver['next_global_step']==6 and driver['pid']==m['driver_pid'] and driver['uid']==m['uid'] and str(driver['job_id'])==str(m['job_id']) and driver['source_commit']==args.expected_source and driver['config_sha256']==m['config_sha256'],'driver payload identity differs')
    sampler=driver['sampler'];require(sampler['size']==169028 and sampler['cursor']==sampler['consumed_cursor']==2560 and sampler['epoch']==0 and len(sampler['order'])==169028 and set(sampler['order'])==set(range(169028)),'driver complete sampler differs')
    torch.Generator().set_state(sampler['generator']);driver_rng=rng(driver['rng'],False)
    require(driver['algorithm']==unique(paths['algorithm'].read_bytes()) and driver['algorithm']['method']=='pure_opsd' and driver['algorithm']['global_step']==5,'driver/controller saved state differs')
    require(not torch.cuda.is_initialized(),'CUDA initialized during inspection')
    for path,pin in pins.items():require(sha(P(path))==pin,'input drifted during audit')
    require(expected_source_files=={str(p.relative_to(repo)):sha(p) for namespace in ('search_r1','verl') for p in sorted((repo/namespace).rglob('*.py'))},'source drifted during audit')
    require(sha(P(__file__).resolve())==inspector_source_sha and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==args.expected_source and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'inspector/repo drifted during audit')
    return dict(version=1,scope='NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE',inspector_source_sha256=inspector_source_sha,cpu_runtime=dict(python_executable=sys.executable,python_version=sys.version,torch_version=torch.__version__,torch_file=str(P(torch.__file__).resolve()),torch_file_sha256=sha(P(torch.__file__).resolve()),numpy_file=str(P(np.__file__).resolve()),numpy_file_sha256=sha(P(np.__file__).resolve())),source_commit=args.expected_source,job_id=str(m['job_id']),run_root=str(root),checkpoint_manifest_sha256=args.manifest_sha256,driver=dict(pid=driver['pid'],state_sha256=pins[str(driver_path)],state_digest=dr['digest'],sampler_cursor=2560,consumed_cursor=2560,size=169028,epoch=0,sampler_digest=digest(sampler),rng_verified=True,rng_content=driver_rng),ranks=ranks,base_named_parameter_digest=base_digest,hf_all_rank_slices_exact=True,hf_files=hf_files,all_files=pins,cuda_is_initialized=False,passed=True,limitations=['Stored content inspection is not re-execution of native same-object restore or approval','No live worker PID/UUID/resource/lease witness is produced','Native Adam ID/group/local-flat shapes are checked; named-to-flat FSDP handle reconstruction is not performed','Teacher buffers are actual saved history-dependent bytes, not forced equal to initial RoPE caches','No signatures or training continuation; actual runtime reports/source/first5 logs must be independently joined'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--expected-source',required=True);p.add_argument('--repo',required=True);p.add_argument('--base-model',required=True);p.add_argument('--base-manifest',required=True);p.add_argument('--base-manifest-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=inspect(a);output=P(a.output);require(output.is_absolute() and output.parent.resolve()==output.parent and not output.is_symlink(),'nonphysical output')
    with output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(scope=result['scope'],actual_job_id=result['job_id'],passed=result['passed'],output=str(output),output_sha256=sha(output),cuda_is_initialized=False)))
if __name__=='__main__': main()
