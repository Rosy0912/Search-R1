"""Actual worker resources at a synchronous calibration boundary, no reward math."""
import hashlib,json,os,pathlib,socket,time,subprocess

def identity(worker):
 return {'pid':os.getpid(),'ppid':os.getppid(),'host':socket.gethostname(),'job':os.environ.get('SLURM_JOB_ID'),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'rank':worker.rank,'role':worker.role,'starttime_ticks':int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}

def modules(worker):
 values=[]
 for name in ('actor_module_fsdp','ref_module_fsdp'):
  if hasattr(worker,name):values.append((name,getattr(worker,name)))
 if getattr(worker,'_is_rollout',False):
  engine=worker.rollout.inference_engine
  values.append(('vllm_model',engine.llm_engine.model_executor.worker.model_runner.model))
 return values

def quiesce(worker,torch,offload_model,offload_optimizer):
 if getattr(worker,'_g1_buffer_restore',None) is not None:raise RuntimeError('worker already calibration quiesced')
 torch.cuda.synchronize();saved=[]
 # Completed outer update/save, and the original next update zeroes grads anyway.
 # Release unused completed gradients before staging a new 32B calibrator.
 if hasattr(worker,'actor_optimizer'):worker.actor_optimizer.zero_grad(set_to_none=True)
 if getattr(worker,'_is_rollout',False):
  engine=worker.rollout.inference_engine;engine.free_cache_engine();engine.offload_model_weights()
 for label,module in modules(worker):
  if label!='vllm_model':offload_model(module,offload_grad=True)
  for name,value in module.named_buffers(remove_duplicate=False):
   if value is not None and value.device.type=='cuda':
    owner_name,_,leaf=name.rpartition('.');owner=module.get_submodule(owner_name) if owner_name else module
    saved.append((label,owner,leaf,value.device));owner._buffers[leaf]=value.to('cpu')
 if hasattr(worker,'actor_optimizer'):offload_optimizer(worker.actor_optimizer)
 torch.cuda.synchronize();worker._g1_buffer_restore=saved
 return dict(identity(worker),status='OFFLOADED_PENDING_ALL_ROLES_ALLOCATOR_CHECK',epoch=time.time(),buffers_moved=len(saved))

def actual_uuid(torch):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
 inventory={index.strip():uuid.strip() for index,uuid in (line.split(',') for line in rows.splitlines())}
 visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[torch.cuda.current_device()].strip()
 expected=visible if visible.startswith('GPU-') else inventory[visible]
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
 actual=sorted({uuid.strip() for pid,uuid in (line.split(',') for line in raw.splitlines()) if int(pid.strip())==os.getpid()})
 if actual!=[expected]:raise RuntimeError('own worker actual NVML PID/UUID identity unavailable or changed')
 return expected,actual

def release_receipt(worker,torch):
 if getattr(worker,'_g1_buffer_restore',None) is None:raise RuntimeError('missing actual offload call')
 torch.cuda.synchronize()
 if not hasattr(torch._C,'_cuda_clearCublasWorkspaces'):raise RuntimeError('fixed own-worker clear API unavailable')
 torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize()
 parameter_devices={label:sorted({str(p.device) for p in module.parameters()}) for label,module in modules(worker)}
 buffer_devices={label:sorted({str(b.device) for b in module.buffers()}) for label,module in modules(worker)}
 allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
 uuid,actual=actual_uuid(torch)
 record=dict(identity(worker),status='RELEASED_ZERO' if allocated==reserved==0 else 'FAILED_NONZERO_ALLOCATOR',allocated=allocated,reserved=reserved,parameter_devices=parameter_devices,buffer_devices=buffer_devices,epoch=time.time(),gpu_uuid=uuid,process_gpu_uuids=actual)
 if allocated or reserved or any(any(not x.startswith('cpu') for x in devices) for devices in list(parameter_devices.values())+list(buffer_devices.values())):raise RuntimeError('actual calibration actor/reference/vllm release not zero: '+json.dumps(record))
 return record

def resume(worker,torch):
 saved=getattr(worker,'_g1_buffer_restore',None)
 if saved is None:raise RuntimeError('missing calibration release before resume')
 for label,owner,leaf,device in saved:owner._buffers[leaf]=owner._buffers[leaf].to(device)
 torch.cuda.synchronize();worker._g1_buffer_restore=None
 # Original FSDP/vLLM context loads/syncs weights and creates KV on next real call.
 return dict(identity(worker),status='BUFFERS_RESTORED_ORIGINAL_CALLER_LOADS_WEIGHTS',epoch=time.time())
