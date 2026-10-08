"""Isolated stage-once FP32 pipeline candidate; no training or deployment entry."""
import hashlib
import inspect
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import capacity_reference as reference

SCOPE = 'FP32_PIPELINE_CANDIDATE_UNVALIDATED_NO_TRAINING'
SOURCE_SHA = 'fede3c23e882139a12a2b50c531177438012959774958fc39e03fb3f9d29e229'
require = reference.require


def snapshot(model):
    """Same ordered raw FP32 identity as reference, bounded8MiB host chunks."""
    hashes = []
    for tensors in (model.named_parameters(), model.named_buffers()):
        digest = hashlib.sha256()
        for name, tensor in tensors:
            require(tensor.dtype == torch.float32, 'Non-FP32 frozen tensor: ' + name)
            digest.update(name.encode() + b'\0')
            digest.update(str(tuple(tensor.shape)).encode() + b'\0')
            # Contiguous model tensors are required; no whole-device CPU copy.
            require(tensor.is_contiguous(), 'Noncontiguous frozen tensor: ' + name)
            flat = tensor.detach().view(-1)
            for start in range(0, flat.numel(), 2097152):
                digest.update(flat[start:start + 2097152].cpu().numpy().tobytes())
        hashes.append(digest.hexdigest())
    return dict(parameter_sha256=hashes[0], buffer_sha256=hashes[1],
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                buffer_devices=sorted({str(b.device) for b in model.buffers()}),
                training_modules=sum(int(m.training) for m in model.modules()),
                requires_grad_parameters=sum(int(p.requires_grad) for p in model.parameters()),
                grad_parameters=sum(int(p.grad is not None) for p in model.parameters()))


class PipelineRuntime(reference.NativeCudaRuntime):
    def synchronize_device(self, device):
        torch.cuda.synchronize(torch.device(device))

    def thread_context(self):
        self.flags()
        streams=[dict(device=i,stream=int(torch.cuda.current_stream(i).cuda_stream)) for i in range(8)]
        require(all(item['stream']==0 for item in streams), 'Nondefault thread stream unreviewed')
        return dict(thread=threading.get_ident(), current_device=torch.cuda.current_device(), actual_flags=dict(
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            autocast_enabled=torch.is_autocast_enabled()),
            streams=streams)


class DevicePermits:
    """Wrap module.forward outside existing dispatch hooks, sync before release."""
    def __init__(self, model, mapping, runtime):
        self.runtime = runtime
        self.saved = []
        self.locks = {device: threading.Lock() for device in set(mapping.values())}
        keys = list(mapping)
        require(not any(a != b and b.startswith(a + '.') for a in keys for b in keys),
                'Nested permit modules would deadlock')
        try:
            for name, device in mapping.items():
                module = model.get_submodule(name)
                original = module.forward
                self.saved.append((module, original))
                lock = self.locks[device]
                def forward(*args, _original=original, _lock=lock, _device=device, **kwargs):
                    with _lock:
                        try:
                            return _original(*args, **kwargs)
                        finally:
                            runtime.synchronize_device(_device)
                module.forward = forward
        except BaseException:
            self.close()
            raise

    def close(self):
        for module, original in reversed(self.saved):
            module.forward = original
        self.saved.clear()


def validate_dispatch_hooks(model):
    # Offload/tied migration and dynamic RoPE are mutable, cannot run concurrently.
    require(not getattr(model.config, 'rope_scaling', None), 'Dynamic/scaled RoPE unreviewed')
    for module in model.modules():
        hook = getattr(module, '_hf_hook', None)
        if hook is not None:
            stack = list(getattr(hook, 'hooks', [hook]))
            for item in stack:
                require(not getattr(item, 'offload', False), 'Concurrent offload hook forbidden')
                require(not getattr(item, 'place_submodules', False), 'Mutable submodule migration hook forbidden')


class PipelineScorer(reference.SharedScorer):
    def __init__(self, teacher, runtime, mapping, allowed_ids, expected_parameters=reference.PARAMETERS_32B):
        from search_r1.baselines.distillation import FrozenTeacher
        require(type(teacher) is FrozenTeacher and reference.method_sha(FrozenTeacher.log_probs)==reference.ORIGINAL_METHOD_SHA,
                'Original FrozenTeacher method differs')
        self.original=FrozenTeacher.log_probs;self.teacher=teacher;self.runtime=runtime;self.mapping=mapping
        self.allowed_ids=set(allowed_ids);self.lock=threading.Lock();self.state='CPU_READY';self.rounds=0
        self.initial=snapshot(teacher.model);self.inventory=reference.parameter_map(teacher.model,mapping)
        require(self.initial['parameter_count']==expected_parameters and self.initial['parameter_devices']==['cpu']
                and all(d=='cpu' for d in self.initial['buffer_devices']), 'Wrong teacher/CPU state')
        require(self.initial['training_modules']==self.initial['requires_grad_parameters']==self.initial['grad_parameters']==0,
                'Teacher not frozen eval')
        require(teacher.model.config._attn_implementation=='eager','Eager attention required')
        original = inspect.unwrap(self.original)
        self.original_file = Path(inspect.getfile(original)).resolve()
        require(hashlib.sha256(self.original_file.read_bytes()).hexdigest() == SOURCE_SHA,
                'Original distillation source file differs')
        self.last_record = None

    def score_round(self, tensors, workers=8, join_timeout=1800):
        require(workers in (1,2,4,8), 'Worker count must be1/2/4/8')
        require(self.lock.acquire(False), 'Concurrent outer phase forbidden')
        record = dict(scope=SCOPE, status='FAILED', workers=workers,
                      original_method_sha256=reference.ORIGINAL_METHOD_SHA,
                      original_file=str(self.original_file), original_file_sha256=SOURCE_SHA)
        permits = None; threads = []; output = None
        errors = []; stop = threading.Event(); warmup_lock = threading.Lock()
        self.last_record = record
        try:
            require(self.state in ('CPU_READY','RELEASED'), 'CPU release boundary absent')
            p,r,a = self.validate(tensors)
            from safetensors.torch import save
            record['input_sha256']=hashlib.sha256(save(tensors)).hexdigest()
            record['initial_cpu']=self.initial
            self.runtime.verify(); record['actual_flags'] = self.runtime.flags()
            self.state = 'STAGING'; record['started_epoch'] = time.time()
            self.runtime.reset_peaks()
            self.teacher.model = self.runtime.dispatch(self.teacher.model, self.mapping)
            self.teacher.device = torch.device(self.runtime.device)
            self.runtime.synchronize(); self.runtime.verify_partition(self.teacher.model,self.inventory)
            staged = snapshot(self.teacher.model); record['staged_frozen'] = staged
            require(reference.frozen_equal(staged,self.initial) and staged['training_modules']==staged['requires_grad_parameters']==staged['grad_parameters']==0, 'Staging changed frozen state')
            validate_dispatch_hooks(self.teacher.model)
            record['gpu_pid_staged'] = self.runtime.gpu_identity()
            permits = DevicePermits(self.teacher.model,self.mapping,self.runtime)
            self.state = 'SCORING'
            results = [None] * len(r); views = []; records = []; completion_order=[]; guard = threading.Lock()
            warmed = threading.Barrier(workers)
            tasks = queue.Queue()
            for i in range(len(r)): tasks.put(i)
            def worker():
                view = SimpleNamespace(model=self.teacher.model,tokenizer=self.teacher.tokenizer,
                                       device=self.teacher.device,scored_tokens=0)
                local = dict(thread=threading.get_ident(), rows=[], warmup_tokens=0)
                try:
                    # Dedicated persistent thread warms its handles serially before admission.
                    with warmup_lock:
                        if stop.is_set(): return
                        local['warmup_started_epoch']=time.time()
                        local['context'] = self.runtime.thread_context()
                        self.original(view,p[:1],r[:1],a[:1],temperature=1.,targets=None,chunk=16)
                        self.runtime.synchronize()
                        local['warmup_completed_epoch']=time.time()
                        local['warmup_tokens'] = view.scored_tokens; view.scored_tokens = 0
                    warmed.wait()
                    while not stop.is_set():
                        try: index = tasks.get_nowait()
                        except queue.Empty: break
                        local.setdefault('first_delivered_row_started_epoch',time.time())
                        value = self.original(view,p[index:index+1],r[index:index+1],a[index:index+1],
                                              temperature=1.,targets=None,chunk=16)
                        require(value.device.type=='cpu', 'GPU output retained')
                        results[index] = value; local['rows'].append(index)
                        with guard: completion_order.append(index)
                except BaseException as exc:
                    # Keep strings, never exception/traceback retaining CUDA temporaries.
                    with guard: errors.append(repr(exc))
                    stop.set(); warmed.abort()
                finally:
                    local['scored_tokens'] = view.scored_tokens
                    with guard: views.append(view); records.append(local)
            threads = [threading.Thread(target=worker,name='teacher-pipeline-'+str(i)) for i in range(workers)]
            for thread in threads: thread.start()
            deadline = time.monotonic()+join_timeout
            for thread in threads: thread.join(max(0,deadline-time.monotonic()))
            require(not any(t.is_alive() for t in threads), 'Live worker: supervisor must terminate own scorer; no offload')
            record['threads_joined_epoch']=time.time();record['threads']=records
            record['completion_order']=completion_order
            require(not errors, 'Row scoring failed: '+repr(errors))
            require(all(value is not None for value in results), 'Incomplete rows')
            output = torch.cat(results)
            expected = int((p != self.teacher.tokenizer.pad_token_id).sum()+a.sum())
            require(sum(v.scored_tokens for v in views)==expected, 'Delivered token counter mismatch')
            require(output.shape==r.shape and output.dtype==torch.float32 and bool(torch.isfinite(output).all())
                    and bool((output[a==0]==0).all()), 'Finite/masked output mismatch')
            record.update(rows=len(r), scored_tokens=expected, warmup_tokens=sum(x['warmup_tokens'] for x in records),
                          allocator_peaks=self.runtime.peaks(), row_order_verified=True)
            record['output_sha256']=hashlib.sha256(save(dict(teacher_log_probs=output,row_indices=tensors['row_indices']))).hexdigest()
            self.teacher.scored_tokens += expected
        except BaseException as exc:
            record['error'] = repr(exc)
        finally:
            if any(t.is_alive() for t in threads):
                # Fail closed, do not remove hooks/migrate while workers still execute.
                self.state='FAILED';record['release_error']='LIVE_THREADS_NO_OFFLOAD_SUPERVISOR_KILL_REQUIRED'
            elif self.state in ('STAGING','SCORING'):
                try:
                    record['all_threads_joined']=True
                    self.runtime.synchronize()
                    if permits is not None: permits.close()
                    self.runtime.unhook(self.teacher.model)
                    self.teacher.model.to('cpu'); self.teacher.device=torch.device('cpu')
                    self.runtime.synchronize(); released=snapshot(self.teacher.model)
                    require(reference.frozen_equal(released,self.initial) and released['parameter_devices']==['cpu']
                            and all(d=='cpu' for d in released['buffer_devices'])
                            and released['training_modules']==released['requires_grad_parameters']==released['grad_parameters']==0,
                            'Frozen CPU release differs')
                    record['released_cpu']=released
                    record['allocators_released']=self.runtime.release_allocators()
                    require(len(record['allocators_released'])==8 and all(x['allocated_bytes']==x['reserved_bytes']==0
                            for x in record['allocators_released']), 'Allocator release nonzero')
                    record['released_epoch']=time.time(); record['whole_phase_wall_seconds']=record['released_epoch']-record['started_epoch']; self.state='RELEASED'
                except BaseException as exc:
                    record['release_error']=repr(exc);self.state='FAILED'
            self.lock.release()
        if 'error' in record or 'release_error' in record: raise RuntimeError(record)
        require(self.state=='RELEASED', 'Result cannot publish before release')
        record['status']='RELEASED';self.rounds+=1
        return output,record
