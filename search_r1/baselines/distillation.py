"""Full-vocabulary OPSD loss and explicit teacher/student target alignment."""
import math
import resource
import sys
import time
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F


def generalized_jsd(student_logits, teacher_logits, beta=0., temperature=1., component_clip=.05):
    if not 0 <= beta <= 1 or temperature <= 0:
        raise ValueError('Invalid JSD beta/temperature')
    s = F.log_softmax(student_logits.float() / temperature, -1)
    t = F.log_softmax(teacher_logits.detach().float() / temperature, -1)
    if beta == 0:
        parts = F.kl_div(s, t, log_target=True, reduction='none')
    elif beta == 1:
        parts = F.kl_div(t, s, log_target=True, reduction='none')
    else:
        m = torch.logaddexp(t + math.log(beta), s + math.log1p(-beta))
        parts = beta * F.kl_div(m, t, log_target=True, reduction='none') + (1 - beta) * F.kl_div(m, s, log_target=True, reduction='none')
    if component_clip is not None:
        if component_clip <= 0:
            raise ValueError('component_clip must be positive or null')
        parts = parts.clamp(max=component_clip)
    return parts.sum(-1)


def teacher_inputs(tokenizer, prompts, responses, response_attention, targets=None, max_length=None):
    """Append private training-answer hint to prompt, then identical student response.

    Return absolute prediction positions for every response target. Observations
    remain in response and attention, but the caller excludes them from loss.
    """
    from .rollout import padded
    pad = tokenizer.pad_token_id
    prefixes, rows = [], []
    for i, prompt in enumerate(prompts):
        prefix = prompt[prompt != pad].tolist()
        if targets is not None:
            answers = targets[i]
            if isinstance(answers, str):
                answers = [answers]
            hint = '\n[Teacher-only training context: accepted answers: ' + repr(list(answers)) + ']\n'
            prefix += tokenizer.encode(hint, add_special_tokens=False)
        prefixes.append(len(prefix))
        response = responses[i][response_attention[i].bool()].tolist()
        row = prefix + response
        if max_length is not None and len(row) > max_length:
            raise ValueError('Teacher context exceeds model limit; no silent truncation')
        rows.append(row)
    ids = padded(rows, pad)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        attention[i, :len(row)] = 1
    positions = torch.tensor(prefixes)[:, None] + response_attention.long().cpu().cumsum(-1) - 2
    return ids, attention, positions


def opsd_loss(student_model, teacher, tensors, tokenizer, targets, settings, denominator):
    """Identical CPU/GPU loss path; full logits retain gradients only for student."""
    width = tensors['responses'].shape[1]
    tid, tmask, positions = teacher_inputs(tokenizer, tensors['prompts'], tensors['responses'],
                                           tensors['attention_mask'][:, -width:], targets,
                                           teacher.model.config.max_position_embeddings)
    with teacher.measure(), torch.no_grad():
        hidden = teacher.model.base_model(input_ids=tid.to(teacher.device),
                                          attention_mask=tmask.to(teacher.device),
                                          use_cache=False).last_hidden_state
    teacher.scored_tokens += int(tmask.sum())
    device = tensors['input_ids'].device
    # Chunking loss does not bound the model's full logits allocation.
    if hasattr(settings, 'max_student_logits_gib'):
        bytes_per_element = 2 if device.type == 'cuda' else 4
        allocation = tensors['input_ids'].numel() * student_model.config.vocab_size * bytes_per_element
        if allocation > settings.max_student_logits_gib * 1024 ** 3:
            raise ValueError('OPSD full-logits estimate exceeds explicit budget; reduce microbatch/length')
    context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with context:
        logits = student_model(input_ids=tensors['input_ids'], attention_mask=tensors['attention_mask'],
                               position_ids=tensors['position_ids'], use_cache=False).logits
        student = logits[:, -width - 1:-1]
    total = torch.where(torch.isfinite(student), student, torch.zeros_like(student)).sum() * 0
    for start in range(0, width, settings.chunk_size):
        stop = min(start + settings.chunk_size, width)
        pos = positions[:, start:stop].clamp(0, hidden.shape[1] - 1).to(teacher.device)
        features = hidden.gather(1, pos[..., None].expand(-1, -1, hidden.shape[-1]))
        with teacher.measure(), torch.no_grad():
            target_logits = teacher.model.get_output_embeddings()(features)
        # CUDA copies may wait for the student stream; exclude that from CPU teacher timing.
        target_logits = target_logits.to(device)
        valid = tensors['loss_mask'][:, start:stop].bool()
        source_logits = torch.where(valid[..., None], student[:, start:stop], 0.)
        target_logits = torch.where(valid[..., None], target_logits, 0.)
        if not torch.isfinite(source_logits).all() or not torch.isfinite(target_logits).all():
            raise ValueError('Nonfinite OPSD logits on supervised tokens')
        divergence = generalized_jsd(source_logits, target_logits,
                                     settings.beta, settings.temperature, settings.component_clip)
        total = total + (divergence * tensors['loss_mask'][:, start:stop]).sum()
    return total / denominator


def global_token_count(mask):
    """Raw collective count; every rank makes the same empty-minibatch decision."""
    total = mask.sum().float()
    world = 1
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        world = torch.distributed.get_world_size()
    return total, world


def global_token_denominator(mask):
    """Account for averaged gradients; an empty count is handled by the actor."""
    total, world = global_token_count(mask)
    return total.clamp_min(1) / world


class FrozenTeacher:
    """Unsharded frozen FP32/eager teacher on an explicitly selected device.

    CUDA placement costs one complete teacher per actor rank; production must
    verify memory coexistence with actor, rollout and local GPU retrieval.
    """
    def __init__(self, path, student_tokenizer, device='cpu'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        self.cpu_seconds = self.wall_seconds = 0.
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.get_vocab() != student_tokenizer.get_vocab():
            raise ValueError('Teacher/student token-to-id vocabularies differ')
        for attr in ('bos_token_id', 'eos_token_id'):
            if getattr(self.tokenizer, attr) != getattr(student_tokenizer, attr):
                raise ValueError('Teacher/student special tokens differ')
        self.tokenizer.pad_token_id = student_tokenizer.pad_token_id
        self.execution_device = torch.device(device)
        if self.execution_device.type not in ('cpu', 'cuda'):
            raise ValueError('Frozen teacher device must be cpu or cuda')
        # Keep the complete frozen teacher off GPU during vLLM rollouts.
        self.device = torch.device('cpu')
        self.transfer_wall_seconds = 0.
        self.cuda_update_entries = 0
        self.cuda_seconds = 0.
        self._cuda_events = []
        if self.execution_device.type == 'cuda':
            # Preserve FP32 teacher arithmetic even in a BF16 student worker.
            # These process settings also prevent accidental later TF32 use.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                        attn_implementation='eager').to(self.device)
        self.model.eval().requires_grad_(False)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.model_bytes = sum(t.numel() * t.element_size()
                               for t in list(self.model.parameters()) + list(self.model.buffers()))
        self.scored_tokens = 0
        self.load_wall_seconds = time.perf_counter() - wall_start
        self.load_cpu_seconds = time.process_time() - cpu_start

    def _validate_execution(self, device):
        """Inspect actual parameters/configuration, rather than placement intent."""
        expected = torch.device(device)
        if expected.type == 'cuda' and expected.index is None:
            expected = torch.device('cuda', torch.cuda.current_device())
        parameters = list(self.model.parameters())
        if not parameters or any(p.dtype != torch.float32 for p in parameters):
            raise RuntimeError('Frozen teacher parameters must remain FP32')
        if getattr(self.model.config, '_attn_implementation', None) != 'eager':
            raise RuntimeError('Frozen teacher attention must remain eager')
        if any(p.device != expected for p in parameters) or any(b.device != expected for b in self.model.buffers()):
            raise RuntimeError('Frozen teacher actual device differs from execution/storage device')
        if expected.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise RuntimeError('Frozen FP32 CUDA teacher requires TF32 disabled')

    @contextmanager
    def on_device(self):
        """Move once per update and restore CPU storage even after an exception."""
        if self.execution_device.type == 'cpu':
            self._validate_execution('cpu')
            yield
            return
        if self.device.type != 'cpu':
            raise RuntimeError('Frozen teacher update device context cannot be nested')
        wall_start = time.perf_counter()
        try:
            self.model.to(self.execution_device)
            torch.cuda.synchronize(self.execution_device)
            self.device = self.execution_device
            self._validate_execution(self.device)
            self.cuda_update_entries += 1
            self.transfer_wall_seconds += time.perf_counter() - wall_start
            yield
        finally:
            try:
                self._flush_cuda_events()
            finally:
                # Timing failures must not bypass an attempt to release teacher
                # GPU parameters. Preserve the original error when cleanup works.
                wall_start = time.perf_counter()
                try:
                    self.model.to('cpu')
                    self.device = torch.device('cpu')
                    torch.cuda.synchronize(self.execution_device)
                    self._validate_execution('cpu')
                finally:
                    self.transfer_wall_seconds += time.perf_counter() - wall_start

    @contextmanager
    def measure(self):
        """Measure teacher intervals; process CPU time includes every worker CPU thread.

        These are interval costs, not a profiler's exclusive CPU attribution.
        Host wall time measures enqueue time for CUDA, not GPU execution.
        CUDA events measure elapsed stream intervals and exclude later transfers;
        they are not exclusive kernel attribution when streams overlap.
        """
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        events = None
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record(torch.cuda.current_stream(self.device))
        try:
            # Teacher base model and projection stay FP32 inside caller autocast.
            with torch.autocast(device_type=self.device.type, enabled=False):
                yield
        finally:
            if events is not None:
                events[1].record(torch.cuda.current_stream(self.device))
                self._cuda_events.append(events)
            self.wall_seconds += time.perf_counter() - wall_start
            self.cpu_seconds += time.process_time() - cpu_start
            if len(self._cuda_events) >= 1024:
                self._flush_cuda_events()

    def _flush_cuda_events(self):
        # Bounded event storage; synchronize only recorded teacher intervals.
        for start, end in self._cuda_events:
            end.synchronize()
            self.cuda_seconds += start.elapsed_time(end) / 1000.
        self._cuda_events.clear()

    def cost_metrics(self):
        self._flush_cuda_events()
        # ru_maxrss includes student/optimizer/worker overhead, not isolated teacher RAM.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        parameters = list(self.model.parameters())
        return {'teacher/cpu_seconds_cumulative_per_rank': self.cpu_seconds,
                'teacher/wall_seconds_cumulative_per_rank': self.wall_seconds,
                'teacher/host_wall_seconds_cumulative_per_rank': self.wall_seconds,
                'teacher/cuda_stream_seconds_cumulative_per_rank': self.cuda_seconds,
                'teacher/update_device_is_cuda_per_rank': int(self.execution_device.type == 'cuda'),
                'teacher/storage_is_cpu_per_rank': int(bool(parameters) and all(p.device.type == 'cpu' for p in parameters)
                                                       and all(b.device.type == 'cpu' for b in self.model.buffers())),
                'teacher/all_parameter_dtype_is_fp32_per_rank': int(bool(parameters) and all(p.dtype == torch.float32 for p in parameters)),
                'teacher/attention_is_eager_per_rank': int(getattr(self.model.config, '_attn_implementation', None) == 'eager'),
                'teacher/cuda_update_entries_cumulative_per_rank': self.cuda_update_entries,
                'teacher/transfer_wall_seconds_cumulative_per_rank': self.transfer_wall_seconds,
                'teacher/load_cpu_seconds_per_rank': self.load_cpu_seconds,
                'teacher/load_wall_seconds_per_rank': self.load_wall_seconds,
                'teacher/model_bytes_per_rank': self.model_bytes,
                'teacher_worker/peak_rss_bytes_per_rank': peak if sys.platform == 'darwin' else peak * 1024}

    def audit_metrics(self, student_optimizer):
        self._validate_execution(self.device)
        parameters = list(self.model.parameters())
        teacher_ids = {id(p) for p in parameters}
        trainable = sum(p.numel() for p in parameters if p.requires_grad)
        gradients = sum(p.grad is not None for p in parameters)
        overlap = sum(id(p) in teacher_ids for group in student_optimizer.param_groups for p in group['params'])
        if trainable or gradients or overlap:
            raise RuntimeError('Frozen teacher acquired gradients or entered the student optimizer')
        return {'teacher/trainable_parameters_per_rank': trainable,
                'teacher/gradient_tensors_per_rank': gradients,
                'teacher/student_optimizer_membership_per_rank': overlap}

    @torch.no_grad()
    def log_probs(self, prompts, responses, attention, temperature=1., targets=None, chunk=16):
        ids, mask, positions = teacher_inputs(self.tokenizer, prompts, responses, attention,
                                              targets, self.model.config.max_position_embeddings)
        ids, mask = ids.to(self.device), mask.to(self.device)
        self.scored_tokens += int(mask.sum())
        with self.measure():
            hidden = self.model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        result = torch.zeros_like(responses, dtype=torch.float32)
        for start in range(0, responses.shape[1], chunk):
            stop = min(start + chunk, responses.shape[1])
            pos = positions[:, start:stop].clamp(0, hidden.shape[1] - 1).to(self.device)
            features = hidden.gather(1, pos[..., None].expand(-1, -1, hidden.shape[-1]))
            with self.measure():
                logits = self.model.get_output_embeddings()(features).float() / temperature
            labels = responses[:, start:stop].to(self.device)
            logp = F.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
            result[:, start:stop] = logp.cpu() * attention[:, start:stop].cpu()
        return result
