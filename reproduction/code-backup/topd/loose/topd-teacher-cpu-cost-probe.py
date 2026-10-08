#!/usr/bin/env python3
"""Hardware-only BASE surrogate through unchanged FrozenTeacher.log_probs.

No controller, rollout, optimizer, Ray, retrieval or training. CPU-only;
measures one trajectory at a time exactly as current TOP-D scoring does.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    status = Path('/proc/self/status').read_text().splitlines()
    return dict(wall=time.perf_counter(), user=value.ru_utime, system=value.ru_stime,
                peak_rss_kib_cumulative=value.ru_maxrss,
                proc_memory=[line for line in status if line.startswith(('VmRSS:', 'VmHWM:'))])


def elapsed(before, after):
    return dict(wall_seconds=after['wall']-before['wall'],
                user_cpu_seconds=after['user']-before['user'],
                system_cpu_seconds=after['system']-before['system'],
                peak_rss_kib_cumulative=after['peak_rss_kib_cumulative'],
                proc_memory_after=after['proc_memory'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/dataset/shaoshitong/search-r1-rl/topd')
    parser.add_argument('--prompt-lengths', default='256', help='Start256; later256,1024,4096')
    parser.add_argument('--response-length', type=int, default=500)
    parser.add_argument('--repetitions', type=int, default=1, choices=(1, 2))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU hardware probe requires CUDA_VISIBLE_DEVICES empty')
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < 32:
        raise RuntimeError('Require own Slurm CPU32 allocation; no GPU allocation')
    if os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Do not allocate GPUs for this CPU probe')
    import socket
    if socket.gethostname().split('.')[0] != 'h100-11':
        raise RuntimeError('Probe must use the own h100-11 local assets/environment')
    root = Path(args.root).resolve()
    if root != Path('/dataset/shaoshitong/search-r1-rl/topd'):
        raise RuntimeError('Unexpected task root')
    lengths = [int(value) for value in args.prompt_lengths.split(',')]
    if not lengths or any(value not in (256, 1024, 4096) for value in lengths):
        raise RuntimeError('Only bounded representative prompt256/1024/4096 cases')
    if args.response_length not in (256, 500):
        raise RuntimeError('Only response256/500 cases')
    import torch
    import transformers
    import tensordict
    if not torch.__version__.startswith('2.4.0') or transformers.__version__ != '4.46.3' or tensordict.__version__ != '0.5.0':
        raise RuntimeError('Use pinned Torch2.4/Transformers4.46.3/TD0.5 environment')
    torch.set_num_threads(16)
    from transformers import AutoTokenizer
    from search_r1.baselines import distillation
    model_path = root/'assets-root/assets/qwen2.5-3b'
    if not model_path.is_dir() or not model_path.resolve().is_relative_to(root):
        raise RuntimeError('Missing own local BASE surrogate')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open('x')  # Preserve old results; do not overwrite.
    def write(record):
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True)+'\n')
        stream.flush()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    record = dict(scope='HARDWARE_BASE_SURROGATE_ONLY_NOT_TRAINED_TEACHER_NOT_TOPD_RUN',
        slurm_job=os.environ['SLURM_JOB_ID'], pid=os.getpid(), model_path=str(model_path.resolve()),
        dtype='float32', device='cpu', torch_threads=torch.get_num_threads(), chunk=16,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, tensordict=tensordict.__version__),
        prompts=lengths, response_length=args.response_length, repetitions=args.repetitions,
        code_source_sha256=hashlib.sha256(Path(distillation.__file__).read_bytes()).hexdigest(),
        model_config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
        sequence_scope='Synthetic repeated ordinary tokens, no gold targets; original teacher_inputs concatenation',
        limitations='No actual rollout distribution or trained teacher; outer2560 extrapolations are estimates, not timings')
    write(dict(kind='metadata', **record))
    before = usage()
    teacher = distillation.FrozenTeacher(str(model_path), tokenizer, 'cpu')
    write(dict(kind='model_load', parameter_count=teacher.parameter_count,
               parameter_bytes=sum(p.numel()*p.element_size() for p in teacher.model.parameters()),
               **elapsed(before, usage())))
    if any(parameter.requires_grad for parameter in teacher.model.parameters()) or teacher.model.training:
        raise RuntimeError('Probe teacher must be frozen/eval')
    token = next(value for value in tokenizer.encode('the answer', add_special_tokens=False)
                 if value not in {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id})
    original_inputs = distillation.teacher_inputs
    state = {}
    def timed_inputs(*values, **kwargs):
        start = usage()
        result = original_inputs(*values, **kwargs)
        state['teacher_inputs'] = elapsed(start, usage())
        state['actual_teacher_sequence_shape'] = list(result[0].shape)
        return result
    distillation.teacher_inputs = timed_inputs
    def hidden_pre(module, values, kwargs):
        state['hidden_start'] = usage()
    def hidden_post(module, values, kwargs, result):
        state['hidden_forward'] = elapsed(state.pop('hidden_start'), usage())
    def head_pre(module, values):
        state['head_start'] = time.perf_counter()
    def head_post(module, values, result):
        state['head_linear_wall_seconds'] += time.perf_counter()-state.pop('head_start')
        state['head_calls'] += 1
    hooks = [teacher.model.base_model.register_forward_pre_hook(hidden_pre, with_kwargs=True),
             teacher.model.base_model.register_forward_hook(hidden_post, with_kwargs=True),
             teacher.model.get_output_embeddings().register_forward_pre_hook(head_pre),
             teacher.model.get_output_embeddings().register_forward_hook(head_post)]
    try:
        for length in lengths:
            prompts = torch.full((1, length), token, dtype=torch.long)
            responses = torch.full((1, args.response_length), token, dtype=torch.long)
            attention = torch.ones_like(responses)
            for repetition in range(args.repetitions):
                state.clear()
                state.update(head_linear_wall_seconds=0., head_calls=0)
                before = usage()
                result = teacher.log_probs(prompts, responses, attention, temperature=1., targets=None, chunk=16)
                total = elapsed(before, usage())
                if result.shape != responses.shape or not bool(torch.isfinite(result).all()):
                    raise RuntimeError('Original log_probs returned invalid/nonfinite output')
                write(dict(kind='score', prompt_length=length, response_length=args.response_length,
                    repetition=repetition, temperature=1., targets_none=True, result_finite=True,
                    result_shape=list(result.shape), scored_tokens_cumulative=teacher.scored_tokens,
                    input_prepare=state['teacher_inputs'], hidden_forward=state['hidden_forward'],
                    actual_teacher_sequence_shape=state['actual_teacher_sequence_shape'],
                    head_linear_wall_seconds=state['head_linear_wall_seconds'], head_calls=state['head_calls'],
                    head_logsoftmax_gather_residual_wall_seconds=total['wall_seconds']-state['hidden_forward']['wall_seconds']-state['teacher_inputs']['wall_seconds'],
                    extrapolated_outer2560_wall_seconds_ESTIMATE=total['wall_seconds']*2560,
                    timing_scope='First call cold; each call measured, no excluded warmup; hooks add small measurement overhead',
                    **total))
                del result
        write(dict(kind='complete', status='HARDWARE_PROBE_COMPLETED_NO_TRAINING'))
    finally:
        distillation.teacher_inputs = original_inputs
        for hook in hooks:
            hook.remove()
        stream.close()


if __name__ == '__main__':
    main()
