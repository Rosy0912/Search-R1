"""Standalone, fixed-prompt 16/32 trajectory timing. Never trains or opens training locks.

Single-GPU microbenchmark, not an eight-rank Tree throughput claim. Real vLLM
outputs stay in process; only aggregate JSON/profiler tables are persisted.
String-stop comparison is diagnostic, not a validated production mu contract.
"""
import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import re
import sys
import time


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, time=time.time(), **values), allow_nan=False), flush=True)


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def read_rows(path, count):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.slice(0, count).to_pylist()
    with path.open() as stream:
        result = []
        for line in stream:
            result.append(json.loads(line))
            if len(result) == count: break
    return result


def prompts_from_rows(tokenizer, rows, system_file):
    system = Path(system_file).read_text() if system_file else None
    result = []
    for row in rows:
        messages = row.get('prompt')
        if messages is None:
            if system is None:
                raise ValueError('Question-only data requires --system-prompt-file; do not invent prompts')
            messages = [dict(role='system', content=system), dict(role='user', content=row['question'])]
        if isinstance(messages, str):
            result.append(tokenizer.encode(messages, add_special_tokens=False))
        else:
            result.append(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    return result


def worker_for_kv(llm):
    executor = llm.llm_engine.model_executor
    worker = getattr(executor, 'driver_worker', None)
    if worker is None:
        raise RuntimeError('This profiling helper requires the actual TP1 driver_worker')
    worker = getattr(worker, 'worker', worker)
    if not all(hasattr(worker, key) for key in ('cache_engine', 'gpu_cache', '_init_cache_engine')):
        raise RuntimeError('Actual vLLM worker KV cache API differs; refuse guessed cache instrumentation')
    return worker


def reset_kv(llm, torch):
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('Cannot reset KV with unfinished requests')
    worker = worker_for_kv(llm)
    torch.cuda.synchronize()
    start = time.perf_counter()
    worker.cache_engine = None
    worker.gpu_cache = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    freed = time.perf_counter() - start
    start = time.perf_counter()
    worker._init_cache_engine()
    torch.cuda.synchronize()
    return dict(kv_free_wall_seconds=freed, kv_init_wall_seconds=time.perf_counter()-start)


def profile_table(profiler):
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('cumulative').print_stats(25)
    return text.getvalue()


def aggregate_cpu(tokenizer, outputs, raw_module, identity, scan_mode):
    """Compare old whole-response decode with actual frozen prefix scan, same IDs."""
    ids = [list(request.outputs[0].token_ids) for request in outputs]
    decoded, legacy_wall = timed_call(lambda: [tokenizer.decode(x, skip_special_tokens=True) for x in ids])
    canonical, canonical_wall = timed_call(lambda: [raw_module.canonical_text(x) for x in decoded])
    encoded, encoding_wall = timed_call(lambda: [tokenizer.encode(x, add_special_tokens=False) for x in canonical])
    # Register/validate real tokenizer once per action batch, as current rollout does.
    context, binding_wall = timed_call(lambda: raw_module._validation_context(tokenizer, identity))
    profiler = cProfile.Profile()
    profiler.enable()
    cpu_start = time.process_time(); start = time.perf_counter()
    cut_lengths = []
    for token_ids, text in zip(ids, canonical):
        if not token_ids:
            cut_lengths.append(0)
            continue
        prefixes, complete, _ = raw_module._scan_utf8_prefixes(tokenizer, token_ids, identity,
            check_native_prefixes=(scan_mode == 'native'), _context=context)
        if '</search>' in prefixes[-1] or '</answer>' in prefixes[-1]:
            choices = [i+1 for i, prefix in enumerate(prefixes) if complete[i] and prefix.startswith(text)]
            if not choices: raise ValueError('No stable original-token canonical boundary')
            cut_lengths.append(choices[0])
        else:
            cut_lengths.append(len(token_ids))
    scan_wall = time.perf_counter()-start; scan_cpu = time.process_time()-cpu_start
    profiler.disable()
    context.check_complete()
    raw_count = sum(map(len, ids)); retained = sum(cut_lengths)
    retained_native_ids = [token_ids[:cut] for token_ids, cut in zip(ids, cut_lengths)]
    return canonical, retained_native_ids, dict(
        whole_decode_wall_seconds=legacy_wall, canonical_parse_wall_seconds=canonical_wall,
        canonical_encode_wall_seconds=encoding_wall, tokenizer_binding_wall_seconds=binding_wall,
        prefix_scan_mode=scan_mode, prefix_scan_wall_seconds=scan_wall, prefix_scan_cpu_seconds=scan_cpu,
        emitted_tokens=raw_count, retained_action_tokens=retained, discarded_tail_tokens=raw_count-retained,
        discarded_tail_fraction=(raw_count-retained)/raw_count if raw_count else 0.,
        cprofile_top_cumulative=profile_table(profiler))


def run_condition(llm, tokenizer, raw_module, identity, prompts, args, size, stop_mode, cache_mode, torch):
    from vllm import SamplingParams
    states = [dict(context=list(prompt), search_calls=0, done=False) for prompt in prompts[:size]]
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    manager=LLMGenerationManager(tokenizer,None,GenerationConfig(max_turns=4,max_start_length=2048,
        max_prompt_length=args.context,max_response_length=500,max_obs_length=500,num_gpus=1,
        force_final_answer=False,search_url=args.url,topk=3))
    stages = []
    total_started = time.perf_counter()
    emit('condition_start', trajectories=size, stop_mode=stop_mode, cache_mode=cache_mode)
    for turn in range(4):
        active = [(i, state) for i, state in enumerate(states) if not state['done'] and len(state['context'])+500 <= args.context]
        if not active: break
        kv = reset_kv(llm, torch) if cache_mode == 'rebuild' else dict(kv_free_wall_seconds=0., kv_init_wall_seconds=0.)
        emit('generation_start', trajectories=size, stop_mode=stop_mode, cache_mode=cache_mode,
             turn=turn, active=len(active), prefill_tokens=sum(len(x['context']) for _, x in active), **kv)
        parameters = dict(temperature=1., top_p=1., top_k=-1, max_tokens=500, logprobs=1,
            seed=args.seed+turn, detokenize=True)
        if stop_mode == 'tag':
            parameters.update(stop=['</search>', '</answer>'], include_stop_str_in_output=True)
        torch.cuda.synchronize(); start = time.perf_counter()
        outputs = llm.generate(prompt_token_ids=[s['context'] for _, s in active],
            sampling_params=SamplingParams(**parameters), use_tqdm=False)
        torch.cuda.synchronize(); generation_wall = time.perf_counter()-start
        emit('generation_complete', trajectories=size, stop_mode=stop_mode, cache_mode=cache_mode,
            turn=turn, generation_wall_seconds=generation_wall,
            emitted_tokens=sum(len(x.outputs[0].token_ids) for x in outputs))
        canonical, retained_native_ids, cpu = aggregate_cpu(tokenizer, outputs, raw_module, identity, args.scan_mode)
        finish_counts = {}
        for request in outputs:
            reason = str(request.outputs[0].finish_reason)
            finish_counts[reason] = finish_counts.get(reason, 0)+1
        row = dict(turn=turn, active=len(active), generation_wall_seconds=generation_wall,
            prefill_tokens=sum(len(x['context']) for _, x in active), finish_reasons=finish_counts, **kv, **cpu)
        emit('cpu_complete', trajectories=size, stop_mode=stop_mode, cache_mode=cache_mode,
             **{k: v for k, v in row.items() if k != 'cprofile_top_cumulative'})
        if args.executor=='current':
            emit('interaction_start',trajectories=size,turn=turn,active=len(active))
            started=time.perf_counter()
            observations,dones,valid,searches=manager.execute_predictions(canonical,tokenizer.pad_token,
                torch.ones(len(active),dtype=torch.bool),do_search=True)
            observation_ids,observation_mask=manager._process_next_obs(observations,return_attention_mask=True)
            for pos,((index,state),tokens) in enumerate(zip(active,retained_native_ids)):
                state['context'].extend(tokens)
                state['context'].extend(observation_ids[pos][observation_mask[pos].bool()].tolist())
                state['search_calls']+=int(searches[pos]);state['done']=bool(dones[pos])
            row.update(interaction_wall_seconds=time.perf_counter()-started,
                retrieval_http_json_wall_seconds=getattr(manager,'retrieval_http_json_wall_seconds',0.),
                valid_actions=sum(valid),search_actions=sum(searches))
            emit('interaction_complete',trajectories=size,turn=turn,**{k:row[k] for k in
                ('interaction_wall_seconds','retrieval_http_json_wall_seconds','valid_actions','search_actions')})
            stages.append(row)
            continue
        searches = []
        for (index, state), text, tokens in zip(active, canonical, retained_native_ids):
            state['context'].extend(tokens)
            if '</answer>' in text:
                state['done'] = True
                continue
            match = re.search(r'<search>(.*?)</search>', text, re.S)
            if match and state['search_calls'] < 4:
                searches.append((index, match.group(1).strip()))
            else:
                state['done'] = True
        retrieval_wall = 0.
        if searches:
            if not args.url:
                raise ValueError('Actual search actions require --url; no invented observations')
            import requests
            emit('retrieval_start', trajectories=size, turn=turn, queries=len(searches))
            start = time.perf_counter()
            response = requests.post(args.url, json=dict(queries=[q for _, q in searches], topk=3,
                return_scores=True), timeout=300)
            response.raise_for_status(); docs = response.json()['result']
            if len(docs) != len(searches): raise ValueError('Retriever cardinality mismatch')
            for (index, _), passages in zip(searches, docs):
                content = '\n'.join(f"Doc {j+1}: {p.get('document',p).get('contents','')}" for j,p in enumerate(passages))
                info = tokenizer.decode(tokenizer.encode(content, add_special_tokens=False)[:480])
                states[index]['context'].extend(tokenizer.encode('\n<information>\n'+info+'\n</information>\n', add_special_tokens=False))
                states[index]['search_calls'] += 1
            retrieval_wall = time.perf_counter()-start
            emit('retrieval_complete', trajectories=size, turn=turn, queries=len(searches), wall_seconds=retrieval_wall)
        row['retrieval_wall_seconds'] = retrieval_wall
        stages.append(row)
    result = dict(trajectories=size, stop_mode=stop_mode, cache_mode=cache_mode, stages=stages,
        history='retained_original_sampled_action_ids_plus_real_retrieval_observation',
        condition_wall_seconds=time.perf_counter()-total_started,
        completed=sum(s['done'] for s in states), budget_terminated=sum(not s['done'] for s in states),
        emitted_tokens=sum(r['emitted_tokens'] for r in stages),
        discarded_tail_tokens=sum(r['discarded_tail_tokens'] for r in stages))
    emit('condition_complete', **{k:v for k,v in result.items() if k != 'stages'})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--tokenizer-sha', required=True)
    parser.add_argument('--url')
    parser.add_argument('--system-prompt-file')
    parser.add_argument('--executor',choices=['current','legacy'],default='current')
    parser.add_argument('--sizes', default='16,32')
    parser.add_argument('--stop-modes', default='eos,tag')
    parser.add_argument('--cache-modes', default='persistent,rebuild')
    parser.add_argument('--scan-mode', choices=['native','linear'], default='native')
    parser.add_argument('--memory-util', type=float, default=.4)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise ValueError('Use a new output directory; preserve previous measurements')
    output.mkdir(parents=True)
    sys.path.insert(0, args.repo)
    import torch
    from vllm import LLM
    import vllm
    from search_r1.baselines import tree_raw_tokens as raw_module
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    emit('load_start', vllm_version=vllm.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if vllm.__version__ != '0.6.3': raise ValueError('This helper is for actual vLLM0.6.3 only')
    start = time.perf_counter()
    llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=1,
        gpu_memory_utilization=args.memory_util, max_model_len=args.context, enforce_eager=True, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    identity = raw_module.tokenizer_identity(tokenizer, args.tokenizer_sha)
    sizes = [int(x) for x in args.sizes.split(',')]
    if not sizes or any(x not in (16,32) for x in sizes): raise ValueError('Only fixed16/32 diagnostic cohorts')
    rows = read_rows(Path(args.data), max(sizes))
    if len(rows) < max(sizes): raise ValueError('Insufficient actual data rows')
    prompts = prompts_from_rows(tokenizer, rows, args.system_prompt_file)
    emit('load_complete', wall_seconds=time.perf_counter()-start,
         prompt_sha256=hashlib.sha256(json.dumps(prompts,separators=(',',':')).encode()).hexdigest(),
         prompts=len(prompts), dtype='bfloat16')
    results = []
    for size in sizes:
        for cache in args.cache_modes.split(','):
            if cache not in ('persistent','rebuild'): raise ValueError('Unknown cache mode')
            for stop in args.stop_modes.split(','):
                if stop not in ('eos','tag'): raise ValueError('Unknown stop mode')
                result = run_condition(llm,tokenizer,raw_module,identity,prompts,args,size,stop,cache,torch)
                results.append(result)
                (output/f'{size}-{cache}-{stop}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(dict(
        scope='single_gpu_diagnostic_not_eight_rank_training_or_native_mu_acceptance',executor=args.executor,
        source_model=args.model, source_data=args.data, results=results), indent=2, allow_nan=False))
    emit('complete', conditions=len(results), output=str(output))


if __name__ == '__main__':
    main()
