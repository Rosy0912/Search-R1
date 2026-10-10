"""GPU leaf for saved-response replay, admitted only by independent CPU parent."""
import argparse
import importlib
from pathlib import Path
import os
import json
import time
try:
    from .common import (require, read_json, read_rows, check_file, file_ref, exclusive_json,
        remote_output, bind_training, semantic_sha)
    from .replay_math import encode_response, replay_response
except ImportError:
    from common import (require, read_json, read_rows, check_file, file_ref, exclusive_json,
        remote_output, bind_training, semantic_sha)
    from replay_math import encode_response, replay_response


def run(request_ref, slot):
    request, rp = read_json(request_ref); cfg, cp = read_json(request['config']); cfg['_config_sha256'] = cp['sha256']
    require(request['schema'] == 'opsd-signal-heldout-replay-request-v1' and request['new_samples_or_training'] == 0,
        'Replay cannot receive a training/sampling request')
    modules = bind_training(cfg); gates = modules['gates']
    physical = cfg['owner']['physical_gpus'][slot]
    require(os.getuid() == cfg['owner']['uid'] == 1001 and os.environ.get('E1_RUN_ID') == request['analysis_run_id']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(physical), 'Replay leaf UID/run/physical mapping differs')
    owner = gates.process(request['controller']['pid'])
    require(gates.same_process(owner, request['controller']) and owner['run_id'] == request['analysis_run_id']
        and str(Path(request['analysis_helper_root']) / 'replay.py') in owner['argv'], 'Replay CPU parent no longer owns leaf')
    check_file(request['analysis_source_manifest']); check_file(request['analysis_publication'])
    try:
        from .replay import verify_analysis_publication, resource_boundary
    except ImportError:
        from replay import verify_analysis_publication, resource_boundary
    verify_analysis_publication(request['analysis_helper_root'], request['analysis_source_manifest'], request['analysis_publication'])
    resource_boundary(cfg, modules, owner, request['analysis_run_id'], released=False)
    bank, bp = read_json(request['bank']); endpoint, ep = read_json(request['endpoint'])
    records, results_ref = read_rows(endpoint['results']); data, data_ref = read_rows(cfg['validation_identity'])
    require(bp['sha256'] == request['bank']['sha256'] and endpoint['completed_outer'] == request['completed_outer']
        and endpoint['version'] == request['version'] and len(data) == len(records) == 512, 'Replay endpoint/bank changed')
    modules['prepare'].verify_runtime(cfg)
    identity = request['input_identity']; modules['evaluation'].verify_metadata(identity['source_files'])
    for ref in identity['adapter_files']: check_file(ref)
    actual_config, _ = read_json(identity['adapter_config'])
    require(modules['evaluation'].validate_adapter_config(actual_config, cfg) == identity['adapter_config_contract'],
        'Replay adapter forward/scaling contract changed')
    directory = remote_output(Path(request['output']) / ('worker-' + str(slot)))
    directory.mkdir(exist_ok=False)
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    require(torch.__version__.split('+')[0] == '2.4.0' and version('transformers') == '4.46.3'
        and version('peft') == '0.13.2', 'Replay frozen actual environment differs')
    require(torch.cuda.device_count() == 1 and torch.cuda.is_available(), 'Replay must see exactly one owned physical GPU')
    evaluator = modules['evaluation_worker']; zero = request['completed_outer'] == 0
    values = load_file(str(Path(identity['adapter']['path']) / 'adapter_model.safetensors'), device='cpu')
    payload = evaluator.tensor_checks(values, identity['expected_adapter_tensor_sha256'], zero); del values
    tokenizer = AutoTokenizer.from_pretrained(cfg['paths']['model'], use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(cfg['paths']['model'], torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model = PeftModel.from_pretrained(base, identity['adapter']['path'], adapter_name='replay_current', is_trainable=False)
    model.set_adapter('replay_current')
    loaded = evaluator.tensor_checks(get_peft_model_state_dict(model, adapter_name='replay_current'),
        identity['expected_adapter_tensor_sha256'], zero)
    forward = evaluator.validate_loaded_lora(model, 'replay_current')
    for parameter in model.parameters(): parameter.requires_grad_(False)
    require(model.get_base_model().get_output_embeddings().weight.dtype == torch.bfloat16, 'Replay head compute dtype changed')
    core = importlib.import_module('search_e1.core'); qa = importlib.import_module('search_r1.qa_protocol')
    process = gates.process(os.getpid())
    started = dict(schema='opsd-signal-heldout-replay-worker-v1', status='RUNNING', slot=slot,
        analysis_run_id=request['analysis_run_id'], request=rp, config=cp, bank=bp, endpoint=ep,
        physical_gpu=physical, gpu_uuid=cfg['owner']['gpu_uuids'][slot], process=process,
        actual_payload=payload, actual_loaded_adapter=loaded, actual_loaded_forward=forward,
        new_samples_or_training=0, parent_owned=True, posthoc=True)
    exclusive_json(directory / 'started.json', started)
    rows_by_index = {r['validation_index']: r for r in records}; bank_index = {r['validation_index']: r for r in bank['entries']}
    completed = 0
    with (directory / 'results.jsonl').open('x') as sink:
        for i in range(slot, 512, 4):
            resource_boundary(cfg, modules, owner, request['analysis_run_id'], released=False)
            q = data[i]; row = rows_by_index[i]; B = bank_index[i]
            require(B['question_id'] == row['question_id'] == q['question_id']
                and B['validation_row_semantic_sha256'] == row['row_semantic_sha256'] == semantic_sha(q), 'Replay question/B identity changed')
            trace, tp = read_json(row['trace']); result = trace['native_result']
            decoded = modules['training_metrics'].protocol(core, qa, result, q['answers'])
            require(decoded == trace['protocol'] and decoded['protocol_em'] == row['protocol_em']
                and semantic_sha(result) == row['native_result_sha256'], 'Replay strict EM/actual saved response changed')
            if B['available']:
                reference_trace, _ = read_json(B['source_zero_trace'])
                require(reference_trace['native_result']['response'] == B['reference_response']
                    and reference_trace['protocol']['protocol_em'] == 1, 'Common B bank changed')
            encoded = encode_response(tokenizer, q['question'], result, B['reference_response'], core, modules['sibling_scores'])
            case = directory / ('validation-%03d' % i); case.mkdir()
            with (case / 'tokens.jsonl').open('x') as tokens:
                def record(value):
                    value.update(question_id=q['question_id'], validation_index=i, endpoint=ep,
                        bank=bp, source_trace=tp, current_adapter_tensor_sha256=identity['expected_adapter_tensor_sha256'])
                    tokens.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                replay = replay_response(model, tokenizer, encoded, modules, record, 'replay_current')
                tokens.flush(); os.fsync(tokens.fileno())
            encoder_ref = exclusive_json(case / 'encoding.json', encoded)
            replay.update(token_records=file_ref(case / 'tokens.jsonl'), encoding=encoder_ref,
                no_reference_generation=True, no_retrieval_calls=True, no_optimizer=True)
            saved = {k: row[k] for k in ('version', 'completed_outer', 'validation_index', 'question_id', 'dataset',
                'row_semantic_sha256', 'protocol_em', 'protocol_valid_final_action', 'failure_class', 'termination', 'search_calls')}
            saved.update(schema='opsd-signal-heldout-replay-case-v1', bank_available=B['available'], bank=bp,
                source_trace=tp, replay=replay, current_adapter_tensor_sha256=identity['expected_adapter_tensor_sha256'])
            sink.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + '\n'); sink.flush(); os.fsync(sink.fileno())
            completed += 1
    require(completed == 128, 'Replay worker must retain128 EM rows even with metric NA')
    after = evaluator.tensor_checks(get_peft_model_state_dict(model, adapter_name='replay_current'),
        identity['expected_adapter_tensor_sha256'], zero)
    modules['evaluation'].verify_metadata(identity['source_files'])
    for ref in identity['adapter_files']: check_file(ref)
    modules['prepare'].verify_runtime(cfg)
    return exclusive_json(directory / 'complete.json', dict(**started, status='COMPLETE', accepted=True,
        rows=128, actual_adapter_after=after, adapter_unchanged=True, results=file_ref(directory / 'results.jsonl'), ended_unix=time.time()))


def main():
    p = argparse.ArgumentParser()
    for name in ('request', 'request-sha256'): p.add_argument('--' + name, required=True)
    p.add_argument('--slot', type=int, choices=range(4), required=True)
    a = p.parse_args(); result = run(dict(path=a.request, sha256=a.request_sha256), a.slot)
    print(json.dumps(dict(accepted=True, worker_complete=result)))


if __name__ == '__main__': main()
