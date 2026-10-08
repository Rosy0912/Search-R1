"""Candidate fixed-weight worker; root launches only after independent admission.

Importing this file is CPU-only. No launcher, optimizer, train step or signals.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def package(name, directory):
    module = types.ModuleType(name); module.__path__ = [str(Path(directory).resolve())]
    sys.modules[name] = module
    return module


def main():
    p = argparse.ArgumentParser()
    for key in ('config', 'config-sha256', 'ticket', 'ticket-sha256', 'phase', 'worker-index'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args(); index = int(a.worker_index)
    here = Path(__file__).resolve().parent
    package('m3exec', here)
    from m3exec.admission import checked, sha, validate_config, validate_ticket, select_phase, fresh_retriever,proc_identity
    from m3exec.continuation import score_tap
    from m3exec.engine import execute_case
    config = checked(dict(path=a.config, sha256=a.config_sha256)); config['_config_sha256'] = a.config_sha256
    validate_config(config)
    if not 0 <= index < 3: raise ValueError('Invalid worker index')
    for prefix, key in (('m3prep', 'cpu_helper_manifest'), ('m3debug', 'debug_helper_manifest'), ('m4identity', 'm4_helper_manifest')):
        manifest = checked(config[key]); directory = Path(config.get(key.replace('_manifest','_source'), Path(config[key]['path']).resolve().parent)).resolve()
        for name, digest in manifest.get('files',manifest).items():
            if sha(directory / name) != digest: raise ValueError('Dependency source mismatch: ' + name)
        package(prefix, directory)
    manifest = checked(config['execution_manifest'])
    for name, digest in manifest.get('files',manifest).items():
        if sha(here / name) != digest: raise ValueError('Execution source changed')
    from m3prep import state as state_api
    from m3debug.common import load_science, atomic_json
    from m3debug.weight_identity import check_weight_manifest
    from m3debug.admission import RetrieverAdmission
    from m3debug.token_probe import probe, FullVocabularyBudget
    from m4identity.identity import check_adapter_labels
    ticket = checked(dict(path=a.ticket, sha256=a.ticket_sha256))
    validate_ticket(ticket, config, a.phase, index)
    summary = checked(config['preparation_summary'])
    if summary['accepted_cpu_preparation'] is not True: raise ValueError('Preparation not accepted')
    cpu_manifest = checked(config['cpu_helper_manifest'])
    if {n:p['sha256'] for n,p in summary['source']['helper'].items()} != cpu_manifest['files']:
        raise ValueError('Preparation was produced by different CPU helper')
    states_proof = summary['states_output']; states_path = Path(states_proof['path'])
    if sha(states_path) != states_proof['sha256']: raise ValueError('Preparation states changed')
    records = [json.loads(line) for line in states_path.read_text().splitlines()]
    case_proof = summary['source']['cases']; case_path = Path(case_proof['path'])
    if sha(case_path) != case_proof['sha256']: raise ValueError('Frozen192 cases changed')
    cases = {c['case_id']:c for c in (json.loads(line) for line in case_path.read_text().splitlines())}
    for record in records:
        if cases[record['case_id']]['case_semantic_sha256'] != record['source_case_sha256']:
            raise ValueError('Preparation/case source binding differs')
    chosen = select_phase(records, a.phase, checked(config['pilot_receipt']) if a.phase == 'remaining' else None, config['run_id'])
    assigned = [r for r in chosen if records.index(r) % 3 == index]
    weights = check_weight_manifest(config['weight_manifest'], config['model'], config['adapters'])
    check_adapter_labels(weights)
    admission = RetrieverAdmission(config['retriever_admission'], config['retriever_url'])
    def live():
        current = ticket
        if 'renewal_path' in ticket:
            path = Path(ticket['renewal_path']); current = checked(dict(path=str(path),sha256=sha(path)))
            if current['owner'] != ticket['owner'] or current.get('renewal_path') != ticket['renewal_path']:
                raise ValueError('Lease owner/path changed')
        validate_ticket(current, config, a.phase, index)
        return fresh_retriever(config,admission)
    live()
    # This is the first optional heavy import; never reached by CPU fixtures.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.logits_process import LogitsProcessorList
    from peft import PeftModel
    from safetensors.torch import load_file
    import importlib.metadata
    if torch.__version__.split('+')[0]!='2.4.0':raise ValueError('Different native Torch version')
    for name,version in {'transformers':'4.46.3','peft':'0.13.2'}.items():
        if importlib.metadata.version(name)!=version:raise ValueError('Different native package: '+name)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one externally assigned GPU required')
    live()
    actual_uuid = torch.cuda.get_device_properties(0).uuid
    if str(actual_uuid) != ticket['gpu_uuid']:
        raise ValueError('Actual GPU UUID differs from reservation')
    for version, path in config['adapters'].items():
        tensors = load_file(str(Path(path) / 'adapter_model.safetensors'), device='cpu')
        if not tensors or any(not torch.isfinite(t).all().item() for t in tensors.values()): raise ValueError('Nonfinite adapter')
        if version == 'source100':
            b = [v for k, v in tensors.items() if 'lora_B' in k]
            if len(b) != 252 or any(torch.count_nonzero(v).item() for v in b): raise ValueError('source100 is not zero-B')
        del tensors
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False
    core, opsd, rollout, science = load_science(config['runtime'], with_torch=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model'], use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(config['model'], torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    live()
    base = base.cuda().eval()
    live()
    model = PeftModel.from_pretrained(base, config['adapters']['source100'], adapter_name='source100', is_trainable=False)
    for version in ('update1', 'update5', 'epoch1'):
        live()
        model.load_adapter(config['adapters'][version], adapter_name=version, is_trainable=False)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if any(p.dtype != torch.float32 for n, p in model.named_parameters() if 'lora_' in n): raise ValueError('Adapter dtype changed')
    generation = model.generation_config.to_dict()
    if generation['repetition_penalty'] != 1.05 or generation['eos_token_id'] != [151645, 151643]: raise ValueError('Inherited generation config changed')
    # Check only consumed selected HF trace artifacts; never scan original845k.
    for record in assigned:
        for version, reconstruction in record['actual_hf_states'].items():
            artifact = reconstruction['artifact']; trace = checked(artifact)
            expected_prefix = tokenizer.apply_chat_template(core.prompt(record['source_train_row']['row']['question']), tokenize=False, add_generation_prompt=True)
            if trace['prompt_text'] != expected_prefix: raise ValueError('HF source chat template differs')
            for event in trace['turns']:
                ids = tokenizer.encode(event['context_text'], add_special_tokens=False)
                if len(ids) != event['context_tokens']: raise ValueError('HF source context token count differs')
                if 'generated_token_ids' in event and tokenizer.decode(event['generated_token_ids'], skip_special_tokens=True) != event['decoded_segment_before_structural_cut']:
                    raise ValueError('Actual HF source IDs do not decode to kept-source provenance')
                if 'inserted_information' in event:
                    info_ids = tokenizer.encode(event['formatted_information_before_truncation'], add_special_tokens=False)[:480]
                    if info_ids != event['inserted_content_token_ids']: raise ValueError('Actual HF observation encoding differs')
            recomputed = state_api.reconstruct_hf(trace)
            if [s['state_sha256'] for s in recomputed['executable_states']] != [s['state_sha256'] for s in reconstruction['executable_states']]:
                raise ValueError('Prepared HF state differs from current bound trace')
    directory = Path(config['output']) / a.phase / f'worker-{index}'
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(directory / 'identity.json', dict(run_id=config['run_id'], ticket=ticket, weights=weights, science=science,
                generation_config=generation, pid=os.getpid(), uid=os.getuid(),proc_starttick=proc_identity(os.getpid())['starttick'],ppid=os.getppid(),
                e1_run_id=os.environ.get('E1_RUN_ID'),gpu_uuid=str(actual_uuid),config_sha256=a.config_sha256,phase=a.phase, assigned=[r['case_id'] for r in assigned]))
    def distribution(scores):
        logits = scores[0].float(); logp = logits.log_softmax(-1)
        values, ids = torch.topk(logp, 32)
        def finite(v): return float(v) if torch.isfinite(v) else str(float(v))
        return dict(top32_ids=ids.detach().cpu().tolist(), top32_logp=[finite(v) for v in values],
                    top32_actual_scores=[finite(v) for v in logits[ids]],
                    argmax_id=int(logits.argmax()), entropy=float(-(logp.exp() * torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))).sum()), normalization='actual score tensor FP32 log_softmax', full_vocabulary_dump=False)
    @contextmanager
    def capture(event):
        live()
        seen = []
        def tap(ids, raw, processed, processors):
            if not seen:
                event['fork_scores'] = dict(raw=distribution(raw), processed=distribution(processed),
                    actual_processor_order=[type(x).__module__ + '.' + type(x).__name__ for x in processors],
                    prefix_input_ids=ids[0].detach().cpu().tolist(), raw_is_unprocessed=True,
                    actual_greedy_target_id=int(processed[0].argmax()),
                    raw_logp_of_processed_greedy_target=float(raw[0].float().log_softmax(-1)[processed[0].argmax()]),
                    raw_logit_of_processed_greedy_target=float(raw[0,processed[0].argmax()]),
                    actual_processed_score_of_greedy_target=float(processed[0,processed[0].argmax()]),
                    raw_rank_of_processed_greedy_target=int((raw[0] > raw[0,processed[0].argmax()]).sum()) + 1,
                    isolated_action_token_scores=[dict(label=tag, token_ids=tokenizer.encode(tag,add_special_tokens=False),
                        scores=[dict(token_id=t,raw_logit=float(raw[0,t]), processed_score=float(processed[0,t]) if torch.isfinite(processed[0,t]) else str(float(processed[0,t])),
                                     raw_rank=int((raw[0]>raw[0,t]).sum())+1,processed_rank=int((processed[0]>processed[0,t]).sum())+1)
                                for t in tokenizer.encode(tag,add_special_tokens=False)]) for tag in ('<think>','<search>','<answer>')],
                    isolated_action_tokens_are_not_joint_tag_or_action_probabilities=True,
                    teacher_training_distribution_equivalence_claimed=False)
            seen.append(1)
        with score_tap(LogitsProcessorList, tap): yield
        event['actual_processor_steps'] = len(seen)
        atomic_json(directory / 'last-completed-unit.json',dict(kind='generate',turn=event.get('turn'),actual_processor_steps=len(seen)))
        if not seen: raise ValueError('Actual HF processor invocation not observed')
    def retrieve(query):
        import requests
        live()
        reply = requests.post(config['retriever_url'], json={'queries': [query], 'topk': 3, 'return_scores': True}, timeout=120)
        reply.raise_for_status(); return reply.json()
    budget = FullVocabularyBudget(0, 3, [], [])
    # Distribution zero-B sanity at real first active fork, no retrieval or new
    # rollout; check source100 enabled versus disabled raw head exact equality.
    ready = next((s for r in assigned for s in r['original_student']['executable_states']), None)
    if ready is None:raise ValueError('Worker has no executable original fork for GPU null-effect control')
    if ready is not None:
        r = next(r for r in assigned if ready in r['original_student']['executable_states'])
        question = r['source_train_row']['row']['question']
        prefix = tokenizer.apply_chat_template(core.prompt(question), tokenize=False, add_generation_prompt=True)
        ids = tokenizer(state_api.context(ready, prefix), return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        if ids.shape[1] + 500 <= 8192:
            model.set_adapter('source100'); model.eval()
            live()
            with torch.no_grad():
                transformer = model.get_base_model().base_model
                head = model.get_base_model().get_output_embeddings()
                enabled = head(transformer(input_ids=ids, use_cache=False).last_hidden_state[0, -1:]).detach().clone()
                with model.disable_adapter():
                    live()
                    disabled = head(transformer(input_ids=ids, use_cache=False).last_hidden_state[0, -1:])
            if not torch.equal(enabled, disabled): raise ValueError('Zero-B source100/base raw fork distributions differ')
            options = dict(max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                           stopping_criteria=rollout.StoppingCriteriaList([rollout.StopTags(tokenizer, ids.shape[1])]))
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
            live()
            with torch.no_grad(): plain = model.generate(ids, **options)
            plain_after = (torch.get_rng_state(), torch.cuda.get_rng_state())
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1]); event = {}
            with torch.no_grad(), capture(event): hooked = model.generate(ids, **options)
            hooked_after = (torch.get_rng_state(), torch.cuda.get_rng_state())
            if not torch.equal(plain, hooked) or any(not torch.equal(x, y) for x, y in zip(plain_after, hooked_after)):
                raise ValueError('Actual processor capture changes generation output/RNG')
            atomic_json(directory / 'zero-b-control.json', dict(accepted=True, raw_logits_equal=True,
                        actual_generate_tap_null_effect=True, control_generated_ids=plain[0, ids.shape[1]:].cpu().tolist(),
                        actual_fork_capture=event, case_id=r['case_id'], state_sha256=ready['state_sha256']))
        else: raise ValueError('First active zero-B fork has no native context budget')
    for record in assigned:
        live(); target = directory / record['case_id']; target.mkdir(mode=0o700)
        atomic_json(target / 'preparation.source.json', record)
        def save(name, value):
            path = target / name
            if path.exists(): raise FileExistsError(str(path))
            atomic_json(path, value)
        def forcing(response, spans, question, reference, version, key):
            live()
            result = probe(model, tokenizer, core, opsd, response, spans, question, reference, version,
                         record['case_id'], target / key, budget, response_origin='new_counterfactual_branch')
            atomic_json(directory / 'last-completed-unit.json',dict(kind='forcing_probe',case_id=record['case_id'],key=key,version=version,status=result['status']))
            return result
        original_reference = next(c['reference'] for c in record['reference_interventions']['conditions'] if c['name']=='original')
        pair = cases[record['case_id']]['original_pair']['row']
        with torch.no_grad():
            # Native fixed original y probes of changed teacher references.
            # Existing original/ordinary M1 probes are not silently regenerated.
            for condition in record['reference_interventions']['conditions']:
                if condition['name'] not in ('remove_final_answer','mismatched'): continue
                key = 'original-y-ref-' + condition['name']
                if condition['status'] != 'ready':
                    save(key + '.summary.json', dict(status='skipped', reason=condition['reason'])); continue
                model.set_adapter('source100')
                live()
                fixed = probe(model, tokenizer, core, opsd, pair['student'], pair['student_policy_spans'],
                              record['source_train_row']['row']['question'], condition['reference'], 'source100',
                              record['case_id'], target / key, budget, response_origin='counterfactual_reference_fixed_original_y')
                fixed.update(reference_status='counterfactual_reference_not_original_success_claim',
                             reference_condition=condition['name'], reference_sha256=condition['reference_sha256'],
                             history_owner='original_vllm', original_training_objective=False, fixed_original_y=True,
                             teacher_adapter='disabled', training_online_loss_claimed=False)
                save(key + '.summary.json', fixed)
                atomic_json(directory / 'last-completed-unit.json',dict(kind='fixed_reference_probe',case_id=record['case_id'],key=key,status=fixed['status']))
            # Existing accepted M1 responses; cross-force, NEVER new rollout.
            for owner, reconstruction in sorted(record['actual_hf_states'].items()):
                trace = checked(reconstruction['artifact'])
                for version in ('source100','update1','update5','epoch1'):
                    model.set_adapter(version); key = 'existing-hf-' + owner + '-forced-' + version
                    fixed = forcing(trace['response'], trace['policy_spans'], record['source_train_row']['row']['question'], original_reference, version, key)
                    fixed.update(history_owner=owner, source_trace=reconstruction['artifact'], branch_policy_only=False,
                                 forcing_reference_condition='original', actual_generation_already_existed=True,
                                 no_new_rollout_for_this_object=True, training_online_loss_claimed=False,
                                 response_is_common_across_four_checkpoints=True)
                    save(key + '.summary.json', fixed)
            result = execute_case(record, model, tokenizer, core, rollout, state_api, retrieve, capture, forcing, save)
        save('completion.json', dict(result, run_id=config['run_id'], phase=a.phase))
    atomic_json(directory / 'worker-final.json', dict(status='WORKER_COMPLETE', run_id=config['run_id'], phase=a.phase,
                assigned_case_ids=[r['case_id'] for r in assigned], completed_case_ids=[r['case_id'] for r in assigned]))


if __name__ == '__main__': main()
