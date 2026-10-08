"""One case, all canonical histories/boundaries and references; no training."""
import copy
from contextlib import nullcontext
from .continuation import run

VERSIONS = ('source100', 'update1', 'update5', 'epoch1')


def execute_case(record, model, tokenizer, core, rollout, state_api, retrieve,
                 score_context, forcing, save):
    source = record['source_train_row']['row']; conditions = record['reference_interventions']['conditions']
    original_reference = next(c['reference'] for c in conditions if c['name'] == 'original')
    histories = [('original_vllm', record['original_student'])]
    histories += [('actual_hf_' + v, states) for v, states in sorted(record['actual_hf_states'].items())]
    slots = []
    for history, states in histories:
        for skipped in states['skipped_boundaries']:
            slots.append(dict(history=history, aliases=[skipped['boundary']], status='skipped', reason=skipped['reason']))
        for state in states['executable_states']:
            # Reconstruct canonical object only, not execution annotations.
            canonical = {k: copy.deepcopy(state[k]) for k in state_api.initial(state['backend'])}
            for role, condition, version in ([('teacher', c, 'source100') for c in conditions] +
                                              [('student', dict(name='none', reference=None, status='ready'), v) for v in ('source100', 'epoch1')]):
                key = f"{history}-{state['state_sha256']}-{role}-{condition['name']}-{version}"
                metadata = dict(history=history, aliases=state['aliases'], state_sha256=state['state_sha256'],
                                role=role, reference_condition=condition['name'], active_adapter='disabled' if role == 'teacher' else version,
                                version=version, history_owner=history, reference_sha256=condition.get('reference_sha256'))
                if condition['status'] != 'ready':
                    slots.append(dict(metadata, status='skipped', reason=condition['reason'])); continue
                model.set_adapter(version)
                with (model.disable_adapter() if role == 'teacher' else nullcontext()):
                    result = run(model, tokenizer, core, rollout, state_api, canonical, source['question'], source['answers'],
                                 condition['reference'], retrieve, score_context)
                result.update(metadata); save(key + '.generation.json', result)
                # Newly authored branch positions alone; history tokens remain as
                # causal context, not re-labelled generated branch supervision.
                branch_spans = result['policy_spans'][len(canonical['policy_spans']):]
                for target_version in VERSIONS:
                    model.set_adapter(target_version)
                    probe = forcing(result['response'], branch_spans, source['question'], original_reference,
                                    target_version, key + '-forced-' + target_version)
                    probe.update(history_owner=history, generated_by=metadata,
                                 forcing_reference_condition='original', response_is_common_across_four_checkpoints=True,
                                 training_online_loss_claimed=False, branch_policy_only=True)
                    save(key + '-forced-' + target_version + '.summary.json', probe)
                slots.append(dict(metadata, status='complete', protocol_em=result['protocol_em'],
                                  new_policy_spans=branch_spans, generation_file=key + '.generation.json'))
    return dict(case_id=record['case_id'], pilot=record['pilot'], slots=slots,
                observations_preserved=True, no_training=True,
                teacher_rollout_is_not_training_q=True)
