"""Revalidate the root same-process five gate from its actual retained prefix."""
from pathlib import Path

from reproduction.common8.actor_probe_calibration_acceptance import _document
from reproduction.common8.actor_probe_retained_prefix import load_prefix
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.runtime import verify_scheduler_receipts
from reproduction.common8.formal_inline_safety import verify_first_five
from reproduction.common8.gpu_training_admission import verify_historical_training_gpu_samples
from reproduction.common8.actor_probe_checkpoint import verify_roundtrip_receipt
from reproduction.common8.actor_probe_evidence import verify_actor_readouts
from reproduction.common8.formal_inline_evaluation import verify_pair
from reproduction.common8.verify_full import verify_five_scheduler_prefix
from search_g1.actor_probe_recovery import identity


def verify_root_five(entry, segment):
    assert entry['origin'] == dict(completed_outer=0,actual_optimizer_writes=0,attempt_index=0,aborted_attempts=0)
    run = entry['run']; base = Path(entry['base'])
    assert run.get('actor_probe_recovery') is None
    assert segment['actual_interval_accepted'] is True and segment['identity'] == identity(run)
    assert segment['first'] == 1 and segment['last'] >= 5 and segment['base'] == str(base)
    prefix = load_prefix(base,run,5); records = prefix['selected_updates']
    assert records == segment['counter_segment']['retained_updates'][:5]
    five, file = _document(base,base/'logs/five-updates-acceptance.json')
    assert five['accepted'] is True and five['run_id'] == run['run_id']
    assert five['policy_safety_mode'] == run['policy_safety_mode']
    assert five['formal_inline_plan_sha256'] == segment['formal_inline_plan_sha256']
    assert [row['step'] for row in five['actual_updates']] == list(range(1,6))
    for saved, actual in zip(five['actual_updates'],records):
        assert saved['probe_version'] == actual['probe_version'] == run['initial_probe_version']
        assert all(float(saved['metrics'][key]) == value for key,value in actual['metrics'].items())
    gates = {str(step):segment['behavior_probability_gates'][str(step)] for step in range(1,6)}
    assert five['behavior_probability_gates'] == gates
    schedulers = verify_scheduler_receipts(base,run,1,5,gates,retained_prefix=prefix)
    for rank in range(4):
        verify_five_scheduler_prefix(five['scheduler_receipts'][str(rank)],schedulers[str(rank)])
    safety = verify_first_five(base,run,gates,schedulers,retained_prefix=prefix)
    assert canonical_bytes(safety) == canonical_bytes(five['formal_inline_first_five'])
    readouts = verify_actor_readouts(base,run,records,retained_prefix=prefix)
    assert five['actual_actor_readouts']['retained_prefix_files'] is None
    assert {k:v for k,v in readouts.items() if k != 'retained_prefix_files'} == \
        {k:v for k,v in five['actual_actor_readouts'].items() if k != 'retained_prefix_files'}
    pair = verify_pair(base,run)
    assert pair == five['paired_development']
    assert five['checkpoint'] == segment['checkpoint_proofs']['5']['receipt']
    assert five['independently_verified_saved_tensor_count'] == segment['checkpoint_proofs']['5']['finite_tensor_count'] == 435
    assert five['changed_parameters'], 'First-five checkpoint has no authenticated parameter delta'
    # The original writer compared all canonical tensors with the official
    # model. Recheck the same claimed delta against actual immutable bytes.
    import torch
    from safetensors import safe_open
    official = Path(run['initial_model']); checkpoint = Path(run['checkpoint_dir'])/'actor/global_step_5'
    original, _ = _document(official,official/'model.safetensors.index.json')
    current, _ = _document(base,checkpoint/'model.safetensors.index.json')
    old = original['weight_map']; new = current['weight_map']; changed = []
    for name in old:
        with safe_open(str(official/old[name]),framework='pt',device='cpu') as reader:
            before = reader.get_tensor(name)
        with safe_open(str(checkpoint/new[name]),framework='pt',device='cpu') as reader:
            after = reader.get_tensor(name)
        if not torch.equal(before,after): changed.append(name)
        del before, after
    assert changed == five['changed_parameters']
    roundtrip = verify_roundtrip_receipt(base,run,segment['full_states']['5'])
    occupancy = verify_historical_training_gpu_samples(base,run,schedulers,five['actual_first_five_gpu_occupancy'])
    return dict(role='actor_probe_root_actual_first_five',identity=identity(run),accepted=True,
        receipt=file,actual_complete_outers=5,actual_optimizer_writes=records[-1]['optimizer_writes_after'],
        safety_selected_evidence_sha256=safety['selected_evidence_sha256'],
        readout_evidence_sha256=document_sha256(readouts),paired_development_file_sha256=pair['file_sha256'],
        changed_parameter_count=len(changed),roundtrip=roundtrip,historical_gpu_occupancy=occupancy,
        current_liveness_claimed=False,full_training_or_final_acceptance=False)
