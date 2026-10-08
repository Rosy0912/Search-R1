"""Record existing actor forwards and guards, without executing an extra forward.

Tensor backend is supplied by the actual bound GPU worker. CPU tests use a
declared callable spy. This module never loads a model, starts Ray or restores RNG.
"""
from contextlib import contextmanager
from qualification_contract_v1 import require


@contextmanager
def record_existing_actor_forwards(actor, backend):
    forward, guard = actor._forward_micro_batch, actor._policy_safety_guard
    records, context = [], []
    gradient_serial = 0

    def observed_guard(monitor, temperature, control, identity, phase, writes):
        require(phase in ('before','after') and not context, 'Unexpected existing monitor nesting')
        context.append(dict(kind='fixed_global256_monitor',phase=phase,
                            mini_index=identity['mini_index'],actual_optimizer_writes=writes))
        try:
            return guard(monitor,temperature,control,identity,phase,writes)
        finally:
            context.pop()

    def observed_forward(*args, **kwargs):
        nonlocal gradient_serial
        # Capture actual inputs before the existing method; return its exact
        # original object. No clone is substituted into model input or loss.
        micro = kwargs.get('micro_batch', args[0] if args else None)
        temperature = kwargs.get('temperature',args[1] if len(args)>1 else None)
        observation_before = backend.capture_observer_read_state()
        snapshot = backend.snapshot_micro_tensors(micro)
        require(backend.capture_observer_read_state() == observation_before,
                'Probability input recorder altered actual state before original forward')
        original_output = forward(*args, **kwargs)
        read_before = backend.capture_observer_read_state()
        try:
            entropy,current = original_output
            output = backend.snapshot_probability_tensors(entropy,current)
            require(backend.micro_digest(micro) == backend.micro_digest(snapshot),
                    'Existing forward changed immutable input tensors')
            if context:
                identity = dict(context[-1])
            else:
                identity = dict(kind='gradient_full2560',serial=gradient_serial,
                                mini_index=gradient_serial//16,micro_index=gradient_serial%16)
                gradient_serial += 1
            records.append(dict(identity=identity,temperature=temperature,actual_micro=snapshot,
                                actual_output=output,source='EXISTING_ACTOR_FORWARD_NO_EXTRA_PROBE'))
        finally:
            # Reading/cloning may not consume RNG, mutate parameters/Adam,
            # change tensor versions/layout/modes or move the live tensors.
            require(backend.capture_observer_read_state() == read_before,
                    'Probability recorder altered actual model/Adam/RNG/layout')
        return original_output

    actor._forward_micro_batch,actor._policy_safety_guard = observed_forward,observed_guard
    try:
        yield records
    finally:
        actor._forward_micro_batch,actor._policy_safety_guard = forward,guard


def validate_recorded_coverage(records):
    gradients = [row for row in records if row['identity']['kind']=='gradient_full2560']
    monitors = [row for row in records if row['identity']['kind']=='fixed_global256_monitor']
    require(len(gradients)==160 and [r['identity']['serial'] for r in gradients]==list(range(160)),
            'Missing/reordered actual160 gradient forwards')
    require(len(monitors)==320, 'Missing actual20×16 monitor forwards')
    for mini in range(10):
        for phase,writes in (('before',mini),('after',mini+1)):
            group=[r for r in monitors if r['identity']['mini_index']==mini and r['identity']['phase']==phase]
            require(len(group)==16 and all(r['identity']['actual_optimizer_writes']==writes for r in group),
                    'Fixed monitor actual pre/post Adam boundary differs')
    require(all(row['temperature']==1 for row in records), 'Existing forward changed temperature')
    return dict(gradient_forward_count=160,monitor_forward_count=320,extra_forward_count=0)
