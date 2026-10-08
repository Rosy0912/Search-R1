"""Explicit empty ranker shards; unchanged nonempty author processing."""
import contextvars
import functools
import hashlib
import inspect
import textwrap

_generation = contextvars.ContextVar('native_ranker_generation', default=None)
MARKER = '_native_empty_ranker'


def empty_batch(prompt_length, response_length):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    p, r = int(prompt_length), int(response_length)
    assert p > 0 and r > 0
    fields = {k: torch.empty((0, n), dtype=torch.long) for k, n in
              [('prompts', p), ('responses', r), ('response_mask', r),
               ('input_ids', p+r), ('attention_mask', p+r), ('position_ids', p+r)]}
    fields['rm_scores'] = torch.empty((0, r), dtype=torch.float32)
    return DataProto(batch=TensorDict(fields, batch_size=[0]),
                     non_tensor_batch={'uid': np.empty(0, dtype=object),
                                       '__num_turns__': np.empty(0, dtype=np.int32)},
                     meta_info={'metrics': [], 'reward_extra_keys': [], MARKER: True})


def is_empty(batch):
    import numpy as np
    import torch
    from verl import DataProto
    assert isinstance(batch, DataProto)
    if MARKER not in batch.meta_info:
        return False
    assert batch.meta_info[MARKER] is True and len(batch) == 0
    fields = {'prompts', 'responses', 'response_mask', 'input_ids',
              'attention_mask', 'position_ids', 'rm_scores'}
    assert batch.batch is not None and set(batch.batch.keys()) == fields
    assert all(v.device.type == 'cpu' and v.ndim == 2 and v.shape[0] == 0
               for v in batch.batch.values())
    p, r = batch.batch['prompts'].shape[1], batch.batch['responses'].shape[1]
    assert p > 0 and r > 0
    for k in fields:
        expected = p if k == 'prompts' else r if k in {
            'responses', 'response_mask', 'rm_scores'} else p+r
        assert batch.batch[k].shape[1] == expected
        assert batch.batch[k].dtype == (torch.float32 if k == 'rm_scores' else torch.long)
    assert set(batch.non_tensor_batch) == {'uid', '__num_turns__'}
    for k, dtype in [('uid', np.dtype(object)), ('__num_turns__', np.dtype(np.int32))]:
        value = batch.non_tensor_batch[k]
        assert isinstance(value, np.ndarray) and value.shape == (0,) and value.dtype == dtype
    return True


def install_worker(worker, record, preserve):
    if getattr(worker, '_native_empty_ranker_installed', False):
        return
    generate, post = worker.generate_sequences, worker._postprocess

    @functools.wraps(generate)
    async def run(instance, batch, *args, **kwargs):
        token = _generation.set(dict(rows=len(batch), validation=bool(
            batch.meta_info.get('validate', False)), post_calls=0))
        try:
            return await generate(instance, batch, *args, **kwargs)
        finally:
            _generation.reset(token)

    @functools.wraps(post)
    def process(instance, inputs):
        state = _generation.get()
        if not inputs:
            assert state and state['rows'] > 0 and not state['validation']
            assert state['post_calls'] == 1, 'Empty main output must still fail'
            preserve('empty-ranker-pre-postprocess', inputs)
            cfg = instance.config.reranker_actor_rollout_ref.rollout
            result = empty_batch(cfg.prompt_length, cfg.response_length)
            record('empty_local_ranker_output', dict(main_rows=state['rows'],
                ranker_rows=0, qualification_not_fabricated=True))
        else:
            result = post(instance, inputs)
        if state is not None:
            state['post_calls'] += 1
        return result

    worker.generate_sequences = run
    worker._postprocess = process
    worker._native_empty_ranker_installed = True
    record('empty_ranker_worker_installed', dict(empty_main_prohibited=True))


def _compile(original, replacements, extra):
    source = textwrap.dedent(inspect.getsource(original))
    for before, after in replacements:
        assert source.count(before) == 1, ('Author source anchor changed', before)
        if '\n' in after:
            indent = source[:source.index(before)].rsplit('\n', 1)[-1]
            assert not indent.strip()
            after = after.replace('\n', '\n' + indent)
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    namespace.update(extra)
    exec(compile(source, '<frozen-native-empty-ranker>', 'exec'), namespace)
    return namespace[original.__name__], hashlib.sha256(source.encode()).hexdigest()


def install_controller(manager, trainer_class, record):
    from verl import DataProto
    if getattr(manager, '_native_empty_ranker_installed', False):
        return

    def concat_ranker(outputs):
        valid = [v for v in outputs if not is_empty(v)]
        assert all(len(v) > 0 for v in valid)
        if valid:
            return DataProto.concat(valid)
        assert outputs and all(is_empty(v) for v in outputs)
        record('empty_global_ranker_output', dict(ranker_rows=0,
            qualification_not_fabricated=True))
        return outputs[0]

    original = manager.generate_sequences.__func__
    new, digest = _compile(original, [
        ('reranker_output = DataProto.concat(reranker_outputs)',
         'reranker_output = _native_concat_ranker(reranker_outputs)'),
        ('reranker_timing, reranker_aggregated_metrics = self._performance_metrics(reranker_metrics, reranker_output)',
         'reranker_timing, reranker_aggregated_metrics = ({}, {}) if len(reranker_output) == 0 else self._performance_metrics(reranker_metrics, reranker_output)'),
        ('**reranker_outputs[0].meta_info',
         '**next((v.meta_info for v in reranker_outputs if len(v) > 0), reranker_outputs[0].meta_info)')
    ], {'_native_concat_ranker': concat_ranker})
    manager.generate_sequences = new.__get__(manager, type(manager))
    manager._native_empty_ranker_installed = True
    if not getattr(trainer_class, '_native_empty_ranker_fit_installed', False):
        fit, fit_digest = _compile(trainer_class.fit, [
            ('if "response_mask" not in main_batch.batch.keys():',
             'if reranker_train_enabled and len(reranker_batch) == 0:\n'
             '    assert _native_is_empty(reranker_batch)\n'
             '    reranker_train_enabled = False\n'
             '    metrics["reranker/skipped_no_rows"] = 1\n'
             '    _native_record("ranker_update_skipped_no_rows", dict(step=self.global_steps, ranker_rows=0, Adam_steps=0, scheduler_steps=0))\n'
             'if "response_mask" not in main_batch.batch.keys():')
        ], {'_native_is_empty': is_empty, '_native_record': record})
        trainer_class.fit = fit
        trainer_class._native_empty_ranker_fit_installed = True
        record('empty_ranker_fit_installed', dict(transformed_source_sha256=fit_digest,
            persistent_trainable_flag_unchanged=True))
    record('empty_ranker_manager_installed', dict(transformed_source_sha256=digest))
