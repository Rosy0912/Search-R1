"""Read-only cost accounting for the unchanged Search-R1 validation path."""
import time
from search_r1.llm_agent.generation import LLMGenerationManager
from .bapo import answer_from_spans, is_idk, legal_action_records


def validation_answer_counts(output, tokenizer):
    from verl.trainer.main_ppo import decode_model_response_spans
    width = output.batch['responses'].shape[1]
    mask = output.batch['attention_mask'][:, -width:].bool() & output.batch['info_mask'][:, -width:].bool()
    counts = dict(idk_count=0, answered_count=0, invalid_count=0)
    for ids, row_mask in zip(output.batch['responses'], mask):
        spans = decode_model_response_spans(tokenizer, ids, row_mask)
        answer = answer_from_spans(spans)
        if answer is None or not legal_action_records([dict(text=s) for s in spans]):
            counts['invalid_count'] += 1
        elif is_idk(answer):
            counts['idk_count'] += 1
        else:
            counts['answered_count'] += 1
    return counts


class CountedRollout:
    def __init__(self, worker, pad_id, cost):
        self.worker, self.pad_id, self.cost = worker, pad_id, cost

    def __getattr__(self, key):
        return getattr(self.worker, key)

    def generate_sequences(self, batch):
        self.cost['prefill_tokens'] += int(batch.batch['attention_mask'].sum())
        self.cost['generation_rows'] += len(batch)
        output = self.worker.generate_sequences(batch)
        # Count before the existing manager removes padding rows or truncates
        # actions at </search>/</answer> and rolling response caps.
        if batch.meta_info.get('preserve_sampling_tokens') is True:
            records = output.non_tensor_batch.get('fresh200_sampling')
            if records is None: raise ValueError('Fresh200 evaluation missing native token records')
            self.cost['generated_tokens'] += sum(len(r['raw_generated_token_ids']) for r in records)
        else:
            self.cost['generated_tokens'] += int((output.batch['responses'] != self.pad_id).sum())
        return output


class BapoValidationManager(LLMGenerationManager):
    def __init__(self, tokenizer, actor_rollout_wg, config, is_validation=False):
        self.cost = dict(generated_tokens=0, prefill_tokens=0, generation_rows=0,
                         trajectories=0, retrieval_calls=0, retrieval_requests=0,
                         retrieval_wall_seconds=0.)
        self.answer_counts = dict(idk_count=0, answered_count=0, invalid_count=0)
        worker = CountedRollout(actor_rollout_wg, tokenizer.pad_token_id, self.cost)
        super().__init__(tokenizer, worker, config, is_validation)

    def run_llm_loop(self, gen_batch, initial_input_ids):
        self.cost['trajectories'] += len(gen_batch)
        if getattr(self.config, 'preserve_sampling_tokens', False):
            if gen_batch.meta_info.get('fresh200_evaluation') is not True or gen_batch.meta_info.get('do_sample') is not False:
                raise ValueError('Fresh200 evaluation requires explicitly labeled greedy role')
        output = super().run_llm_loop(gen_batch, initial_input_ids)
        for key, value in validation_answer_counts(output, self.tokenizer).items():
            self.answer_counts[key] += value
        return output

    def answer_metrics(self):
        n = self.cost['trajectories']
        return dict(**self.answer_counts,
                    idk_ratio=self.answer_counts['idk_count'] / max(1, n),
                    answered_coverage=self.answer_counts['answered_count'] / max(1, n))

    def _batch_search(self, queries):
        self.cost['retrieval_calls'] += len(queries)
        self.cost['retrieval_requests'] += 1
        start = time.perf_counter()
        try:
            return super()._batch_search(queries)
        finally:
            self.cost['retrieval_wall_seconds'] += time.perf_counter() - start
