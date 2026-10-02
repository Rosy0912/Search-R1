"""Read-only cost accounting for the unchanged Search-R1 validation path."""
import time
from search_r1.llm_agent.generation import LLMGenerationManager


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
        self.cost['generated_tokens'] += int((output.batch['responses'] != self.pad_id).sum())
        return output


class BapoValidationManager(LLMGenerationManager):
    def __init__(self, tokenizer, actor_rollout_wg, config, is_validation=False):
        self.cost = dict(generated_tokens=0, prefill_tokens=0, generation_rows=0,
                         trajectories=0, retrieval_calls=0, retrieval_requests=0,
                         retrieval_wall_seconds=0.)
        worker = CountedRollout(actor_rollout_wg, tokenizer.pad_token_id, self.cost)
        super().__init__(tokenizer, worker, config, is_validation)

    def run_llm_loop(self, gen_batch, initial_input_ids):
        self.cost['trajectories'] += len(gen_batch)
        return super().run_llm_loop(gen_batch, initial_input_ids)

    def _batch_search(self, queries):
        self.cost['retrieval_calls'] += len(queries)
        self.cost['retrieval_requests'] += 1
        start = time.perf_counter()
        try:
            return super()._batch_search(queries)
        finally:
            self.cost['retrieval_wall_seconds'] += time.perf_counter() - start
