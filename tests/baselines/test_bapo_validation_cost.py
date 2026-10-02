"""Cost telemetry includes actual padded model work, before output trimming."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from verl import DataProto
from search_r1.baselines.bapo_validation import BapoValidationManager
from search_r1.llm_agent.generation import LLMGenerationManager


class ValidationCost(unittest.TestCase):
    def test_counts_padding_before_original_manager_trims_output(self):
        calls = []
        def generate(batch):
            calls.append(batch)
            # Three real rows plus a padded copy: lengths1,2,3,1.
            return DataProto.from_dict({'responses': torch.tensor([[4,0,0],[5,6,0],[7,8,9],[4,0,0]])})
        worker = SimpleNamespace(world_size=4, generate_sequences=generate)
        tokenizer = SimpleNamespace(pad_token_id=0)
        config = SimpleNamespace(num_gpus=4, max_prompt_length=4096, max_obs_length=500, max_start_length=2048)
        manager = BapoValidationManager(tokenizer, worker, config, True)
        batch = DataProto.from_dict({'input_ids': torch.tensor([[1,2],[3,0],[4,5]]),
                                    'attention_mask': torch.tensor([[1,1],[1,0],[1,1]])})
        output = manager._generate_with_gpu_padding(batch)
        self.assertEqual(output.batch['responses'].shape[0], 3)
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual(manager.cost['prefill_tokens'], 7)
        self.assertEqual(manager.cost['generated_tokens'], 7)
        self.assertEqual(manager.cost['generation_rows'], 4)
        # Returned real rows are byte-for-byte unchanged, padding excluded.
        self.assertEqual(output.batch['responses'].tolist(), [[4,0,0],[5,6,0],[7,8,9]])

    def test_search_and_trajectory_counts_do_not_change_results(self):
        tokenizer = SimpleNamespace(pad_token_id=0)
        config = SimpleNamespace(num_gpus=4, max_prompt_length=4096, max_obs_length=500, max_start_length=2048)
        manager = BapoValidationManager(tokenizer, SimpleNamespace(), config, True)
        result = {'result': ['one','two']}
        with patch.object(LLMGenerationManager, '_batch_search', return_value=result) as search:
            self.assertIs(manager._batch_search(['q1','q2']), result)
            search.assert_called_once_with(['q1','q2'])
        self.assertEqual(manager.cost['retrieval_calls'], 2)
        self.assertEqual(manager.cost['retrieval_requests'], 1)
        self.assertGreaterEqual(manager.cost['retrieval_wall_seconds'], 0.)
        batch = DataProto.from_dict({'input_ids': torch.ones(3,2,dtype=torch.long)})
        with patch.object(LLMGenerationManager, 'run_llm_loop', return_value=result):
            self.assertIs(manager.run_llm_loop(batch, batch.batch['input_ids']), result)
        self.assertEqual(manager.cost['trajectories'], 3)


if __name__ == '__main__':
    unittest.main()
