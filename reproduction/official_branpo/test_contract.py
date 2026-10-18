import copy
import json
import os
from pathlib import Path
import unittest
from .contract import INSTRUCTION, job_name, task_from_common, training_overrides
from .retrieval import common_results


class ContractTest(unittest.TestCase):
    def row(self):
        return {'prompt': [{'role': 'user', 'content': INSTRUCTION + '\nQuestion: Exact question? '}],
                'reward_model': {'ground_truth': {'target': ['First Answer', '2,718']}},
                'extra_info': {'index': 'nq:train:0'}, 'data_source': 'nq'}

    def test_exact_question_gold_id_without_normalization(self):
        row = self.row()
        before = copy.deepcopy(row)
        task = task_from_common(row)
        self.assertEqual(task, dict(question='Exact question? ',
            ground_truth=['First Answer', '2,718'], data_source='nq', index='nq:train:0'))
        self.assertEqual(row, before)

    def test_unknown_common_prompt_rejected(self):
        row = self.row()
        row['prompt'][0]['content'] = 'Question: made up'
        with self.assertRaises(ValueError):
            task_from_common(row)

    def test_empty_gold_rejected(self):
        row = self.row()
        row['reward_model']['ground_truth']['target'] = []
        with self.assertRaises(ValueError):
            task_from_common(row)

    def test_no_unrequested_scale(self):
        self.assertNotEqual(job_name('3b'), job_name('7b'))
        with self.assertRaises(ValueError):
            job_name('4b')

    def test_actual_source_recipe_and_200_real_updates(self):
        upstream = Path(os.environ.get('BRANPO_UPSTREAM',
            str(Path(__file__).resolve().parents[3] / 'official-branpo')))
        script = (upstream / 'agent/search/start_branpo_trainer.sh').read_text()
        for size in ['3b', '7b']:
            options = training_overrides(script, size, '/same/init', '/exclusive/output')
            values = dict(x.split('=', 1) for x in options)
            self.assertEqual(values['actor_rollout_ref.actor.ppo_mini_batch_size'], '512')
            self.assertEqual(values['actor_rollout_ref.rollout.n'], '4')
            self.assertEqual(values['data.max_response_length'], '3072')
            self.assertEqual(int(values['trainer.total_steps']) - int(values['trainer.start_step']), 200)
            self.assertEqual(values['actor_rollout_ref.model.path'], '/same/init')
            self.assertEqual(values['+rllm.env.env_args.max_steps'], '4')
            with self.assertRaises(ValueError):
                training_overrides(script.replace('rollout.n=4', 'rollout.n=2'), size, '/m', '/o')

    def test_retrieval_order_content_score_preserved(self):
        docs = [{'document': {'id': i, 'contents': ' Title\nExact\n text '}, 'score': 9.25-i}
                for i in range(3)]
        before = copy.deepcopy(docs)
        result = common_results({'result': [docs]}, 1, 3)
        self.assertEqual([x['id'] for x in result[0]], [0, 1, 2])
        self.assertEqual(result[0][0]['contents'], ' Title\nExact\n text ')
        self.assertEqual(result[0][1]['score'], 8.25)
        self.assertEqual(docs, before)

    def test_partial_retrieval_cannot_be_used(self):
        with self.assertRaises(ValueError):
            common_results({'result': [[]]}, 1, 3)
        with self.assertRaises(ValueError):
            common_results({'result': []}, 1, 3)


if __name__ == '__main__':
    unittest.main()
