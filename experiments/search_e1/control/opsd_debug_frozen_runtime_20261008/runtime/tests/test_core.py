import unittest
from search_e1.core import mine_pairs, reward, response_encoding, clipped_forward_kl

class TestCore(unittest.TestCase):
    def test_em(self):
        self.assertEqual(reward('<answer>The Paris.</answer>', ['Paris']), 1)
        self.assertEqual(reward('<think>Paris</think>', ['Paris']), 0)
    def test_pairs(self):
        def row(text, score, calls):
            return dict(question_id='q', question='Q', answers=['A'], response=text, reward=score, search_calls=calls)
        rows = [row('short',1,1),row('long long',1,2),row('WRONG!',0,0)]
        pair = mine_pairs(rows)[0]
        self.assertEqual(pair['reference'], 'short')
        self.assertEqual(pair['student'], 'WRONG!')
        self.assertEqual(mine_pairs([row('wrong',0,1),row('bad',0,2)]), [])
    def test_mask(self):
        class CharTokenizer:
            def __call__(self,text,**kw):
                return {'input_ids':list(range(len(text))), 'offset_mapping':[(i,i+1) for i in range(len(text))]}
        text = '<think>x</think><information>doc</information><answer>A</answer>'
        _, mask = response_encoding(CharTokenizer(), text)
        self.assertEqual(sum(mask), len('<think>x</think><answer>A</answer>'))
        _, mask = response_encoding(CharTokenizer(), 'a<information>unclosed')
        self.assertEqual(sum(mask),1)
        text='A\n<information>doc</information>\nB'
        _, mask = response_encoding(CharTokenizer(), text, [(0,1),(len(text)-1,len(text))])
        self.assertEqual(sum(mask),2)  # External whitespace is excluded by rollout provenance.
    def test_kl_direction_clipping_and_teacher_grad(self):
        import torch
        s=torch.tensor([[0.,-100.]],requires_grad=True)
        t=torch.tensor([[-5.,5.]],requires_grad=True)
        got=clipped_forward_kl(s,t,10.)
        sl=s.log_softmax(-1); tl=t.detach().log_softmax(-1)
        expected=(tl.exp()*(tl-sl)).clamp(max=10.).sum(-1)
        torch.testing.assert_close(got,expected)
        got.sum().backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(s.grad)
        self.assertLess(got.item(),10.)  # negative contribution is retained
    def test_zero_kl(self):
        import torch
        x=torch.randn(4,17)
        torch.testing.assert_close(clipped_forward_kl(x,x),torch.zeros(4),atol=1e-6,rtol=0)
if __name__ == '__main__':
    unittest.main()
