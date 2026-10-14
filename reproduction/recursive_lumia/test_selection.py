import unittest
from collections import Counter
from .selection import Selector,quotas

class SelectionTests(unittest.TestCase):
    def setUp(self):self.sources=['nq']*4000+['hotpotqa']*4192
    def refresh(self,s,step):
        c=s.candidates(step)
        r={i:([0,1,0,1,0] if i%3 else [0]*5) for i in c}
        s.update(step,c,r)
    def test_150_round_budget_strata_uniqueness_and_sliding_cap(self):
        s=Selector(self.sources);history=[]
        for step in range(1,151):
            if (step-1)%25==0:self.refresh(s,step)
            rows=s.choose();history.append(rows)
            self.assertEqual(len(rows),512);self.assertEqual(len(set(rows)),512)
            count=Counter(i for b in history[-16:] for i in b)
            self.assertLessEqual(max(count.values()),4)
            src=Counter(self.sources[i] for i in rows)
            for source in src:self.assertLess(abs(src[source]-512*self.sources.count(source)/8192),2)
        self.assertEqual(sum(map(len,history)),76800)
    def test_checkpoint_restores_next_choice_without_reading_future_rewards(self):
        s=Selector(self.sources);self.refresh(s,1)
        for _ in range(13):s.choose()
        state=s.state_dict();expected=[s.choose() for _ in range(5)]
        t=Selector(self.sources);t.load_state_dict(state)
        self.assertEqual(expected,[t.choose() for _ in range(5)])
    def test_homogeneous_screen_falls_back_to_uniform(self):
        s=Selector(self.sources);c=s.candidates(1)
        s.update(1,c,{i:[0]*5 for i in c})
        self.assertIsNone(s.pool);self.assertEqual(len(s.choose()),512)
    def test_rejects_non_binary_or_filtered_group(self):
        s=Selector(self.sources);c=s.candidates(1)
        with self.assertRaises(AssertionError):s.update(1,c,{i:[0,1] for i in c})
        with self.assertRaises(AssertionError):s.update(1,c,{i:[0,0,0,.5,1] for i in c})
    def test_candidates_are_train_only_and_repeatable(self):
        s=Selector(self.sources);c=s.candidates(1)
        self.assertEqual(c,s.candidates(1));self.assertEqual(len(set(c)),4096)
        self.assertTrue(all(0<=i<len(self.sources) for i in c))
        self.assertEqual(sum(quotas(2048,s.sizes).values()),2048)

if __name__=='__main__':unittest.main()
