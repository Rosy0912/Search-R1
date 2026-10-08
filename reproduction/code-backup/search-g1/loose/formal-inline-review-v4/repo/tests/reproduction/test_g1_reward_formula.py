import ast, pathlib, unittest, types
p=pathlib.Path(__file__).resolve().parents[2]/'verl/trainer/main_ppo_eco.py'
tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EcoRewardManager');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_score_probe');ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns)
class RewardFormula(unittest.TestCase):
    def score(self,n,correct=True,answer='Paris',b=.2,d=.7):
        self_=types.SimpleNamespace(PENALTY=.1,MAX_SEARCH=2,probe_gate=True,probe_base=0.,probe_alpha_cost=False,w_cost=.02,wrong_floor=.05,bal_ground=.6,bal_nosearch_bonus=.5,wrong_ground_scale=0.)
        return ns['_score_probe'](self_,dict(n_search=n,correct=correct,answer=answer),dict(flip=d,boundary=b))[0]
    def test_source_repeat_cost_is_fixed_after_first_search(self):
        self.assertAlmostEqual(self.score(1),1+.6*.8*.7)
        for n in (2,3,4):self.assertAlmostEqual(self.score(n),self.score(1)-.02)
    def test_direct_correctness_and_strict_outcome_order(self):
        self.assertAlmostEqual(self.score(0),1+.5*.2)
        self.assertEqual(self.score(2,False),.05)
        self.assertEqual(self.score(2,False,None),-.1)
        self.assertGreater(self.score(4,b=1,d=0),self.score(2,False))
if __name__=='__main__':unittest.main()
