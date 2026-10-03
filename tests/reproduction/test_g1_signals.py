import unittest
from search_g1.signals import contexts, model_search_count, validate_probe_reply

class Signals(unittest.TestCase):
    def test_observation_answer_and_search_markup_never_count(self):
        a='<search>q</search>';o='<information><answer>Paris</answer><search>x</search></information>';r=a+o+'<answer>Rome</answer>'
        spans=[[0,len(a)],[len(a+o),len(r)]]
        self.assertEqual(contexts('prompt example <answer>bad</answer>',r,spans)['answer'],'Rome')
        self.assertEqual(model_search_count(r,spans),1)
        self.assertEqual(contexts('',a+o,[[0,len(a)]]),None)
    def test_tags_cannot_cross_observation_boundary(self):
        r='<answer>Paris</answer>'
        self.assertIsNone(contexts('',r,[[0,8],[10,len(r)]]))
    def test_commit_state_is_last_answer_token_not_closing_tag(self):
        r='<answer>Paris</answer> extra'
        c=contexts('P',r,[[0,len(r)]]);self.assertEqual(c['answer_state'],'P<answer>Paris');self.assertEqual(c['commit'],'P<answer>')
    def test_wrong_missing_or_nonfinite_probe_reply_is_rejected(self):
        items=[{'id':'0'}]
        for reply in [{'version':'v','results':[]},{'version':'v','results':[{'id':'1','version':'v','flip':.2,'boundary':.3}]},{'version':'v','results':[{'id':'0','version':'v','flip':float('nan'),'boundary':.3}]}]:
            with self.assertRaises(ValueError):validate_probe_reply(reply,items)
    def test_valid_probes_preserve_immutable_batch_version(self):
        rows=[{'id':'0','version':'v50','flip':.7,'boundary':.2}]
        self.assertEqual(validate_probe_reply({'version':'v50','results':rows},[{'id':'0'}]),(rows,'v50'))
if __name__=='__main__':unittest.main()
