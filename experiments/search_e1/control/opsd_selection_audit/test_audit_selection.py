"""Small synthetic CPU fixtures; never read the full experiment inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_selection import Audit, analyze_group, canonical, load_science, policy_only_text, stat_id, validate_row


def response(think="alpha", answer="gold", information="X"*1000, invalid=False):
    first = "<think>" + think + "</think><search>q</search>"
    external = "<information>" + information + "</information>"
    last = "<think>finish</think>" + ("<answer>" + answer if invalid else "<answer>" + answer + "</answer>")
    text = first + external + last
    return text, [[0,len(first)],[len(first+external),len(text)]]


def row(qid, sibling, dataset="nq", answer="gold", think="alpha", information="X"*1000, invalid=False):
    text, spans = response(think,answer,information,invalid)
    return dict(question_id=qid, question="synthetic fixture question", answers=["gold"], data_source=dataset,
                response=text, search_calls=1,policy_spans=spans,sibling=sibling,termination="answer",
                token_count=100,reward=int(answer=="gold" and not invalid))


def fixtures():
    cc = [row("fixture-cc",i) for i in range(5)]
    cw = [row("fixture-cw",0),row("fixture-cw",1,answer="bad",information="Y"*1000),
          row("fixture-cw",2,answer="bad",think="different"*20),
          row("fixture-cw",3,answer="bad"),row("fixture-cw",4,answer="bad")]
    hard = [row("fixture-hard",i,dataset="hotpotqa",answer="bad",invalid=i%2==0) for i in range(5)]
    return [cc,cw,hard]


class SelectionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime=os.environ.get("E1_AUDIT_TEST_RUNTIME")
        if not runtime:
            raise RuntimeError("Set E1_AUDIT_TEST_RUNTIME to the persisted original frozen runtime")
        cls.core,cls.qa,distance,cls.science=load_science(runtime)
        cls.distance=staticmethod(distance)

    def run_fixture(self, rows, pairs=None, expected_questions=3, partial=False, max_line=8<<20, change_stat=False, wrong_hash=False, bad_json=False):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root=Path(temp.name); rp=root/"rollouts.jsonl"; pp=root/"pairs.jsonl"
        if pairs is None:
            pairs=self.core.mine_pairs(rows)
        rb=b"".join((canonical(r)+"\n").encode() for r in rows)
        if bad_json: rb=b'{ not json\n'+rb.split(b'\n',1)[1]
        if partial: rb=rb[:-1]
        pb=b"".join((canonical(r)+"\n").encode() for r in pairs)
        rp.write_bytes(rb);pp.write_bytes(pb)
        expected={n:dict(stat_id(p),rows=len(values),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                  for n,p,values in (("rollouts",rp,rows),("pairs",pp,pairs))}
        if wrong_hash: expected["rollouts"]["sha256"]="0"*64
        audit=Audit(root/"private-output",self.core,self.qa,self.distance,self.science,expected,
                    expected_questions,max_line=max_line,min_free=0)
        if change_stat:
            original=audit.ingest_rollout
            def changed(row, p):
                original(row,p)
                if p["line"]==1:
                    s=rp.stat();os.utime(rp,ns=(s.st_atime_ns,s.st_mtime_ns+1))
            audit.ingest_rollout=changed
        result=audit.run()
        cases=[json.loads(x) for x in (root/"private-output/cases.private.jsonl").read_text().splitlines()]
        return result,cases

    def test_full_noncontiguous_groups_and_native_exact(self):
        groups=fixtures()
        rows=[groups[g][i] for i in range(5) for g in range(3)]
        result,cases=self.run_fixture(rows)
        self.assertTrue(result["accepted"],result["errors"])
        self.assertEqual(result["counters"]["unique_rollout_questions"],3)
        self.assertEqual(result["counters"]["noncontiguous_reappearing_questions"],3)
        self.assertEqual(result["counters"]["additional_question_runs"],12)
        self.assertEqual(result["counters"]["pair_native_exact"],2)
        cw=next(c for c in cases if c["group"]=="correct_wrong")
        self.assertEqual(cw["selection"]["full_chosen"],1)
        self.assertEqual(cw["selection"]["policy_chosen"],2)
        self.assertTrue(cw["selection"]["chosen_sibling_changed"])
        hard=next(c for c in result["cells"] if c["group"]=="all5_wrong")
        self.assertEqual(hard["failure_final_invalid"],3)
        self.assertEqual(hard["failure_valid_gold_mismatch"],2)

    def test_first_source_order_ties_not_sibling_order(self):
        cc=fixtures()[0]
        cc=[cc[i] for i in (3,2,1,0,4)]
        entries=[(r,{"line":i+1},validate_row(r,self.core,self.qa)) for i,r in enumerate(cc)]
        result, expected=analyze_group(entries,self.core,self.distance)
        self.assertEqual(result["reference"]["sibling"],3)
        self.assertEqual(result["reference"]["tied_siblings"],[3,2,1,0,4])
        self.assertEqual(result["selection"]["full_chosen"],2)
        self.assertEqual(result["selection"]["full_max_ties"],[2,1,0,4])
        self.assertEqual(result["selection"]["candidate_mode"],"other_success_fallback")
        self.assertEqual(expected,self.core.mine_pairs(cc)[0])

    def test_reference_search_then_token_tie_and_failure_priority(self):
        rows=fixtures()[1]
        rows[3]=row("fixture-cw",3)
        rows[3]["search_calls"]=0
        rows[3]["token_count"]=200
        rows[0]["token_count"]=1
        entries=[(r,{},validate_row(r,self.core,self.qa)) for r in rows]
        result,_=analyze_group(entries,self.core,self.distance)
        self.assertEqual(result["reference"]["sibling"],3)
        self.assertNotIn(0,result["selection"]["candidate_siblings"])
        self.assertEqual(result["selection"]["candidate_mode"],"failed_first")
        rows[0]["search_calls"]=0;rows[0]["token_count"]=100
        result,_=analyze_group([(r,{},validate_row(r,self.core,self.qa)) for r in rows],self.core,self.distance)
        self.assertEqual(result["reference"]["sibling"],0)

    def test_policy_text_exact_subtraction_and_unicode(self):
        text="<think>中🙂<information>AUTHORED</information>文</think>EXT<answer>gold</answer>"
        first_end=text.index("EXT")
        r=dict(response=text,policy_spans=[[0,first_end],[first_end+3,len(text)]])
        self.assertEqual(policy_only_text(r,self.core.information_spans),"<think>中🙂文</think><answer>gold</answer>")

    def test_duplicate_sibling_and_missing_group_fail(self):
        rows=[r for group in fixtures() for r in group]
        rows[1]=dict(rows[0])
        result,_=self.run_fixture(rows)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollout_duplicate_sibling"],1)
        self.assertGreater(result["errors"]["invalid_question_group_or_native_selection"],0)

    def test_duplicate_and_missing_pairs_fail(self):
        rows=[r for group in fixtures() for r in group]
        pairs=self.core.mine_pairs(rows)
        result,_=self.run_fixture(rows,[pairs[0],pairs[0]])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["pair_duplicate_question"],1)
        self.assertEqual(result["errors"]["missing_eligible_pair"],1)

    def test_pair_selection_mismatch_fails(self):
        rows=[r for group in fixtures() for r in group]
        pairs=self.core.mine_pairs(rows);pairs[1]["student"]="synthetic mismatched student"
        result,_=self.run_fixture(rows,pairs)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["pair_selection_or_identity_mismatch"],1)

    def test_strict_reward_and_invalid_mask_fail(self):
        rows=[r for group in fixtures() for r in group]
        rows[0]["reward"]=0
        rows[2]["policy_spans"]=[[10,2]]
        result,_=self.run_fixture(rows)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollout_strict_reward_mismatch"],1)
        self.assertEqual(result["errors"]["rollout_bad_schema_or_mask"],1)

    def test_partial_final_line_and_oversize_are_hashed_failures(self):
        rows=[r for group in fixtures() for r in group]
        result,_=self.run_fixture(rows,partial=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollouts_partial_final_row"],1)
        self.assertTrue(result["inputs"]["rollouts"]["known_sha_matches"])
        result,_=self.run_fixture(rows,max_line=64)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollouts_oversize_row"],15)
        self.assertTrue(result["inputs"]["rollouts"]["known_sha_matches"])

    def test_structural_defect_does_not_change_final_reward(self):
        r=row("fixture-structure",0)
        # Defect belongs to an earlier authored span; native final action still valid.
        r["response"]=r["response"].replace("</think>","",1)
        r["policy_spans"][0][1]-=8
        r["policy_spans"][1]=[x-8 for x in r["policy_spans"][1]]
        metrics=validate_row(r,self.core,self.qa)
        self.assertEqual(metrics["strict_reward"],1)
        self.assertFalse(metrics["transcript_structure"]["balanced"])

    def test_known_hash_and_stat_changes_reject(self):
        rows=[r for group in fixtures() for r in group]
        result,_=self.run_fixture(rows,wrong_hash=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollouts_known_hash_mismatch"],1)
        result,_=self.run_fixture(rows,change_stat=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollouts_stat_changed"],1)

    def test_bad_pair_types_and_extra_fields_reject(self):
        rows=[r for group in fixtures() for r in group]
        pairs=self.core.mine_pairs(rows);pairs[0]["reference_reward"]=True
        pairs[1]["unexpected_fixture_field"]=1
        result,_=self.run_fixture(rows,pairs)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["pair_bad_schema"],1)
        self.assertEqual(result["errors"]["pair_unexpected_or_missing_field"],1)

    def test_bad_json_hashes_complete_input_and_science_rejects_wrong_source(self):
        rows=[r for group in fixtures() for r in group]
        result,_=self.run_fixture(rows,bad_json=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"]["rollouts_bad_json_row"],1)
        self.assertTrue(result["inputs"]["rollouts"]["known_sha_matches"])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"search_e1/core.py";p.parent.mkdir();p.write_text("# deliberately wrong CPU fixture source\n")
            with self.assertRaisesRegex(ValueError,"Frozen source identity mismatch"):
                load_science(directory)


if __name__=="__main__":
    unittest.main()
