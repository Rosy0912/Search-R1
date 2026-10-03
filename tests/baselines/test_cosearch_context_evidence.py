"""Oversized full-candidate inputs must be preserved rather than truncated."""
import json
import pytest
from search_r1.baselines.cosearch import check_candidate_context


def test_context_failure_keeps_exact_full_candidates_and_ids(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv('SEARCH_R1_BASELINE_OUTPUT',str(tmp_path))
    docs=[[dict(document=dict(id=str(j),contents='Original full passage '+str(j))) for j in range(50)] for _ in range(2)]
    encoded=[[1]*20,list(range(40))]
    with pytest.raises(ValueError,match='context exceeds model limit'):
        check_candidate_context(encoded,4,32,['first','second'],['q1','q2'],docs)
    files=list((tmp_path/'ranker-context-failures').glob('*.json'));assert len(files)==1
    record=json.loads(files[0].read_text())
    assert record['rows']==[1] and record['prompt_tokens']==[40]
    assert record['cases']==[dict(row=1,question='second',query='q2',documents=docs[1],input_ids=encoded[1])]
    assert docs[1][-1]['document']['contents']=='Original full passage 49'
    assert 'COSEARCH_CONTEXT_EXCEEDED_JSON=' in capsys.readouterr().out


def test_exact_context_boundary_does_not_change_inputs_or_create_failure(tmp_path,monkeypatch):
    monkeypatch.setenv('SEARCH_R1_BASELINE_OUTPUT',str(tmp_path))
    encoded=[list(range(28))];docs=[[dict(document=dict(contents='full'))]*50]
    check_candidate_context(encoded,4,32,['q'],['query'],docs)
    assert encoded==[list(range(28))] and not (tmp_path/'ranker-context-failures').exists()
