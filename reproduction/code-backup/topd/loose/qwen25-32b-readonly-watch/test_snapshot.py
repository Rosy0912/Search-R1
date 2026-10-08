import importlib.util
from pathlib import Path
import json
import types
import pytest
p=Path(__file__).with_name('snapshot.py');spec=importlib.util.spec_from_file_location('readonly_snapshot',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('state',['RUNNING','FAILED'])
def test_raw_status_and_actual_counts(tmp_path,monkeypatch,state):
    root=tmp_path/'root';out=root/'logs/qwen25-32b-eval64-123/evaluation';out.mkdir(parents=True)
    (out/'qa.jsonl').write_text('{}\n{}\n');(out/'generation-01.json').write_text('{}');(out/'readiness.json').write_text('{}')
    monkeypatch.setattr(m,'ROOT',root)
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return types.SimpleNamespace(stdout='JobId=123 JobState='+state,stderr='',returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run)
    actual=m.observe(123)
    assert calls==[['scontrol','show','job','-dd','123']]
    assert actual['qa_rows']==2 and actual['generation_files']==['generation-01.json']
    assert actual['final_state_observed']==(state=='FAILED')
    saved=json.loads((root/'logs/qwen25-32b-readonly-watch-123.jsonl').read_text())
    assert saved['scontrol_stdout']=='JobId=123 JobState='+state
