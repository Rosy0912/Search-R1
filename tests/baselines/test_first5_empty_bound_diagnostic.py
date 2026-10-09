"""Manifest-only CPU regressions; not a training or first-five acceptance."""
import hashlib
import json
from unittest.mock import patch
import pytest
from search_r1.baselines import fresh200_gate as gate

EMPTY_SHA=hashlib.sha256(b'').hexdigest()

def manifest(tmp_path,name,bound=True,inside=False,sha=EMPTY_SHA):
    run=tmp_path/'runs'/'continued';directory=run/'checkpoints'/'step5'
    parent=run if inside else tmp_path/'runs'/'original'
    parent.mkdir(parents=True,exist_ok=True);path=parent/name;path.touch()
    originals={str(path):sha}if bound else {}
    with patch.object(gate,'validate_gate_config',return_value=(200,57)),patch.object(gate,'partial_lineage',return_value={'originals':originals}):
        result=gate.write_manifest([path],directory,{'baseline_aligned':True},'a'*40,run,5,[],{'completed_outers':5,'baseline_aligned':True})
    return path,json.loads(result.read_text())

@pytest.mark.parametrize('name',['gpu-exit.err','gpu-exit.out','driver-time.log'])
def test_empty_original_diagnostic_retains_exact_hash_and_zero_bytes(tmp_path,name):
    path,value=manifest(tmp_path,name)
    assert next(x for x in value['files']if x['path']==str(path))==dict(path=str(path),sha256=EMPTY_SHA,bytes=0)
    assert value['version']==3

@pytest.mark.parametrize('name',['rank_0.pt','driver.pt','report.json','audit.jsonl','resolved-config.yaml'])
def test_empty_checkpoint_and_structured_proof_still_rejected_even_bound(tmp_path,name):
    with pytest.raises(ValueError,match='empty checkpoint'):manifest(tmp_path,name)

@pytest.mark.parametrize('inside,bound,sha',[(False,False,EMPTY_SHA),(False,True,'0'*64),(True,False,EMPTY_SHA)])
def test_empty_diagnostic_requires_exact_original_lineage_binding(tmp_path,inside,bound,sha):
    with pytest.raises(ValueError,match='empty checkpoint'):manifest(tmp_path,'gpu-exit.err',bound,inside,sha)
