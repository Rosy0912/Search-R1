import copy, hashlib, json
import pytest
import build

def setup_quality(tmp_path,monkeypatch,correct):
    monkeypatch.setattr(build,'ROOT',tmp_path)
    rows=[dict(uid=str(i),sample_position=i,em=float(i<correct)) for i in range(64)]
    content={'questions.json':[dict(uid=str(i)) for i in range(64)],'qa.jsonl':rows,'result.json':dict(status='COMPLETE64_UNREVIEWED_NO_TRAINING',quality=dict(correct=correct,denominator=64)),'effective-config.json':{},'final-meta.json':{}}
    for n,v in content.items():
        (tmp_path/n).write_text(''.join(json.dumps(x)+'\n' for x in v) if n=='qa.jsonl' else json.dumps(v))
    (tmp_path/'final.safetensors').write_bytes(b'raw tensor fixture; not an actual64 receipt')
    q=dict(status='QUALITY64_RAW_RECOMPUTED_UNREVIEWED',metric='EM',greedy=True,denominator=64,correct=correct,data_sha256='af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756',binding_sha256='binding',raw_files={n:build.ref(tmp_path/n) for n in list(content)+['final.safetensors']})
    return q,dict(data_sha256=q['data_sha256'])

@pytest.mark.parametrize('count,branch',[(26,'OFFICIAL_BASE'),(25,'SEARCH_G1_GRPO100'),(0,'SEARCH_G1_GRPO100')])
def test_actual_quality_branch_boundary(tmp_path,monkeypatch,count,branch):
    q,b=setup_quality(tmp_path,monkeypatch,count);assert build.quality_check(q,b,'binding')==branch

def test_actual_raw_sha_tamper_rejected(tmp_path,monkeypatch):
    q,b=setup_quality(tmp_path,monkeypatch,26);(tmp_path/'final.safetensors').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Raw quality bytes drift'):build.quality_check(q,b,'binding')

def test_actual_false_row_tally_rejected(tmp_path,monkeypatch):
    q,b=setup_quality(tmp_path,monkeypatch,26);q['correct']=27
    with pytest.raises(RuntimeError,match='tally'):build.quality_check(q,b,'binding')

def test_full771_shape_contract():
    shapes=build.shapes();assert len(shapes)==771
    numel=0
    for dims in shapes.values():
        n=1
        for d in dims:n*=d
        numel+=n
    assert numel==32763876352
    assert shapes['model.layers.63.self_attn.k_proj.weight']==[1024,5120]
    assert shapes['model.layers.63.mlp.down_proj.weight']==[5120,27648]

def test_canonical_external_review_excludes_own_link():
    payload=dict(status='REVIEWED',scope='source_quality');h=build.canonical(payload)
    payload['independent_review']=dict(path='external',sha256='externalSHA')
    assert build.canonical({k:v for k,v in payload.items() if k!='independent_review'})==h
    assert build.canonical(payload)!=h
