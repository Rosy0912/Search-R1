"""Independent additional strict UTF8 boundary matrix; synthetic CPU only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import tempfile

HERE=Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    fixture=runpy.run_path(str(HERE/'UTF8_INDEPENDENT_PROBE.py'))
Tokenizer=fixture['SyntheticTokenizer'];raw=fixture['raw']

def check(tokenizer,identity,data,parts,tagged):
    emitted=list(range(1000,1000+len(parts)))+[151645]
    tokenizer.pieces.update(dict(zip(emitted[:-1],parts)))
    text=data.decode('utf8');canonical=raw.canonical_text(text)
    tokenizer.pieces[300]=canonical.encode('utf8')
    tokenizer.encode=lambda text,add_special_tokens=False: [300]
    emission=dict(protocol=raw.PROTOCOL,schema_version=1,raw_emitted_token_ids=emitted,
        true_emitted_length=len(emitted),raw_selected_logprobs=[-.25*(i+1) for i in range(len(emitted))],
        sampling_identity={'synthetic_matrix':True},prompt_token_ids=[42])
    emission['sample_id']=raw.fingerprint(emission['sampling_identity'])
    action,ledger=raw.sampled_action(tokenizer,emission,emitted+[151643]*(500-len(emitted)),[300],canonical,identity)
    raw.validate_action_utf8_proof(ledger,tokenizer)
    # Independent complete byte-boundary and canonical cut computation.
    prefixes=[];aggregate=b'';boundary=[]
    for i,piece in enumerate(parts+[b'']):
        aggregate+=piece
        try:decoded=aggregate.decode('utf8');boundary.append(i+1);prefixes.append((i+1,decoded))
        except UnicodeDecodeError:pass
    cut=next(i for i,text in prefixes if text.startswith(canonical)) if tagged else len(emitted)
    assert action==emitted[:cut] and ledger['selected_logprobs']==emission['raw_selected_logprobs'][:cut]
    assert ledger['utf8_boundary_proof']['complete_original_token_boundaries']==boundary
    return dict(original_tokens=len(emitted),cut=cut,pending_prefixes=len(ledger['utf8_boundary_proof']['pending_prefix_events']))

results=[]
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp);(path/'tokenizer.json').write_text('{"strictly_synthetic":true}')
    tokenizer=Tokenizer(path);sha=hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest()
    identity=raw.tokenizer_identity(tokenizer,sha)
    samples=['\x00','\x7f','\u0080','\u07ff','\u0800','\ud7ff','\ue000','\ufffc','\ufffe','\uffff','\U00010000','\U0010ffff','é中😀']
    for text in samples:
        data=text.encode('utf8')
        for mask in range(1 << (len(data)-1)):
            boundaries=[i+1 for i in range(len(data)-1) if mask & (1<<i)]+[len(data)]
            starts=[0]+boundaries[:-1];parts=[data[a:b] for a,b in zip(starts,boundaries)]
            case=check(tokenizer,identity,data,parts,False)
            results.append(dict(text_utf8_hex=data.hex(),partition_boundaries=boundaries,**case))
    # Whole-token boundary cannot end inside a suffix scalar after a canonical tag.
    for suffix in ('é','中','😀'):
        tail=suffix.encode('utf8');parts=[b'<search>x</search>'+tail[:1],tail[1:],b'tail']
        case=check(tokenizer,identity,b''.join(parts),parts,True)
        assert case['cut']==2
        results.append(dict(tagged=True,suffix_utf8_hex=tail.hex(),**case))
print(json.dumps(dict(status='INDEPENDENT_STRICT_UTF8_BOUNDARY_MATRIX_PASSED',cases=len(results),
    actual_gpu_executed=False,actual_remote_executed=False,actual_official_assets_loaded=False,
    source_sha256=hashlib.sha256(fixture['RAW'].read_bytes()).hexdigest(),matrix=results),indent=2,sort_keys=True))
