"""Read-only h1009 CPU diagnostic: existing row metadata and official tokenizer.

No model/PT/GPU/training, no filtering or changing historical raw/config/source.
Print only parameter metadata and seven failed row diagnostic counts/hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
assert socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
starts=time.perf_counter(),time.process_time()
plan_path=ROOT/'instruct300/preparation-f7-v8-20261006/HISTORICAL_RAW_CPU_PLAN.json'
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()=='7227b2586bc584d05613b8b5c5d54657812ff1978f435d8916e60faa00443bda'
plan=json.loads(plan_path.read_bytes())
report_path=ROOT/'instruct300/candidates/fresh200-mu-f7-v8/CPU_PREP_CONFIG/HISTORICAL_RAW_CPU_PREFLIGHT.json'
report_raw=report_path.read_bytes()
assert hashlib.sha256(report_raw).hexdigest()=='958d592e11f12233241ae3da121764fbdc011bb728066aa062c7e70edfdcc430'
report=json.loads(report_raw)
failed=[r for r in report['records'] if r['job_id']=='3295' and r['status'].endswith('REJECTED')]
assert len(failed)==7
from transformers import AutoTokenizer
import torch
assert not torch.cuda.is_initialized()
model=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
assert hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest()=='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
by_rank={}
for record in failed:by_rank.setdefault(record['rank'],[]).append(record)
result=[];first_parameters=None
for rank,coordinates in sorted(by_rank.items()):
    entry=plan['runs'][0]['rank_ready'][rank]
    raw=Path(entry['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    ready=json.loads(raw)
    if first_parameters is None:first_parameters={'before':ready['sampling_params_before'],'after':ready['sampling_params_after']}
    generated={r['input_row']:r for r in ready['observer']['generated']}
    for coord in coordinates:
        r=generated[coord['row']];ids=r['token_ids']
        full=tokenizer.decode(ids,skip_special_tokens=True)
        first=None;restored=None
        for stop in range(1,len(ids)+1):
            prefix=tokenizer.decode(ids[:stop],skip_special_tokens=True)
            if first is None and '\ufffd' in prefix:first=stop
            if first is not None and stop>first and '\ufffd' not in prefix:
                restored=stop;break
        result.append({'job_id':'3295','rank':rank,'row':coord['row'],'request_id':r['request_id'],
            'raw_emitted_length':len(ids),'finish_reason':r['finish_reason'],'full_has_replacement':'\ufffd' in full,
            'first_replacement_prefix_tokens':first,'next_valid_prefix_tokens':restored,
            'full_text_sha256':hashlib.sha256(full.encode()).hexdigest(),
            'raw_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'READ_ONLY_HISTORICAL_UTF8_DIAGNOSTIC_NOT_SOURCE_OR_TRAINING_ACCEPTANCE',
    'node':socket.gethostname(),'source_report_sha256':hashlib.sha256(report_raw).hexdigest(),
    'decoder_state':json.loads(tokenizer.backend_tokenizer.decoder.__getstate__()),
    'cleanup_spaces':tokenizer.clean_up_tokenization_spaces,'rows':result,'original_sampling_parameters':first_parameters,
    'wall_seconds':time.perf_counter()-starts[0],'process_cpu_seconds':time.process_time()-starts[1],
    'cuda_initialized':False,'optimizer_writes':0,'generation_calls':0,'model_weight_loads':0,'tensor_asset_loads':0},ensure_ascii=False))
