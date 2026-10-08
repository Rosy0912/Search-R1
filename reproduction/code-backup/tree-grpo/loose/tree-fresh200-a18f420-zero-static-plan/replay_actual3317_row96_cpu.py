"""Same-h100 CPU replay of exactly the retained FAILED3317 offending sample.

Only original UTF-8 prefixes, official tokenizer identity and sample immutability
are checked. No action parser, generation, weights, tensors or training acceptance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import time
import types


ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
MODEL=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
COMMIT='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
HELPER_SHA='5ebffa13473b111f5b73992679370d1c90603f2f9da5bdfee52db8b5736750c6'
TOKENIZER_SHA='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
REJECTION=ROOT/'runs/tree-fresh200-3317/output/token-preservation-rejections/5d85e5687d954bdaa0a1043052dbd8dc.json'
REJECTION_SHA='91237cb19325f3f8cc5467a550fffd73345b7b41da921065ae0a56d8c57e41fb'
REJECTION_BYTES=10115714
ROW=96
SAMPLE_ID='4a659f5a308a597087edf7206b2c01d09862285fb77b01f04d58d43670cd9259'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def local_path(value):
    path=Path(value)
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve()),'Own same-node path required')
    return path


def bound_bytes(entry):
    path=local_path(entry['path'])
    require(set(entry)=={'path','sha256','bytes'} and type(entry['bytes']) is int and entry['bytes']>0,
            'Exact ordinary-file SHA and byte count required')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size==entry['bytes'],'Bound ordinary file size differs')
        raw=stream.read();after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Bound file changed during read')
    require(sha(raw)==entry['sha256'],'Bound file SHA differs')
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    require(socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Own h100-9 CPU-only replay requires explicitly empty CUDA visibility')
    plan_path=local_path(args.plan)
    plan_raw=bound_bytes(dict(path=str(plan_path),sha256=args.plan_sha,bytes=plan_path.stat().st_size))
    plan=json.loads(plan_raw)
    require(plan['replay_stage_directory']==str(plan_path.parent),'Own standalone replay stage differs')
    require(plan['code_commit']==COMMIT and plan['sample_selector']==dict(row_index=ROW,sample_id=SAMPLE_ID,
            emitted_tokens=500,native_emissions_in_rejection=784,first_literal_fffd_token_prefix=159),
            'Predetermined unfiltered offending sample selector changed')
    require(plan['official_model_path']==str(MODEL) and plan['official_tokenizer_json_sha256']==TOKENIZER_SHA,
            'Official same-node tokenizer binding differs')
    require(plan['rejection']==dict(path=str(REJECTION),sha256=REJECTION_SHA,bytes=REJECTION_BYTES),
            'Exact retained FAILED3317 rejection binding differs')
    output=local_path(args.output)
    require(plan['output']==str(output) and output.parent==plan_path.parent and not output.exists(),
            'Own no-clobber report path differs')
    producer=plan['producer']
    require(local_path(producer['path']).resolve()==Path(__file__).resolve(), 'Replay producer path differs')
    bound_bytes(producer)
    entry=plan['raw_helper'];source=local_path(entry['path'])
    require(entry['sha256']==HELPER_SHA and source==plan_path.parent/'tree_raw_tokens.py',
            'Exact standalone staged raw-helper path/SHA required')
    helper_raw=bound_bytes(entry)
    helper=types.ModuleType('bound_actual3317_utf8_helper');helper.__file__=str(source)
    exec(compile(helper_raw,str(source),'exec'),helper.__dict__)
    from transformers import AutoTokenizer
    import torch
    require(not torch.cuda.is_initialized(),'CPU tokenizer import initialized CUDA')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    identity=helper.tokenizer_identity(tokenizer,TOKENIZER_SHA)
    require(tokenizer.pad_token_id==151643 and tokenizer.eos_token_id==151645,'Official PAD/EOS changed')
    rejection=json.loads(bound_bytes(plan['rejection']))
    require(rejection['status']=='CURRENT_ROLLOUT_REJECTED_BEFORE_RETURN'
            and rejection['probability_accepted'] is False and rejection['tokenizer_identity']==identity,
            'Historical rejection or exact live tokenizer identity differs')
    context=rejection['failure_context'];emissions=context['native_emissions']
    require(type(emissions) is list and len(emissions)==784,'Retained rejected batch identity differs')
    emission=emissions[ROW];ids=emission['raw_emitted_token_ids'];values=emission['raw_selected_logprobs']
    prompt=context['contexts'][ROW]
    require(emission['sample_id']==SAMPLE_ID and emission['sample_id']==helper.fingerprint(emission['sampling_identity'])
            and emission['schema_version']==1 and emission['protocol']==helper.PROTOCOL and emission['finished'] is True,
            'Original sample identity/protocol/finality differs')
    require(type(emission['true_emitted_length']) is int and emission['true_emitted_length']==500
            and type(ids) is list and len(ids)==500 and all(type(token) is int for token in ids)
            and type(values) is list and len(values)==500
            and all(type(value) in (int,float) and math.isfinite(value) for value in values)
            and emission['prompt_token_ids']==prompt,'Original IDs/scalars/context differ')
    frozen=dict(emission_sha256=helper.fingerprint(emission),ids_sha256=helper.fingerprint(ids),
        selected_logprobs_sha256=helper.fingerprint(values),context_sha256=helper.fingerprint(prompt),
        legacy_text_sha256=helper.fingerprint(context['legacy_text'][ROW]))
    started=time.perf_counter(),time.process_time()
    decoded,complete,proof=helper._scan_utf8_prefixes(tokenizer,ids,identity,check_native_prefixes=True)
    first_literal=next((stop for stop,text in enumerate(decoded,1) if '\ufffd' in text),None)
    special=set(identity['utf8_decoder_binding']['skip_special_token_ids'])
    original_bytes=b''.join(bytes(helper.BYTELEVEL_INVERSE[char] for char in tokenizer.convert_ids_to_tokens(token))
        for token in ids if token not in special)
    require(first_literal==159 and b'\xef\xbf\xbd' in original_bytes and complete[-1] is True
            and proof['final_utf8_complete'] is True and proof['native_prefix_agreement_count']==500
            and proof['original_emitted_byte_sha256']==sha(original_bytes),
            'Original legal literal U+FFFD and all500 native prefixes not proven')
    after=dict(emission_sha256=helper.fingerprint(emission),ids_sha256=helper.fingerprint(ids),
        selected_logprobs_sha256=helper.fingerprint(values),context_sha256=helper.fingerprint(prompt),
        legacy_text_sha256=helper.fingerprint(context['legacy_text'][ROW]))
    require(frozen==after, 'Original sample mutated during replay')
    require(helper.tokenizer_identity(tokenizer,TOKENIZER_SHA)==identity
            and not torch.cuda.is_initialized(),'Live tokenizer identity/CUDA changed during replay')
    report=dict(status='HISTORICAL_RAW_CPU_PREFLIGHT_OBSERVED_NOT_FRESH_TRAINING_ACCEPTED',
        code_commit=COMMIT,tokenizer_identity=identity,plan_sha256=args.plan_sha,producer_sha256=producer['sha256'],
        raw_helper_sha256=HELPER_SHA,rejection_sha256=REJECTION_SHA,rows=1,rejected_rows=0,
        scope='Exactly predetermined unfiltered historical FAILED3317 row96: live official tokenizer identity, all500 original-byte UTF8/native prefixes and IDs/logprobs/context immutability only.',
        sample_selector=plan['sample_selector'],original_sample_hashes_before=frozen,original_sample_hashes_after=after,
        utf8_proof_sha256=helper.fingerprint(proof),original_emitted_byte_sha256=sha(original_bytes),
        native_prefix_agreement_count=500,native_prefix_agreement_sha256=proof['native_prefix_agreement_sha256'],
        final_native_text_sha256=proof['final_native_text_sha256'],first_literal_fffd_token_prefix=first_literal,
        valid_literal_utf8_efbfbd_present=True,final_strict_original_bytes_complete=True,final_pending_byte_count=0,
        wall_seconds=time.perf_counter()-started[0],cpu_seconds=time.process_time()-started[1],
        model_weight_loads=0,tensor_asset_loads=0,generation_calls=0,optimizer_writes=0,cuda_initializations=0,
        cuda_initialized=False,original_sampler_actor_gate_accepted=False,fresh_tree_rollout_accepted=False,
        actual_native24_accepted=False,actual_behavior_mu_accepted=False,actual_full_tree_accepted=False,
        limitations='Only one historical sampled emission UTF8/identity proof. No sampled_action/legacy action-parser, all784 rows, historical5120 rows, native24 sampling support, fresh Tree/score-cap, mu, weights or training acceptance. Original sample is neither repaired nor filtered.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(status=report['status'],rows=1,rejected_rows=0,path=str(output),
        sha256=sha(output.read_bytes()),raw_helper_sha256=HELPER_SHA,actual_gpu_accepted=False)),flush=True)


if __name__=='__main__':main()
