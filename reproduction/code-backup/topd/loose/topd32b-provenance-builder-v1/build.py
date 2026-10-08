"""CPU-only source/quality assembly; independent review is external; never submits."""
import argparse, hashlib, importlib.util, json, os, socket, time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
REVISION='1818d35814b8319459f4bd55ed1ac8709630f003'
METADATA='e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'
PARAM='a80140df56c39e1e0f99e29473bd1b6f3668d47cccfa05240514f560705d0d16'
BUFFER='b778c6814b0f038120e64a78d067aff5c4d65c1c15e7824a7a9b6f632b56f924'
CPU_RECEIPT='d261a5537a98eab95d45dc7db0fd5cb8b56a8930f756524743fe704917c74948'
GATE_SHA='ab6324524187ef70c213709fcde105e01444b6c4a5a15d7abf65005313382e84'

def need(x,msg):
    if not x:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def canonical(p):return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ref(p):
    p=Path(p).resolve();need(ROOT in p.parents,'OwnROOT evidence only');return dict(path=str(p),sha256=sha(p))
def save(p,d):
    p=Path(p);need(ROOT/'logs' in p.resolve().parents,'Ownlogs publication only')
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def gate():
    p=ROOT/'tools/topd-32b-native-score-gate-v6/score_gate.py';need(sha(p)==GATE_SHA,'Original canonical gate drift')
    spec=importlib.util.spec_from_file_location('canonical_gate',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def shapes():
    d={'model.embed_tokens.weight':[152064,5120],'model.norm.weight':[5120],'lm_head.weight':[152064,5120]}
    for i in range(64):
        p='model.layers.%d.'%i
        for n in ['input_layernorm','post_attention_layernorm']:d[p+n+'.weight']=[5120]
        for n,out in [('q_proj',5120),('k_proj',1024),('v_proj',1024)]:
            d[p+'self_attn.'+n+'.weight']=[out,5120];d[p+'self_attn.'+n+'.bias']=[out]
        d[p+'self_attn.o_proj.weight']=[5120,5120]
        for n,s in [('gate_proj',[27648,5120]),('up_proj',[27648,5120]),('down_proj',[5120,27648])]:d[p+'mlp.'+n+'.weight']=s
    return d

def quality_check(q,b,binding_sha):
    need(q['status']=='QUALITY64_RAW_RECOMPUTED_UNREVIEWED' and q['metric']=='EM' and q['greedy'] is True and q['denominator']==64,'Complete original greedy64 recomputation required')
    need(q['data_sha256']==b['data_sha256']=='af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756','Data identity mismatch')
    need(type(q['correct']) is int and 0<=q['correct']<=64,'Integer correct count required')
    need(q['binding_sha256']==binding_sha,'Quality binding SHA mismatch')
    required={'questions.json','final.safetensors','final-meta.json','qa.jsonl','result.json','effective-config.json'}
    need(set(q['raw_files'])==required,'All raw quality evidence required')
    for r in q['raw_files'].values():need(ref(r['path'])==r,'Raw quality bytes drift')
    qs=json.loads(Path(q['raw_files']['questions.json']['path']).read_text()); rows=[json.loads(x) for x in Path(q['raw_files']['qa.jsonl']['path']).read_text().splitlines()]
    need(len(qs)==len(rows)==64 and len({x['uid'] for x in qs})==64,'Exactly64 unique rows required')
    need([x['sample_position'] for x in rows]==list(range(64)) and [x['uid'] for x in rows]==[x['uid'] for x in qs],'Rows/order mismatch')
    need(all(type(x['em']) in (int,float) and x['em'] in (0,1) for x in rows) and sum(int(x['em']) for x in rows)==q['correct'],'Actual row tally mismatch')
    result=json.loads(Path(q['raw_files']['result.json']['path']).read_text())
    need(result['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING' and result['quality']['correct']==q['correct'] and result['quality']['denominator']==64,'Actual completed result mismatch')
    return 'OFFICIAL_BASE' if q['correct']>=26 else 'SEARCH_G1_GRPO100'

def fresh_inventory(b,g):
    model=Path(b['model_directory']).resolve();need(model==ROOT/'assets-root/assets/qwen2.5-32b','Official own model path required')
    metadata_path=Path(b['official_metadata_path']);need(ref(metadata_path)['sha256']==METADATA and b['official_metadata_sha256']==METADATA and b['revision']==REVISION,'Official revision/metadata mismatch')
    metadata=json.loads(metadata_path.read_text());need(metadata['id']=='Qwen/Qwen2.5-32B' and metadata['sha']==REVISION,'Official model identity mismatch')
    official={x['rfilename']:x for x in metadata['siblings']};names=set(b['model_files']);shards=sorted(n for n in names if n.endswith('.safetensors'))
    inv=dict(status='FULL_HF_FILES_VERIFIED',model_directory=str(model),shards=shards,files={})
    need(len(shards)==17 and len(names)==24,'Official24/17 inventory required');g.check_hf_file_set(dict(inv,files={n:{} for n in names}),model,'OFFICIAL_QWEN2_5_32B_BASE')
    signatures={}
    for n in sorted(names):
        p=model/n;need(not p.is_symlink(),'Symlink model rejected');s=p.stat();sig=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns);h=sha(p);after=p.stat()
        need(sig==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Changed during fresh hash')
        row=official[n];need(s.st_size==row['size'] and h==b['model_files'][n],'Fresh HF/binding mismatch')
        if 'lfs' in row:need(h==row['lfs']['sha256'],'Official LFS mismatch')
        else:
            raw=p.read_bytes();need(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['blobId'],'Official Git blob mismatch')
        inv['files'][n]=dict(size=s.st_size,sha256=h);signatures[n]=sig
    from safetensors import safe_open
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];actual={}
    for n in shards:
        with safe_open(str(model/n),framework='pt',device='cpu') as f:
            for k in f.keys():
                need(k not in actual and index.get(k)==n and f.get_slice(k).get_dtype()=='BF16','Duplicate/unindexed/dtype mismatch');actual[k]=f.get_slice(k).get_shape()
    need(len(index)==len(actual)==771 and actual==shapes(),'Exact official771 tensor shapes differ')
    g.check_hf_tensor_index(model,shards);g.check_tokenizer_config_identity(model)
    for n,sig in signatures.items():
        s=(model/n).stat();need(sig==(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns),'HF changed during inventory')
    inv.update(hostname=socket.gethostname(),pid=os.getpid(),epoch=time.time(),header_keys=771,tensor_shapes_sha256=canonical(actual),parameter_count=g.PARAMETERS,official_revision=REVISION,official_metadata_sha256=METADATA,scope='FRESH_H11_CPU_SOURCE_ONLY_NO_GPU_NO_TRAINING')
    return inv

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');
    for n in ['binding','quality','cpu-identity','cpu-review','output-directory']:a.add_argument('--'+n,required=True)
    a=sub.add_parser('finalize');
    for n in ['draft','independent-review','output']:a.add_argument('--'+n,required=True)
    a=p.parse_args();need(socket.gethostname().split('.')[0] in ('h10011','h100-11'),'Only native h11');need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only CVDempty required')
    if a.command=='finalize':
        d=json.loads(Path(a.draft).read_text());need(d['status']=='PENDING_INDEPENDENT_REVIEW','Only pending draft');payload=d['review_target_payload'];review=json.loads(Path(a.independent_review).read_text())
        need(review['status']=='PASS' and review['approval_scope']=='32B_TEACHER_SOURCE_AND_QUALITY_ONLY' and review['reviewed_payload_sha256']==canonical(payload),'External independent review mismatch')
        payload['independent_review']=ref(a.independent_review);gate().check_identity(payload,ROOT/'tools/topd-32b-pipeline-candidate');save(a.output,payload);print('CANONICAL_PROVENANCE_SOURCE_QUALITY_ONLY_NO_GPU_ACCEPTANCE');return
    canonical_binding_file_sha=ref(a.binding)['sha256'];b=json.loads(Path(a.binding).read_text());q=json.loads(Path(a.quality).read_text());branch=quality_check(q,b,canonical_binding_file_sha)
    out=Path(a.output_directory).resolve();need(ROOT/'logs' in out.parents and out.is_dir(),'Existing ownlogs output directory required')
    if branch!='OFFICIAL_BASE':save(out/'decision.json',dict(scope='QUALITY_BRANCH_ONLY_NO_TEACHER_PUBLICATION',correct=q['correct'],denominator=64,next_path='SEARCH_G1_GRPO100',quality_result=ref(a.quality),binding=ref(a.binding)));print('SEARCH_G1_GRPO100_REQUIRED_NO_SECOND_CUTOFF');return
    cpu_ref=ref(a.cpu_identity);need(cpu_ref['sha256']==CPU_RECEIPT,'Original3027 receipt bytes required');cpu=json.loads(Path(a.cpu_identity).read_text());review=json.loads(Path(a.cpu_review).read_text());s=cpu['snapshot']
    need(review['status']=='PASS' and review['approval_scope']=='OFFICIAL_32B_CPU_FP32_IDENTITY_ONLY_NOT_TEACHER_QUALITY' and review['reviewed_receipt_sha256']==CPU_RECEIPT and review['parameter_sha256']==PARAM and review['buffer_sha256']==BUFFER,'CPU independent source review mismatch')
    need(cpu['status']=='CPU_FP32_IDENTITY_CAPTURE_UNREVIEWED_NOT_TEACHER_QUALITY' and cpu['hostname']=='h100-11' and cpu['cuda_initialized'] is False and cpu['model_directory']==b['model_directory'] and cpu['hf_file_sha256']==b['model_files'],'Actual original CPU source mismatch')
    need(s['parameter_sha256']==PARAM and s['buffer_sha256']==BUFFER and s['parameter_count']==32763876352 and s['parameter_bytes']==131055505408 and s['parameter_devices']==s['buffer_devices']==['cpu'] and s['training_modules']==s['requires_grad_parameters']==s['grad_parameters']==0,'Actual fullFP32 identity mismatch')
    g=gate();inv=fresh_inventory(b,g);save(out/'model-inventory.json',inv)
    payload=dict(status='REVIEWED',scope='REVIEWED_32B_TEACHER_SOURCE_AND_QUALITY_NOT_GPU_ACCEPTANCE',teacher_kind='OFFICIAL_QWEN2_5_32B_BASE',official_revision=REVISION,official_metadata_path=b['official_metadata_path'],official_metadata_sha256=METADATA,data_sha256=b['data_sha256'],expected_fp32_parameter_sha256=PARAM,expected_fp32_buffer_sha256=BUFFER,quality_result=ref(a.quality),model_inventory=ref(out/'model-inventory.json'),cpu_identity=cpu_ref,cpu_identity_independent_review=ref(a.cpu_review),evaluation_binding=ref(a.binding),candidate_files=g.CANDIDATE_FILES)
    save(out/'provenance-draft.json',dict(status='PENDING_INDEPENDENT_REVIEW',scope='NOT_USABLE_AS_PROVENANCE',review_target_payload=payload,review_target_payload_sha256=canonical(payload)))
    print('DRAFT_ONLY_EXTERNAL_SOURCE_QUALITY_REVIEW_REQUIRED_NO_GPU')
if __name__=='__main__':main()
