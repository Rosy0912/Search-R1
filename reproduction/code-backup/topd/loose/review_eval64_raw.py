"""Independent CPU-only raw reconstruction; never runs a model or submits jobs."""
import argparse,collections,hashlib,importlib.util,json,re,string
from pathlib import Path
DATA_SHA='af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756'
BINDING_SHA='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if hasattr(x,'tolist'):return plain(x.tolist())
    if hasattr(x,'item'):return x.item()
    return x

def normalize(x):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in x.lower() if c not in string.punctuation)).split())

def f1(a,b):
    a,b=normalize(a).split(),normalize(b).split()
    return 2*sum((collections.Counter(a)&collections.Counter(b)).values())/(len(a)+len(b)) if a and b else 0.

def spans(tok,ids,mask):
    output=[];i=0
    while i<len(ids):
        j=i+1
        while j<len(ids) and mask[j]==mask[i]:j+=1
        if mask[i]:output.append(tok.decode(ids[i:j]))
        i=j
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation',required=True);parser.add_argument('--binding',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.evaluation).resolve();out=Path(args.output).resolve()
    require(ROOT/'logs' in folder.parents and ROOT/'logs' in out.parents,'Own raw evidence paths only')
    require(not out.exists(),'Never overwrite independent review')
    # Completeness is required before any metrics; missing QA is PENDING, never 0/64.
    required=['questions.json','final.safetensors','final-meta.json','qa.jsonl','result.json','effective-config.json']
    require(all((folder/n).is_file() for n in required),'PENDING_COMPLETE_RAW64')
    require(sha(args.binding)==BINDING_SHA,'Actual source binding drift')
    binding=json.loads(Path(args.binding).read_text());cfg=json.loads((folder/'effective-config.json').read_text())
    data=Path(binding['data_path']);model=Path(binding['model_directory']);require(sha(data)==DATA_SHA,'Original64 bytes drift')
    require(cfg['binding_sha256']==BINDING_SHA and cfg['code_commit']=='413a10c50f32febaafc83db8170e2b005e0e979f' and cfg['data_sha256']==DATA_SHA,'Actual config/source identity')
    expected={'seed':42,'tp':8,'dtype':'bfloat16','eager':True,'quantization':None,'max_turns':4,'start':2048,'rolling':4096,'response':500,'observation':500,'topk':3,'force_final_answer':False,'max_model_len':4596}
    require(all(cfg[k]==v for k,v in expected.items()),'Actual evaluation protocol changed')
    require(cfg['sampling']==dict(n=1,best_of=1,temperature=0.,top_p=1.,top_k=-1,min_p=0.,max_tokens=500,detokenize=False,ignore_eos=False),'Actual greedy sampling changed')
    import torch,pandas as pd
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    require(not torch.cuda.is_initialized(),'CPU reader must remain CUDA-free')
    for n in ['tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','config.json']:
        require(sha(model/n)==binding['model_files'][n],'Actual tokenizer/config bytes changed')
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    require(tok.pad_token_id==cfg['pad'] and tok.eos_token_id==cfg['eos'] and tok.bos_token_id==cfg['bos'],'Tokenizer special-ID drift')
    require(hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest()==cfg['tokenizer_vocab_sha256'],'Token-ID map drift')
    protocol_path=ROOT/'repo-terminal-fixed413/search_r1/qa_protocol.py'
    require(sha(protocol_path)==binding['source_files']['search_r1/qa_protocol.py'],'Terminal protocol source drift')
    spec=importlib.util.spec_from_file_location('actual_terminal_protocol',protocol_path);protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    frame=pd.read_parquet(data);require(len(frame)==64,'Raw original data rows changed');frame=frame.sample(64,random_state=42)
    questions=json.loads((folder/'questions.json').read_text());require(len(questions)==64,'Missing questions')
    for i,(_,row) in enumerate(frame.iterrows()):
        original=plain(row.to_dict());uid=hashlib.sha256(json.dumps(original,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        require(questions[i]==dict(sample_position=i,uid=uid,source=row['data_source'],original_row=original),'Resampled/reordered/modified questions')
    require(len({q['uid'] for q in questions})==64,'Repeated QA identities')
    final=load_file(str(folder/'final.safetensors'),device='cpu');meta=json.loads((folder/'final-meta.json').read_text());qa=[json.loads(l) for l in (folder/'qa.jsonl').read_text().splitlines()];result=json.loads((folder/'result.json').read_text())
    require(result['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING' and result['final_sha256']==sha(folder/'final.safetensors') and result['qa_sha256']==sha(folder/'qa.jsonl'),'Result raw SHA/status drift')
    require(len(qa)==64 and all(t.device.type=='cpu' and t.dtype==torch.long and len(t)==64 for t in final.values()),'Actual64 CPU int64 tensors required')
    width=final['responses'].shape[1];require(width<=4096,'Final rolling response exceeds4096')
    for key in ['attention_mask','info_mask']:require(bool(((final[key]==0)|(final[key]==1)).all()),'Nonbinary final masks')
    require(torch.equal(final['input_ids'],torch.cat([final['prompts'],final['responses']],dim=1)),'Final token assembly changed')
    # Independently rebuild initial and rolling prompts from the original questions.
    rolling=[]
    for i,q in enumerate(questions):
        chat=q['original_row']['prompt']
        text=tok.apply_chat_template(chat,add_generation_prompt=True,tokenize=False) if tok.chat_template else chat[0]['content']
        ids=tok.encode(text,add_special_tokens=False);require(0<len(ids)<=4096,'Original prompt token length changed')
        padded=[tok.pad_token_id]*(4096-len(ids))+ids
        require(final['prompts'][i].tolist()==padded[-2048:],'Final left prompt differs from original question')
        rolling.append([v for v in ids if v!=tok.pad_token_id])
    active=[True]*64;right=[[]for _ in range(64)];authored=[[]for _ in range(64)];turn_counts=[0]*64;search_counts=[0]*64
    turns=sorted(folder.glob('turn-*.json'));generations=sorted(folder.glob('generation-*.json'));require(1<=len(turns)<=4 and len(turns)==len(generations),'Turn budget or generation replay changed')
    for k,(turn_path,gen_path) in enumerate(zip(turns,generations),1):
        turn=json.loads(turn_path.read_text());gen=json.loads(gen_path.read_text());gt=load_file(str(gen_path.with_suffix('.safetensors')),device='cpu')
        require(turn['turn']==k and gen['call']==k and gen['tensors_sha256']==sha(gen_path.with_suffix('.safetensors')),'Generation source/call mismatch')
        require(turn['active_mask']==active,'Turn active-set drift')
        indices=[i for i,a in enumerate(active)if a];n=len(indices);padded=n+(8-n%8)%8
        require(len(gen['raw_response_token_ids'])==padded and len(gen['prompt_token_ids'])==padded and len(gen['results'])==padded,'Actual DP8 duplicates/count changed')
        for j in range(padded):
            source_index=indices[j] if j<n else indices[0]
            require(gen['prompt_token_ids'][j]==rolling[source_index],'Actual generation prompt differs from independently reconstructed question/search history')
            raw=gen['raw_response_token_ids'][j];require(len(raw)<=500 and gen['results'][j]['generated_tokens']==len(raw),'Generated length changed')
            require(gt['responses'][j,:len(raw)].tolist()==raw and bool((gt['responses'][j,len(raw):]==tok.pad_token_id).all()),'Raw model tokens differ from generation tensor')
            p=gt['prompts'][j].tolist();p=p[next(i for i,v in enumerate(p)if v!=tok.pad_token_id):];require(p==gen['prompt_token_ids'][j],'Actual prompt mismatch')
        predictions=['']*64
        for j,i in enumerate(indices):
            text=tok.decode(gen['raw_response_token_ids'][j],skip_special_tokens=True)
            text=text.split('</search>')[0]+'</search>' if '</search>' in text else text.split('</answer>')[0]+'</answer>' if '</answer>' in text else text
            predictions[i]=text;turn_counts[i]+=1
        require(predictions==turn['predictions'],'Executor text not actual processed model tokens')
        obs=tok(turn['observations'],padding='longest',return_tensors='pt',add_special_tokens=False)['input_ids'][:,:500]
        for i in range(64):
            resp=[v for v in tok.encode(predictions[i],add_special_tokens=False)if v!=tok.pad_token_id];observation=[v for v in obs[i].tolist()if v!=tok.pad_token_id]
            rolling[i]=(rolling[i]+resp+observation)[-4096:]
            right[i]=(right[i]+resp+observation)[:4096];authored[i]=(authored[i]+[1]*len(resp)+[0]*len(observation))[:4096]
            action=re.search(r'<(search|answer)>(.*?)</\1>',predictions[i],re.DOTALL);kind=action.group(1) if action else None
            require(turn['is_search'][i]==int(active[i] and kind=='search') and turn['valid_action'][i]==int(active[i] and kind in ('search','answer')) and turn['dones'][i]==int(not active[i] or kind=='answer'),'Actual executor flags changed')
            search_counts[i]+=turn['is_search'][i];active[i]=active[i] and not turn['dones'][i]
        search=folder/f'search-{k:02d}.json'
        if search.exists():
            sr=json.loads(search.read_text());require(sr['request']['topk']==3 and len(sr['response']['result'])==sum(turn['is_search']) and all(len(v)==3 for v in sr['response']['result']),'Actual HTTP top3/count drift')
    require(meta['turns_stats']==turn_counts and meta['active_mask']==active and meta['valid_search_stats']==search_counts,'Final meta differs from actual turn trace')
    rows=[]
    for i,q in enumerate(questions):
        r=final['responses'][i].tolist();mask=(final['attention_mask'][i,-width:] & final['info_mask'][i,-width:]).tolist()
        expected_ids=right[i]+[tok.pad_token_id]*(width-len(right[i]));expected_mask=authored[i]+[0]*(width-len(authored[i]))
        require(r==expected_ids and mask==expected_mask,'Observation/model information mask or final tokens differ')
        decoded=spans(tok,r,mask);answer=protocol.extract_final_answer(decoded);targets=q['original_row']['reward_model']['ground_truth']['target'];targets=[targets]if isinstance(targets,str)else targets
        em=float(answer is not None and any(normalize(answer)==normalize(t) for t in targets));score=max((f1(answer,t)for t in targets),default=0.)if answer is not None else 0.
        record=qa[i];require(record['sample_position']==i and record['uid']==q['uid'] and record['model_spans']==decoded and record['answer']==answer and record['em']==em and abs(record['f1']-score)<1e-12 and record['generated_tokens']==sum(mask),'Stored QA does not equal independent terminal token decode/EM/F1')
        require(record['turns']==turn_counts[i] and record['unfinished']==active[i] and record['actual_search_calls']==search_counts[i],'Stored QA metadata drift')
        rows.append(dict(sample_position=i,uid=q['uid'],em=em,f1=score,answer_present=answer is not None,generated_tokens=sum(mask),actual_search_calls=search_counts[i]))
    correct=sum(int(v['em'])for v in rows);require(result['quality']['correct']==correct and result['quality']['denominator']==64 and result['quality']['candidate_quality_accepted']==(correct>=26),'Stored quality threshold mismatch')
    require(not torch.cuda.is_initialized(),'Reader initialized CUDA')
    payload=dict(status='INDEPENDENT_RAW64_RECONSTRUCTION_PASS_NOT_TRAINING_APPROVAL',correct=correct,denominator=64,metric='EM',greedy=True,data_sha256=DATA_SHA,candidate_quality_accepted=correct>=26,next_path='TOPD_TEACHER_BACKEND_VALIDATION'if correct>=26 else 'SEARCH_G1_GRPO100',f1_mean=sum(v['f1']for v in rows)/64,actual_turns=len(turns),rows=rows,raw_files={p.name:sha(p)for p in folder.iterdir()if p.is_file()and p.suffix in ('.json','.jsonl','.safetensors')},parent_cuda_initialized=False,torch_version=torch.__version__)
    with out.open('x')as f:json.dump(payload,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in payload.items()if k not in ('rows','raw_files')}))

if __name__=='__main__':main()
