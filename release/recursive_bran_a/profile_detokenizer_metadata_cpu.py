"""CPU-only comparison of the actual vLLM incremental decoder, synthetic IDs."""
import copy,time,cProfile,pstats,io,json,hashlib,argparse
from transformers import AutoTokenizer
from vllm.transformers_utils.tokenizer import get_cached_tokenizer
from vllm.transformers_utils.detokenizer import detokenize_incrementally
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
original=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
cached=get_cached_tokenizer(copy.deepcopy(original))
ids=original.encode('We should search for the exact answer. <search>capital of France</search> '*32,add_special_tokens=False)[:192]
result={}
for label,t in [('original',original),('engine_cached_copy',cached)]:
 profile=cProfile.Profile();profile.enable();start=time.perf_counter();texts=[]
 for row in range(32):
  previous=None;prefix=read=0;text=''
  for n in range(1,len(ids)+1):
   new,decoded,prefix,read=detokenize_incrementally(t,ids[:n],previous,prefix,read,True,True)
   previous=(previous or [])+new;text+=decoded
  texts.append(text)
  print(json.dumps(dict(stage=label,complete_rows=row+1,seconds=time.perf_counter()-start)),flush=True)
 wall=time.perf_counter()-start;profile.disable();buf=io.StringIO();pstats.Stats(profile,stream=buf).sort_stats('cumulative').print_stats(8)
 result[label]=dict(seconds=wall,rows=32,tokens_per_row=len(ids),text_sha=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),profile=buf.getvalue())
assert result['original']['text_sha']==result['engine_cached_copy']['text_sha']
print(json.dumps(result),flush=True)
