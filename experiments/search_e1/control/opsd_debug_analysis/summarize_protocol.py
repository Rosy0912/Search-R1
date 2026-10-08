"""CPU-only frozen-protocol audit of completed diagnostic generation cases.

Output is PRIVATE analysis data; do not commit actual answers/traces to Git.
No model/tokenizer, GPU, HTTP, training, or changes to the running diagnostics.
Literal invalid-tag matches are NOT strict EM or evidence of semantic correctness.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import types

HASHES={'search_e1/core.py':'0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc',
 'upstream/Search-R1/search_r1/qa_protocol.py':'9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'}
VERSIONS=('teacher','source100','update1','update5','epoch1')
ANSWER=re.compile(r'<answer>(.*?)</answer>',re.S)
TAGS=re.compile(r'<(/?)(think|search|answer|information)>')


def sha_bytes(data):return hashlib.sha256(data).hexdigest()

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def stat(path):
 s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def read_stable(path):
 path=Path(path);before=stat(path);data=path.read_bytes()
 if stat(path)!=before:raise RuntimeError('Immutable completed file changed: '+str(path))
 return json.loads(data),dict(path=str(path.resolve()),sha256=sha_bytes(data),stat=before)


def load_science(runtime):
 runtime=Path(runtime).resolve();proof={}
 for rel,wanted in HASHES.items():
  _,item=read_source(runtime/rel)
  if item['sha256']!=wanted:raise ValueError('Frozen source hash differs: '+rel)
  proof[rel]=item
 # Directly load dependency-free frozen modules. A minimal package prevents any
 # upstream package initializer/model dependency from executing on this CPU path.
 parent='search_r1';name=parent+'.qa_protocol';path=runtime/'upstream/Search-R1/search_r1/qa_protocol.py'
 if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=path:raise ValueError('Foreign qa_protocol already imported')
 if parent not in sys.modules:
  package=types.ModuleType(parent);package.__path__=[str(path.parent)];sys.modules[parent]=package
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,path);qa=importlib.util.module_from_spec(spec);sys.modules[name]=qa;spec.loader.exec_module(qa)
 qa=sys.modules[name]
 spec=importlib.util.spec_from_file_location('_frozen_debug_analysis_core',runtime/'search_e1/core.py')
 core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 return core,qa,proof


def read_source(path):
 before=stat(path);data=Path(path).read_bytes()
 if stat(path)!=before:raise RuntimeError('Source changed')
 return data,dict(path=str(Path(path).resolve()),sha256=sha_bytes(data),stat=before)


def overlaps(a,b,ranges):return any(a<y and b>x for x,y in ranges)


def unclosed_think_before(text,offset):
 # This is a conservative diagnostic cue, not a replacement protocol parser.
 balance=0
 for tag in re.finditer(r'</?think>',text[:offset]):balance+=-1 if tag.group().startswith('</') else 1
 return balance>0


def analyze_generation(generation,answers,core,qa):
 if generation.get('status')=='not_applicable_no_successful_reference':
  return dict(status='skipped',reason='no_successful_reference',strict_em=None)
 if generation.get('kind')!='actual_autoregressive_generation' or generation.get('teacher_forced') is not False:
  raise ValueError('Expected genuine autoregressive generation artifact')
 response=generation['response'];bounds=generation['policy_spans'];previous=0;spans=[]
 for start,end in bounds:
  if type(start)!=int or type(end)!=int or not previous<=start<=end<=len(response):raise ValueError('Invalid policy provenance bounds')
  spans.append(response[start:end]);previous=end
 answer=qa.extract_final_answer(spans);strict=core.reward(response,answers,bounds)
 if generation.get('protocol_em')!=strict:raise ValueError('Recorded strict EM disagrees with frozen reward')
 if isinstance(answers,str):answers=[answers]
 last=qa.parse_model_action(spans[-1]) if spans else None
 information=core.information_spans(response);gaps=[];cursor=0
 for start,end in bounds:
  if cursor<start:gaps.append([cursor,start])
  cursor=end
 if cursor<len(response):gaps.append([cursor,len(response)])
 tag_records=[];eligible=[]
 for match in ANSWER.finditer(response):
  start,end=match.span();policy=[i for i,(a,b) in enumerate(bounds) if a<=start and end<=b]
  scope='last_policy_span' if policy and policy[0]==len(bounds)-1 else 'earlier_policy_span' if policy else 'outside_or_crossing_policy_span'
  info=overlaps(start,end,information)
  item=dict(character_range=[start,end],payload_sha256=sha_bytes(match.group(1).encode()),scope=scope,information_overlap=info,
    excluded_from_terminal_candidate=scope!='last_policy_span' or info,
    unclosed_think_before_tag=unclosed_think_before(spans[policy[0]],start-bounds[policy[0]][0]) if policy else False)
  tag_records.append(item)
  if not item['excluded_from_terminal_candidate']:eligible.append((match,item))
 candidate=eligible[-1] if answer is None and eligible else None
 literal=bool(candidate and any(core.normalize(candidate[0].group(1))==core.normalize(gold) for gold in answers))
 calls=generation['search_calls']
 if type(calls)!=int or not 0<=calls<=4:raise ValueError('Invalid executed search count')
 turns=generation.get('turns',[])
 queries=[turn for turn in turns if 'query' in turn]
 if len(queries)!=calls:raise ValueError('Executed search count/turn provenance mismatch')
 valid=answer is not None
 reason=None if valid else 'no_policy_span' if not spans else 'last_valid_action_is_search' if last and last[0]=='search' else 'last_span_not_standalone_terminal_answer'
 return dict(status='analyzed',strict_em=strict,strict_final_answer=answer,final_answer_protocol_valid=valid,
   last_action_protocol_valid=last is not None,last_action_kind=last[0] if last else None,protocol_invalid_reason=reason,
   invalid_tag_candidate_matches_gold=literal,
   invalid_tag_candidate_interpretation='Literal normalized tag text only; never strict EM or semantic correctness',
   invalid_tag_candidate_range=candidate[1]['character_range'] if candidate else None,
   invalid_tag_candidate_unclosed_think=bool(candidate and candidate[1]['unclosed_think_before_tag']),
   no_executed_search=calls==0,search_calls=calls,termination=generation.get('termination'),
   policy_spans=[dict(character_range=[a,b],sha256=sha_bytes(response[a:b].encode()),last=i==len(bounds)-1,
     parsed_action_kind=qa.parse_model_action(response[a:b])[0] if qa.parse_model_action(response[a:b]) else None) for i,(a,b) in enumerate(bounds)],
   information_excluded_ranges=[list(x) for x in information],outside_policy_ranges=gaps,answer_tag_ranges=tag_records,
   structural_tag_ranges=[dict(tag=m.group(),character_range=list(m.span()),information_overlap=overlaps(*m.span(),information),
     inside_policy=any(a<=m.start() and m.end()<=b for a,b in bounds)) for m in TAGS.finditer(response)],
   response_sha256=sha_bytes(response.encode()),response_characters=len(response),
   generated_token_count=sum(len(t.get('generated_token_ids',[])) for t in turns),
   mode='real_generated_response_rechecked_by_frozen_strict_protocol; no token_loss_or_online_training_claim')


def completed_prefix(path):
 # Workers may append later. Audit only the complete-line prefix present at this
 # read; do not claim its hash identifies the final whole ongoing file.
 before=stat(path)
 with Path(path).open('rb') as stream:data=stream.read(before['bytes'])
 if len(data)!=before['bytes']:raise RuntimeError('Results prefix shortened during read')
 now=stat(path)
 if now['device']!=before['device'] or now['inode']!=before['inode'] or now['bytes']<before['bytes']:raise RuntimeError('Results file replaced/truncated')
 end=data.rfind(b'\n')+1;prefix=data[:end]
 return [json.loads(x) for x in prefix.splitlines()],dict(path=str(Path(path).resolve()),prefix_bytes=end,
  prefix_sha256=sha_bytes(prefix),snapshot_stat=before,trailing_partial_line_skipped=end<len(data),whole_final_file_hash_claimed=False)


def summarize(runtime,cases_dir,run_dir,output):
 core,qa,science=load_science(runtime);cases_dir=Path(cases_dir);run_dir=Path(run_dir);output=Path(output)
 cases_bytes,cases_proof=read_source(cases_dir/'cases.jsonl');manifest,manifest_proof=read_stable(cases_dir/'manifest.json')
 if sha_bytes(cases_bytes)!=manifest['cases']['sha256']:raise ValueError('Case manifest content hash mismatch')
 cases={}
 for case in [json.loads(x) for x in cases_bytes.splitlines()]:
  expected=case['case_semantic_sha256'];payload={k:v for k,v in case.items() if k!='case_semantic_sha256'}
  if sha_bytes(canonical(payload))!=expected or case['case_id'] in cases:raise ValueError('Case semantic identity mismatch/duplicate')
  cases[case['case_id']]=case
 config,config_proof=read_stable(run_dir/'config.json')
 records=[];prefixes=[];skips=Counter();seen=set();counts=defaultdict(Counter)
 for path in sorted(run_dir.glob('worker-*/case-results.jsonl')):
  results,prefix=completed_prefix(path);prefixes.append(prefix)
  skips['trailing_partial_results_line']+=prefix['trailing_partial_line_skipped']
  for result in results:
   if result.get('status')!='complete':skips['not_complete_result']+=1;continue
   key=result['case_id']
   if key in seen or key not in cases:raise ValueError('Duplicate/foreign completed case')
   case=cases[key]
   if result['source_case_sha256']!=case['case_semantic_sha256'] or result['dataset']!=case['dataset'] or result['group']!=case['group']:raise ValueError('Completed case provenance mismatch')
   seen.add(key);case_dir=path.parent/key;answers=case['source_train_row']['row']['answers']
   entry=dict(case_id=key,question_id=case['question_id'],dataset=case['dataset'],group=case['group'],
    source_case_sha256=case['case_semantic_sha256'],completed_case_result_sha256=sha_bytes(canonical(result)),versions={})
   for version in VERSIONS:
    name='teacher-generation.json' if version=='teacher' else 'student-generation-'+version+'.json'
    generation,trace=read_stable(case_dir/name)
    if version=='teacher' and case['group']=='all5_wrong' and generation.get('status')!='not_applicable_no_successful_reference':raise ValueError('Hard case has fabricated teacher')
    if generation.get('status')!='not_applicable_no_successful_reference':
     if version=='teacher' and (generation.get('active_adapter')!='disabled' or generation.get('privileged_reference') is not True):raise ValueError('Teacher is not frozen/privileged generation')
     if version!='teacher' and (generation.get('active_adapter')!=version or generation.get('privileged_reference') is not False):raise ValueError('Student adapter/reference label mismatch')
    analyzed=analyze_generation(generation,answers,core,qa)
    if analyzed['status']=='analyzed' and version!='teacher' and result['versions'][version]['generation_em']!=analyzed['strict_em']:raise ValueError('Completed case aggregate EM disagrees with generation')
    analyzed['trace']=trace;entry['versions'][version]=analyzed
    cell=counts[(version,case['dataset'],case['group'])];cell['completed_case_artifacts']+=1
    if analyzed['status']=='skipped':cell['skipped']+=1;skips['teacher_no_successful_reference']+=1;continue
    cell['analyzed']+=1;cell['strict_correct']+=analyzed['strict_em'];cell['final_answer_protocol_valid']+=analyzed['final_answer_protocol_valid']
    cell['protocol_valid_wrong_answer']+=analyzed['final_answer_protocol_valid'] and not analyzed['strict_em']
    cell['final_answer_protocol_invalid']+=not analyzed['final_answer_protocol_valid'];cell['no_executed_search']+=analyzed['no_executed_search']
    cell['invalid_tag_candidate_matches_gold']+=analyzed['invalid_tag_candidate_matches_gold'];cell['invalid_tag_candidate_unclosed_think']+=analyzed['invalid_tag_candidate_unclosed_think']
   records.append(entry)
 skips['selected_cases_not_yet_complete']=len(cases)-len(seen)
 totals=defaultdict(Counter)
 for (version,dataset,group),cell in counts.items():totals[version].update(cell)
 summary=dict(schema='frozen-protocol-postprocess-v1',created_unix=time.time(),run_id=config['run_id'],
  scope='Completed stratified diagnostic cases only; not native512/seven-benchmark EM or proof of semantic correctness',
  interpretation='invalid_tag_candidate_matches_gold is excluded from strict_correct; no rawtop32 or loss interpreted as historical online loss',
  selected_cases=len(cases),completed_cases=len(records),skip_counts=dict(skips),
  source=dict(science=science,cases=cases_proof,case_manifest=manifest_proof,run_config=config_proof,completed_results_prefixes=prefixes),
  by_version={v:dict(c) for v,c in sorted(totals.items())},
  cells=[dict(version=v,dataset=d,group=g,**dict(c)) for (v,d,g),c in sorted(counts.items())])
 output.mkdir(parents=True,exist_ok=False)
 with (output/'cases.jsonl').open('x') as f:
  for r in records:f.write(canonical(r).decode()+'\n')
 summary['cases_output']=dict(path=str((output/'cases.jsonl').resolve()),sha256=sha_bytes((output/'cases.jsonl').read_bytes()),rows=len(records))
 with (output/'summary.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return summary


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('runtime','cases-dir','run-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();summary=summarize(args.runtime,args.cases_dir,args.run_dir,args.output)
 print(json.dumps(dict(output=str(args.output),completed_cases=summary['completed_cases'],skip_counts=summary['skip_counts'])))

if __name__=='__main__':main()
