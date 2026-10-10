from pathlib import Path
p=Path('work/search-e1-first20-report-publish-20261010/experiments/search_e1/control/opsd_teacher_offpolicy200_v1')
s=(p/'teacher_diag.py').read_text().replace("return original_prompt(question,reference=case['reference'])","return original_prompt(question,reference=case['reference']) if case.get('_privileged',True) else original_prompt(question)")
(p/'teacher_diag.py').write_text(s)
(p/'paired_diag.py').write_text('''"""One paired autonomous case per outer; isolated RNG, no training contribution."""
import os,json,sys,argparse,random,hashlib
from pathlib import Path
from checkpoint import checked_json,fileproof,exclusive_json,require,checked_file
from sampling import read_rows
from teacher_diag import capture

def main():
 a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True);v=a.parse_args()
 req,ref=checked_json(dict(path=v.request,sha256=v.request_sha256));cfg,_=checked_json(req['config'])
 require(os.environ.get('E1_RUN_ID')==cfg['run_id'] and os.getuid()==1005,'Diagnostic ownership mismatch')
 import torch,numpy as np
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 from search_e1 import rollout as native,core
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 pairs=read_rows(checked_file(req['pairs'])['path']);root=Path(req['output']);root.mkdir()
 if not pairs:exclusive_json(root/'complete.json',dict(accepted=True,count=0,reason='No successful reference',original_training_objective=False));return
 pair=min(pairs,key=lambda x:hashlib.sha256(json.dumps([42,req['outer_index'],x['question_id']]).encode()).hexdigest())
 tok=AutoTokenizer.from_pretrained(cfg['paths']['model'],use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(cfg['paths']['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 if req['student_checkpoint']:
  cp,_=checked_json(req['student_checkpoint']);adapter=cp.get('adapter') or cp.get('adapter_model')
  # Actual manifest layout is adapter_directory plus payload files.
  files=cp['files'];paths=[x['path'] for x in files if Path(x['path']).name=='adapter_model.safetensors'] if isinstance(files,list) else []
  if not paths:
   paths=[str(Path(req['student_checkpoint']['path']).parent/'adapter_model.safetensors')]
  model=PeftModel.from_pretrained(model,str(Path(paths[0]).parent),is_trainable=False).eval()
 agent=native.SearchAgent(model,tok,cfg['paths']['retriever_url'],max_new_tokens=500,max_context=8192)
 out=[]
 from contextlib import nullcontext
 for role in ('student_ordinary','teacher_privileged'):
  case=dict(pair,_privileged=role=='teacher_privileged')
  ctx=model.disable_adapter() if role=='teacher_privileged' and hasattr(model,'disable_adapter') else nullcontext()
  with ctx, (root/(role+'-http.jsonl')).open('x') as http:
   with capture(agent,native,case,http) as trace:result=agent.run(case['question'],temperature=0.)
   proxy=trace.pop('http_proxy');require(not proxy.logging_errors and not trace['logging_errors'],'Incomplete diagnostic trace')
  out.append(exclusive_json(root/(role+'.json'),dict(role=role,result=result,strict_reward=core.reward(result['response'],pair['answers'],result['policy_spans']),trace=trace,question_id=pair['question_id'],student_checkpoint=req['student_checkpoint'],original_training_objective=False,denominator_contribution=0)))
 exclusive_json(root/'complete.json',dict(accepted=True,count=1,outputs=out,request=ref,original_training_objective=False))
if __name__=='__main__':main()
''')
s=(p/'controller.py').read_text();a=s.index('    def teacher_diagnostic(');z=s.index('    def verify_teacher_child(',a)
method='''    def teacher_diagnostic(self,outer,index,manifest):
        root=outer/'teacher-diagnostics';root.mkdir(exist_ok=True)
        done=root/'complete.json'
        if done.exists():return checked_json(fileproof(done))[0]
        request=exclusive_json(root/'paired-request.json',dict(config=self.config_ref,pairs=manifest['pairs'],outer_index=index,student_checkpoint=manifest['previous_complete_checkpoint'],output=str(root/'paired')))
        command=[self.config['paths']['python'],'-B',str(Path(self.config['paths']['helper_root'])/'paired_diag.py'),'--request',request['path'],'--request-sha256',request['sha256']]
        self.gate('sampling-start',[])
        job=self.spawn(command,root/'paired.log','sampling-running',dict(CUDA_VISIBLE_DEVICES='0'))
        terminal=self.wait([job],'sampling-running')
        child,ref=checked_json(fileproof(root/'paired/complete.json'));require(child['accepted'],'Paired autonomous diagnostic incomplete')
        return exclusive_json(done,dict(accepted=True,child=ref,exit_proof=terminal,original_training_objective=False))

'''
s=s[:a]+method+s[z:];(p/'controller.py').write_text(s)
