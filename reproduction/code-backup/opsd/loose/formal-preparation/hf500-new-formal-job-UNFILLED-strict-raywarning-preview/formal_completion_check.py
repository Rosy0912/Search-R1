"""Read-only actual formal500 prerequisites; no future job ID or self-approval."""
import os,re
from pathlib import Path
from training_evidence import loads,own_file,digest,verify_slurm,verify_outers,COMMIT
from periodic_validation import verify_periodic_validations
OWN=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
FORMAL_JOB='UNFILLED_ACTUAL_NEW_FORMAL_JOB_ID'
FORMAL=OWN/('runs/opsd-formal-'+FORMAL_JOB)
def verify(receipt,base=OWN):
 assert re.fullmatch('[0-9]+',FORMAL_JOB),'Actual newly submitted formal job ID is not filled'
 run=base/('runs/opsd-formal-'+FORMAL_JOB);r=loads(own_file(str(receipt),base).read_text())
 assert set(r)=={'source_commit','formal_job','evidence'}
 assert r['source_commit']==COMMIT and r['formal_job']==FORMAL_JOB
 roles={'raw_log','source_commit','slurm_terminal','exit','finished','completeness','expected_final_counts','independent_review'}
 assert set(r['evidence'])==roles
 paths={}
 for key,item in r['evidence'].items():
  assert set(item)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',item['sha256'])
  p=own_file(item['path'],base if key=='independent_review' else run)
  assert digest(p)==item['sha256'];paths[key]=p
 assert paths['source_commit'].read_text().strip()==COMMIT
 verify_slurm(paths['slurm_terminal'].read_text(),FORMAL_JOB)
 terminal=paths['slurm_terminal'].read_text()
 for key,value in (('NodeList','h100-10'),('NumNodes','1'),('NumCPUs','192'),('TresPerNode','gres:gpu:h100:8')):
  assert re.findall(r'\b'+key+r'=([^\s]+)',terminal)==[value]
 assert paths['exit'].read_text().strip()=='0'
 assert paths['finished'].read_text().strip()=='FORMAL_500_OUTERS_AND_FINAL_EVALUATION_EXITED'
 raw=paths['raw_log'].read_text();verify_outers(raw,500);verify_periodic_validations(raw)
 expected=loads(paths['expected_final_counts'].read_text())
 assert len(expected)==7 and all(type(n) is int and n>0 for n in expected.values()) and sum(expected.values())==51713
 marker='SEARCH_R1_FINAL_METRICS_JSON='
 finals=[loads(line.split(marker,1)[1]) for line in raw.splitlines() if marker in line]
 assert len(finals)==1
 final=finals[0];assert final['val/qa_count/all']==51713
 assert {k[len('val/qa_count/'):] for k in final if k.startswith('val/qa_count/')}==set(expected)|{'all'}
 assert {k[len('val/evaluated_count/'):] for k in final if k.startswith('val/evaluated_count/')}==set(expected)
 for source,count in expected.items():assert final['val/qa_count/'+source]==count and final['val/evaluated_count/'+source]==count
 for source in set(expected)|{'all'}:
  for kind in ('f1','em','answer_format','format_rate'):assert 0<=final[f'val/{kind}/{source}']<=1
 c=loads(paths['completeness'].read_text())
 assert c['passed'] is True and c['actual_outer_updates']==500 and c['final_questions']==51713 and c['source_commit']==COMMIT
 assert c['final_per_source_counts']==expected and c['checkpoint_steps']==[5,100,200,300,400,500] and c['fresh_hf_reload_pending'] is True
 review=loads(paths['independent_review'].read_text())
 assert review['source_commit']==COMMIT and review['formal_job']==FORMAL_JOB and review['accepted'] is True
 assert review['reviewer'] and review['report']
 assert review['bound_evidence_sha256']=={k:v['sha256'] for k,v in r['evidence'].items() if k!='independent_review'}
 checkpoint=run/'output/checkpoints/opsd-gpu-formal500/actor/global_step_500'
 assert checkpoint.is_dir() and len(list(checkpoint.glob('*.safetensors')))==3
 for name in ('config.json','tokenizer.json','model.safetensors.index.json'):own_file(str(checkpoint/name),run)
 assert len(loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'])==435
 return {'source_commit':COMMIT,'formal_job':FORMAL_JOB,'actual_outer_updates':500,'final_questions':51713,'receipt_sha256':digest(Path(receipt)),'checkpoint':str(checkpoint)}
if __name__=='__main__':
 result=verify(os.environ['OPSD_FORMAL_COMPLETION_RECEIPT'])
 print('ACTUAL_FORMAL500_FINAL51713_AND_TERMINAL_RECEIPT_VERIFIED',result,flush=True)
