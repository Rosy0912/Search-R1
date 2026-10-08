"""CPU-only summaries of completed cases; diagnostic strata are not benchmarks."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
from common import atomic_json,file_sha
VERSIONS=('source100','update1','update5','epoch1')
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/data/shaoshitong/search-e1-opsd-debug-20261008'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 pointer=json.loads((a.base/'receipts/current-debug-run.json').read_text());run=Path(pointer['output']);cases=json.loads((a.base/'cases/manifest.json').read_text());collected=[];seen=set();status=[]
 for worker in sorted(run.glob('worker-*')):
  if (worker/'status.json').exists():status.append(json.loads((worker/'status.json').read_text()))
  results=worker/'case-results.jsonl'
  if not results.exists():continue
  for line in results.read_text().splitlines(keepends=True):
   if not line.endswith('\n'):continue
   c=json.loads(line);assert c['status']=='complete' and c['case_id'] not in seen;seen.add(c['case_id']);collected.append(c)
 cells=defaultdict(list)
 for c in collected:cells[(c['dataset'],c['group'])].append(c)
 summary=[]
 for (dataset,group),items in sorted(cells.items()):
  correct={v:sum(c['versions'][v]['generation_em'] for c in items) for v in VERSIONS}
  flips={v:dict(improved=sum(c['versions']['source100']['generation_em']==0 and c['versions'][v]['generation_em']==1 for c in items),worsened=sum(c['versions']['source100']['generation_em']==1 and c['versions'][v]['generation_em']==0 for c in items)) for v in VERSIONS[1:]}
  losses={}
  for version in VERSIONS:
   origins=defaultdict(list)
   for c in items:
    for probe in c['versions'][version]['token_probes']:
     name=probe['preserved_trajectory_name'];origins[name].append(probe)
   losses[version]={name:dict(probes=len(rows),completed=sum(r['status']=='complete' for r in rows),skipped=sum(r['status']=='skipped' for r in rows),tokens=sum(r.get('response_tokens',0) for r in rows if r['status']=='complete'),valid_policy_tokens=sum(r.get('valid_policy_tokens',0) for r in rows if r['status']=='complete'),mean_trajectory_loss=(sum(r['trajectory_mean'] for r in rows if r.get('trajectory_mean') is not None)/sum(r.get('trajectory_mean') is not None for r in rows)) if any(r.get('trajectory_mean') is not None for r in rows) else None,frozen_loss_checks_passed=all(r.get('frozen_pair_loss_check',{}).get('passed',True) for r in rows)) for name,rows in origins.items()}
  teachers=[c['teacher_generation']['protocol_em'] for c in items if 'protocol_em' in c['teacher_generation']]
  summary.append(dict(dataset=dataset,group=group,completed_cases=len(items),planned_cases=32,student_correct=correct,paired_checkpoint_flips=flips,teacher_available=len(teachers),teacher_correct=sum(teachers),token_loss_origins=losses))
 report=dict(created_unix=time.time(),run_id=pointer['run_id'],run_path=str(run),controller_status=pointer['status'],completed_cases=len(collected),planned_cases=192,cases_manifest_sha256=file_sha(a.base/'cases/manifest.json'),case_source_sha256=cases['cases']['sha256'],cells=summary,workers=[{k:s.get(k) for k in ['worker_index','pid','proc_starttick','status','completed_cases','assigned_cases','gpu','last_failure']} for s in status],scope='Partial completed-case summaries only unless192 complete; fixed stratified greedy HF diagnostic, not native512/full7. Teacher privilege and checkpoint effects are associations, not causal attribution. No training.')
 atomic_json(a.output,report);print(json.dumps(dict(completed_cases=len(collected),planned_cases=192,output=str(a.output))))
if __name__=='__main__':main()
