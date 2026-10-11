"""Read only persisted recursive events; incomplete questions stay incomplete."""
import argparse, collections, json, time
from pathlib import Path

def summarize(root):
    events=collections.Counter(); stops=collections.Counter(); generations=collections.Counter()
    completed=[]; errors=[]; partial=0; invalid=[]
    for path in sorted(root.glob('*/*/events.jsonl')):
        rows=[]
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: partial+=1
        for r in rows:
            events[r['event']]+=1
            if r['event']=='branch_end':
                if r['status']=='complete': generations[r['generation']]+=1
                else: errors.append(dict(path=str(path),event=r))
        ends=[r for r in rows if r['event']=='question_end']
        if ends:
            r=ends[-1]; completed.append(r);stops[r['stop_reason']]+=1
            if r['attempts']>10 or len(r['final_ids'])!=len(set(r['final_ids'])) or len(ends)!=1:
                invalid.append(str(path))
    n=len(completed); attempts=sum(r['attempts'] for r in completed)
    same=sum(r['same_reward_candidates'] for r in completed)
    fill=sum(r['fill_count'] for r in completed)
    return dict(timestamp=time.time(),event_counts=dict(events),completed_questions=n,
      shortfall_questions=sum(not r['complete_group'] for r in completed),
      accepted_branches=sum(r['accepted_branches'] for r in completed),
      actual_continuations=attempts,same_reward_candidates=same,fills=fill,
      fill_fraction=fill/sum(len(r['final_ids']) for r in completed) if n else None,
      same_fraction=same/attempts if attempts else None,
      pool_use_fraction=fill/same if same else None,
      generation_histogram=dict(generations),stop_reasons=dict(stops),
      incomplete_json_lines=partial,errors=errors,invariant_failures=invalid,
      scope='Persisted question_end only; not optimizer or outer completion acceptance')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',type=Path,required=True)
    args=p.parse_args(); print(json.dumps(summarize(args.events),ensure_ascii=False))
