"""Read-only, outer-separated counters with explicit completed/observed scopes."""
import argparse, collections, json, math, time
from pathlib import Path

def summarize(root):
    outers={}
    for path in sorted(root.glob('*/*/events.jsonl')):
        rows=[];truncated=0
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:truncated+=1
        if not rows:continue
        start=rows[0];outer=start['outer']
        o=outers.setdefault(outer,dict(outer=outer,run_id=start['run_id'],source_commit=start['source_commit'],
             config_sha256=start['config_sha256'],started_questions=0,closed_questions=0,
             final_trajectories=0,shortfalls=0,continuations_closed=0,contrast_accepts_closed=0,
             same_candidates_closed=0,fill_closed=0,strict_correct_selected_closed=0,
             strict_correct_initial_closed=0,max_attempts_closed=0,
             all_observed_events=collections.Counter(),all_observed_candidate_generation=collections.Counter(),
             closed_candidate_generation=collections.Counter(),closed_stop_reasons=collections.Counter(),
             violations=[],execution_errors=0,incomplete_json_lines=0,entropy_sum=0.))
        o['started_questions']+=1;o['incomplete_json_lines']+=truncated
        ids={};accepted=set();active_search=None;pending=None;last_attempt=0
        ends=[];candidate_gen=collections.Counter()
        for r in rows:
            e=r['event'];o['all_observed_events'][e]+=1
            def bad(reason):o['violations'].append(dict(path=str(path),event_seq=r.get('event_seq'),reason=reason))
            if (r.get('outer')!=outer or r.get('run_id')!=o['run_id'] or
                r.get('source_commit')!=o['source_commit'] or r.get('config_sha256')!=o['config_sha256']):bad('identity changed')
            if e=='search_start':
                eligible=r['eligible'];n=len(eligible)
                for q in eligible:
                    if q['generation']==0 and q['id'] not in ids:
                        ids[q['id']]=(0,start['initial_reward']);accepted.add(q['id'])
                    if q['id'] not in accepted:bad('eligible is not accepted')
                    if ids.get(q['id'],(-1,))[0]!=q['generation']:bad('eligible generation differs')
                    if q.get('weight')!=1 or n==0 or not math.isclose(q['probability'],1/n,abs_tol=1e-14):bad('not uniform eligible-parent selection')
                parent=r['parent_id']
                if parent not in {q['id'] for q in eligible}:bad('parent not eligible')
                gen=ids.get(parent,(-1,))[0]
                earlier={k for k,v in ids.items() if v[0]<gen}
                if set(r['lcps'])!=earlier:bad('comparison set is not strictly lower generation')
                boundary=max(r['lcps'].values(),default=0)
                if boundary!=r['boundary']:bad('boundary differs from maximum recorded LCP')
                positions=r['legal_positions']
                if not positions or positions!=list(range(boundary,positions[-1]+1)) or r['start'] not in positions:bad('illegal recorded position interval')
                active_search=r
            elif e=='branch_start':
                if r['attempt']!=last_attempt+1 or r['attempt']>start['budget']:bad('attempt budget/order differs')
                if active_search is None or r['parent_id']!=active_search['parent_id']:bad('branch parent differs from active search')
                elif not r['boundary']<=r['action_index']<=active_search['start']:bad('branch outside fixed backoff boundary')
                last_attempt=r['attempt'];pending=r
            elif e=='branch_end':
                if pending is None or pending['attempt']!=r['attempt']:bad('branch terminal missing matching start')
                if r['status']=='error':o['execution_errors']+=1
                else:
                    parent=ids.get(pending['parent_id']) if pending else None
                    if parent is None or r['generation']!=parent[0]+1 or r['parent_reward']!=parent[1]:bad('child generation/parent reward differs')
                    if r['id'] in ids:bad('duplicate sampled identity')
                    if r['reward'] not in (0,1):bad('non-binary strict reward')
                    contrast=parent is not None and r['reward']!=parent[1]
                    if r['source']!=('contrast_accept' if contrast else 'same_reward'):bad('contrast/pool classification differs')
                    ids[r['id']]=(r['generation'],r['reward'])
                    if contrast:accepted.add(r['id'])
                    candidate_gen[r['generation']]+=1;o['all_observed_candidate_generation'][r['generation']]+=1
                pending=None
            elif e=='question_end':ends.append(r)
        if len(ends)>1:o['violations'].append(dict(path=str(path),reason='multiple question terminals'))
        if ends:
            end=ends[-1];final=end['final_ids'];n=len(final)
            if len(set(final))!=n or any(k not in ids for k in final):
                o['violations'].append(dict(path=str(path),reason='final selected identities unknown or duplicated'))
            if end['attempts']!=last_attempt or end['attempts']>start['budget']:
                o['violations'].append(dict(path=str(path),reason='terminal budget differs'))
            if end['complete_group']!=(n==start['target']):
                o['violations'].append(dict(path=str(path),reason='terminal group completeness differs'))
            o['closed_questions']+=1;o['final_trajectories']+=n;o['shortfalls']+=int(not end['complete_group'])
            o['continuations_closed']+=end['attempts'];o['contrast_accepts_closed']+=end['accepted_branches']
            o['same_candidates_closed']+=end['same_reward_candidates'];o['fill_closed']+=end['fill_count']
            correct=sum(ids.get(k,(0,0))[1] for k in final);o['strict_correct_selected_closed']+=correct
            o['strict_correct_initial_closed']+=start['initial_reward'];o['max_attempts_closed']=max(o['max_attempts_closed'],end['attempts'])
            o['closed_candidate_generation'].update(candidate_gen);o['closed_stop_reasons'][end['stop_reason']]+=1
            p=correct/n if n else 0.;o['entropy_sum']+=(-p*math.log(p)-(1-p)*math.log1p(-p)) if 0<p<1 else 0.
    for o in outers.values():
        n=o['closed_questions'];o['mean_selected_binary_reward_entropy_closed']=o.pop('entropy_sum')/n if n else None
        o['mean_continuations_closed']=o['continuations_closed']/n if n else None
        o['selected_training_trajectory_strict_em_closed']=o['strict_correct_selected_closed']/o['final_trajectories'] if o['final_trajectories'] else None
        o['initial_root_strict_em_closed']=o['strict_correct_initial_closed']/n if n else None
        o['fill_fraction_closed']=o['fill_closed']/o['final_trajectories'] if o['final_trajectories'] else None
    return dict(timestamp=time.time(),outers=[outers[k] for k in sorted(outers)],
       scope='Recorded structural event checks and closed-question aggregates only; no true-LCP text reconstruction, optimizer/logit/gradient recomputation or dev EM claim')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.events),ensure_ascii=False))
