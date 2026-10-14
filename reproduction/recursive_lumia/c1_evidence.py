"""Read-only checks of actual C1 tree events and consumed-round accounting."""
import json
from pathlib import Path


def validate_question(events):
    if (not events or events[0]['event']!='question_start' or events[-1]['event']!='question_end'
            or [e['event_seq'] for e in events]!=list(range(1,len(events)+1))):
        raise ValueError('Incomplete C1 question event stream')
    first,last=events[0],events[-1]
    if first['budget']!=10 or first['target']!=5 or first['initial_reward'] not in (0,1):
        raise ValueError('C1 question changed the original budget/target/reward')
    root=last['final_ids'][0]
    starts=[e for e in events if e['event']=='branch_start']
    ends=[e for e in events if e['event']=='branch_end']
    if (len(starts)!=len(ends) or len(starts)>10 or last['attempts']!=len(starts)
            or [e['attempt'] for e in starts]!=list(range(1,len(starts)+1))
            or [e['attempt'] for e in ends]!=list(range(1,len(ends)+1))):
        raise ValueError('C1 did not preserve its finite continuation budget')
    for e in events:
        if e['event']=='search_start':
            eligible=e['eligible']
            if (e.get('ablation_variant')!='root_only' or len(eligible)!=1
                    or eligible[0]['id']!=root or eligible[0]['generation']!=0
                    or e['parent_id']!=root):
                raise ValueError('A C1 search used a non-root parent')
    if any(e['parent_id']!=root for e in starts):
        raise ValueError('A C1 continuation used a non-root parent')
    accepted=[];same=[]
    for e in ends:
        if (e['status']!='complete' or e['generation']!=1
                or e['parent_reward']!=first['initial_reward'] or e['reward'] not in (0,1)):
            raise ValueError('C1 child generation/reward/parent evidence differs')
        contrast=e['reward']!=e['parent_reward']
        if e['source']!=('contrast_accept' if contrast else 'same_reward'):
            raise ValueError('C1 changed reward-difference acceptance')
        (accepted if contrast else same).append(e['id'])
    if len(set([root]+accepted+same))!=1+len(ends):
        raise ValueError('C1 candidate identities are repeated')
    fill=[e for e in events if e['event']=='fill_end']
    if len(fill)!=1:
        raise ValueError('C1 final fill evidence is missing')
    chosen=fill[0]['selected_ids']
    if (fill[0]['pool_ids']!=same or len(chosen)!=len(set(chosen))
            or not set(chosen).issubset(same) or len(chosen)!=min(5-1-len(accepted),len(same))
            or last['final_ids']!=[root]+accepted+chosen
            or len(last['final_ids'])>5 or last['accepted_branches']!=len(accepted)
            or last['same_reward_candidates']!=len(same)):
        raise ValueError('C1 changed the accepted group or original fill rule')
    return dict(root=root,branches=len(ends),contrast=len(accepted),same=len(same),filled=len(chosen))


def verify_events(output, record, outer=5):
    output=Path(output);questions={};branches=contrast=same=0
    for path in sorted((output/'recursive-events').glob('*/*/events.jsonl')):
        raw=path.read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('Unfinished event stream cannot establish a complete C1 boundary')
        rows=[json.loads(line) for line in raw.splitlines()]
        if not rows or rows[0]['outer']!=outer:continue
        question=rows[0]['question_id']
        expected=dict(question_id=question,outer=outer,run_id=record['run_id'],
                      source_commit=record['source_commit'],config_sha256=record['config_sha256'])
        if question in questions or any(any(row[k]!=v for k,v in expected.items()) for row in rows):
            raise ValueError('C1 event identity or question coverage differs')
        result=validate_question(rows)
        root=json.loads((path.parent/(result['root']+'.json')).read_text())
        if (root['source']!='initial' or root['state']['trajectory']!=result['root']
                or root['state']['tree']!=rows[0]['root_id']):
            raise ValueError('The sole C1 parent is not the saved initial root')
        questions[question]=result
        branches+=result['branches'];contrast+=result['contrast'];same+=result['same']
    if len(questions)!=512 or branches==0:
        raise ValueError('Need actual C1 branch evidence covering all512 training questions')
    return dict(outer=outer,questions=len(questions),branches=branches,contrast=contrast,same=same,
                root_only_verified=True,reward_acceptance_and_fill_verified=True)


def validate_rounds(audit, declared_totals):
    if audit['completed_rounds']!=150 or len(audit['attempts'])!=150:
        raise ValueError('C1 has not completed150 consumed training rounds')
    total=0;zero=0;guards=0
    for outer,r in enumerate(audit['attempts'],1):
        writes=r['actual_adam_writes_per_rank'];guard=r['normal_guard']
        if r['outer']!=outer or type(writes) is not int or not 0<=writes<=20:
            raise ValueError('Actual optimizer opportunity accounting differs')
        if writes<20:
            if (not guard or guard['status']!='NORMAL_SAFETY_GUARD_ROUND_CONSUMED'
                    or guard['optimizer_writes']!=writes or guard['opportunities_processed']!=20
                    or guard['opportunities_skipped']!=20-writes):
                raise ValueError('Skipped updates lack explicit normal-guard evidence')
        if r['reported_pg_loss_finite'] is not True or any(
                rank['all_observed_loss_finite'] is False for rank in r['ranks']):
            raise ValueError('Completed C1 round lacks finite observed losses')
        total+=writes;zero+=writes==0;guards+=bool(guard)
        if r['cumulative_adam_writes_per_rank']!={str(i):total for i in range(8)}:
            raise ValueError('Cumulative actual Adam counts differ from observed round writes')
    if declared_totals!={str(i):total for i in range(8)}:
        raise ValueError('Completion declaration inflates or omits actual Adam updates')
    # Zero-write normal-guard rounds are admissible. Do not require the
    # conservative audit.initial_acceptance_verified positive-write flag.
    if not audit['sixth_real_generation_observed'] or not all(audit['native5'][key] for key in (
            'pre_evaluation_saved','post_evaluation_saved','same_world_reload_verified',
            'all_native_files_hash_verified')):
        raise ValueError('Initial5 native boundary and actual sixth generation are unverified')
    return dict(consumed_rounds=150,actual_adam_per_rank=total,zero_write_rounds=zero,normal_guard_rounds=guards)
