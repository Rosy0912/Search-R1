"""CPU results acceptance, preserving terminal skips; never infer teacher benefit."""
import json
from pathlib import Path


def owned_leftovers(processes,owner):
    return [p for p in processes if (p['pid'],p['uid'],p['starttick']) != (owner['pid'],owner['uid'],owner['starttick'])]


def phase_acceptance(run, phase, expected, run_id, children, sha, region, state_api, core):
    from .numeric_regions import probe as numeric_probe
    found = {}; artifacts = []; summaries = []; generations = 0; skips = 0
    for index in range(3):
        folder = Path(run) / phase / f'worker-{index}'
        final = json.loads((folder / 'worker-final.json').read_text())
        identity = json.loads((folder / 'identity.json').read_text())
        child = next(c for c in children if c['worker_index']==index)
        if final['status'] != 'WORKER_COMPLETE' or final['run_id'] != run_id or final['phase'] != phase or identity['pid'] != child['pid'] or identity['uid'] != child['uid'] or identity['proc_starttick']!=child['starttick'] or identity['e1_run_id']!=run_id or identity['gpu_uuid']!=child['gpu_uuid']:
            raise ValueError('Worker terminal identity mismatch')
        control = json.loads((folder / 'zero-b-control.json').read_text())
        if control.get('accepted') is not True or control.get('actual_generate_tap_null_effect') is not True:
            raise ValueError('GPU zero-B/null-effect gate missing')
        expected_ids = [r['case_id'] for i,r in enumerate(expected['all']) if i%3==index and r in expected[phase]]
        if final['assigned_case_ids'] != expected_ids or final['completed_case_ids'] != expected_ids:
            raise ValueError('Exact assigned/completed cohort differs')
        for case_id in expected_ids:
            target = folder / case_id; record = json.loads((target / 'preparation.source.json').read_text())
            source = next(r for r in expected[phase] if r['case_id']==case_id)
            if state_api.sha(record) != state_api.sha({k:v for k,v in source.items() if not k.startswith('_')}): raise ValueError('Case prepared state identity changed')
            completion = json.loads((target / 'completion.json').read_text())
            if completion['case_id'] != case_id or completion['run_id'] != run_id or completion['phase'] != phase or case_id in found:
                raise ValueError('Duplicate/missing/wrong run completion')
            histories=[('original_vllm',source['original_student'])]+[('actual_hf_'+v,x) for v,x in sorted(source['actual_hf_states'].items())]
            conditions=source['reference_interventions']['conditions'];expected_active=set();expected_inactive=[]
            for history,states in histories:
                expected_inactive += [(history,x['boundary'],x['reason']) for x in states['skipped_boundaries']]
                for active in states['executable_states']:
                    expected_active.update((history,active['state_sha256'],'teacher',c['name'],'source100') for c in conditions)
                    expected_active.update((history,active['state_sha256'],'student','none',v) for v in ('source100','epoch1'))
            actual_active=[];actual_inactive=[]
            for x in completion['slots']:
                if 'state_sha256' in x:actual_active.append((x['history'],x['state_sha256'],x['role'],x['reference_condition'],x['version']))
                else:actual_inactive.append((x['history'],x['aliases'][0],x['reason']))
            if set(actual_active)!=expected_active or len(actual_active)!=len(expected_active) or sorted(actual_inactive)!=sorted(expected_inactive):
                raise ValueError('Actual branch/skip inventory differs from prepared protocol')
            required_summaries={f'existing-hf-{v}-forced-{t}.summary.json' for v in source['actual_hf_states'] for t in ('source100','update1','update5','epoch1')}
            required_summaries.update('original-y-ref-'+c['name']+'.summary.json' for c in conditions if c['name'] in ('remove_final_answer','mismatched'))
            for slot in completion['slots']:
                if slot['status']=='skipped': skips+=1;continue
                if slot['status']!='complete':raise ValueError('Unknown result status')
                generation_path=target/slot['generation_file']; trace=json.loads(generation_path.read_text()); generations+=1
                from search_r1.qa_protocol import extract_final_answer
                spans=trace['policy_spans'];response=trace['response']
                answer=extract_final_answer([response[a:b] for a,b in spans])
                if core.reward(response,source['source_train_row']['row']['answers'],spans)!=trace['protocol_em'] or (answer is not None)!=trace['protocol_valid'] or answer!=trace['accepted_final_answer']:
                    raise ValueError('Frozen native CPU reward/final protocol mismatch')
                condition=next(c for c in conditions if c['name']==slot['reference_condition']) if slot['role']=='teacher' else {'reference':None}
                content=core.prompt(source['source_train_row']['row']['question'],condition['reference'])[0]['content']
                if content not in trace['prompt_text']:raise ValueError('Condition prompt literal content differs')
                current={k:trace['initial_state'][k] for k in state_api.initial(trace['source_history_backend'])}
                for event in trace['turns']:
                    if state_api.sha(current)!=state_api.sha(event['state_before']):raise ValueError('Runtime state before mismatch')
                    if event['context_text'] != state_api.context(current,trace['prompt_text']):raise ValueError('Actual executor context differs')
                    if event.get('skipped'):
                        if event['skipped']!='context_limit_without_independent_truncation' or event['context_tokens']+500<=8192:
                            raise ValueError('False context-limit skip')
                        current['terminal']='context_limit';current['forced_pending']=False;break
                    if event['context_tokens']+500>8192:raise ValueError('Generation exceeded original context budget')
                    current=state_api.advance(current,event['policy_segment'],event.get('inserted_information'))
                    if state_api.sha(current)!=state_api.sha(event['state_after']):raise ValueError('Runtime transition mismatch')
                if state_api.sha(current)!=state_api.sha(trace['final_state']):raise ValueError('Final state mismatch')
                stem=slot['generation_file'].removesuffix('.generation.json')
                required_summaries.update(stem+'-forced-'+v+'.summary.json' for v in ('source100','update1','update5','epoch1'))
                artifacts.append(dict(path=str(generation_path),sha256=sha(generation_path)))
            if {x.name for x in target.glob('*.summary.json')}!=required_summaries:raise ValueError('Exact cross-forcing/probe summary inventory differs')
            invariant_groups={}
            # Probe numerical reconstruction uses independently frozen M2 CPU API.
            for path in sorted(target.glob('*.summary.json')):
                summary=json.loads(path.read_text())
                if summary.get('status')=='skipped':skips+=1;continue
                if summary.get('status')!='complete':raise ValueError('Unknown probe status')
                token=path.with_name(path.name.removesuffix('.summary.json'))/'tokens.jsonl'
                # Fixed original-y and existing M1 trace summaries use their own
                # response; continuation branch names bind exact generation owner.
                origin=summary['response_origin']
                if origin=='counterfactual_reference_fixed_original_y':
                    pair=source['_original_pair'];response=pair['student'];spans=pair['student_policy_spans']
                elif path.name.startswith('existing-hf-'):
                    owner=summary['history_owner'];artifact=source['actual_hf_states'][owner]['artifact']
                    trace=json.loads(Path(artifact['path']).read_text());response=trace['response'];spans=trace['policy_spans']
                else:
                    name=path.name.split('-forced-')[0]+'.generation.json';trace=json.loads((target/name).read_text())
                    response=trace['response'];spans=trace['policy_spans'][len(trace['initial_state']['policy_spans']):]
                proof=numeric_probe(region,summary,token,response,spans,teacher_present=True)
                if proof.get('reconstruction_passed') is not True or summary.get('all_token_metrics_finite') is not True:raise ValueError('CPU region/native finite loss reconstruction failed')
                summaries.append(dict(path=str(path),sha256=sha(path),tokens_sha256=sha(token)))
                if origin!='counterfactual_reference_fixed_original_y':
                    key=path.name.rsplit('-forced-',1)[0]
                    invariant_groups.setdefault(key,[]).append((token,proof,summary['version']))
            for key,items in invariant_groups.items():
                if len(items)!=4 or {x[2] for x in items}!={'source100','update1','update5','epoch1'}:
                    raise ValueError('Four-checkpoint common response probe missing')
                for other in items[1:]:region.teacher_invariance(items[0][0],other[0],items[0][1]['token_file'],other[1]['token_file'])
            found[case_id]=True
    ids=[r['case_id'] for r in expected[phase]]
    if set(found)!=set(ids):raise ValueError('Cohort completion coverage differs')
    return dict(accepted=True,run_id=run_id,phase=phase,completed_case_ids=ids,
                state_reconstruction_accepted=True,null_effect_accepted=True,
                actual_generation_branches=generations,skip_count=skips,
                generation_artifacts=artifacts,probe_summaries=summaries,
                teacher_benefit_conclusion=False)
