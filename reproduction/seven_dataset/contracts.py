"""File/row contracts for the fixed seven-dataset study; no training imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path

COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa':12576}, musique=2417, bamboogle=125)
TOTAL = 51713
FINAL_SHA = '491a7643170d06d7c4a25906aced2504329cb83652b2e9e53159507306141713'
IDS_SHA = '4e9df91a75beba40c3316576f6da049851e736df4a1e5c873f4f0cdb883fd4b1'
DATA_SHA = '7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
GROUPS = {'original','C1-root-only','C2-random-final','budget-5','budget-20',
          'generation-2','position-early','position-late'}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2), b''):h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()


def communication_transport(plan):
    """Allow only the exact transport independently exercised on eight GPUs."""
    transport=plan.get('communication_transport',{})
    if not transport:return {}
    require(transport=={'NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1'},
            'Unverified communication transport')
    path=Path(plan['communication_probe'])
    require(sha(path)==plan['communication_probe_sha256'],'Communication probe changed')
    proof=json.loads(path.read_text())
    require(proof['status']=='EIGHT_RANK_COMMUNICATION_PROBE_PASSED' and
            proof['transport']==transport and {r['rank'] for r in proof['ranks']}==set(range(8)) and
            all({c['mib'] for c in r['cases']}=={1,64,256} and
                all(c['broadcast_and_reduce_values_verified'] for c in r['cases'])
                for r in proof['ranks']), 'Eight-rank communication probe incomplete')
    return dict(transport)


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def validate_milestone(group, step):
    allowed = [50] if group in ('position-early','position-late') else [50,100,150]
    if group=='original':allowed.append(200)
    require(group in GROUPS and type(step) is int and step in allowed,'Unrequested milestone')


def milestone_job_name(group, step, checkpoint_identity, backend):
    validate_milestone(group,step)
    require(len(checkpoint_identity)==64 and all(c in '0123456789abcdef' for c in checkpoint_identity),
            'Invalid model identity')
    if backend=='pj_pt':
        short={'original':'orig','C1-root-only':'c1','C2-random-final':'c2',
               'budget-5':'b5','budget-20':'b20','generation-2':'d2',
               'position-early':'early','position-late':'late'}[group]
        return 'rt17-seven-'+short+'-'+str(step)+'-'+checkpoint_identity[:8]
    require(backend=='slurm_lumia','Unsupported scheduler')
    return 'rt16-seven-'+group.lower().replace('generation','depth')+'-'+str(step)+'-'+checkpoint_identity[:8]


def validate_scheduler_binding(plan, intent, submission, environ, plan_sha):
    """Bind a stable model job name to the scheduler's actual allocation ID."""
    require(intent['job']==plan['job'] and intent['plan_sha256']==plan_sha,
            'Submission intent does not bind this plan')
    require(environ.get('SEARCH_R1_EXECUTION_BACKEND')==plan['backend'] and
            environ.get('RECURSIVE_SOURCE_COMMIT')==plan['evaluation_code_commit'],
            'Actual execution backend/source differs')
    if plan['backend']=='pj_pt':
        require(not environ.get('SLURM_JOB_ID') and environ.get('SEARCH_R1_JOB_ID')==plan['job'],
                'PT allocation identity differs')
        return plan['job']
    require(plan['backend']=='slurm_lumia','Unsupported scheduler')
    actual=environ.get('SLURM_JOB_ID','')
    require(actual.isdigit() and environ.get('SLURM_JOB_NAME')==plan['job'],
            'Actual Slurm allocation name/ID differs')
    require(submission is not None and submission['job_id']==actual and
            submission['job']==plan['job'] and submission['plan_sha256']==plan_sha,
            'Slurm allocation lacks its exact successful submission receipt')
    return actual


def validate_manifest(plan, manifest):
    validate_milestone(plan['group'],plan['outer'])
    i=manifest['identity'];step=plan['outer']
    require(manifest['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and
            i['completed_step']==step and manifest['driver']['next_step']==step+1,
            'Not a completed milestone boundary')
    require(i['config_sha256']==plan['training_config_sha256'] and
            i['data_identity']==DATA_SHA and
            i['job_id']==plan['training_job'] and i['code_commit']==plan['training_code_commit'],
            'Training identity differs')
    require(i['guard_continue_accounting']['completed_outer_updates']==step,
            'Milestone lacks actual completed updates')
    workers=manifest['worker_states']
    require(len(workers)==8 and {w['identity']['rank'] for w in workers}==set(range(8)),
            'Milestone does not cover eight ranks')
    require(all(w['identity']['world_size']==8 and w['identity']['completed_step']==step and
                w['identity']['config_sha256']==i['config_sha256'] and
                w['identity']['job_id']==i['job_id'] and w['identity']['code_commit']==i['code_commit'] and
                w['full_worker_state_finite'] and w['scheduler_last_epoch']==step
                for w in workers),'Worker milestone identity differs')
    require(manifest['hf_identity']['files']==plan['model_files'], 'HF file identity differs')
    required={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
              'model.safetensors.index.json'}
    require(required<=set(plan['model_files']) and
            any(n.endswith('.safetensors') for n in plan['model_files']), 'Incomplete HF export')
    return i


def validate_model_files(plan):
    root=Path(plan['model']).resolve(strict=True)
    require(root.is_dir(),'Model directory missing')
    for name,expected in plan['model_files'].items():
        require(Path(name).name==name,'HF file name escapes directory')
        p=root/name
        require(p.is_file() and not p.is_symlink() and sha(p)==expected,'HF file changed: '+name)
    index=json.loads((root/'model.safetensors.index.json').read_text())
    shards=set(index['weight_map'].values())
    require(shards and shards=={n for n in plan['model_files'] if n.endswith('.safetensors')},
            'Model index does not cover exactly the bound shards')
    return digest(plan['model_files'])


def validate_expected_rows(expected):
    ids=[r['question_id'] for r in expected]
    require(len(ids)==TOTAL and len(set(ids))==TOTAL and ids_hash(ids)==IDS_SHA,
            'Full split ID coverage/order differs')
    require(Counter(r['source'] for r in expected)==COUNTS,'Seven source counts differ')
    return ids


def read_prediction_rows(path):
    """JSONL is delimited by LF, not Unicode characters inside JSON strings."""
    rows=[]
    with Path(path).open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'),'Unterminated prediction record')
            require(line.strip(),'Empty prediction record')
            rows.append(json.loads(line))
    return rows


def validate_replacement_state(state, elapsed, start, reason=None):
    if reason == 'UNSTARTED_DEPENDENCY_BLOCKED':
        require(state.split()[0]=='CANCELLED' and elapsed==0 and start in ('Unknown','None',''),
                'Queued replacement requires cancelled allocation that never started')
    else:
        require(reason is None and state=='FAILED','Failed replacement state differs')


def accept_rows(rows,expected,checkpoint_identity,em_check):
    validate_expected_rows(expected)
    require(len(rows)==TOTAL,'Incomplete full evaluation')
    groups={source:0 for source in COUNTS}
    for index,(row,want) in enumerate(zip(rows,expected)):
        require(row['final_index']==index and row['question_id']==want['question_id'] and
                row['source']==want['source'] and row['gold_answers']==want['gold_answers'] and
                row['checkpoint_identity']==checkpoint_identity,'Row identity/order/gold mismatch')
        prediction=row['prediction']
        require(prediction is None or isinstance(prediction,str),'Invalid prediction')
        em=int(bool(em_check(prediction,want['gold_answers']))) if prediction is not None else 0
        require(row['em'] in (0,1) and row['em']==em,'Saved EM differs from prediction')
        groups[row['source']]+=em
    by_source={s:dict(correct=groups[s],count=n,em=groups[s]/n) for s,n in COUNTS.items()}
    return dict(count=TOTAL,datasets=by_source,correct=sum(groups.values()),
                equal_dataset_em=sum(r['em'] for r in by_source.values())/len(by_source),
                all_question_em=sum(groups.values())/TOTAL,ordered_ids_sha256=IDS_SHA)


def compare_paired(baseline,candidate):
    require(len(baseline)==len(candidate)==TOTAL,'Paired comparison needs complete seven sets')
    wins=losses=0
    for a,b in zip(baseline,candidate):
        require(a['question_id']==b['question_id'] and a['source']==b['source'] and
                a['gold_answers']==b['gold_answers'],'Paired question identities differ')
        require(a['em'] in (0,1) and b['em'] in (0,1),'Paired EM must be binary')
        wins+=int(b['em']>a['em']);losses+=int(b['em']<a['em'])
    return dict(count=TOTAL,candidate_only_correct=wins,baseline_only_correct=losses,
                delta_all_question_em=(wins-losses)/TOTAL)
