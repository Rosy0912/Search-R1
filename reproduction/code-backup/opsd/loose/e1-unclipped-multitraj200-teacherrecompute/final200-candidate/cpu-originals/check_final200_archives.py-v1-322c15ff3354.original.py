"""Replay all200 actual paired training archives; never final-test/effect proof."""
import argparse
from pathlib import Path
import time

SCOPE='ALL200_RAW_PAIRED_TRAINING_ARCHIVES_AND_DATA_CURSOR_CONTENT_INSPECTION_NOT_FINAL_EVALUATION'


def load_common(expected_sha):
    import hashlib,os,re,stat,types
    p=Path(__file__).resolve().parent/'common.py';before=p.stat()
    if not(re.fullmatch('[0-9a-f]{64}',expected_sha) and p.resolve(strict=True)==p and before.st_uid==os.getuid()):raise ValueError('Exact owned common source required')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    names=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    if not stat.S_ISREG(opened.st_mode) or any(tuple(getattr(x,k) for k in names)!=tuple(getattr(before,k) for k in names) for x in (opened,after,p.stat())) or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Common changed before exact-byte compilation')
    module=types.ModuleType('final200_common');module.__file__=str(p);exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),module.__dict__)
    module.read_file(p,expected_sha);return module


def validate_question_row(q,row,position,ordinal,archive,c):
    c.typed(q['dataset_row_position'],ordinal,'Archived dataset row differs from stored200 sampler order')
    c.typed(q['question_position'],position,'Archived question position differs')
    prompt=archive.plain(row['prompt']);extra=archive.plain(row['extra_info']);reward=archive.plain(row['reward_model'])
    c.typed(q['question_uid'],extra['index'],'Archived question ID differs from actual common parquet row')
    c.typed(q['data_source'],row['data_source'],'Archived source differs from actual common parquet row')
    c.require(c.canonical(q['original_prompt'])==c.canonical(prompt) and c.canonical(q['accepted_answers'])==c.canonical(reward['ground_truth']['target']),'Archived input/privileged target differs from actual common parquet row')
    c.typed(q['accepted_answers_provenance'],'common_training_reward_model.ground_truth.target','Archived target provenance differs')
    prefix=archive.E1_INSTRUCTION+'\nQuestion: '
    c.require(len(prompt)==1 and prompt[0]['role']=='user' and prompt[0]['content'].startswith(prefix),'Actual common instruction differs')
    c.typed(q['question'],prompt[0]['content'][len(prefix):],'Archived question text differs from common row')


def check(a,c):
    execution=c.execution_identity(a.execution_job_id,a.execution_step_id);started=time.monotonic()
    inventory,irec=c.read_json(a.inventory,a.inventory_sha256);c.validate_inventory(inventory)
    context=c.accepted_context(a.accepted5_receipt_sha256);c.require(inventory['accepted5']==context['accepted5'],'Historical context changed')
    native,nrec=c.read_json(a.native_report,a.native_report_sha256)
    for key,target in dict(version=1,scope='NATIVE_CPU_FINAL200_STORED_CHECKPOINT_CONTENT_INSPECTION_NOT_LIVE_RESTORE',passed=True,source_commit=c.SOURCE,config_sha256=c.CONFIG,job_id='3304',run_root=str(c.RUN),checkpoint_step=200,inventory_sha256=a.inventory_sha256,inspector_source_sha256=a.native_inspector_sha256,cuda_is_initialized=False).items():c.typed(native.get(key),target,'Native200 archive prerequisite differs: '+key)
    c.require(c.canonical(native['clock'])==c.canonical(c.clock()) and native['accepted5']==context['accepted5'],'Native200 clock/context differs')
    _,isource=c.read_file(a.native_inspector,a.native_inspector_sha256)
    modules,records,source_map,import_audit=c.cpu_modules(context);archive=modules['trajectory_archive'];state=modules['full_state']
    records.update({r['physical_path']:r for r in (irec,nrec,isource)})
    import torch
    import pandas as pd
    from transformers import AutoTokenizer
    c.require(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'NativeTorch2.4 CPU only')
    files=c.verified_members(inventory);records.update({rec['physical_path']:rec for rec in list(inventory['checkpoint_files'].values())+list(inventory['archive_files'].values())})
    config=context['config'];base=Path(config['actor_rollout_ref']['model']['path']);modules['e1_profile'].verify_official_assets(base,base/'VERIFIED-MODEL-MANIFEST.json')
    official,orec=c.read_json(base/'VERIFIED-MODEL-MANIFEST.json',c.MODEL);records[orec['physical_path']]=orec
    for item in official['files']:
        _,rec=c.read_file(base/item['file'],item['sha256']);records[rec['physical_path']]=rec
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    data_manifest=context['manifest']['files']['data_manifest'];common,crec=c.read_json(data_manifest['path'],c.COMMON_DATA,c.RUN);records[crec['physical_path']]=crec
    train=Path(config['data']['train_files']);split=common['splits']['train'];_,trec=c.read_file(train,split['sha256']);records[trec['physical_path']]=trec
    frame=pd.read_parquet(train,columns=['prompt','data_source','reward_model','extra_info']);c.require(len(frame)==169028,'Actual common training row count differs')
    dpt=files['driver/driver.pt'];driver=torch.load(dpt,map_location='cpu',weights_only=False)
    c.require(state.state_digest(driver)==inventory['driver_report']['digest']==native['driver']['state_digest'],'Actual200 driver content differs from native qualification')
    order=driver['sampler']['order'];c.require(type(order) is list and len(order)==169028 and all(type(i) is int for i in order) and set(order)==set(range(169028)) and driver['sampler']['cursor']==driver['sampler']['consumed_cursor']==102400,'Actual200 sampler permutation/cursor differs')
    worker_pids={w['rank']:w['pid'] for w in context['workers']};summaries=[];uids=set()
    for outer in range(1,201):
        directory=c.RUN/'trajectories'/f'outer_{outer:06d}'
        identity=dict(version=1,profile='multitraj200',outer_step=outer,job_id='3304',source_commit=c.SOURCE,config_sha256=c.CONFIG,common_manifest_sha256=c.COMMON_DATA,teacher_revision=archive.REVISION,teacher_model_path=config['baselines']['distillation']['teacher_path'],driver_pid=context['manifest']['driver_pid'],directory=str(directory))
        role=f'trajectory_outer_{outer:06d}/paired-manifest.json';rec=inventory['archive_files'][role]
        paired,_=c.read_json(rec['physical_path'],rec['sha256'],c.RUN)
        c.require(c.canonical(paired['identity'])==c.canonical(identity),'Exact archived outer identity differs')
        for key,target in dict(version=1,student_count=2560,teacher_count=512,teacher_diagnostic_answers_used_as_targets=False,actual_optimizer_steps_per_rank=10.0).items():c.typed(paired.get(key),target,'Paired200 writer field differs: '+key)
        for name in c.ARCHIVE_NAMES:
            key=f'trajectory_outer_{outer:06d}/{name}';record=inventory['archive_files'][key]
            if key in context['manifest']['files']:c.require(dict(path=record['physical_path'],sha256=record['sha256'])==context['manifest']['files'][key],'Historical1..5 archive bytes changed')
        raw=archive.validate_archive(directory,outer,identity,worker_pids,tokenizer)
        request_rec=inventory['archive_files'][f'trajectory_outer_{outer:06d}/questions.json'];request,_=c.read_json(request_rec['physical_path'],request_rec['sha256'],c.RUN)
        for position,q in enumerate(request['questions']):
            ordinal=order[(outer-1)*512+position];row=frame.iloc[ordinal].to_dict();validate_question_row(q,row,position,ordinal,archive,c)
            c.require(q['question_uid'] not in uids,'Repeated question before exhausting epoch0');uids.add(q['question_uid'])
        summaries.append(dict(outer=outer,paired_manifest_sha256=rec['sha256'],student_count=raw['student_count'],teacher_count=raw['teacher_count'],actual_worker_bindings_verified=raw['actual_worker_bindings_verified']))
    c.require(len(uids)==102400 and sum(x['student_count'] for x in summaries)==512000 and sum(x['teacher_count'] for x in summaries)==102400 and all(x['actual_worker_bindings_verified'] is True for x in summaries),'All200 archive counts differ')
    own=c.read_file(Path(__file__).resolve())[1];records[own['physical_path']]=own;commonrec=c.read_file(Path(c.__file__),a.common_sha256)[1];records[commonrec['physical_path']]=commonrec
    c.require(not torch.cuda.is_initialized(),'CPU archive replay initialized CUDA');c.recheck(records);c.source_files(c.REPO,context['closure'])
    return dict(version=1,scope=SCOPE,passed=True,source_commit=c.SOURCE,config_sha256=c.CONFIG,job_id='3304',run_root=str(c.RUN),checkpoint_step=200,inventory_sha256=a.inventory_sha256,native_report_sha256=a.native_report_sha256,native_inspector_source_sha256=a.native_inspector_sha256,inspector_source_sha256=own['sha256'],common_source_sha256=commonrec['sha256'],execution_identity=execution,accepted5=context['accepted5'],import_audit=import_audit,outer_count=200,member_count=4000,student_count=512000,teacher_count=102400,unique_common_question_count=102400,archived_questions_exact_stored200_sampler_and_common_parquet=True,official_tokenizer_replayed=True,outers=summaries,verified_input_records=records,cuda_is_initialized=False,seconds=time.monotonic()-started,limitations=['Training paired diagnostics only, not final51713 raw evaluation','Privileged teacher short generation is independent diagnostic, not supervised target or teacher search trajectory','Stored200 cursor/order is qualified separately from actual logs/optimizer history','No training restore, terminal, resource release or effect claim'])


def main():
    p=argparse.ArgumentParser()
    for name in ('common-sha256','inventory','inventory-sha256','accepted5-receipt-sha256','native-report','native-report-sha256','native-inspector','native-inspector-sha256','execution-job-id','execution-step-id','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();c=load_common(a.common_sha256);value=check(a,c);record=c.write_exclusive(a.output,value)
    print(c.canonical(dict(scope=SCOPE,passed=True,output=record,stored_training_job_id='3304',execution_identity=value['execution_identity'])).decode())


if __name__=='__main__':main()
