#!/usr/bin/env python3
"""Close ignored-source and upstream-dependency gaps in isolated backup branches."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('publication',ROOT/'publish_code.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
EXTRA={'.c','.cc','.cpp','.h','.hpp','.cu','.cuh','.rs','.go','.js','.ts','.tsx','.cmake'}
SESSIONS={'search-g1':'01a0fb1b-d3ff-74b3-aa13-49ca57871f92','gigpo':'01a0fcbb-863f-7450-926d-db537f648265','tree-grpo':'01a0fcbc-0b46-7502-8b66-d8b21a0e032d','bapo':'01a0fcbd-41d8-7910-a410-372ce5c8870c','opsd':'01a0fcbd-c50e-7cf1-8073-256a13ac9f7e','topd':'01a0fcbe-15f8-7600-ae84-9487d924560a','cosearch':'01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef','search-e1':'01a1028b-b626-7f91-93db-0099ff1623c5'}

for config in p.source_config():
    label=config['label']; repo=ROOT/label; original=Path(config['repo'])
    result_path=ROOT/(label+'-result.json'); result=json.loads(result_path.read_text())
    mf=repo/'reproduction/code-backup'/label/'FILES-SHA256.json'; meta=json.loads(mf.read_text())
    records={f['destination']:f for f in meta['files']}; changed=0
    def save(path,destination,role):
        global changed
        raw=p.stable_bytes(path); dst=str(destination); old=records.get(dst)
        if old and old['sha256']==p.digest(raw):return
        target=repo/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(path.stat().st_mode&0o777)
        records[dst]=dict(source=str(path),destination=dst,bytes=len(raw),sha256=p.digest(raw),role=role);changed+=1
    p.EXTENSIONS.update(EXTRA)
    for f,relative in p.loose_files(config):
        save(f,Path('reproduction/code-backup')/label/'loose'/relative,'latest_loose_source_final_sweep')
    patterns=['*'+x for x in p.EXTENSIONS]
    ignored=p.git(original,'ls-files','--others','--ignored','--exclude-standard','-z','--',*patterns).stdout.split(b'\0')
    for raw in ignored:
        if not raw:continue
        name=raw.decode();f=original/name
        if f.is_file() and not f.is_symlink() and not any(part in p.SKIP_DIRS or 'site-packages' in part for part in Path(name).parts) and f.stat().st_size<=20*1024*1024:
            save(f,Path(name),'ignored_current_checkout_implementation')
    if label=='cosearch':
        author=Path(config['root'])/'work/cosearch-author-review'
        for name in ['LICENSE','.gitmodules']:
            raw=p.git(author,'show','763bf8c4d7caa363ad845d39ddfd53b81ae377bd:'+name).stdout
            target=repo/'reproduction/code-backup/cosearch/author_source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            records[str(target.relative_to(repo))]=dict(source='author_commit:763bf8c4:'+name,destination=str(target.relative_to(repo)),bytes=len(raw),sha256=p.digest(raw),role='author_license_and_pinned_submodule')
        sub='598e61bd1d36895726d28a8d06b3a15bed19f5d3'
        tree=p.git(original,'ls-tree','-r','--name-only',sub).stdout.decode().splitlines()
        for name in tree:
            if Path(name).suffix.lower() not in p.EXTENSIONS|{'.md','.txt','.json'} and Path(name).name not in {'LICENSE','Makefile','Dockerfile'}:continue
            raw=p.git(original,'show',sub+':'+name).stdout
            if p.SECRET.search(raw):raise RuntimeError('Credential pattern in pinned upstream dependency')
            target=repo/'reproduction/code-backup/cosearch/author_source/Search-R1'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            records[str(target.relative_to(repo))]=dict(source='pinned_Search-R1_commit:'+sub+':'+name,destination=str(target.relative_to(repo)),bytes=len(raw),sha256=p.digest(raw),role='exact_author_pinned_Search-R1_submodule')
        meta['current_runtime_author_commit']='763bf8c4d7caa363ad845d39ddfd53b81ae377bd';meta['author_Search_R1_submodule_commit']=sub
    if label=='search-e1':
        for name in ['publish_code.py','verify_remote.py','supplement_backups.py']:
            save(ROOT/name,Path('reproduction/code-backup/publication-tools')/name,'publication_and_verification_tool')
        save(Path('/Users/shaoshitong/Documents/remote_codex_control/memories/code-publication-before-node-release.md'),Path('reproduction/code-backup/publication-tools/code-publication-before-node-release.md'),'user_required_persistent_backup_rule')
    # Refresh changed tracked source/doc files only in this task's copied repository.
    names=p.git(original,'diff','--name-only','-z','HEAD').stdout.split(b'\0')+p.git(original,'ls-files','--others','--exclude-standard','-z').stdout.split(b'\0')
    for raw in names:
        if not raw:continue
        name=raw.decode();f=original/name
        if f.is_file() and not f.is_symlink() and f.suffix.lower() in p.EXTENSIONS|{'.md','.json','.txt'} and f.stat().st_size<=10*1024*1024:
            save(f,Path(name),'latest_current_checkout_working_copy')
    meta.update(session_id=SESSIONS[label],final_source_sweep_utc=datetime.now(timezone.utc).isoformat(),original_head_at_final_sweep=p.git(original,'rev-parse','HEAD').stdout.decode().strip(),files=list(records.values()))
    mf.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    p.git(repo,'add','--all');p.git(repo,'add','--force','--','reproduction/code-backup')
    forced=[f['destination'] for f in records.values() if f['role']=='ignored_current_checkout_implementation']
    if forced:p.git(repo,'add','--force','--',*forced)
    p.git(repo,'commit','--quiet','-m','Bind session identity and preserve ignored source and pinned dependencies')
    sha=p.git(repo,'rev-parse','HEAD').stdout.decode().strip();p.verify_files(repo,sha,meta['files'])
    result.update(session_id=SESSIONS[label],commit=sha,status='committed_pending_push',snapshot_files=len(meta['files']),snapshot_bytes=sum(f['bytes'] for f in meta['files']),final_source_sweep_utc=meta['final_source_sweep_utc'],original_head_at_final_sweep=meta['original_head_at_final_sweep'],supplement_changed_files=changed)
    result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'label':label,'commit':sha,'snapshot_files':len(meta['files']),'supplement_changes':changed}),flush=True)
