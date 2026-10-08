"""Independent M3 controller; actual CPU gates before fixed-weight GPU workers.

Never signal another process. Failure stops NEW admissions, waits own spawned
children to exit normally, keeps every output and releases only its own locks.
"""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    from worker import package
    here=Path(__file__).resolve().parent;package('m3exec',here)
    from m3exec.admission import checked,sha,select_phase,validate_config,fresh_retriever,validate_m4_pane,proc_identity,fresh_base_weights,validate_m4_worker_identity
    from m3exec.acceptance import phase_acceptance
    from m3exec.ownership import strict_scan
    from m3exec.supervision import supervise
    from m3exec.release_wait import await_release as wait_for_release
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);a=p.parse_args()
    c=checked(dict(path=a.config,sha256=a.config_sha256));c['_config_sha256']=a.config_sha256;validate_config(c)
    if os.getuid()!=1005 or c.get('uid')!=1005 or os.environ.get('E1_RUN_ID')!=c['run_id']:
        raise ValueError('Controller requires actual UID1005/private matching E1_RUN_ID')
    if c.get('physical_gpus')!=[4,5,6]:raise ValueError('Only K1 GPU4-6 are authorized')
    for prefix,key in [('m3debug','debug_helper_manifest'),('m3prep','cpu_helper_manifest'),('m4identity','m4_helper_manifest'),('m2region','region_helper_manifest')]:
        manifest=checked(c[key]);source=Path(c.get(key.replace('_manifest','_source'),Path(c[key]['path']).resolve().parent)).resolve();entries=manifest.get('files',manifest)
        if set(entries)!={x.name for x in source.glob('*.py')}:raise ValueError('Dependency inventory changed')
        for n,d in entries.items():
            if '/' in n or sha(source/n)!=d:raise ValueError('Dependency source changed')
        package(prefix,source)
    manifest=checked(c['execution_manifest'])
    if set(manifest['files'])!={x.name for x in here.glob('*.py')}:raise ValueError('M3 inventory differs')
    for n,d in manifest['files'].items():
        if '/' in n or sha(here/n)!=d:raise ValueError('M3 execution source differs')
    publication=checked(c['publication_proof'])
    if publication.get('accepted') is not True or publication['source_manifest_sha256']!=c['execution_manifest']['sha256'] or publication['source_commit']!=publication['remote_readback_sha'] or len(publication['source_commit'])!=40 or publication['branch']!='search-e1':
        raise ValueError('M3 actual publication/readback proof missing')
    from m4identity import gates as g,controller as m4,utils as u,identity as i
    from m3debug.common import atomic_json,load_science
    from m3debug.admission import RetrieverAdmission
    from m3debug.weight_identity import check_weight_manifest
    from m3prep import state
    from m2region import summarize_regions as region
    from m3exec.admission import fresh_runtime
    runtime_proof=fresh_runtime(c)
    m4config=checked(c['m4_config']);m4config['actual_config_path']=c['m4_config']['path']
    if m4config['debug_source_manifest']['sha256']!=c['debug_helper_manifest']['sha256'] or Path(m4config['runtime']).resolve()!=Path(c['runtime']).resolve() or Path(m4config['model']).resolve()!=Path(c['model']).resolve() or sha(m4config['weight_manifest'])!=sha(c['weight_manifest']):
        raise ValueError('M3 source/runtime/weights differ from actual M4')
    if m4config['helper_source_manifest']['sha256']!=c['m4_helper_manifest']['sha256']:raise ValueError('M4 actual source manifest differs')
    g.publication(m4config,Path(c.get('m4_helper_source',Path(c['m4_helper_manifest']['path']).resolve().parent)))
    if c['approved_gpu_uuids']!=i.GPU_UUIDS or Path(c['debug_root']).resolve()!=Path(m4config['debug_root']).resolve() or Path(c['m4_output_root']).resolve()!=Path(m4config['output_root']).resolve():raise ValueError('GPU/resource lock scope differs from actual M4')
    support=u.load_support(m4config)
    own=u.proc_identity(os.getpid());owner={k:own[k] for k in ('pid','uid','starttick')}
    run=Path(c['output']).resolve();run.mkdir(parents=True,exist_ok=False,mode=0o700)
    locks=[];children=[];admission=None
    def save(name,value):atomic_json(run/name,value);return dict(path=str(run/name),sha256=sha(run/name))
    def release_m4():
        # Root config binds preserved M4 run. Terminal files can appear while
        # this CPU controller waits; absence never means success.
        binding=checked(c['m4_tmux_binding'])
        if binding.get('m4_run')!=c['m4_run'] or binding.get('preserved_before_exit') is not True:
            raise ValueError('M4 preserved owned tmux binding missing')
        raw=subprocess.check_output(['tmux','display-message','-p','-t',binding['pane_id'],'#{pane_dead}\t#{pane_dead_status}\t#{pane_id}\t#{pane_pid}\t#{pane_start_command}'],text=True)
        try:actual_tmux=validate_m4_pane(binding,raw,proc_identity(binding['tmux_server']['pid']))
        except ValueError as error:raise g.NotReady(str(error)) from error
        old=Path(c['m4_run']).resolve();final_path=old/'four-fixed-weights-native512-accepted.json'
        if not final_path.exists():raise g.NotReady('M4 full final not written')
        final=checked(dict(path=str(final_path),sha256=sha(final_path)))
        if final.get('accepted') is not True or final.get('rows')!=2048 or set(final.get('versions',{}))!=set(i.VERSIONS):raise g.NotReady('M4 full2048 not accepted')
        load_science(c['runtime'],with_torch=False)
        rows,_=i.cohort(m4config)
        collected=[];phase_proofs=[];old_known=[];old_controllers=[]
        for phase in ('endpoints','early'):
            path=old/(phase+'-children-final.json');terminal=checked(dict(path=str(path),sha256=sha(path)))
            if terminal.get('complete') is not True or len(terminal['children'])!=3 or any(v!=0 for v in terminal['child_exit_codes'].values()):raise g.NotReady('M4 actual child terminal rc0 missing')
            for child in terminal['children']:
                identity=child['identity'];old_known.append(identity)
                status=json.loads((old/phase/f"worker-{child['worker_index']}"/'status.json').read_text())
                validate_m4_worker_identity(status,identity)
                try:live=u.proc_identity(identity['pid'])
                except (FileNotFoundError,ProcessLookupError):continue
                if (live['uid'],live['starttick'])==(identity['uid'],identity['starttick']):raise g.NotReady('M4 child remains live')
            entries,summary=m4.phase_records(old,phase,m4config,rows);collected+=entries
            if phase=='endpoints' and not m4.endpoint_acceptance(summary,m4config)['accepted']:raise ValueError('M4239/203 endpoint mismatch')
            phase_proofs.append(dict(path=str(path),sha256=sha(path)))
        actual=i.summarize_records(collected,rows,i.VERSIONS)
        if actual['versions']!=final['versions']:raise ValueError('M4 CPU full final metrics mismatch')
        start_paths=list(old.glob('controller-start-*.json'))
        if not start_paths:raise g.NotReady('M4 actual controller identity missing')
        for path in start_paths:
            start=json.loads(path.read_text());identity=start['controller'];old_known.append(identity);old_controllers.append(identity)
            try:live=u.proc_identity(identity['pid'])
            except (FileNotFoundError,ProcessLookupError):continue
            if (live['uid'],live['starttick'])==(identity['uid'],identity['starttick']):raise g.NotReady('M4 controller remains live')
        old_run=old.name.removeprefix('native512-')
        if binding.get('pane_process'):old_known.append(binding['pane_process'])
        process_audit=strict_scan(g,old_run,1005,min(old_controllers,key=lambda x:x['starttick']),old_known,[str(old),str(Path(c['m4_helper_source'])/'worker.py'),str(Path(c['m4_helper_source'])/'controller.py')],c.get('independent_cpu_exclusion_proofs',m4config.get('independent_cpu_exclusion_proofs',[])),expected_live_owner=owner,live=u.proc_identity)
        if not process_audit['accepted_no_unexpected_owned_or_opaque']:raise g.NotReady('M4 owned/opaque process audit blocks release')
        return dict(accepted=True,run_id=c['run_id'],m4_run_id=old_run,actual_tmux_exit=actual_tmux,full_final=dict(path=str(final_path),sha256=sha(final_path)),phase_proofs=phase_proofs,process_audit=process_audit,owned_run_processes=[])
    def resource(allow=None):
        devices,apps=g.gpu_inventory();chosen=[devices[x] for x in [4,5,6]]
        if [d['uuid'] for d in chosen]!=c['approved_gpu_uuids']:raise g.NotReady('GPU UUID changed')
        permitted={x['pid']:x for x in (allow or [])};foreign=[]
        for app in apps:
            if app['gpu_uuid'] not in c['approved_gpu_uuids']:continue
            expected=permitted.get(app['pid'])
            try:identity=u.proc_identity(app['pid'])
            except (FileNotFoundError,ProcessLookupError):
                if expected and expected['process'].poll() is not None:continue # CUDA cleanup lag for a known exited worker.
                raise g.NotReady('Unidentified live GPU allocation')
            if not expected or (identity['uid'],identity['starttick'])!=(expected['uid'],expected['starttick']) or app['gpu_uuid']!=expected['gpu_uuid']:foreign.append(app)
        if foreign:raise g.NotReady('Foreign compute process on GPU4-6')
        if not allow and any(d['actual_free_mib']<20480 for d in chosen):raise g.NotReady('Actual GPU startup free below20GiB')
        free=shutil.disk_usage(run).free
        if free<100<<30:raise g.NotReady('Disk free below100GiB')
        health=fresh_retriever(c,admission) if admission else None
        return dict(accepted=True,run_id=c['run_id'],sample_unix=time.time(),devices=chosen,selected_apps=apps,foreign_selected_apps=[],disk_free_bytes=free,health=health)
    def await_release(label,known):
        def read_release():
            audit=strict_scan(g,c['run_id'],1005,owner,[{k:x[k] for k in ('pid','uid','starttick')} for x in known],
                  [str(here/'worker.py'),str(run)],c.get('independent_cpu_exclusion_proofs',m4config.get('independent_cpu_exclusion_proofs',[])),expected_live_owner=owner,live=u.proc_identity)
            known_clear=True;kernel_proofs=[]
            for child in known:
                try:actual=u.proc_identity(child['pid'])
                except (FileNotFoundError,ProcessLookupError):kernel_proofs.append(dict(pid=child['pid'],absent=True));continue
                kernel_proofs.append(actual)
                if child['starttick'] is None or actual['starttick']==child['starttick']:known_clear=False
            blockers=audit['unexpected_blockers']
            owned_blockers=[b for b in blockers if b['reason'] in ('known_owned_or_run_or_argv_or_descendant_present','known_owned_pid_same_starttick_present_even_if_uid_changed')]
            apps=g.gpu_inventory()[1];own_apps=[]
            for app in apps:
                if app['gpu_uuid'] not in c['approved_gpu_uuids']:continue
                try:actual=u.proc_identity(app['pid'])
                except (FileNotFoundError,ProcessLookupError):raise RuntimeError('Selected GPU allocation identity currently unknown')
                child=next((x for x in known if x['pid']==app['pid']),None)
                if (child and (child['starttick'] is None or actual['starttick']==child['starttick'])) or any(b.get('pid')==app['pid'] for b in owned_blockers):own_apps.append(app)
            no_owned=known_clear and not owned_blockers
            return dict(status='RELEASE_VERIFIED' if audit['accepted_no_unexpected_owned_or_opaque'] and no_owned and not own_apps else 'HOLD_WAIT_RELEASE',
                        released=audit['accepted_no_unexpected_owned_or_opaque'] and no_owned and not own_apps,
                        known_identities_absent_or_reused=known_clear,own_cuda_clear=not own_apps,
                        only_independent_opaque_blockers=bool(blockers) and no_owned and all(b['reason']=='opaque_live_process_not_proven_infrastructure' for b in blockers),
                        process_audit=audit,kernel_known_identity_proofs=kernel_proofs,own_gpu_apps=own_apps)
        return wait_for_release(read_release,lambda value:save(label+'-process-release-audit.json',value),time.sleep)
    try:
        # CPU wait20s, never duplicate M4 work or infer terminal from absence.
        while True:
            try:
                m1=g.m1_completed(m4config,support);release=release_m4()
                for path in (Path(c['debug_root'])/'experiment.lock',Path(c['m4_output_root'])/'experiment.lock',run.parent/'experiment.lock'):
                    handle=path.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
                break
            except (g.NotReady,FileNotFoundError,BlockingIOError,subprocess.CalledProcessError) as error:
                for handle in locks:handle.close()
                locks=[];save('waiting.json',dict(status='WAITING_ACTUAL_RELEASE',reason=str(error),owner=owner));time.sleep(20)
        weights=check_weight_manifest(c['weight_manifest'],c['model'],c['adapters']);i.check_adapter_labels(weights)
        save('fresh-base12-start.json',fresh_base_weights(c['weight_manifest'],c['model']))
        core,_=load_science(c['runtime'],with_torch=False)
        admission=RetrieverAdmission(c['retriever_admission'],c['retriever_url']);admission.check_live()
        summary=checked(c['preparation_summary']);proof=summary['states_output']
        if sha(proof['path'])!=proof['sha256'] or not summary['accepted_cpu_preparation']:raise ValueError('CPU preparation invalid')
        records=[json.loads(line) for line in Path(proof['path']).read_text().splitlines()];pilot=select_phase(records,'pilot')
        case_proof=summary['source']['cases']
        if sha(case_proof['path'])!=case_proof['sha256']:raise ValueError('Case bytes changed')
        cases={x['case_id']:x for x in (json.loads(line) for line in Path(case_proof['path']).read_text().splitlines())}
        enriched=[dict(r,_original_pair=cases[r['case_id']]['original_pair']['row']) for r in records]
        from m3prep.references import select
        if [x['case']['case_id'] for x in select(list(cases.values()))]!=[r['case_id'] for r in records]:raise ValueError('Actual hash-selected64 case order differs')
        if summary['selected_cases']!=64 or summary['pilot_cases']!=16 or len({r['question_id'] for r in records})!=64:raise ValueError('Fixed64/16 preparation coverage differs')
        for r in records:
            case=cases[r['case_id']]
            if state.sha({k:v for k,v in case.items() if k!='case_semantic_sha256'})!=r['source_case_sha256']:raise ValueError('Case semantic identity differs')
        expected={'all':enriched,'pilot':[r for r in enriched if r['pilot']],'remaining':[r for r in enriched if not r['pilot']]}
        releaseproof=save('m4-release-proof.json',release)
        lockproof=save('lock-proof.json',dict(accepted=True,run_id=c['run_id'],owner=owner,locks=[str(h.name) for h in locks]))
        resource();save('controller-start.json',dict(owner=owner,run_id=c['run_id'],m1=m1,weights=weights,config_sha256=a.config_sha256))
        # Immutable worker config gets pilot receipt path before it exists; only
        # remaining workers read it after acceptance writes the expected digest.
        phase_c=dict(c)
        for phase in ('pilot','remaining'):
            fresh_runtime(c)
            save(phase+'-fresh-base12-start.json',fresh_base_weights(c['weight_manifest'],c['model']))
            check_weight_manifest(c['weight_manifest'],c['model'],c['adapters'])
            children=[];resource();inventory=save(phase+'-startup-resource.json',resource())
            if phase=='remaining':phase_c['pilot_receipt']=pilot_proof
            configpath=run/(phase+'-worker-config.json');atomic_json(configpath,phase_c);configsha=sha(configpath)
            def lease(index):
                return dict(schema='m3-external-resource-ticket-v1',accepted=True,run_id=c['run_id'],phase=phase,worker_index=index,
                    owner=owner,gpu_uuid=c['approved_gpu_uuids'][index],config_sha256=configsha,expiry_unix=time.time()+90,
                    renewal_path=str(run/(phase+f'-lease-{index}.json')),m4_fully_released=True,root_lock_held=True,
                    m4_release_proof=releaseproof,lock_proof=lockproof,gpu_inventory_proof=inventory)
            spawn_error=None
            try:
                for index,gpu in enumerate([4,5,6]):
                    ticketpath=run/(phase+f'-initial-ticket-{index}.json');atomic_json(ticketpath,lease(index));atomic_json(run/(phase+f'-lease-{index}.json'),lease(index))
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),E1_RUN_ID=c['run_id'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
                    argv=[c['python'],'-B',str(here/'worker.py'),'--config',str(configpath),'--config-sha256',configsha,'--ticket',str(ticketpath),'--ticket-sha256',sha(ticketpath),'--phase',phase,'--worker-index',str(index)]
                    with (run/(phase+f'-worker-{index}.log')).open('x') as log:child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owned_child=dict(process=child,worker_index=index,gpu_uuid=c['approved_gpu_uuids'][index],pid=child.pid,uid=1005,starttick=None)
                    children.append(owned_child)
                    identity=u.proc_identity(child.pid);owned_child.update(identity)
                    if identity['uid']!=1005 or identity['argv']!=argv or u.environment(child.pid).get('E1_RUN_ID')!=c['run_id'] or u.environment(child.pid).get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Actual worker process/argv/env differs')
                    save(phase+'-children-start.json',[{k:v for k,v in x.items() if k!='process'} for x in children])
            except BaseException as error:spawn_error=repr(error)
            def observe():
                nonlocal inventory
                inventory=save(phase+'-resource-'+str(time.time_ns())+'.json',resource(children))
            def renew(child):atomic_json(run/(phase+f"-lease-{child['worker_index']}.json"),lease(child['worker_index']))
            def progress(error):save(phase+'-progress.json',dict(codes={str(x['pid']):x['process'].poll() for x in children},monitor_error=error))
            def revoke(error):
                for x in children:
                    value=lease(x['worker_index']);value.update(accepted=False,hold_reason=error)
                    atomic_json(run/(phase+f"-lease-{x['worker_index']}.json"),value)
            monitor_error=supervise(children,observe,renew,progress,time.sleep,spawn_error,on_failure=revoke)
            codes={str(x['pid']):x['process'].returncode for x in children};save(phase+'-children-final.json',dict(codes=codes,children=[{k:v for k,v in x.items() if k!='process'} for x in children],spawn_error=spawn_error,monitor_error=monitor_error))
            # Wait all own descendants and CUDA release; no signals/no retry.
            await_release(phase,children)
            if len(children)!=3 or spawn_error or monitor_error or any(v!=0 for v in codes.values()):raise ValueError('M3 worker/resource failure HOLD')
            fresh_runtime(c)
            save(phase+'-fresh-base12-end.json',fresh_base_weights(c['weight_manifest'],c['model']))
            check_weight_manifest(c['weight_manifest'],c['model'],c['adapters'])
            accepted=phase_acceptance(run,phase,expected,c['run_id'],children,sha,region,state,core)
            pilot_proof=save(phase+'-accepted.json',accepted)
            resource()
        save('controller-final.json',dict(status='COMPLETE',accepted=True,run_id=c['run_id'],completed_cases=64,no_signals_sent=True,owner=owner))
        return 0
    except BaseException as error:
        for x in children:x['process'].wait()
        if children:
            try:await_release('failure',children)
            except RuntimeError as release_error:
                if not str(release_error).startswith('HOLD: independent opaque audit blockers;'):raise
                save('failure-release-error.json',dict(error=repr(release_error),known_owned_cuda_release_verified=True))
        save('controller-final.json',dict(status='HOLD',accepted=False,run_id=c['run_id'],error=repr(error),traceback=traceback.format_exc(),owner=owner,no_signals_sent=True))
        return 3
    finally:
        for handle in locks:handle.close()

if __name__=='__main__':raise SystemExit(main())
