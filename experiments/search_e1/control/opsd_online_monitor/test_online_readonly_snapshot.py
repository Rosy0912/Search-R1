"""CPU-only negative/live-race fixtures; no node/GPU/HTTP/model calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import os
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('snapshot_fixture', HERE/'online_readonly_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data)+'\n')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Snapshot(unittest.TestCase):
    def test_actual_private_config_publication_and_observer_field_compatibility(self):
        # Actual persisted receipts, never copied into this source tree/Git.
        # Path translation and a frozen receipt-time clock are explicit: this
        # fixture tests schema, not present Node process/memory health.
        base=Path(os.environ.get('OPSD_ONLINE_SNAPSHOT_TEST_PROOFS',str(HERE.parents[4]/'search-e1-analysis-output/opsd-online-launch-20261008')))
        config,cp=m.value(base/'config.json','d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17')
        config['_config_sha256']=cp['sha256']
        helper=HERE.parent/'opsd_online'
        manifest,mp=m.value(helper/'source-manifest.json',config['helper_source_manifest']['sha256'])
        prefix='_snapshot_actual_config_science_fixture';names=[]
        import sys
        try:
            m.exact_module(prefix,helper/'__init__.py',b'',True);names.append(prefix)
            for name in ('checkpoint','prepare'):
                raw,_=m.read(helper/(name+'.py'),manifest['files'][name+'.py']['sha256'])
                module=m.exact_module(prefix+'.'+name,helper/(name+'.py'),raw);names.append(prefix+'.'+name)
            self.assertEqual(module.validate_config(config,launch=True),[])
        finally:
            for name in reversed(names):sys.modules.pop(name,None)
        binding,_=m.value(base/'cpu-merge-recovery/final-progress/receipts/current-controller.json')
        argv=list(binding['owner']['argv_tokens'])
        replacements={'--source-manifest':str(HERE.parent/'opsd_online_cpu_merge_recovery/source-manifest.json'),
                      '--publication':str(base/'cpu-merge-recovery/actual-recovery-evidence/receipts/cpu-merge-recovery-publication.json'),
                      '--wait-root':str(HERE.parent/'opsd_online_wait_recovery'),
                      '--wait-publication':str(base/'wait-recovery-publication.json')}
        for flag,path in replacements.items():argv[argv.index(flag)+1]=path
        argv[2]=str(HERE.parent/'opsd_online_cpu_merge_recovery/recovery.py')
        external=m.external_identity(argv,config)
        self.assertEqual([r['source_commit'] for r in external],['51eb6c574dd726b929dffcd6c0abfdc7b441a5d4','2cbcc1b4754eff51a52b213240579f5db0e10f87'])
        observer_binding,bp=m.value(base/'memory-binding.json')
        live,lp=m.value(base/'cpu-merge-recovery/final-progress/receipts/memory-live.json')
        native=types.SimpleNamespace(process=lambda pid:dict(observer_binding['observer_identity']),same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','starttick')))
        def mutable(path):
            return (live,lp) if str(path)==observer_binding['live_report_path'] else (observer_binding,bp)
        with patch.object(m,'mutable_value',side_effect=mutable),patch.object(m,'ref_value',return_value=(manifest,mp)),patch.object(m.time,'time',return_value=live['updated_unix']+1):
            report=m.observer(config,native)
        self.assertEqual(report['age_seconds'],1)

    def owner_case(self):
        owner = dict(pid=1,uid=1005,starttick=20,run_id='r',argv=['python','--config-sha256','s','--run-id','r'])
        binding = dict(accepted=True,run_id='r',config_sha256='s',owner=dict(owner,argv_tokens=owner['argv']))
        config = dict(run_id='r',_config_sha256='s',owner=dict(uid=1005))
        native = types.SimpleNamespace(process=lambda pid:dict(owner),same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','starttick')))
        return owner,binding,config,native

    def test_pid_reused_and_wrong_argv_rejected(self):
        for field in ('starttick','run_id','argv'):
            owner,binding,config,native=self.owner_case()
            self.assertEqual(m.controller_identity(config,binding,native)['pid'],1)
            owner[field] = 99 if field=='starttick' else ('foreign' if field=='run_id' else ['other'])
            with self.assertRaises(m.Invalid): m.controller_identity(config,binding,native)

    def test_foreign_gpu_opaque_and_oracle_oversize(self):
        config=dict(run_id='r',owner=dict(gpu_uuids=['g%d'%i for i in range(8)]),resources=dict(training_minimum_free_mib=1024,startup_minimum_free_mib=20480))
        devices=[dict(index=i,uuid='g%d'%i,actual_free_mib=22000) for i in range(8)]
        app=dict(pid=7,gpu_uuid='g7',used_mib=600)
        native=types.SimpleNamespace(inventory=lambda:(devices,[app]),process=lambda pid:dict(pid=7,uid=1005,starttick=20,run_id=None),same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','starttick')),
            owned_gpu_process=lambda *args: (_ for _ in ()).throw(m.Invalid('opaque/foreign')))
        oracle=dict(pid=7,uid=1005,starttick=20)
        self.assertTrue(m.gpu_check(config,dict(pid=1),[],oracle,native)['actual_free_limit_met'])
        for field,new in [('pid',99),('gpu_uuid','foreign'),('used_mib',1025)]:
            old=app[field];app[field]=new
            with self.assertRaises(m.Invalid):m.gpu_check(config,dict(pid=1),[],oracle,native)
            app[field]=old

    def test_unused_card_uses_startup_floor_and_cpu_parent_context_is_attention(self):
        config=dict(run_id='r',owner=dict(gpu_uuids=['g%d'%i for i in range(8)]),resources=dict(training_minimum_free_mib=1024,startup_minimum_free_mib=20480))
        devices=[dict(index=i,uuid='g%d'%i,actual_free_mib=22000) for i in range(8)];devices[0]['actual_free_mib']=2048
        apps=[];owner=dict(pid=1,uid=1005,starttick=20,run_id='r')
        native=types.SimpleNamespace(inventory=lambda:(devices,apps),process=lambda pid:dict(owner,pid=pid),same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','starttick')),owned_gpu_process=lambda *args:dict(owner,pid=2))
        oracle=dict(pid=7,uid=1005,starttick=20)
        self.assertFalse(m.gpu_check(config,owner,[],oracle,native)['actual_free_limit_met'])
        apps.append(dict(pid=2,gpu_uuid='g0',used_mib=500))
        result=m.gpu_check(config,owner,[],oracle,native)
        self.assertTrue(result['actual_free_limit_met']);self.assertEqual(result['occupied_by_owned_run_uuids'],['g0'])
        apps[0]['pid']=1
        result=m.gpu_check(config,owner,[],oracle,native)
        self.assertIn('CPU_only_controller_has_GPU_context',result['alerts'])
        self.assertFalse(result['actual_free_limit_met']);self.assertEqual(result['occupied_by_owned_run_uuids'],[])

    def test_current_partial_line_count_no_private_text_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'outers/outer-00003/sampling-attempt-00000/rollouts-0.jsonl'
            path.parent.mkdir(parents=True);path.write_bytes(b'{"private_question":"not-exported"}\npartial')
            result=m.current(dict(paths=dict(output_root=str(root))),dict(outer_index=3,phase='SAMPLING'))
            self.assertEqual(result['sampling_parts'][0]['complete_line_count'],1)
            self.assertFalse(result['sampling_parts'][0]['final_line_complete'])
            self.assertNotIn('not-exported',json.dumps(result))

    def test_shard_overfull_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'outers/outer-00000/sampling-attempt-00000/rollouts-1.jsonl';p.parent.mkdir(parents=True);p.write_bytes(b'{}\n'*321)
            with self.assertRaises(m.Invalid):m.current(dict(paths=dict(output_root=str(root))),dict(outer_index=0,phase='SAMPLING'))

    def test_manifest_mutation_before_execution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'acceptance.py';p.write_bytes(b'VALUE=1\n')
            ref=write(root/'manifest.json',dict(files={'acceptance.py':dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())}))
            p.write_bytes(b'VALUE=2\n')
            with self.assertRaises(m.Invalid):m.source_tree(root,ref)

    def test_source_exec_uses_bytes_not_same_path_cached_object(self):
        import sys
        name='_readonly_fixture_exact_bytes'
        try:
            module=m.exact_module(name,Path('/tmp/bound.py'),b'VALUE=17\n')
            self.assertEqual(module.VALUE,17)
            with self.assertRaises(m.Invalid):m.exact_module(name,Path('/tmp/bound.py'),b'VALUE=18\n')
        finally:sys.modules.pop(name,None)

    def test_line_count_race_reported_not_training_failure(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p';p.write_bytes(b'{}\n')
            with patch.object(m,'stat',side_effect=[dict(bytes=3),dict(bytes=6)]):row=m.line_count(p)
            self.assertFalse(row['stable']);self.assertEqual(row['complete_line_count'],1)

    def test_completed_reads_full_states_only_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=dict(run_id='r',_config_sha256='s',paths=dict(output_root=str(root)),runtime_manifest=dict(sha256='runtime'),source_model_manifest=dict(sha256='source'))
            previous=None;updates=0;exposure=0
            for i,K in enumerate((33,2)):
                outer=root/'outers'/('outer-%05d'%i);updates+= (K+31)//32;exposure+=K
                cp=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id='r',config_sha256='s',runtime_identity='runtime',source_model_identity='source',optimizer_updates=updates,cumulative_pair_exposure=exposure,cursor=dict(outer_index=i,next_pair_cursor=K,usable_pair_count=K),protection_reason=['outer_end'])
                ref=write(root/'checkpoints'/('cp%d'%i)/'manifest.json',cp)
                write(outer/'outer-manifest.json',dict(run_id='r',config_sha256='s',outer_index=i,cumulative_optimizer_updates_before=updates-(K+31)//32,P=K,previous_complete_checkpoint=previous))
                write(outer/'training/outer-complete.json',dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id='r',outer_index=i,K=K,P=K,optimizer_updates_delta=(K+31)//32,optimizer_updates=updates,scheduler_advanced_updates=0,checkpoint=ref));previous=ref
            calls=[]
            def checker(ref,*args):calls.append(ref);return {},dict(manifest=ref)
            result=m.completed(config,types.SimpleNamespace(check_checkpoint=checker))
            self.assertEqual(result['total_usable_pair_exposure'],35);self.assertEqual(result['optimizer_updates'],3)
            self.assertEqual(calls,[previous])


if __name__=='__main__':unittest.main()
