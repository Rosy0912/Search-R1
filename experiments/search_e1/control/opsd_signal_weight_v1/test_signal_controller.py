"""CPU contracts for four physical cards/eight logical slots and strict gates."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PACKAGE='_opsd_signal_specific_fixtures'
package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package
prepare=importlib.import_module(PACKAGE+'.prepare')
controller=importlib.import_module(PACKAGE+'.controller')
gates=importlib.import_module(PACKAGE+'.gates')
cp=importlib.import_module(PACKAGE+'.checkpoint')
memory=importlib.import_module(PACKAGE+'.memory_observer')


def oracle_fixture():
    ids=[[4*i+j for j in range(4)] for i in range(16)]
    scores=[[1.-j*.1 for j in range(4)] for _ in range(16)]
    return dict(queries=['synthetic fixture '+str(i) for i in range(16)],cpu_ids=ids,gpu_ids=copy.deepcopy(ids),
        cpu_scores=scores,gpu_scores=copy.deepcopy(scores),checks=[dict(same_ids=True,max_score_delta=0.) for _ in range(16)],
        score_atol=1e-5,tf32=False,encoder='unchanged CPU FP32 E5',shard_bounds=[[0,21015324]],
        ntotal=21015324,d=768,concurrent_512_requests=2,source_manifest_sha256='1'*64)


class SignalControl(unittest.TestCase):
    def cfg(self):
        return prepare.template('/weka_3/shaoshitong/new/group-G_POS','G_POS','study','/weka_3/shaoshitong/new')

    def test_exact_double_accumulation_contract_and_old_groups_rejected(self):
        c=self.cfg();self.assertEqual(c['scientific']['world_size'],4)
        self.assertEqual(c['scientific']['gradient_accumulation_full_batch_per_rank'],8)
        self.assertEqual(c['scientific']['original_eight_rank_accumulation'],4)
        self.assertEqual(c['scientific']['global_batch'],32)
        self.assertEqual(c['scientific']['micro_batch'],1)
        self.assertEqual(c['scientific']['outer_iterations'],25)
        self.assertEqual(c['evaluation']['completed_outers'],[15,25])
        for old in ('C0','C1','M','R','W','P','J'):
            with self.assertRaises(ValueError):prepare.template('/weka_3/group',old,'study','/weka_3/study')

    def test_mapping_keeps_eight_logical_seeds_and_four_physical_cards(self):
        c=self.cfg();self.assertEqual([controller.logical_gpu(c,r) for r in range(8)],[4,5,6,7,4,5,6,7])
        for bad in (-1,8,True):
            with self.assertRaises(ValueError):controller.logical_gpu(c,bad)
        c['sampling']['logical_rank_to_slot']=[0]*8
        with self.assertRaises(ValueError):controller.logical_gpu(c,4)

    def test_schema_uid_cache_and_budget_mutations_rejected(self):
        for keys,value in [(('owner','uid'),1005),(('scientific','outer_iterations'),10),
                (('scientific','gradient_accumulation_full_batch_per_rank'),4),
                (('sampling','gpu_memory_utilization'),.9),(('sampling','num_gpu_blocks_override'),45342),
                (('resources','training_minimum_free_mib'),1024),
                (('paths','retriever_url'),'http://127.0.0.1:18003/retrieve')]:
            c=self.cfg();c[keys[0]][keys[1]]=value
            with self.subTest(keys=keys),self.assertRaises(ValueError):prepare.validate_config(c)

    def test_original_oracle_arrays_reconstructed_and_corruption_rejected(self):
        value=oracle_fixture();self.assertEqual(gates.verify_existing_joined_gate(value)['queries'],16)
        for mutate in (lambda x:x['gpu_ids'][1].__setitem__(0,999),
                       lambda x:x['gpu_scores'][0].__setitem__(0,float('nan')),
                       lambda x:x.__setitem__('score_atol',.1),
                       lambda x:x.__setitem__('shard_bounds',[[0,1024]]),
                       lambda x:x['checks'][0].__setitem__('max_score_delta',1e-3)):
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(gates.Hold):gates.verify_existing_joined_gate(bad)

    def test_foreign_gpu_zero_to_three_is_ignored_but_owned_card_foreign_rejected(self):
        c=self.cfg();c.update(run_id='a'*32,_config_sha256='b'*64)
        owner=dict(pid=9,uid=1001,starttick=3,run_id='a'*32)
        cards=[dict(index=i,uuid='foreign-'+str(i),actual_free_mib=1) for i in range(4)]
        cards += [dict(index=i+4,uuid=u,actual_free_mib=23000) for i,u in enumerate(c['owner']['gpu_uuids'])]
        transition=dict(accepted=True,cluster='HKUST',gpu_uuids=c['owner']['gpu_uuids'])
        apps=[dict(pid=1,gpu_uuid='foreign-0',used_mib=48000),dict(pid=77,gpu_uuid=cards[-1]['uuid'],used_mib=738)]
        with patch.object(gates.os,'getuid',return_value=1001),patch.object(gates,'process',return_value=owner),\
             patch.object(gates,'bound',return_value=transition),patch.object(gates,'scoped',side_effect=lambda cfg,r:r),\
             patch.object(gates,'publication_admission',return_value={}),patch.object(gates,'storage_admission',return_value={}),\
             patch.object(gates,'retriever_admission',return_value={'owner':{'pid':77}}),\
             patch.object(gates,'scoped_cpu_processes',return_value={}),patch.object(gates,'memory_observer_admission',return_value={}),\
             patch.object(gates,'inventory',side_effect=lambda:(cards,apps)),\
             patch.object(gates,'owned_gpu_process',side_effect=gates.Hold('foreign owns requested card')) as checked:
            result=gates.require_launch(c,'sampling-start','a'*32,owner,[])
            self.assertEqual([d['index'] for d in result['devices']],[4,5,6,7]);checked.assert_not_called()
            apps.append(dict(pid=88,gpu_uuid=cards[4]['uuid'],used_mib=100))
            with self.assertRaises(gates.Hold):gates.require_launch(c,'sampling-running','a'*32,owner,[])

    def test_two_wave_actual_release_precedes_next_four_starts_and_complete_train_em(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);outer=root/'outer-00000';outer.mkdir()
            obj=object.__new__(controller.Controller);obj.config=self.cfg();obj.run_id='a'*32
            obj.config_ref=dict(path='/fixture-config.json',sha256='b'*64)
            obj.owner=dict(pid=9,uid=1001,starttick=3,run_id=obj.run_id);obj.output=root
            events=[];jobs_by_rank={}
            obj.gate=lambda phase,*a,**k:events.append(('gate',phase)) or dict(accepted=True)
            questions=[dict(question_id='q'+str(i),question='synthetic '+str(i),answers=['A'],data_source='nq') for i in range(512)]
            identity=dict(kind='raw-source100',sha256='c'*64)
            core=types.SimpleNamespace(reward=lambda response,answers,spans:0)
            fake_package=types.ModuleType('search_e1');fake_package.core=core
            def spawn(command,log,phase,extra_env):
                rank=int(command[command.index('--rank')+1]);events.append(('start',rank,extra_env['CUDA_VISIBLE_DEVICES']))
                output=Path(command[command.index('--output')+1]);trace=Path(command[command.index('--trace')+1]);http=Path(command[command.index('--http-trace')+1])
                rows=[dict(q,sibling=s,response='wrong',policy_spans=[[0,5]],reward=0,search_calls=0)
                      for q in questions[rank::8] for s in range(5)]
                controller.write_rows(output,rows);trace.write_text('{}\n');http.write_text('')
                process=types.SimpleNamespace(pid=1000+rank);log.write_text('synthetic CPU lifecycle fixture\n')
                child=dict(pid=process.pid,uid=1001,starttick=100+rank,run_id=obj.run_id)
                job=(process,None,child);jobs_by_rank[rank]=job
                cp.exclusive_json(str(output)+'.worker-complete.json',dict(accepted=True,rank=rank,rows=320,run_id=obj.run_id,
                    config_sha256='b'*64,variant='G_POS',precision={'mode':'bf16'},pid=process.pid,uid=1001,
                    actual_llm_load={'actual_dtype':'torch.bfloat16'},sampling_model_identity='c'*64,
                    rollouts=cp.fileproof(output),generation_trace=cp.fileproof(trace),http_trace=cp.fileproof(http),http_requests=0))
                return job
            def wait(jobs,phase):
                ranks=[j[0].pid-1000 for j in jobs];events.append(('actual_wait_release',ranks))
                exits=[dict(identity=j[2],actual_wait_returncode=0) for j in jobs]
                ref=cp.exclusive_json(root/('exit-'+str(ranks[0])+'.json'),dict(run_id=obj.run_id,config_sha256='b'*64,
                    phase=phase,child_exit_evidence=exits))
                return dict(accepted=True,child_exit_evidence=exits,actual_terminal=ref,
                    gpu_release={'all_owned_release':{'accepted':True}})
            obj.spawn=spawn;obj.wait=wait
            with patch.object(controller,'bind_model_files',side_effect=lambda config,value:value),\
                 patch.object(controller,'verify_scoring',return_value={}),\
                 patch.object(controller,'verify_http_trace',side_effect=lambda ref,*a:cp.checked_file(ref)),\
                 patch.object(controller,'verify_rollouts',return_value=([],dict(P=0,questions=512,classes={'all_wrong':512}))),\
                 patch.dict(sys.modules,{'search_e1':fake_package}):
                value,ref=obj.sample(outer,questions,identity)
            starts=[e for e in events if e[0]=='start'];self.assertEqual(starts,[( 'start',i,str(4+i%4)) for i in range(8)])
            release_index=events.index(('actual_wait_release',[0,1,2,3]))
            self.assertGreater(events.index(('start',4,'4')),release_index)
            self.assertEqual(value['train_em']['overall']['denominator'],2560)
            self.assertEqual(value['train_em']['all5_wrong'],512)
            self.assertEqual(value['train_em']['overall']['em'],0.)
            obj.verify_exit(value['exit_proof'],'sampling-running',8)

    def test_observer_stays_live_until_actual_study_driver_release_ack(self):
        self.observer_fixture()

    def test_observer_preserves_old_driver_ack_until_current_driver_ack(self):
        self.observer_fixture(initial='old')

    def test_observer_absent_driver_ack_is_not_a_valid_exit(self):
        self.observer_fixture(initial='current',absent_first=True)

    def test_observer_current_driver_invalid_ack_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'current-driver wait0'):
            self.observer_fixture(initial='invalid')

    def observer_fixture(self,initial=None,absent_first=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.cfg();cfg['paths']['receipts']=str(root/'receipts');cfg['paths']['output_root']=str(root/'runs')
            cfg.update(run_id='a'*32);receipts=Path(cfg['paths']['receipts']);receipts.mkdir();output=Path(cfg['paths']['output_root']);output.mkdir()
            cfg['gates']['scoped_cpu_processes_path']=str(receipts/'scoped-cpu-processes.json')
            source=Path(memory.__file__).resolve();manifest=cp.exclusive_json(root/'manifest.json',dict(files={'memory_observer.py':cp.fileproof(source)}))
            cfg['helper_source_manifest']=manifest;config_ref=cp.exclusive_json(root/'config.json',cfg)
            driver=dict(pid=10,uid=1001,starttick=30,ppid=1,pgid=10,argv=['python','study_runner.py'],run_id='study-independent')
            owner=dict(pid=20,uid=1001,starttick=33,run_id='a'*32)
            cp.exclusive_json(cfg['gates']['scoped_cpu_processes_path'],dict(roles=[dict(role='study_driver',identity=driver)]))
            terminal=cp.exclusive_json(root/'controller-terminal.json',dict(actual_wait_returncode=0,controller=owner))
            cp.exclusive_json(output/'status.json',dict(run_id='a'*32,phase='TRAIN_COMPLETE',config_sha256=config_ref['sha256'],
                outer_iterations=25,resource_release={'all_owned_release':{'accepted':True}},owner=owner))
            cards=[dict(index=i+4,uuid=u,actual_free_mib=40000) for i,u in enumerate(cfg['owner']['gpu_uuids'])]
            observer=dict(pid=9009,uid=1001,starttick=3,run_id='observer-independent')
            pauses=[]
            ack=dict(schema='opsd-signal-study-driver-release-v1',accepted=True,
                run_id='a'*32,config_sha256=config_ref['sha256'],actual_wait_returncode=0,release={'accepted':True},
                driver_identity=driver,controller_terminal=terminal)
            if initial is not None:
                initial_ack=copy.deepcopy(ack)
                if initial=='old':
                    initial_ack['driver_identity']['pid']=999
                    initial_ack['config_sha256']='old-owner-configuration'
                elif initial=='invalid':initial_ack['actual_wait_returncode']=1
                cp.exclusive_json(receipts/'study-driver-release.json',initial_ack)
            def pause(seconds):
                pauses.append(seconds)
                self.assertFalse((receipts/'memory-terminal.json').exists())
                cp.atomic_json(receipts/'study-driver-release.json',ack)
            driver_reads=[]
            def read_process(pid):
                if pid==9009:return observer
                driver_reads.append(pid)
                if absent_first and len(driver_reads)==1:raise FileNotFoundError('Synthetic previous owner is actually absent')
                return driver
            with patch.object(memory.os,'getuid',return_value=1001),patch.object(memory.os,'getpid',return_value=9009),\
                 patch.object(memory,'process',side_effect=read_process),\
                 patch.object(memory,'inventory',return_value=(cards,[])),patch.object(memory.time,'sleep',side_effect=pause):
                memory.observe(config_ref['path'],receipts/'memory-live.json',receipts/'memory-binding.json',.2)
            accepted=json.loads((receipts/'memory-terminal.json').read_text())
            self.assertEqual(pauses,[.2]);self.assertTrue(accepted['accepted']);self.assertTrue(accepted['no_signals_sent'])
            self.assertEqual(accepted['samples'],2);self.assertEqual(accepted['actual_driver_identity'],driver)
            self.assertEqual(set(map(int,json.loads((receipts/'memory-live.json').read_text())['minimum_actual_free_mib'])),{4,5,6,7})


if __name__=='__main__':unittest.main()
