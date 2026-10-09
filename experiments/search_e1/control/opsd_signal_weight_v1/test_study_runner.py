"""CPU process-lifecycle fixtures; never launches a child or reads /proc/GPU."""
import copy
import json
import os
from pathlib import Path as RealPath
import tempfile
import types
import unittest
from unittest.mock import patch
import study_runner as runner
from checkpoint import exclusive_json, fileproof


REMOTE='/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'


class LifecycleFixture:
    def __init__(self,root,completed_prior=False,controller_code=0):
        self.root=RealPath(root).resolve();self.helper=RealPath(runner.__file__).parent.resolve();self.started=[];self.events=[];self.processes={}
        self.pid=9000;self.controller_code=controller_code
        self.publication=exclusive_json(self.root/'publication.json',dict(accepted=True))
        self.source=exclusive_json(self.root/'source.json',dict(files={}))
        self.owner=dict(pid=os.getpid(),uid=1001,starttick=111,run_id='study-private-marker',argv=['python',str(self.helper/'study_runner.py')])
        groups=[];self.cfg_by_path={}
        for variant in runner.VARIANTS:
            base=self.root/variant;(base/'receipts').mkdir(parents=True);(base/'runs').mkdir()
            cfg=dict(variant=variant,study_id='study',run_id=variant+'-run',owner={'uid':1001},
                paths=dict(helper_root=str(self.helper),output_root=str(base/'runs'),receipts=str(base/'receipts'),python='/fixture/python'),
                gates=dict(scoped_cpu_processes_path=str(base/'receipts'/'scoped-cpu-processes.json'),memory_binding_path=str(base/'receipts'/'memory-binding.json'),publication=self.publication),
                helper_source_manifest=self.source)
            ref=exclusive_json(base/'config.json',cfg);self.cfg_by_path[ref['path']]=(cfg,ref);groups.append(dict(variant=variant,config=ref))
            if completed_prior and variant==runner.VARIANTS[0]:self.finish_group(cfg,ref)
        self.study=exclusive_json(self.root/'study.json',dict(study_id='study',study_base=REMOTE,helper_root=str(self.helper),
                helper_source_manifest=self.source,publication=self.publication,groups=groups))
    def path(self,value):
        value=str(value)
        return RealPath(str(self.root)+value[len(REMOTE):] if value.startswith(REMOTE) else value)
    def finish_group(self,cfg,ref):
        output=RealPath(cfg['paths']['output_root'])
        (output/'status.json').write_text(json.dumps(dict(phase='TRAIN_COMPLETE',run_id=cfg['run_id'],config_sha256=ref['sha256'],outer_iterations=25)))
        (output/'training-em-25-complete.json').write_text(json.dumps(dict(accepted=True,rows=[{'outer_index':i} for i in range(25)])))
        for count in (15,25):
            destination=output/'evaluations'/('outer-%02d'%count);destination.mkdir(parents=True,exist_ok=True)
            (destination/'parent-complete.json').write_text(json.dumps(dict(accepted=True,completed_outer=count)))
    def process(self,pid):
        if pid==os.getpid():return self.owner
        if pid not in self.processes:raise FileNotFoundError('Fixture process absent')
        return self.processes[pid].identity
    def popen(self,command,env,stdout,stderr,start_new_session):
        self.pid+=1;cfgpath=command[command.index('--config')+1];cfg,ref=self.cfg_by_path[cfgpath]
        role='observer' if command[1].endswith('memory_observer.py') else 'controller'
        identity=dict(pid=self.pid,uid=1001,starttick=self.pid*10,run_id=env['E1_RUN_ID'],argv=command,ppid=os.getpid(),pgid=self.pid)
        process=FakeProcess(self,role,cfg,ref,identity,command)
        self.processes[self.pid]=process;self.started.append(process);self.events.append((role+'-start',cfg['variant']))
        if role=='observer':
            destination=RealPath(cfg['gates']['memory_binding_path'])
            destination.write_text(json.dumps(dict(observer_identity=identity)))
        return process
    def bootstrap(self,cfg,run_id,owner):
        roles=json.loads(RealPath(cfg['gates']['scoped_cpu_processes_path']).read_text())['roles']
        if [r['role'] for r in roles]!=['study_driver']:raise AssertionError('Bootstrap must precede observer and register only driver')
        self.events.append(('bootstrap',cfg['variant']));return {'accepted':True}
    def observer_admission(self,cfg):
        roles=json.loads(RealPath(cfg['gates']['scoped_cpu_processes_path']).read_text())['roles']
        if [r['role'] for r in roles]!=['study_driver','memory_observer']:raise AssertionError('Observer registry must precede gate')
        self.events.append(('observer-admitted',cfg['variant']));return {'accepted':True}
    def release(self,cfg,run_id,owner,known_children):
        observer=next(p for p in self.started if p.role=='observer' and p.cfg['variant']==cfg['variant'])
        if observer.poll() is not None:raise AssertionError('Observer exited before independent driver release/ACK')
        child=next(p for p in self.started if p.role=='controller' and p.cfg['variant']==cfg['variant'])
        if child.waited is not True:raise AssertionError('Release audit borrowed poll instead of actual wait')
        self.events.append(('release-before-ACK',cfg['variant']));return {'accepted':True}
    def contexts(self):
        return [patch.object(runner,'Path',self.path),patch.object(runner.os,'getuid',return_value=1001),
            patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','E1_RUN_ID':'study-private-marker'}),
            patch.object(runner,'validate_config',return_value=[]),patch.object(runner.gates,'process',side_effect=self.process),
            patch.object(runner.gates,'require_launch_without_memory',side_effect=self.bootstrap),
            patch.object(runner.gates,'memory_observer_admission',side_effect=self.observer_admission),
            patch.object(runner.gates,'require_external_study_release',side_effect=self.release),
            patch.object(runner.subprocess,'Popen',side_effect=self.popen),patch.object(runner.time,'sleep',return_value=None)]


class FakeProcess:
    def __init__(self,fixture,role,cfg,ref,identity,command):
        self.fixture=fixture;self.role=role;self.cfg=cfg;self.ref=ref;self.identity=identity;self.command=command
        self.pid=identity['pid'];self.returncode=None;self.polls=0;self.waited=False
    def poll(self):
        self.polls+=1
        if self.role=='observer':
            ack=RealPath(self.cfg['paths']['receipts'])/'study-driver-release.json'
            if ack.exists():self.returncode=0
        elif self.polls>=3:
            self.fixture.finish_group(self.cfg,self.ref);self.returncode=self.fixture.controller_code
        return self.returncode
    def wait(self):
        self.waited=True
        if self.returncode is None:raise AssertionError('Fixture refuses wait without real exit')
        self.fixture.events.append((self.role+'-wait',self.cfg['variant']));return self.returncode


class RunnerTests(unittest.TestCase):
    def execute(self,fixture,resume=False):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for context in fixture.contexts():stack.enter_context(context)
            return runner.run(fixture.study,resume)
    def test_identity_requires_live_matching_child(self):
        child=types.SimpleNamespace(pid=77,poll=lambda:None)
        with patch.object(runner.gates,'process',return_value={'uid':1001,'run_id':'same','pid':77,'starttick':10}):
            self.assertEqual(runner.actual_identity(child,'same')['starttick'],10)
        with patch.object(runner.gates,'process',return_value={'uid':1001,'run_id':'different'}):
            with self.assertRaises(ValueError):runner.actual_identity(child,'same')
    def test_exited_child_is_not_actual_identity(self):
        child=types.SimpleNamespace(pid=77,poll=lambda:0,returncode=0)
        with self.assertRaises(RuntimeError):runner.actual_identity(child,'same')
    def test_serial_four_arms_actual_wait_release_and_observer_ACK(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=LifecycleFixture(directory);ref=self.execute(fixture)
            result=json.loads(RealPath(ref['path']).read_text())
            self.assertEqual(result['completed_groups'],4);self.assertTrue(result['analysis_still_required'])
            self.assertEqual([p.cfg['variant'] for p in fixture.started if p.role=='controller'],list(runner.VARIANTS))
            self.assertEqual(len(fixture.started),8)
            for variant in runner.VARIANTS:
                events=[e[0] for e in fixture.events if e[1]==variant]
                self.assertLess(events.index('bootstrap'),events.index('observer-start'))
                self.assertLess(events.index('observer-admitted'),events.index('controller-start'))
                self.assertLess(events.index('controller-wait'),events.index('release-before-ACK'))
                self.assertLess(events.index('release-before-ACK'),events.index('observer-wait'))
    def test_nonzero_controller_does_not_start_next_group(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=LifecycleFixture(directory,controller_code=2)
            with self.assertRaises(ValueError):self.execute(fixture)
            self.assertEqual([p.cfg['variant'] for p in fixture.started if p.role=='controller'],['G_POS'])
            terminals=list(fixture.root.glob('sequence-attempts/*/actual-controller-exit.json'))
            self.assertEqual(json.loads(terminals[0].read_text())['actual_wait_returncode'],2)
            self.assertFalse((fixture.root/'G_POS'/'receipts'/'study-driver-release.json').exists())
    def test_existing_group_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=LifecycleFixture(directory,completed_prior=True)
            with self.assertRaises(ValueError):self.execute(fixture)
            self.assertFalse(fixture.started)
    def test_completed_status_cannot_bypass_new_resume_controller_actual_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=LifecycleFixture(directory,completed_prior=True)
            self.execute(fixture,resume=True)
            controllers=[p for p in fixture.started if p.role=='controller']
            self.assertEqual([p.cfg['variant'] for p in controllers],list(runner.VARIANTS))
            self.assertIn('--resume',controllers[0].command)
            self.assertTrue(controllers[0].waited)
    def test_absent_old_observer_preserved_UNKNOWN_not_falsely_wait0(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=LifecycleFixture(directory,completed_prior=True)
            cfg,ref=next(iter(fixture.cfg_by_path.values()))
            binding=RealPath(cfg['gates']['memory_binding_path'])
            old=dict(experiment_run_id=cfg['run_id'],config_sha256=ref['sha256'],observer_identity=dict(pid=8001,uid=1001,starttick=80010,run_id='old-observer'))
            binding.write_text(json.dumps(old));original=fileproof(binding)
            self.execute(fixture,resume=True)
            copies=list(binding.parent.glob('preserved-observers/*/memory-binding.json'))
            self.assertEqual(len(copies),1);self.assertEqual(fileproof(copies[0])['sha256'],original['sha256'])
            proof=json.loads((copies[0].parent/'absence-and-preservation.json').read_text())
            self.assertEqual(proof['previous_wait_returncode'],'UNKNOWN');self.assertTrue(proof['kernel_process_absent'])
    def test_reused_old_observer_PID_never_stopped_or_relaunched(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=LifecycleFixture(directory,completed_prior=True);cfg,ref=next(iter(fixture.cfg_by_path.values()))
            old_identity=dict(pid=8001,uid=1001,starttick=80010,run_id='old-observer')
            binding=RealPath(cfg['gates']['memory_binding_path'])
            binding.write_text(json.dumps(dict(experiment_run_id=cfg['run_id'],config_sha256=ref['sha256'],observer_identity=old_identity)))
            fixture.processes[8001]=types.SimpleNamespace(identity=dict(old_identity,starttick=80011))
            with self.assertRaises(ValueError):self.execute(fixture,resume=True)
            self.assertFalse(fixture.started)
            self.assertEqual(json.loads(binding.read_text())['observer_identity'],old_identity)


if __name__=='__main__':unittest.main()
