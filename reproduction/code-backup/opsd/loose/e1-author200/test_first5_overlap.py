"""Local CPU fixtures only: no SSH, real Slurm step, signature or GPU work."""
import argparse
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_overlap_fixture',HERE/'first5_overlap.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture_process(pid,job=None,required=True):
    return {'pid':pid,'uid':0,'start_ticks':'CPU_FIXTURE','present':True,'live':True,'scope':'CPU_TEST_FIXTURE_ONLY'}


def test_exact_real_command_is_one_cpu_and_zero_gpu():
    argv=m.command('3261',m.ROOT/'runs/formal-3261/evidence/FIXTURE_PATH_ONLY')
    assert argv[:2]==['srun','--jobid=3261']
    assert '--overlap' in argv and '--gres=none' in argv and '--cpus-per-task=1' in argv
    for value in ('CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1'):
        assert value in argv
    assert argv[argv.index('--tools-manifest-sha256')+1]==m.FIRST5_SHA
    assert argv[argv.index('--job-id')+1]=='3261'
    assert str(m.FIRST5/'inspect_first5_cpu.py') in argv
    for value in argv:assert not any(bad in value for bad in ('sbatch','scancel','--gres=gpu','receipt_payload'))


@pytest.mark.parametrize('exit_code',[0,7])
def test_controller_waits_actual_cpu_fixture_subprocess_and_preserves_streams(tmp_path,monkeypatch,exit_code):
    monkeypatch.setattr(m,'pause_context',lambda _: {'scope':'CPU_FIXTURE_NOT_ACTUAL_PAUSE'})
    monkeypatch.setattr(m,'process',fixture_process)
    argv=[sys.executable,'-c',"import sys;print('CPU_FIXTURE_stdout');print('CPU_FIXTURE_stderr',file=sys.stderr);sys.exit("+str(exit_code)+")"]
    result=m.controller({'command':argv},tmp_path)
    assert result==exit_code
    assert (tmp_path/'srun-stdout-original.txt').read_text()=='CPU_FIXTURE_stdout\n'
    assert (tmp_path/'srun-stderr-original.txt').read_text()=='CPU_FIXTURE_stderr\n'
    original=json.loads((tmp_path/'srun-exit-original.json').read_text())
    assert type(original['returncode']) is int and original['returncode']==exit_code
    assert original['waited_by_controller_pid']==os.getpid()


@pytest.mark.parametrize('exit_code',[0,9])
def test_supervisor_records_actual_waited_controller_exit(tmp_path,monkeypatch,exit_code):
    monkeypatch.setattr(m,'pause_context',lambda _: {'scope':'CPU_FIXTURE_NOT_ACTUAL_PAUSE'})
    monkeypatch.setattr(m,'process',fixture_process)
    argv=[sys.executable,'-c','import sys;print("CPU_FIXTURE_controller");sys.exit('+str(exit_code)+')']
    monkeypatch.setattr(m,'child_command',lambda *args:argv)
    assert m.supervise({},tmp_path,Path('/CPU_FIXTURE_ONLY'),'0'*64)==exit_code
    original=json.loads((tmp_path/'controller-exit-original.json').read_text())
    assert original['returncode']==exit_code and original['waited_by_supervisor_pid']==os.getpid()
    assert (tmp_path/'controller-stdout-original.txt').read_text()=='CPU_FIXTURE_controller\n'


def test_signal_mapping_does_not_fake_exit_zero():
    assert m.waited(-15)==143 and m.waited(-9)==137 and m.waited(12)==12


def arguments(tmp_path):
    run=tmp_path/'run';evidence=run/'evidence';evidence.mkdir(parents=True)
    return run,argparse.Namespace(job_id='123',output=str(evidence/'new'),checkpoint_manifest_sha256='a'*64,
            pause_nonce='b'*64,driver_start_ticks='12345',overlap_tools_manifest_sha256='c'*64)


def test_existing_launch_marker_covers_unknown_outcome_and_prevents_repeat(tmp_path,monkeypatch):
    run,args=arguments(tmp_path)
    monkeypatch.setattr(m,'run_root',lambda _:run)
    monkeypatch.setattr(m,'physical',lambda path,directory=False:Path(path))
    monkeypatch.setattr(m,'pause_context',lambda _: {'scope':'CPU_FIXTURE_NOT_ACTUAL_PAUSE'})
    marker=run/'evidence/first5-cpu-overlap-launch.json';marker.write_bytes(b'CPU_FIXTURE_UNKNOWN_OUTCOME')
    monkeypatch.setattr(m.subprocess,'Popen',lambda *args,**kw:pytest.fail('Duplicate launch must not spawn'))
    with pytest.raises(ValueError,match='Existing launch'):m.start(args)
    assert marker.read_bytes()==b'CPU_FIXTURE_UNKNOWN_OUTCOME' and not Path(args.output).exists()


def test_no_actual_pause_or_wrong_nonce_cannot_create_output(tmp_path,monkeypatch):
    run,args=arguments(tmp_path)
    monkeypatch.setattr(m,'run_root',lambda _:run)
    monkeypatch.setattr(m,'physical',lambda path,directory=False:Path(path))
    def reject(_):raise ValueError('Fixture actual PAUSED/nonce mismatch')
    monkeypatch.setattr(m,'pause_context',reject)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *args,**kw:pytest.fail('Unverified pause must not spawn'))
    with pytest.raises(ValueError,match='PAUSED'):m.start(args)
    assert not Path(args.output).exists() and list((run/'evidence').iterdir())==[]


def completed(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'physical',lambda path,directory=False:Path(path))
    config={'job_id':'123','checkpoint_manifest_sha256':'a'*64}
    records={'controller-exit-original.json':{'returncode':0},'srun-exit-original.json':{'returncode':0},
         'inspection-completed-review-required.json':{'scope':'CPU_STORED_STATE_AND_HF_INSPECTION_ONLY_NOT_SIGNATURE_OR_RELEASE',
              'source_commit':m.SOURCE,'job_id':'123','manifest_sha256':'a'*64,'cuda_is_initialized':False},
         'cpu-step-original.json':{'job_id':'123','uid':0,'manifest_sha256':'a'*64,
              'cuda_visible_devices':'','cuda_is_initialized':False,'cgroup':'1:freezer:/slurm_h100-10/uid_0/job_123/step_1\n'}}
    for name in ('native','hf'):
        records[name+'-exit-original.json']={'returncode':0}
        records[name+'.json']={'passed':True,'source_commit':m.SOURCE,'job_id':'123','checkpoint_manifest_sha256':'a'*64}
    for name,record in records.items():(tmp_path/name).write_text(json.dumps(record))
    return config,records


def test_completion_requires_real_native_and_hf_and_cpu_step_identity(tmp_path,monkeypatch):
    config,records=completed(tmp_path,monkeypatch)
    assert m.completion(config,tmp_path) is True
    cases=[('controller-exit-original.json','returncode',False),('srun-exit-original.json','returncode',False),
         ('cpu-step-original.json','uid',False),('cpu-step-original.json','cuda_visible_devices','0'),
         ('cpu-step-original.json','cuda_is_initialized',True),('cpu-step-original.json','job_id','124'),
         ('inspection-completed-review-required.json','source_commit','0'*40),
         ('native.json','passed',False),('hf.json','checkpoint_manifest_sha256','0'*64),
         ('native-exit-original.json','returncode',False),('hf-exit-original.json','returncode',False)]
    for name,key,value in cases:
        bad=copy.deepcopy(records[name]);bad[key]=value;(tmp_path/name).write_text(json.dumps(bad))
        with pytest.raises(ValueError):m.completion(config,tmp_path)
        (tmp_path/name).write_text(json.dumps(records[name]))
    for name in ('controller-exit-original.json','srun-exit-original.json'):
        (tmp_path/name).write_text(json.dumps({'returncode':7}))
        assert m.completion(config,tmp_path) is False
        (tmp_path/name).write_text(json.dumps(records[name]))


def test_json_identity_cannot_use_duplicates_nonfinite_or_nonactual_pins():
    for raw in (b'{"a":1,"a":2}',b'{"x":NaN}'):
        with pytest.raises(ValueError):m.unique(raw)
    for value in (None,True,'short','0'*63):
        with pytest.raises(ValueError):m.pin(value)
    assert m.pin('a'*64)=='a'*64


def test_plain_module_has_no_transport_signing_restart_or_gpu_initialization():
    raw=(HERE/'first5_overlap.py').read_text()
    compile(raw,'CPU_FIXTURE_SOFTWARE_ONLY','exec')
    for forbidden in ('openssl','torch.cuda.init','ssh ','pkill','sbatch','scancel','ray stop'):
        assert forbidden not in raw
    assert 'start_new_session=True' in raw and 'finally:returncode=child.wait()' in raw
    assert 'srun-exit-original.json' in raw and 'controller-exit-original.json' in raw
    assert '.open(\'xb\')' in raw and 'acceptance-5-receipt.json' not in raw
