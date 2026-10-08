"""Local CPU fixtures only. No actual review, Slurm, signature or release is made."""
import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('unsigned_runner_cpu_fixture',HERE/'unsigned_receipt_overlap.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old_spec=importlib.util.spec_from_file_location('reviewed_overlap_fixture_api',HERE/'first5_overlap.py')
b=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(b)


def fixture_base(monkeypatch,root=None):
    module=SimpleNamespace(physical=lambda p,directory=False:Path(p),unique=b.unique,pin=b.pin,
          write_original=b.write_original,waited=b.waited,
          process=lambda pid,*args,**kw:{'pid':pid,'uid':0,'present':True,'start_ticks':'CPU_FIXTURE',
                                        'cgroup_original':'1:freezer:/uid_0/job_123/step_1\n'},
          run_root=lambda job:root if root else b.run_root(job),
          pause_context=lambda _: {'scope':'CPU_FIXTURE_NOT_ACTUAL_PAUSE'})
    monkeypatch.setattr(m,'base',lambda:module)
    return module


def test_exact_commands_use_fixed_producer_and_cpu_step_without_signing(monkeypatch):
    fixture_base(monkeypatch)
    config={'job_id':'3261','checkpoint_manifest_sha256':'a'*64}
    for name in ('evidence_spec','root_review','independent_review'):
        config[name]={'path':'/CPU_FIXTURE_ONLY/'+name,'sha256':'b'*64}
    output=m.ROOT/'runs/formal-3261/evidence/CPU_FIXTURE_ONLY'
    argv=m.producer_command(config,output)
    assert argv[1]==str(m.FIRST5/'receipt_payload.py')
    assert argv[argv.index('--tools-manifest-sha256')+1]==m.FIRST5_SHA
    assert argv[argv.index('--output')+1]==str(output/'unsigned-step5-payload.json')
    for field in ('--evidence-spec-sha256','--root-review-sha256','--independent-review-sha256'):
        assert argv[argv.index(field)+1]=='b'*64
    step=m.srun_command(config,Path('/CPU_FIXTURE_ONLY/launch'),'c'*64)
    assert step[:2]==['srun','--jobid=3261']
    for required in ('--overlap','--gres=none','--cpus-per-task=1','CUDA_VISIBLE_DEVICES=',
                     'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','produce'):
        assert required in step
    assert 'openssl' not in argv+step and 'scancel' not in argv+step and '--gres=gpu' not in step


def context_fixture(tmp_path,monkeypatch):
    run=tmp_path/'CPU_TEST_FIXTURE_ONLY';(run/'full_state/global_step_5').mkdir(parents=True)
    common=fixture_base(monkeypatch,run)
    monkeypatch.setattr(m,'verify_tools',lambda _:common)
    p=HERE/'first5-12424d01025e/receipt_payload.py'
    spec=importlib.util.spec_from_file_location('actual_fixed_receipt_api_for_fixture',p)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    monkeypatch.setattr(m,'producer',lambda:api)
    manifest={'version':1,'method':'pure_opsd','source_commit':m.SOURCE,'run_root':str(run),'uid':os.getuid(),
              'global_step':5,'actual_outer_updates':5,'next_global_step':6,'horizon':200,'warmup_steps':0,
              'world_size':8,'driver_pid':12345,'job_id':'123','pause_nonce':'a'*64,'config_sha256':'b'*64}
    mp=run/'full_state/global_step_5/manifest.json';mp.write_text(json.dumps(manifest))
    config={'source_commit':m.SOURCE,'scope':m.SCOPE,'job_id':'123','runner_manifest_sha256':'c'*64,
            'first5_tools_manifest_sha256':m.FIRST5_SHA,'checkpoint_manifest_sha256':m.hashfile(mp),
            'pause_nonce':manifest['pause_nonce'],'driver_start_ticks':'12345'}
    evidence={}
    for name in ('resources','retrieval','hf_reload','rank_audit','service'):
        path=run/(name+'.json');record={'scope':'CPU_TEST_FIXTURE_ONLY_NO_ACTUAL_ACCEPTANCE'}
        if name in ('hf_reload','rank_audit'):
            record.update(passed=True,source_commit=m.SOURCE,job_id='123',
                          checkpoint_manifest_sha256=config['checkpoint_manifest_sha256'],cuda_is_initialized=False)
        path.write_text(json.dumps(record));evidence[name]={'path':str(path),'sha256':m.hashfile(path)}
    path=run/'spec.json';path.write_text(json.dumps(evidence));config['evidence_spec']={'path':str(path),'sha256':m.hashfile(path)}
    for name,role in [('root_review','ROOT_AUDITOR'),('independent_review','INDEPENDENT_REVIEWER')]:
        record={'version':1,'scope':'CPU_TEST_FIXTURE_ONLY_NO_ACTUAL_ACCEPTANCE','status':'PASS_ACTUAL_FIRST5_EVIDENCE',
                'role':role,'source_commit':m.SOURCE,'job_id':'123','driver_pid':manifest['driver_pid'],
                'pause_nonce':manifest['pause_nonce'],'checkpoint_manifest_sha256':config['checkpoint_manifest_sha256'],'evidence':evidence}
        path=run/(name+'.json');path.write_text(json.dumps(record));config[name]={'path':str(path),'sha256':m.hashfile(path)}
    return config,run,evidence


def test_actual_review_protocol_is_required_and_no_generic_software_pass_substitutes(tmp_path,monkeypatch):
    config,run,evidence=context_fixture(tmp_path,monkeypatch)
    assert m.context(config)['scope']=='CPU_FIXTURE_NOT_ACTUAL_PAUSE'
    for name in ('root_review','independent_review'):
        path=Path(config[name]['path']);original=json.loads(path.read_text())
        for key,value in [('status','PASS_SOFTWARE_ONLY'),('role','WRONG'),('job_id','124'),
                          ('source_commit','0'*40),('driver_pid',54321),('pause_nonce','0'*64),
                          ('checkpoint_manifest_sha256','0'*64),('evidence',{})]:
            bad=copy.deepcopy(original);bad[key]=value;path.write_text(json.dumps(bad));config[name]['sha256']=m.hashfile(path)
            with pytest.raises(ValueError):m.context(config)
        path.write_text(json.dumps(original));config[name]['sha256']=m.hashfile(path)


def test_missing_reviews_and_unfinished_cpu_reports_refuse_start_eligibility(tmp_path,monkeypatch):
    config,run,evidence=context_fixture(tmp_path,monkeypatch)
    for name in ('hf_reload','rank_audit'):
        path=Path(evidence[name]['path']);raw=path.read_bytes();path.unlink()
        with pytest.raises(FileNotFoundError):m.context(config)
        path.write_bytes(raw)
    review=Path(config['independent_review']['path']);review.unlink()
    with pytest.raises(FileNotFoundError):m.context(config)


@pytest.mark.parametrize('exit_code',[0,7])
def test_controller_waits_real_local_cpu_subprocess_and_retains_nonzero(tmp_path,monkeypatch,exit_code):
    fixture_base(monkeypatch);monkeypatch.setattr(m,'context',lambda _: {'scope':'CPU_FIXTURE'})
    argv=[sys.executable,'-c','import sys;print("CPU_fixture_srun");sys.exit('+str(exit_code)+')']
    monkeypatch.setattr(m,'srun_command',lambda *args:argv)
    assert m.controller({},tmp_path,Path('/CPU_FIXTURE'),'a'*64)==exit_code
    record=json.loads((tmp_path/'srun-exit-original.json').read_text())
    assert record['returncode']==exit_code and record['waited_by_controller_pid']==os.getpid()
    assert (tmp_path/'srun-stdout-original.txt').read_text()=='CPU_fixture_srun\n'


def test_supervisor_records_actual_controller_exit(tmp_path,monkeypatch):
    fixture_base(monkeypatch);monkeypatch.setattr(m,'context',lambda _: {'scope':'CPU_FIXTURE'})
    monkeypatch.setattr(m,'child_command',lambda *args:[sys.executable,'-c','import sys;sys.exit(9)'])
    assert m.supervise({},tmp_path,Path('/CPU_FIXTURE'),'a'*64)==9
    record=json.loads((tmp_path/'controller-exit-original.json').read_text())
    assert record['returncode']==9 and record['waited_by_supervisor_pid']==os.getpid()


def test_once_only_marker_covers_unknown_outcome(tmp_path,monkeypatch):
    run=tmp_path/'run';(run/'evidence').mkdir(parents=True);common=fixture_base(monkeypatch,run)
    monkeypatch.setattr(m,'verify_tools',lambda _:common);monkeypatch.setattr(m,'context',lambda _: {'scope':'CPU_FIXTURE'})
    args=argparse.Namespace(job_id='123',source_commit=m.SOURCE,runner_manifest_sha256='a'*64,
        first5_tools_manifest_sha256=m.FIRST5_SHA,checkpoint_manifest_sha256='b'*64,pause_nonce='c'*64,
        driver_start_ticks='12345',output=str(run/'evidence/new'))
    for name in ('evidence_spec','root_review','independent_review'):
        setattr(args,name,'/CPU_FIXTURE_ONLY/'+name);setattr(args,name+'_sha256','d'*64)
    marker=run/'evidence/first5-unsigned-receipt-overlap-launch.json';marker.write_bytes(b'CPU_FIXTURE_UNKNOWN_OUTCOME')
    monkeypatch.setattr(m.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Must not retry an existing launch'))
    with pytest.raises(ValueError,match='Existing unsigned launch'):m.start(args)
    assert marker.read_bytes()==b'CPU_FIXTURE_UNKNOWN_OUTCOME' and not Path(args.output).exists()


def test_software_contains_no_review_generation_signing_remote_or_gpu_operation():
    raw=(HERE/'unsigned_receipt_overlap.py').read_text();compile(raw,'CPU_SOFTWARE_FIXTURE_ONLY','exec')
    for forbidden in ('openssl','ssh ','scancel','sbatch','torch.cuda.init','root-review.json\').open'):
        assert forbidden not in raw
    assert 'module.review_record' in raw and 'role,evidence' in raw
    assert 'finally:returncode=child.wait()' in raw and 'start_new_session=True' in raw
    assert 'CUDA_VISIBLE_DEVICES=' in raw and "--gres=none" in raw
