"""CPU temporary locks and synthetic /proc records; NO actual GPU/Slurm acceptance."""
import copy,fcntl,json,os,subprocess
from pathlib import Path
import pytest
from hf_activation_capture import capture_lease,lock_records,validate_process_raw
from pre_stage_runtime import PRESCRIPTION_FIELDS,verify_prescription,verify_isolated_sources

def lockline(stat):return f'1: FLOCK ADVISORY WRITE {os.getpid()} {os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino} 0 EOF\n'

def test_reassert_inherited_cpu_temp_descriptor_only(tmp_path):
 path=tmp_path/'lease';path.write_text('');proc=tmp_path/'proc';proc.mkdir();fd=os.open(path,os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);(proc/'locks').write_text(lockline(path.stat()))
 try:assert capture_lease(fd,path,proc)['reasserted_same_descriptor'] is True
 finally:os.close(fd)

def test_foreign_descriptor_not_proved_by_same_inode_label(tmp_path):
 path=tmp_path/'lease';path.write_text('');proc=tmp_path/'proc';proc.mkdir();fd=os.open(path,os.O_RDWR);foreign=os.open(path,os.O_RDWR);fcntl.flock(foreign,fcntl.LOCK_EX|fcntl.LOCK_NB);(proc/'locks').write_text(lockline(path.stat()))
 try:
  with pytest.raises(BlockingIOError):capture_lease(fd,path,proc)
 finally:os.close(fd);os.close(foreign)

def test_wrong_lease_descriptor_or_kernel_record_rejects(tmp_path):
 p=tmp_path/'a';p.write_text('');other=tmp_path/'b';other.write_text('');proc=tmp_path/'proc';proc.mkdir();(proc/'locks').write_text('');fd=os.open(p,os.O_RDWR)
 try:
  with pytest.raises(ValueError):capture_lease(fd,other,proc)
  with pytest.raises(ValueError):capture_lease(fd,p,proc)
 finally:os.close(fd)

@pytest.mark.parametrize('extra',['training_hf_receipt_sha256','reviewed_stage_sha256','runtime_supplement','future_corrected_terminal','anything_else'])
def test_exact_prescription_rejects_future_and_hash_cycles(extra):
 stage={key:'CPU_SCHEMA_FIXTURE' for key in PRESCRIPTION_FIELDS};stage['version']=1;stage['phase']='prescribed-pre-generation';final=copy.deepcopy(stage);stage[extra]='not allowed'
 with pytest.raises(ValueError,match='Exact prescription'):verify_prescription(stage,final)

def test_prescription_correspondence_not_output_self_labels():
 stage={key:'CPU_SCHEMA_FIXTURE' for key in PRESCRIPTION_FIELDS};stage['version']=1;stage['phase']='prescribed-pre-generation';final=copy.deepcopy(stage)
 verify_prescription(stage,final);final['input_manifest_sha256']='wrong'
 with pytest.raises(ValueError,match='differs'):verify_prescription(stage,final)

def test_child_source_import_cannot_shadow_by_parent_cwd(tmp_path,monkeypatch):
 from search_r1.baselines import corrected_evaluation as ce
 from pre_stage_runtime import SOURCE_NAMES
 repo=Path(ce.__file__).resolve().parents[2]
 stage=dict(training_commit='8e60976d364ab7d7ae51714415813c1359dafe9a',eval_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),source_files={name:ce.digest(repo/name) for name in SOURCE_NAMES})
 p=tmp_path/'source-stage.json';p.write_text(json.dumps(stage));shadow=tmp_path/'shadow';(shadow/'search_r1/baselines').mkdir(parents=True)
 (shadow/'search_r1/__init__.py').write_text('');(shadow/'search_r1/baselines/__init__.py').write_text('');(shadow/'search_r1/baselines/corrected_evaluation.py').write_text("raise RuntimeError('FOREIGN CWD SOURCE SHOULD NEVER LOAD')")
 monkeypatch.chdir(shadow);verify_isolated_sources(repo,p)
 stage['source_files']['search_r1/qa_protocol.py']='0'*64;p.write_text(json.dumps(stage))
 with pytest.raises(subprocess.CalledProcessError):verify_isolated_sources(repo,p)

def test_launcher_never_runs_without_actual_job(monkeypatch,tmp_path):
 from hf_launcher import run
 monkeypatch.delenv('SLURM_JOB_ID',raising=False)
 with pytest.raises(ValueError,match='Actual Slurm'):run(tmp_path/'missing',tmp_path,tmp_path,'0'*64)

@pytest.mark.parametrize('name',['hf_launcher.py','hf_activation_capture.py'])
def test_new_launcher_capture_entries_remain_blocked(name):
 import sys
 result=subprocess.run([sys.executable,str(Path(__file__).parent/name)],capture_output=True,text=True)
 assert result.returncode==78 and not result.stdout and 'BLOCKED' in result.stderr
