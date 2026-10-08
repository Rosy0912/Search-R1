import importlib.util,pathlib,json,pytest
s=importlib.util.spec_from_file_location('stress',pathlib.Path(__file__).with_name('stress.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_bad_deployed_source_refused_before_import(tmp_path):
 p=tmp_path/'stage_cleanup.py';p.write_text('raise AssertionError("must never import")')
 with pytest.raises(ValueError,match='source drift'):m.load(tmp_path,'stage_cleanup.py')
def test_primary_receipt_exclusive_preserves_first_error(tmp_path):
 p=tmp_path/'error.json';m.durable(p,{'error':'first'})
 with pytest.raises(FileExistsError):m.durable(p,{'error':'second'})
 assert json.loads(p.read_text())=={'error':'first'}
def test_no_GPU_job_entry_resource_request():
 entry=pathlib.Path(__file__).with_name('native.sbatch').read_text();assert '--gres' not in entry and '--cpus-per-task=4' in entry and '--mem=8G' in entry and "CUDA_VISIBLE_DEVICES=''" in entry
