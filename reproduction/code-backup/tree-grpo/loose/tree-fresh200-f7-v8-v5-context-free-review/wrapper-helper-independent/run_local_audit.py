"""Independent local source/metadata + tiny CPU tests, never dispatch/assets."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
WRAP = BASE / 'work/tree-fresh200-mu-runtime-v8'
HELP = BASE / 'work/tree-fresh200-dispatch-v5'
SOURCE = BASE / 'work/tree-behavior-mu-ppo'
COMMIT = 'f7e762af365a2beb9429e4cb3b6d5f0d808d4d7d'

def digest(raw): return hashlib.sha256(raw).hexdigest()
def record(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=digest(raw), bytes=len(raw))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def save(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return record(path)

sys.dont_write_bytecode = True
B = load('independent_fresh200_builder', WRAP/'build_package_draft.py')
C = load('independent_fresh200_common', HELP/'fresh200_dispatch_common.py')
assert sys.flags.optimize == 0
pins = {
    WRAP/'FILE_HASHES_V8.json': 'e6d3ece49a466e8aa29f72fc8c87eb4b202b5b44fa455119dee356a8e25a1956',
    WRAP/'SOURCE_CPU_RECEIPT_V8.json': '8af8ddebc8e165763fc65bd243e608f9b211c93be5f52a0a941e600340965a0a',
    HELP/'FILE_HASHES_V5.json': '82f276a8e2889c343fa67e23257ec3fd7a2f3e77c96b2a6e578295066c564f95',
    HELP/'SOURCE_CPU_RECEIPT_V5.json': '8b6e90966a84cda94110c2411a2efef5374d8579fbc01f998113d256a76a87b7',
}
for path, expected in pins.items(): assert record(path)['sha256'] == expected, path
inventories = {str(path.parent): json.loads(path.read_bytes()) for path in pins if path.name.startswith('FILE_HASHES')}
all_inventory_mismatches = []
for directory, inventory in inventories.items():
    for name, expected in inventory['files'].items():
        observed = record(Path(directory)/name)
        if observed['sha256'] != expected['sha256'] or observed['bytes'] != expected['bytes']:
            all_inventory_mismatches.append(dict(directory=directory, name=name, observed=observed, expected=expected))
assert not all_inventory_mismatches
inv = inventories[str(WRAP)]
inputs = json.loads((WRAP/'INPUTS.schema.json').read_bytes())
assert set(inv['wrapper_files']) == set(B.WRAPPER_FILES) == C.WRAPPER_FILES
assert len(B.WRAPPER_FILES) == 15 and len(B.RUNTIME_ENTRIES) == 4 and len(C.SOURCE_INTERFACE_FILES) == 15
prepared = B.prepare_wrapper_inventory(inv, inputs, WRAP)
assert all(digest(raw) == inv['wrapper_files'][name] for name, raw in prepared.items())

checks = []
def rejected(label, function):
    try: function()
    except (KeyError, ValueError) as error:
        checks.append(dict(name=label, outcome='REJECTED', error=repr(error)))
    else: raise AssertionError('unexpected accepted case: '+label)
bad = copy.deepcopy(inv); bad['files'] = {name: entry['sha256'] for name,entry in bad['files'].items()}
rejected('flat_wrapper_sha_map', lambda: B.prepare_wrapper_inventory(bad, inputs, WRAP))
for field,value in [('bytes',True),('bytes','5'),('bytes',-1),('sha256','F'*64)]:
    bad = copy.deepcopy(inv); bad['files']['supervisor.py'][field] = value
    rejected('wrapper_record_'+field+'_'+str(value), lambda bad=bad: B.prepare_wrapper_inventory(bad, inputs, WRAP))
bad = copy.deepcopy(inv); bad['files']['supervisor.py']['bytes'] += 1
rejected('actual_wrapper_byte_count_mismatch', lambda: B.prepare_wrapper_inventory(bad, inputs, WRAP))
for name in B.RUNTIME_ENTRIES:
    bad = copy.deepcopy(inputs); del bad['runtime_entries'][name]
    rejected('missing_runtime_entry_'+name, lambda bad=bad: B.prepare_wrapper_inventory(inv, bad, WRAP))
    bad = copy.deepcopy(inputs); bad['runtime_entries'][name]['sha256'] = '0'*64
    rejected('changed_runtime_entry_'+name, lambda bad=bad: B.prepare_wrapper_inventory(inv, bad, WRAP))
with tempfile.TemporaryDirectory(dir=OUT, prefix='boundary-') as tmp:
    link = Path(tmp)/'entry.sh'; link.symlink_to(WRAP/'run_tree.sh')
    bad=copy.deepcopy(inputs); bad['runtime_entries']['run_tree.sh']['path']=str(link)
    rejected('symlink_runtime_entry', lambda: B.prepare_wrapper_inventory(inv, bad, WRAP))

owner_path = BASE/'work/tree-fresh200-guard-continue-owner/REPAIRED_SOURCE_INTERFACE_CPU_RECEIPT.json'
owner = json.loads(owner_path.read_bytes())
assert owner['status'] == 'FRESH200_REPAIRED_SOURCE_INTERFACE_CPU_OWNER_VERIFIED'
assert owner['code_commit'] == COMMIT and owner['source_clean'] is True
assert all(owner[key] is False for key in ['actual_gpu_executed','actual_remote_executed','training_release'])
assert set(owner['source_files']) == C.SOURCE_INTERFACE_FILES
head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=SOURCE, text=True).strip()
clean = not subprocess.check_output(['git','status','--porcelain'], cwd=SOURCE, text=True).strip()
assert head == COMMIT and clean
source_map = {}
for name in sorted(C.SOURCE_INTERFACE_FILES):
    observed = record(SOURCE/name)
    git_raw = subprocess.check_output(['git','show',COMMIT+':'+name], cwd=SOURCE)
    assert observed['sha256'] == owner['source_files'][name] == digest(git_raw)
    source_map[name] = dict(observed, matches_git_head=True, matches_actual_owner_receipt=True)
wrapper_map = {name:dict(record(WRAP/name),matches=True) for name in sorted(B.WRAPPER_FILES)}
helper_map = {name:record(HELP/name) for name in ('fresh200_dispatch_common.py','prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py')}
entry_map = {name:dict(record(Path(inputs['runtime_entries'][name]['path'])), matches_wrapper=True) for name in B.RUNTIME_ENTRIES}
maps = save('BYTE_MAPS.json', dict(git_head=head, source_clean=clean, source_files=source_map,
    wrapper_files=wrapper_map, helper_files=helper_map, runtime_entries=entry_map,
    pins={str(path):record(path) for path in pins}, source_owner_receipt=record(owner_path),
    all_frozen_inventory_entries_verified=True, actual_gpu_executed=False, actual_remote_executed=False, actual_slurm_executed=False))

dry_path = WRAP/'V8_ACTUAL_TRACKED_RUNTIME_DRY_SELECTION.json'
dry = json.loads(dry_path.read_bytes())
tracked = subprocess.check_output(['git','ls-files','-z'],cwd=SOURCE).decode().strip('\0').split('\0')
selected = {name for name in tracked if B.source_file_allowed(name)}
assert selected == set(dry['selected_files'])
assert set(tracked)-selected == set(dry['excluded_files'])
assert (len(tracked),len(selected),len(tracked)-len(selected)) == (737,236,501)
for name in sorted(selected):
    observed=record(SOURCE/name)
    assert observed['sha256'] == dry['selected_files'][name]['sha256']
    assert observed['bytes'] == dry['selected_files'][name]['bytes']
for name in ('source/repo/verl/weights.pt','source/repo/search_r1/e5_Flat.index',
             'source/repo/verl/wiki-18.jsonl','source/repo/verl/wiki-18.jsonl.gz',
             'source/repo/public/chart.json','source/repo/reproduction/old.json',
             'source/repo/tests/fixture.json','source/repo/model.json'):
    rejected('forbidden_package_asset_or_path_'+name,lambda name=name:C.relative_file(name))
path_examples = {}
for name in ('source/repo/verl/assets/model.json','source/repo/search_r1/tests/fixture.json'):
    try: C.relative_file(name); path_examples[name]='ALLOWED_BY_PREFIX_AND_SUFFIX'
    except ValueError as error: path_examples[name]=repr(error)
selection = save('SOURCE_SELECTION_CHECK.json',dict(dry_selection=record(dry_path),git_head=head,
    tracked_count=len(tracked),selected_count=len(selected),excluded_count=len(tracked)-len(selected),
    selected_current_code_bytes_reverified=True,excluded_binary_bodies_read=False,
    guaranteed_scope='Actual frozen Git runtime prefix/suffix selection, not semantic content classification or dynamic dependency proof',
    allowed_in_prefix_metadata_examples=path_examples))
boundaries = save('PURE_FUNCTION_BOUNDARIES.json',dict(actual_frozen_prepare_wrapper_inventory_pass=True,
    wrapper_count=15,entry_count=4,source_owner_count=15,checks=checks,source_owner_receipt_verified_against_current_clean_git=True,
    owner_test_PASS_treated_as_independent_acceptance=False,actual_gpu_executed=False,actual_remote_executed=False,actual_slurm_executed=False))

test_specs = [('HELPER_CPU_TESTS', HELP, ['test_fresh200_dispatch_cpu']),
              ('OBSERVER_CPU_TESTS', WRAP, ['test_validation_adapter_cpu']),
              ('OPTIMIZE_BOUNDARY_CPU_TESTS', WRAP, ['test_runtime_contract_cpu.OptimizationBoundaryTests'])]
test_results = []
for label,cwd,targets in test_specs:
    command=[sys.executable,'-B','-m','unittest','-v',*targets]
    before=time.monotonic()
    result=subprocess.run(command,cwd=cwd,text=True,capture_output=True,timeout=60,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',CUDA_VISIBLE_DEVICES=''))
    elapsed=time.monotonic()-before
    (OUT/(label+'.stdout')).write_text(result.stdout)
    (OUT/(label+'.stderr')).write_text(result.stderr)
    test_results.append(dict(name=label,command=command,cwd=str(cwd),actual_exit_code=result.returncode,
        actual_seconds=elapsed,stdout=record(OUT/(label+'.stdout')),stderr=record(OUT/(label+'.stderr'))))
    assert result.returncode==0,result.stderr
save('LOCAL_CPU_TEST_RESULTS.json',dict(tests=test_results,actual_remote_executed=False,
    actual_gpu_executed=False,actual_slurm_executed=False,actual_official_model_assets_loaded=False,
    actual_historical_pt_loaded=False,actual_hydra_preparation_executed=False,
    test_scope='Synthetic fixtures and exact guard text with all hardware/proc/cancel calls stubbed; no helper main, real sbatch or production backend'))
for path,expected in pins.items(): assert record(path)['sha256']==expected
for name,entry in helper_map.items(): assert record(HELP/name)['sha256']==entry['sha256']
for name,entry in wrapper_map.items(): assert record(WRAP/name)['sha256']==entry['sha256']
print(json.dumps(dict(status='INDEPENDENT_LOCAL_BYTE_AND_CPU_CHECKS_PASSED',byte_maps=maps,
    boundaries=boundaries,source_selection=selection,tests=test_results,actual_gpu_executed=False,
    actual_remote_executed=False,actual_slurm_executed=False),indent=2))
