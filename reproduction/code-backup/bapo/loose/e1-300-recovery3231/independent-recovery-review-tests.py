"""Independent counterexamples; imports fixture only, never edits recovery core."""
import importlib.util,json,tempfile
from pathlib import Path
from pytest import MonkeyPatch
from search_r1.baselines import recovery as r
p=Path('work/bapo-e1-300-recovery/tests/recovery/test_recovery_state.py');spec=importlib.util.spec_from_file_location('creator_fixture',p);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
findings=[]
with tempfile.TemporaryDirectory()as tmp,MonkeyPatch.context()as mp:
 m,path,c,run,permit,permission=fixture.fixture_manifest(Path(tmp).resolve(),mp)
 # The exact public gate accepts the creator fixture's gpu={"actual":true}.
 r.validate_manifest(path,c,'new',run,permission)
 findings.append(dict(test='ARBITRARY_JSON_ACCEPTED_AS_GPU_WITNESS',result='REPRODUCED',gpu_json=json.loads(Path(permit['gpu_witnesses'][0]['path']).read_text()),accepted_without='8rank/model/Adam/driver/process/Slurm/source/runtime proof'))
 # Replace it with a text file: hashes remain correct, but no GPU semantics exist.
 gpu=permit['gpu_witnesses'][0];Path(gpu['path']).write_text('not json and not a GPU witness');gpu['bytes']=Path(gpu['path']).stat().st_size;gpu['sha256']=r.digest(gpu['path']);permission.write_text(json.dumps(permit))
 r.validate_manifest(path,c,'new',run,permission)
 findings.append(dict(test='NON_JSON_TEXT_ACCEPTED_AS_GPU_WITNESS',result='REPRODUCED'))
# Separately demonstrate truncation boundary; immutable actual25 file remains unchanged.
p=fixture.native();p['optimizer']['state'][0]['step'].fill_(500.5);r.validate_native(p,fixture.report(p),0,8)
findings.append(dict(test='ADAM_FRACTIONAL_500_5_ACCEPTED',result='REPRODUCED',scope='validator boundary only; actual old immutable producer step500 not alleged corrupt'))
output=dict(status='MUST_FIX_INDEPENDENT_RECOVERY_REVIEW_COUNTEREXAMPLES',findings=findings)
Path('work/evidence/recovery3231-independent-core-counterexamples.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
