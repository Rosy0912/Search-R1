"""CPU exact source activation relationship; never execute activated source."""
import copy,hashlib,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
import source_bundle_check as gate

def prepared(tmp_path):
 template=json.loads((HERE/'source-manifest.json').read_text());records={}
 for name in template['file_sources']:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(gate.activated_bytes(name,(HERE/name).read_bytes()));records[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 return {'schema':'corrected-eval-activated-tool-bundle-v1','eval_commit':template['eval_commit'],'template_manifest_sha256':hashlib.sha256((HERE/'source-manifest.json').read_bytes()).hexdigest(),'file_sources':records}

def test_exact_guard_only_transform(tmp_path):gate.verify_bundle(prepared(tmp_path))
@pytest.mark.parametrize('mutation',['guard','code','missing','wrong_ancestry','other_commit'])
def test_any_other_activation_or_source_relationship_rejected(tmp_path,mutation):
 value=prepared(tmp_path)
 if mutation in ('guard','code'):
  p=Path(value['file_sources']['eval_job.py']['path']);p.write_bytes((HERE/'eval_job.py').read_bytes() if mutation=='guard' else p.read_bytes()+b'\nprint("changed")\n');value['file_sources']['eval_job.py']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 elif mutation=='missing':value['file_sources'].pop('eval_supervisor.py')
 elif mutation=='wrong_ancestry':value['template_manifest_sha256']='0'*64
 elif mutation=='other_commit':value['eval_commit']='0'*40
 with pytest.raises(ValueError):gate.verify_bundle(value)
