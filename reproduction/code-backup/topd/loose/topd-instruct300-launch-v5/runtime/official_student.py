"""Validate already-computed native inventory against official asset metadata."""
import json,pathlib
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
MANIFEST_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
PUBLIC_SHA='2182e6ac33d1c7f1116d1c85ea0531f1e14565bcdbb14668ddc69e07a37302f3'
def verify(inventory,binding):
 root=pathlib.Path(inventory['directory']);files=inventory['files']
 if binding['revision']!=REVISION or binding['repo']!='Qwen/Qwen2.5-3B-Instruct':raise ValueError('Official student revision differs')
 for key,name,expected in [('verified_manifest','VERIFIED-MODEL-MANIFEST.json',MANIFEST_SHA),('public_source_api','PUBLIC-SOURCE-API.json',PUBLIC_SHA)]:
  if binding[key]!={'path':str(root/name),'sha256':expected} or files[name]['sha256']!=expected:raise ValueError('Official student source metadata drift')
 manifest=json.loads((root/'VERIFIED-MODEL-MANIFEST.json').read_text());public=json.loads((root/'PUBLIC-SOURCE-API.json').read_text())
 if manifest['repo']!=binding['repo'] or manifest['revision']!=REVISION or public['id']!=binding['repo'] or public['sha']!=REVISION:raise ValueError('Official student source identity mismatch')
 expected={r['file']:{'size':r['bytes'],'sha256':r['sha256']} for r in manifest['files']}
 if binding['official_files']!=expected or set(files)!=set(expected)|{'VERIFIED-MODEL-MANIFEST.json','PUBLIC-SOURCE-API.json'}:raise ValueError('Official student exact asset set drift')
 for name,record in expected.items():
  if files[name]!=record:raise ValueError('Official student actual file SHA/size drift '+name)
 return {'repo':binding['repo'],'revision':REVISION,'official_files':expected,'verified_manifest':binding['verified_manifest'],'public_source_api':binding['public_source_api'],'source':'ALREADY_COMPUTED_FRESH_HF_INVENTORY_NO_SECOND_WEIGHT_SCAN'}
