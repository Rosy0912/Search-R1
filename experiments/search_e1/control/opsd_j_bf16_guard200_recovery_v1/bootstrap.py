import sys,json,hashlib,importlib.util
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010')
cfg=json.loads((B/'receipts/training-config.json').read_text());H=Path(cfg['paths']['helper_root']);R=Path(__file__).parent
sys.path.insert(0,str(H))
manifest=json.loads((R/'source-manifest.json').read_text())
assert hashlib.sha256((H/'miner_v1.py').read_bytes()).hexdigest()==manifest['original_miner_sha256']
for name,entry in manifest['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==entry['sha256']
spec=importlib.util.spec_from_file_location('miner_v1',R/'miner_v1.py');module=importlib.util.module_from_spec(spec);sys.modules['miner_v1']=module;spec.loader.exec_module(module)
