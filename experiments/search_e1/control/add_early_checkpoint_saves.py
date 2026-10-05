"""Add saves2/5 to a new isolated R1 copy; never alter an original runtime."""
import os,json,hashlib,ast
from pathlib import Path
b=Path(os.environ['E1_BASE']);rt=b/'runtime/r1'
assert b.name=='search-e1-config-study-r1-aligned-8gpu-20261005'
manifest=b/'receipts/r1-runtime.sha256.json';m=json.loads(manifest.read_text())
assert all(hashlib.sha256((rt/k).read_bytes()).hexdigest()==v for k,v in m.items())
audit=b/'receipts/early-checkpoints-overlay.json'
if not audit.exists():
 p=rt/'upstream/Search-R1/verl/trainer/ppo/ray_trainer.py';old=p.read_text()
 needle='if self.config.trainer.save_freq > 0 and \\\n+                            self.global_steps % self.config.trainer.save_freq == 0:'
 assert old.count(needle)==1
 new=old.replace(needle,'if self.global_steps in [2, 5] or (self.config.trainer.save_freq > 0 and \\\n+                            self.global_steps % self.config.trainer.save_freq == 0):',1)
 ast.parse(new);p.write_text(new)
 key=str(p.relative_to(rt));m[key]=hashlib.sha256(p.read_bytes()).hexdigest()
 manifest.write_text(json.dumps(m,indent=2))
 audit.write_text(json.dumps(dict(changed_file=key,original_sha256=hashlib.sha256(old.encode()).hexdigest(),new_sha256=m[key],extra_checkpoint_steps=[2,5],training_math_changes=False),indent=2))
print('NEW_RUNTIME_SAVES_VERIFIED',len(m))
