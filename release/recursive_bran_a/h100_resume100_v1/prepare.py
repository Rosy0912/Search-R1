"""Create isolated run configuration; do not change original config or checkpoint."""
import hashlib,json,os,uuid
from pathlib import Path
from omegaconf import OmegaConf
b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012');r=b/'recovery-from100-20261013'
(r/'receipts').mkdir(exist_ok=True)
assert not (r/'receipts/configuration.json').exists(),'Already prepared; inspect, do not overwrite'
old=json.loads((b/'receipts/configuration.json').read_text())
parent=b/'checkpoints/training_state/global_step_100/post-evaluation/manifest.json';raw=parent.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='bb0599ced9db88eaa5dfec21ead76d214a8a1f5b4a64aa7baa4b85a41ceebce2'
m=json.loads(raw);assert m['boundary_phase']=='post-evaluation'
for x in m['worker_states']+[m['driver']]:assert Path(x['path']).is_file()
retr=r/'retriever';(retr/'receipts').mkdir(parents=True,exist_ok=True)
(retr/'assets').symlink_to(b/'retriever/assets')
token=retr/'receipts/private-probe-token';token.write_bytes((b/'retriever/receipts/private-probe-token').read_bytes());token.chmod(0o600)
cfg=OmegaConf.load(old['config']);cfg.trainer.default_local_dir=str(r/'checkpoints');cfg.retriever.control_token_file=str(token)
text=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)))
p=r/'receipts/resolved.yaml';p.write_text(text)
new=dict(old,run_id=uuid.uuid4().hex,config=str(p),config_sha256=hashlib.sha256(text.encode()).hexdigest(),source_commit=os.environ['RBA_RECOVERY_SOURCE_COMMIT'],fresh_authorized=False,parent_checkpoint=str(parent))
(r/'receipts/configuration.json').write_text(json.dumps(new,indent=2))
binding=dict(protocol='recursive_a_sparse_schedule_continuation_v1',handoff_contract='h100_saved100_after_allocation_terminal_v1',parent_completed_step=100,parent_optimizer_calls=2000,parent_manifest=str(parent),parent_manifest_sha256=hashlib.sha256(raw).hexdigest(),parent_config=old['config'],parent_config_sha256=old['config_sha256'],parent_identity=m['identity'],child_checkpoint_dir=str(r/'checkpoints'),child_control_token_file=str(token),limits='Saved100 replay; 101-118 and partial119 unpersisted lost; vLLM RNG not bitwise restored')
(r/'receipts/continuation-binding.json').write_text(json.dumps(binding,indent=2))
print(json.dumps(new),flush=True)
