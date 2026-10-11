"""Verify real immutable inputs/private namespace only; no Torch/GPU/HTTP."""
import json,sys
import entry,common

def main():
 hf=sys.modules['_signal_k5_HF_resources']
 values=[]
 import prepare
 for g,digest in hf.SHAS.items():
  ref=dict(path=str(entry.S/('group-'+g)/'receipts/config.json'),sha256=digest)
  c,proof=common.read_json(ref);c['_config_sha256']=proof['sha256'];prepare.validate_config(c,launch=True)
  modules=common.bind_training(c)
  assert c['owner']['physical_gpus']==[0,1,2,3] and modules['gates'].inventory.__module__=='_signal_k5_migration_resources'
  values.append(g)
 for ref in entry.G_TERMINALS.values():assert entry.pinned(ref)['actual_wait_returncode']==0
 assert 'torch' not in sys.modules
 print(json.dumps(dict(accepted=True,four_configs=values,original_math_byte_frozen=True,private_source_only_training_namespace=True,G_actual_wait0_inputs_present=True,torch_imported=False,gpu_started=False,terminal='HOLD_ALL100_AND_NINE_HF_ENDPOINTS_NOT_COMPLETE')))
if __name__=='__main__':main()
