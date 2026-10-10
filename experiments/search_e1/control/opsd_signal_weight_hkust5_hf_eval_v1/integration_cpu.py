"""Real CPU input integration, no model/Torch/GPU/HTTP calls."""
import json,sys,time
from pathlib import Path
import entry

def main():
 m=entry.setup(dict(path=str(entry.S/'receipts/hkust5-isolation-binding-v2.json'),sha256=entry.BINDING_SHA))
 from dependencies import verify_core
 dependency=verify_core()
 study=m.read(dict(path=str(entry.S/'receipts/study-manifest.json'),sha256='86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e'))
 configs=[m.read(g['config']) for g in study['groups']]
 for c,g in zip(configs,study['groups']):c['_config_sha256']=g['config']['sha256']
 import prepare
 for c in configs:prepare.validate_config(c,launch=True)
 config=configs[0];m.modules(config)
 import evaluation
 complete=json.loads((Path(config['paths']['output_root'])/'outers/outer-00014/training/outer-complete.json').read_text())
 identity,_=evaluation.input_identity(config,15,complete['checkpoint'])
 assert len(identity['source_files'])==12 and identity['expected_adapter_tensor_sha256']
 assert m.PROTOCOL=='hf-sdpa-greedy-rep1p05-unmerged-fp32-lora-batch32-fixed512-v1'
 assert 'torch' not in sys.modules
 print(json.dumps(dict(accepted=True,four_configs_valid=True,GP15_full_source_and_raw_input_identity_pass=True,checkpoint=complete['checkpoint'],batch_core=dependency,torch_imported=False,gpu_started=False,terminal='HOLD_ALL100_NOT_COMPLETE',observed_unix=time.time())))
if __name__=='__main__':main()
