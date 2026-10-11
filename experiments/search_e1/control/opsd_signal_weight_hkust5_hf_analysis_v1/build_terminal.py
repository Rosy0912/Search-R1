"""Create only the real combined lineage packet after all100; no inferred old exit."""
import argparse,json
from pathlib import Path
import entry,common

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 # entry.py has already checked the own source and installed the exact HF guard.
 hf=__import__('_signal_k5_HF_resources')
 actual=hf.terminal();out=common.remote_output(a.output)
 require=common.require;require(actual['training_outer_total']==100,'All100 required')
 value=dict(schema='opsd-signal-k5-analysis-terminal-v1',accepted=True,migration_binding=entry.BINDING,
  migration_training_complete=common.file_ref(entry.S/'receipts/migration-training-complete.json'),
  configs={g:dict(path=str(entry.S/('group-'+g)/'receipts/config.json'),sha256=s) for g,s in hf.SHAS.items()},
  train_ledgers={g:common.file_ref(entry.S/('group-'+g)/'runs/training-em-25-complete.json') for g in hf.SHAS},
  G_actual_terminal=entry.G_TERMINALS,H_actual_terminal={g:common.file_ref(entry.S/'receipts'/('migration-'+g+'-terminal.json')) for g in ['H_POS','H_NEG']},
  historical_K4_whole_supervisor_exit_NOT_inferred=True)
 import replay
 cfg,_=common.read_json(value['configs']['G_POS']);cfg['_config_sha256']=value['configs']['G_POS']['sha256'];replay.verify_four_group_terminal(value,common.bind_training(cfg))
 print(json.dumps(common.exclusive_json(out,value)))
if __name__=='__main__':main()
