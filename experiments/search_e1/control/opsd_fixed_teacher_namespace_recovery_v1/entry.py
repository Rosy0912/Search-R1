import os,json,pathlib,hashlib
from bootstrap import apply,B,H
apply()
from controller import Controller
from checkpoint import fileproof,checked_json,exclusive_json,require
R=pathlib.Path(__file__).parent
class Recovery(Controller):
 def spawn(self,command,log,phase,extra_env=None):
  if str(H/'trainer.py') in command:
   k=command.index(str(H/'trainer.py'));command=command[:k]+[str(R/'child_entry.py')]+command[k:]
  return super().spawn(command,log,phase,extra_env)
def main():
 pub=json.loads((R/'publication.json').read_text());require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha'],'Unpublished recovery')
 m=json.loads((R/'source-manifest.json').read_text());require(all(hashlib.sha256((R/n).read_bytes()).hexdigest()==v for n,v in m['files'].items()),'Recovery source mismatch')
 cfg=fileproof(B/'receipts/training-config.json');c,_=checked_json(cfg);require(c['run_id']==os.environ['E1_RUN_ID'],'Wrong same-run environment')
 status=json.loads((B/'runs/status.json').read_text());require(status['phase']=='FAILED_OWNED_RELEASE_CONFIRMED' and not pathlib.Path('/proc',str(status['owner']['pid'])).exists(),'Previous owner not released')
 exclusive_json(B/'receipts/namespace-recovery-launch.json',dict(accepted=True,publication=pub,config=cfg,previous_status=status,pid=os.getpid(),same_sampling_reused=True,no_optimizer_reset=True,change='Exact pruning namespace only'))
 owner=Recovery(cfg,c['run_id'],resume=True)
 try:owner.run()
 except Exception as e:
  if owner.owns_output:owner.contain_failure(e)
  raise
if __name__=='__main__':main()
