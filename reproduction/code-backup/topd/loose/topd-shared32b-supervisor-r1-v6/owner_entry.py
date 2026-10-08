"""New bootstrap route for pinned owner body. Never changes original scoring math."""
import argparse,json,os,pathlib,sys
from contracts import *
from dependencies import load,gate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--mode',required=True);ap.add_argument('--template',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();c=json.loads(pathlib.Path(a.config).read_text())
 selected=gate(c,a.mode);owner=load('pinned_shared_owner',c['owner_source'])
 if a.mode=='profile':
  def bootstrap_assets(manifest,repo,pipeline):
   require(manifest['kind']=='NATIVE_HARDWARE_PROBE_NO_TRAINING','bootstrap cannot production mode')
   require(pathlib.Path(repo).resolve()==ROOT/'repo-shared32b','bootstrap actual source')
   source=artifact(manifest['source_manifest']);require(source['commit']==D23,'bootstrap source commit')
   p=selected['provenance'];proof={'fp32_parameter_sha256':p['expected_fp32_parameter_sha256'],'teacher_provenance':c['teacher_provenance'],'teacher_gate_source':c['teacher_gate_source']}
   owner.fresh_files(pathlib.Path(pipeline),owner.PIPELINE_PINS)
   return proof,pathlib.Path(selected['model']),source
  owner.owner_assets=bootstrap_assets
 sys.argv=[str(c['owner_source']['path']),'--manifest-template',a.template,'--directory',a.directory,'--repo',c['repo'],'--pipeline',str(ROOT/'tools/topd-32b-pipeline-candidate')]
 owner.main()
 from lifecycle import wait
 # Preserve known CUDA PID until parent verifies final actor update and stops active monitor.
 permission=wait(pathlib.Path(a.directory).parent/'authorized-terminal-exit.json',[],[],3600)
 require(permission['job']==os.environ['SLURM_JOB_ID'] and permission['parent_pid']==os.getppid(),'Owned terminal lifecycle permission mismatch')
if __name__=='__main__':main()
