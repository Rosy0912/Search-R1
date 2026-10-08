"""Offline metadata generation only; no submission, receipt, approval or signing."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
OLD='89beac273f783d8e48d7d3e34da040f753514d90'
FORMAL_PIN='209513e681cd5814a1aa7c1952e28a1e95038da5e724127e248275bfe54e4653'
V2_PIN='74856b5f245cd7186ca7b40bcf2f949e3cfe26c59875ac279df46a43d5fffb58'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,msg):
 if not x:raise ValueError(msg)
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def generate(repo,source,output,base):
 need(re.fullmatch('[0-9a-f]{40}',source) and source!=OLD,'Require actual new full commit')
 need(repo.is_absolute() and repo.resolve(strict=True)==repo,'Physical repo required')
 need(git(repo,'rev-parse','HEAD')==source and not git(repo,'status','--porcelain'),'Require actual clean committed HEAD')
 need(set(git(repo,'diff','--name-only',OLD,source).splitlines())=={'search_r1/baselines/actor.py','search_r1/baselines/checkpoint_gate.py','tests/baselines/test_opsd_e1_checkpoint_gate.py'},'Unexpected source delta')
 templates=[('formal',base/'formal-tools-89beac2','manifest.json',FORMAL_PIN),('first5',base/'first5-tools-service-marker-live-state-v2','candidate-manifest.json',V2_PIN)]
 for _,directory,name,pin in templates:need(sha(directory/name)==pin,'Template manifest changed')
 fm=json.loads((templates[0][1]/'manifest.json').read_text());v2=json.loads((templates[1][1]/'candidate-manifest.json').read_text())
 for n,pin in fm['tools'].items():need(sha(templates[0][1]/n)==pin,'Formal template drift')
 for n,pin in v2['files'].items():need(sha(templates[1][1]/n)==pin,'V2 template drift')
 # Critical source must be the independently reviewed implementation bytes.
 need(sha(repo/'search_r1/baselines/actor.py')=='a9e5005f9d5e48f2cba652ee88b2ff2f3b0bb671dde8ce70ea7653eeb82c5ed9','Actor differs from reviewed eager selector')
 need(sha(repo/'search_r1/baselines/checkpoint_gate.py')=='74c3c13929c4f5cb784b90b7fd568ee80dfe1d37c8efaa28f97aae60b3492632','Gate differs from strict parser reviewed bytes')
 source_files={n:sha(repo/n) for n in fm['source_files']}
 for n,pin in fm['source_files'].items():
  if n not in ('search_r1/baselines/actor.py','search_r1/baselines/checkpoint_gate.py'):need(source_files[n]==pin,'Nonreviewed source/math/config changed')
 need(output.is_absolute() and output.parent.resolve(strict=True)==output.parent,'Physical output parent required');output.mkdir()
 for kind,directory,_,_ in templates:
  dest=output/kind;dest.mkdir();names=fm['tools'] if kind=='formal' else {'receipt_payload.py','hf_cpu_reload.py'}
  tools={}
  for n in sorted(names):
   raw=(directory/n).read_bytes();count=raw.count(OLD.encode());expected=(1 if n in ('formal_supervisor.py','cpu_import_check.py','hardware_rank.py','receipt_payload.py','hf_cpu_reload.py') else 0)
   need(count==expected,'Unexpected source replacement count: '+n)
   (dest/n).write_bytes(raw.replace(OLD.encode(),source.encode()));tools[n]=sha(dest/n)
  manifest=dict(fm) if kind=='formal' else json.loads((base/'first5-tools/manifest.json').read_text())
  manifest.update(source_commit=source,source_files=source_files,tools=tools,scope='REBOUND_CANDIDATE_NOT_REVIEWED_NOT_DEPLOYED_NO_ACCEPTANCE')
  (dest/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 report={'source_commit':source,'repo':str(repo),'templates':{'formal':FORMAL_PIN,'first5_v2':V2_PIN},'output_manifests':{k:sha(output/k/'manifest.json') for k in ('formal','first5')},'pending':['formal shared-parser delta + independent review','native actual new source inventory external SHA','new owned allocation/preGPU ready review','fresh actual five state/HF/driver evidence and independently signed receipt'], 'actual_first5_accepted':False}
 (output/'REBINDING.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(generate(a.repo,a.source,a.output,Path(__file__).resolve().parent.parent),sort_keys=True))
