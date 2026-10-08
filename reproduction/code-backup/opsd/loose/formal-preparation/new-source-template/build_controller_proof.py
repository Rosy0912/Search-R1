"""Local-only preparation from actual retained metadata, with no remote operations."""
import copy,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from verify_formal_prerequisites import digest,loads,own_file
from controller_exit_evidence import verify_controller_exit
ROOT=HERE.parents[1]
COMMIT='8e60976d364ab7d7ae51714415813c1359dafe9a'
REMOTE=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-new-acceptance-2994')
def build(output):
 out=Path(output);assert not out.exists();out.mkdir(parents=True)
 metadata=loads((ROOT/'job2994-2996-actual-metadata.json').read_text())
 names={'controller_origin':'controller-original-reviewer-origin.json','head_config':'controller-head-config.txt','start_snapshot':'slurm-job.txt','source_commit_evidence':'commit.txt'}
 texts={'controller_origin':(ROOT/'opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2994_controller_reviewer_raw.json').read_text(),'head_config':(ROOT/'job2994-controller-head-config.txt').read_text()}
 for field in ('start_snapshot','source_commit_evidence'):
  item=metadata[str(REMOTE/'evidence'/names[field])]
  texts[field]=item['text']
 for field,text in texts.items():
  p=out/names[field];p.write_text(text)
  if field in ('start_snapshot','source_commit_evidence'):assert digest(p)==metadata[str(REMOTE/'evidence'/names[field])]['sha256']
 data={'proof_kind':'controller_normal_exit','source_commit':COMMIT,'job_id':'2994','node':'h100-10','cpus':192,'controller_host':'h100-16','source_log_path':'/var/log/slurm/slurmctld.log'}
 for field,name in names.items():data[field]={'path':str((out/name).resolve()),'sha256':digest(out/name)}
 result=verify_controller_exit(data,'2994',COMMIT,out.resolve(),own_file,digest,loads)
 for field,name in names.items():data[field]['path']=str(REMOTE/'evidence'/name)
 (out/'controller-normal-exit-proof.json').write_text(json.dumps(data,indent=2)+'\n')
 (out/'local-proof-validation.json').write_text(json.dumps({'actual_retained_controller_proof_checked':result,'local_only_no_upload':True,'remote_proof_sha256':digest(out/'controller-normal-exit-proof.json'),'file_sha256':{name:digest(out/name) for name in names.values()}},indent=2)+'\n')
 return out
if __name__=='__main__':assert len(sys.argv)==2;print(build(sys.argv[1]))
