"""Render immutable candidate only after exact reviewed source commit is available."""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=['capacity_job.sh','capacity_entry.py','rank_entry.py','resource_sample.py','retriever_entry.py']
SOURCES=['reproduction/baselines/e1-300/microbatch_probe_e1.py','reproduction/h20/retriever_local.py','search_r1/search/retrieval_server.py','search_r1/baselines/preflight.py','search_r1/baselines/e1_profile.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(repo,source,output):
 repo=Path(repo);output=Path(output)
 if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('Actual reviewed source commit required')
 if not repo.is_absolute() or repo.resolve(strict=True)!=repo:raise ValueError('Physical actual source repo required')
 if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=source or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():raise ValueError('Exact clean reviewed source required')
 if sha(repo/SOURCES[0])!='bbe157e70f959fb654f0d10836d91f32a1c6af0919eacdec8b2caf5c446e9b2b':raise ValueError('Frozen actualprobeSHA changed')
 if output.exists() or output.is_symlink():raise FileExistsError('Never overwrite reviewed tools')
 output.mkdir()
 for f in FILES:shutil.copyfile(HERE/'capacity-tools'/f,output/f)
 manifest={'source_commit':source,'tools':{f:sha(output/f) for f in FILES},'source_files':{f:sha(repo/f) for f in SOURCES},'retrieval_asset_manifest_sha256':'87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87','scope':'CPU-prepared capacity wrapper only; no actual GPU acceptance'}
 p=output/'manifest.json';p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return sha(p)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(render(a.repo,a.source,a.output))
