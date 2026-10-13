"""Render exact own shared systemd argv only; never starts a service."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from production_contract import ROOT,require,read_bound

def render(package,manifest_sha,release_path,release_sha):
    package=Path(package);release_path=Path(release_path)
    require(package.is_absolute() and package.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve()),'Own shared candidate required')
    require(release_path.is_absolute() and release_path.resolve().is_relative_to(ROOT.resolve()),'Own external release required')
    require(all(type(h) is str and re.fullmatch('[0-9a-f]{64}',h) for h in (manifest_sha,release_sha)),'Exact package/release pins required')
    value,_,_=read_bound(package/'manifest.json',manifest_sha)
    require(value['dispatch']['kind']=='shared_systemd' and value['dispatch']['slurm_reserved_gpus']==0,'Separate shared dispatch protocol required')
    source=value['code_commit'];require(re.fullmatch('[0-9a-f]{40}',source),'Published source identity required')
    unit='sst-tree-nested-branch-fresh200-'+source[:8]+'.service'
    environment={
      'TREE_SHARED_TRAINING_MODE':'1','TREE_SHARED_SYSTEMD_UNIT':unit,'TREE_SHARED_RESOURCE_PROTOCOL':'training_memory_headroom_shared_v1',
      'SLURM_JOB_ID':'not-slurm;systemd:'+source,
      'TREE_INSTRUCT_TOOLS':str(package),'TREE_INSTRUCT_PACKAGE_SHA256':manifest_sha,
      'TREE_PRODUCTION_RELEASE_PATH':str(release_path),'TREE_INSTRUCT_REVIEW_SHA256':release_sha,
      'PYTHONPATH':str(package)+':'+str(ROOT/'tools')+':'+str(package/'source/repo'),
      'PYTHONOPTIMIZE':'0','CUDA_VISIBLE_DEVICES':','.join(value['dispatch']['expected_gpu_uuids'])}
    # Own .service has no scheduler reservation or foreign job cgroup/device policy.
    return ['systemd-run','--unit='+unit,'--property=CPUQuota=3200%','--property=MemoryMax=256G',
            '--property=TasksMax=4096','--property=KillMode=control-group','--property=TimeoutStopSec=60',
            '--property=RuntimeMaxSec=604800','--property=Restart=no','--property=Nice=5',
            *['--setenv='+key+'='+item for key,item in sorted(environment.items())],
            str(ROOT/'env/bin/python'),str(package/'shared_launch.py')]

def payload(package,manifest_sha,release_path,release_sha):
    return (json.dumps(render(package,manifest_sha,release_path,release_sha),sort_keys=True,separators=(',',':'))+'\n').encode()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','manifest-sha','release-path','release-sha','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();data=payload(a.package,a.manifest_sha,a.release_path,a.release_sha);path=Path(a.output)
    with path.open('xb') as stream:stream.write(data)
    print(json.dumps(dict(status='SHARED_SYSTEMD_ARGV_RENDERED_NOT_STARTED',path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),physical_gpus=8,slurm_reserved_gpus=0)))
if __name__=='__main__':main()
