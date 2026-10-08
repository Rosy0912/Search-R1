"""Source-only candidate entry: validate envelope before srun/preflight."""
import argparse
import os
from pathlib import Path
import socket
import sys
from probe_package import ROOT, file_sha, read_bound, require, same_node_path, validate_package_files, validate_release


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--package',required=True)
    parser.add_argument('--manifest-sha256','--manifest-sha',required=True); parser.add_argument('--release',required=True)
    parser.add_argument('--release-sha256','--release-sha',required=True); args=parser.parse_args()
    package=same_node_path(args.package)
    require(package == Path(__file__).resolve().parent, 'Entry source/package differs')
    manifest=read_bound(package/'package-manifest.json',args.manifest_sha256)
    validate_package_files(package,manifest)
    release_path=same_node_path(args.release)
    release=read_bound(release_path,args.release_sha256)
    template=read_bound(package/'bundle-template.json',manifest['files']['bundle-template.json'])
    validate_release(manifest,args.manifest_sha256,template,release)
    require(socket.gethostname().split('.')[0]=='h100-9' and
            os.environ.get('SLURM_JOB_NODELIST')=='h100-9' and os.environ.get('SLURM_JOB_GPUS'),
            'Entry must run in actual h100-9 GPU Slurm allocation')
    python=str(ROOT/'env/bin/python')
    env=dict(os.environ,PATH=str(ROOT/'env/bin')+':/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(package/'repo')+':'+str(package)+':'+str(ROOT/'tools'),
             TREE_PROBE_PACKAGE_MANIFEST_PATH=str(package/'package-manifest.json'),
             TREE_PROBE_PACKAGE_MANIFEST_SHA256=args.manifest_sha256,
             TREE_PROBE_RELEASE_PATH=str(same_node_path(args.release)),TREE_PROBE_RELEASE_SHA256=args.release_sha256)
    command=['srun','--ntasks=1','--kill-on-bad-exit=1',python,'-m','search_r1.baselines.preflight','--',
             python,str(package/'supervisor.py'),'--package',str(package),'--manifest-sha256',args.manifest_sha256,
             '--release',str(release_path),'--release-sha256',args.release_sha256]
    os.chdir(package/'repo')
    os.execvpe(command[0],command,env)


if __name__=='__main__': main()
