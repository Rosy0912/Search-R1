"""Build metadata for a staged source package; does not create GPU release."""
import argparse
from pathlib import Path
import re
from probe_package import (PROTOCOL,SELECTION_SHA,atomic_json,bundle_payload,decode,digest,file_sha,require)


def build_manifest(package_root,spec):
    root=Path(package_root).resolve()
    template=decode((root/'bundle-template.json').read_bytes())
    require(re.fullmatch('[0-9a-f]{40}',spec['runner_commit']),'Frozen runner commit required')
    require(template['runner_commit']==spec['runner_commit'],'Bundle runner commit differs')
    require(template['selection_file']['sha256']==SELECTION_SHA,'CPU3277 selection must be fixed')
    require('release' not in template,'Separate release is excluded from the static template')
    files={}
    for path in sorted(root.rglob('*')):
        parts=path.relative_to(root).parts
        if '.git' in parts:continue
        require('__pycache__' not in parts and path.suffix not in ('.pyc','.pyo'),
                'Stage without unbound Python bytecode')
        require(not path.is_symlink(),'Alias in source package: '+str(path))
        if not path.is_file() or path.name=='package-manifest.json':continue
        require(path.suffix not in {'.pt','.safetensors'},'Actual tensor/HF assets must stay in existing same-node dirs')
        files[path.relative_to(root).as_posix()]=file_sha(path)
    required={'start.sh','start.py','supervisor.py','probe_package.py','dispatch_once.py',
              'bundle-template.json','repo/search_r1/baselines/preflight.py',
              'repo/search_r1/baselines/probability_probe_runner.py',
              'repo/search_r1/baselines/probability_probe_loader.py',
              'repo/search_r1/baselines/probability_probe_capture.py',
              'repo/search_r1/baselines/probability_probe_io.py',
              'repo/search_r1/baselines/probability_probe_contracts.py',
              'repo/search_r1/baselines/probability_probe_monitor.py'}
    require(required <= set(files),'Stage complete exact reviewed runner repository before freeze')
    for field in ('reviewed_resource_adapter','probe_source_files','installed_sources','installed_versions'):
        require(spec[field]==template[field],'Spec/template source binding differs: '+field)
    require(type(spec['torchrun_port']) is int and 1024<=spec['torchrun_port']<65536,'Explicit loopback port required')
    return dict(schema_version=1,scope='FROZEN_SOURCE_CONTENT_WITHOUT_GPU_RELEASE',protocol=PROTOCOL,
        files=files,external_files=spec['external_files'],runner_commit=spec['runner_commit'],
        selection_sha256=SELECTION_SHA,bundle_payload_sha256=digest(bundle_payload(template)),
        reviewed_resource_adapter=spec['reviewed_resource_adapter'],probe_source_files=spec['probe_source_files'],
        installed_sources=spec['installed_sources'],installed_versions=spec['installed_versions'],
        torchrun_port=spec['torchrun_port'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-root',required=True);p.add_argument('--spec',required=True)
    a=p.parse_args();root=Path(a.package_root)
    manifest=build_manifest(root,decode(Path(a.spec).read_bytes()))
    record=atomic_json(root/'package-manifest.json',manifest)
    print('SOURCE_PACKAGE_FROZEN_WITHOUT_GPU_RELEASE',record['sha256'])


if __name__=='__main__':main()
