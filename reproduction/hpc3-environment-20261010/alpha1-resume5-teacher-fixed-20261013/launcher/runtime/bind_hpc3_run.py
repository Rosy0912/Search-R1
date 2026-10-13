"""Bind an approved configuration to a unique actual Slurm run, without launching."""
import argparse
import json
import os
import pathlib
import re
import socket

def bind(config, job, node):
    if not re.fullmatch(r'[0-9]+', job) or not re.fullmatch(r'[A-Za-z0-9_.-]+', node):
        raise ValueError('Invalid actual allocation identity')
    old = config['evidence_root']
    new = str(pathlib.Path(old).parent.parent / ('topd-hpc3-' + job) / 'evidence')
    def remap(v):
        if isinstance(v, dict): return {k: remap(x) for k, x in v.items()}
        if isinstance(v, list): return [remap(x) for x in v]
        if isinstance(v, str): return v.replace(old, new)
        return v
    c = remap(config)
    c['training_overrides'] = [
        'baselines.local_runtime_host=' + node if s.startswith('baselines.local_runtime_host=') else s
        for s in c['training_overrides']]
    c['node'] = node
    c['prepared_allocation'] = {'job': job, 'node': node}
    return c

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--template', required=True)
    a.add_argument('--output', required=True)
    args = a.parse_args()
    node = socket.gethostname().split('.')[0]
    if node != os.environ.get('SLURMD_NODENAME', os.environ.get('SLURM_JOB_NODELIST')):
        raise ValueError('Actual Slurm host differs')
    c = bind(json.loads(pathlib.Path(args.template).read_text()), os.environ['SLURM_JOB_ID'], node)
    p = pathlib.Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(c, f, indent=2)
