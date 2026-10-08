#!/usr/bin/env python3
"""Synchronous original G1 refit; unavailable staged32B binding fails closed."""
import argparse, hashlib, json, os, pathlib, types
import requests
from ts_coevolve import refit

def refresh(config,checkpoint,step,session=None,refitter=refit):
    binding=config.get('staged_dump_binding')
    if not isinstance(binding,dict) or binding.get('scope')!='INDEPENDENTLY_REVIEWED_G1_32B_STAGED_DUMP':
        raise RuntimeError('32B staged dump binding not supplied; no calibration or training readiness')
    source=pathlib.Path(binding['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=binding['source_sha256']: raise ValueError('staged dump source mismatch')
    if source.resolve()!=pathlib.Path('ts_probe_dump.py').resolve(): raise ValueError('refit must call this pinned original-path dump')
    manifest=pathlib.Path(config['manifest']); old=manifest.read_bytes()
    session=session or requests.Session();session.trust_env=False
    base=config['probe_url'].rsplit('/',1)[0]
    if not base.startswith('http://127.0.0.1:'): raise ValueError('own loopback probe only')
    pause=session.post(base+'/barrier_pause',timeout=600);pause.raise_for_status()
    if pause.json().get('status')!='PAUSED_ENCODER_RELEASED': raise ValueError('unknown pause receipt')
    try:
        args=types.SimpleNamespace(**config['refit_args'])
        if not refitter(config['experiment'],step,str(checkpoint),args,dict(os.environ)): raise RuntimeError('original G1 refit failed')
        resumed=session.post(base+'/barrier_resume',timeout=1200);resumed.raise_for_status()
        identity=resumed.json(); payload=json.loads(manifest.read_bytes())
        if identity.get('status')!='READY' or identity.get('round')!=step or identity.get('encoder')!=str(checkpoint) or identity.get('manifest_sha256')!=hashlib.sha256(manifest.read_bytes()).hexdigest(): raise ValueError('new encoder receipt mismatch')
        return identity
    except BaseException:
        # Keep the actual previous complete head/encoder manifest available.
        tmp=pathlib.Path(str(manifest)+'.restore.tmp')
        with open(tmp,'wb') as f:f.write(old);f.flush();os.fsync(f.fileno())
        os.replace(tmp,manifest)
        session.post(base+'/barrier_resume',timeout=1200).raise_for_status()
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--outer-step',required=True,type=int);a=ap.parse_args()
    refresh(json.loads(pathlib.Path(a.config).read_text()),a.checkpoint,a.outer_step)
if __name__=='__main__':main()
