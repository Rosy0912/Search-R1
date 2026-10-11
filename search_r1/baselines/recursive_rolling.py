"""Reclaim only accepted same-configuration complete older boundary states."""
import json, shutil
from pathlib import Path

PROTECTED={5,50,100,150,200}
def prune(base, current, config_sha, source_commit):
    base=Path(base).resolve()
    latest=base/'training_state'/f'global_step_{current}'/'manifest.json'
    record=json.loads(latest.read_text())
    if record['status']!='COMPLETE_SAME_WORLD_STATE_SAVED':
        raise ValueError('Rolling checkpoint not fully accepted')
    identity=record['identity']
    if identity['config_sha256']!=config_sha or identity['code_commit']!=source_commit:
        raise ValueError('Rolling identity mismatch')
    for directory in (base/'training_state').glob('global_step_*'):
        step=int(directory.name.removeprefix('global_step_'))
        if step in PROTECTED or step>=current-1:continue
        manifest=directory/'manifest.json'
        if not manifest.is_file():continue
        old=json.loads(manifest.read_text()); i=old['identity']
        if (old['status']!='COMPLETE_SAME_WORLD_STATE_SAVED' or
            i['config_sha256']!=config_sha or i['code_commit']!=source_commit or
            i['job_id']!=identity['job_id']):continue
        actor=Path(old['hf_checkpoint']).resolve()
        expected=base/'actor'/directory.name
        if actor!=expected or actor.parent!=base/'actor':
            raise ValueError('Refuse unexpected checkpoint path')
        # Preserve boundary identity/manifest; remove only recoverable old payload.
        for p in directory.iterdir():
            if p.name=='manifest.json':continue
            if p.is_dir():shutil.rmtree(p)
            else:p.unlink()
        if actor.exists():shutil.rmtree(actor)
        (directory/'reclaimed.json').write_text(json.dumps(dict(reclaimed_at_outer=current,
             protected=False,manifest_preserved=True)))
