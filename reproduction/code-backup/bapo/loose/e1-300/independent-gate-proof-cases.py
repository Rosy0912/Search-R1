"""Independent local CPU proof-consistency oracle; no training acceptance."""
import json,runpy,tempfile,os
from pathlib import Path
from search_r1.baselines.checkpoint_gate import validate_release,digest
from search_r1.baselines.full_state import atomic_save,state_digest
ns=runpy.run_path('work/bapo-e1-300/tests/baselines/test_e1_checkpoint_gate.py')
records=[]
for kind in ['valid','config-header','rank-alias','driver-step','foreign-pid']:
    with tempfile.TemporaryDirectory() as tmp:
        repo,mp,rp,value=ns['fixture'](Path(tmp).resolve());m=json.loads(mp.read_text())
        if kind=='config-header':m['config_sha256']='0'*64;value['config_sha256']='0'*64
        if kind=='rank-alias':
            for report in m['rank_roundtrips'][1:]:
                report['path']=m['rank_roundtrips'][0]['path'];report['sha256']=m['rank_roundtrips'][0]['sha256']
        if kind in ['driver-step','foreign-pid']:
            path=Path(next(x['path'] for x in m['files'] if x['path'].endswith('/driver-reload.json')))
            report=json.loads(path.read_text())
            if kind=='driver-step':report['step']=999
            else:
                import torch
                state=torch.load(report['path'],weights_only=False,map_location='cpu');state['driver_pid']=999
                atomic_save(state,report['path']);report.update(pid=999,sha256=digest(report['path']),state_digest=state_digest(state))
            path.write_text(json.dumps(report))
        for member in m['files']:member['sha256']=digest(member['path'])
        mp.write_text(json.dumps(m));value['checkpoint_manifest_sha256']=digest(mp)
        for item in value['evidence']:item['sha256']=digest(item['path'])
        rp.write_text(json.dumps(value))
        try:
            # The offline validator can be independently replayed with the actual
            # expected live driver PID. The fixture baseline records current PID.
            import inspect
            kwargs={'expected_driver_pid':os.getpid()} if 'expected_driver_pid' in inspect.signature(validate_release).parameters else {}
            validate_release(rp,mp,repo,**kwargs);accepted=True;error=None
        except Exception as exc:accepted=False;error=str(exc)
        records.append(dict(case=kind,accepted=accepted,error=error))
assert records[0]['accepted'],records
assert all(not x['accepted'] for x in records[1:]),records
Path('work/evidence/e1-300-gate-independent-final-proof-cases.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
