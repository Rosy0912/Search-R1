"""Local boundary tests only; node mutations are hard-disabled."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('repair_candidate',HERE/'node_completion_repair.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def fixture():
    evidence=HERE.parents[2]/'evidence'
    health=json.loads((evidence/'aligned-em200-hardware3345-health6.actual.json').read_text())
    observation=json.loads((evidence/'aligned-em200-node-queue3345-observation2.actual.json').read_text())
    old=json.loads((evidence/'aligned-em200-old3315-completion-readonly-observation2.actual.json').read_text())
    row=lambda text:dict(exit=0,stdout=text,stderr='')
    rows=dict(node=observation['node'],old=observation['old3315'],new=health['slurm'],
        node_queue=row('3315|COMPLETING|h100-5|old|'+h.OLD_COMMENT+'\n'),
        queue=row('3345|PENDING|(null)|new|'+h.NEW_COMMENT+'\n'),pids=old['pids'],steps=old['steps'],
        gpu=health['gpu'],version=row('slurm-wlm 21.08.5\n'),source=row(h.SOURCE+'\n'),old_source=row(h.OLD_SOURCE+'\n'),
        processes=row('755855 1 slurmstepd: [3315.0]\n'))
    daemon=dict(pid=755855,start_ticks=1433986891,boot_id=h.BOOT,exe_sha256=h.EXE_SHA)
    return rows,daemon

def test_saved_actual_admission_is_historical_not_execution_permission():
    rows,daemon=fixture();h.check_snapshot(rows,daemon)
    assert h.NODE_RECONCILIATION_ADMISSION is False

@pytest.mark.parametrize('change',['CPU','memory','foreign','gpu','birth','duplicate','started'])
def test_foreign_allocation_or_unknown_identity_rejected(change):
    rows,daemon=fixture()
    if change=='CPU':rows['node']['stdout']=rows['node']['stdout'].replace('CPUAlloc=0','CPUAlloc=1')
    if change=='memory':rows['node']['stdout']=rows['node']['stdout'].replace('AllocMem=0','AllocMem=1')
    if change=='foreign':rows['node_queue']['stdout']+='999|RUNNING|h100-5|foreign|foreign\n'
    if change=='gpu':rows['gpu']['stdout']='999,GPU-unknown,foreign\n'
    if change=='birth':daemon['start_ticks']+=1
    if change=='duplicate':rows['queue']['stdout']+='3346|PENDING|(null)|new|'+h.NEW_COMMENT+'\n'
    if change=='started':rows['new']['stdout']=rows['new']['stdout'].replace('JobState=PENDING','JobState=RUNNING')
    with pytest.raises(ValueError):h.check_snapshot(rows,daemon)

def test_apply_disabled_before_auth_io_or_any_command(tmp_path):
    calls=[]
    with pytest.raises(ValueError,match='BLOCKED'):
        h.apply(tmp_path/'missing-plan','x',tmp_path/'missing-authorization','x',tmp_path/'out',runner=lambda args:calls.append(args))
    assert not calls and not(tmp_path/'out').exists()

def test_plan_exclusive_and_explicitly_blocked(tmp_path):
    path=tmp_path/'plan.json';value=h.plan(path)
    assert value['execution_admission']is False and value['mutation_executed']is False and value['authorization_present']is False
    assert value['commands']==[h.DOWN,h.RESUME]
    with pytest.raises(FileExistsError):h.plan(path)

def test_existing_receipt_not_overwritten(tmp_path):
    path=tmp_path/'intent.json';h.save(path,{'original':True})
    with pytest.raises(FileExistsError):h.save(path,{'original':False})
    assert json.loads(path.read_text())=={'original':True}
