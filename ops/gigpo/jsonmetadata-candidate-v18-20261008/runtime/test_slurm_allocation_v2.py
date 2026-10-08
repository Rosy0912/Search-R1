"""UNEXECUTED allocation fixtures; no scheduler command or allocation created."""
import ast
from pathlib import Path

import pytest
from slurm_allocation_v2 import validate_exclusive_allocation

JOB = ('JobId=999 JobState=RUNNING Requeue=0 Restarts=0 NodeList=h100-8 BatchHost=h100-8 '
       'NumNodes=1 OverSubscribe=NO NumCPUs=192 TresPerNode=gres:gpu:h100:8')
NODE = ('NodeName=h100-8 CPUTot=192 CPUAlloc=192 Gres=gpu:h100:8(S:0-1) '
        'CfgTRES=cpu=192,gres/gpu=8 GresUsed=gpu:h100:8(IDX:0-7)')
# Prospective running-allocation fixture uses the observed node/config schema.
# The preserved actual post-failure idle node is separately refused below.
NODE_NO_GPU_ACCOUNTING = ('NodeName=h100-8 CPUTot=192 CPUAlloc=192 Gres=gpu:h100:8 '
                          'Version=21.08.5 CfgTRES=cpu=192,mem=2000000M,billing=192 '
                          'GresUsed=gpu:h100:8(IDX:0-7)')
CONFIG_NO_GPU_ACCOUNTING = ('AccountingStorageTRES   = cpu,mem,energy,node,billing,fs/disk,vmem,pages\n'
                            'AccountingStorageType   = accounting_storage/none\n'
                            'GresTypes               = gpu\n'
                            'SelectType              = select/cons_tres\n'
                            'SelectTypeParameters    = CR_CORE_MEMORY\n')
CONFIG_GPU_ACCOUNTING = CONFIG_NO_GPU_ACCOUNTING.replace(
    'pages\n', 'pages,gres/gpu\n').replace('accounting_storage/none', 'accounting_storage/slurmdbd')
ENV = dict(SLURM_JOB_ID='999', SLURM_JOB_NODELIST='h100-8', SLURM_JOB_GPUS='0-7',
           CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')


def validate(job=JOB, node=NODE_NO_GPU_ACCOUNTING, allocations='999|h100-8\n',
             config=CONFIG_NO_GPU_ACCOUNTING, environment=ENV):
    return validate_exclusive_allocation(job, node, allocations, environment, 'h100-8',
                                         scheduler_config_raw=config)


def test_actual_effective_scheduler_allocation():
    result = validate(node=NODE, config=CONFIG_GPU_ACCOUNTING)
    assert result['job_id'] == '999' and result['header_is_not_allocation_evidence']
    assert result['gpu_inventory_evidence'] == 'EXPLICIT_CFG_TRES_EIGHT_GPU_TOTAL'


def test_observed_no_gpu_accounting_requires_actual_detailed_allocation():
    result = validate()
    assert result['gpu_ids'] == list(range(8))
    assert result['gpu_inventory_evidence'] == 'CONFIG_CORROBORATED_UNACCOUNTED_TYPED_EIGHT_H100'
    assert result['raw_job'] == JOB and result['raw_node'] == NODE_NO_GPU_ACCOUNTING
    assert result['raw_scheduler_config'] == CONFIG_NO_GPU_ACCOUNTING
    assert result['raw_node_allocations'] == '999|h100-8\n'
    assert result['production_thresholds_selected'] is False


def test_unrelated_non_gpu_accounting_entries_are_not_pinned():
    config = CONFIG_NO_GPU_ACCOUNTING.replace('energy,node,', '').replace(',fs/disk,vmem,pages', '')
    assert validate(config=config)['gpu_ids'] == list(range(8))


@pytest.mark.parametrize('job,node,allocations', [
    (JOB.replace('OverSubscribe=NO', 'OverSubscribe=YES'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('OverSubscribe=NO', ''), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('JobId=999', 'JobId=998'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('JobState=RUNNING', 'JobState=PENDING'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('NumCPUs=192', 'NumCPUs=128'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('NumNodes=1', 'NumNodes=2'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('Requeue=0', 'Requeue=1'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('Restarts=0', 'Restarts=1'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB.replace('gres:gpu:h100:8', 'gres:gpu:8'), NODE_NO_GPU_ACCOUNTING, '999|h100-8'),
    (JOB, NODE_NO_GPU_ACCOUNTING.replace('IDX:0-7', 'IDX:0-3'), '999|h100-8'),
    (JOB, NODE_NO_GPU_ACCOUNTING, '999|h100-8\n998|h100-8'),
    (JOB, NODE_NO_GPU_ACCOUNTING, '999|h100-8\n999|h100-8'),
    (JOB, NODE_NO_GPU_ACCOUNTING, ''),
    (JOB, NODE_NO_GPU_ACCOUNTING, '999|h100-8|extra'),
    (JOB, NODE_NO_GPU_ACCOUNTING.replace('NodeName=h100-8', 'NodeName=h100-7'), '999|h100-8'),
    (JOB, NODE_NO_GPU_ACCOUNTING.replace('CPUAlloc=192', 'CPUAlloc=0').replace(
        'gpu:h100:8(IDX:0-7)', 'gpu:h100:0(IDX:N/A)'), '999|h100-8'),
])
def test_missing_ambiguous_or_shared_allocation_refused(job, node, allocations):
    with pytest.raises(ValueError):
        validate(job=job, node=node, allocations=allocations)


@pytest.mark.parametrize('node', [
    NODE_NO_GPU_ACCOUNTING.replace(' GresUsed=gpu:h100:8(IDX:0-7)', ''),
    NODE_NO_GPU_ACCOUNTING.replace('Gres=gpu:h100:8', 'Gres=gpu:8'),
    NODE_NO_GPU_ACCOUNTING.replace('Gres=gpu:h100:8', 'Gres=gpu:h100:80'),
    NODE_NO_GPU_ACCOUNTING.replace('Gres=gpu:h100:8', 'Gres=gpu:h100:4,gpu:a100:4'),
    NODE_NO_GPU_ACCOUNTING.replace('gpu:h100:8(IDX:0-7)', 'gpu:h100:8(IDX:0-7),gpu:a100:1(IDX:8)'),
    NODE_NO_GPU_ACCOUNTING + ' GresUsed=gpu:h100:8(IDX:0-7)',
    NODE_NO_GPU_ACCOUNTING + ' Gres=gpu:h100:8',
    NODE_NO_GPU_ACCOUNTING + ' CfgTRES=cpu=192',
    NODE_NO_GPU_ACCOUNTING.replace('GresUsed=', 'GresUsed ='),
    NODE_NO_GPU_ACCOUNTING.replace('CPUAlloc=192', 'CPUAlloc=+192'),
    NODE_NO_GPU_ACCOUNTING.replace('cpu=192,', 'cpu=128,'),
    NODE_NO_GPU_ACCOUNTING.replace('cpu=192,', 'cpu=192,cpu=192,'),
    NODE_NO_GPU_ACCOUNTING.replace('billing=192', 'billing=192,'),
    NODE_NO_GPU_ACCOUNTING + '\n' + NODE_NO_GPU_ACCOUNTING,
])
def test_missing_duplicate_or_malformed_node_fields_refused(node):
    with pytest.raises(ValueError):
        validate(node=node)


@pytest.mark.parametrize('gpu_tres', [
    'gres/gpu=4', 'gres/gpu=9', 'gres/gpu=8,gres/gpu=8',
    'gres/gpu=8,gres/gpu:h100=4', 'gres/gpu:h100=8',
    'gres/gpu=8,gres/gpu:a100=8', 'gres/gpu=8.0', 'gres/gpu=', 'GRES/GPU=8',
    'gpu=8', 'gres//gpu=8', 'gres/gpu::h100=8',
])
def test_explicit_conflicting_or_malformed_gpu_totals_refused(gpu_tres):
    with pytest.raises(ValueError):
        validate(node=NODE_NO_GPU_ACCOUNTING.replace('billing=192', 'billing=192,' + gpu_tres),
                 config=CONFIG_GPU_ACCOUNTING)


def test_consistent_explicit_generic_and_typed_gpu_totals():
    result = validate(node=NODE.replace('gres/gpu=8', 'gres/gpu=8,gres/gpu:h100=8'),
                      config=CONFIG_GPU_ACCOUNTING)
    assert result['gpu_inventory_evidence'] == 'EXPLICIT_CFG_TRES_EIGHT_GPU_TOTAL'


@pytest.mark.parametrize('config', [
    '',
    CONFIG_NO_GPU_ACCOUNTING.replace('AccountingStorageType   = accounting_storage/none\n', ''),
    CONFIG_NO_GPU_ACCOUNTING.replace('accounting_storage/none', 'accounting_storage/slurmdbd'),
    CONFIG_NO_GPU_ACCOUNTING.replace('pages\n', 'pages,gres/gpu\n'),
    CONFIG_NO_GPU_ACCOUNTING.replace('pages\n', 'pages,gres/gpu:h100\n'),
    CONFIG_NO_GPU_ACCOUNTING.replace('pages\n', 'pages,gpu\n'),
    CONFIG_NO_GPU_ACCOUNTING.replace('cpu,mem,', 'cpu,cpu,mem,'),
    CONFIG_NO_GPU_ACCOUNTING.replace('cpu,mem,', 'cpu,'),
    CONFIG_NO_GPU_ACCOUNTING.replace('pages\n', 'pages,\n'),
    CONFIG_NO_GPU_ACCOUNTING.replace('SelectType              = select/cons_tres', 'SelectType              = select/linear'),
    CONFIG_NO_GPU_ACCOUNTING.replace('CR_CORE_MEMORY', 'CR_CORE'),
    CONFIG_NO_GPU_ACCOUNTING.replace('GresTypes               = gpu', 'GresTypes               = (null)'),
    CONFIG_NO_GPU_ACCOUNTING + 'GresTypes = gpu\n',
    CONFIG_NO_GPU_ACCOUNTING.replace('SelectType              =', 'SelectType              :'),
])
def test_missing_conflicting_duplicate_or_malformed_configuration_refused(config):
    with pytest.raises(ValueError):
        validate(config=config)


def test_explicit_node_gpu_totals_cannot_contradict_no_gpu_accounting():
    with pytest.raises(ValueError):
        validate(node=NODE)


@pytest.mark.parametrize('job', [
    JOB + ' JobId=999', JOB + ' NumCPUs=192',
    JOB.replace('NumCPUs=192', 'NumCPUs =192'),
    JOB.replace('NumCPUs=192', 'NumCPUs=192.0'),
    JOB.replace('TresPerNode=gres:gpu:h100:8', ''),
])
def test_duplicate_missing_or_malformed_job_fields_refused(job):
    with pytest.raises(ValueError):
        validate(job=job)


@pytest.mark.parametrize('key,value', [
    ('SLURM_JOB_GPUS', ''), ('SLURM_JOB_GPUS', '0-6'),
    ('SLURM_JOB_GPUS', '0-7,7'), ('SLURM_JOB_GPUS', '7-0'),
    ('SLURM_JOB_GPUS', '0-999999999'), ('SLURM_JOB_GPUS', '0-7x'),
    ('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,6'),
    ('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,'),
    ('SLURM_JOB_ID', ''), ('SLURM_JOB_NODELIST', 'h100-7'),
])
def test_missing_duplicate_or_malformed_environment_refused(key, value):
    with pytest.raises(ValueError):
        validate(environment=dict(ENV, **{key: value}))


def test_collector_requests_detail_and_binds_raw_scheduler_config():
    tree = ast.parse((Path(__file__).parent / 'calibration_run_v1.py').read_text())
    command_lists = [ast.literal_eval(node) for node in ast.walk(tree)
                     if isinstance(node, ast.List) and node.elts
                     and all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                             for item in node.elts)]
    assert ['scontrol', 'show', 'node', 'h100-8', '-o', '-d'] in command_lists
    assert ['scontrol', 'show', 'config'] in command_lists
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'validate_exclusive_allocation']
    assert len(calls) == 1
    assert any(keyword.arg == 'scheduler_config_raw' and isinstance(keyword.value, ast.Name)
               and keyword.value.id == 'scheduler_config' for keyword in calls[0].keywords)
