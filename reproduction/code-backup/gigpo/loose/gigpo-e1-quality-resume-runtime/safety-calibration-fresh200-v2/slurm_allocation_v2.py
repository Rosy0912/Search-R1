"""Fail closed on actual scheduler allocation, never on header intent alone."""
import re


def fields(record):
    lines = record.strip().splitlines()
    if len(lines) != 1:
        raise ValueError('Expected one actual scheduler record')
    pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=([^\s]*)', lines[0])
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Ambiguous scheduler field: ' + key)
        result[key] = value
    return result


def gpu_ids(value):
    result = []
    for part in value.split(','):
        if re.fullmatch(r'\d+', part):
            result.append(int(part))
        elif re.fullmatch(r'\d+-\d+', part):
            start, end = map(int, part.split('-'))
            if end < start:
                raise ValueError('Reversed allocation GPU range')
            result.extend(range(start, end + 1))
        else:
            raise ValueError('Unsupported allocated GPU IDs')
    if len(result) != len(set(result)):
        raise ValueError('Duplicate allocated GPU IDs')
    return sorted(result)


def validate_exclusive_allocation(job_raw, node_raw, node_jobs_raw, environment, hostname):
    job, node = fields(job_raw), fields(node_raw)
    job_id = environment['SLURM_JOB_ID']
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(job.get('JobId') == job_id and job.get('JobState') == 'RUNNING', 'Different/inactive actual Slurm JobId')
    require(job.get('Requeue') == '0' and job.get('Restarts') == '0', 'Requeue/restart forbidden')
    require(job.get('NodeList') == job.get('BatchHost') == node.get('NodeName') == 'h100-8'
            and hostname.split('.')[0] == 'h100-8' and environment['SLURM_JOB_NODELIST'] == 'h100-8'
            and job.get('NumNodes') == '1', 'Actual allocation is not solely h100-8')
    require(job.get('OverSubscribe') == 'NO', 'Missing/ambiguous effective exclusive allocation flag')
    require(int(job['NumCPUs']) == int(node['CPUTot']) == int(node['CPUAlloc'])
            and int(job['NumCPUs']) >= 128, 'Actual job does not hold the full exclusive node CPU allocation')
    require('gpu:h100:8' in node['Gres'].split('(')[0]
            and re.search(r'(?:^|,)gres/gpu=8(?:,|$)', node['CfgTRES']), 'Actual node full eight-H100 inventory differs')
    require(re.search(r'(?:^|,)gpu:h100:8\(IDX:0-7\)(?:,|$)', node['GresUsed']),
            'Actual scheduler node eight-GPU allocation is missing/ambiguous')
    require(job.get('TresPerNode') == 'gres:gpu:h100:8', 'Actual job GPU allocation request differs')
    require(gpu_ids(environment['SLURM_JOB_GPUS']) == list(range(8)), 'Actual JobId does not hold all eight GPU IDs')
    require(len(environment['CUDA_VISIBLE_DEVICES'].split(',')) == 8, 'Eight actual visible GPU leases required')
    allocations = []
    for line in node_jobs_raw.strip().splitlines():
        values = line.strip().split('|')
        require(len(values) == 2 and values[1] == 'h100-8' and values[0].isdecimal(),
                'Ambiguous scheduler node allocation inventory')
        allocations.append(values[0])
    require(allocations == [job_id], 'Another running/suspended/completing scheduler allocation occupies h100-8')
    return dict(status='ACTUAL_SCHEDULER_EXCLUSIVE_FULL_H100_8_ALLOCATION', job_id=job_id,
        node='h100-8', effective_oversubscribe=job['OverSubscribe'], gpu_ids=list(range(8)),
        raw_job=job_raw, raw_node=node_raw, raw_node_allocations=node_jobs_raw,
        header_is_not_allocation_evidence=True, production_thresholds_selected=False)
