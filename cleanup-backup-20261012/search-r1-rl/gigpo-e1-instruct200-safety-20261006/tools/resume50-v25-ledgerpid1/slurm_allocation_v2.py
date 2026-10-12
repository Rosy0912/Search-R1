"""Fail closed on actual scheduler allocation, never on header intent alone."""
import re


def fields(record, required=()):
    if not isinstance(record, str):
        raise ValueError('Missing actual scheduler record')
    lines = record.strip().splitlines()
    if len(lines) != 1:
        raise ValueError('Expected one actual scheduler record')
    pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=([^\s]*)', lines[0])
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Ambiguous scheduler field: ' + key)
        result[key] = value
    for key in required:
        occurrences = re.findall(r'(?:^|\s)' + re.escape(key) + r'(?=[=:\s]|$)', lines[0])
        if len(occurrences) != 1 or not result.get(key):
            raise ValueError('Missing/malformed/ambiguous scheduler field: ' + key)
    return result


def scheduler_config_fields(record):
    required = {'AccountingStorageTRES', 'AccountingStorageType', 'GresTypes',
                'SelectType', 'SelectTypeParameters'}
    if not isinstance(record, str) or not record.strip():
        raise ValueError('Missing actual scheduler configuration')
    result = {}
    for line in record.splitlines():
        match = re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*', line)
        if match and match[1] in required:
            key, value = match[1], match[2]
            if key in result or not value:
                raise ValueError('Missing/ambiguous scheduler configuration field: ' + key)
            result[key] = value
        elif any(re.match(r'\s*' + key + r'(?=[=:\s]|$)', line) for key in required):
            raise ValueError('Malformed scheduler configuration field')
    if set(result) != required:
        raise ValueError('Missing actual scheduler configuration fields')
    return result


def tres_fields(value):
    result = {}
    for item in value.split(','):
        match = re.fullmatch(r'([a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*)?)?)=([^\s,=]+)', item)
        if not match or match[1] in result:
            raise ValueError('Malformed/duplicate actual node CfgTRES')
        result[match[1]] = match[2]
    return result


def gpu_ids(value):
    if not isinstance(value, str):
        raise ValueError('Missing allocated GPU IDs')
    result = []
    for part in value.split(','):
        if re.fullmatch(r'[0-9]+', part):
            result.append(int(part))
        elif re.fullmatch(r'[0-9]+-[0-9]+', part):
            start, end = map(int, part.split('-'))
            if end < start or end - start > 7:
                raise ValueError('Reversed/oversized allocation GPU range')
            result.extend(range(start, end + 1))
        else:
            raise ValueError('Unsupported allocated GPU IDs')
    if len(result) != len(set(result)):
        raise ValueError('Duplicate allocated GPU IDs')
    return sorted(result)


def validate_exclusive_allocation(job_raw, node_raw, node_jobs_raw, environment, hostname, *, scheduler_config_raw):
    job = fields(job_raw, ('JobId', 'JobState', 'Requeue', 'Restarts', 'NodeList', 'BatchHost',
                          'NumNodes', 'OverSubscribe', 'NumCPUs', 'TresPerNode'))
    node = fields(node_raw, ('NodeName', 'CPUTot', 'CPUAlloc', 'Gres', 'CfgTRES', 'GresUsed'))
    config = scheduler_config_fields(scheduler_config_raw)
    cfg_tres = tres_fields(node['CfgTRES'])
    job_id = environment.get('SLURM_JOB_ID')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(isinstance(job_id, str) and re.fullmatch(r'[0-9]+', job_id)
            and job.get('JobId') == job_id and job.get('JobState') == 'RUNNING', 'Different/inactive actual Slurm JobId')
    require(job.get('Requeue') == '0' and job.get('Restarts') == '0', 'Requeue/restart forbidden')
    require(job.get('NodeList') == job.get('BatchHost') == node.get('NodeName') == 'h100-8'
            and hostname.split('.')[0] == 'h100-8' and environment.get('SLURM_JOB_NODELIST') == 'h100-8'
            and job.get('NumNodes') == '1', 'Actual allocation is not solely h100-8')
    require(job.get('OverSubscribe') == 'NO', 'Missing/ambiguous effective exclusive allocation flag')
    require(job['NumCPUs'] == node['CPUTot'] == node['CPUAlloc'] == cfg_tres.get('cpu') == '192',
            'Actual job does not hold the full exclusive node CPU allocation')
    require(re.fullmatch(r'gpu:h100:8(?:\(S:0-1\))?', node['Gres']),
            'Actual node full eight-H100 inventory differs')
    require(node['GresUsed'] == 'gpu:h100:8(IDX:0-7)',
            'Actual scheduler node eight-GPU allocation is missing/ambiguous')
    require(config['GresTypes'] == 'gpu' and config['SelectType'] == 'select/cons_tres'
            and config['SelectTypeParameters'] == 'CR_CORE_MEMORY', 'Different actual scheduler GPU selection configuration')
    accounting_tres = config['AccountingStorageTRES'].split(',')
    require(all(re.fullmatch(r'[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*)?)?', item)
                for item in accounting_tres)
            and len(accounting_tres) == len(set(accounting_tres)), 'Malformed/duplicate scheduler accounting TRES')
    gpu_totals = {key: value for key, value in cfg_tres.items()
                  if key.split(':')[0].split('/')[-1] == 'gpu'}
    if gpu_totals:
        require(gpu_totals.get('gres/gpu') == '8'
                and all(key in {'gres/gpu', 'gres/gpu:h100'} and value == '8'
                        for key, value in gpu_totals.items()), 'Conflicting actual node GPU CfgTRES totals')
        require('gres/gpu' in accounting_tres, 'Node GPU totals contradict actual scheduler accounting configuration')
        gpu_inventory_evidence = 'EXPLICIT_CFG_TRES_EIGHT_GPU_TOTAL'
    else:
        # This is evidence-backed absence of GPU accounting, never absence of
        # GPU allocation proof. Typed Gres/GresUsed and every job gate still hold.
        require(config['AccountingStorageType'] == 'accounting_storage/none'
                and {'cpu', 'mem', 'billing'} <= set(accounting_tres)
                and not any(item == 'gpu' or item.startswith('gpu:') or item == 'gres/gpu'
                            or item.startswith('gres/gpu:') for item in accounting_tres),
                'Missing GPU CfgTRES is not corroborated by actual non-GPU accounting configuration')
        gpu_inventory_evidence = 'CONFIG_CORROBORATED_UNACCOUNTED_TYPED_EIGHT_H100'
    require(job.get('TresPerNode') == 'gres:gpu:h100:8', 'Actual job GPU allocation request differs')
    require(gpu_ids(environment.get('SLURM_JOB_GPUS')) == list(range(8)), 'Actual JobId does not hold all eight GPU IDs')
    visible = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(visible) == len(set(visible)) == 8 and all(visible), 'Eight actual visible GPU leases required')
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
        raw_scheduler_config=scheduler_config_raw, gpu_inventory_evidence=gpu_inventory_evidence,
        parsed_scheduler_config=config, parsed_node_cfg_tres=cfg_tres,
        header_is_not_allocation_evidence=True, production_thresholds_selected=False)
