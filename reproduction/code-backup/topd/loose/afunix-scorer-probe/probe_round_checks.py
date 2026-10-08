"""Hardware request construction and strict native eight-scorer receipt gates."""
def request_rows(tokens,pad,step):
    if len(set(tokens))<2 or pad in tokens:raise ValueError('Need at least two ordinary nonpad token IDs')
    if step not in (1,2,3):raise ValueError('Hardware round must be1/2/3')
    width=32 if step==1 else 500 if step==2 else 4096
    prompts=[];responses=[];attention=[]
    for row in range(16):
        prefix=[tokens[(row>>(j%4))&1] for j in range(256)]
        sequence=[tokens[(row>>((j+1)%4))&1] for j in range(width)]
        active=width
        if step==3:
            if row==15:prefix=[tokens[(row>>(j%4))&1] for j in range(2048)];active=4096
            else:prefix=[pad]*1792+prefix;active=32 if row%2==0 else 500
        prompts.append(prefix);responses.append(sequence);attention.append([1]*active+[0]*(width-active))
    return prompts,responses,attention

def validate_round_receipts(receipts,manifest,step):
    services=manifest['services'];hashes=set()
    keys=('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')
    if len(receipts)!=8 or len(services)!=8:raise ValueError('Require8 actual scorer receipts')
    if len({r['pid'] for r in receipts})!=8:raise ValueError('Scorer PIDs must be distinct')
    for rank,(r,s) in enumerate(zip(receipts,services)):
        if any(r[key]!=manifest[key] for key in keys):raise ValueError('Actual receipt frozen identity differs')
        if (r['rank'],r['pid'],r['uuid'],r['cgroup'],r['outer_step'],r['rows'])!=(rank,s['pid'],s['uuid'],manifest['cgroup'],step,2):raise ValueError('Actual rank/PID/UUID/cgroup/row/round differs')
        if r['status']!='RELEASED' or r['allocator_released']!={'allocated_bytes':0,'reserved_bytes':0}:raise ValueError('Actual scorer allocator must release completely')
        if not r['targets_none'] or r['dtype']!='float32' or r['tf32'] or r['attention']!='eager':raise ValueError('Hardware scorer math/privacy differs')
        if r['gpu_pid_phase_sample']['pid']!=s['pid'] or r['gpu_pid_phase_sample']['uuid']!=s['uuid']:raise ValueError('Actual GPU PID/UUID differs from lease')
        if r['gpu_pid_released']['pid']!=s['pid'] or r['gpu_pid_released']['uuid']!=s['uuid']:raise ValueError('Released own GPU PID differs')
        initial=r['loaded_cpu'];released=r['released_cpu'];hashes.add(initial['parameter_sha256'])
        if initial['devices']!=['cpu'] or released['devices']!=['cpu']:raise ValueError('CPU boundary device differs')
        for state in (initial,released,r['gpu_before'],r['gpu_after']):
            if state is None:continue
            if state['parameter_sha256']!=initial['parameter_sha256'] or state['dtypes']!=['torch.float32']:raise ValueError('Frozen FP32 parameters differ')
            if state['training'] or any(state[k]!=0 for k in ('training_modules_count','requires_grad_parameters','grad_parameters')):raise ValueError('Scorer must stay frozen/eval')
        if any(d!='cpu' for d in released['buffer_devices']) or released['devices']!=['cpu'] or initial['devices']!=['cpu']:raise ValueError('Released model parameters/buffers notCPU')
        if step==1 and (r['gpu_before'] is None or r['gpu_before']['devices']!=['cuda:0']):raise ValueError('First GPU boundary missing')
        if step==manifest['outer_updates'] and (r['gpu_after'] is None or r['gpu_after']['devices']!=['cuda:0']):raise ValueError('Last GPU boundary missing')
    if len(hashes)!=1:raise ValueError('Teacher replica hashes differ')
    if step==3:
        # Rank7 has short row14 (256+32) plus untruncated longest row15 (2048+4096).
        if receipts[7]['scored_tokens_delta']!=256+32+2048+4096:raise ValueError('Maximum6144context was not actually fully scored')


def coexistence_evidence(samples,services,vllm_pids,mapping,retriever_pid):
    """Require actual NVML PID/UUID model residency, independent of process-alive checks."""
    if len(services)!=8 or len(set(vllm_pids))!=8:raise ValueError('Need8 distinct model owners')
    vllm={item['uuid']:pid for item,pid in zip(mapping,vllm_pids)}
    if set(vllm)!={item['uuid'] for item in services}:raise ValueError('Coexistence GPU mappings differ')
    found=[]
    for sample in samples:
        actual={}
        for line in sample['processes']:
            uid,pid,memory=[x.strip() for x in line.split(',')]
            actual[(uid,int(pid))]=float(memory)
        scorers=all(actual.get((item['uuid'],item['pid']),0)>=11000 for item in services)
        actors=all(actual.get((uid,pid),0)>=10000 for uid,pid in vllm.items())
        retriever=all((item['uuid'],retriever_pid) in actual for item in services)
        if scorers and actors and retriever:
            found.append(dict(time=sample['time'],phase=sample['phase'],scorer_memory_mib={item['uuid']:actual[(item['uuid'],item['pid'])] for item in services},vllm_memory_mib={uid:actual[(uid,pid)] for uid,pid in vllm.items()},retriever_pid=retriever_pid))
    return found


def resource_envelope(process_samples,parent_sample):
    rows={row['pid']:row for row in process_samples+[parent_sample] if not row.get('exited')}
    rss=0;hwm=0
    for row in rows.values():
        for line in row['memory']:
            key,value,unit=line.split()
            if unit!='kB':raise ValueError('Unexpected procfs memory unit')
            if key=='VmRSS:':rss+=int(value)
            elif key=='VmHWM:':hwm+=int(value)
    return dict(summed_rss_kib=rss,summed_per_pid_hwm_kib=hwm,alive_sampled_pids=sorted(rows),user_cpu_ticks=sum(row['user_cpu_ticks'] for row in rows.values()),system_cpu_ticks=sum(row['system_cpu_ticks'] for row in rows.values()),scope='summed RSS includes shared pages repeatedly; sampled not unique physical RAM or guaranteed instantaneous peak')
