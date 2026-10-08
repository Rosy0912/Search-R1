import datetime, hashlib, json, os, re, subprocess
BASE='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fresh200-train-3328'
FIXED={'Tree raw-token protocol: Native raw token IDs/PAD unsupported','Identity-only wrapper is restricted to this direct V4 package','Own GPU retrieval startup failed','Own GPU guard failed','Own Ray failed','Own dependent service died','Actual Tree training failed','Guard died before training completion','Final guard missing/stale/STOP','Original8 must be preserved, no unexpected GPU remainder','Fresh200 rolling completed state requires native training state','Rolling native worker snapshot coverage is incomplete'}
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3328,'terminal_files':[],'safe_errors':[],'log_metadata':[]}
for rel in ['run-result.json','cost.json','termination-cost.json']:
    path=os.path.join(BASE,rel)
    if not os.path.isfile(path):continue
    with open(path,'rb') as stream:content=stream.read()
    value=json.loads(content)
    result['terminal_files'].append({'name':rel,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'mtime':os.path.getmtime(path)})
    if rel=='run-result.json':
        result['native_wall_seconds']=value.get('wall_seconds')
        result['cleanup_residual_empty']=value.get('cleanup_residual')=={}
        result['original_eight_pid_start_times_preserved']=value.get('original_eight_pid_start_times_preserved')
        result['optimizer_committed']=value.get('optimizer_committed')
        result['scheduler_advanced']=value.get('scheduler_advanced')
        result['failed_unrecorded_partial_optimizer_writes_unknown']=value.get('failed_unrecorded_partial_optimizer_writes_unknown')
        for key in ['error','cleanup_error','termination_record_error','actual_progress_evidence_error']:
            error=value.get(key)
            if not isinstance(error,str):continue
            record={'origin':rel,'field':key,'error_type':value.get('error_type'),'fixed_error_message':error if error in FIXED else None,'unmatched_description_omitted':error not in FIXED}
            result['safe_errors'].append(record)
for path in [os.path.join(BASE,n) for n in ['stage.log','retriever.log','guard.log','ray.log']]+['/root/slurm-3328.out']:
    if not os.path.isfile(path):continue
    record={'name':os.path.relpath(path,BASE) if path.startswith(BASE+'/') else path,'bytes':os.path.getsize(path),'mtime':os.path.getmtime(path),'source_files_lines':[],'exception_classes':[],'fixed_fault_markers':[],'exact_fixed_exception_messages':[]}
    with open(path,errors='replace') as stream:
        for line in stream:
            line=re.sub(r'\x1b\[[0-9;]*m','',line)
            match=re.search(r'File "(/[^"\n]+)", line (\d+)',line)
            if match:
                entry={'file':match.group(1),'line':int(match.group(2))}
                if entry not in record['source_files_lines']:record['source_files_lines'].append(entry)
            for cls in re.findall(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Refusal|Stop|Panic))\b',line):
                if cls not in record['exception_classes']:record['exception_classes'].append(cls)
            for message in FIXED:
                if message in line and message not in record['fixed_fault_markers']:record['fixed_fault_markers'].append(message)
            exact = re.search(r'\b(ValueError|RuntimeError):\s*(.*)',line)
            if exact and exact.group(2).strip() in FIXED:
                item={'exception_class':exact.group(1),'fixed_error_message':exact.group(2).strip()}
                if item not in record['exact_fixed_exception_messages']:record['exact_fixed_exception_messages'].append(item)
            checks={'faiss_allocation_error':r'cudaMalloc|failed to allocate|alloc error|CUDA error.*out of memory','abort_or_terminate':r'terminate called|Aborted|SIGABRT|Fatal Python error','process_killed':r'\bKilled\b|SIGKILL|oom-kill','invalid_utf8':r'invalid utf-?8|invalid start byte|UnicodeDecodeError','bind_address_in_use':r'Address already in use','native_faiss_gpu_search_error':r'GpuIndex|faiss::|faiss\.swigfaiss'}
            for marker,pattern in checks.items():
                if re.search(pattern,line,re.I) and marker not in record['fixed_fault_markers']:record['fixed_fault_markers'].append(marker)
    result['log_metadata'].append(record)
control=subprocess.run(['scontrol','show','job','3328','-o'],capture_output=True,text=True,timeout=20)
queue=subprocess.run(['squeue','-h','-j','3328','-o','%i|%T|%R'],capture_output=True,text=True,timeout=20)
result['positive_slurm_raw_stdout']=control.stdout if control.returncode==0 and 'JobId=3328 ' in control.stdout else None
result['slurm_returncode']=control.returncode
result['own_queue_raw_stdout']=queue.stdout
result['own_queue_returncode']=queue.returncode
print(json.dumps(result,sort_keys=True))
