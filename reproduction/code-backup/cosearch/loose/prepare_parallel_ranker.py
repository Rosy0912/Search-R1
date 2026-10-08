from pathlib import Path
p=Path('work/cosearch/search_r1/baselines/cosearch.py');s=p.read_text()
s=s.replace('    events = []\n    for row in range(input_ids.shape[0]):', '''    scores = torch.stack([F.log_softmax(logits.float() / temperature, -1).gather(
        -1, response[:, t, None]).squeeze(-1) for t, logits in enumerate(output.logits)], dim=1)
    events = []
    for row in range(input_ids.shape[0]):''',1)
s=s.replace('        logp = torch.stack([F.log_softmax(output.logits[t][row].float() / temperature, -1)[tokens[t]]\n                            for t in range(length)])','        logp = scores[row, :length]')
pos=s.index('\nclass GenerativeRanker:')
s=s[:pos]+'''
class RankerSequenceLoss(torch.nn.Module):
    """One scalar per sampled sequence; never gather vocabulary logits."""
    def __init__(self, model, temperature, clip_ratio):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.clip_ratio = clip_ratio

    def forward(self, input_ids, attention_mask, response_lengths, old_logp, advantages, max_response_length):
        positions = attention_mask.cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        logits = self.model(input_ids, attention_mask=attention_mask, position_ids=positions,
                            use_cache=False, num_logits_to_keep=max_response_length + 1).logits[:, :-1]
        logp = F.log_softmax(logits.float() / self.temperature, -1).gather(
            -1, input_ids[:, -max_response_length:, None]).squeeze(-1)
        if not torch.isfinite(logp).all() or not torch.isfinite(old_logp).all():
            raise ValueError('Nonfinite parallel ranker scores')
        ratio = (logp - old_logp).exp()
        if not torch.isfinite(ratio).all():
            raise ValueError('Nonfinite parallel ranker ratios')
        mask = torch.arange(max_response_length, device=logp.device)[None] >= (
            max_response_length - response_lengths[:, None])
        objective = torch.minimum(ratio * advantages[:, None],
            ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * advantages[:, None])
        return -(objective * mask).sum(-1) / response_lengths

''' + s[pos:]
# Preserve original one-GPU mode; 8-GPU mode is a separate explicit opt-in.
s=s.replace("        if not torch.cuda.is_initialized():", "        count = int(getattr(self.config, 'phase_gpu_devices', 1))\n        expected_mask = ','.join(leased) if count == 8 else uid\n        if count not in (1, 8):\n            raise ValueError('Ranker phase GPU count must be one or all eight owned devices')\n        if not torch.cuda.is_initialized():")
s=s.replace("            os.environ['CUDA_VISIBLE_DEVICES'] = uid", "            os.environ['CUDA_VISIBLE_DEVICES'] = expected_mask")
s=s.replace("        if os.environ.get('CUDA_VISIBLE_DEVICES') != uid or torch.cuda.device_count() != 1:", "        if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_mask or torch.cuda.device_count() != count:")
s=s.replace("        self.device = torch.device('cuda:0')", "        master = leased.index(uid) if count == 8 else 0\n        self.phase_devices = [master] + [i for i in range(count) if i != master]\n        self.device = torch.device('cuda', master)\n        torch.cuda.set_device(master)")
s=s.replace("            if actual != {uid}:", "            if uid not in actual or not actual.issubset(set(leased)):")
s=s.replace('            torch.cuda.reset_peak_memory_stats()', '            for i in self.phase_devices:\n                torch.cuda.reset_peak_memory_stats(i)')
s=s.replace("                torch.cuda.synchronize()\n                self.cost['gpu_peak_allocated_bytes']", "                for i in self.phase_devices:\n                    torch.cuda.synchronize(i)\n                self.cost['gpu_peak_by_device'] = {str(i):dict(allocated=torch.cuda.max_memory_allocated(i),\n                                                    reserved=torch.cuda.max_memory_reserved(i)) for i in self.phase_devices}\n                self.cost['gpu_peak_allocated_bytes']")
s=s.replace('                torch.cuda.empty_cache()', '                for i in self.phase_devices:\n                    with torch.cuda.device(i):\n                        torch.cuda.empty_cache()')
start=s.index('        offset = 0\n',s.index('    def _rank_batch'))
end=s.index('        result = []\n',start)
s=s[:start]+'''        groups=[]
        offset=0
        while offset < len(order):
            group=[]
            for i in order[offset:offset+batch_size]:
                if group and (len(group)+1)*(len(encoded[i])+self.config.max_new_tokens)>budget:
                    break
                group.append(i)
            groups.append(group);offset+=len(group)
        devices = getattr(self, 'phase_devices', [self.device.index or 0]) if getattr(self.config, 'inference_parallel', False) else [self.device]
        replicas = None
        gpu_models = [self.model]
        generation_started=time.perf_counter()
        try:
            if len(devices)>1:
                if self.model.config.model_type != 'qwen2' or self.model.config.rope_scaling:
                    raise ValueError('Parallel ranker inference requires pinned static-RoPE Qwen2')
                replica_at=time.perf_counter()
                replicas=torch.nn.parallel.replicate(self.model, devices, detach=True)
                gpu_models=replicas
                for i,model in zip(devices,gpu_models):
                    if any(p.device.index != i for p in model._former_parameters.values()):
                        raise RuntimeError('Ranker replica is on the wrong GPU')
                self.cost['replication_wall_seconds']=self.cost.get('replication_wall_seconds',0.)+time.perf_counter()-replica_at
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                completed=0
                for wave in range(0,len(groups),len(devices)):
                    futures=[]
                    for j,group in enumerate(groups[wave:wave+len(devices)]):
                        inputs=self.tokenizer.pad({'input_ids':[encoded[i] for i in group]},padding=True,return_tensors='pt')
                        sampling=GenerationConfig(max_new_tokens=self.config.max_new_tokens,do_sample=not self.validation,
                            temperature=self.config.temperature,top_p=1.,top_k=0,num_beams=1,
                            pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.model.generation_config.eos_token_id,
                            bos_token_id=self.model.generation_config.bos_token_id,return_dict_in_generate=True,
                            output_logits=not self.validation)
                        futures.append((group,executor.submit(self._generate_batch, gpu_models[j], devices[j], inputs, sampling)))
                    for group,future in futures:
                        events,wall=future.result()
                        self.cost['generation_gpu_worker_seconds']=self.cost.get('generation_gpu_worker_seconds',0.)+wall
                        for i,event in zip(group,events):rows[i]=event
                        completed+=len(group)
                    if wave==0 or completed%100<sum(len(g) for g,_ in futures) or completed==len(order):
                        print('COSEARCH_BATCH_PROGRESS_JSON='+json.dumps(dict(validation=self.validation,
                              completed=completed,total=len(order),wall_seconds=time.perf_counter()-started)),flush=True)
                    futures.clear()
        finally:
            gpu_models.clear()
            replicas=None
            self.cost['generation_wall_seconds']=self.cost.get('generation_wall_seconds',0.)+time.perf_counter()-generation_started
''' + s[end:]
pos=s.index('    def execute(')
s=s[:pos]+'''    def _generate_batch(self, model, device, inputs, sampling):
        started=time.perf_counter()
        device=torch.device('cuda',device) if isinstance(device,int) else device
        from contextlib import nullcontext
        # grad mode and active CUDA device are thread-local.
        with torch.cuda.device(device) if device.type=='cuda' else nullcontext():
            inputs=inputs.to(device)
            with torch.no_grad():
                output=model.generate(**inputs,generation_config=sampling)
                if not self.validation:
                    events=unpack_generation(output,inputs.input_ids,inputs.attention_mask,
                                             sampling.eos_token_id,self.config.temperature)
                else:
                    response=output.sequences[:,inputs.input_ids.shape[1]:]
                    eos=sampling.eos_token_id
                    eos_ids=eos if isinstance(eos,(list,tuple)) else [eos]
                    events=[]
                    for row in range(response.shape[0]):
                        ends=[t for t,token in enumerate(response[row].tolist()) if token in eos_ids]
                        length=ends[0]+1 if ends else response.shape[1]
                        prompt=inputs.input_ids[row][inputs.attention_mask[row].bool()]
                        events.append(dict(input_ids=torch.cat((prompt,response[row,:length])).cpu(),response_length=length))
            del output,inputs
        return events,time.perf_counter()-started

''' + s[pos:]
start=s.index('                for i in rows:\n',s.index('    def _update'))
end=s.index('                norm = ',start)
legacy=s[start:end]
s=s[:start]+'''                if getattr(self.config, 'training_parallel', False):
                    width=max(events[i]['input_ids'].numel() for i in rows)
                    response_width=max(events[i]['response_length'] for i in rows)
                    ids=torch.stack([F.pad(events[i]['input_ids'],(width-events[i]['input_ids'].numel(),0),
                                           value=self.tokenizer.pad_token_id) for i in rows]).to(self.device)
                    mask=torch.stack([F.pad(torch.ones_like(events[i]['input_ids']),
                                            (width-events[i]['input_ids'].numel(),0)) for i in rows]).to(self.device)
                    old=torch.stack([F.pad(events[i]['old_logp'],(response_width-events[i]['response_length'],0)) for i in rows]).to(self.device)
                    lengths=torch.tensor([events[i]['response_length'] for i in rows],device=self.device)
                    row_advantages=advantage[rows].to(self.device)
                    sequence_loss=RankerSequenceLoss(self.model,self.config.temperature,self.config.clip_ratio)
                    parallel=torch.nn.DataParallel(sequence_loss,device_ids=self.phase_devices,output_device=self.device.index)
                    losses=parallel(ids,mask,lengths,old,row_advantages,max_response_length=response_width)
                    (losses.sum()/len(rows)).backward()
                    del losses,parallel,sequence_loss,ids,mask,old,lengths,row_advantages
                    self.cost['training_forward_tokens']+=sum(events[i]['input_ids'].numel() for i in rows)
                    self.cost['training_padding_tokens']=self.cost.get('training_padding_tokens',0)+width*len(rows)-sum(events[i]['input_ids'].numel() for i in rows)
                else:
''' + ''.join('    '+line+'\n' for line in legacy.splitlines()) + s[end:]
p.write_text(s)
p=Path('work/cosearch/verl/trainer/config/ppo_baselines.yaml');s=p.read_text().replace('    phase_gpu: false','    phase_gpu: false\n    phase_gpu_devices: 1\n    inference_parallel: false\n    training_parallel: false');p.write_text(s)
p=Path('work/cosearch/search_r1/baselines/main.py');s=p.read_text().replace("audit_env.update(CUDA_VISIBLE_DEVICES=os.environ['COSEARCH_RANKER_GPU_UUID'],", "audit_env.update(CUDA_VISIBLE_DEVICES=(','.join(__import__('json').loads(os.environ['COSEARCH_RANKER_LEASE_UUIDS']))\n                         if config.baselines.ranker.phase_gpu_devices==8 else os.environ['COSEARCH_RANKER_GPU_UUID']),");p.write_text(s)
p=Path('work/cosearch/search_r1/baselines/controller.py');s=p.read_text().replace("        if ranker.keep != config.retriever.topk:", "        if (ranker.phase_gpu_devices not in (1,8)\n                or (ranker.inference_parallel or ranker.training_parallel) and (not ranker.phase_gpu or ranker.phase_gpu_devices!=8)):\n            raise ValueError('Parallel ranker requires all eight owned phase GPUs')\n        if ranker.keep != config.retriever.topk:");p.write_text(s)
