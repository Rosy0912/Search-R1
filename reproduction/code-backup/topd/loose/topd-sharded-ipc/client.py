"""CPU driver shared-session client; no automatic retry after uncertain ACK."""
import os
from pathlib import Path
import secrets
import socket

import torch
from safetensors.torch import load,save

from protocol import IDENTITY,PROTOCOL,canonical,digest,in_parent_scope,peer,private_directory,read_private_json,receive,require,send
from session import BINDING,validate_manifest


class SharedTeacherClient:
    def __init__(self,manifest,directory):
        validate_manifest(manifest);self.manifest=manifest;self.directory=private_directory(directory)
        self.manifest_sha=digest(canonical(manifest));self.scored_tokens=0;self.parameter_count=None;self.last_receipts=[];self.failed=False;self.next_step=1
        path=Path(manifest['service']['socket'])
        require(path.parent.resolve()==self.directory and len(os.fsencode(str(path)))<108,'Own Unix socket directory/length differs')
        require(manifest['timeout_seconds']>0 and manifest['timeout_seconds']<=3600,'Bounded prototype deadline required')

    def base(self,command,nonce,step):
        return dict({key:self.manifest[key] for key in IDENTITY},protocol=PROTOCOL,command=command,capability=self.manifest['capability'],manifest_sha256=self.manifest_sha,nonce=nonce,outer_step=step)

    def validate_result(self,metadata,payload,tensors,nonce,step):
        require(set(metadata)==set(IDENTITY)|{'protocol','status','nonce','outer_step','receipt','output_sha256'},'Result fields differ')
        require(all(metadata[key]==self.manifest[key] for key in IDENTITY) and metadata['protocol']==PROTOCOL
                and metadata['status']=='RESULT_RELEASED' and metadata['nonce']==nonce and metadata['outer_step']==step,'Result identity/release differs')
        require(digest(payload)==metadata['output_sha256'],'Actual output frame SHA differs')
        receipt=read_private_json(self.directory,metadata['receipt']);sidecar=read_private_json(self.directory,receipt['sidecar'])
        require(set(sidecar)=={'scope','inventory','ledger'} and sidecar['scope']=='CPU_FIXTURE_PARTITION_INVENTORY_NOT_GPU_PROOF','Missing original partition/ledger evidence')
        require(all(receipt[key]==self.manifest[key] for key in IDENTITY) and receipt['protocol']==PROTOCOL
                and receipt['status']=='RELEASED_ALL8_ZERO' and receipt['kind']==self.manifest['kind']
                and receipt['nonce']==nonce and receipt['outer_step']==step and receipt['rows']==len(tensors['responses'])
                and receipt['input_sha256']==digest(save(tensors)) and receipt['output_sha256']==digest(payload),'Receipt request/result identity differs')
        require(receipt.get('frozen_source_binding')==BINDING,'Receipt original source binding differs')
        require(receipt['service_identity']==self.manifest['service'] and receipt['pid']==self.manifest['service']['pid']
                and receipt['leased_uuids']==self.manifest['leased_uuids'],'One-service actual owner/UUID receipt differs')
        for actual,expected in [(receipt['actual_driver_peer'],self.manifest['driver']),(receipt['actual_service_identity'],self.manifest['service'])]:
            require(actual and actual['pid']==expected['pid'] and actual['starttime']==expected['starttime']
                    and in_parent_scope(actual['cgroups'],self.manifest['parent_cgroup']),'Raw actual peer identity missing/drift')
        require(receipt['row_indices_sha256']==digest(save(dict(row_indices=tensors['row_indices'])))
                and receipt['attention_sha256']==digest(save(dict(attention=tensors['attention']))),'Receipt row/mask SHA differs')
        raw=receipt['original_scorer_receipt'];require(raw['status']=='RELEASED' and raw['runtime_kind']=='CPU_ONLY_FIXTURE_NOT_HARDWARE_EVIDENCE'
                and 'error' not in raw and 'release_error' not in raw,'Original scorer release failed')
        require(raw['original_method_sha256']==BINDING['original_method_sha256'] and raw['original_file_sha256']==self.manifest['original_wrapper_file_sha256']==receipt['original_wrapper_file_sha256'],'Actual original method/file SHA differs')
        if hasattr(self,'last_actual_server_peer'):
            observed=self.last_actual_server_peer;recorded=receipt['actual_service_identity']
            require(all(recorded[k]==observed[k] for k in ('pid','starttime','cgroups')),'Socket peer differs from raw actual service')
        require(raw['input_sha256']==receipt['input_sha256'] and raw['output_sha256']==receipt['output_sha256'],'Raw original scorer hash differs')
        allocation=raw['allocators_released'];require(len(allocation)==8 and {a['device'] for a in allocation}==set(range(8))
                and all(a['allocated_bytes']==a['reserved_bytes']==0 for a in allocation),'One of eight release allocators invalid')
        before,after=raw['initial_cpu'],raw['released_cpu']
        require(after['parameter_devices']==['cpu'] and all(device=='cpu' for device in after['buffer_devices'])
                and all(before[key]==after[key] for key in ['parameter_sha256','buffer_sha256','parameter_count','parameter_bytes'])
                and after['training_modules']==after['requires_grad_parameters']==after['grad_parameters']==0,'Actual frozen CPU release snapshot differs')
        require(receipt['temperature']==1. and receipt['chunk']==16 and receipt['teacher_targets_none'] is True
                and receipt['batch_size']==self.manifest['scoring_batch_rows'],'Math/noGold settings differ')
        output=load(payload);require(set(output)=={'teacher_log_probs','row_indices'} and torch.equal(output['row_indices'],tensors['row_indices']),'Output global row order differs')
        values=output['teacher_log_probs'];require(values.dtype==torch.float32 and values.shape==tensors['responses'].shape and values.device.type=='cpu'
                and bool(torch.isfinite(values).all()) and bool((values[tensors['attention']==0]==0).all()),'Output finite/mask/shape differs')
        expected=int((tensors['prompts']!=self.manifest['pad_token_id']).sum()+tensors['attention'].sum())
        require(raw['scored_tokens']==expected and raw['rows']==len(values),'Score tokens/global rows differ')
        ledger=sidecar['ledger'];require(ledger and ledger[0]['row_start']==0 and ledger[-1]['row_stop']==len(values)
                and all(item['chunk_index']==index and item['row_start']==(0 if index==0 else ledger[index-1]['row_stop']) for index,item in enumerate(ledger)),'Original chunk ledger incomplete')
        return values,receipt,expected

    def score(self,prompts,responses,attention,temperature,chunk,step,transport_rows=320):
        require(not self.failed,'Unknown prior ACK; client permanently failed, no training retry')
        require(step==self.next_step and temperature==1. and chunk==16 and type(transport_rows) is int and transport_rows>0,'Outer/settings differ')
        require(prompts.device.type==responses.device.type==attention.device.type=='cpu','Driver inputs must remainCPU')
        tensors=dict(prompts=prompts.contiguous(),responses=responses.contiguous(),attention=attention.contiguous(),row_indices=torch.arange(len(responses)))
        require(all(t.dtype==torch.long for t in tensors.values()) and len(prompts)==len(responses)==len(attention)==self.manifest['rows_per_rollout'],'Global input dtype/rows differ')
        nonce=secrets.token_hex(32);expected=int((prompts!=self.manifest['pad_token_id']).sum()+attention.sum());ledger=[]
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
                connection.settimeout(self.manifest['timeout_seconds']);connection.connect(self.manifest['service']['socket'])
                self.last_actual_server_peer=peer(connection,self.manifest['service'],self.manifest['parent_cgroup'])
                metadata=self.base('BEGIN_SCORE',nonce,step);metadata.update(rows=len(responses),prompt_width=prompts.shape[1],response_width=responses.shape[1],temperature=1.,chunk=16,
                    chunk_count=(len(responses)+transport_rows-1)//transport_rows,scoring_batch_rows=self.manifest['scoring_batch_rows'],input_sha256=digest(save(tensors)),expected_scored_tokens=expected)
                send(connection,metadata);ack,raw=receive(connection)
                require(ack==dict(status='CPU_BEGIN_ACCEPTED',nonce=nonce,outer_step=step) and not raw,'BEGIN ACK differs')
                for index,start in enumerate(range(0,len(responses),transport_rows)):
                    stop=min(start+transport_rows,len(responses));part={key:value[start:stop].contiguous() for key,value in tensors.items()};payload=save(part)
                    metadata=self.base('INPUT_CHUNK',nonce,step);metadata.update(chunk_index=index,row_start=start,row_stop=stop,input_sha256=digest(payload))
                    send(connection,metadata,payload);ack,raw=receive(connection)
                    require(ack==dict(status='CPU_CHUNK_ACCEPTED',nonce=nonce,outer_step=step,chunk_index=index) and not raw,'Chunk ACK differs')
                    ledger.append(dict(chunk_index=index,row_start=start,row_stop=stop,input_sha256=digest(payload)))
                metadata=self.base('COMMIT_SCORE',nonce,step);metadata.update(input_sha256=digest(save(tensors)),ledger_sha256=digest(canonical(ledger)))
                send(connection,metadata);result,payload=receive(connection);values,receipt,actual_tokens=self.validate_result(result,payload,tensors,nonce,step)
                require(read_private_json(self.directory,receipt['sidecar'])['ledger']==ledger,'Server ledger differs from actual sent chunks')
                metadata=self.base('ACK_RESULT',nonce,step);metadata.update(output_sha256=digest(payload),receipt_sha256=result['receipt']['sha256'])
                send(connection,metadata);ack,raw=receive(connection)
                require(ack==dict(status='ACKED',nonce=nonce,outer_step=step) and not raw,'Unknown result ACK')
            self.scored_tokens+=actual_tokens;self.parameter_count=receipt['original_scorer_receipt']['released_cpu']['parameter_count'];self.last_receipts=[receipt];self.next_step+=1
            return values
        except Exception:
            self.failed=True;raise  # No auto reconnect, no hidden repeated GPU stage.
