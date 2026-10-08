"""One outer request, CPU chunks, one frozen score_round, all-eight release gate."""
import os
import inspect
import time

import torch
from safetensors.torch import load,save

from protocol import IDENTITY,PROTOCOL,canonical,digest,file_sha,require,write_private_json

SCORER_SHA='2d5da3fdd94b637c1625677b302df4df21395ef138f9de2ce484035eab7298ea'
METHOD_SHA='e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d'
DISTILLATION_SHA='fede3c23e882139a12a2b50c531177438012959774958fc39e03fb3f9d29e229'
BINDING=dict(scorer_sha256=SCORER_SHA,original_method_sha256=METHOD_SHA,original_file_sha256=DISTILLATION_SHA)

BEGIN_FIELDS=set(IDENTITY)|{'protocol','command','capability','manifest_sha256','nonce','outer_step','rows','prompt_width','response_width','temperature','chunk','chunk_count','scoring_batch_rows','input_sha256','expected_scored_tokens'}
CHUNK_FIELDS=set(IDENTITY)|{'protocol','command','capability','manifest_sha256','nonce','outer_step','chunk_index','row_start','row_stop','input_sha256'}
COMMIT_FIELDS=set(IDENTITY)|{'protocol','command','capability','manifest_sha256','nonce','outer_step','input_sha256','ledger_sha256'}
ACK_FIELDS=set(IDENTITY)|{'protocol','command','capability','manifest_sha256','nonce','outer_step','output_sha256','receipt_sha256'}


def tensor_sha(tensors):return digest(save(tensors))


def manifest_identity(manifest):return {key:manifest[key] for key in IDENTITY}


def validate_manifest(manifest):
    require(manifest.get('frozen_source_binding')==BINDING,'Frozen external/original source binding differs')
    require(manifest['protocol']==PROTOCOL and manifest['backend']=='local_gpu_fp32_sharded_shared','Wrong shared backend/protocol')
    require(manifest['kind']=='CPU_FIXTURE_NO_ALGORITHM','Prototype currently CPU fixture only; no hardware/training acceptance')
    require(manifest['rows_per_rollout']>0 and manifest['rows_per_rollout']<=2560 and manifest['outer_updates']>0,'Rollout count differs')
    require(manifest['temperature']==1. and manifest['chunk']==16 and manifest['scoring_batch_rows'] in (1,4,8),'Original score settings differ')
    require(manifest['max_prompt']==2048 and manifest['max_response']==4096,'Context contract differs')
    require(len(manifest['leased_uuids'])==len(set(manifest['leased_uuids']))==8,'Eight distinct UUID contract required')
    require(type(manifest['capability']) is str and len(manifest['capability'])==64 and all(c in '0123456789abcdef' for c in manifest['capability']),'256-bit capability required')
    require('service' in manifest and 'services' not in manifest,'Exactly one shared service required')


class SharedSession:
    def __init__(self,scorer,manifest,directory):
        validate_manifest(manifest)
        require(file_sha(inspect.getfile(type(scorer).__init__))==SCORER_SHA and file_sha(inspect.getfile(inspect.unwrap(scorer.original)))==DISTILLATION_SHA
                and digest(inspect.getsource(scorer.original).encode())==METHOD_SHA,'Actual original scorer/source SHA differs')
        require(manifest['original_wrapper_file_sha256']==file_sha(inspect.getfile(scorer.original)),'Actual no_grad wrapper dependency SHA differs')
        require(scorer.runtime.kind=='CPU_ONLY_FIXTURE_NOT_HARDWARE_EVIDENCE','This external prototype may not start native CUDA')
        self.scorer=scorer;self.manifest=manifest;self.directory=directory
        self.manifest_sha=digest(canonical(manifest));self.next_step=1;self.state='CPU_READY';self.begin_record=None
        self.peer_identity=None;self.actual_service_identity=None
        self.chunks=[];self.ledger=[];self.seen_nonces=set();self.receipt=None;self.output_sha=None

    def authenticate(self,metadata,fields,command):
        require(set(metadata)==fields and metadata['command']==command and metadata['protocol']==PROTOCOL,'Wrong request fields/protocol')
        require(all(metadata[key]==self.manifest[key] for key in IDENTITY),'Request identity drift')
        require(metadata['capability']==self.manifest['capability'] and metadata['manifest_sha256']==self.manifest_sha,'Request authorization/manifest drift')
        require(type(metadata['outer_step']) is int and metadata['outer_step']==self.next_step<=self.manifest['outer_updates'],'Outer order differs')
        nonce=metadata['nonce'];require(type(nonce) is str and len(nonce)==64 and all(c in '0123456789abcdef' for c in nonce),'Bad nonce')
        if self.begin_record:require(nonce==self.begin_record['nonce'],'In-flight nonce differs')

    def begin(self,metadata,payload):
        self.authenticate(metadata,BEGIN_FIELDS,'BEGIN_SCORE')
        require(self.state in ('CPU_READY','ACKED') and not payload,'Concurrent request/BEGIN payload forbidden')
        require(metadata['nonce'] not in self.seen_nonces,'Duplicate nonce forbidden')
        require(type(metadata['rows']) is int and metadata['rows']==self.manifest['rows_per_rollout'],'Wrong global row count')
        require(type(metadata['prompt_width']) is int and 0<metadata['prompt_width']<=2048
                and type(metadata['response_width']) is int and 0<metadata['response_width']<=4096,'Context shape differs')
        require(metadata['temperature']==1. and metadata['chunk']==16 and metadata['scoring_batch_rows']==self.manifest['scoring_batch_rows'],'Original score settings drift')
        require(type(metadata['chunk_count']) is int and 0<metadata['chunk_count']<=metadata['rows']
                and type(metadata['expected_scored_tokens']) is int and metadata['expected_scored_tokens']>0,'Invalid chunk/token budget')
        require(type(metadata['input_sha256']) is str and len(metadata['input_sha256'])==64,'Missing full input hash')
        self.begin_record=metadata.copy();self.chunks=[];self.ledger=[];self.state='ASSEMBLING'
        return dict(status='CPU_BEGIN_ACCEPTED',nonce=metadata['nonce'],outer_step=self.next_step)

    def accept_chunk(self,metadata,payload):
        self.authenticate(metadata,CHUNK_FIELDS,'INPUT_CHUNK');require(self.state=='ASSEMBLING','Chunk outside assembler')
        begin=self.begin_record
        require(type(metadata['chunk_index']) is int and metadata['chunk_index']==len(self.chunks)<begin['chunk_count'],'Chunk duplicate/order differs')
        start=sum(t['responses'].shape[0] for t in self.chunks);stop=metadata['row_stop']
        require(type(metadata['row_start']) is int and metadata['row_start']==start and type(stop) is int and start<stop<=begin['rows'],'Chunk row coverage differs')
        require(digest(payload)==metadata['input_sha256'],'Actual chunk wire SHA differs')
        tensors=load(payload);require(set(tensors)=={'prompts','responses','attention','row_indices'},'Only token tensors allowed; no gold')
        p,r,a,i=(tensors[k] for k in ('prompts','responses','attention','row_indices'))
        require(all(t.dtype==torch.long and t.device.type=='cpu' for t in tensors.values()),'Chunk requires CPUint64')
        require(p.shape==(stop-start,begin['prompt_width']) and r.shape==a.shape==(stop-start,begin['response_width'])
                and torch.equal(i,torch.arange(start,stop)),'Chunk shape/global row order differs')
        require(bool(((a==0)|(a==1)).all()),'Chunk mask not binary')
        require(all(int(v) in self.scorer.allowed_ids for v in p.flatten()) and all(int(v) in self.scorer.allowed_ids for v in r.flatten()),'Unmapped student token')
        self.chunks.append(tensors);self.ledger.append(dict(chunk_index=len(self.ledger),row_start=start,row_stop=stop,input_sha256=metadata['input_sha256']))
        return dict(status='CPU_CHUNK_ACCEPTED',nonce=begin['nonce'],outer_step=self.next_step,chunk_index=metadata['chunk_index'])

    def commit(self,metadata,payload):
        self.authenticate(metadata,COMMIT_FIELDS,'COMMIT_SCORE');require(self.state=='ASSEMBLING' and not payload,'COMMIT state/payload differs')
        begin=self.begin_record
        require(len(self.chunks)==begin['chunk_count'] and self.ledger[-1]['row_stop']==begin['rows'],'Incomplete global rollout')
        require(digest(canonical(self.ledger))==metadata['ledger_sha256'],'Chunk ledger SHA differs')
        tensors={key:torch.cat([part[key] for part in self.chunks]) for key in self.chunks[0]}
        require(tensor_sha(tensors)==metadata['input_sha256']==begin['input_sha256'],'Full tensor input SHA differs')
        expected=int((tensors['prompts']!=self.scorer.teacher.tokenizer.pad_token_id).sum()+tensors['attention'].sum())
        require(expected==begin['expected_scored_tokens'],'Global scoretoken count differs')
        self.state='SCORING';output,raw=self.scorer.score_round(tensors,batch_size=begin['scoring_batch_rows'])
        self.state='VERIFYING_RELEASE'
        require(raw['status']=='RELEASED' and 'error' not in raw and 'release_error' not in raw,'Scorer not released')
        allocations=raw['allocators_released']
        require(len(allocations)==8 and {a['device'] for a in allocations}==set(range(8))
                and all(a['allocated_bytes']==a['reserved_bytes']==0 for a in allocations),'All-eight actual release barrier missing')
        require(raw['released_cpu']['parameter_devices']==['cpu'] and all(d=='cpu' for d in raw['released_cpu']['buffer_devices']),'Teacher retained GPU model/buffers')
        require(raw['scored_tokens']==expected and raw['rows']==begin['rows'],'Actual scorer accounting differs')
        output_raw=save(dict(teacher_log_probs=output,row_indices=tensors['row_indices']))
        require(raw['input_sha256']==begin['input_sha256'] and raw['output_sha256']==digest(output_raw),'Raw scorer tensor identity differs')
        sidecar=dict(scope='CPU_FIXTURE_PARTITION_INVENTORY_NOT_GPU_PROOF',inventory=self.scorer.inventory,ledger=self.ledger)
        sidecar_ref=write_private_json(self.directory,f'partition-round{self.next_step}.json',sidecar)
        record=dict(manifest_identity(self.manifest),protocol=PROTOCOL,kind=self.manifest['kind'],status='RELEASED_ALL8_ZERO',
                    frozen_source_binding=BINDING,original_wrapper_file_sha256=self.manifest['original_wrapper_file_sha256'],nonce=begin['nonce'],outer_step=self.next_step,pid=os.getpid(),service_identity=self.manifest['service'],actual_driver_peer=self.peer_identity,actual_service_identity=self.actual_service_identity,
                    leased_uuids=self.manifest['leased_uuids'],rows=begin['rows'],input_sha256=begin['input_sha256'],output_sha256=digest(output_raw),
                    row_indices_sha256=tensor_sha(dict(row_indices=tensors['row_indices'])),attention_sha256=tensor_sha(dict(attention=tensors['attention'])),
                    temperature=1.,chunk=16,batch_size=begin['scoring_batch_rows'],teacher_targets_none=True,scored_tokens_cumulative=self.scorer.teacher.scored_tokens,
                    sidecar=sidecar_ref,original_scorer_receipt=raw,publication_epoch=time.time(),scope='CPU_FIXTURE_IPC_NOT_GPU_OR_TRAINING_ACCEPTANCE')
        self.receipt=write_private_json(self.directory,f'receipt-round{self.next_step}.json',record);self.output_sha=digest(output_raw)
        self.state='RESULT_READY';self.chunks=[]
        return dict(manifest_identity(self.manifest),protocol=PROTOCOL,status='RESULT_RELEASED',nonce=begin['nonce'],outer_step=self.next_step,
                    receipt=self.receipt,output_sha256=self.output_sha),output_raw

    def acknowledge(self,metadata,payload):
        self.authenticate(metadata,ACK_FIELDS,'ACK_RESULT')
        require(self.state=='RESULT_READY' and not payload and metadata['output_sha256']==self.output_sha
                and metadata['receipt_sha256']==self.receipt['sha256'],'Unknown/incorrect ACK; no training retry')
        nonce=self.begin_record['nonce'];step=self.next_step;self.seen_nonces.add(nonce);self.next_step+=1
        self.state='ACKED';self.begin_record=None;self.ledger=[]
        return dict(status='ACKED',nonce=nonce,outer_step=step)

    def abort(self):
        self.chunks=[];self.ledger=[];self.state='FAILED'  # Scorer.score_round owns actual finally release.
