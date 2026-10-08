"""One GPU owner/eight local Unix rank endpoints; unvalidated candidate only."""
import os
import socket
import stat
import threading
import time
from pathlib import Path
import torch
from safetensors.torch import load,save
from .protocol import (IDENTITY,canonical,digest,peer,private_directory,process_identity,
                      receive,send,require,write_private_json)

PROTOCOL='TOPD32B_ONE_OWNER_EIGHT_RANK_ENDPOINTS_V1'


def identity(manifest):return {key:manifest[key] for key in IDENTITY}


def validate_manifest(m):
    require(m['protocol']==PROTOCOL and m['gpu_owner_process_count']==1 and m['rank_endpoint_count']==8,
            'Require one honest GPU owner/eight rank endpoints')
    require(m['kind'] in ('CPU_FIXTURE_NO_ALGORITHM','NATIVE_HARDWARE_PROBE_NO_TRAINING','NATIVE_32B_TOPD_SHARED_PIPELINE'),
            'Unknown backend kind')
    if m['kind']=='NATIVE_32B_TOPD_SHARED_PIPELINE':
        from .readiness import validate_training_readiness
        validate_training_readiness(m)
    require(len(m['endpoints'])==8 and [e['rank'] for e in m['endpoints']]==list(range(8)), 'Rank endpoint order')
    require(len({e['socket'] for e in m['endpoints']})==8, 'Unique sockets required')
    require(all(e['pid']==m['owner']['pid'] and e['starttime']==m['owner']['starttime'] for e in m['endpoints']),
            'Rank sockets must describe same actual owner PID/starttime')
    require(len(m['leased_uuids'])==len(set(m['leased_uuids']))==8, 'Eight GPU lease identities')
    require(type(m['rows_per_rank']) is int and 0<m['rows_per_rank']<=320, 'Bounded contiguous rank rows')
    require(m['workers']==8 and m['temperature']==1. and m['chunk']==16,'Unchanged original math')
    require(0<m['timeout_seconds']<=3600 and m['outer_updates']>0,'Bounded phase budget')
    require(isinstance(m['capability'],str) and len(m['capability'])==64,'Private capability required')


class Coordinator:
    """CPU gather8 → one stage/score/release → all8 results → all8 ACK barrier."""
    def __init__(self,scorer,manifest,directory):
        validate_manifest(manifest)
        self.scorer=scorer;self.m=manifest;self.directory=private_directory(directory)
        if manifest['kind']=='NATIVE_32B_TOPD_SHARED_PIPELINE':
            from .readiness import validate_training_readiness
            proof=validate_training_readiness(manifest)
            require(scorer.initial['parameter_sha256']==proof['fp32_parameter_sha256'],'Actual CPU owner model differs from reviewed teacher')
        if manifest['kind'] in ('NATIVE_HARDWARE_PROBE_NO_TRAINING','NATIVE_32B_TOPD_SHARED_PIPELINE'):
            import inspect
            from .protocol import file_sha
            require(type(scorer).__name__=='PipelineScorer' and type(scorer.runtime).__name__=='PipelineRuntime'
                    and scorer.runtime.kind=='NATIVE_UNVALIDATED_PROTOTYPE', 'Actual native shared pipeline only')
            require(file_sha(inspect.getfile(type(scorer)))=='8d7a847daa6ffc84d62c24c0cdb6250c285c8433883fc37fdfe72b6f0254edde'
                    and scorer.initial['parameter_count']==32763876352, 'Current actual full32B candidate only')
            require(manifest['owner']['pid']==os.getpid() and process_identity(os.getpid())['starttime']==manifest['owner']['starttime'], 'Actual oneowner identity')
            scorer.runtime.verify()
            require(scorer.runtime.uuids==manifest['leased_uuids'], 'Actual leased UUID ordering')
        self.condition=threading.Condition();self.step=1;self.state='READY';self.parts={};self.acks=set()
        self.result=None;self.receipt=None;self.nonce=None;self.failed=None;self.seen=set()

    def fail(self,error):
        with self.condition:
            self.failed=str(error);self.state='FAILED';self.condition.notify_all()

    def authenticate(self,meta,command,rank):
        require(set(meta)==set(IDENTITY)|{'protocol','command','capability','outer_step','nonce','rank','input_sha256'},'Exact token-only metadata')
        require(all(meta[k]==self.m[k] for k in IDENTITY) and meta['protocol']==PROTOCOL
                and meta['command']==command and meta['capability']==self.m['capability'],'Request identity/authentication')
        require(type(rank) is int and 0<=rank<8 and meta['rank']==rank,'Rank identity')
        require(type(meta['outer_step']) is int and meta['outer_step']==self.step<=self.m['outer_updates'],'Actual outer order')
        require(type(meta['nonce']) is str and len(meta['nonce'])==64 and all(c in '0123456789abcdef' for c in meta['nonce']) and meta['nonce'] not in self.seen,'Unique phase nonce')
        require(self.nonce in (None,meta['nonce']),'Cross-rank phase nonce mismatch')

    def wait(self,predicate,deadline):
        while not predicate():
            require(not self.failed,'Failed global phase: '+str(self.failed))
            remaining=deadline-time.monotonic()
            require(remaining>0,'Missing rank/result/ACK timeout; actor forbidden')
            self.condition.wait(remaining)
        require(not self.failed,'Failed global phase: '+str(self.failed))

    def submit(self,rank,meta,payload,actual_peer):
        deadline=time.monotonic()+self.m['timeout_seconds']
        try:
            with self.condition:
                self.authenticate(meta,'SCORE_RANK',rank)
                require(self.state in ('READY','GATHERING') and rank not in self.parts,'Duplicate/concurrent rank')
                require(digest(payload)==meta['input_sha256'],'Rank payload SHA')
                part=load(payload)
                require(set(part)=={'prompts','responses','attention','row_indices'},'Tokens only; no Gold')
                p,r,a,i=[part[k] for k in ('prompts','responses','attention','row_indices')]
                count=self.m['rows_per_rank'];start=rank*count
                require(all(t.device.type=='cpu' and t.dtype==torch.long for t in part.values()),'CPU int64 only')
                require(p.ndim==r.ndim==2 and len(p)==len(r)==count and r.shape==a.shape
                        and 0<p.shape[1]<=2048 and 0<r.shape[1]<=4096,'Exact per-rank shapes')
                require(torch.equal(i,torch.arange(start,start+count)),'Actual global row indices')
                require(bool(((a==0)|(a==1)).all()),'Binary mask')
                self.nonce=meta['nonce'];self.state='GATHERING'
                self.parts[rank]=dict(tensors=part,input_sha256=meta['input_sha256'],actual_peer=actual_peer)
                leader=len(self.parts)==8
                if leader:self.state='SCORING'
                else:
                    self.wait(lambda:self.result is not None,deadline)
            if leader:
                tensors={k:torch.cat([self.parts[j]['tensors'][k] for j in range(8)]) for k in part}
                # No rank output exists until global original math + all8 release completes.
                values,raw=self.scorer.score_round(tensors,workers=8)
                require(raw['status']=='RELEASED' and raw['all_threads_joined'] is True
                        and 'error' not in raw and 'release_error' not in raw,'Global worker/release barrier')
                alloc=raw['allocators_released']
                require(len(alloc)==8 and {x['device'] for x in alloc}==set(range(8))
                        and all(x['allocated_bytes']==x['reserved_bytes']==0 for x in alloc),'All8 allocator zero')
                before,after=raw['initial_cpu'],raw['released_cpu']
                require(all(before[k]==after[k] for k in ('parameter_sha256','buffer_sha256','parameter_count','parameter_bytes'))
                        and after['parameter_devices']==['cpu'] and all(d=='cpu' for d in after['buffer_devices'])
                        and after['training_modules']==after['requires_grad_parameters']==after['grad_parameters']==0,'Full frozen CPU boundary')
                require(values.shape==tensors['responses'].shape and values.dtype==torch.float32
                        and values.device.type=='cpu' and bool(torch.isfinite(values).all())
                        and bool((values[tensors['attention']==0]==0).all()),'Finite masked output')
                require(raw['input_sha256']==digest(save(tensors)) and raw['output_sha256']==digest(save(dict(teacher_log_probs=values,row_indices=tensors['row_indices']))), 'Global scorer input/output identity')
                record=dict(identity(self.m),protocol=PROTOCOL,kind=self.m['kind'],status='GLOBAL_ALL8_RELEASED',
                            gpu_owner_process_count=1,rank_endpoint_count=8,owner=self.m['owner'],nonce=self.nonce,
                            outer_step=self.step,rank_inputs=[dict(rank=j,input_sha256=self.parts[j]['input_sha256'],actual_peer=self.parts[j]['actual_peer']) for j in range(8)],
                            original_scorer_receipt=raw,global_input_sha256=digest(save(tensors)),
                            global_output_sha256=digest(save(dict(teacher_log_probs=values,row_indices=tensors['row_indices']))),
                            publication_epoch=time.time(),scope='UNREVIEWED_CANDIDATE_NOT_ALGORITHM_ACCEPTANCE')
                with self.condition:
                    require(not self.failed,'Rank failure during scoring; discard released results')
                    ref=write_private_json(self.directory,'global-release-step'+str(self.step)+'.json',record)
                    self.result=values;self.receipt=ref;self.state='RESULT_READY';self.condition.notify_all()
            with self.condition:
                require(not self.failed,'Failed global phase')
                start=rank*self.m['rows_per_rank'];stop=start+self.m['rows_per_rank']
                body=save(dict(teacher_log_probs=self.result[start:stop].contiguous(),row_indices=torch.arange(start,stop)))
                return dict(identity(self.m),protocol=PROTOCOL,status='GLOBAL_RELEASED_RANK_RESULT',rank=rank,
                            outer_step=self.step,nonce=self.nonce,receipt=self.receipt,output_sha256=digest(body)),body
        except BaseException as exc:
            self.fail(repr(exc));raise

    def acknowledge(self,rank,meta,payload):
        deadline=time.monotonic()+self.m['timeout_seconds']
        try:
            with self.condition:
                self.authenticate(meta,'ACK_RANK',rank)
                require(not payload and self.state in ('RESULT_READY','ACKED') and rank not in self.acks
                        and meta['input_sha256']==self.receipt['sha256'],'Result ACK binding')
                step=self.step;nonce=self.nonce;self.acks.add(rank)
                if len(self.acks)==8:
                    write_private_json(self.directory,'global-ack-step'+str(step)+'.json',dict(identity(self.m),
                        protocol=PROTOCOL,status='ALL8_ACKED_ACTOR_RELEASE_BARRIER',outer_step=step,nonce=nonce,
                        ranks=list(range(8)),release_receipt=self.receipt,ack_epoch=time.time(),scope=self.m['kind']))
                    self.state='ACKED';self.condition.notify_all()
                self.wait(lambda:len(self.acks)==8,deadline)
                # Keep completed phase until explicit new server lifetime/phase reset.
                return dict(status='ALL8_ACKED_ACTOR_RELEASE_BARRIER',outer_step=step,nonce=nonce)
        except BaseException as exc:
            self.fail(repr(exc));raise

    def advance(self):
        with self.condition:
            require(self.state=='ACKED' and len(self.acks)==8 and not self.failed,'ACK barrier incomplete')
            self.seen.add(self.nonce);self.step+=1;self.parts={};self.acks=set();self.result=None;self.receipt=None;self.nonce=None;self.state='READY'


class RankUnixServer:
    """SamePID eight sockets; one serialized outer loop, threadperendpoint."""
    def __init__(self,coordinator):
        self.c=coordinator;self.listeners=[];self.created=[]

    def start(self):
        directory=self.c.directory
        try:
            for endpoint in self.c.m['endpoints']:
                path=Path(endpoint['socket'])
                require(path.parent.resolve()==directory and len(os.fsencode(path))<108 and not path.exists(),'Private short fresh socket')
                listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.settimeout(self.c.m['timeout_seconds'])
                listener.bind(str(path));os.chmod(path,0o600);listener.listen(1)
                self.listeners.append(listener);self.created.append((path,path.stat().st_ino))
        except BaseException:
            self.close();raise

    def serve(self):
        try:
            while self.c.step<=self.c.m['outer_updates']:
                errors=[]
                def handler(rank):
                    try:
                        with self.listeners[rank].accept()[0] as connection:
                            connection.settimeout(self.c.m['timeout_seconds'])
                            actual=peer(connection,self.c.m['driver'],self.c.m['parent_cgroup'])
                            meta,body=receive(connection)
                            result,body=self.c.submit(rank,meta,body,actual);send(connection,result,body)
                            meta,body=receive(connection)
                            result=self.c.acknowledge(rank,meta,body);send(connection,result)
                    except BaseException as exc:
                        errors.append(repr(exc));self.c.fail(repr(exc))
                threads=[threading.Thread(target=handler,args=(r,),name='rank-afunix-'+str(r)) for r in range(8)]
                for t in threads:t.start()
                for t in threads:t.join(self.c.m['timeout_seconds']+1)
                require(not errors and not any(t.is_alive() for t in threads),'Rank connection failed; no nextphase or actor')
                self.c.advance()
        finally:self.close()

    def close(self):
        for listener in self.listeners:listener.close()
        self.listeners=[]
        for path,inode in self.created:
            try:value=path.lstat()
            except FileNotFoundError:continue
            require(stat.S_ISSOCK(value.st_mode) and value.st_uid==os.getuid() and value.st_ino==inode,'Own socket replaced')
            path.unlink()
        self.created=[]


class RankTeacherClient:
    def __init__(self,manifest,directory):
        validate_manifest(manifest);self.m=manifest;self.directory=private_directory(directory)
        self.failed=False;self.step=1;self.scored_tokens=0;self.last_receipts=[];self.parameter_count=None

    def score(self,prompts,responses,attention,temperature,chunk,step):
        require(not self.failed and step==self.step and temperature==1. and chunk==16,'No replay/settings change')
        require(all(t.device.type=='cpu' and t.dtype==torch.long for t in (prompts,responses,attention)),'CPU int64 only')
        count=self.m['rows_per_rank'];require(len(prompts)==len(responses)==len(attention)==8*count,'All8 ranks mandatory')
        import secrets
        from .protocol import read_private_json
        nonce=secrets.token_hex(32);results=[None]*8;errors=[];receipt_refs=[None]*8
        def base(rank,command,sha):
            return dict(identity(self.m),protocol=PROTOCOL,command=command,capability=self.m['capability'],
                        outer_step=step,nonce=nonce,rank=rank,input_sha256=sha)
        def worker(rank):
            try:
                start=rank*count;stop=start+count
                tensors=dict(prompts=prompts[start:stop].contiguous(),responses=responses[start:stop].contiguous(),
                             attention=attention[start:stop].contiguous(),row_indices=torch.arange(start,stop))
                body=save(tensors)
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
                    connection.settimeout(self.m['timeout_seconds']);connection.connect(self.m['endpoints'][rank]['socket'])
                    peer(connection,self.m['owner'],self.m['parent_cgroup'])
                    send(connection,base(rank,'SCORE_RANK',digest(body)),body)
                    meta,body=receive(connection)
                    require(set(meta)==set(IDENTITY)|{'protocol','status','rank','outer_step','nonce','receipt','output_sha256'}
                            and all(meta[k]==self.m[k] for k in IDENTITY) and meta['protocol']==PROTOCOL
                            and meta['status']=='GLOBAL_RELEASED_RANK_RESULT' and meta['rank']==rank
                            and meta['outer_step']==step and meta['nonce']==nonce and meta['output_sha256']==digest(body),'Result binding')
                    result=load(body)
                    require(set(result)=={'teacher_log_probs','row_indices'} and torch.equal(result['row_indices'],tensors['row_indices']),'Returned order')
                    values=result['teacher_log_probs']
                    require(values.shape==responses[start:stop].shape and values.dtype==torch.float32 and bool(torch.isfinite(values).all())
                            and bool((values[attention[start:stop]==0]==0).all()),'Returned finite/mask')
                    receipt=read_private_json(self.directory,meta['receipt'])
                    require(receipt['status']=='GLOBAL_ALL8_RELEASED' and receipt['nonce']==nonce and receipt['outer_step']==step
                            and all(receipt[k]==self.m[k] for k in IDENTITY),'Global release receipt')
                    require(receipt['rank_inputs'][rank]['input_sha256']==digest(save(tensors)),'Own rank source SHA')
                    raw=receipt['original_scorer_receipt'];alloc=raw['allocators_released']
                    initial,final=raw['initial_cpu'],raw['released_cpu']
                    require(raw['status']=='RELEASED' and 'error' not in raw and 'release_error' not in raw
                            and all(initial[k]==final[k] for k in ('parameter_sha256','buffer_sha256','parameter_count','parameter_bytes'))
                            and final['parameter_devices']==['cpu'] and all(d=='cpu' for d in final['buffer_devices'])
                            and final['training_modules']==final['requires_grad_parameters']==final['grad_parameters']==0, 'Actual frozen CPU boundary before rank ACK')
                    require(raw['all_threads_joined'] is True and len(alloc)==8 and {a['device'] for a in alloc}==set(range(8))
                            and all(a['allocated_bytes']==a['reserved_bytes']==0 for a in alloc),'All8 allocator barrier before rank ACK')
                    receipt_refs[rank]=meta['receipt'];results[rank]=values
                    send(connection,base(rank,'ACK_RANK',meta['receipt']['sha256']))
                    ack,body=receive(connection)
                    require(ack==dict(status='ALL8_ACKED_ACTOR_RELEASE_BARRIER',outer_step=step,nonce=nonce) and not body,'Unknown global ACK; actor forbidden')
            except BaseException as exc:errors.append(repr(exc))
        threads=[threading.Thread(target=worker,args=(r,)) for r in range(8)]
        for t in threads:t.start()
        for t in threads:t.join(self.m['timeout_seconds']+1)
        if errors or any(t.is_alive() for t in threads):
            self.failed=True;raise RuntimeError('Rank/ACK barrier failed: '+repr(errors))
        if not all(ref==receipt_refs[0] for ref in receipt_refs):
            self.failed=True;raise RuntimeError('Ranks disagree global release')
        output=torch.cat(results)
        self.last_receipts=[read_private_json(self.directory,receipt_refs[0])]
        global_tensors=dict(prompts=prompts.contiguous(),responses=responses.contiguous(),attention=attention.contiguous(),row_indices=torch.arange(len(responses)))
        receipt=self.last_receipts[0]
        if receipt['global_input_sha256']!=digest(save(global_tensors)) or receipt['global_output_sha256']!=digest(save(dict(teacher_log_probs=output,row_indices=global_tensors['row_indices']))):
            self.failed=True;raise RuntimeError('Reassembled global input/output SHA mismatch; actor forbidden')
        self.parameter_count=self.last_receipts[0]['original_scorer_receipt']['released_cpu']['parameter_count']
        self.scored_tokens+=int((prompts!=self.m['pad_token_id']).sum()+attention.sum());self.step+=1
        return output


def require_global_ack_before_actor(client,step):
    """Fresh disk barrier verification, never teacher/backend readiness approval."""
    from .protocol import artifact,read_private_json
    require(not client.failed and client.step==step+1 and len(client.last_receipts)==1,
            'Actor blocked: scoring or anyrankACK incomplete')
    receipt=client.last_receipts[0]
    require(receipt['status']=='GLOBAL_ALL8_RELEASED' and receipt['outer_step']==step
            and all(receipt[k]==client.m[k] for k in IDENTITY),'Actor step/run binding')
    name='global-ack-step'+str(step)+'.json'
    path=artifact(client.directory,name)
    import json
    ack=json.loads(path.read_text())
    require(ack['status']=='ALL8_ACKED_ACTOR_RELEASE_BARRIER' and ack['outer_step']==step
            and ack['nonce']==receipt['nonce'] and ack['ranks']==list(range(8))
            and all(ack[k]==client.m[k] for k in IDENTITY),'Actor blocked: fresh ACK artifact invalid')
    raw=read_private_json(client.directory,ack['release_receipt'])
    require(raw==receipt,'Actor blocked: raw released receipt drift')
    return ack


def load_production_client(manifest_path,teacher_path,student_tokenizer):
    """Explicit TOP-D adapter constructor; actual missing dependencies failclosed."""
    import json
    from .readiness import ROOT,artifact,validate_training_readiness
    path=Path(manifest_path).resolve()
    require(ROOT/'logs' in path.parents and not path.is_symlink(),'Own persisted manifest required')
    manifest=json.loads(path.read_text())
    require(manifest['kind']=='NATIVE_32B_TOPD_SHARED_PIPELINE','Hardware/CPU client forbidden in actor training')
    proof=validate_training_readiness(manifest)
    teacher=artifact(proof['selected_teacher'])
    require(Path(teacher_path).resolve()==Path(teacher['model_directory']).resolve()
            and ROOT in Path(teacher_path).resolve().parents,'Actual selected teacher path identity')
    vocab=digest(canonical(student_tokenizer.get_vocab()))
    require(vocab==teacher['student_token_to_id_sha256']==teacher['teacher_token_to_id_sha256']
            and student_tokenizer.bos_token_id==teacher['bos_token_id']
            and student_tokenizer.eos_token_id==teacher['eos_token_id']
            and student_tokenizer.pad_token_id==manifest['pad_token_id'],'Actual student/teacher tokenizer alignment')
    actual=process_identity(os.getpid())
    require(actual['pid']==manifest['driver']['pid'] and actual['starttime']==manifest['driver']['starttime'], 'Actual Ray driver identity')
    client=RankTeacherClient(manifest,Path(manifest['endpoints'][0]['socket']).parent)
    client.parameter_count=proof['parameter_count']
    return client
