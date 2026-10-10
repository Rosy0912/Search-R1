import sys,json,hashlib,time,os
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010')
R=Path(__file__).parent
sys.path.insert(0,str(B/'source/guard-recovery-5585c5a/opsd_j_bf16_guard200_recovery_v1'))
from entry import Recovery
from controller import *
class Periodic(Recovery):
    def sample(self,outer,questions,identity):
        return Controller.sample(self,outer,questions,identity)
    def evaluate(self,number,checkpoint):
        destination=B/'periodic512'/f'eval-{number:03d}'
        done=destination/'summary.json'
        if done.exists():
            value=json.loads(done.read_text());require(value['checkpoint']==checkpoint and value['count']==512,'Changed existing evaluation');return
        destination.mkdir(parents=True,exist_ok=True)
        self.stage('EVALUATING_512',completed_outers=number,checkpoint=checkpoint)
        self.gate('sampling-start',[])
        jobs=[]
        for rank in range(8):
            command=[self.config['paths']['python'],'-B',str(R/'worker.py'),'--config',self.config_ref['path'],'--checkpoint',checkpoint['path'],'--checkpoint-sha256',checkpoint['sha256'],'--rank',str(rank),'--output',str(destination)]
            jobs.append(self.spawn(command,destination/f'worker-{rank}.log','sampling-running',dict(CUDA_VISIBLE_DEVICES=str(rank))))
        terminal=self.wait(jobs,'sampling-running');exclusive_json(destination/'actual-exit-release.json',terminal)
        rows=[]
        for rank in range(8):
            rows.extend(read_rows(destination/f'results-{rank}.jsonl'))
        require(len(rows)==512 and sorted(x['index'] for x in rows)==list(range(512)),'Incomplete or duplicate512')
        totals={}
        for x in rows:
            v=totals.setdefault(x['dataset'],dict(count=0,hits=0));v['count']+=1;v['hits']+=x['em']
        exclusive_json(done,dict(count=512,hits=sum(x['em'] for x in rows),em=sum(x['em'] for x in rows)/512,by_dataset=totals,checkpoint=checkpoint,completed_outers=number,actual_exit_release=terminal,finished_unix=time.time(),protocol='native HF SDPA BF16base FP32unmergedadapter greedy rep1.05',exclusive_GPU=True))
    def run(self):
        # Approval is checked before creating experimental outputs or merging a model.
        initial=self.gate('cpu-merge',[])
        Path(self.config['paths']['base']).mkdir(parents=True,exist_ok=True)
        self.lock=open(Path(self.config['paths']['base'])/'online-experiment.lock','a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in self.config['resources']['shared_gpu_lease_paths']+[self.config['resources']['study_lease_path']]:
            lease_path=Path(name);require(lease_path.parent.is_dir(),'Shared lease parent must be CPU prepared')
            lease=lease_path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);self.shared_locks.append(lease)
        self.gate('cpu-merge',[])
        verify_runtime(self.config)
        verify_common(self.config)
        if not self.resume:
            require(not self.output.exists(),'Fresh output root already exists; explicit same-run resume required')
            self.output.mkdir(parents=True)
            exclusive_json(self.output/'run-identity.json',dict(schema='opsd-online-run-identity-v1',run_id=self.run_id,
                variant=self.config['variant'],study_id=self.config['study_id'],contract=science_contract(self.config),
                config=self.config_ref,config_sha256=self.config_ref['sha256'],initial_owner=self.owner,initial_gate=initial,
                retention=self.config['retention'],no_old_adam_restored=False,continuation=self.config['continuation']))
            self.owns_output=True
        else:
            identity=json.loads((self.output/'run-identity.json').read_text())
            require(identity['run_id']==self.run_id and identity['config_sha256']==self.config_ref['sha256'],'Cannot resume another run/config')
            self.owns_output=True
        prefix=self.config['continuation']
        previous=prefix['prefix_checkpoint'];verified,_=check_checkpoint(previous,self.config,self.run_id,50,4)
        require(verified['cumulative_pair_exposure']==1497,'Accepted five-round exposure differs')
        updates=50;cursor=dict(cycle=0,offset=2560)
        for index in range(5,self.config['scientific']['outer_iterations']):
            outer=self.output/'outers'/f'outer-{index:05d}';outer.mkdir(parents=True,exist_ok=True)
            completed=outer/'training'/'outer-complete.json'
            manifest_path=outer/'outer-manifest.json'
            if completed.exists():
                manifest,_=checked_json(fileproof(manifest_path))
                self.verify_outer_inputs(manifest,index,updates,previous,cursor)
                require((outer/'teacher-diagnostics'/'complete.json').exists(),'Completed outer teacher diagnostic proof missing')
                self.teacher_diagnostic(outer,index,manifest)
                exits=list((outer/'trainer-logs').glob('attempt-*-exit.json'))
                require(exits,'Completed outer has no actual trainer exit proof; independent terminal recovery required')
                verified=False
                for path in exits:
                    value,_=checked_json(fileproof(path))
                    if value.get('accepted') is True:self.verify_exit(value,'training-running',1);verified=True
                require(verified,'Completed outer has no successful committed trainer phase')
                value,_=check_outer_complete(fileproof(completed),self.config,self.run_id,index,updates,manifest['P'])
                updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after']
                if (index+1)%10==0:self.evaluate(index+1,previous)
                continue
            if manifest_path.exists():
                manifest,manifest_ref=checked_json(fileproof(manifest_path))
                questions=self.verify_outer_inputs(manifest,index,updates,previous,cursor)
            else:
                questions_path=outer/'questions.jsonl';selection_path=outer/'train-selection.json'
                if questions_path.exists() or selection_path.exists():
                    require(questions_path.exists() and selection_path.exists(),'Incomplete CPU selection preserved; root review required')
                    questions=read_rows(questions_path);selection,_=checked_json(fileproof(selection_path))
                    expected_rows,expected_selection=cohort(self.config,cursor)
                    require(questions==expected_rows and selection==expected_selection,'Resumed selected questions differ from bound TRAIN stream')
                else:
                    questions,selection=cohort(self.config,cursor);write_rows(questions_path,questions);exclusive_json(selection_path,selection)
                if previous is None and self.config['precision']['mode']=='bf16':
                    model_identity=dict(kind='raw-source100',path=self.config['paths']['model'],sha256=self.config['source_model_manifest']['sha256'],
                        source_model_identity=self.config['source_model_manifest']['sha256'],optimizer_updates=0,adapter=None,
                        checkpoint=None,adapter_tensor_sha256=self.config['common_initial_adapter_tensor_sha256'],
                        variant=self.config['variant'],precision=self.config['precision'],common_initial_adapter=self.config['common_initial_adapter_manifest'])
                else:
                    model_identity=self.cpu_merge(outer,index,previous,updates)
                self.stage('SAMPLING',outer_index=index,optimizer_updates=updates,latest_sampling_model=model_identity['sha256'])
                sampling_path=outer/'sampling-complete.json'
                if sampling_path.exists():
                    sampling,_=checked_json(fileproof(sampling_path));require(all(sampling['sampling_model_identity'].get(k)==model_identity.get(k) for k in ('kind','path','sha256','source_model_identity','checkpoint','adapter_tensor_sha256','optimizer_updates')),'Changed sampling model on resume')
                    checked_file(sampling['rollouts']);checked_file(sampling['pairs'])
                    self.verify_exit(sampling['exit_proof'],'sampling-running',8)
                else:sampling,_=self.sample(outer,questions,model_identity)
                model_identity=sampling['sampling_model_identity']
                manifest=dict(schema='opsd-online-outer-manifest-v1',run_id=self.run_id,config_sha256=self.config_ref['sha256'],
                    runtime_identity=self.config['runtime_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256'],
                    outer_index=index,outer_dir=str(outer),questions=fileproof(questions_path),train_cursor=selection,
                    variant=self.config['variant'],contract=science_contract(self.config),mining_version=self.config['mining']['version'],
                    common_cohort_manifest=self.config['common_cohort_manifest'],common_initial_adapter_manifest=self.config['common_initial_adapter_manifest'],
                    outer_old_adapter=previous or self.config['common_initial_adapter_manifest'],
                    rollouts=sampling['rollouts'],pairs=sampling['pairs'],P=sampling['stats']['P'],sampling_stats=sampling['stats'],
                    sampling_model_identity=model_identity,previous_complete_checkpoint=previous,explicit_fresh=previous is None,
                    cumulative_optimizer_updates_before=updates)
                manifest_ref=exclusive_json(manifest_path,manifest)
            self.teacher_diagnostic(outer,index,manifest)
            resume=latest_resume_checkpoint(self.output,self.config,self.run_id,index)
            command=[self.config['paths']['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(Path(self.config['paths']['helper_root'])/'trainer.py'),'--config',self.config_ref['path'],
                '--config-sha256',self.config_ref['sha256'],'--outer-manifest',manifest_ref['path'],
                '--outer-manifest-sha256',manifest_ref['sha256']]
            if resume:
                command+=['--resume-checkpoint',resume['path'],'--resume-checkpoint-sha256',resume['sha256']]
                try:from .retry_state import select_retry_capsule
                except ImportError:from retry_state import select_retry_capsule
                retry=select_retry_capsule(outer/'training',resume,self.config)
                if retry:command+=['--retry-capsule-manifest',retry['path'],'--retry-capsule-sha256',retry['sha256']]
            self.stage('TRAINING',outer_index=index,optimizer_updates_before=updates,resume_checkpoint=resume)
            logs=outer/'trainer-logs';logs.mkdir(exist_ok=True);attempt=len(list(logs.glob('attempt-*.log')))
            self.gate('training-start',[])
            job=self.spawn(command,logs/f'attempt-{attempt:05d}.log','training-running',dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'))
            terminal=self.wait([job],'training-running');exclusive_json(logs/f'attempt-{attempt:05d}-exit.json',terminal)
            value,evidence=check_outer_complete(fileproof(completed),self.config,self.run_id,index,updates,manifest['P'])
            updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after']
            self.stage('OUTER_COMPLETE',outer_index=index,actual_optimizer_updates=updates,P=value['P'],K=value['K'],checkpoint=previous)
            if (index+1)%10==0:self.evaluate(index+1,previous)
            from checkpoint import prune_merged_caches
            prune_merged_caches(self.output,self.run_id,self.config_ref['sha256'])
        release=self.wait_release([])
        self.stage('TRAIN_COMPLETE',outer_iterations=200,actual_optimizer_updates=updates,latest_checkpoint=previous,resource_release=release,
            scope='Full200 online outers including preserved first5; no512 evaluation')

if __name__=='__main__':
    cfg=B/'receipts/training-config.json';x=Periodic(fileproof(cfg),'d040ebac9fa24ec28d43976299fd4f71',resume=True)
    try:x.run()
    except Exception as exc:
        if x.owns_output:x.contain_failure(exc)
        raise
