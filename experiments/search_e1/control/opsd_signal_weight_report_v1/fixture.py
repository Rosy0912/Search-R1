"""Clearly synthetic aggregate data and native-format interface fixtures.

No production evidence or statistical estimates are created by this module.
"""
import copy
import hashlib
from pathlib import Path
try:
    from .common import VARIANTS, ENDPOINTS, encode
    from .render import TRAIN_FEATURES, HELDOUT_FEATURES, PRIMARY
except ImportError:
    from common import VARIANTS, ENDPOINTS, encode
    from render import TRAIN_FEATURES, HELDOUT_FEATURES, PRIMARY


def example_data():
    curves={}; models={}; train_groups={}; held_models={}; paired={}; comparisons=[]
    def count(h,n):
        invalid=min(4,n-h)
        return dict(denominator=n,questions=n,hits=h,em=h/n,final_invalid=invalid,valid_gold_mismatch=n-h-invalid,no_executed_search=9)
    for vi,v in enumerate(VARIANTS):
        curves[v]=[dict(outer=i+1,strict_correct=1200+vi*20-i*2,denominator=2560,em=(1200+vi*20-i*2)/2560,
            optimizer_updates=(i+1)*10,K=300,P=310,unweighted_post_loss=.4/(i+1),weighted_post_loss=.41/(i+1),
            raw_loss=.45/(i+1),weight=dict(mean=1.01,sum=303,sumsq=330,min=.1,max=1.9,ess=278.)) for i in range(25)]
        relations={f:{label:dict(measured_questions=7000,population_questions=12800,missing_questions=5800,
            pearson=(-.18 if f in ('student_entropy','answer_gap') else .03),spearman=-.12,raw_direction_auc=.42,
            pearson_question_block=dict(interval95=[-.22,-.14],valid_replicates=2000,requested_replicates=2000,
                 blocks=12800,NA_reason=None)) for label in ('selected_reward','sampling_s_over_5')} for f in TRAIN_FEATURES}
        train_groups[v]=dict(population_questions=12800,population_sampled_trajectories=64000,sampling_strict_correct=sum(r['strict_correct'] for r in curves[v]),
            sampling_train_em=sum(r['em'] for r in curves[v])/25,all5_wrong_noB=5000,eligible_with_B=7800,
            selected_D_encoding_skips=300,actual_trained_questions=7500,CC_CW_exposure={'CC':3200,'CW':4300},
            self_pair_exposure=1100,CC_CW_weighted_exposure={'CC':3232,'CW':4343},selected_weight=dict(sum=7575,mean=1.01,ess=7000),
            actual_batches=250,clipped_batches=150,mean_preclip_norm=1.4,regions={},selected_D_relationships=relations)
    for v,t in ENDPOINTS:
        h=239 if v=='source100' else 235+VARIANTS.index(v)*4+(2 if t==25 else 0)
        o=count(h,512);nq=count(h//2,256);hot=count(h-h//2,256)
        models[v+'/'+str(t)]=dict(overall=o,datasets={'nq':nq,'hotpotqa':hot})
        features={f:dict(questions=512,metric_available=239 if f!='student_entropy' else 512,
            metric_unavailable=273 if f!='student_entropy' else 0,missing_reasons={'no_common_zero_successful_reference':273} if f!='student_entropy' else {},
            correct_available=190,wrong_available=49,subset_em=190/239,pearson=None if v=='source100' and f!='student_entropy' else -.15,
            spearman=None if v=='source100' and f!='student_entropy' else -.19,original_increasing_direction_EM_AUC=None if v=='source100' and f!='student_entropy' else .39,
            association_NA_reason='only_one_EM_class_in_metric_subset' if v=='source100' and f!='student_entropy' else None,
            adjusted_partial_pearson=None if v=='source100' else -.13) for f in HELDOUT_FEATURES}
        held_models[v+'/'+str(t)]=dict(all512=o,datasets={'nq':nq,'hotpotqa':hot},bank_coverage=239,bank_subset=count(239,239) if v=='source100' else count(190,239),
            features=features,regions={})
    keys=list(PRIMARY)+['source100/0->'+v+'/'+str(t) for v in VARIANTS for t in (15,25)]+[v+'/15->'+v+'/25' for v in VARIANTS]
    for key in keys:
        left,right=key.split('->');lh=models[left]['overall']['hits'];rh=models[right]['overall']['hits'];g=max(0,rh-lh)+10;l=max(0,lh-rh)+10
        delta=(rh-lh)*100/512
        paired[key]=dict(questions=512,gained=g,lost=l,preserved=lh-l,still_wrong=512-lh-g,net_hits=g-l,delta_percentage_points=delta,
            lost_final_invalid=2,lost_valid_gold_mismatch=l-2,primary_comparison=key in PRIMARY,delta_direction=right+' minus '+left,
            ci95=dict(low=delta-2,high=delta+2,confidence=.95,unit='percentage_points',interval='single_comparison_percentile'))
        if key in PRIMARY:paired[key]['exact_mcnemar_holm']=dict(raw_p=.6,holm_p=1.,reject_at_family_alpha=False)
    def metric(source=1):return dict(tensors=504,parameters=300,changed_tensors=504,source_l2=source,destination_l2=2.,delta_l2=.8,
        relative_l2=.8/source if source else None,relative_l2_na_reason=None if source else 'ZERO_SOURCE_PARAMETER_NORM')
    for v in VARIANTS:
        for a,b in (('common_zero','outer15'),('outer15','outer25'),('common_zero','outer25')):
            comparisons.append(dict(variant=v,source=a,destination=b,source_updates=0 if a=='common_zero' else 150,
                destination_updates=150 if b=='outer15' else 250,destination_pair_exposure=4500 if b=='outer15' else 7500,
                metrics=dict(aggregate=metric(),by_factor={'A':metric(),'B':metric(0 if a=='common_zero' else 1)})))
    return dict(study_id='SYNTHETIC-FIXTURE-NOT-PRODUCTION',synthetic=True,curves=curves,models=models,
        analyses=dict(effects=dict(schema='opsd-signal-weight-final-effects-v1',accepted=True,formal_predictions=4608,models=copy.deepcopy(models),paired=paired,
            primary_family_size=2,primary_time=25,best_checkpoint_selection=False,bootstrap=dict(repetitions=10000,seed=42,strata=['nq','hotpotqa'])),
            train_relationship=dict(schema='opsd-signal-train-analysis-summary-v1',accepted=True,study_id='SYNTHETIC-FIXTURE-NOT-PRODUCTION',
                complete_100_outers=True,completed_outers=100,groups=train_groups),
            heldout_relationship=dict(schema='opsd-signal-heldout-response-relations-v1',accepted=True,formal_EM_rows=4608,models=held_models)),
        debug=dict(schema='opsd-signal-debug-retention-v1',accepted=True,full_study_complete=True,complete_outers=100,sampling_workers=800,
            teacher_parents=100,teacher_cases=800,teacher_not_applicable_parents=0,study_id='SYNTHETIC-FIXTURE-NOT-PRODUCTION',
            referenced_inventory=dict(files=1000,bytes=100000,sha256='0'*64)),
        movement=dict(schema='opsd-signal-parameter-movement-v1',accepted=True,full_study_complete=True,mode='final',comparisons=comparisons),
        coverage={},request={},request_ref=None)


def native_bundle(root):
    """Tiny model-free exact-schema dependency graph for meaningful reader tests."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=example_data();counter=[0]
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encode(value))
        return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    helper=root/'fixture-helper';helper.mkdir();(helper/'stub.py').write_text('# Synthetic source inventory only\n')
    raw=(helper/'stub.py').read_bytes();manifest=write('helper-manifest.json',dict(files={'stub.py':dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))}))
    publication=write('publication.json',dict(accepted=True,branch='search-e1',source_commit='1'*40,remote_readback_sha='1'*40,source_manifest_sha256=manifest['sha256']))
    source=dict(helper_root=str(helper.resolve()),source_manifest=manifest,publication=publication)
    study=write('study.json',dict(study_id=data['study_id']));configs={};ledgers={};finals={};outer_rows=[];coverage_groups=[];end_refs={};movement_endpoints={}
    for v in VARIANTS:
        cfg=write(v+'/config.json',dict(variant=v,study_id=data['study_id'],run_id='fixture-run-'+v))
        configs[v]=cfg;rows=[];movement_endpoints[v]=[]
        for i,r in enumerate(data['curves'][v]):
            cp=write(v+'/cp-'+str(i)+'.json',dict(run_id='fixture-run-'+v,optimizer_updates=r['optimizer_updates'],
                cursor=dict(outer_index=i,next_pair_cursor=r['K'],usable_pair_count=r['K'])))
            row=dict(schema='opsd-signal-outer-summary-v1',accepted=True,variant=v,run_id='fixture-run-'+v,config_sha256=cfg['sha256'],outer_index=i,
                train_denominator=2560,train_post_update=False,train_correct=r['strict_correct'],train_em=r['em'],raw_checkpoint=cp,
                optimizer_updates=r['optimizer_updates'],K=r['K'],P=r['P'],actual_outer_unweighted_post_loss=r['unweighted_post_loss'],
                actual_outer_weighted_post_loss=r['weighted_post_loss'],actual_outer_raw_loss=r['raw_loss'],weight=r['weight'])
            rows.append(row);outer_rows.append(dict(variant=v,outer_index=i,train_correct=r['strict_correct'],checkpoint=cp,optimizer_updates=r['optimizer_updates'],K=r['K']))
            if i in (14,24):movement_endpoints[v].append(dict(outer_index=i,completed_outer=i+1,evidence={'manifest':cp}))
        finals[v]=cp;ledgers[v]=write(v+'/ledger.json',dict(accepted=True,variant=v,run_id='fixture-run-'+v,outer_iterations=25,rows=rows))
        coverage_groups.append(dict(variant=v,config=cfg,train_ledger=ledgers[v]))
    evaluations=[];coverage_evals=[];zero=None
    for v,t in sorted(ENDPOINTS):
        variant='G_POS' if v=='source100' else v;cfg=configs[variant];name=v+'/'+str(t);model=data['models'][name]
        raw=None if t==0 else next(r['checkpoint'] for r in outer_rows if r['variant']==variant and r['outer_index']==t-1)
        summary=write('endpoints/'+v+'-'+str(t)+'-summary.json',dict(schema='opsd-signal-native512-summary-v1',accepted=True,run_id='fixture-run-'+variant,
            version=v,completed_outer=t,overall=model['overall'],datasets=model['datasets']))
        child=write('endpoints/'+v+'-'+str(t)+'-child.json',dict(schema='opsd-signal-native512-complete-v1',accepted=True,study_id=data['study_id'],
            run_id='fixture-run-'+variant,variant=variant,config=cfg,version=v,completed_outer=t,primary_slots=512,actual_child_exit_and_release=True,
            checkpoint_metadata_unchanged=True,summary=summary,raw_checkpoint=None if raw is None else dict(kind='raw_end_of_outer',manifest=raw)))
        parent=write('endpoints/'+v+'-'+str(t)+'-parent.json',dict(schema='opsd-signal-evaluation-parent-v1',accepted=True,run_id='fixture-run-'+variant,
            config_sha256=cfg['sha256'],completed_outer=t,child=child,raw_checkpoint=raw))
        entry=dict(parent=parent,child=child,summary=summary);evaluations.append(entry);end_refs[name]=entry
        coverage_evals.append(dict(parent=parent,child=child,hits=model['overall']['hits']))
        if t==0:zero=write('common-zero.json',dict(schema='opsd-signal-common-zero-reference-v1',accepted=True,study_id=data['study_id'],parent=parent,child=child))
    coverage_request=write('coverage-request.json',dict(schema='opsd-signal-final-coverage-request-v1',study_manifest=study,groups=coverage_groups,common_zero=zero))
    def packet(kind,value,native_request=None):
        request=write(kind+'-native-request.json',native_request or {})
        value=copy.deepcopy(value);value['request']=request;output=write(kind+'-output.json',value)
        if kind=='train_relationship':
            complete=write('train-native-complete.json',dict(accepted=True,complete_100_outers=True,request=request,outputs={'summary':output}));stdout=write(kind+'-stdout.json',complete)
        else:stdout=write(kind+'-stdout.json',dict(accepted=True,output=output))
        execution=write(kind+'-execution.json',dict(accepted=True,actual_wait_returncode=0,stdout=stdout,source_manifest=manifest,publication=publication,torch_imported=False))
        return dict(source,mode='native_cli_artifact',actual_stdout=stdout,execution=execution,output=output)
    analyses={};analysis_evidence={}
    for kind,value in data['analyses'].items():
        p=packet(kind,value);binding=write(kind+'-binding.json',dict(schema='opsd-signal-analysis-input-binding-v1',input_format_bridge_only=True,
            original_output=p['output'],study_id=data['study_id'],configs=configs,final_checkpoints=finals,formal_evaluations=[e['child'] for e in evaluations]))
        p['input_binding']=binding;analyses[kind]=p;analysis_evidence[kind]=dict(execution=dict(report=p['output']),input_binding=binding)
    coverage=dict(schema='opsd-signal-final-coverage-v1',accepted=True,coverage_accepted=True,final_accepted=True,complete_groups=4,complete_outers=100,
        train_trajectory_slots=256000,formal_native512_slots=4608,study_id=data['study_id'],torch_imported=False,study_manifest=study,request=coverage_request,
        final_checkpoints=finals,outer_rows=outer_rows,evaluations=coverage_evals,
        final_requirements=dict(effect_and_relationship_analysis=dict(accepted=True,evidence=analysis_evidence),
            final_numeric_adam=dict(accepted=True),protected_and_outer_end_backups=dict(accepted=True),
            published_source=dict(accepted=True),full_vocab_scope=dict(accepted=True)))
    cov=packet('coverage',coverage)
    # packet adds a request pointer; formal actual CLI must name the real assembler request.
    coverage['request']=coverage_request;cov['output']=write('coverage-output.json',coverage)
    cov['actual_stdout']=write('coverage-stdout.json',dict(accepted=True,output=cov['output']))
    cov['execution']=write('coverage-execution.json',dict(accepted=True,actual_wait_returncode=0,stdout=cov['actual_stdout'],source_manifest=manifest,publication=publication,torch_imported=False))
    movement=copy.deepcopy(data['movement']);movement.update(configs=configs,endpoints=movement_endpoints,study=study)
    debug=copy.deepcopy(data['debug']);debug.update(study_manifest=study,per_outer=[dict(variant=r['variant'],outer_index=r['outer_index'],
        run_id='fixture-run-'+r['variant'],config=configs[r['variant']],end_checkpoint=r['checkpoint']) for r in outer_rows])
    req=dict(schema='opsd-signal-final-report-request-v1',study_id=data['study_id'],source=source,coverage=cov,coverage_request=coverage_request,
        configs=configs,train_ledgers=ledgers,common_zero=zero,evaluations=evaluations,analysis=analyses,
        debug_retention=write('debug-retention.json',debug),parameter_movement=write('movement.json',movement))
    return write('report-request.json',req),req,write
