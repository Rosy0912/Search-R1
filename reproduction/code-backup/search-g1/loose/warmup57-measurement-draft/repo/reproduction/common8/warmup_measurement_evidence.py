"""Typed two-outer warmup grammar; old diagnostic arithmetic stays unchanged."""
import copy
from collections.abc import Mapping,Sequence
from reproduction.common8 import safety_calibration_evidence as arithmetic
from reproduction.common8.formal_contract import consumed_lr
from reproduction.common8.warmup_measurement_plan import ROLE,OUTER_COUNT,WRITES,MONITORS


def verify_segment(rows_by_rank,binding):
    need=arithmetic._require
    need(isinstance(binding,Mapping) and type(binding['schema_version']) is int and
         binding['schema_version']==2 and binding['role']==ROLE,'Explicit new measurement schema/role required')
    need(binding['outer_count']==OUTER_COUNT and binding['initial_optimizer_writes']==0,'Only fresh two-outer phase allowed')
    arithmetic._hash(binding['calibration_sha256'],'new plan binding')
    need(isinstance(rows_by_rank,Mapping) and set(rows_by_rank)==set(range(8)) and
         all(type(rank) is int for rank in rows_by_rank),'Exactly eight actual ranks required')
    enforcement=dict(negative_tail='observe_only',old_policy_kl='enforced',numerical_finite='enforced',dual_clip='enforced')
    observations={};identities={outer:{} for outer in (1,2)};gates={}
    for rank in range(8):
        rows=rows_by_rank[rank]
        need(isinstance(rows,Sequence) and not isinstance(rows,(str,bytes)) and len(rows)==122,
             'Missing/extra actual monitor/write/budget events')
        cursor=0
        for outer in (1,2):
            identity=None;frozen=None
            for mini in range(20):
                prior=(outer-1)*20+mini
                before,write,after=rows[cursor:cursor+3];cursor+=3
                for row,stage,writes in ((before,'before_minibatch',prior),(after,'after_minibatch',prior+1)):
                    arithmetic._row(row,rank,binding,'POLICY_MONITOR');arithmetic._context(row,outer,mini,prior)
                    need(row['stage']==stage and row['optimizer_writes']==writes,'Actual pre/post order differs')
                    need(row['calibration_sha256']==binding['calibration_sha256'] and row['execution_role']==ROLE and
                         row['enforcement']==enforcement,'Actual new-role brakes/plan differ')
                    need(row['objective_weight_definition']==arithmetic.OBJECTIVE_WEIGHT_DEFINITION and
                         row['monitoring_scope']==arithmetic.MONITORING_SCOPE and row['rejection_count']==0,
                         'Wrong original objective monitoring or prior rejection')
                    arithmetic._identity(row['monitor_identity'])
                    if identity is None:
                        identity=copy.deepcopy(row['monitor_identity']);identities[outer][rank]=identity
                    need(row['monitor_identity']==identity,'Within-outer frozen monitor identity changed')
                    statistics=arithmetic._statistics(row)
                    need(row['allowed'] is True and statistics['kl_estimate_complete'] is True and
                         statistics['sampled_old_policy_kl']<=.01,'Actual KL/numerical brake failed')
                    constant=tuple(statistics[name] for name in ('effective_weight_mass','negative_effective_weight_mass',
                                                               'effective_token_count','negative_token_count'))
                    if frozen is None:frozen=constant
                    need(frozen==constant,'Frozen masks/advantages changed within outer')
                    key=(outer,mini,stage)
                    if key not in observations:
                        observations[key]=dict(run_id=binding['run_id'],context=copy.deepcopy(row['context']),
                            stage=stage,optimizer_writes=writes,calibration_sha256=binding['calibration_sha256'],
                            execution_role=ROLE,enforcement=copy.deepcopy(enforcement),**statistics)
                    else:
                        need(all(observations[key][name]==statistics[name] for name in arithmetic.GLOBAL_FIELDS),
                             'Globally reduced statistics disagree across ranks')
                arithmetic._row(write,rank,binding,'OPTIMIZER_WRITE');arithmetic._context(write,outer,mini,prior)
                for event in (before,write,after):
                    gate=event['context']['behavior_probability_gate_sha256'];arithmetic._hash(gate,'actual all2560 gate')
                    if outer not in gates:gates[outer]=gate
                    need(gate==gates[outer],'Within-outer/across-rank behavior gate changed')
                need(type(write['write_count']) is int and write['write_count']==prior+1,'Missing actual optimizer write')
                rates=write['learning_rates'];need(isinstance(rates,list) and bool(rates),'Missing actual LR')
                need(all(arithmetic._finite(lr,'learning_rate')==consumed_lr(outer) for lr in rates),'Wrong real consumed warmup LR')
                for name in ('gradient_norm_before','gradient_norm_after'):arithmetic._finite(write[name],name)
                need(write['gradient_norm_after']<=1.+arithmetic.GRADIENT_CLIP_FLOAT32_TOLERANCE,'Post-clip norm exceeded clip1')
            budget=rows[cursor];cursor+=1;arithmetic._row(budget,rank,binding,'OUTER_BUDGET')
            for name,value in dict(logical_update=outer,local_trajectories=320,local_mini=16,
                                   local_micro=2,optimizer_writes=20,cumulative_writes=outer*20).items():
                need(type(budget[name]) is int and budget[name]==value,'Wrong actual outer budget '+name)
    for observation in observations.values():
        outer=observation['context']['logical_update']
        need(observation['effective_token_count']==sum(sum(x['actor_token_lengths']) for x in identities[outer].values()),
             'Global actual token count differs')
        arithmetic._same_arithmetic(observation['effective_weight_mass'],16.,'original objective global weight')
    for outer in (1,2):
        for mini in range(1,20):arithmetic._adjacent(observations[(outer,mini-1,'after_minibatch')],observations[(outer,mini,'before_minibatch')])
    need(len(observations)==MONITORS,'Exactly80 deduplicated global pre/post observations required')
    return dict(schema_version=2,role='warmup57_measurement_structural_verification',execution_role=ROLE,
        run_id=binding['run_id'],calibration_sha256=binding['calibration_sha256'],outer_count=OUTER_COUNT,
        optimizer_writes_per_rank=WRITES,global_monitors=list(observations.values()),
        monitor_identities={str(outer):{str(rank):identity for rank,identity in values.items()}
                            for outer,values in identities.items()},enforcement=enforcement,
        behavior_gate_sha256={str(outer):value for outer,value in gates.items()},
        production_admission=False)
