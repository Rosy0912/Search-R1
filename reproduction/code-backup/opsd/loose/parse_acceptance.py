import json,re,sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
rows=[]
for line in v['metric_lines']:
 d={k:float(x) for part in line.split(' - ') if ':' in part for k,x in [part.rsplit(':',1)]}
 if d['step']<1: continue
 rank_checks={}
 for rank in range(8):
  expected={'actor/samples':320,'actor/optimizer_steps':10,'actor/skipped_empty_minibatches':0,'actor/old_log_probs_present':1,'actor/old_log_probs_finite':1,'actor/loss_mask_valid':1,'actor/opsd_loss_finite':1,'actor/grad_norm_finite':1,'actor/allocator_is_native':1,'actor/allocator_configuration_is_default':1,'teacher/update_device_is_cuda_per_rank':1,'teacher/storage_is_cpu_per_rank':1,'teacher/all_parameter_dtype_is_fp32_per_rank':1,'teacher/attention_is_eager_per_rank':1,'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0,'teacher/cuda_update_entries_cumulative_per_rank':int(d['step'])}
  keys={k+'/rank_'+str(rank):x for k,x in expected.items()}
  missing=[k for k in keys if k not in d]
  wrong={k:{'actual':d.get(k),'expected':x} for k,x in keys.items() if k in d and d[k]!=x}
  rank_checks[str(rank)]={'missing':missing,'wrong':wrong,'passed':not missing and not wrong}
 rows.append({'step':int(d['step']),'timing':{k:x for k,x in d.items() if k.startswith('timing_s/')},'rank_checks':rank_checks,'metrics':d})
result={'sample_utc':v['sample_utc'],'job_id':v['job_id'],'outer_updates_completed':len(rows),'steps':[x['step'] for x in rows],'outer_rows':rows,'console_values_rounded_3_decimals':True,'training_finished':v['training_finished'],'exit':v['exit']}
out=Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem+'-parsed.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(result['sample_utc'],result['steps'],'finished',result['training_finished'])
for row in rows: print('step',row['step'],'timing',row['timing'],'allrankpass',all(x['passed'] for x in row['rank_checks'].values()),'failed',{k:x for k,x in row['rank_checks'].items() if not x['passed']})
