import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import pathlib,json,hashlib
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-grpo100-common-v5-r1/evidence/actor-resource-audit');out=[]
for rank in range(8):
 p=R/f'actor-updates-rank{rank}-outer55.json';raw=p.read_bytes();d=json.loads(raw);u=d['updates'][-1];assert str(d['job'])=='3228' and d['rank']==rank and u['actual_outer_update_ordinal']==55
 out.append({'rank':rank,'original_path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'local_rows':u['local_rows'],'micro_forwards':len(u['micro_forward_records']),'optimizer_steps':len(u['optimizer_step_records']),'all_optimizer_finite':all(x['finite'] for x in u['optimizer_step_records']),'learning_rates':sorted(set(v for x in u['optimizer_step_records'] for v in x['actual_learning_rates'])),'parameter_sample_before':u['parameter_sample_before'],'parameter_sample_after':u['parameter_sample_after'],'parameter_sample_changed':u['parameter_sample_before']!=u['parameter_sample_after'],'sample_scope':u['sample_scope'],'allocator_oom_max':max(x['pre_backward_memory']['original_cuda_memory_stats']['num_ooms'] for x in u['micro_forward_records'])})
print(json.dumps({'scope':'READ_ONLY_EXISTING_ACTUAL_RANK_LOGS_NOT_FULLMODEL_DIFF_OR_INDEPENDENT_AUDIT','job_id':3228,'outer':55,'ranks':out}))
'''
r=send(source,attempts=3,retry_safe=True,timeout=120);pathlib.Path('outputs/standard3228-outer55-during-training-rank-read.json').write_bytes(r.stdout);print(r.stdout.decode());print(r.stderr.decode());sys.exit(r.returncode)
