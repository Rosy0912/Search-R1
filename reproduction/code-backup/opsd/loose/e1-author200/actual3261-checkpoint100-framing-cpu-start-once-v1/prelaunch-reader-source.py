
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
tools=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-90c2695a7f46')
sys.path.insert(0,str(tools))
import checkpoint_overlap as c,checkpoint_cpu_pipeline as pipeline,author200_observation as shared
step=int(sys.argv[1]);pin='90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
assert step in (75,100,125,150,175,200)
c.tools(pin);allocation=c.allocation()
prefixes={};originals={};stats={}
for name,p in [('driver',Path(os.readlink('/proc/1533312/fd/1'))),('training',c.RUN/'train.log')]:
 c.physical(p);before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  opened=os.fstat(f.fileno());raw=f.read(before.st_size);opened_after=os.fstat(f.fileno())
 after=p.stat();identity=('st_dev','st_ino','st_uid')
 assert all(tuple(getattr(v,k) for k in identity)==tuple(getattr(before,k) for k in identity) for v in (opened,opened_after,after))
 assert len(raw)==before.st_size and opened_after.st_size>=before.st_size and after.st_size>=before.st_size
 closed=raw[:raw.rfind(b'\n')+1];assert closed
 prefixes[name]={'path':str(p),'bytes':len(closed),'sha256':hashlib.sha256(closed).hexdigest()};originals[name]=closed
 stats[name]={'device':before.st_dev,'inode':before.st_ino,'uid':before.st_uid,'captured_size':before.st_size,'size_after':after.st_size,'excluded_tail_bytes':len(raw)-len(closed)}
config={'checkpoint_step':step,'log_prefixes':prefixes}
command=pipeline.observer_command(config,c.RUN/'evidence'/('checkpoint-'+str(step)+'-cpu-inspection-once-v1'))
parser=argparse.ArgumentParser();shared.parser_anchors(parser);parser.add_argument('--checkpoint-step',type=int);parser.add_argument('--output-directory')
args=parser.parse_args(command[2:]);context=shared.load_accepted_context(args)
driver=shared.parse_records(originals['driver'],context);aggregate=shared.parse_records(originals['training'],context)
join=shared.join_log_prefixes(driver,aggregate,context,step);assert join['aggregate_last_outer']>=step
shared.check_evaluation_sequence(driver,join['authoritative_last_outer'])
actor=c.RUN/'checkpoints/actor'/('global_step_'+str(step))
members=[actor/n for n in ('config.json','generation_config.json','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json')]
members += [actor/'training_state'/('rank_'+str(r)+s) for r in range(8) for s in ('.pt','.json')]
members += [c.RUN/'full_state'/('global_step_'+str(step))/n for n in ('driver.pt','driver.json')]+[c.RUN/'checkpoints/algorithm'/('global_step_'+str(step))/'state.json']
assert len(members)==31
for p in members:c.physical(p)
marker=c.RUN/'evidence'/('checkpoint-cpu-'+str(step)+'-launch.json');out=c.RUN/'evidence'/('checkpoint-'+str(step)+'-cpu-inspection-once-v1')
assert not marker.exists() and not marker.is_symlink() and not out.exists() and not out.is_symlink()
allocation_after=c.allocation()
print(json.dumps({'scope':'ACTUAL_CLOSED_PREFIX_AND31_MEMBERS_PRELAUNCH_NOT_FULL_CONTENT_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'tools_manifest_sha256':pin,'source_commit':c.SOURCE,'checkpoint_step':step,'log_prefixes':prefixes,'prefix_stats':stats,'log_join':join,
 'allocation_before':allocation,'allocation_after':allocation_after,'expected_members':list(map(str,members)),
 'latest_outer':driver['outers'][-1],'ready_for_one_cpu_inspection':True},sort_keys=True))
