import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import pathlib,json,struct,math,time
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-grpo100-common-v5-r1');a=json.loads((R/'actual-outer.json').read_text());assert 5 in a['actual_completed_outer_steps']
p=R/'checkpoints/actor/global_step_5';idx=json.loads((p/'model.safetensors.index.json').read_text());wm=idx['weight_map'];assert len(wm)==771
c=json.loads((R/'evidence/effective-runtime-config.json').read_text());base=pathlib.Path(c['base_model']);orig=json.loads((base/'model.safetensors.index.json').read_text());assert set(wm)==set(orig['weight_map'])
seen={};files=[];nbytes=0;dtypebytes={'F32':4,'BF16':2,'F16':2,'I64':8,'I32':4}
for name in sorted(set(wm.values())):
 f=p/name
 with f.open('rb') as stream:
  hlen=struct.unpack('<Q',stream.read(8))[0];assert 0<hlen<32*1024*1024;header=json.loads(stream.read(hlen))
 end=0
 for key,val in header.items():
  if key=='__metadata__':continue
  assert key not in seen and wm[key]==name and val['dtype']=='F32';lo,hi=val['data_offsets'];assert hi-lo==math.prod(val['shape'])*dtypebytes[val['dtype']];seen[key]=val;end=max(end,hi);nbytes+=hi-lo
 assert f.stat().st_size==8+hlen+end
 files.append({'name':name,'size':f.stat().st_size,'header_bytes':hlen,'tensor_count':len(header)-('__metadata__' in header)})
assert set(seen)==set(wm) and idx['metadata']['total_size']==nbytes
config=json.loads((p/'config.json').read_text());assert config['model_type']=='qwen2'
required=['tokenizer.json','tokenizer_config.json','generation_config.json','config.json'];assert all((p/name).is_file() for name in required)
print(json.dumps({'scope':'LIVE_TRAINING_CHECKPOINT5_COMPLETE_FILES_AND_HEADERS_NOT_FULL_WEIGHT_HASH_OR_FINITE_SCAN','epoch':time.time(),'job_id':3228,'checkpoint':str(p),'actual_completed':a['actual_completed_outer_steps'],'tensor_count':len(seen),'tensor_bytes':nbytes,'all_dtype':'F32','source_index_keys_equal':True,'files':files,'small_files':{name:(p/name).stat().st_size for name in required}}))
'''
r=send(source,attempts=3,retry_safe=True,timeout=120);pathlib.Path('outputs/standard3228-checkpoint5-live-presence.json').write_bytes(r.stdout);print(r.stdout.decode());print(r.stderr.decode());sys.exit(r.returncode)
