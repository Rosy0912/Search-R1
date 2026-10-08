import copy,hashlib,importlib.util,json,os
from pathlib import Path
import pytest
D=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('candidate_gate',D/'checkpoint_gate_candidate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
ACTUAL=D.parent/'formal3198-first5-observed-20261004T134348Z/run/train.log'
MARKER='OPSD_E1_OUTER_JSON=';DRIVER=469564;WORKER=470520;WARNPATH=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')
def parse(line,workers=(WORKER,)):return g.parse_log_marker(line,MARKER,driver_pid=DRIVER,job_id='3198',worker_pids=workers,checkpoint_warning_path=WARNPATH)
def payload():return {'driver_pid':DRIVER,'job_id':'3198','actual_outer_updates':4,'global_step':4,'metrics':{}}
def warning():return f"\x1b[36m(WorkerDict pid={WORKER})\x1b[0m {WARNPATH}:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.\x1b[32m [repeated 7x across cluster]\x1b[0m"
def test_actual3198_row4_interleaving_exact_object_not_failedjob_acceptance():
 line=next(v for v in ACTUAL.read_text().splitlines() if MARKER in v and 'torch.cpu.amp.autocast' in v)
 expected,end=json.JSONDecoder().raw_decode(line.split(MARKER,1)[1]);assert expected['actual_outer_updates']==4
 with pytest.raises(json.JSONDecodeError):json.loads(line.split(MARKER,1)[1])
 assert parse(line)==expected
@pytest.mark.parametrize('prefix',['','  ',f'(main_task pid={DRIVER}) ',f'\x1b[36m(main_task pid={DRIVER})\x1b[0m '])
def test_plain_and_typed_driver_prefix(prefix):assert parse(prefix+MARKER+json.dumps(payload())+'  ')==payload()
@pytest.mark.parametrize('suffix',[warning(),warning().replace('\x1b[36m','').replace('\x1b[32m','').replace('\x1b[0m',''),warning().split('\x1b[32m')[0],warning().replace('7x','1x'),warning().replace('7x','8x')])
def test_exact_typed_warning_only(suffix):assert parse(MARKER+json.dumps(payload())+suffix)==payload()
@pytest.mark.parametrize('suffix',[' arbitrary', '{}','\n{}','OPSD_E1_OUTER_JSON={}', 'FULL_STATE_RANK_JSON={}',warning()+'{}',warning()+' extra',warning().replace(str(WORKER),'999999'),warning().replace('1399','1400'),warning().replace('checkpoint.py','foreign.py'),warning().replace('7x','9x'),warning().replace('7x','0x'),warning().replace('FutureWarning','RuntimeWarning'),warning().replace(' instead.',' instead. {}'),warning().replace('\x1b[0m','',1)])
def test_arbitrary_second_event_unknown_warning_pid_and_ansi_rejected(suffix):
 with pytest.raises(ValueError):parse(MARKER+json.dumps(payload())+suffix)
@pytest.mark.parametrize('raw',['{"x":1,"x":2}','{"x":{"y":1,"y":2}}','{"x":NaN}','{"x":Infinity}','{"x":1e309}','{"x":{"y":-1e309}}','[]','1'])
def test_duplicate_nonfinite_and_nonobject_rejected(raw):
 with pytest.raises(ValueError):parse(MARKER+raw)
@pytest.mark.parametrize('field,value',[('driver_pid',1),('job_id','3196')])
def test_wrong_body_driver_job_rejected(field,value):
 p=payload();p[field]=value
 with pytest.raises(ValueError):parse(MARKER+json.dumps(p))
def test_foreign_driver_and_unrecognized_prefix_rejected():
 for prefix in ['hello ', '(main_task pid=1) ', f'\x1b[36m(main_task pid={DRIVER}) ',f'(WorkerDict pid={WORKER}) ']:
  with pytest.raises(ValueError):parse(prefix+MARKER+json.dumps(payload()))
def test_actual_manifest_pids_all_sha_owned_and_no_alias(tmp_path):
 files={}
 for rank in range(8):
  pt=tmp_path/f'rank_{rank}.pt';pt.write_bytes(b'CPU_PARSER_UNIT_NOT_NATIVE_STATE');files[f'actor/training_state/rank_{rank}.pt']={'path':str(pt),'sha256':g.sha256(pt)}
  p=tmp_path/f'rank_{rank}.json';p.write_text(json.dumps({'version':3,'algorithm':'pure_opsd','status':'EXACT_FULL_STATE_RELOAD_OK','model_state_dict_type':'SHARDED_STATE_DICT','model_dtensor_entries':1,'path':str(pt),'sha256':g.sha256(pt),'state_sha256':g.sha256(pt),'rank':rank,'world_size':8,'step':5,'pid':100+rank}));files[f'actor/training_state/rank_{rank}.json']={'path':str(p),'sha256':g.sha256(p)}
 m={'world_size':8,'files':files};assert g.manifest_worker_pids(m,tmp_path)==set(range(100,108))
 p=tmp_path/'rank_7.json';row=json.loads(p.read_text());row['pid']=100;p.write_text(json.dumps(row));files['actor/training_state/rank_7.json']['sha256']=g.sha256(p)
 with pytest.raises(ValueError):g.manifest_worker_pids(m,tmp_path)
 p.write_text('{}')
 with pytest.raises(ValueError):g.manifest_worker_pids(m,tmp_path)


def test_all_eight_typed_native_writer_records_and_transport_pid_mismatch():
 workers=set(range(100,108))
 for pid in sorted(workers):
  row={'pid':pid,'rank':pid-100,'step':5,'world_size':8}
  line=f'\x1b[36m(WorkerDict pid={pid})\x1b[0m FULL_STATE_RANK_JSON='+json.dumps(row)
  assert g.parse_log_marker(line,'FULL_STATE_RANK_JSON=',driver_pid=DRIVER,job_id='3198',worker_pids=workers)==row
  row['pid']=999
  with pytest.raises(ValueError):g.parse_log_marker(line.split('FULL_STATE_RANK_JSON=')[0]+'FULL_STATE_RANK_JSON='+json.dumps(row),'FULL_STATE_RANK_JSON=',driver_pid=DRIVER,job_id='3198',worker_pids=workers)
