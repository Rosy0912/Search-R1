"""Read pinned offline W&B history; never contacts a server."""
import argparse,json,pathlib
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
assert not a.output.exists()
files=list(a.run.rglob('*.wandb'));assert len(files)==1, 'Need one own offline run'
d=DataStore();d.open_for_scan(str(files[0]));rows=[]
while True:
 b=d.scan_data()
 if b is None:break
 r=pb.Record();r.ParseFromString(b)
 if r.HasField('history'):
  row={('/'.join(x.nested_key) if x.nested_key else x.key):json.loads(x.value_json) for x in r.history.item}
  if '_step' not in row and r.history.HasField('step'):row['_step']=r.history.step.num
  rows.append(row)
assert rows, 'No actual metric records'
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as out:json.dump(dict(source_offline_file=str(files[0]),history=rows),out,indent=2,allow_nan=False)
print('OFFLINE_METRICS_EXTRACTED',len(rows),str(a.output))
