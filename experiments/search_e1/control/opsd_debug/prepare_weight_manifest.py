"""CPU-only fresh source/adapter hashing against the accepted source transfer."""
import argparse,json,time
from pathlib import Path
from common import file_sha,stat_identity,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 proof=a.base/'receipts/raw-epoch1-cpu-merge-20261008.json'
 assert file_sha(proof)=='cec8bcf50cd7fb611473bd3fe16a24c679626493a37c99a584c1364f4057966c'
 old=json.loads(proof.read_text());assert old['status']=='ACCEPTED'
 expected=old['inputs_before']['source_transfer']['files']
 model=a.base/'assets/archive100';files=[]
 assert len(expected)==12
 for relative,v in sorted(expected.items()):
  f=model/relative;s=stat_identity(f);h=file_sha(f)
  assert s==stat_identity(f) and h==v['sha256'] and s['bytes']==v['bytes'],relative
  files.append(dict(s,relative_path=relative,sha256=h,sha256_origin='Fresh CPU read compared to accepted archive100 transfer'))
 adapters={}
 for label,directory in [('source100','best_adapter'),('update1','adapter_update_1'),('update5','adapter_update_5'),('epoch1','adapter_epoch_1')]:
  d=a.base/'runs/opsd'/directory;m=d/'adapter_model.safetensors';c=d/'adapter_config.json';ms=stat_identity(m);cs=stat_identity(c)
  adapters[label]=dict(path=str(d),adapter_model_sha256=file_sha(m),adapter_config_sha256=file_sha(c),model_stat=ms,config_stat=cs)
  assert ms==stat_identity(m) and cs==stat_identity(c)
 assert adapters['source100']['adapter_model_sha256']=='8cbebc6cb10203bf93a2c8b7271d31020ab1d1e7f1958a6f94f5890af3206c3d'
 assert adapters['epoch1']['adapter_model_sha256']=='8eef75aafa06eb98dc21a6b0ab85195800fc373692d633faa0ecf461fa45afd0'
 result=dict(created_unix=time.time(),source100=dict(path=str(model),config_sha256=file_sha(model/'config.json'),files=files),adapters=adapters,proof=dict(path=str(proof),sha256=file_sha(proof)),scope='No CUDA imports or model updates; all source/adapter bytes freshly hashed')
 atomic_json(a.output,result);print(json.dumps(dict(output=str(a.output),source_files=len(files),adapter_count=len(adapters))))
if __name__=='__main__':main()
