"""Fresh streaming SHA checks of the exact TOP-D initial weights and fixed split."""
import hashlib,json,time
from pathlib import Path
BASE=Path('/data/user/jxu957/experiments/topd-20261008')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    protocol=json.loads((BASE/'outputs/training-preparation/hpc3-topd200-protocol-draft.json').read_text())
    model=Path(protocol['base_model']);data=Path(protocol['common_data']['directory'])
    manifest=json.loads((model/'VERIFIED-MODEL-MANIFEST.json').read_text())
    assert manifest['repo']=='Qwen/Qwen2.5-3B-Instruct'
    assert manifest['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    records=[]
    for item in manifest['files']:
        p=model/item['file'];actual=sha(p)
        assert p.stat().st_size==item['bytes'] and actual==item['sha256'],str(p)
        records.append({'path':str(p),'resolved':str(p.resolve()),'bytes':p.stat().st_size,'sha256':actual})
    for item in protocol['input_bindings']:
        p=Path(item['path']);actual=sha(p)
        assert actual==item['sha256'],str(p)
        records.append({'path':str(p),'resolved':str(p.resolve()),'bytes':p.stat().st_size,'sha256':actual})
    assert sha(data/'manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    result={'status':'FRESH_MODEL_AND_FIXED_DATA_SHA_PASS','epoch':time.time(),'model':str(model),'data':str(data),'records':records,'rows_expected':{'train':169028,'validation':512,'final':51713},'actual_parquet_rows_pending':True}
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
