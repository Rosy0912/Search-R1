from pathlib import Path
import os,sys,json,hashlib
root=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');repo=root/'repo'
sys.path.insert(0,str(repo))
from search_r1.baselines.baseline_alignment_data import prepared_manifest, verify_order_manifest, object_hash
common=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data')
assert hashlib.sha256((common/'manifest.json').read_bytes()).hexdigest()=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
man=prepared_manifest([str(common/'train.parquet')],[str(common/'validation.parquet')],[str(common/'final.parquet')],[str(common/'stage.parquet')],stage_train_overlap_policy='disjoint')
assert [man['splits'][k]['rows']for k in ['train','validation','test','stage']]==[169028,512,51713,64]
order_path=root/'inputs/baseline-observed-order.json';order=json.loads(order_path.read_text());verify_order_manifest(order,man['splits']['train']['question_ids'],512)
man.update(original_common_manifest=str(common/'manifest.json'),original_common_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',baseline_order_manifest=str(order_path),baseline_order_manifest_sha256=hashlib.sha256(order_path.read_bytes()).hexdigest(),reference_scope='NEW_COMMON169028_REFERENCE_NOT_HISTORICAL169103',source_commit='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84')
path=root/'inputs/baseline-aligned-data.json';raw=(json.dumps(man,sort_keys=True,ensure_ascii=False)+'\n').encode()
with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'splits':{k:{'rows':v['rows'],'semantic_sha256':v['semantic_sha256']}for k,v in man['splits'].items()},'actual_baseline_order':man['baseline_order_manifest_sha256'],'batches':len(order['actual_batches']),'first_batch_sha256':order['actual_batches'][0]['batch_sha256'] if 'batch_sha256'in order['actual_batches'][0]else object_hash(order['actual_batches'][0]),'scope':'ACTUAL_FIXED_COMMON_DATA_AND_REFERENCE_ORDER_VERIFIED'}))
