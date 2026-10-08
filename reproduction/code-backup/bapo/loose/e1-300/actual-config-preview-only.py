import pathlib,os,subprocess,json,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004');env=dict(os.environ)
for k,v in dict(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(r/'repo'),BAPO_E1_DATA=str(r/'common-data'),BAPO_E1_MODEL=str(r/'assets/qwen2.5-3b-instruct-aa8e725'),BAPO_E1_RUN=str(r/'logs/prospective-config-only-not-run'),BAPO_E1_DATA_MANIFEST_SHA256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',BAPO_E1_RETRIEVER_URL='http://127.0.0.1:18083/retrieve',RAY_DEDUP_LOGS='0').items():env[k]=v
start=time.monotonic()
with(r/'logs/prospective-config.yaml').open('x')as out,(r/'logs/prospective-config.stderr').open('x')as err:
 p=subprocess.run([str(r/'env/bin/python'),'-m','search_r1.baselines.main','--config-name','ppo_bapo_e1_300','baselines.algorithm_source_commit='+ (r/'source-expected.txt').read_text().strip(),'--cfg','job','--resolve'],cwd=r/'repo',env=env,stdout=out,stderr=err)
record={'status':'PROSPECTIVE_CONFIG_ONLY_NOT_ACTUAL_RUN','exit_code':p.returncode,'wall_seconds':time.monotonic()-start,'cuda_visible_devices':'','source_commit':(r/'source-expected.txt').read_text().strip()}
(r/'logs/prospective-config-record.json').write_text(json.dumps(record));print(json.dumps(record));raise SystemExit(p.returncode)
