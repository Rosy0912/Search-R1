import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import pathlib,hashlib,sys,time,json
path=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/check_real_checkpoint.py')
raw=path.read_bytes();expected='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
assert hashlib.sha256(raw).hexdigest()==expected
source=raw.decode();assert source.count('torch.set_num_threads(8)')==1
# Reviewer-only CPU resource cap; do not mutate the frozen checker file.
source=source.replace('torch.set_num_threads(8)','torch.set_num_threads(4)')
output=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/reviewer-recovery3234-hf120-'+str(time.time_ns())+'.json')
sys.argv=[str(path),'--base','/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725','--checkpoint','/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output/checkpoints/tree-recovery300/actor/global_step_120','--output',str(output)]
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
report=json.loads(output.read_text());report.update(reviewer_original_checker_sha256=expected,reviewer_cpu_threads=4,reviewer_only_inmemory_resource_cap='set_num_threads8->4; file unmodified',reviewer_remote_report=str(output))
print('REVIEWER_HF120_RESULT '+json.dumps(report),flush=True)
