import argparse,json,pathlib,sys
sys.dont_write_bytecode=True
ROOT=pathlib.Path("/dataset/shaoshitong/search-r1-rl/topd")
p=argparse.ArgumentParser();p.add_argument("--teacher",required=True);p.add_argument("--socket-manifest",required=True);p.add_argument("--run-directory",required=True);a=p.parse_args()
for value in (a.teacher,a.socket_manifest,a.run_directory):
 if ROOT not in pathlib.Path(value).resolve().parents:raise ValueError("Only same-node owned paths")
r=json.loads((pathlib.Path(__file__).parent/"recipe.json").read_text())
command=[str(ROOT/"env/bin/python"),"-m","verl.trainer.main_ppo","--config-name=ppo_baselines"]+r["overrides"]+["baselines.distillation.teacher_path="+a.teacher,"baselines.topd.socket_manifest="+a.socket_manifest,"trainer.default_local_dir="+a.run_directory+"/checkpoints"]
print(json.dumps({"command":command,"cwd":str(ROOT/"tools/topd-instruct300-source-v1"),"teacher_required":"actual GRPO100 full HF; pending until delivered","does_not_submit":True}))
