import json
from pathlib import Path
p=Path(__file__).parent
f=sorted(p.glob("runtime-watch-3328-observation-*.json"))[-1]
s=json.loads(f.read_text())
sl=s.get("slurm_identity",{}).get("fields",{})
r=s["files"].get("resource-samples.jsonl",{}).get("recent",[])
g=r[-1].get("gpu_util_per_card",[]) if r else []
cpu=s.get("own_cpu_delta",{})
print(json.dumps({"at":s["observed_at"],"state":sl.get("JobState"),"runtime":sl.get("RunTime"),"rollout_files":s["files"]["output/training-rollout-cost/attempt-*.json"]["count"],"rank_files":s["files"]["output/fresh-state/outer-*-attempt-*-rank??.json"]["count"],"attempt_files":s["files"]["checkpoints/training_state/attempt-results/attempt-*.json"]["count"],"policy_latest_valid":s["policy_safety_events"].get("valid_json_file_count",s["policy_safety_events"].get("valid_selected_json_file_count")),"policy_invalid":s["policy_safety_events"].get("invalid_json_file_count",s["policy_safety_events"].get("invalid_selected_json_file_count")),"policy_scope":s["policy_safety_events"].get("statistics_scope"),"rolling_directories":s.get("rolling_completed_native_metadata",{}).get("directories"),"errors":s.get("anchored_log_metadata",{}).get("stage.log",{}).get("exception_classes"),"stage_bytes":s.get("anchored_log_metadata",{}).get("stage.log",{}).get("size_bytes"),"retriever_bytes":s.get("anchored_log_metadata",{}).get("retriever.log",{}).get("size_bytes"),"GPU_util_range":[min(g),max(g)] if g else None,"process_cpu_delta_seconds":round((cpu.get("utime_delta_jiffies",0)+cpu.get("stime_delta_jiffies",0))/cpu.get("clock_ticks_per_second",100),2)},ensure_ascii=False))
