"""Root-only documentation update from the actual immutable 3289 snapshot."""
import hashlib
import json
from pathlib import Path

BASE = Path.cwd()
OUT = BASE / "outputs/gigpo-e1-policy-safety"
LIVE = OUT / "CALIBRATION3289-LIVE-20261006T015243Z-v9.json"
live_sha = hashlib.sha256(LIVE.read_bytes()).hexdigest()
assert live_sha == "e33b31692017715edb87b6f34bc840734c9cb70d72be3514a20e8d4b0e1791e8"
actual = json.loads(LIVE.read_text())["actual"]
assert actual["job_id"] == "3289" and "JobState=RUNNING" in actual["scheduler"]["stdout"]
run = actual["actual_run"]
records = {Path(r["path"]).name: r for r in actual["originals"]}
backend = records["actual-vllm-backend-identity.json"]
restore = records["restore25-before-baseline.json"]
assert backend["data"]["status"] == "PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY"
assert restore["data"]["complete_saved_state_equal"] is True
assert restore["data"]["original_adam"] == 500 and restore["data"]["cursor"] == 12800

def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temp = path.with_name(path.name + ".root-update.tmp")
    with temp.open("x") as f:
        f.write(raw)
    temp.replace(path)

state_path = BASE / "work/gigpo-e1-instruct300-state.json"
state = json.loads(state_path.read_text())
fields = ["current_job", "progress_scope", "latest_calibration_job_status", "calibration_status",
          "current_verified_wait_evidence", "progress_note", "last_update_utc"]
state.setdefault("historical_flat_fields_before_actual3289_gpu", {k: state.get(k) for k in fields})
state.update(phase="POLICY_SAFETY_V9_CALIBRATION3289_RUNNING_BASELINE_EVALUATION",
             active_job="3289", job="3289", current_job="3289", run=run,
             actual_iterations=25, new_formal_iterations=0,
             latest_calibration_job_status="RUNNING",
             calibration_status="ACTUAL_FLASH8_AND_FULL25_RESTORE_PRODUCED_BASELINE_EVALUATION_RUNNING",
             last_update_utc=actual["utc"], last_checked_utc=actual["utc"],
             current_verified_wait_evidence=str(LIVE),
             progress_scope="Formal retained25/300; new formal0; isolated calibration3289 contributes0.",
             progress_note="Actual eight-rank Flash metadata and complete25 restore produced; baseline evaluation running. No probability acceptance, new Adam, production threshold, or formal update accepted.")
prep = state["policy_safety"]["backend_v9_preparation"]
prep.update(status=state["calibration_status"], actual_live_path=str(LIVE), actual_live_sha256=live_sha,
            actual_live_utc=actual["utc"], actual_backend_metadata_sha256=backend["sha256"],
            actual_backend_metadata_produced=True, actual_full25_restore_before_baseline_sha256=restore["sha256"],
            actual_restore25_producer_equal=True, probability_accepted=False,
            production_tolerances_selected=False, production_thresholds_selected=False)
state["goal_turn_classification"] = dict(current_turn="progress", verified_handle="Slurm job3289",
    authoritative_evidence=str(LIVE), authoritative_evidence_sha256=live_sha, utc=actual["utc"],
    reason="Actual eight-rank Flash metadata and full25 restore produced; baseline evaluation executing in the same unique calibration job.")
write(state_path, state)

impl_path = OUT / "IMPLEMENTATION-STATUS.json"
impl = json.loads(impl_path.read_text())
old_fields = [k for k in impl if k.startswith("calibration_") and k not in ["calibration_scope"]]
impl.setdefault("historical_calibration_flat_before_actual3289_gpu", {k: impl[k] for k in old_fields})
for key in ["calibration_terminal_evidence", "calibration_terminal_evidence_sha256", "calibration_failure_reason",
            "calibration_failure_review", "calibration_failure_review_sha256", "calibration_terminal_sha256",
            "calibration_result_sha256", "calibration_baseline_evaluation", "calibration_baseline_evaluation_sha256",
            "calibration_restore25_before_baseline_review", "calibration_restore25_before_baseline_review_sha256"]:
    impl[key] = None
impl.update(updated_utc=actual["utc"], phase=state["phase"], calibration_status=state["calibration_status"],
    calibration_job="3289", calibration_run=run, calibration_helpers=str(Path(run).parents[1]/"tools/policy-safety-calibration-52befe9-v9"),
    calibration_controls=str(Path(run).parents[1]/"tools/policy-safety-calibration-controls-52befe9-v9"),
    calibration_live_evidence=str(LIVE), calibration_live_evidence_sha256=live_sha,
    calibration_last_confirmed_utc=actual["utc"], calibration_prob_material_exists=False,
    calibration_failure_review_status=None, calibration_exit_code=None, calibration_cleanup=None,
    calibration_cgroup_removed=None, calibration_baseline_evidence=None, calibration_baseline_evidence_sha256=None,
    calibration_restore25_before_baseline_independent_acceptance=False,
    calibration_restore25_before_baseline_sha256=restore["sha256"],
    calibration_restore25_before_baseline_producer_equal=True,
    calibration_restore25_before_baseline_adam=500, calibration_restore25_before_baseline_scheduler=25,
    calibration_restore25_before_baseline_cursor=12800, calibration_actual_8gpu_factory_initialized=True,
    calibration_backend_metadata_sha256=backend["sha256"], calibration_backend_metadata_produced=True,
    calibration_runtime_gpu_accepted=False, calibration_complete=False,
    probability_identity_accepted=False, actual_retained_iterations=25, new_formal_iterations=0)
write(impl_path, impl)

report = Path("/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct300/REPORT.md")
text = report.read_text()
start, end = "<!-- CURRENT-POLICY-SAFETY-STATUS -->", "<!-- END-CURRENT-POLICY-SAFETY-STATUS -->"
current = f"""{start}
当前状态（北京时间2026-10-06 09:52）：正式训练保留 **25/300轮，新增0轮**。三项防护已接入代码，八卡数值与完整状态诊断通过；正式刹车阈值仍须用健康第25轮的实际轨迹校准。

唯一八卡校准作业 **3289** 在武汉H100集群 **h100-8** 实际RUNNING。当前已产出八卡全部36层实际FlashAttention身份和第25轮完整恢复记录，模型、Adam500、scheduler25及数据游标12800与原保存状态一致；正在重新执行基准评估，尚未接受概率一致性或新增参数更新。真实目录 `{run}`。校准不计入训练300轮；生产阈值与概率容差未选定，正式续训尚未启动。

恢复来源：`{restore['data']['restore_checkpoint']}`，同512题F1 **32.3198%**、EM **26.1719%**。源码 `52befe968c856162d1bc79cd0171bb38309f4f9e`。证据目录 `{OUT}`，实际快照SHA `{live_sha}`。全部历史代码、日志、轨迹与checkpoint保留，历史回滚分支轮数不拼入当前进度。
{end}"""
assert text.count(start) == text.count(end) == 1
report.write_text(text[:text.index(start)] + current + text[text.index(end)+len(end):])
design = OUT / "DESIGN.md"
text = design.read_text()
start, end = "<!-- CURRENT-IMPLEMENTATION -->", "<!-- END-CURRENT-IMPLEMENTATION -->"
current = f"""{start}
当前实施状态（北京时间2026-10-06 09:52）：正式保留25/300，新增0。三项防护源52befe9已提交，八卡数值/分布式/完整恢复诊断3267已独立接受。唯一校准3289实际RUNNING，已产出八卡36层实际Flash后端及第25轮完整恢复原件，正在基准评估。
实际概率一致性和10Adam校准尚未接受；生产KL/尾部阈值与概率容差为空，禁止正式续训。v8未提交、3281失败及全部原件保留，校准和历史回滚轮数不计当前25/300。以下早期快照保留为历史。
{end}"""
assert text.count(start) == text.count(end) == 1
design.write_text(text[:text.index(start)] + current + text[text.index(end)+len(end):])
print(json.dumps(dict(updated=[str(state_path), str(impl_path), str(report), str(design)], actual_evidence=str(LIVE), sha256=live_sha)))
