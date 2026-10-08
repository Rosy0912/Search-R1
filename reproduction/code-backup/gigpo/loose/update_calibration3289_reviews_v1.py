"""Record completed narrow reviews while the unique calibration continues."""
import hashlib
import json
from pathlib import Path

base = Path.cwd()
out = base / "outputs/gigpo-e1-policy-safety"
proofs = {
    "backend": ("CALIBRATION3289-BACKEND-METADATA-INDEPENDENT-REVIEW-v1.json", "db8e126e415da656fa9684d4b69e2d203361651d5857227756d69ce5e7fdced3"),
    "restore": ("CALIBRATION3289-RESTORE25-INDEPENDENT-REVIEW-v1.json", "76d7ceae596144c1cca7a88c1ca9c82279df2e29e008055b6a2055e6c45d585b"),
    "formal_template": ("FORMAL-DEPLOYMENT-TEMPLATE-INDEPENDENT-REVIEW-v1.json", "643ccbd88fa2b8473b5183c71914fbf1aeebe39312facec7bb7516556129b377"),
    "formal_implementation": ("FORMAL-DEPLOYMENT-TEMPLATE-IMPLEMENTATION-v1.json", "ff97e439ba02d208120bcf20f74ea6a7dbb15c374de498010be4a79913c112ec"),
    "formal_scope": ("FORMAL-DEPLOYMENT-PHYSICAL-SCOPE-CLARIFICATION-v1.json", "86f4d13dc5ae09c27716fdc83a2c8867a8f49734d7a3414175840e0892c487ed"),
}
verified = {}
for key, (name, sha) in proofs.items():
    path = out / name
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    verified[key] = dict(path=str(path), bytes=len(raw), sha256=sha)
live = out / "CALIBRATION3289-LIVE-20261006T020342Z-v9.json"
live_sha = hashlib.sha256(live.read_bytes()).hexdigest()
assert live_sha == "5db39abc919d3799954965d6d59e8a7428e26df959b6fc959c8367454cf76133"
actual = json.loads(live.read_text())["actual"]
assert actual["job_id"] == "3289" and "JobState=RUNNING" in actual["scheduler"]["stdout"]

def write(path, value):
    tmp = path.with_name(path.name + ".reviews.tmp")
    with tmp.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)

state_path = base / "work/gigpo-e1-instruct300-state.json"
state = json.loads(state_path.read_text())
state.update(last_checked_utc=actual["utc"], last_update_utc=actual["utc"],
             current_verified_wait_evidence=str(live),
             deployment_review_sha256="a7ce7599d2c764efd203276a47fc6f5d3e75f2742ca83206680a640b77173218")
prep = state["policy_safety"]["backend_v9_preparation"]
prep.update(actual_live_path=str(live), actual_live_sha256=live_sha, actual_live_utc=actual["utc"],
            target_backend_metadata_accepted=True, actual_backend_metadata_review=verified["backend"],
            actual_restore_control_receipt_accepted=True, actual_restore_control_review=verified["restore"],
            actual_saved_PT_independent_replay_performed=False, probability_accepted=False)
formal = dict(status="INDEPENDENT_STATIC_CPU_PASS_NOT_READY_NOT_INSTALLED",
    proofs={k: v for k, v in verified.items() if k.startswith("formal_")},
    control_cpu_tests=55, repeats_native_gpu=False, extra_optional_AB_not_mandatory=True,
    actual_calibration_probability_acceptance=None, actual_calibration_10Adam_acceptance=None,
    production_thresholds=None, probability_tolerances=None, target_written=False, submitted=False,
    physical_scope="Future mature proposal admission reads complete file and evidence SHA before the physical helper inherited-asset metadata loop; this preparation did no target scan.")
state["policy_safety"]["formal_deployment_template_v1"] = formal
state["goal_turn_classification"].update(authoritative_evidence=str(live), authoritative_evidence_sha256=live_sha,
    utc=actual["utc"], reason="Actual backend and full25 restore control receipts independently accepted in narrow scopes, formal deployment templates independently prepared; unique calibration3289 remains live in baseline evaluation.")
write(state_path, state)
impl_path = out / "IMPLEMENTATION-STATUS.json"
impl = json.loads(impl_path.read_text())
impl.update(updated_utc=actual["utc"], last_checked_utc=actual["utc"],
    calibration_backend_metadata_independent_acceptance=True,
    calibration_backend_metadata_review=verified["backend"],
    calibration_restore25_before_baseline_independent_acceptance=True,
    calibration_restore25_before_baseline_review=verified["restore"]["path"],
    calibration_restore25_before_baseline_review_sha256=verified["restore"]["sha256"],
    calibration_restore_acceptance_scope="ACTUAL_RESTORE_CONTROL_ORIGINALS_NOT_LIVE_PT_REPLAY",
    calibration_live_evidence=str(live), calibration_live_evidence_sha256=live_sha,
    calibration_last_confirmed_utc=actual["utc"], formal_deployment_template_v1=formal)
write(impl_path, impl)
report = Path("/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct300/REPORT.md")
text = report.read_text()
line = "2026-10-06 10:03 独审更新：3289实际八卡36层Flash后端元数据接受SHA db8e126e415da656fa9684d4b69e2d203361651d5857227756d69ce5e7fdced3；第25轮恢复控制原件接受SHA76d7ceae596144c1cca7a88c1ca9c82279df2e29e008055b6a2055e6c45d585b，仅控制收据范围，未再次读取PT做独立重演。CPU preflight组的导入记录不冒充实际GPU worker逐卡采集。正式部署模板55项独立控制CPU通过SHA643ccbd88fa2b8473b5183c71914fbf1aeebe39312facec7bb7516556129b377，仍NOT_READY、未安装未提交；实际概率与10Adam校准、生产阈值和容差仍为空。3289实际RUNNING，基准评估继续，正式25/300、新0。\n"
assert line not in text
with report.open("a") as f:
    f.write("\n" + line)
print(json.dumps(dict(actual_live=str(live), live_sha256=live_sha, reviews=verified)))
