"""Strict CPU startup/resource/publication/M1-release gates; never signal jobs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

try:
    from .utils import checked_json, environment, file_sha, proc_identity, stat_identity
    from .identity import GPU_UUIDS, cohort, native_files, check_adapter_labels, validate_config
except ImportError:
    from utils import checked_json, environment, file_sha, proc_identity, stat_identity
    from identity import GPU_UUIDS, cohort, native_files, check_adapter_labels, validate_config


class NotReady(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise NotReady(reason)


def publication(config, source_dir):
    """Root attests actual commit/push/readback; local presence is insufficient."""
    manifest, manifest_proof = checked_json(config["helper_source_manifest"])
    entries = manifest["files"]
    require(entries and all(Path(name).suffix == ".py" for name in entries), "Empty/unsafe M4 Python source manifest")
    require(set(entries) == {path.name for path in Path(source_dir).glob("*.py")}, "M4 source inventory differs from publication")
    for name, digest in entries.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe M4 manifest path")
        require(file_sha(Path(source_dir) / name) == digest, "M4 source hash differs: " + name)
    proof, proof_identity = checked_json(config["publication_proof"])
    require(proof.get("accepted") is True, "Latest M4 source publication is not accepted")
    require(proof.get("source_manifest_sha256") == manifest_proof["sha256"], "Publication does not bind this M4 source manifest")
    require(proof.get("source_commit") == proof.get("remote_readback_sha") and len(proof.get("source_commit", "")) == 40,
            "Actual M4 commit/push/remote readback identity is missing")
    require(proof.get("branch") == "search-e1" and proof.get("verified_unix", 0) > 0, "Wrong/undated M4 publication proof")
    return dict(manifest=manifest_proof, publication=proof_identity, source_commit=proof["source_commit"], remote_readback_sha=proof["remote_readback_sha"])


def m1_completed(config, support, proc_reader=proc_identity, owned_reader=None):
    release, release_identity = checked_json(config["m1_release_receipt"])
    require(release.get("accepted") is True and release.get("completed_cases") == 192,
            "M1 full 192-case release is not accepted")
    require(release.get("source_and_weight_end_verified") is True, "M1 final source/weight checks were not verified")
    require(release["weight_manifest_sha256"] == file_sha(config["weight_manifest"]), "M1/M4 fixed weight manifest differs")
    require(release["debug_source_manifest_sha256"] == config["debug_source_manifest"]["sha256"], "M1 final source manifest differs")
    controller = release["controller"]
    require(controller["uid"] == config["uid"] and controller.get("exit_code") == 0, "M1 controller actual exit0 is missing")
    final, final_identity = checked_json(release["controller_final"])
    exit_proof, exit_identity = checked_json(release["tmux_exit_proof"])
    validate_exit_proof(release, exit_proof)
    require(final["status"] == "WORKERS_COMPLETE" and final["run_id"] == release["m1_run_id"], "M1 controller final is not complete")
    require(final["controller_pid"] == controller["pid"], "M1 controller final PID differs")
    children = final["children"]
    require(len(children) == 3 and set(final["child_exit_codes"]) == {str(child["pid"]) for child in children}, "M1 final child identities are incomplete")
    require(all(code == 0 for code in final["child_exit_codes"].values()), "A M1 worker did not exit0")
    output = Path(final["output"]).resolve()
    require(output.parent == (Path(config["debug_root"]) / "runs").resolve(), "M1 output outside its preserved debug run")
    # verify_cases provides all original semantic hashes and fixed 192-cell identity.
    import importlib.util
    worker_source = Path(config["diagnostic_source"]).resolve() / "worker.py"
    spec = importlib.util.spec_from_file_location("native512_frozen_m1_case_verifier", worker_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_cases, case_identity = module.verify_cases(Path(config["debug_root"]) / "cases")
    expected = {row["case_id"]: row for row in expected_cases}
    seen, statuses = set(), []
    for worker_index, child in enumerate(children):
        require(child["uid"] == config["uid"] and child["physical_gpu"] == [4, 5, 6][worker_index] and child["gpu_uuid"] == GPU_UUIDS[worker_index], "M1 owned worker identity differs")
        directory = output / f"worker-{worker_index}"
        status = json.loads((directory / "status.json").read_text())
        require(status["status"] == "COMPLETE" and status["completed_cases"] == 64, "M1 worker lacks complete64 final status")
        require(status["pid"] == child["pid"] and status["uid"] == child["uid"] and status["proc_starttick"] == child["starttick"] and
                status["ppid"] == controller["pid"] and status["pgid"] == child["pid"] and status["worker_index"] == worker_index,
                "M1 final worker status is not bound to actual owned PID/tick/UID/parent/PGID")
        result_path = directory / "case-results.jsonl"
        require(file_sha(result_path) == status["results_sha256"], "M1 complete result file changed")
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        require(len(rows) == 64, "M1 worker actual result count differs")
        for row in rows:
            key = row["case_id"]
            require(key in expected and key not in seen and row["status"] == "complete", "M1 incomplete/duplicate/unknown result case")
            original = expected[key]
            require(row["case_index"] == original["case_index"] and row["case_index"] % 3 == worker_index and row["source_case_sha256"] == original["case_semantic_sha256"], "M1 case source/physical mapping changed")
            require(set(row["versions"]) == {"source100", "update1", "update5", "epoch1"}, "M1 case lacks fixed model versions")
            seen.add(key)
        statuses.append(dict(path=str(directory / "status.json"), sha256=file_sha(directory / "status.json"), results_sha256=status["results_sha256"], completed_cases=64))
    require(seen == set(expected), "M1 does not contain all192 fixed unique cases")
    owned = [controller] + children
    for process in owned:
        try:
            current = proc_reader(process["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(current["uid"] != process["uid"] or current["starttick"] != process["starttick"], "M1 owned process has not exited: " + str(process["pid"]))
    leftovers = owned_reader(release["m1_run_id"], config["uid"]) if owned_reader else owned_run_processes(release["m1_run_id"], config["uid"])
    require(not leftovers, "M1 run still has owned descendants/orphans: " + str(leftovers))
    return dict(release=release_identity, controller_final=final_identity, tmux_actual_exit=exit_identity,
                actual_complete_unique_cases=192, workers=statuses, owned_run_processes=[], preserved_old_run=str(output),
                no_signals_sent=True, cases_sha256=case_identity["sha256"])


def validate_exit_proof(release, exit_proof):
    require(exit_proof.get("pane_dead") is True and exit_proof.get("pane_dead_status") == 0,
            "Actual owned tmux terminal exit0 is not present")
    require(exit_proof.get("controller") == release["controller"],
            "Tmux actual terminal exit proof is not bound to the exact controller UID/tick/PID/exit")
    require(exit_proof.get("m1_run_id") == release["m1_run_id"], "Tmux exit proof names another M1 run")
    preservation, preservation_identity = checked_json(exit_proof["preservation_receipt"])
    require(preservation_identity["sha256"] == "d872cfc0da2329e319167f6d8d6ae85ac48ce309d21fd83d92c6b75bf61ed45b",
            "Actual owned M1 remain-on-exit preservation evidence differs")
    # Root's normalized terminal receipt binds the preserved tmux pane/binding
    # and its final actual exit reading, never disappearance alone.
    require(exit_proof.get("actual_tmux_reading") is not None and exit_proof.get("observed_unix", 0) > 0,
            "Normalized M1 exit receipt lacks the actual terminal reading/time")


def owned_run_processes(run_id, uid):
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid == uid and environment(int(directory.name)).get("E1_RUN_ID") == run_id:
                found.append(proc_identity(int(directory.name)))
        except (FileNotFoundError, ProcessLookupError):
            pass  # Process vanished while CPU-reading its identity.
        except PermissionError as error:
            raise NotReady("Cannot inspect a same-UID process for an M1 orphan: " + str(error)) from error
    return found


def gpu_inventory():
    def rows(kind, fields):
        raw = subprocess.check_output(["nvidia-smi", "--query-" + kind + "=" + fields, "--format=csv,noheader,nounits"], text=True)
        return [[part.strip() for part in line.split(",")] for line in raw.splitlines() if line.strip()]
    devices = {int(row[0]): dict(index=int(row[0]), uuid=row[1], actual_free_mib=int(row[2]), used_mib=int(row[3]), total_mib=int(row[4]))
               for row in rows("gpu", "index,uuid,memory.free,memory.used,memory.total")}
    apps = [dict(gpu_uuid=row[0], pid=int(row[1]), used_gpu_memory=row[2]) for row in rows("compute-apps", "gpu_uuid,pid,used_gpu_memory")]
    return devices, apps


def resource_gate(config, inventory_reader=gpu_inventory, disk_reader=shutil.disk_usage):
    devices, apps = inventory_reader()
    chosen = [devices[index] for index in config["physical_gpus"]]
    require([device["uuid"] for device in chosen] == config["gpu_uuids"], "Physical GPU4–6 UUID mapping changed")
    require(all(device["actual_free_mib"] >= config.get("minimum_free_mib", 20000) for device in chosen), "GPU4–6 actual memory.free below startup threshold")
    require(not [app for app in apps if app["gpu_uuid"] in config["gpu_uuids"]], "GPU4–6 contains a compute process; do not stop it")
    free = disk_reader(config["output_root"]).free
    require(free >= config.get("minimum_disk_bytes", 100 << 30), "Insufficient diagnostic output disk space")
    return dict(sample_unix=time.time(), devices=chosen, selected_compute_processes=[], disk_free_bytes=free, no_signals_sent=True)


def acquire_locks(config):
    """Use the SAME M1 resource lock, retain ownership through every M4 phase."""
    handles = []
    try:
        for path in (Path(config["debug_root"]) / "experiment.lock", Path(config["output_root"]) / "experiment.lock"):
            handle = path.open("a")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise NotReady("The M1/M4 resource lock is still held")


def input_gate(config, support):
    validate_config(config)
    require(os.getuid() == config["uid"], "Wrong actual HKUST UID")
    rows, validation_proof = cohort(config)
    _, native_proof = native_files(config)
    weights = support["weight_identity"].check_weight_manifest(config["weight_manifest"], config["model"], config["adapters"])
    check_adapter_labels(weights)
    core, science = support["common"].load_science(config["runtime"], with_torch=False)
    admission = support["admission"].RetrieverAdmission(config["retriever_admission"], config["retriever_url"], allow_fallback=False)
    health = admission.check_live()
    return rows, admission, dict(validation=validation_proof, native=native_proof, weights=weights, science=science,
                                admitted_retriever=admission.identity, actual_startup_health=health,
                                route_changed_from_native18004_to_admitted18005=True,
                                admission_numeric_gate_does_not_prove_every_future_query_bitwise=True)
