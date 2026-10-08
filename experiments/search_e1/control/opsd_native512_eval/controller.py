"""CPU controller for one complete, fixed-weight native512 M4 evaluation.

All startup gates precede worker spawn. NOT_READY never launches CUDA. Failed
endpoints HOLD without running update1/5. Already accepted endpoints are never
rerun by --resume-run; partial model evaluations cannot be resumed as complete.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import atomic_json, checked_json, file_sha, load_support, object_sha, proc_identity
    from identity import PHASES, VERSIONS, summarize_records
    from gates import NotReady, acquire_locks, input_gate, m1_completed, publication, require, resource_gate, owned_run_processes
else:
    from .utils import atomic_json, checked_json, file_sha, load_support, object_sha, proc_identity
    from .identity import PHASES, VERSIONS, summarize_records
    from .gates import NotReady, acquire_locks, input_gate, m1_completed, publication, require, resource_gate, owned_run_processes


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--resume-run", help="Only reuse complete, accepted endpoint phase. Never skip partial evaluations.")
    return parser.parse_args()


def phase_records(path, phase, config, rows):
    # Re-score every saved native result on CPU with the actual frozen reward.
    # The node trace/hash and the compact record must describe the same slot.
    import importlib
    core = importlib.import_module("search_e1.core")
    require(Path(core.__file__).resolve() == Path(config["runtime"]).resolve() / "search_e1/core.py", "Wrong frozen core in M4 merge")
    records, workers = [], []
    for worker_index in range(3):
        directory = Path(path) / phase / f"worker-{worker_index}"
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text())
        require(status["status"] == "COMPLETE" and status["phase"] == phase and status["worker_index"] == worker_index,
                "Incomplete actual M4 worker phase")
        require(status["final_source_weights_validation_science_publication_admission_verified"] is True,
                "M4 final immutable input checks missing")
        results_path = directory / "results.jsonl"
        require(file_sha(results_path) == status["results_sha256"], "M4 final results bytes changed")
        entries = [json.loads(line) for line in results_path.read_text().splitlines()]
        require(len(entries) == status["expected_model_question_slots"] == status["completed_model_question_slots"], "M4 worker count mismatch")
        require(status["config"]["sha256"] == file_sha(config["actual_config_path"]), "M4 worker configuration differs")
        for record in entries:
            require(record["worker_index"] == worker_index, "Cross-worker result substituted")
            reference = record["trace"]
            trace, _ = checked_json(reference)
            require(trace["native_result_unmodified"] is True and trace["hooks_restored"] is True and trace["version"] == record["version"] and trace["validation_index"] == record["validation_index"],
                    "M4 actual trace does not bind native result and restored hook scope")
            validation_index = record["validation_index"]
            require(trace["question_id"] == rows[validation_index]["question_id"] and trace["question"] == rows[validation_index]["question"] and trace["answers"] == rows[validation_index]["answers"],
                    "Trace/native512 original question and gold identity differ")
            require(object_sha(trace["native_result"]) == record["native_result_sha256"], "Compact record/native result semantic SHA differs")
            from search_r1.qa_protocol import extract_final_answer
            native = trace["native_result"]
            spans = native["policy_spans"]
            authored = [native["response"][start:end] for start, end in spans]
            score = core.reward(native["response"], rows[validation_index]["answers"], spans)
            answer = extract_final_answer(authored)
            require(score == record["protocol_em"] == trace["protocol"]["protocol_em"] and
                    (answer is not None) == record["protocol_valid_final_action"] == trace["protocol"]["protocol_valid_final_action"],
                    "Native CPU fresh reward/protocol differs from compact record")
            require(trace["generation_options_modified"] is False and trace["return_objects_modified"] is False,
                    "Trace did not preserve native generation options/return object")
            if record["rank_slot"] == 0:
                require(record["null_effect_equivalence"]["accepted"] is True, "Required actual native/hook null-effect check not accepted")
            records.append(record)
        workers.append(dict(worker_index=worker_index, status_path=str(status_path), status_sha256=file_sha(status_path),
                            result_path=str(results_path), result_sha256=status["results_sha256"], actual_complete_slots=len(entries)))
    summary = summarize_records(records, rows, PHASES[phase])
    summary["workers"] = workers
    return records, summary


def endpoint_acceptance(summary, config):
    actual = {version: summary["versions"][version]["correct"] for version in ("source100", "epoch1")}
    accepted = actual == config["historical_correct"]
    return dict(schema="native512-m4-endpoint-acceptance-v1", accepted=accepted,
                actual_correct=actual, expected_historical_correct=config["historical_correct"], denominator=512,
                route_changed_from_native18004_to_admitted18005=True,
                matching_counts_do_not_prove_identical_historical_question_success_sets=True,
                old_native_per_question_outputs_not_saved=True,
                decision="CONTINUE_EARLY_FIXED_WEIGHTS" if accepted else "HOLD_ENDPOINT_MISMATCH",
                summary=summary, verified_unix=time.time())


def run_phase(phase, config, run, controller, config_identity, admission, pointer_path):
    output = run / phase
    output.mkdir(exist_ok=False)
    ticket = dict(schema="native512-m4-launch-ticket-v1", accepted=True, phase=phase, run_id=run.name.removeprefix("native512-"),
                  controller={key: controller[key] for key in ("pid", "uid", "starttick")},
                  config_sha256=config_identity["sha256"], phase_output=str(output),
                  publication_source_manifest_sha256=config["helper_source_manifest"]["sha256"],
                  resource_free_gate=resource_gate(config), actual_retriever_health=admission.check_live(),
                  issued_unix=time.time(), no_training=True)
    ticket_path = run / (phase + "-launch-ticket.json")
    atomic_json(ticket_path, ticket)
    argv = [config["python"], str(Path(__file__).parent / "worker.py"), "--config", config_identity["path"],
            "--config-sha256", config_identity["sha256"], "--launch-ticket", str(ticket_path),
            "--launch-ticket-sha256", file_sha(ticket_path), "--output", str(output), "--phase", phase]
    children = []
    spawn_error = None
    try:
        for index, gpu in enumerate(config["physical_gpus"]):
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), E1_RUN_ID=ticket["run_id"], PYTHONUNBUFFERED="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1")
            log_path = run / f"{phase}-worker-{index}.log"
            child_argv = argv + ["--worker-index", str(index)]
            with log_path.open("x") as log:
                process = subprocess.Popen(child_argv, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = proc_identity(process.pid)
            children.append(dict(process=process, worker_index=index, physical_gpu=gpu,
                                 gpu_uuid=config["gpu_uuids"][index], identity=identity, argv=child_argv, log=str(log_path)))
            atomic_json(run / (phase + "-children-start.json"), [{key: value for key, value in child.items() if key != "process"} for child in children])
    except BaseException as error:
        spawn_error = dict(error=repr(error), traceback=traceback.format_exc())
        atomic_json(run / (phase + "-spawn-failure.json"), spawn_error)
    # Never release locks while an already spawned child is still running;
    # never send a signal to it, even when another child failed to spawn/run.
    while True:
        codes = {str(child["identity"]["pid"]): child["process"].poll() for child in children}
        atomic_json(pointer_path, dict(status="RUNNING", phase=phase, run_id=ticket["run_id"], output=str(run),
                                       controller=controller, child_exit_codes=codes, updated_unix=time.time(), spawn_error=spawn_error))
        if all(code is not None for code in codes.values()):
            break
        time.sleep(5)
    complete = len(children) == 3 and spawn_error is None and all(code == 0 for code in codes.values())
    final = dict(phase=phase, complete=complete, child_exit_codes=codes, spawn_error=spawn_error,
                 children=[{key: value for key, value in child.items() if key != "process"} for child in children],
                 actual_terminal_unix=time.time(), no_signals_sent=True)
    atomic_json(run / (phase + "-children-final.json"), final)
    require(complete, "A M4 worker did not complete exit0; preserve all partial outputs and HOLD")
    # CUDA cleanup may lag process exit briefly. It is a resource-release wait,
    # not a new pilot/evaluation. No model is restarted by this wait.
    deadline = time.monotonic() + 55
    while True:
        try:
            require(not owned_run_processes(ticket["run_id"], config["uid"]), "M4 phase still has an owned descendant")
            released = resource_gate(config)
            break
        except NotReady:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    final["gpu_release"] = released
    atomic_json(run / (phase + "-children-final.json"), final)
    return final


def main():
    args = arguments()
    config, config_identity = checked_json(dict(path=args.config, sha256=args.config_sha256))
    config["actual_config_path"] = config_identity["path"]  # local execution bookkeeping only, not generation settings
    base = Path(config["output_root"]).resolve()
    base.mkdir(parents=True, exist_ok=True)
    (base / "receipts").mkdir(exist_ok=True)
    pointer = base / "receipts/current-native512-run.json"
    locks = []
    run = None
    try:
        publication_proof = publication(config, Path(__file__).parent)
        support = load_support(config)
        # M1 must have really finished, with captured controller exit0, before
        # health/model/startup resources are admitted. No signals are sent.
        release = m1_completed(config, support)
        locks = acquire_locks(config)
        rows, admission, inputs = input_gate(config, support)
        resources = resource_gate(config)
        controller = proc_identity(os.getpid())
        if args.resume_run:
            run = Path(args.resume_run).resolve()
            require(run.parent == base / "runs", "Resume path outside the independent M4 output")
            original, _ = checked_json(dict(path=str(run / "config.json"), sha256=config_identity["sha256"]))
            require(not (run / "early").exists(), "Never silently resume/skip a partial early evaluation")
            accepted = json.loads((run / "endpoints-accepted.json").read_text())
            require(accepted["accepted"] is True, "Only fully accepted endpoints can be reused")
            endpoints, endpoint_summary = phase_records(run, "endpoints", config, rows)
            require(endpoint_acceptance(endpoint_summary, config)["accepted"], "Previously accepted endpoint files no longer reproduce239/203")
        else:
            (base / "runs").mkdir(exist_ok=True)
            run = base / "runs" / ("native512-" + uuid.uuid4().hex)
            run.mkdir()
            # Preserve the exact immutable supplied configuration bytes.
            (run / "config.json").write_bytes(Path(config_identity["path"]).read_bytes())
            require(file_sha(run / "config.json") == config_identity["sha256"], "Configuration copy differed")
        atomic_json(run / (f"controller-start-{os.getpid()}.json"), dict(controller=controller, config=config_identity,
                    publication=publication_proof, m1_actual_release=release, inputs=inputs, startup_resources=resources,
                    resumed_accepted_endpoints=bool(args.resume_run), started_unix=time.time(), no_training=True))
        if not args.resume_run:
            run_phase("endpoints", config, run, controller, config_identity, admission, pointer)
            endpoints, endpoint_summary = phase_records(run, "endpoints", config, rows)
            accepted = endpoint_acceptance(endpoint_summary, config)
            atomic_json(run / "endpoints-accepted.json", accepted)
            if not accepted["accepted"]:
                atomic_json(pointer, dict(status="HOLD_ENDPOINT_MISMATCH", output=str(run), endpoint_acceptance=accepted,
                                         no_early_workers_started=True, no_training=True, ended_unix=time.time()))
                return 4
        run_phase("early", config, run, controller, config_identity, admission, pointer)
        early, early_summary = phase_records(run, "early", config, rows)
        summary = summarize_records(endpoints + early, rows, VERSIONS)
        summary.update(accepted=True, no_training=True, config_sha256=config_identity["sha256"], endpoint_acceptance=accepted,
                       early_phase=early_summary, completed_unix=time.time(), output=str(run))
        atomic_json(run / "four-fixed-weights-native512-accepted.json", summary)
        atomic_json(pointer, dict(status="COMPLETE", output=str(run), summary_path=str(run / "four-fixed-weights-native512-accepted.json"),
                                 summary_sha256=file_sha(run / "four-fixed-weights-native512-accepted.json"), no_training=True,
                                 controller_exit_code=0, ended_unix=time.time()))
        return 0
    except BaseException as error:
        state = "NOT_READY" if run is None else "HOLD_RUNTIME_OR_IDENTITY_FAILURE"
        receipt = dict(status=state, output=str(run) if run else None, error=repr(error), traceback=traceback.format_exc(),
                       time_unix=time.time(), no_signals_sent=True, no_training=True,
                       workers_started_this_attempt=run is not None)
        atomic_json(base / "receipts" / ("not-ready-" + str(time.time_ns()) + ".json"), receipt)
        atomic_json(pointer, receipt)
        return 3
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
