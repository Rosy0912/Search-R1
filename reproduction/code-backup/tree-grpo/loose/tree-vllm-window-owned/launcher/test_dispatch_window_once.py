"""Synthetic CPU-only launcher checks; no real scheduler/GPU/service calls.

generation_call=5 identifies the saved window. This suite does not execute the
window: its runtime scope is one original generate invocation per rank, 320
prompts and 80 original actor micro4 scores per rank, and nine global anchors.
All submission receipts below are synthetic transport fixtures, never releases.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


sys.dont_write_bytecode = True
LAUNCHER = Path(__file__).resolve().parent


def load_local_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, LAUNCHER / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = load_local_module("_window_dispatch_test_package", "probe_package.py")
with patch.dict(sys.modules, {"probe_package": package}):
    dispatch = load_local_module("_window_dispatch_test_dispatch", "dispatch_once.py")


def mock_receipt(stdout="991;synthetic-cluster", code=0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=code)


def fake_queue(raw, details, expansions):
    """Return an in-memory stand-in for all three scheduler read commands."""
    def read(argv, **kwargs):
        if argv[0] == "squeue":
            return raw
        if argv[:3] == ["scontrol", "show", "job"]:
            return details[argv[3]]
        if argv[:3] == ["scontrol", "show", "hostnames"]:
            return expansions[argv[3]]
        raise AssertionError("Unexpected scheduler command: " + repr(argv))
    return read


def synthetic_window_inputs():
    """Only signed metadata is constructed; no source path is opened."""
    return dict(
        status='ORIGINAL_CALL5_METADATA_PLAN_FROZEN_PENDING_TENSOR_READ',
        actual_tensor_rows_verified=False,
        source_position_report=dict(path=str(package.ROOT/'synthetic-only/position.json'),sha256='c'*64),
        generation_call=5,
        original_selection_sha256=package.SELECTION_SHA,
        original_ready_sha256=package.ORIGINAL_READY_SHA,
        original_config_sha256=package.ORIGINAL_CONFIG_SHA,
        maximum_new_emitted_tokens=1280000,
        anchors=package.window_anchors(),
        ranks=[dict(
            rank=rank, local_rows=320, expected_actor_micro4_calls=80,
            original_physical_rows=list(range(320)), prompt_width=256,
            source_full_input_identity={key:dict(dtype='torch.int64',shape=[320,756],shape_and_bytes_sha256='f'*64)
                for key in ('input_ids','attention_mask','position_ids')},
            source_snapshot=dict(
                path=str(package.ROOT / ("synthetic-only/rank-%02d-snapshot" % rank)),
                sha256="a" * 64,
            ),
            source_record=dict(
                path=str(package.ROOT / ("synthetic-only/rank-%02d-record" % rank)),
                sha256="b" * 64,
            ),
        ) for rank in range(8)],
    )


def synthetic_envelopes():
    """R3 envelope fixture extended with the new mandatory window bindings."""
    inputs = synthetic_window_inputs()
    profile = dict(
        installed_kernel_binaries=copy.deepcopy(package.KERNEL_BINARY_PINS),
        diagnostic_profile=package.WINDOW_PROFILE,
        window_contract=copy.deepcopy(package.WINDOW_CONTRACT),
        runner_module=package.RUNNER_MODULE,
        window_inputs_sha256=package.digest(inputs),
    )
    template = dict(
        output=package.OUTPUT_TOKEN, status=package.BUNDLE_STATUS, runner_commit="1" * 40,
        selection_file=dict(path="/unused/synthetic-metadata", sha256=package.SELECTION_SHA),
        reviewed_resource_adapter=dict(path=str(package.ROOT / "synthetic-only/adapter.py"), sha256="2" * 64),
        probe_source_files=[dict(path=str(package.ROOT / "synthetic-only/runner.py"), sha256="3" * 64)],
        installed_sources=[dict(path="/unused/synthetic-site/source.py", sha256="4" * 64)],
        installed_versions=dict(torch="synthetic", vllm="synthetic"),
        window_profile=package.WINDOW_PROFILE, window_inputs=inputs,
        vllm_installed_sources={"/unused/synthetic-site/vllm/__init__.py": "6" * 64},
        **copy.deepcopy(profile),
    )
    manifest = dict(
        schema_version=1, protocol=copy.deepcopy(package.PROTOCOL),
        bundle_payload_sha256=package.digest(package.bundle_payload(template)),
        selection_sha256=package.SELECTION_SHA, runner_commit=template["runner_commit"],
        files={"bundle-template.json": "7" * 64},
        **copy.deepcopy(profile),
    )
    for field in (
        "reviewed_resource_adapter", "probe_source_files", "installed_sources",
        "installed_versions", "vllm_installed_sources",
    ):
        manifest[field] = copy.deepcopy(template[field])
    receipt = dict(
        status=package.RELEASE_STATUS, protocol=copy.deepcopy(package.PROTOCOL),
        package_manifest_sha256="5" * 64,
        bundle_payload_sha256=manifest["bundle_payload_sha256"],
        selection_sha256=package.SELECTION_SHA, runner_commit="1" * 40,
        actual_optimizer_writes=0, adapter_sha256="2" * 64,
        dispatch_intent_path=package.DISPATCH_INTENT,
        **copy.deepcopy(profile),
    )
    return template, manifest, receipt


class WindowReleaseTests(unittest.TestCase):
    def test_complete_synthetic_new_window_envelope_is_valid(self):
        template, manifest, receipt = synthetic_envelopes()
        self.assertTrue(package.validate_release(manifest, "5" * 64, template, receipt))

    def test_old_or_arbitrary_dispatch_intents_rejected(self):
        for name in (
            "PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json",
            "PROBABILITY_PROBE_ONCE-3279.json",
            "SINGLE_TP1_WINDOW_AFTER_3282_ONCE-copy.json",
        ):
            with self.subTest(intent=name):
                template, manifest, receipt = synthetic_envelopes()
                receipt["dispatch_intent_path"] = str(package.ROOT / "instruct300" / name)
                with self.assertRaisesRegex(ValueError, "new fixed intent"):
                    package.validate_release(manifest, "5" * 64, template, receipt)

    def test_old_native_release_status_does_not_authorize_window(self):
        template, manifest, receipt = synthetic_envelopes()
        for field in ("diagnostic_profile", "window_contract", "runner_module", "window_inputs_sha256"):
            receipt.pop(field)
        with self.assertRaisesRegex(ValueError, "Old native release"):
            package.validate_release(manifest, "5" * 64, template, receipt)

    def test_window_profile_runner_and_inputs_must_match_all_envelopes(self):
        changes = (
            ("diagnostic_profile", "old-native"),
            ("runner_module", "search_r1.baselines.probability_probe_runner"),
            ("window_inputs_sha256", "9" * 64),
        )
        for position in range(3):
            for field, wrong in changes:
                with self.subTest(envelope=position, field=field):
                    envelopes = synthetic_envelopes()
                    envelopes[position][field] = wrong
                    template, manifest, receipt = envelopes
                    with self.assertRaises(ValueError):
                        package.validate_release(manifest, "5" * 64, template, receipt)

    def test_call5_is_window_id_with_one_invocation_and_global_nine_anchors(self):
        self.assertEqual(package.WINDOW_CONTRACT["generation_call_id"], 5)
        self.assertEqual(package.WINDOW_CONTRACT["generation_invocations_per_rank"], 1)
        self.assertEqual(package.WINDOW_CONTRACT["prompts_per_rank"], 320)
        self.assertEqual(package.WINDOW_CONTRACT["actor_microbatch_rows"], 4)
        self.assertEqual(package.WINDOW_CONTRACT["actor_comparison_calls_per_rank"], 80)
        self.assertEqual(package.WINDOW_CONTRACT["emitted_cap_per_row"], 500)
        expected = [dict(id="rank%02d-first" % rank, rank=rank, input_row=0, response_offset=0) for rank in range(8)]
        expected.append(dict(id="rank04-row160-offset17", rank=4, input_row=160, response_offset=17))
        self.assertEqual(package.window_anchors(), expected)

    def test_call_count_drift_cannot_be_signed_into_all_envelopes(self):
        template, manifest, receipt = synthetic_envelopes()
        for value in (template, manifest, receipt):
            value["window_contract"]["generation_invocations_per_rank"] = 5
        with self.assertRaises(ValueError):
            package.validate_release(manifest, "5" * 64, template, receipt)

    def test_complete_rank_window_metadata_rejects_partial_or_reordered_rows(self):
        for field, wrong in (
            ("local_rows", 319), ("expected_actor_micro4_calls", 79),
            ("original_physical_rows", list(range(319))), ("prompt_width", 4097),
        ):
            with self.subTest(field=field):
                inputs = synthetic_window_inputs()
                inputs["ranks"][4][field] = wrong
                with self.assertRaises(ValueError):
                    package.validate_window_inputs(inputs)
        inputs = synthetic_window_inputs()
        inputs["ranks"].reverse()
        with self.assertRaises(ValueError):
            package.validate_window_inputs(inputs)

    def test_nonzero_updates_and_boolean_zero_receipts_rejected(self):
        for value in (1, False):
            with self.subTest(actual_optimizer_writes=value):
                template, manifest, receipt = synthetic_envelopes()
                receipt["actual_optimizer_writes"] = value
                with self.assertRaises(ValueError):
                    package.validate_release(manifest, "5" * 64, template, receipt)
        for field in ("optimizer_writes", "backward_calls"):
            with self.subTest(protocol=field):
                template, manifest, receipt = synthetic_envelopes()
                manifest["protocol"][field] = receipt["protocol"][field] = 1
                with self.assertRaises(ValueError):
                    package.validate_release(manifest, "5" * 64, template, receipt)

    def test_dispatch_cli_cannot_bypass_fixed_intent_and_binds_window_identity(self):
        package_path = package.ROOT / "synthetic-only/window-package"
        release_path = package.ROOT / "synthetic-only/release.json"
        for intent in (package.DISPATCH_INTENT, str(package.ROOT / "instruct300/arbitrary.json")):
            with self.subTest(intent=intent):
                template, manifest, receipt = synthetic_envelopes()
                argv = [
                    "dispatch_once.py", "--package", str(package_path),
                    "--manifest-sha256", "5" * 64, "--release", str(release_path),
                    "--release-sha256", "8" * 64, "--intent", intent,
                ]
                with patch.object(sys, "argv", argv), \
                     patch.object(dispatch.socket, "gethostname", return_value="h100-9"), \
                     patch.object(dispatch, "read_bound", side_effect=[manifest, template, receipt]), \
                     patch.object(dispatch, "validate_package_files"), \
                     patch.object(dispatch, "submit_once", return_value={"synthetic": True}) as submit, \
                     patch.object(dispatch.subprocess, "run", side_effect=AssertionError("No real submit")), \
                     patch.object(dispatch.subprocess, "check_output", side_effect=AssertionError("No real queue")), \
                     patch("builtins.print"):
                    if intent != package.DISPATCH_INTENT:
                        with self.assertRaisesRegex(ValueError, "fixed single-attempt intent"):
                            dispatch.main()
                        submit.assert_not_called()
                    else:
                        dispatch.main()
                        submit.assert_called_once()
                        identity = submit.call_args.args[2]
                        self.assertEqual(identity["diagnostic_profile"], package.WINDOW_PROFILE)
                        self.assertEqual(identity["window_inputs_sha256"], package.digest(template["window_inputs"]))
                        self.assertEqual(identity["window_contract"], package.WINDOW_CONTRACT)


class AllocationContractTests(unittest.TestCase):
    def test_fixed_window_intent_has_new_identity(self):
        self.assertEqual(
            Path(package.DISPATCH_INTENT),
            package.ROOT / "instruct300/SINGLE_TP1_WINDOW_AFTER_3282_ONCE.json",
        )

    def test_zero_update_resource_protocol(self):
        expected = dict(
            node="h100-9", nodes=1, tasks=1, gpu_count=8, exclusive=True,
            requested_cpus=32, memory_request="256G", time_limit_minutes=30,
            no_requeue=True, optimizer_writes=0, backward_calls=0,
        )
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(package.PROTOCOL[key], value)
        self.assertIs(type(package.PROTOCOL["optimizer_writes"]), int)
        self.assertIs(type(package.PROTOCOL["backward_calls"]), int)

    def test_actual_command_exact_resources_and_pins(self):
        args = dispatch.command("/same node/window", "M", "/same node/release", "R")
        self.assertEqual(args[0], "sbatch")
        for item in (
            "--parsable", "--ntasks=1", "--gres=gpu:8", "--exclusive",
            "--cpus-per-task=32", "--mem=256G", "--time=00:30:00", "--no-requeue",
        ):
            with self.subTest(option=item):
                self.assertEqual(args.count(item), 1)
        self.assertEqual(args[args.index("-w") + 1], "h100-9")
        self.assertEqual(args[args.index("-N") + 1], "1")
        self.assertEqual(args[args.index("--package") + 1], "/same node/window")
        self.assertEqual(args[args.index("--manifest-sha256") + 1], "M")
        self.assertEqual(args[args.index("--release") + 1], "/same node/release")
        self.assertEqual(args[args.index("--release-sha256") + 1], "R")
        self.assertIn("/same node/window/start.sh", args)
        self.assertFalse(any(item.startswith("--wrap") for item in args))

    def test_actual_allocation_accepts_normalized_memory(self):
        text = (
            "JobId=991 JobState=RUNNING NodeList=h100-9 OverSubscribe=NO "
            "TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=64 "
            "TimeLimit=00:30:00 Requeue=0 MinMemoryNode=256G"
        )
        self.assertEqual(package.allocation(text, "991"), 64)
        self.assertEqual(package.allocation(text.replace("256G", "262144M"), "991"), 64)
        for old, new in (
            ("gres:gpu:8", "gres:gpu:7"), ("CPUs/Task=32", "CPUs/Task=16"),
            ("256G", "200G"), ("00:30:00", "01:00:00"), ("Requeue=0", "Requeue=1"),
        ):
            with self.subTest(change=(old, new)), self.assertRaises(ValueError):
                package.allocation(text.replace(old, new), "991")


class SingleAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "SINGLE_TP1_WINDOW_AFTER_3282_ONCE.json"
        self.calls = []

    def submit(self, argv):
        self.assertTrue(self.path.is_file(), "Intent must be durable before submission")
        self.calls.append(argv)
        return mock_receipt()

    def result(self):
        return json.loads(self.path.with_suffix(self.path.suffix + ".result.json").read_text())

    def test_success_is_one_attempt_with_no_execution_acceptance(self):
        identity = {"package_manifest_sha256": "a" * 64, "release_sha256": "b" * 64}
        queue = {"status": "synthetic refreshed queue"}
        argv = ["synthetic-sbatch", "window"]
        result = dispatch.submit_once(self.path, argv, identity, self.submit, pre_submit=lambda: queue)
        marker = json.loads(self.path.read_text())
        self.assertEqual(result["job_id"], "991")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(marker["status"], "SUBMISSION_ATTEMPTED_OUTCOME_UNKNOWN")
        self.assertEqual(marker["identity"], identity)
        self.assertEqual(marker["command_sha256"], package.digest(argv))
        self.assertEqual(marker["refreshed_queue_evidence"], queue)
        self.assertFalse(marker["automatic_retry_allowed"])
        self.assertEqual(result["status"], "SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE")
        self.assertFalse(result["actual_gpu_probe_accepted"])

    def test_duplicate_cannot_change_original_identity_or_submit_again(self):
        dispatch.submit_once(self.path, ["synthetic-sbatch"], {"version": "first"}, self.submit)
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            dispatch.submit_once(self.path, ["different"], {"version": "second"}, self.submit)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(len(self.calls), 1)

    def test_nonzero_response_is_unknown_and_cannot_retry(self):
        def failure(argv):
            self.assertTrue(self.path.is_file())
            self.calls.append(argv)
            return mock_receipt("991", 1)
        with self.assertRaises(ValueError):
            dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, failure)
        self.assertEqual(self.result()["status"], "SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RETRY")
        with self.assertRaises(ValueError):
            dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, self.submit)
        self.assertEqual(len(self.calls), 1)

    def test_timeout_is_unknown_and_cannot_retry(self):
        def timeout(argv):
            self.assertTrue(self.path.is_file())
            self.calls.append(argv)
            raise subprocess.TimeoutExpired(argv, 30)
        with self.assertRaises(subprocess.TimeoutExpired):
            dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, timeout)
        self.assertEqual(self.result()["status"], "SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RETRY")
        with self.assertRaises(ValueError):
            dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, self.submit)
        self.assertEqual(len(self.calls), 1)

    def test_ambiguous_response_never_guesses_or_retries(self):
        for index, answer in enumerate(("991\n992", "Submitted batch job 991", "991;c;extra", "0", "")):
            with self.subTest(answer=answer):
                path = self.path.with_name("synthetic-unknown-%d.json" % index)
                submit = Mock(return_value=mock_receipt(answer))
                with self.assertRaises(ValueError):
                    dispatch.submit_once(path, ["synthetic-sbatch"], {}, submit)
                with self.assertRaises(ValueError):
                    dispatch.submit_once(path, ["synthetic-sbatch"], {}, submit)
                self.assertEqual(submit.call_count, 1)
                self.assertFalse(json.loads(path.read_text())["automatic_retry_allowed"])

    def test_failure_to_persist_intent_prevents_submission(self):
        with patch.object(dispatch, "atomic_json", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, self.submit)
        self.assertFalse(self.calls)

    def test_confirmation_write_failure_still_blocks_retry(self):
        original = dispatch.atomic_json
        def persist(path, value):
            if Path(path) != self.path:
                raise OSError("synthetic confirmation write failure")
            return original(path, value)
        with patch.object(dispatch, "atomic_json", side_effect=persist):
            with self.assertRaises(OSError):
                dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, self.submit)
        with self.assertRaises(ValueError):
            dispatch.submit_once(self.path, ["synthetic-sbatch"], {}, self.submit)
        self.assertEqual(len(self.calls), 1)


class QueueGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic-intent.json"
        self.submit = Mock(return_value=mock_receipt())

    def attempt_with_read(self, read):
        with patch.object(dispatch.subprocess, "check_output", side_effect=read):
            return dispatch.submit_once(
                self.path, ["synthetic-sbatch"], {}, self.submit,
                pre_submit=lambda: dispatch.refreshed_queue_gate(self.path),
            )

    def assert_no_attempt(self):
        self.assertFalse(self.path.exists())
        self.submit.assert_not_called()
        evidence = list(self.path.parent.glob(self.path.name + ".queue-*.json"))
        self.assertEqual(len(evidence), 1)
        return json.loads(evidence[0].read_text())

    def test_unknown_queue_blocks_before_intent_and_submission(self):
        def unknown(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 15)
        with self.assertRaises(ValueError):
            self.attempt_with_read(unknown)
        self.assertIn("TimeoutExpired", self.assert_no_attempt()["unknown_error"])

    def test_h1009_running_and_pending_gpu_jobs_block(self):
        for state in ("RUNNING", "PENDING"):
            with self.subTest(state=state):
                raw = "81|tree-window|%s|h100-9|gpu:8\n" % state
                details = {"81": "JobId=81 NodeList=h100-9 ReqNodeList=h100-9 ReqTRES=cpu=32,gres/gpu=8"}
                read = fake_queue(raw, details, {"h100-9": "h100-9\n"})
                self.path = Path(self.temp.name) / ("synthetic-%s.json" % state)
                with self.assertRaises(ValueError):
                    self.attempt_with_read(read)
                self.assertEqual(len(self.assert_no_attempt()["conflicts"]), 1)

    def test_empty_or_blank_expanded_node_blocks(self):
        for index, expansion in enumerate(("", "\n", " \n", " h100-9\n", "h100-9 \n")):
            with self.subTest(expansion=expansion):
                self.path = Path(self.temp.name) / ("synthetic-empty-%d.json" % index)
                raw = "81|tree-window|PENDING|(null)|gpu:8\n"
                details = {"81": "JobId=81 NodeList=(null) ReqNodeList=h100-9 ReqTRES=gres/gpu=8"}
                with self.assertRaises(ValueError):
                    self.attempt_with_read(fake_queue(raw, details, {"h100-9": expansion}))
                self.assertIn("expanded node identity", self.assert_no_attempt()["unknown_error"])

    def test_missing_node_identity_is_potential_h1009_conflict(self):
        raw = "81|tree-window|PENDING|(null)|gpu:8\n"
        details = {"81": "JobId=81 NodeList=(null) ReqNodeList=(null) ReqTRES=gres/gpu=8"}
        with self.assertRaises(ValueError):
            self.attempt_with_read(fake_queue(raw, details, {}))
        self.assertTrue(self.assert_no_attempt()["conflicts"][0]["requested_or_actual_node_unknown"])

    def test_known_other_node_and_cpu_only_jobs_allow_one_attempt(self):
        raw = "81|tree-cpu|RUNNING|h100-9|N/A\n82|tree-other|PENDING|h100-16|gpu:8\n"
        details = {
            "81": "JobId=81 NodeList=h100-9 ReqNodeList=h100-9 ReqTRES=cpu=4,mem=32G",
            "82": "JobId=82 NodeList=(null) ReqNodeList=h100-16 ReqTRES=gres/gpu=8",
        }
        result = self.attempt_with_read(fake_queue(raw, details, {"h100-16": "h100-16\n"}))
        self.assertEqual(result["job_id"], "991")
        self.submit.assert_called_once()

    def test_mismatched_job_identity_blocks(self):
        raw = "81|tree-window|PENDING|h100-16|gpu:8\n"
        details = {"81": "JobId=82 NodeList=h100-16 ReqNodeList=h100-16 ReqTRES=gres/gpu=8"}
        with self.assertRaises(ValueError):
            self.attempt_with_read(fake_queue(raw, details, {"h100-16": "h100-16\n"}))
        self.assertIn("identity differs", self.assert_no_attempt()["unknown_error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
