"""CPU fixtures only: transparent object identity, merge and admission guards."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from .hooks import capture_native
    from .identity import GPU_UUIDS, object_sha, rank_slots, summarize_records, validate_cohort_rows
    from .controller import endpoint_acceptance
    from . import controller as controller_module
    from .gates import NotReady, resource_gate, validate_exit_proof
    from .worker import decode_protocol
except ImportError:
    from hooks import capture_native
    from identity import GPU_UUIDS, object_sha, rank_slots, summarize_records, validate_cohort_rows
    from controller import endpoint_acceptance
    import controller as controller_module
    from gates import NotReady, resource_gate, validate_exit_proof
    from worker import decode_protocol


class FakeTensor:
    def __init__(self, data):
        self.data = data
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.data


class FakeTokenizer:
    def __init__(self):
        self.result = SimpleNamespace(input_ids=FakeTensor([[1, 2, 3]]))
    def __call__(self, *args, **kwargs):
        self.call = (args, kwargs)
        return self.result
    def decode(self, *args, **kwargs):
        return "unchanged"


class FakeModel:
    def __init__(self, output):
        self.output = output
    def generate(self, *args, **kwargs):
        self.call = (args, kwargs)
        self._get_logits_processor(generation_config=SimpleNamespace(repetition_penalty=1.05))
        return self.output
    def _get_logits_processor(self, *args, **kwargs):
        self.processor_list = [SimpleNamespace(penalty=1.05)]
        return self.processor_list


class HooksTests(unittest.TestCase):
    def fixtures(self, modeloutput=False):
        tensor = FakeTensor([[1, 2, 3, 4]])
        output = SimpleNamespace(sequences=tensor) if modeloutput else tensor
        model, tokenizer = FakeModel(output), FakeTokenizer()
        response = SimpleNamespace(content=b'{"result":[[{"contents":"full doc","score":0.9,"id":"42"}]]}', status_code=200)
        calls = []
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return response
        requests = SimpleNamespace(post=post)
        return model, tokenizer, output, response, calls, requests

    def test_tensor_and_modeloutput_return_identity_options_and_scope(self):
        for modeloutput in (False, True):
            model, tokenizer, output, response, calls, requests = self.fixtures(modeloutput)
            agent = SimpleNamespace(model=model, tokenizer=tokenizer)
            original_generate, original_post = model.generate, requests.post
            stop = SimpleNamespace(start=3)
            options = dict(max_new_tokens=500, do_sample=False, pad_token_id=151645, stopping_criteria=[stop])
            with capture_native(agent, requests) as trace:
                encoding = agent.tokenizer("actual prompt", return_tensors="pt", add_special_tokens=False)
                self.assertIs(encoding, tokenizer.result)
                received = model.generate(encoding.input_ids, **options)
                self.assertIs(received, output)
                self.assertIs(model.call[0][0], encoding.input_ids)
                self.assertEqual(set(model.call[1]), set(options))
                self.assertIs(model.call[1]["stopping_criteria"], options["stopping_criteria"])
                reply = requests.post("http://example/retrieve", json={"queries": ["q"], "topk": 3, "return_scores": True}, timeout=120)
                self.assertIs(reply, response)
                self.assertEqual(agent.tokenizer.decode([1]), "unchanged")
            self.assertIs(agent.tokenizer, tokenizer)
            self.assertEqual(model.generate, original_generate)
            self.assertNotIn("generate", vars(model))
            self.assertIs(requests.post, original_post)
            self.assertEqual(len(calls), 1)
            self.assertTrue(trace["hooks_restored"])
            self.assertEqual(trace["events"][0]["args"], ["actual prompt"])
            self.assertEqual(trace["events"][1]["output_ids"], [[1, 2, 3, 4]])
            processors = next(event for event in trace["events"] if event["kind"] == "actual_hf_logits_processors")
            self.assertEqual(processors["processors"][0]["attributes"], {"penalty": 1.05})
            self.assertTrue(processors["returned_original_list_object"])
            self.assertNotIn("_get_logits_processor", vars(model))
            self.assertEqual(trace["events"][-1]["response_json"]["result"][0][0]["contents"], "full doc")

    def test_exception_restores_own_attribute_and_response_module(self):
        model, tokenizer, _, _, _, requests = self.fixtures()
        error = RuntimeError("native error")
        def fail(*args, **kwargs):
            raise error
        model.generate = fail
        agent = SimpleNamespace(model=model, tokenizer=tokenizer)
        original_post = requests.post
        with self.assertRaises(RuntimeError) as caught:
            with capture_native(agent, requests) as trace:
                model.generate(FakeTensor([[1]]), do_sample=False)
        self.assertIs(caught.exception, error)
        self.assertIs(model.generate, fail)
        self.assertIs(agent.tokenizer, tokenizer)
        self.assertIs(requests.post, original_post)
        self.assertTrue(trace["hooks_restored"])

    def test_http_failure_is_not_retried_and_hooks_restore(self):
        model, tokenizer, _, _, _, requests = self.fixtures()
        count = []
        def fail(*args, **kwargs):
            count.append(1)
            raise TimeoutError("native timeout")
        requests.post = fail
        agent = SimpleNamespace(model=model, tokenizer=tokenizer)
        with self.assertRaises(TimeoutError):
            with capture_native(agent, requests):
                requests.post("url", timeout=120)
        self.assertEqual(len(count), 1)
        self.assertIs(requests.post, fail)
        self.assertIs(agent.tokenizer, tokenizer)

    def test_nested_hooks_rejected_without_touching_original(self):
        model, tokenizer, _, _, _, requests = self.fixtures()
        agent = SimpleNamespace(model=model, tokenizer=tokenizer)
        with capture_native(agent, requests):
            with self.assertRaises(RuntimeError):
                with capture_native(agent, requests):
                    pass
        self.assertIs(agent.tokenizer, tokenizer)


def rows_fixture():
    return [dict(question_id=str(index), question="question " + str(index), answers=["answer"], data_source="nq" if index % 2 == 0 else "hotpotqa") for index in range(512)]


def records_fixture(rows, versions=("source100", "epoch1")):
    return [dict(version=version, validation_index=index, question_id=row["question_id"], row_semantic_sha256=object_sha(row),
                 logical_rank=index % 8, rank_slot=index // 8, worker_index=(index % 8) % 3, rank_seed=42,
                 protocol_em=int(index < (239 if version == "source100" else 203)), status="complete")
            for version in versions for index, row in enumerate(rows)]


class MergeTests(unittest.TestCase):
    def test_complete_original_512_slices_and_exact_counts(self):
        rows = rows_fixture()
        validate_cohort_rows(rows)
        records = records_fixture(rows)
        summary = summarize_records(records, rows, ("source100", "epoch1"))
        self.assertEqual(summary["versions"]["source100"]["correct"], 239)
        self.assertEqual(summary["versions"]["epoch1"]["correct"], 203)
        self.assertEqual(summary["paired_against_source100"]["epoch1"]["correct_to_wrong"], 36)
        self.assertEqual([rank_slots(index) for index in range(3)], [[0, 3, 6], [1, 4, 7], [2, 5]])
        config = {"historical_correct": {"source100": 239, "epoch1": 203}}
        self.assertTrue(endpoint_acceptance(summary, config)["accepted"])
        summary["versions"]["epoch1"]["correct"] = 204
        held = endpoint_acceptance(summary, config)
        self.assertFalse(held["accepted"])
        self.assertEqual(held["decision"], "HOLD_ENDPOINT_MISMATCH")

    def test_partial_duplicate_wrong_slice_or_seed_rejected(self):
        rows = rows_fixture()
        records = records_fixture(rows)
        bad = [records[:-1], records + [records[0]]]
        for field, value in (("logical_rank", 3), ("worker_index", 2), ("rank_seed", 43), ("question_id", "foreign"), ("protocol_em", 0.5)):
            changed = copy.deepcopy(records)
            changed[0][field] = value
            bad.append(changed)
        for variant in bad:
            with self.assertRaises(ValueError):
                summarize_records(variant, rows, ("source100", "epoch1"))

    def test_513_cohort_and_duplicate_ids_rejected(self):
        rows = rows_fixture()
        for variant in (rows + [rows[0]], rows[:-1]):
            with self.assertRaises(ValueError):
                validate_cohort_rows(variant)
        rows[1]["question_id"] = rows[0]["question_id"]
        with self.assertRaises(ValueError):
            validate_cohort_rows(rows)


class GateTests(unittest.TestCase):
    def config(self):
        return dict(physical_gpus=[4, 5, 6], gpu_uuids=GPU_UUIDS, output_root="fixture", minimum_free_mib=20000, minimum_disk_bytes=100 << 30)

    def inventory(self):
        devices = {index: dict(index=index, uuid=uuid, actual_free_mib=23000) for index, uuid in zip([4, 5, 6], GPU_UUIDS)}
        return devices, []

    def test_resource_foreign_process_low_actualfree_and_wrong_uuid_rejected(self):
        config = self.config()
        disk = lambda path: SimpleNamespace(free=200 << 30)
        self.assertEqual(len(resource_gate(config, self.inventory, disk)["devices"]), 3)
        for change in ("foreign", "free", "uuid", "disk"):
            devices, apps = self.inventory()
            if change == "foreign": apps.append(dict(gpu_uuid=GPU_UUIDS[1], pid=999))
            if change == "free": devices[4]["actual_free_mib"] = 19999
            if change == "uuid": devices[4]["uuid"] = "foreign"
            disk_reader = (lambda path: SimpleNamespace(free=99 << 30)) if change == "disk" else disk
            with self.assertRaises(NotReady):
                resource_gate(config, lambda: (devices, apps), disk_reader)

    def test_disappearance_or_live_pane_never_proves_exit0(self):
        controller = dict(pid=33834, uid=1005, starttick=922457688, exit_code=0)
        release = dict(controller=controller, m1_run_id="894021c8f55f436f96bb3cb146ab05bb")
        proof = dict(pane_dead=True, pane_dead_status=0, controller=controller, m1_run_id=release["m1_run_id"],
                     preservation_receipt={"path": "fixture", "sha256": "d872cfc0da2329e319167f6d8d6ae85ac48ce309d21fd83d92c6b75bf61ed45b"},
                     actual_tmux_reading={"pane_dead": "1", "pane_dead_status": "0"}, observed_unix=1)
        with patch("gates.checked_json", return_value=({}, dict(sha256=proof["preservation_receipt"]["sha256"]))):
            validate_exit_proof(release, proof)
            for field, value in (("pane_dead", False), ("pane_dead_status", 1), ("actual_tmux_reading", None), ("observed_unix", 0)):
                changed = dict(proof, **{field: value})
                with self.assertRaises(NotReady):
                    validate_exit_proof(release, changed)
            changed = dict(proof, controller=dict(controller, starttick=1))
            with self.assertRaises(NotReady):
                validate_exit_proof(release, changed)

    def test_not_ready_m1_exit_never_spawns_worker_or_opens_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps(dict(output_root=str(Path(temporary) / "output"))))
            from utils import file_sha
            args = SimpleNamespace(config=str(config_path), config_sha256=file_sha(config_path), resume_run=None)
            with patch.object(controller_module, "arguments", return_value=args), \
                 patch.object(controller_module, "publication", return_value={}), \
                 patch.object(controller_module, "load_support", return_value={}), \
                 patch.object(controller_module, "m1_completed", side_effect=NotReady("M1 still running")), \
                 patch.object(controller_module, "input_gate") as inputs, \
                 patch.object(controller_module, "run_phase") as launch:
                self.assertEqual(controller_module.main(), 3)
                launch.assert_not_called()
                inputs.assert_not_called()
                proof = json.loads((Path(temporary) / "output/receipts/current-native512-run.json").read_text())
                self.assertEqual(proof["status"], "NOT_READY")
                self.assertFalse(proof["workers_started_this_attempt"])


class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        remote = Path("/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runtime/opsd-preparation3-xformers")
        local = Path(__file__).resolve().parents[6] / "work/search-e1-analysis-output/opsd-preparation3-xformers-backup-20261008/snapshot/runtime/opsd-preparation3-xformers"
        runtime = Path(os.environ["M4_TEST_FROZEN_RUNTIME"]) if os.environ.get("M4_TEST_FROZEN_RUNTIME") else remote if remote.exists() else local
        if not runtime.exists():
            raise unittest.SkipTest("Set M4_TEST_FROZEN_RUNTIME to the actual preserved runtime for native protocol CPU fixtures")
        from utils import file_sha
        qa = runtime / "upstream/Search-R1/search_r1/qa_protocol.py"
        core = runtime / "search_e1/core.py"
        if file_sha(qa) != "9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd" or file_sha(core) != "0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc":
            raise AssertionError("Native protocol fixture source differs from actual frozen science")
        cls.before_path = list(sys.path)
        sys.path.insert(0, str(runtime / "upstream/Search-R1"))
        spec = importlib.util.spec_from_file_location("native512_actual_core_fixture", core)
        cls.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core)
        cls.runtime = runtime.resolve()

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.before_path

    def result(self, response, spans=None):
        return dict(response=response, policy_spans=spans if spans is not None else [(0, len(response))], search_calls=0, termination="answer")

    def test_valid_answer_and_unclosed_think_lexical_correct_but_invalid(self):
        response = "<think>reason</think><answer>Ben Kingsley</answer>"
        valid = decode_protocol(self.core, self.result(response), ["Ben Kingsley"])
        self.assertEqual(valid["protocol_em"], 1)
        self.assertTrue(valid["protocol_valid_final_action"])
        self.assertEqual(valid["extracted_final_answer"], "Ben Kingsley")
        response = "<think>reason<answer>Ben Kingsley</answer>"
        invalid = decode_protocol(self.core, self.result(response), ["Ben Kingsley"])
        self.assertEqual(invalid["protocol_em"], 0)
        self.assertFalse(invalid["protocol_valid_final_action"])

    def test_only_last_authored_action_counts_not_info_or_earlier_answer(self):
        first = "<think>reason</think><answer>Ben Kingsley</answer>"
        info = "\n<information><answer>Ben Kingsley</answer></information>\n"
        last = "<think>search</think><search>Ben Kingsley</search>"
        response = first + info + last
        spans = [(0, len(first)), (len(first) + len(info), len(response))]
        protocol = decode_protocol(self.core, self.result(response, spans), ["Ben Kingsley"])
        self.assertEqual(protocol["protocol_em"], 0)
        self.assertFalse(protocol["protocol_valid_final_action"])
        protocol = decode_protocol(self.core, self.result(info, []), ["Ben Kingsley"])
        self.assertEqual(protocol["protocol_em"], 0)
        self.assertFalse(protocol["protocol_valid_final_action"])

    def test_phase_records_all_workers_multiple_rows_real_frozen_reward(self):
        from utils import file_sha, atomic_json, canonical
        rows = rows_fixture()
        records = records_fixture(rows)
        response = "<think>reason</think><answer>answer</answer>"
        native = self.result(response)
        protocol = decode_protocol(self.core, native, ["answer"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text("{}\n")
            config = dict(runtime=str(self.runtime), actual_config_path=str(config_path))
            for worker_index in range(3):
                directory = root / "endpoints" / f"worker-{worker_index}"
                directory.mkdir(parents=True)
                entries = []
                for record in records:
                    if record["worker_index"] != worker_index:
                        continue
                    index = record["validation_index"]
                    trace_path = directory / (record["version"] + "-" + str(index) + ".json")
                    trace = dict(native_result_unmodified=True, hooks_restored=True, version=record["version"], validation_index=index,
                                 question_id=rows[index]["question_id"], question=rows[index]["question"], answers=rows[index]["answers"],
                                 native_result=native, protocol=protocol, generation_options_modified=False, return_objects_modified=False)
                    atomic_json(trace_path, trace)
                    record = dict(record, native_result_sha256=object_sha(native), protocol_em=1, protocol_valid_final_action=True,
                                  trace=dict(path=str(trace_path), sha256=file_sha(trace_path)), null_effect_equivalence=dict(accepted=True))
                    entries.append(record)
                results_path = directory / "results.jsonl"
                results_path.write_bytes(b"".join(canonical(record) + b"\n" for record in entries))
                atomic_json(directory / "status.json", dict(status="COMPLETE", phase="endpoints", worker_index=worker_index,
                            final_source_weights_validation_science_publication_admission_verified=True,
                            expected_model_question_slots=len(entries), completed_model_question_slots=len(entries),
                            results_sha256=file_sha(results_path), config=dict(sha256=file_sha(config_path))))
            with patch("importlib.import_module", return_value=self.core):
                merged, summary = controller_module.phase_records(root, "endpoints", config, rows)
            self.assertEqual(len(merged), 1024)
            self.assertEqual([item["worker_index"] for item in summary["workers"]], [0, 1, 2])
            self.assertEqual(summary["versions"]["source100"]["denominator"], 512)
            self.assertEqual(summary["versions"]["epoch1"]["correct"], 512)


if __name__ == "__main__":
    unittest.main()
