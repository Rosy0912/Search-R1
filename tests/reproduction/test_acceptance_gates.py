"""Execute real acceptance scripts on tiny CPU safetensors and run-local logs.

Only hardcoded project paths are replaced through AST. No verifier algorithm,
safe_open implementation, GPU, network, or fake tensor reader is substituted.
"""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[2]
ADAPTATION = Path(os.environ.get("SEARCH_R1_ADAPTATION_DIR", (ROOT / "reproduction/h20" if (ROOT / "reproduction/h20").exists() else ROOT.parent)))
SOURCES = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
PARAM = "model.layers.0.self_attn.q_proj.weight"


class AcceptanceGatesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.started = time.time() - 10
        self.smoke = self.base / "checkpoints/search-r1-reviewed-smoke/actor/global_step_2"
        self.full = self.base / "checkpoints/paper-grpo-qwen2.5-3b/actor/global_step_500"
        self.weights(self.base / "assets/qwen2.5-3b", 1)
        self.weights(self.smoke, 2)
        self.weights(self.full, 2)
        audit = self.base / "data/paper-grpo/dataset-audit.json"
        audit.parent.mkdir(parents=True)
        audit.write_text(json.dumps({"eval_counts": {s: 2 for s in SOURCES}}))
        self.metrics = {"val/evaluated_count/" + s: 2 for s in SOURCES}
        self.metrics.update({"val/test_score/" + s: 0.5 for s in SOURCES})
        self.log = self.base / "logs/paper-grpo-qwen2.5-3b.log"
        self.log.parent.mkdir(parents=True)
        self.write_metrics()

    def tearDown(self):
        self.directory.cleanup()

    def weights(self, path, value):
        path.mkdir(parents=True, exist_ok=True)
        mapping = {PARAM: "model-00001-of-00002.safetensors", "second.weight": "model-00002-of-00002.safetensors"}
        for key, file in mapping.items():
            save_file({key: np.full((2, 2), value, dtype=np.float32)}, str(path / file))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))

    def write_metrics(self):
        self.log.write_text("SEARCH_R1_FINAL_METRICS_JSON=" + json.dumps(self.metrics) + "\n")

    def run_gate(self, name):
        tree = ast.parse((ADAPTATION / (name + ".py")).read_text())
        replacement = str(self.base)

        class PathsOnly(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value.startswith("/data/search-r1"):
                    return ast.copy_location(ast.Constant(node.value.replace("/data/search-r1", replacement, 1)), node)
                return node

        tree = ast.fix_missing_locations(PathsOnly().visit(tree))
        with patch.dict(os.environ, {"SEARCH_R1_RUN_STARTED": str(self.started)}), redirect_stdout(io.StringIO()):
            exec(compile(tree, str(ADAPTATION / (name + ".py")), "exec"), {"__name__": "__main__"})

    def test_current_smoke_checkpoint_passes(self):
        self.run_gate("verify_checkpoint")

    def test_stale_smoke_shard_is_rejected(self):
        shard = self.smoke / "model-00002-of-00002.safetensors"
        os.utime(shard, (self.started - 100, self.started - 100))
        with self.assertRaises(AssertionError):
            self.run_gate("verify_checkpoint")

    def test_unchanged_smoke_parameter_is_rejected(self):
        self.weights(self.smoke, 1)
        with self.assertRaises(AssertionError):
            self.run_gate("verify_checkpoint")

    def test_complete_final_checkpoint_and_seven_evals_pass(self):
        self.run_gate("verify_full_run")

    def test_missing_final_shard_is_rejected(self):
        (self.full / "model-00002-of-00002.safetensors").unlink()
        with self.assertRaises(AssertionError):
            self.run_gate("verify_full_run")

    def test_corrupt_final_shard_is_rejected(self):
        (self.full / "model-00002-of-00002.safetensors").write_bytes(b"corrupt header")
        with self.assertRaises(Exception):
            self.run_gate("verify_full_run")

    def test_stale_final_shard_is_rejected(self):
        shard = self.full / "model-00002-of-00002.safetensors"
        os.utime(shard, (self.started - 100, self.started - 100))
        with self.assertRaises(AssertionError):
            self.run_gate("verify_full_run")

    def test_incomplete_dataset_evaluation_is_rejected(self):
        self.metrics["val/evaluated_count/musique"] = 1
        self.write_metrics()
        with self.assertRaises(AssertionError):
            self.run_gate("verify_full_run")

    def test_missing_dataset_metric_is_rejected(self):
        del self.metrics["val/test_score/bamboogle"]
        self.write_metrics()
        with self.assertRaises(KeyError):
            self.run_gate("verify_full_run")

    def test_invalid_em_is_rejected(self):
        self.metrics["val/test_score/nq"] = 1.01
        self.write_metrics()
        with self.assertRaises(AssertionError):
            self.run_gate("verify_full_run")

    def test_missing_final_marker_is_rejected(self):
        self.log.write_text("step:501 val/test_score/nq:0.5\n")
        with self.assertRaises(AssertionError):
            self.run_gate("verify_full_run")


if __name__ == "__main__":
    unittest.main()
