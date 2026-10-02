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

    def prepare_five(self):
        self.five=self.base/'checkpoints/paper-grpo-qwen2.5-3b/actor/global_step_5'
        self.weights(self.five,2)
        (self.base/'logs/training-run-started.txt').write_text(str(self.started))
        self.log.write_text(''.join(f'step:{i} - critic/rewards/mean:0.125 - actor/grad_norm:0.300\n' for i in range(1,7)))

    def test_five_real_updates_fresh_weights_pass(self):
        self.prepare_five()
        self.run_gate('verify_five_updates')

    def test_fewer_than_five_updates_is_rejected(self):
        self.prepare_five()
        self.log.write_text('step:5 - val/test_score/nq:0.125\n')
        with self.assertRaises(AssertionError):
            self.run_gate('verify_five_updates')

    def test_nonfinite_update_is_rejected(self):
        self.prepare_five()
        self.log.write_text(self.log.read_text().replace('actor/grad_norm:0.300','actor/grad_norm:nan'))
        with self.assertRaises(AssertionError):
            self.run_gate('verify_five_updates')

    def test_stale_five_checkpoint_is_rejected(self):
        self.prepare_five()
        shard=self.five/'model-00002-of-00002.safetensors'
        os.utime(shard,(self.started-100,self.started-100))
        with self.assertRaises(AssertionError):
            self.run_gate('verify_five_updates')

    def mutate_checkpoint(self, gate, mutation):
        if gate=='verify_five_updates':
            self.prepare_five()
            directory=self.five
        else:
            directory=self.full
        index=directory/'model.safetensors.index.json'
        mapping=json.loads(index.read_text())['weight_map']
        if mutation=='omit':
            # Self-consistent index+remaining shard: old gates accepted this.
            missing=mapping.pop('second.weight')
            (directory/missing).unlink()
        elif mutation=='shape':
            save_file({'second.weight':np.full((3,2),2,dtype=np.float32)},str(directory/mapping['second.weight']))
        elif mutation=='nan':
            # Keep the sampled q_proj finite/changed; corrupt another tensor.
            save_file({'second.weight':np.full((2,2),np.nan,dtype=np.float32)},str(directory/mapping['second.weight']))
        elif mutation=='reshard':
            mapping={key:'model-all.safetensors' for key in mapping}
            save_file({key:np.full((2,2),2,dtype=np.float32) for key in mapping},str(directory/'model-all.safetensors'))
        else:
            raise AssertionError(mutation)
        index.write_text(json.dumps({'weight_map':mapping}))

    def test_five_self_consistent_omitted_parameter_is_rejected(self):
        self.mutate_checkpoint('verify_five_updates','omit')
        with self.assertRaisesRegex(AssertionError,'keys differ'):
            self.run_gate('verify_five_updates')

    def test_full_self_consistent_omitted_parameter_is_rejected(self):
        self.mutate_checkpoint('verify_full_run','omit')
        with self.assertRaisesRegex(AssertionError,'keys differ'):
            self.run_gate('verify_full_run')

    def test_five_wrong_parameter_shape_is_rejected(self):
        self.mutate_checkpoint('verify_five_updates','shape')
        with self.assertRaisesRegex(AssertionError,'shape differs'):
            self.run_gate('verify_five_updates')

    def test_full_wrong_parameter_shape_is_rejected(self):
        self.mutate_checkpoint('verify_full_run','shape')
        with self.assertRaisesRegex(AssertionError,'shape differs'):
            self.run_gate('verify_full_run')

    def test_five_non_sampled_parameter_nan_is_rejected(self):
        self.mutate_checkpoint('verify_five_updates','nan')
        with self.assertRaisesRegex(AssertionError,'Nonfinite checkpoint tensor'):
            self.run_gate('verify_five_updates')

    def test_full_non_sampled_parameter_nan_is_rejected(self):
        self.mutate_checkpoint('verify_full_run','nan')
        with self.assertRaisesRegex(AssertionError,'Nonfinite checkpoint tensor'):
            self.run_gate('verify_full_run')

    def test_five_valid_reshard_is_accepted(self):
        self.mutate_checkpoint('verify_five_updates','reshard')
        self.run_gate('verify_five_updates')

    def test_full_valid_reshard_is_accepted(self):
        self.mutate_checkpoint('verify_full_run','reshard')
        self.run_gate('verify_full_run')

    def prepare_tied_qwen(self, gate, same_values=True, tied=True):
        if gate=='verify_five_updates':
            self.prepare_five()
            directory=self.five
        else:
            directory=self.full
        canonical=self.base/'assets/qwen2.5-3b'
        embedding='model.embed_tokens.weight'
        alias='lm_head.weight'
        config={'model_type':'qwen2','architectures':['Qwen2ForCausalLM'],
                'tie_word_embeddings':True}
        for path,value in [(canonical,1),(directory,2)]:
            index=path/'model.safetensors.index.json'
            mapping=json.loads(index.read_text())['weight_map']
            mapping[embedding]='embedding.safetensors'
            save_file({embedding:np.full((2,2),value,dtype=np.float32)},str(path/'embedding.safetensors'))
            if path==directory:
                mapping[alias]='head.safetensors'
                save_file({alias:np.full((2,2),value if same_values else value+1,dtype=np.float32)},str(path/'head.safetensors'))
            index.write_text(json.dumps({'weight_map':mapping}))
            settings=dict(config)
            if path==directory and not tied:
                settings['tie_word_embeddings']=False
            (path/'config.json').write_text(json.dumps(settings))

    def test_five_equal_configured_tied_qwen_duplicate_is_accepted(self):
        self.prepare_tied_qwen('verify_five_updates')
        self.run_gate('verify_five_updates')

    def test_full_equal_configured_tied_qwen_duplicate_is_accepted(self):
        self.prepare_tied_qwen('verify_full_run')
        self.run_gate('verify_full_run')

    def test_five_unequal_tied_qwen_duplicate_is_rejected(self):
        self.prepare_tied_qwen('verify_five_updates',same_values=False)
        with self.assertRaisesRegex(AssertionError,'Inconsistent tied'):
            self.run_gate('verify_five_updates')

    def test_full_unequal_tied_qwen_duplicate_is_rejected(self):
        self.prepare_tied_qwen('verify_full_run',same_values=False)
        with self.assertRaisesRegex(AssertionError,'Inconsistent tied'):
            self.run_gate('verify_full_run')

    def test_five_untied_config_cannot_whitelist_extra_head(self):
        self.prepare_tied_qwen('verify_five_updates',tied=False)
        with self.assertRaisesRegex(AssertionError,'keys differ'):
            self.run_gate('verify_five_updates')

    def test_full_untied_config_cannot_whitelist_extra_head(self):
        self.prepare_tied_qwen('verify_full_run',tied=False)
        with self.assertRaisesRegex(AssertionError,'keys differ'):
            self.run_gate('verify_full_run')


if __name__ == "__main__":
    unittest.main()
