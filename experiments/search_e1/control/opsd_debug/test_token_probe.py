"""CPU toy model checks actual frozen loss, mask/head shapes and bounded dumps.

Only audited math functions are extracted from the archived source; its older
reward function is never imported or used. Toy data/model are test fixtures.
"""
import ast
import contextlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.utils.checkpoint import checkpoint

if __package__:
    from .token_probe import FullVocabularyBudget, probe
else:
    from token_probe import FullVocabularyBudget, probe


def frozen_math():
    root = Path(__file__).resolve().parents[2] / "archive/search_e1"
    namespace = dict(torch=torch, checkpoint=checkpoint, re=re)
    names = {"core.py": ["prompt", "information_spans", "response_encoding", "clipped_forward_kl"],
             "opsd.py": ["encoded_pair", "pair_loss"]}
    for filename, functions in names.items():
        source = (root / filename).read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "INSTRUCTION" for target in node.targets):
                namespace["INSTRUCTION"] = ast.literal_eval(node.value)
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                snippet = "\n".join(source.splitlines()[node.lineno-1:node.end_lineno])
                exec(compile(snippet, str(root / filename), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names["core.py"]}), SimpleNamespace(**{name: namespace[name] for name in names["opsd.py"]})


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(character) % 64 for character in text],
                    offset_mapping=[(index, index+1) for index in range(len(text))])

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        text = messages[0]["content"] + "assistant:"
        return self(text)["input_ids"] if tokenize else text

    def convert_ids_to_tokens(self, token):
        return "fixture-token-" + str(token)

    def decode(self, tokens):
        return "".join(chr(token + 32) for token in tokens)


class TinyTransformer(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.embedding = torch.nn.Embedding(64, 4)

    def forward(self, input_ids, use_cache=False):
        hidden = self.embedding(input_ids).cumsum(dim=1) * 0.01
        if self.state.active:
            hidden = hidden + torch.tensor([0.2, -0.1, 0.05, 0.0])
        return SimpleNamespace(last_hidden_state=hidden)


class ShapeHead(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 64, bias=False)
        self.shapes = []

    def forward(self, inputs):
        self.shapes.append(tuple(inputs.shape))
        return super().forward(inputs)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.state = SimpleNamespace(active=True)
        self.transformer = TinyTransformer(self.state)
        self.head = ShapeHead()

    def get_base_model(self):
        return SimpleNamespace(base_model=self.transformer, get_output_embeddings=lambda: self.head)

    @contextlib.contextmanager
    def disable_adapter(self):
        previous = self.state.active
        self.state.active = False
        try:
            yield
        finally:
            self.state.active = previous


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model, self.tokenizer = TinyModel(), TinyTokenizer()
        self.core, self.opsd = frozen_math()
        self.response = "<think>" + "x" * 38 + "</think>\n<information>document text</information>\n<answer>ok</answer>"
        start = self.response.index("<information>")
        end = self.response.index("</information>") + len("</information>")
        self.spans = [[0, start], [end, len(self.response)]]

    def tearDown(self):
        self.temporary.cleanup()

    def run_probe(self, reference="successful fixture reference", budget=None, question="fixture", name="probe"):
        budget = budget or FullVocabularyBudget(0, 3, [], [])
        return probe(self.model, self.tokenizer, self.core, self.opsd, self.response, self.spans,
                     question, reference, "epoch1", "fixture-case", self.root / name, budget)

    def test_all_token_offsets_masks_loss_and_native_head_shapes(self):
        summary = self.run_probe()
        rows = [json.loads(line) for line in Path(summary["tokens_file"]["path"]).read_text().splitlines()]
        self.assertEqual([row["token_index"] for row in rows], list(range(len(self.response))))
        self.assertTrue(summary["frozen_pair_loss_check"]["passed"])
        self.assertEqual(summary["trajectory_mean"], summary["frozen_pair_loss"])
        self.assertTrue(summary["frozen_pair_loss_check"]["per_token_contribution_sum_matches"])
        for row in rows:
            if row["information_mask"]:
                self.assertFalse(row["policy_mask"])
                self.assertEqual(row["actual_masked_trajectory_loss_contribution"], 0)
        valid = summary["valid_policy_tokens"]
        expected_rows = [32] * (valid // 32) + ([valid % 32] if valid % 32 else [])
        first_policy_heads = self.model.head.shapes[:3 * len(expected_rows)]
        self.assertEqual([shape[0] for shape in first_policy_heads], [size for size in expected_rows for _ in range(3)])
        final_frozen_heads = self.model.head.shapes[-2 * len(expected_rows):]
        self.assertEqual([shape[0] for shape in final_frozen_heads], [size for size in expected_rows for _ in range(2)])

    def test_hard_case_has_no_teacher_or_opsd_loss(self):
        summary = self.run_probe(reference=None)
        self.assertIsNone(summary["trajectory_mean"])
        self.assertIn("not_applicable", summary["frozen_pair_loss_check"]["status"])
        rows = [json.loads(line) for line in Path(summary["tokens_file"]["path"]).read_text().splitlines()]
        self.assertTrue(all("teacher_privileged" not in row["distributions"] for row in rows))
        self.assertTrue(all(row["teacher_student_clipped_kl"] is None for row in rows))

    def test_joint_context_skip_without_independent_truncation(self):
        summary = self.run_probe(question="x" * 8200)
        self.assertEqual(summary["status"], "skipped")
        self.assertFalse((self.root / "probe/tokens.jsonl").exists())

    def test_full_vocabulary_cap_skip_keeps_all_token_losses(self):
        budget = FullVocabularyBudget(1, 3, ["fixture-case"], ["epoch1"])
        summary = self.run_probe(budget=budget)
        self.assertEqual(summary["full_vocabulary_dump"]["status"], "cap_skip")
        self.assertEqual(summary["tokens_file"]["rows"], len(self.response))
        self.assertTrue(summary["frozen_pair_loss_check"]["passed"])

    def test_full_vocabulary_dump_has_full_finite_fp32_shapes(self):
        import numpy as np
        budget = FullVocabularyBudget(8 << 20, 3, ["fixture-case"], ["epoch1"])
        summary = self.run_probe(budget=budget)
        dump = summary["full_vocabulary_dump"]
        self.assertEqual(dump["status"], "complete")
        self.assertEqual(len(dump["files"]), 6)
        for file in dump["files"].values():
            array = np.load(file["path"], mmap_mode="r")
            self.assertEqual(array.shape, (len(self.response), 64))
            self.assertEqual(array.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(array).all())
            values = array if file["quantity"] == "probabilities" else np.exp(array)
            self.assertTrue(np.allclose(values.sum(axis=1), 1.0))
        self.assertLessEqual(budget.used, budget.cap)

    def test_new_generated_response_is_separate_diagnostic_not_training_loss(self):
        budget = FullVocabularyBudget(8 << 20, 3, ["fixture-case"], ["epoch1"])
        summary = probe(self.model, self.tokenizer, self.core, self.opsd, self.response, self.spans,
                        "fixture", "successful fixture reference", "epoch1", "fixture-case", self.root / "new-response",
                        budget, response_origin="new_student_generation")
        self.assertFalse(summary["actual_autoregressive_generation"])
        self.assertFalse(summary["original_training_objective"])
        self.assertEqual(summary["response_origin"], "new_student_generation")
        self.assertTrue(summary["frozen_pair_loss_check"]["passed"])
        self.assertEqual(summary["full_vocabulary_dump"]["status"], "not_requested_generated_response_scope")
        self.assertEqual(budget.used, 0)


if __name__ == "__main__":
    unittest.main()
