"""CPU-only structural/provenance tests; synthetic fixtures are never experiments."""
import json
from pathlib import Path
import tempfile
import unittest

if __package__:
    from .select_cases import select
else:
    from select_cases import select


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rollouts, self.pairs, self.train = [], [], []
        for dataset in ("nq", "hotpotqa"):
            for group in ("correct_correct", "correct_wrong", "all5_wrong"):
                for index in range(4):
                    question_id = f"{dataset}:train:{group}:{index}"
                    question = dict(question_id=question_id, question="Fixture " + question_id,
                                    answers=["ok"], data_source=dataset)
                    self.train.append(question)
                    siblings = []
                    for sibling in range(5):
                        correct = group == "correct_correct" or group == "correct_wrong" and sibling == 0
                        response = f"<think>fixture {sibling}</think><answer>{'ok' if correct else 'bad'}</answer>"
                        row = dict(question, sibling=sibling, reward=int(correct), response=response,
                                   policy_spans=[[0, len(response)]], search_calls=0,
                                   token_count=len(response), termination="answer")
                        siblings.append(row)
                        self.rollouts.append(row)
                    if group != "all5_wrong":
                        self.pairs.append(dict(question_id=question_id, question=question["question"], answers=["ok"],
                                               reference=siblings[0]["response"], student=siblings[1]["response"],
                                               reference_reward=1, student_reward=int(group == "correct_correct"),
                                               student_policy_spans=siblings[1]["policy_spans"]))
        self.write_inputs()

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self):
        for name, rows in [("rollouts", self.rollouts), ("pairs", self.pairs), ("train", self.train)]:
            (self.root / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def strict_fixture_reward(response, answers, spans):
        if spans != [[0, len(response)]]:
            raise ValueError("Fixture policy provenance mismatch")
        return int(response.endswith("<answer>ok</answer>"))

    def build(self, name="selected", per_cell=2):
        return select(self.root / "rollouts.jsonl", self.root / "pairs.jsonl", self.root / "train.jsonl",
                      self.root / name, per_cell=per_cell, seed=42,
                      expected_questions=24, expected_pairs=len(self.pairs), reward=self.strict_fixture_reward)

    def test_six_cells_repeatable_source_provenance_and_hard_no_teacher(self):
        first = self.build()
        second = self.build("selected-again")
        self.assertEqual(first["selected_cases"], 12)
        self.assertEqual(first["cases"]["sha256"], second["cases"]["sha256"])
        self.assertTrue(first["strict_reward_checked"])
        cases = [json.loads(line) for line in Path(first["cases"]["path"]).read_text().splitlines()]
        for dataset in ("nq", "hotpotqa"):
            for group in ("correct_correct", "correct_wrong", "all5_wrong"):
                self.assertEqual(sum(case["dataset"] == dataset and case["group"] == group for case in cases), 2)
        for case in cases:
            self.assertEqual(len(case["original_siblings"]), 5)
            self.assertTrue(all(len(sibling["provenance"]["row_sha256"]) == 64 for sibling in case["original_siblings"]))
            self.assertEqual(len(case["source_train_row"]["provenance"]["semantic_sha256"]), 64)
            if case["group"] == "all5_wrong":
                self.assertIsNone(case["original_pair"])
                self.assertIn("not_applicable", case["teacher_reference_status"])
            else:
                self.assertEqual(case["original_pair"]["row"]["reference_reward"], 1)

    def test_incomplete_sibling_group_rejected(self):
        self.rollouts.pop(1)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "five-sibling"):
            self.build()

    def test_duplicate_sibling_rejected(self):
        self.rollouts[1]["sibling"] = 0
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "five-sibling"):
            self.build()

    def test_false_successful_reference_rejected(self):
        self.pairs[0]["reference"] = "<answer>not from the original siblings</answer>"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Pair response/provenance"):
            self.build(per_cell=4)

    def test_hard_pair_cannot_be_fabricated(self):
        row = next(row for row in self.rollouts if "all5_wrong" in row["question_id"])
        self.pairs.append(dict(question_id=row["question_id"], question=row["question"], answers=row["answers"],
                               reference=row["response"], student=row["response"], reference_reward=1,
                               student_reward=0, student_policy_spans=row["policy_spans"]))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Hard excluded"):
            self.build(per_cell=4)

    def test_reward_provenance_is_checked(self):
        self.rollouts[0]["policy_spans"] = []
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.build(per_cell=4)

    def test_partial_immutable_line_rejected(self):
        path = self.root / "rollouts.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        with self.assertRaisesRegex(ValueError, "partial line"):
            self.build()

    def test_existing_output_not_overwritten(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()


if __name__ == "__main__":
    unittest.main()
