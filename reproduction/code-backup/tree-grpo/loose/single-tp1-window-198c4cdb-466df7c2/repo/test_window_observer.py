"""Tiny CPU fixtures; no actual checkpoints, historical PTs, vLLM, or CUDA."""
import unittest
from types import SimpleNamespace as NS

import torch

from window_observer import WindowObserver


def probability(value):
    return NS(logprob=float(value), rank=1, decoded_token=None)


def settings(prompt_logprobs=None):
    return NS(n=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
              presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
              min_tokens=0, max_tokens=500, logprobs=1,
              prompt_logprobs=prompt_logprobs, seed=None, logits_processors=None)


def group(seq_id, prompt, prior, sample_row, is_prompt, prompt_rows=()):
    return NS(seq_ids=[seq_id], seq_data={seq_id: NS(prompt_token_ids=tuple(prompt),
                                                   output_token_ids=tuple(prior))},
              sampling_params=settings(1 if prompt_rows else None), is_prompt=is_prompt,
              sample_indices=[sample_row], prompt_logprob_indices=list(prompt_rows))


def model_input(groups, request_map, selected, query_lens):
    tokens, positions = [], []
    for grp, query in zip(groups, query_lens):
        data = grp.seq_data[grp.seq_ids[0]]
        context = list(data.prompt_token_ids+data.output_token_ids)
        tokens.extend(context[-query:])
        positions.extend(range(len(context)-query, len(context)))
    metadata = NS(seq_groups=groups, selected_token_indices=torch.tensor(selected),
                  skip_sampler_cpu_output=False)
    return NS(sampling_metadata=metadata, input_tokens=torch.tensor(tokens),
              input_positions=torch.tensor(positions), query_lens=query_lens,
              request_ids_to_seq_ids=request_map)


class Processor:
    def __init__(self):
        self.calls = 0
        self.matrix = None

    def _get_logits(self):
        self.calls += 1
        return self.matrix.clone()


class Sampler:
    def __init__(self):
        self.calls = 0
        self.targets = {}
        self.fail = False

    def forward(self, logits, sampling_metadata):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Native sampler fixture failure")
        distributions = logits.float().log_softmax(-1)
        # The native sampler may mutate incoming logits. Observers must clone.
        logits.add_(100)
        outputs = []
        for grp in sampling_metadata.seq_groups:
            samples = []
            for parent, seq_id in enumerate(grp.seq_ids):
                token = self.targets[seq_id]
                row = grp.sample_indices[0]+parent
                samples.append(NS(parent_seq_id=seq_id, output_token=token,
                                  logprobs={token: probability(distributions[row, token])}))
            outputs.append(NS(samples=samples, prompt_logprobs=None))
        return NS(outputs=outputs)


def completion(request_id, prompt, samples, finish_reason="length", prompt_logprobs=None):
    return NS(request_id=request_id, prompt_token_ids=prompt, prompt_logprobs=prompt_logprobs,
              finished=True, outputs=[NS(index=0, token_ids=[s.output_token for s in samples],
                                         logprobs=[s.logprobs for s in samples],
                                         finish_reason=finish_reason, stop_reason=None,
                                         cumulative_logprob=sum(s.logprobs[s.output_token].logprob for s in samples))])


class ObserverTest(unittest.TestCase):
    def observer(self, anchors):
        model = NS(logits_processor=Processor(), sampler=Sampler())
        obs = WindowObserver(model, {"fixture_only": "not_actual_runtime_source"}, anchors, 0)
        return model, obs

    def run_step(self, obs, model, native_input, matrix, targets):
        model.logits_processor.matrix = torch.tensor(matrix, dtype=torch.bfloat16)
        model.sampler.targets = targets
        obs.observe_model_input(native_input)
        logits = model.logits_processor._get_logits()
        output = model.sampler.forward(logits, native_input.sampling_metadata)
        return output, logits

    def test_duplicate_prompts_and_changed_group_order_use_actual_ids(self):
        anchors = [{"id": "b0", "rank": 0, "input_row": 1, "response_offset": 0},
                   {"id": "a1", "rank": 0, "input_row": 0, "response_offset": 1}]
        model, obs = self.observer(anchors)
        obs.observe_prompts([[1, 2], [1, 2]])
        obs.register_request("17", [1, 2], 0)
        obs.register_request("31", [1, 2], 1)
        with obs:
            native = model_input([group(42, [1, 2], [], 0, True), group(77, [1, 2], [], 1, True)],
                                 {"31": [42], "17": [77]}, [1, 3], [2, 2])
            first, mutated = self.run_step(obs, model, native, [[1, 2, 3, 4], [5, 6, 7, 8]], {42: 3, 77: 1})
            b0, a0 = first.outputs[0].samples[0], first.outputs[1].samples[0]
            native = model_input([group(77, [1, 2], [1], 0, False), group(42, [1, 2], [3], 1, False)],
                                 {"17": [77], "31": [42]}, [0, 1], [1, 1])
            second, _ = self.run_step(obs, model, native, [[9, 10, 11, 12], [13, 14, 15, 16]], {77: 2, 42: 0})
            a1, b1 = second.outputs[0].samples[0], second.outputs[1].samples[0]
            # Completion order does not need to equal input row or sample group order.
            obs.observe_generated([completion("31", [1, 2], [b0, b1]), completion("17", [1, 2], [a0, a1])])
        receipt = obs.receipt()
        self.assertTrue(receipt["observer_evidence_complete"], receipt["errors"])
        records = {record["id"]: record for record in receipt["anchors"]}
        self.assertEqual(records["b0"]["request_id"], "31")
        self.assertEqual(records["a1"]["context_token_ids"], [1, 2, 1])
        self.assertEqual(records["b0"]["context_positions"], [0, 1])
        self.assertEqual(records["a1"]["context_positions"], [0, 1, 2])
        self.assertEqual(records["a1"]["input_position"], 2)
        tensors = obs.tensor_evidence()
        self.assertTrue(torch.equal(tensors["b0"]["head_logits"], torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(tensors["b0"]["head_logits"], tensors["b0"]["sampler_input_logits"]))
        self.assertFalse(torch.equal(mutated[0], tensors["b0"]["head_logits"]))
        self.assertEqual(model.logits_processor.calls, 2)
        self.assertEqual(model.sampler.calls, 2)
        self.assertNotIn("forward", vars(model.sampler))
        self.assertNotIn("_get_logits", vars(model.logits_processor))

    def test_prompt_logprob_rows_are_not_sample_rows(self):
        model, obs = self.observer([{"id": "last", "rank": 0, "input_row": 0, "response_offset": 0}])
        obs.observe_prompts([[1, 2, 3]])
        obs.register_request("9", [1, 2, 3], 0)
        with obs:
            native = model_input([group(40, [1, 2, 3], [], 2, True, (0, 1))], {"9": [40]}, [0, 1, 2], [3])
            out, _ = self.run_step(obs, model, native, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], {40: 0})
            obs.observe_generated([completion("9", [1, 2, 3], out.outputs[0].samples,
                                             prompt_logprobs=[None, {2: probability(-1)}, {3: probability(-2)}])])
        record = obs.receipt()["anchors"][0]
        self.assertTrue(obs.receipt()["observer_evidence_complete"])
        self.assertEqual(record["native_logits_row"], 2)
        self.assertEqual(record["hidden_state_index"], 2)
        self.assertEqual(obs.receipt()["generated"][0]["prompt_logprobs"][0], None)

    def test_missing_mapping_defers_error_and_calls_native_once(self):
        model, obs = self.observer([{"id": "a", "rank": 0, "input_row": 0, "response_offset": 0}])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            native = model_input([group(1, [1, 2], [], 0, True)], None, [1], [2])
            out, _ = self.run_step(obs, model, native, [[1, 2, 3]], {1: 0})
            obs.observe_generated([completion("1", [1, 2], out.outputs[0].samples)])
        self.assertFalse(obs.receipt()["observer_evidence_complete"])
        self.assertEqual(model.sampler.calls, 1)
        self.assertEqual(model.logits_processor.calls, 1)

    def test_native_failure_propagates_and_methods_restore(self):
        model, obs = self.observer([])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        model.sampler.fail = True
        with self.assertRaisesRegex(RuntimeError, "Native sampler"):
            with obs:
                self.run_step(obs, model, model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2]), [[1, 2]], {1: 0})
        self.assertEqual(model.sampler.calls, 1)
        self.assertNotIn("forward", vars(model.sampler))
        self.assertFalse(obs.receipt()["observer_evidence_complete"])

    def test_early_native_stop_marks_unreached_anchor_absent(self):
        model, obs = self.observer([{"id": "late", "rank": 0, "input_row": 0, "response_offset": 10}])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            out, _ = self.run_step(obs, model, model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2]), [[1, 2]], {1: 0})
            obs.observe_generated([completion("1", [1, 2], out.outputs[0].samples, "stop")])
        self.assertTrue(obs.receipt()["observer_evidence_complete"], obs.receipt()["errors"])
        self.assertEqual(obs.receipt()["anchors"][0]["status"], "absent_native_early_finish")
        self.assertEqual(obs.tensor_evidence(), {})

    def test_length_finish_cannot_explain_missing_anchor(self):
        model, obs = self.observer([{"id": "late", "rank": 0, "input_row": 0, "response_offset": 10}])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            out, _ = self.run_step(obs, model, model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2]), [[1, 2]], {1: 0})
            obs.observe_generated([completion("1", [1, 2], out.outputs[0].samples)])
        self.assertFalse(obs.receipt()["observer_evidence_complete"])

    def test_bad_request_binding_and_unseen_input_reject(self):
        model, obs = self.observer([])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [9, 9], 0)
        obs.register_request("2", [1, 2], 1)
        with obs:
            pass
        self.assertFalse(obs.receipt()["observer_evidence_complete"])
        self.assertEqual(len(obs.requests), 0)

    def test_cached_context_mutation_is_not_guessed(self):
        model, obs = self.observer([{"id": "a", "rank": 0, "input_row": 0, "response_offset": 0}])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            native = model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2])
            obs.observe_model_input(native)
            model.logits_processor.matrix = torch.tensor([[1, 2]])
            logits = model.logits_processor._get_logits()
            native.sampling_metadata.seq_groups[0].seq_data[1].output_token_ids = (7,)
            model.sampler.targets = {1: 0}
            out = model.sampler.forward(logits, native.sampling_metadata)
            obs.observe_generated([completion("1", [1, 2], out.outputs[0].samples)])
        self.assertFalse(obs.receipt()["observer_evidence_complete"])
        self.assertEqual(model.sampler.calls, 1)

    def test_unobserved_cached_prefix_refuses_full_positions(self):
        model, obs = self.observer([{"id": "cached", "rank": 0, "input_row": 0, "response_offset": 1}])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            # This is the first observation: position 2 is real, 0 and 1 are
            # cached and have never been seen by this observer.
            native = model_input([group(1, [1, 2], [3], 0, False)], {"1": [1]}, [0], [1])
            self.run_step(obs, model, native, [[1, 2, 3, 4]], {1: 0})
        self.assertFalse(obs.receipt()["observer_evidence_complete"])
        self.assertTrue(any("Unobserved cached prefix" in error for error in obs.errors))
        self.assertNotIn("cached", obs.anchor_records)
        self.assertEqual(model.logits_processor.calls, 1)
        self.assertEqual(model.sampler.calls, 1)

    def test_repeated_query_index_changed_token_defers_refusal(self):
        model, obs = self.observer([])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            self.run_step(obs, model, model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2]), [[1, 2, 3, 4]], {1: 3})
            self.run_step(obs, model, model_input([group(1, [1, 2], [3], 0, False)], {"1": [1]}, [0], [1]), [[1, 2, 3, 4]], {1: 0})
            # Native position 2 was already observed with token 3. Replaying
            # the same index with a changed token cannot inherit that evidence.
            self.run_step(obs, model, model_input([group(1, [1, 2], [2], 0, False)], {"1": [1]}, [0], [1]), [[1, 2, 3, 4]], {1: 0})
        self.assertFalse(obs.receipt()["observer_evidence_complete"])
        self.assertTrue(any("Repeated native context index changed" in error for error in obs.errors))
        self.assertEqual(model.logits_processor.calls, 3)
        self.assertEqual(model.sampler.calls, 3)

    def test_repeated_query_changed_position_defers_refusal(self):
        model, obs = self.observer([])
        obs.observe_prompts([[1, 2]])
        obs.register_request("1", [1, 2], 0)
        with obs:
            self.run_step(obs, model, model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2]), [[1, 2, 3]], {1: 0})
            replay = model_input([group(1, [1, 2], [], 0, True)], {"1": [1]}, [1], [2])
            replay.input_positions[1] = 8
            self.run_step(obs, model, replay, [[1, 2, 3]], {1: 0})
        self.assertFalse(obs.receipt()["observer_evidence_complete"])
        self.assertTrue(any("Native token/position segment differs" in error for error in obs.errors))
        self.assertEqual(model.sampler.calls, 2)


if __name__ == "__main__":
    unittest.main()
