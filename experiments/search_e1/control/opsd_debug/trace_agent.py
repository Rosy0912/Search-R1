"""Instrument the frozen HF SearchAgent's control flow without editing it."""
import re
import time


class TraceSearchAgent:
    def __init__(self, model, tokenizer, core, rollout, url, check_health=None):
        self.model, self.tokenizer, self.core, self.rollout, self.url = model, tokenizer, core, rollout, url
        self.max_new_tokens, self.max_context = 500, 8192
        self.check_health = check_health

    def run(self, question, answers, reference=None):
        import requests
        import torch
        prefix = self.tokenizer.apply_chat_template(self.core.prompt(question, reference),
                                                   tokenize=False, add_generation_prompt=True)
        self.model.eval()
        response, calls, spans, turns = "", 0, [], []

        def finished(termination):
            return dict(kind="actual_autoregressive_generation", privileged_reference=reference is not None,
                        prompt_text=prefix, response=response, policy_spans=spans, search_calls=calls,
                        termination=termination, protocol_em=self.core.reward(response, answers, spans),
                        temperature=0.0, top_p=None, max_new_tokens_per_turn=500,
                        max_context=8192, max_search_calls=4, topk_documents=3,
                        turns=turns, teacher_forced=False)

        with torch.no_grad():
            for turn in range(5):
                forced = "\n<information>Search budget exhausted. Give the final answer now inside <answer> and </answer>.</information>\n" if turn == 4 else ""
                text = prefix + response + forced
                ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(next(self.model.parameters()).device)
                event = dict(turn=turn, context_tokens=int(ids.shape[1]), forced_information=forced,
                             context_text=text, generation_started_unix=time.time())
                turns.append(event)
                if ids.shape[1] + self.max_new_tokens > self.max_context:
                    event["skipped"] = "context_limit_without_independent_truncation"
                    return finished("context_limit")
                options = dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                               pad_token_id=self.tokenizer.eos_token_id,
                               stopping_criteria=self.rollout.StoppingCriteriaList([
                                   self.rollout.StopTags(self.tokenizer, ids.shape[1])]))
                output = self.model.generate(ids, **options)
                generated_ids = output[0, ids.shape[1]:].detach().cpu().tolist()
                segment = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                event.update(generated_token_ids=generated_ids, decoded_segment_before_structural_cut=segment,
                             generation_finished_unix=time.time())
                ends = [segment.find(tag) + len(tag) for tag in ("</search>", "</answer>") if tag in segment]
                if ends:
                    segment = segment[:min(ends)]
                response += forced
                start = len(response)
                response += segment
                spans.append((start, len(response)))
                event.update(policy_span=[start, len(response)], policy_segment=segment)
                if "</answer>" in segment:
                    return finished("answer")
                match = re.search(r"<search>(.*?)</search>", segment, re.S)
                if not match or calls == 4:
                    return finished("invalid_or_budget")
                query = match.group(1).strip()
                event.update(query=query, retrieval_started_unix=time.time(), retriever_url=self.url)
                if self.check_health is not None:
                    event["admitted_retriever_health_before_request"] = self.check_health()
                reply = requests.post(self.url, json={"queries": [query], "topk": 3, "return_scores": True}, timeout=120)
                reply.raise_for_status()
                payload = reply.json()
                passages = payload["result"][0]
                if len(passages) != 3:
                    raise RuntimeError("Diagnostic retriever did not return exactly top3")
                documents, formatted = [], []
                for index, passage in enumerate(passages):
                    document = passage.get("document", passage)
                    contents = document.get("contents", document.get("text", ""))
                    found_ids = {key: document[key] for key in ("id", "doc_id", "document_id", "corpus_id", "index") if key in document}
                    documents.append(dict(rank=index + 1, identifiers=found_ids, identifier_missing=not found_ids,
                                          score=passage.get("score"), document=document, contents=contents))
                    formatted.append(f"Doc {index + 1}: {contents}")
                info = "\n".join(formatted)
                info_ids = self.tokenizer.encode(info, add_special_tokens=False)[:480]
                inserted = "\n<information>\n" + self.tokenizer.decode(info_ids) + "\n</information>\n"
                response += inserted
                calls += 1
                event.update(retrieval_finished_unix=time.time(), retrieval_payload=payload, top3_documents=documents,
                             formatted_information_before_truncation=info, inserted_information=inserted,
                             inserted_content_token_ids=info_ids, observation_content_budget=480)
        return finished("budget")
