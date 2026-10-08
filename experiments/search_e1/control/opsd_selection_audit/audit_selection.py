"""Independent CPU-only, disk-grouped audit of the frozen OPSD pair selection.

Original inputs are each read exactly once. Private SQLite holds source text;
published summary contains aggregate counts and identities, never QA bodies.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sqlite3
import sys
import time

SCIENCE_HASHES = {
    "search_e1/core.py": "0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc",
    "search_e1/opsd.py": "61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037",
    "search_e1/rollout.py": "a88506797ea7fe24e61e1ad53ba4487aef87bdd69ec395aef26c94973b3d629a",
    "upstream/Search-R1/search_r1/qa_protocol.py": "9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd",
}
RUNTIME_MANIFEST_SHA = "3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc"
KNOWN_INPUTS = {
    "rollouts": {"sha256": "fb4e94fb1efa600757a604db576807d9fd3dbd65a7c75bf36f11b0d2ba89d124", "rows": 845515, "bytes": 3866825184},
    "pairs": {"sha256": "a81a3ae61bb402f6b99dd35ae552c4bb76df8da391f3620c17fbd742ba2b1f40", "rows": 100974, "bytes": 829404707},
}
RAPIDFUZZ_VERSION = "3.13.0"
MAX_LINE_BYTES = 8 << 20
MAX_ADDRESS_SPACE = 32 << 30
TAG = re.compile(r"<(/?)(think|search|answer|information)>")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stat_id(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_science(runtime, manifest_path=None):
    """Import only hashed frozen core/QA; these imports do not import torch."""
    runtime = Path(runtime).resolve()
    identities = {}
    for rel, expected in SCIENCE_HASHES.items():
        path = runtime / rel
        before = stat_id(path)
        actual = sha_file(path)
        if actual != expected or stat_id(path) != before:
            raise ValueError("Frozen source identity mismatch: " + rel)
        identities[rel] = dict(before, sha256=actual)
    if manifest_path is not None:
        m = Path(manifest_path)
        if sha_file(m) != RUNTIME_MANIFEST_SHA:
            raise ValueError("Frozen 305-file runtime manifest mismatch")
        identities["runtime_manifest"] = dict(stat_id(m), sha256=RUNTIME_MANIFEST_SHA)
    sys.path.insert(0, str(runtime / "upstream/Search-R1"))
    sys.path.insert(0, str(runtime))
    core = importlib.import_module("search_e1.core")
    qa = importlib.import_module("search_r1.qa_protocol")
    for module, rel in ((core, "search_e1/core.py"), (qa, "upstream/Search-R1/search_r1/qa_protocol.py")):
        if Path(module.__file__).resolve() != runtime / rel:
            raise ValueError("Wrong previously imported science module: " + rel)
    import rapidfuzz
    from rapidfuzz.distance import Levenshtein
    if rapidfuzz.__version__ != RAPIDFUZZ_VERSION:
        raise ValueError("Require the original rapidfuzz version " + RAPIDFUZZ_VERSION)
    implementation = importlib.import_module(Levenshtein.normalized_distance.__module__)
    identities["rapidfuzz"] = dict(version=rapidfuzz.__version__, module_path=rapidfuzz.__file__,
                                   module_sha256=sha_file(rapidfuzz.__file__),
                                   implementation_module=implementation.__name__,
                                   implementation_path=implementation.__file__,
                                   implementation_sha256=sha_file(implementation.__file__))
    return core, qa, Levenshtein.normalized_distance, identities


def source_still_same(identities):
    for identity in identities.values():
        if "mtime_ns" in identity:
            if stat_id(identity["path"]) != {k: identity[k] for k in stat_id(identity["path"])}:
                return False
            if sha_file(identity["path"]) != identity["sha256"]:
                return False
    return True


def dataset_name(value):
    if value == "nq":
        return value
    if value in ("hotpot", "hotpotqa"):
        return "hotpotqa"
    raise ValueError("Unexpected dataset")


def policy_slices(row):
    text, previous, result = row["response"], 0, []
    for bounds in row["policy_spans"]:
        if not isinstance(bounds, list) or len(bounds) != 2 or any(type(x) is not int for x in bounds):
            raise ValueError("Invalid policy span schema")
        a, b = bounds
        if not previous <= a <= b <= len(text):
            raise ValueError("Invalid policy span bounds")
        result.append(text[a:b])
        previous = b
    return result


def policy_only_text(row, information_spans):
    """Concatenate authored character spans minus frozen external-info intervals.

    This is a character-distance analysis, not a scoring or token-mask variant.
    No separator is invented; span joining is recorded in the definition.
    """
    intervals = information_spans(row["response"])
    result = []
    for a, b in row["policy_spans"]:
        cursor = a
        for x, y in intervals:
            if y <= cursor or x >= b:
                continue
            if x > cursor:
                result.append(row["response"][cursor:min(x, b)])
            cursor = max(cursor, min(y, b))
            if cursor >= b:
                break
        if cursor < b:
            result.append(row["response"][cursor:b])
    return "".join(result)


def structure(text):
    """Literal reserved tag balance only; deliberately independent of reward."""
    stack, unexpected, nested = [], 0, 0
    for m in TAG.finditer(text):
        closing, kind = m.groups()
        if not closing:
            nested += bool(stack)
            stack.append(kind)
        elif stack and stack[-1] == kind:
            stack.pop()
        else:
            unexpected += 1
    return dict(balanced=not stack and not unexpected, unexpected_closing=unexpected,
                unclosed_opening=len(stack), nested_reserved_opening=nested)


def validate_row(row, core, qa):
    if not isinstance(row, dict):
        raise ValueError("Rollout is not an object")
    for field in ("question_id", "question", "response"):
        if not isinstance(row.get(field), str):
            raise ValueError("Invalid rollout string field " + field)
    if not row["question_id"] or len(row["question_id"]) > 512:
        raise ValueError("Invalid question ID length")
    if not isinstance(row.get("answers"), (str, list)) or (isinstance(row["answers"], list) and any(not isinstance(x, str) for x in row["answers"])):
        raise ValueError("Invalid answers schema")
    for field in ("sibling", "search_calls", "reward"):
        if type(row.get(field)) is not int:
            raise ValueError("Invalid rollout integer field " + field)
    if row["sibling"] not in range(5) or row["reward"] not in (0, 1) or row["search_calls"] < 0:
        raise ValueError("Invalid sibling/reward/search_calls")
    if "token_count" in row and (type(row["token_count"]) is not int or row["token_count"] < 0):
        raise ValueError("Invalid token_count")
    if not isinstance(row.get("policy_spans"), list):
        raise ValueError("Missing policy provenance")
    dataset_name(row.get("data_source"))
    spans = policy_slices(row)
    strict = core.reward(row["response"], row["answers"], row["policy_spans"])
    action = qa.extract_final_answer(spans)
    failure = None if strict else "final_invalid" if action is None else "valid_gold_mismatch"
    return dict(strict_reward=strict, reward_matches=strict == row["reward"], failure=failure,
                transcript_structure=structure(row["response"]),
                policy_structure=[structure(s) for s in spans])


def analyze_group(entries, core, distance):
    rows = [item[0] for item in entries]
    if len(rows) != 5 or sorted(r["sibling"] for r in rows) != list(range(5)):
        raise ValueError("Missing or duplicate five-sibling group")
    for field in ("question_id", "question", "answers", "data_source"):
        if any(r[field] != rows[0][field] for r in rows):
            raise ValueError("Sibling identity mismatch: " + field)
    nsuccess = sum(r["reward"] for r in rows)
    group = "all5_wrong" if not nsuccess else "correct_correct" if nsuccess == 5 else "correct_wrong"
    result = dict(question_id=rows[0]["question_id"], dataset=dataset_name(rows[0]["data_source"]), group=group,
                  successes=nsuccess, sibling_order=[r["sibling"] for r in rows],
                  provenance=[p for _, p, _ in entries],
                  sibling_diagnostics=[dict(sibling=r["sibling"], **metrics) for r, _, metrics in entries],
                  eligible=bool(nsuccess), reference=None, selection=None)
    native = core.mine_pairs(rows)
    if not nsuccess:
        if native:
            raise ValueError("Frozen mine emitted an excluded all5_wrong pair")
        return result, None
    correct = [i for i, r in enumerate(rows) if r["reward"] == 1]
    ref_key = lambda i: (rows[i]["search_calls"], rows[i].get("token_count", len(rows[i]["response"])))
    ref_index = min(correct, key=ref_key)
    ref_ties = [i for i in correct if ref_key(i) == ref_key(ref_index)]
    candidates = [i for i, r in enumerate(rows) if i != ref_index and r["reward"] == 0]
    mode = "failed_first"
    if not candidates:
        candidates = [i for i in range(len(rows)) if i != ref_index]
        mode = "other_success_fallback"
    policy_texts = [policy_only_text(r, core.information_spans) for r in rows]
    full = {i: distance(rows[ref_index]["response"], rows[i]["response"]) for i in candidates}
    policy = {i: distance(policy_texts[ref_index], policy_texts[i]) for i in candidates}
    full_index = max(candidates, key=full.__getitem__)
    policy_index = max(candidates, key=policy.__getitem__)
    selected, ref = rows[full_index], rows[ref_index]
    expected = dict(question_id=ref["question_id"], question=ref["question"], answers=ref["answers"],
                    reference=ref["response"], student=selected["response"], reference_reward=ref["reward"],
                    student_reward=selected["reward"], student_policy_spans=selected["policy_spans"])
    if native != [expected]:
        raise ValueError("Shadow selection differs from the hashed native core")
    full_ties = [i for i in candidates if full[i] == full[full_index]]
    policy_ties = [i for i in candidates if policy[i] == policy[policy_index]]
    result["reference"] = dict(sibling=ref["sibling"], search_calls=ref_key(ref_index)[0],
                               token_count_or_length=ref_key(ref_index)[1],
                               tied_siblings=[rows[i]["sibling"] for i in ref_ties])
    result["selection"] = dict(candidate_mode=mode, candidate_siblings=[rows[i]["sibling"] for i in candidates],
                               full_chosen=selected["sibling"], policy_chosen=rows[policy_index]["sibling"],
                               full_max_ties=[rows[i]["sibling"] for i in full_ties],
                               policy_max_ties=[rows[i]["sibling"] for i in policy_ties],
                               full_max=full[full_index], policy_max=policy[policy_index],
                               chosen_sibling_changed=full_index != policy_index,
                               chosen_response_changed=selected["response"] != rows[policy_index]["response"],
                               maximum_tie_set_changed=full_ties != policy_ties,
                               full_choice_not_policy_max=full_index not in policy_ties,
                               chosen_change_due_only_to_policy_tie=full_index != policy_index and full_index in policy_ties,
                               maxima_sets_disjoint=not set(full_ties).intersection(policy_ties),
                               distances=[dict(sibling=rows[i]["sibling"], full=full[i], policy=policy[i],
                                               response_sha256=hashlib.sha256(rows[i]["response"].encode()).hexdigest(),
                                               policy_text_sha256=hashlib.sha256(policy_texts[i].encode()).hexdigest())
                                          for i in candidates])
    return result, expected


class Audit:
    def __init__(self, output, core, qa, distance, science, expected, expected_questions,
                 max_line=MAX_LINE_BYTES, min_free=32 << 30):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        if shutil.disk_usage(self.output).free < min_free:
            raise ValueError("Insufficient private audit scratch space")
        self.core, self.qa, self.distance, self.science = core, qa, distance, science
        self.expected, self.expected_questions, self.max_line = expected, expected_questions, max_line
        self.errors, self.counters, self.cells = Counter(), Counter(), defaultdict(Counter)
        for dataset in ("nq","hotpotqa"):
            for group in ("correct_correct","correct_wrong","all5_wrong"):
                self.cells[dataset+"/"+group]["questions"] = 0
        self.inputs, self.started, self.last_progress = {}, time.time(), 0
        self.issue_stream = (self.output / "issues.private.jsonl").open("x", encoding="utf-8")
        self.case_stream = (self.output / "cases.private.jsonl").open("x", encoding="utf-8")
        self.pair_stream = (self.output / "pairs.private.jsonl").open("x", encoding="utf-8")
        self.db = sqlite3.connect(str(self.output / "source.private.sqlite"))
        self.db.executescript("""
        PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL;
        PRAGMA cache_size=-65536; PRAGMA temp_store=FILE; PRAGMA mmap_size=0;
        CREATE TABLE questions(qid TEXT PRIMARY KEY, first_line INTEGER, runs INTEGER,
             valid INTEGER DEFAULT 0, eligible INTEGER DEFAULT 0, dataset TEXT, cell TEXT,
             expected_pair TEXT, pair_seen INTEGER DEFAULT 0);
        CREATE INDEX first_seen ON questions(first_line);
        CREATE TABLE rows(qid TEXT, sibling INTEGER, source_line INTEGER, payload TEXT,
             provenance TEXT, metrics TEXT, PRIMARY KEY(qid,sibling));
        CREATE INDEX source_order ON rows(qid,source_line);
        """)

    def issue(self, name, provenance=None, qid=None):
        self.errors[name] += 1
        self.issue_stream.write(canonical(dict(kind=name, provenance=provenance, question_id=qid)) + "\n")

    def progress(self, phase, force=False):
        now = time.time()
        if force or now - self.last_progress >= 60:
            self.last_progress = now
            atomic_json(self.output / "progress.json", dict(phase=phase, pid=os.getpid(),
                        timestamp_unix=now, elapsed_sec=now-self.started, counters=dict(self.counters),
                        errors=dict(self.errors), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    def scan(self, name, visitor):
        """One physical pass also hashes bad/oversize/partial lines exactly once."""
        expected = self.expected[name]
        path = Path(expected["path"])
        before = stat_id(path)
        for k in ("path", "device", "inode", "bytes", "mtime_ns", "ctime_ns"):
            if k in expected and before[k] != expected[k]:
                raise ValueError("Pre-scan input identity mismatch: " + name + "/" + k)
        digest, rows, offset = hashlib.sha256(), 0, 0
        with path.open("rb") as f:
            opened = os.fstat(f.fileno())
            if (opened.st_dev, opened.st_ino) != (before["device"], before["inode"]):
                raise ValueError("Input changed while opening: " + name)
            while True:
                line = f.readline(self.max_line + 1)
                if not line:
                    break
                rows += 1
                row_hash, size = hashlib.sha256(), 0
                chunks, oversized = [], False
                while True:
                    digest.update(line); row_hash.update(line); size += len(line)
                    if size <= self.max_line:
                        chunks.append(line)
                    else:
                        oversized = True
                        chunks.clear()
                    if line.endswith(b"\n"):
                        complete = True
                        break
                    line = f.readline(self.max_line + 1)
                    if not line:
                        complete = False
                        break
                p = dict(line=rows, byte_offset=offset, bytes=size, row_sha256=row_hash.hexdigest())
                offset += size
                self.counters[name + "_physical_rows"] = rows
                if oversized:
                    self.issue(name + "_oversize_row", p)
                elif not complete:
                    self.issue(name + "_partial_final_row", p)
                else:
                    try:
                        row = json.loads(b"".join(chunks))
                    except (ValueError, UnicodeDecodeError):
                        self.issue(name + "_bad_json_row", p)
                    else:
                        visitor(row, p)
                self.progress("SCAN_" + name.upper())
        after = stat_id(path)
        sha = digest.hexdigest()
        self.inputs[name] = dict(before=before, after=after, rows=rows, sha256=sha,
                                 stat_stable=before == after, known_sha_matches=sha == expected["sha256"],
                                 known_rows_match=rows == expected["rows"], byte_count=offset)
        if before != after:
            self.issue(name + "_stat_changed")
        if sha != expected["sha256"]:
            self.issue(name + "_known_hash_mismatch")
        if rows != expected["rows"]:
            self.issue(name + "_known_row_count_mismatch")
        if offset != before["bytes"]:
            self.issue(name + "_byte_count_mismatch")
        self.db.commit()

    def ingest_rollout(self, row, p):
        try:
            metrics = validate_row(row, self.core, self.qa)
        except (ValueError, TypeError, KeyError):
            self.issue("rollout_bad_schema_or_mask", p)
            return
        if not metrics["reward_matches"]:
            self.issue("rollout_strict_reward_mismatch", p, row["question_id"])
        qid = row["question_id"]
        new_run = getattr(self, "last_qid", None) != qid
        self.last_qid = qid
        self.db.execute("INSERT OR IGNORE INTO questions(qid,first_line,runs) VALUES(?,?,0)", (qid,p["line"]))
        if new_run:
            self.db.execute("UPDATE questions SET runs=runs+1 WHERE qid=?", (qid,))
        try:
            self.db.execute("INSERT INTO rows VALUES(?,?,?,?,?,?)", (qid,row["sibling"],p["line"],canonical(row),canonical(p),canonical(metrics)))
        except sqlite3.IntegrityError:
            self.issue("rollout_duplicate_sibling", p, qid)
        if p["line"] % 5000 == 0:
            self.db.commit()

    def grouped_audit(self):
        total = self.db.execute("SELECT count(*) FROM questions").fetchone()[0]
        self.counters["unique_rollout_questions"] = total
        if total != self.expected_questions:
            self.issue("known_question_count_mismatch")
        for qid, _, runs in self.db.execute("SELECT qid,first_line,runs FROM questions ORDER BY first_line"):
            entries = [(json.loads(r),json.loads(p),json.loads(m)) for r,p,m in self.db.execute(
                "SELECT payload,provenance,metrics FROM rows WHERE qid=? ORDER BY source_line", (qid,))]
            if runs > 1:
                self.counters["noncontiguous_reappearing_questions"] += 1
                self.counters["additional_question_runs"] += runs-1
            try:
                result, expected = analyze_group(entries, self.core, self.distance)
            except (ValueError, TypeError, KeyError):
                self.issue("invalid_question_group_or_native_selection", qid=qid)
                continue
            result["source_group_runs"] = runs
            result["native_shadow_exact"] = True
            cell = self.cells[result["dataset"] + "/" + result["group"]]
            cell["questions"] += 1
            cell["eligible_pairs"] += int(result["eligible"])
            for diag in result["sibling_diagnostics"]:
                cell["rollout_rows"] += 1
                if diag["failure"]:
                    cell["failure_" + diag["failure"]] += 1
                cell["transcript_literal_tag_balanced"] += int(diag["transcript_structure"]["balanced"])
                cell["transcript_nested_reserved"] += int(bool(diag["transcript_structure"]["nested_reserved_opening"]))
                cell["nonfinal_policy_spans"] += max(0,len(diag["policy_structure"])-1)
                cell["nonfinal_policy_literal_tag_unbalanced"] += sum(not x["balanced"] for x in diag["policy_structure"][:-1])
                cell["strict_success_with_unbalanced_transcript"] += int(diag["strict_reward"] == 1 and not diag["transcript_structure"]["balanced"])
            if result["eligible"]:
                sel = result["selection"]
                for key in ("chosen_sibling_changed", "chosen_response_changed", "maximum_tie_set_changed",
                            "full_choice_not_policy_max","chosen_change_due_only_to_policy_tie","maxima_sets_disjoint"):
                    cell[key] += int(sel[key])
                cell["reference_tie_questions"] += int(len(result["reference"]["tied_siblings"])>1)
                cell["full_student_tie_questions"] += int(len(sel["full_max_ties"])>1)
                cell["policy_student_tie_questions"] += int(len(sel["policy_max_ties"])>1)
                cell["student_" + sel["candidate_mode"]] += 1
            self.db.execute("UPDATE questions SET valid=1,eligible=?,dataset=?,cell=?,expected_pair=? WHERE qid=?",
                (int(result["eligible"]),result["dataset"],result["group"],canonical(expected) if expected else None,qid))
            self.case_stream.write(canonical(result) + "\n")
            self.counters["audited_questions"] += 1
            self.counters["native_expected_pairs"] += int(result["eligible"])
            if self.counters["audited_questions"] % 1000 == 0:
                self.db.commit()
            self.progress("GROUP_SELECTION_AUDIT")
        self.db.commit()

    def visit_pair(self, row, p):
        if (not isinstance(row, dict) or any(not isinstance(row.get(k),str) for k in ("question_id","question","reference","student"))
                or not row["question_id"] or len(row["question_id"])>512
                or any(type(row.get(k)) is not int or row[k] not in (0,1) for k in ("reference_reward","student_reward"))
                or not isinstance(row.get("student_policy_spans"),list)
                or not isinstance(row.get("answers"),(str,list))):
            self.issue("pair_bad_schema", p)
            return
        qid = row["question_id"]
        question = self.db.execute("SELECT valid,eligible,expected_pair,pair_seen,dataset,cell FROM questions WHERE qid=?", (qid,)).fetchone()
        if question is None:
            self.issue("pair_unknown_question", p, qid)
            return
        valid, eligible, expected_text, seen, dataset, group = question
        self.db.execute("UPDATE questions SET pair_seen=pair_seen+1 WHERE qid=?", (qid,))
        if seen:
            self.issue("pair_duplicate_question", p, qid)
        if not valid or not eligible:
            self.issue("pair_for_invalid_or_excluded_question", p, qid)
            return
        cell = self.cells[dataset + "/" + group]
        cell["pair_rows"] += 1
        expected = json.loads(expected_text)
        mismatch = [key for key in expected if row.get(key) != expected[key]]
        # Unexpected keys are recorded; missing keys are mismatch even if value None.
        mismatch += [key for key in expected if key not in row]
        reference_matches, student_matches = [], []
        for sibling,payload,provenance in self.db.execute("SELECT sibling,payload,provenance FROM rows WHERE qid=? ORDER BY source_line",(qid,)):
            original=json.loads(payload)
            identity=dict(sibling=sibling,provenance=json.loads(provenance))
            if original["response"]==row["reference"] and original["reward"]==row["reference_reward"]:
                reference_matches.append(identity)
            if (original["response"]==row["student"] and original["reward"]==row["student_reward"]
                    and original["policy_spans"]==row["student_policy_spans"]):
                student_matches.append(identity)
        self.pair_stream.write(canonical(dict(question_id=qid,provenance=p,previous_pair_rows=seen,
                                  native_pair_exact=not mismatch, mismatch_fields=sorted(set(mismatch)),
                                  reference_matching_source_rows=reference_matches,student_matching_source_rows=student_matches,
                                  expected_semantic_sha256=hashlib.sha256(canonical(expected).encode()).hexdigest(),
                                  actual_semantic_sha256=hashlib.sha256(canonical(row).encode()).hexdigest())) + "\n")
        if mismatch:
            self.issue("pair_selection_or_identity_mismatch", p, qid)
            cell["pair_mismatch"] += 1
            with (self.output / "pair-mismatch.private.jsonl").open("a", encoding="utf-8") as f:
                f.write(canonical(dict(question_id=qid,provenance=p,mismatch_fields=sorted(set(mismatch)),
                                       expected_semantic_sha256=hashlib.sha256(canonical(expected).encode()).hexdigest(),
                                       actual_semantic_sha256=hashlib.sha256(canonical(row).encode()).hexdigest())) + "\n")
        else:
            cell["pair_native_exact"] += 1
            self.counters["pair_native_exact"] += 1
        if set(row) != set(expected):
            self.issue("pair_unexpected_or_missing_field", p, qid)
        if p["line"] % 5000 == 0:
            self.db.commit()

    def run(self):
        fatal = None
        try:
            self.progress("START", True)
            self.scan("rollouts", self.ingest_rollout)
            self.grouped_audit()
            self.scan("pairs", self.visit_pair)
            for qid, seen in self.db.execute("SELECT qid,pair_seen FROM questions WHERE valid=1 AND eligible=1 AND pair_seen<>1"):
                if not seen:
                    self.issue("missing_eligible_pair", qid=qid)
            if self.counters["native_expected_pairs"] != self.expected["pairs"]["rows"]:
                self.issue("native_pair_count_mismatch")
            if self.counters["pair_native_exact"] != self.expected["pairs"]["rows"]:
                self.issue("native_matched_pair_count_mismatch")
            if not source_still_same(self.science):
                self.issue("frozen_science_changed_during_audit")
            for name, observed in self.inputs.items():
                if stat_id(self.expected[name]["path"]) != observed["before"]:
                    self.issue(name + "_changed_after_its_scan")
        except Exception as exc:
            fatal = dict(type=type(exc).__name__, message=str(exc))
            self.issue("fatal_audit_exception")
        finally:
            self.db.commit()
            self.db.close()
            for stream in (self.issue_stream, self.case_stream,self.pair_stream):
                stream.flush(); os.fsync(stream.fileno()); stream.close()
        accepted = not self.errors
        required_checks = dict(
            complete_original_rollout_rows=self.inputs.get("rollouts",{}).get("rows") == self.expected["rollouts"]["rows"],
            complete_original_pair_rows=self.inputs.get("pairs",{}).get("rows") == self.expected["pairs"]["rows"],
            complete_unique_questions=self.counters["unique_rollout_questions"] == self.expected_questions,
            all_native_pairs_match=self.counters["pair_native_exact"] == self.expected["pairs"]["rows"],
            bad_json_or_partial_or_oversize_rows=sum(v for k,v in self.errors.items() if any(x in k for x in ("bad_json","partial_final","oversize"))),
            bad_schema_or_mask_rows=self.errors["rollout_bad_schema_or_mask"]+self.errors["pair_bad_schema"],
            duplicate_siblings=self.errors["rollout_duplicate_sibling"],duplicate_pairs=self.errors["pair_duplicate_question"],
            missing_eligible_pairs=self.errors["missing_eligible_pair"],selection_or_identity_mismatches=self.errors["pair_selection_or_identity_mismatch"],
            invalid_five_sibling_groups=self.errors["invalid_question_group_or_native_selection"],
            strict_reward_mismatches=self.errors["rollout_strict_reward_mismatch"],
            known_input_sha_matches=all(self.inputs.get(n,{}).get("known_sha_matches",False) for n in self.expected),
            input_metadata_stable=all(self.inputs.get(n,{}).get("stat_stable",False) for n in self.expected),
        )
        cells = []
        for key, counter in sorted(self.cells.items()):
            dataset, group = key.split("/")
            d = dict(dataset=dataset,group=group,**dict(counter))
            denom = d.get("eligible_pairs",0)
            d["selection_changed_rate_among_eligible"] = d.get("chosen_sibling_changed",0)/denom if denom else None
            cells.append(d)
        summary = dict(schema="opsd-selection-audit-v1", accepted=accepted,
            status="ACCEPTED" if accepted else "FAILED_AUDIT", pid=os.getpid(), started_unix=self.started,
            finished_unix=time.time(), elapsed_sec=time.time()-self.started, inputs=self.inputs,
            frozen_science=self.science, counters=dict(self.counters), errors=dict(self.errors), cells=cells,
            required_checks=required_checks,
            fatal=fatal, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope="CPU character selection/protocol/structure audit only; no CUDA/HTTP, tokenizer loss scan, new model inference or training causal claim.",
            policy_distance_definition="Concatenate original authored policy_spans after subtracting frozen core.information_spans; no invented delimiter; same candidate set and reference; rapidfuzz normalized Levenshtein default arguments.",
            tie_rule="Native Python min/max stable first-in-original-file-order; exact float equality defines ties, no tolerance.",
            change_rate_definition="Count questions whose selected original sibling changes divided by eligible questions. Distance differences are not an observation contribution percentage.",
            ambiguity_scope="Pairs do not contain sibling IDs. Multiple matching source rows are preserved; native first-order predicted IDs and distance tie bounds do not prove unique stored-pair origin.",
            structure_scope="Literal reserved tag stack audit only, independent of strict final-action reward; nonfinal span defects remain separate.",
            private_artifacts=[str(self.output/n) for n in ("source.private.sqlite","cases.private.jsonl","pairs.private.jsonl","issues.private.jsonl")],
            algorithm=dict(original_input_passes={"rollouts":1,"pairs":1},grouping="SQLite by question_id; original source_line order; noncontiguous runs detected, never assumed continuous",cpu_processes=1,sqlite_cache_mib=64,max_line_bytes=self.max_line,linux_address_space_limit_bytes=MAX_ADDRESS_SPACE))
        atomic_json(self.output / "summary.json", summary)
        self.progress(summary["status"], True)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",required=True)
    p.add_argument("--runtime-manifest",required=True)
    p.add_argument("--cases-manifest",required=True)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    if sys.platform != "linux":
        raise ValueError("Production admission requires Linux RLIMIT_AS; local fixtures call Audit directly")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(MAX_ADDRESS_SPACE, hard) if hard != resource.RLIM_INFINITY else MAX_ADDRESS_SPACE
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    os.umask(0o077)
    core, qa, distance, science = load_science(args.runtime,args.runtime_manifest)
    science["audit_source"] = dict(stat_id(__file__),sha256=sha_file(__file__))
    mp = Path(args.cases_manifest)
    science["cases_manifest"] = dict(stat_id(mp),sha256=sha_file(mp))
    manifest = json.loads(mp.read_text())
    expected = {name:manifest["sources"][name] for name in KNOWN_INPUTS}
    for name, identity in expected.items():
        if any(identity[k] != v for k,v in KNOWN_INPUTS[name].items()):
            raise ValueError("Manifest does not bind the approved complete original data: " + name)
    summary = Audit(args.output,core,qa,distance,science,expected,169103).run()
    print(json.dumps(dict(accepted=summary["accepted"],status=summary["status"],summary_path=str(Path(args.output).resolve()/"summary.json"),
                         counters=summary["counters"],errors=summary["errors"]),sort_keys=True))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
