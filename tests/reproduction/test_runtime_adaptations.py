"""CPU-only identity and immutable-asset regressions; no SSH/network/GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ADAPTATION = Path(os.environ.get("SEARCH_R1_ADAPTATION_DIR", (ROOT / "reproduction/h20" if (ROOT / "reproduction/h20").exists() else ROOT.parent)))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ADAPTATION / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset(payload, use_sha256=True):
    return {
        "path": "test/input.json", "size": len(payload),
        "kind": "dataset", "repo": "test/source", "revision": "immutable", "file": "input.json",
        "sha256": hashlib.sha256(payload).hexdigest() if use_sha256 else None,
        "git_blob_sha1": hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
    }


class IdentityTest(unittest.TestCase):
    def setUp(self):
        self.module = load("process_identity")
        self.record = {"pid": 42, "start_ticks": "1234", "uid": 1000,
                       "args": ["python", str(self.module.BASE / "retriever_local.py")]}

    def test_exact_identity_is_accepted(self):
        with patch.object(self.module, "identity", return_value=dict(self.record)):
            self.assertTrue(self.module.owned_retriever(self.record))

    def test_reused_pid_other_uid_or_other_command_is_rejected(self):
        for field, value in [("start_ticks", "1235"), ("uid", 1001), ("args", ["python", "other.py"])]:
            changed = dict(self.record, **{field: value})
            with self.subTest(field=field), patch.object(self.module, "identity", return_value=changed):
                self.assertFalse(self.module.owned_retriever(self.record))

    def test_same_nonretriever_identity_is_rejected(self):
        changed = dict(self.record, args=["python", "other.py"])
        with patch.object(self.module, "identity", return_value=changed):
            self.assertFalse(self.module.owned_retriever(changed))

    def test_missing_process_and_malformed_record_fail_closed(self):
        with patch.object(self.module, "identity", side_effect=FileNotFoundError):
            self.assertFalse(self.module.owned_retriever(self.record))
        for record in [None, {}, {"pid": "bad"}]:
            self.assertFalse(self.module.owned_retriever(record))

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux procfs required")
    def test_real_proc_starttime_and_uid_are_parsed_without_gpu(self):
        record = self.module.identity(os.getpid())
        self.assertEqual(record["pid"], os.getpid())
        self.assertEqual(record["uid"], os.getuid())
        self.assertGreater(int(record["start_ticks"]), 0)
        self.assertTrue(record["args"])


class AssetsTest(unittest.TestCase):
    def setUp(self):
        self.module = load("download_assets")
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.module.BASE = self.base
        self.payload = b'{"question":"test?"}\n'

    def tearDown(self):
        self.directory.cleanup()

    def test_sha256_and_git_blob_reject_same_length_mutation(self):
        path = self.base / "sample"
        for mode in [True, False]:
            with self.subTest(sha256=mode):
                a = asset(self.payload, mode)
                path.write_bytes(self.payload)
                self.assertTrue(self.module.valid(path, a))
                path.write_bytes(b"!" + self.payload[1:])
                self.assertFalse(self.module.valid(path, a))

    def test_git_blob_header_is_required(self):
        path = self.base / "sample"
        path.write_bytes(self.payload)
        a = asset(self.payload, False)
        a["git_blob_sha1"] = hashlib.sha1(self.payload).hexdigest()
        self.assertFalse(self.module.valid(path, a))

    def test_invalid_full_partial_is_preserved_before_fresh_download(self):
        a = asset(self.payload)
        dest = self.base / "assets" / a["path"]
        dest.parent.mkdir(parents=True)
        partial = dest.with_name(dest.name + ".partial")
        rejected_payload = b"!" + self.payload[1:]
        partial.write_bytes(rejected_payload)
        state = partial.with_name(partial.name + ".aria2")
        state.write_bytes(b"old resume state")

        def successful_fetch(command, **kwargs):
            output = Path(command[command.index("-o") + 1])
            self.assertFalse(output.exists(), "Must not resume a known-corrupt complete partial")
            output.write_bytes(self.payload)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(self.module.subprocess, "run", side_effect=successful_fetch):
            self.assertTrue(self.module.fetch(a))
        self.assertEqual(dest.read_bytes(), self.payload)
        preserved = list(dest.parent.glob(partial.name + ".rejected.*"))
        self.assertEqual(len(preserved), 2)
        self.assertEqual(sorted(p.read_bytes() for p in preserved),
                         sorted([rejected_payload, b"old resume state"]))

    def test_sparse_aria2_final_logical_size_preserves_incomplete_progress(self):
        a = asset(self.payload)
        a['size'] = 65 * 1024**2
        dest = self.base / "assets" / a['path']
        dest.parent.mkdir(parents=True)
        partial = dest.with_name(dest.name + '.partial')
        with partial.open('wb') as f:
            f.truncate(a['size'])
        state = partial.with_name(partial.name + '.aria2')
        state.write_bytes(b'in-progress bitmap fixture')
        with patch.object(self.module.subprocess, 'run', return_value=
                subprocess.CompletedProcess(['aria2c'], 1, '', 'interrupted')) as run:
            self.assertFalse(self.module.fetch(a))
        self.assertEqual(run.call_args.args[0][0], 'aria2c')
        self.assertEqual(partial.stat().st_size, a['size'])
        self.assertEqual(state.read_bytes(), b'in-progress bitmap fixture')
        self.assertFalse(list(dest.parent.glob(partial.name + '.rejected.*')))
        self.assertFalse(dest.exists())

    def test_network_failure_keeps_partial_and_does_not_promote(self):
        a = asset(self.payload)
        dest = self.base / "assets" / a["path"]
        partial = dest.with_name(dest.name + ".partial")

        def failed_fetch(command, **kwargs):
            partial.write_bytes(self.payload[:3])
            return subprocess.CompletedProcess(command, 28, "", "timeout")

        with patch.object(self.module.subprocess, "run", side_effect=failed_fetch):
            self.assertFalse(self.module.fetch(a))
        self.assertEqual(partial.read_bytes(), self.payload[:3])
        self.assertFalse(dest.exists())

    def test_all_manifest_entries_have_official_content_digest(self):
        entries = json.loads((ADAPTATION / "assets.json").read_text())
        self.assertEqual(len(entries), 30)
        for a in entries:
            with self.subTest(path=a["path"]):
                self.assertTrue(a.get("sha256") or a.get("git_blob_sha1"))
                self.assertEqual(len(a["revision"]), 40)


if __name__ == "__main__":
    unittest.main()
