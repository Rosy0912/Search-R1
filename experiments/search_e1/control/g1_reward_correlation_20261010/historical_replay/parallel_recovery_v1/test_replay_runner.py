import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

HERE = Path(__file__).parent

def module(name):
    s = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

core = module('replay_core'); runner = module('replay_gpu')

class Gates(unittest.TestCase):
    def config(self):
        return dict(gpu_index=5, gpu_uuid='exactGPU', initial_min_free_mib=32768,
                    running_min_free_mib=8192, max_replay_memory_mib=24576)
    def pf(self, processes=(), free=40000):
        return types.SimpleNamespace(resource_snapshot=lambda *a: dict(candidate_gpu=dict(uuid='exactGPU', actual_free_mib=free),
            compute_processes=list(processes)), same_process=lambda a,b: all(a[k] == b[k] for k in ['pid','start_ticks','sid','uid']))
    def own(self): return dict(pid=42, start_ticks=900, sid=42, uid=0, command_sha256='ownexact')
    def test_initial_shared_original_retriever(self):
        p = dict(pid=1, matched_recorded_pid_tick_sid_uid=True, recorded_command_sha256='same', command_sha256='same')
        v = runner.resource_gate(Path('/none'), {}, self.config(), self.pf([p]), self.own(), core, True)
        self.assertTrue(v['only_bound_original_run_and_exact_replay'])
    def test_exact_self_allowed_running(self):
        p = dict(self.own(), matched_recorded_pid_tick_sid_uid=False, memory_mib=18000)
        runner.resource_gate(Path('/none'), {}, self.config(), self.pf([p]), self.own(), core)
    def test_reused_self_PID_rejected(self):
        p = dict(self.own(), start_ticks=901, matched_recorded_pid_tick_sid_uid=False, memory_mib=1000)
        with self.assertRaises(core.Hold): runner.resource_gate(Path('/none'), {}, self.config(), self.pf([p]), self.own(), core)
    def test_foreign_same_UID_rejected(self):
        p = dict(self.own(), pid=43, matched_recorded_pid_tick_sid_uid=False, memory_mib=1000)
        with self.assertRaises(core.Hold): runner.resource_gate(Path('/none'), {}, self.config(), self.pf([p]), self.own(), core)
    def test_self_limit_hold(self):
        p = dict(self.own(), matched_recorded_pid_tick_sid_uid=False, memory_mib=24577)
        with self.assertRaises(core.Hold): runner.resource_gate(Path('/none'), {}, self.config(), self.pf([p]), self.own(), core)
    def test_directfree_initial_hold(self):
        with self.assertRaises(core.Hold): runner.resource_gate(Path('/none'), {}, self.config(), self.pf(free=32767), self.own(), core, True)
    def test_directfree_running_hold(self):
        with self.assertRaises(core.Hold): runner.resource_gate(Path('/none'), {}, self.config(), self.pf(free=8191), self.own(), core)
    def test_UUID_drift_hold(self):
        config = dict(self.config(), gpu_uuid='wrong')
        with self.assertRaises(core.Hold): runner.resource_gate(Path('/none'), {}, config, self.pf(), self.own(), core)
    def test_C_fields_all_original_ID_join(self):
        # Real C decoder mocked, but tests streaming canonical digest + full-mask
        # length, no invalid filtering, and independent decoded SHA join.
        row = dict(trajectory_id='id', input_ids=[3,4,5], attention_mask=[1,1,1],
                   info_mask=[1,1,0], position_ids=[0,1,2], prompt_width=1)
        actor = [dict(row, trajectory_id=str(i)) for i in range(2560)]
        c = dict(final_action_valid=False, model_authored_search_count=2,
            decoded_prompt_sha256=__import__('hashlib').sha256(b'prompt').hexdigest(),
            decoded_response_sha256=__import__('hashlib').sha256(b'response').hexdigest(),
            model_ranges_sha256=__import__('hashlib').sha256(core.canonical([[0,4]])).hexdigest())
        cs = [dict(c, trajectory_id=str(i)) for i in range(2560)]
        verified = []
        ch = types.SimpleNamespace(verify_streamed_actor=lambda r,i,g: None,
            actor_parts=lambda r: ([3],[4,5],[1,0]),
            verify_actor_array_digest=lambda digest,count,gate: verified.append((digest,count)))
        signals = types.SimpleNamespace(decode_segments=lambda *a: ('response', [[0,4]]),
            contexts=lambda *a: None, model_search_count=lambda *a: 2)
        tok = types.SimpleNamespace(decode=lambda ids:'prompt')
        items, lengths, digest = core.request_items(iter(actor), cs, {}, ch, tok, signals)
        self.assertEqual(len(items), 2560); self.assertEqual(lengths, [3]*2560)
        self.assertEqual(items[-1]['valid_rlen'], 2); self.assertEqual(verified[-1][1],2560)
        # No raw IDs are retained in the requests after each streamed row.
        self.assertNotIn('input_ids', items[0]); self.assertNotIn('info_mask', items[0])
    def test_C_ID_order_corruption_hold(self):
        ch = types.SimpleNamespace(verify_streamed_actor=lambda *a:None, actor_parts=lambda *a:None)
        with self.assertRaises(core.Hold): core.request_items(iter([{'trajectory_id':'a'}]), [{'trajectory_id':'b'}], {}, ch, None, None)
    def test_load_module_has_no_Torch(self):
        import sys
        self.assertNotIn('torch', sys.modules)

if __name__ == '__main__': unittest.main()
