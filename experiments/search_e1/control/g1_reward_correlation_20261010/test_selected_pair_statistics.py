"""Synthetic-only CPU fixtures; no historical questions/rows, model or HTTP."""
import copy
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import selected_pair_statistics as bridge
import pairwise_pearson as pearson
import reconstruct_train_raw_c_preflight_v2 as c_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bridge.canonical(value)); return sha(path)


def jsonl(path, rows):
    Path(path).write_bytes(b''.join(bridge.canonical(x) + b'\n' for x in rows))
    return sha(path)


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


SYNTHETIC_REWARD_SOURCE = '''class Synthetic:
    def _score_probe(self, row, pr):
        if row['answer'] is None:
            return -self.PENALTY, 0., None
        b, d = float(pr['boundary']), float(pr['flip'])
        if not row['correct']:
            return self.wrong_floor, 0., None
        if row['n_search'] == 0:
            return 1. + self.bal_nosearch_bonus * b, d, 'true'
        cost = min(1., max(0., row['n_search']-1)/max(1., self.MAX_SEARCH-1))
        return 1. + self.bal_ground*(1.-b)*d-self.w_cost*cost, d, 'true'
'''


class SyntheticFixture:
    """All toy 512x5 identities and finite invented probabilities, temporary only."""
    def __init__(self, root, steps):
        self.root = Path(root).resolve(); self.steps = steps; self.run_id = 'a' * 32
        self.native = self.root/'native'; self.D = self.root/'diagnostic'; self.C = self.D/'C'
        self.native.mkdir(); self.D.mkdir(); self.C.mkdir()
        source = self.native/'repo/verl/trainer/main_ppo_eco.py'
        source.parent.mkdir(parents=True); source.write_text(SYNTHETIC_REWARD_SOURCE)
        self.run = dict(run_id=self.run_id, source_sha256={'verl/trainer/main_ppo_eco.py':sha(source)})
        runsha = put(self.native/'logs/run-manifest.json', self.run)
        self.C_inputs = dict(run_id=self.run_id, helper_sha256='c'*64, sources=self.run['source_sha256'])
        self.config = dict(native_run_id=self.run_id, original_base=str(self.native), diagnostic_base=str(self.D),
            selected_steps=steps, run_manifest_sha256=runsha, C_directory=str(self.C),
            C_inputs_sha256=put(self.C/'inputs.json', self.C_inputs), C_helper_sha256='c'*64, R_sources=[])
        self.R_dirs = {}; self.C_rows = {}; self.R_rows = {}; self.updates = []
        # The original actual ledger has ALL earlier steps; statistics select only requested steps.
        for step in range(1, max(steps)+1):
            attempt = f'{step:032x}'; version = 'probe-round-'+str(((step-1)//50)*50)
            self.updates.append(dict(step=step, run_id=self.run_id, attempt_id=attempt, attempt_index=0,
                probe_version=version, behavior_trace_sha256=h('trace'+str(step)), metrics={'critic/rewards/mean':0.}))
        # Even one producer per new step must work; retained step1 can have a dense config.
        for step in steps:
            producer = self.D/('old-stopped-dense' if step == 1 else 'worker-'+str(step)); producer.mkdir()
            code = self.D/('source-'+str(step)); code.mkdir(); (code/'worker.py').write_text('# synthetic producer\n')
            manifestsha = put(code/'source-manifest.json', dict(source_sha256={'worker.py':sha(code/'worker.py')}))
            rcfg = dict(output=str(producer), expected_run_id=self.run_id, base=str(self.native), C_base=str(self.C),
                run_manifest_sha256=runsha, C_helper_sha256='c'*64, reward_constants=bridge.CONSTANTS,
                diagnostic_run_id=f'{step+1000:032x}')
            if step == 1:
                rcfg.update(start_step=1,end_step=146)
            else:
                rcfg['selected_steps']=[step]
            cfgsha = put(code/'config.json', rcfg)
            inputsha = put(producer/'inputs.json', dict(original_run_id=self.run_id, config_sha256=cfgsha,
                source_manifest_sha256=manifestsha, raw_online_R_saved=False))
            self.config['R_sources'].append(dict(selected_steps=[step], config_path=str(code/'config.json'),
                config_sha256=cfgsha, source_directory=str(code), source_manifest_path=str(code/'source-manifest.json'),
                source_manifest_sha256=manifestsha, inputs_sha256=inputsha))
            self.R_dirs[step] = producer
            update = self.updates[step-1]; attempt=update['attempt_id']; snapshot=f'{self.run_id}:{step}:policy-{attempt}'
            rows=[]; ids=[]; upstream={}
            for n in range(2560):
                group=h('parent'+str(n//5)); identity=f'{self.run_id}-{step}-attempt-{attempt}-{n//5}-{group}-sample-{n%5}'
                C=int((n+step)%7<3); valid=bool(C or n%17)
                rows.append(dict(trajectory_id=identity, C_raw=C, final_action_valid=valid,
                    model_authored_search_count=n%4, original_online_reward=False, reconstructed_strict_C=True,
                    run_id=self.run_id, step=step, attempt_id=attempt, policy_snapshot_id=snapshot,
                    upstream_group_sha256=group, probe_version_for_future_R_replay=update['probe_version']))
                ids.append(identity); upstream[identity]=group
            gate=dict(accepted=True, trajectory_count=2560, run_id=self.run_id, logical_update=step,
                source_sha256=self.run['source_sha256'], policy_snapshot_id=snapshot, attempt_id=attempt, attempt_index=0,
                private_trace_sha256=update['behavior_trace_sha256'], trace_sha256=update['behavior_trace_sha256'],
                trajectory_ids=ids, upstream_group_sha256=upstream, row_tensor_sha256={x:h(x) for x in ids},
                actor_token_counts=[2]*2560, actor_rows_sha256=h('synthetic actor rows'+str(step)))
            update['behavior_probability_gate_sha256']=put(self.native/'logs'/f'behavior-probability-attempt-{attempt}.json',gate)
            self.C_rows[step]=rows
            Cpath=self.C/f'TRAIN-{step}-raw-C.jsonl'; Csha=jsonl(Cpath,rows)
            Csummary=dict(step=step,attempt_id=attempt,trajectory_count=2560,native_EM_check_pass=True,
                probe_version_for_future_R_replay=update['probe_version'],complete_trace_fresh_sha256=update['behavior_trace_sha256'],
                C_hits_exact=sum(r['C_raw'] for r in rows),raw_C_remote_path=str(Cpath),raw_C_fresh_sha256=Csha)
            Csummarysha=put(self.C/f'TRAIN-{step}-summary.json',Csummary)
            Rrows=[]; score=bridge.load_native_score(source,sha(source),bridge.CONSTANTS)
            order=list(reversed(range(2560)))
            for position,n in enumerate(order):
                C=rows[n]; b=(n%13)/12.; d=(n%11)/10.; pyR=score(C,b,d); R=struct.unpack('<f',struct.pack('<f',pyR))[0]
                Rrows.append(dict(trajectory_id=C['trajectory_id'],run_id=self.run_id,step=step,attempt_id=attempt,
                    original_sample_index=n,balanced_request_index=position,upstream_group_sha256=C['upstream_group_sha256'],
                    C_raw=C['C_raw'],final_action_valid=C['final_action_valid'],model_authored_search_count=C['model_authored_search_count'],
                    b_reconstructed=b,d_reconstructed=d,probe_version=update['probe_version'],
                    C_record_sha256=hashlib.sha256(bridge.canonical(C)).hexdigest(),original_online_R_available=False,
                    R_reconstructed=R,R_python_before_FP32=pyR))
            self.R_rows[step]=Rrows; Rpath=producer/f'TRAIN-{step}-pairs-R-reconstructed.jsonl'; Rsha=jsonl(Rpath,Rrows)
            moments=pearson.ExactBinaryMoments()
            for row in Rrows: moments.add(row['C_raw'],row['R_reconstructed'])
            report=moments.report(); update['metrics']['critic/rewards/mean']=report['R_mean']
            values=[r['R_reconstructed'] for r in Rrows]
            comparison=dict(passed=True,precise_native_update_R_mean=report['R_mean'],precise_mean_tolerance=2e-6,
                precise_mean_absolute_difference=0.,native_scalar_agreement_does_not_prove_online_per_row_bitwise_identity=True,
                native_printed={'critic/score/mean':round(report['R_mean'],3),'critic/score/min':round(min(values),3),
                    'critic/score/max':round(max(values),3),'train/perf/em':round(report['C_mean'],3)})
            summary=dict(status='ACCEPTED_RECONSTRUCTED_R_STEP',step=step,N=2560,trajectory_count=2560,actual_probe_rows=2560,
                commit_micro4_batches=640,boundary_micro4_batches=640,probe_version=update['probe_version'],
                original_trace_sha256=update['behavior_trace_sha256'],actor_rows_sha256=gate['actor_rows_sha256'],
                original_online_row_rewards_available=False,independent_C_summary_sha256=Csummarysha,
                independent_C_rows_sha256=Csha,output_path=str(Rpath),output_sha256=Rsha,
                post_balance_permutation_sha256=hashlib.sha256(bridge.canonical(order)).hexdigest(),native_scalar_comparison=comparison,
                sum_C=report['sumC'],sum_R=report['sumR'],sum_R2=report['sumR2'],sum_CR=report['sumCR'])
            put(producer/f'TRAIN-{step}-summary.json',summary)
        jsonl(self.native/'logs/actual-updates.jsonl',self.updates)

    def analyze(self):
        return bridge.analyze_scope(self.config,pearson,c_gate)

    def mutate_R(self, fn, step=20):
        fn(self.R_rows[step]); Rpath=self.R_dirs[step]/f'TRAIN-{step}-pairs-R-reconstructed.jsonl'
        newsha=jsonl(Rpath,self.R_rows[step]); path=self.R_dirs[step]/f'TRAIN-{step}-summary.json'
        summary=json.loads(path.read_bytes()); summary['output_sha256']=newsha; put(path,summary)

    def mutate_R_summary(self, fn, step=20):
        path=self.R_dirs[step]/f'TRAIN-{step}-summary.json'; summary=json.loads(path.read_bytes()); fn(summary); put(path,summary)


class CompleteFixtures(unittest.TestCase):
    def test_complete_sparse8_20480_not_dense_and_no_global_completion(self):
        with tempfile.TemporaryDirectory() as root:
            f=SyntheticFixture(root,bridge.SELECTED_STEPS); result=f.analyze()
            self.assertEqual(result['total_pairs'],20480); self.assertEqual(result['completed_steps'],8)
            self.assertEqual([r['step'] for r in result['steps']],bridge.SELECTED_STEPS)
            self.assertTrue(result['not_all_TRAIN_steps_1_to_150']); self.assertFalse(result['original_online_R_available'])
            self.assertTrue(all(x['producer_whole_scope_completion_not_required'] for x in result['source_proofs']))
            self.assertFalse(any(p.name == 'accepted-summary.json' for p in f.D.rglob('*')))
            direct=pearson.ExactBinaryMoments()
            for step in bridge.SELECTED_STEPS:
                for row in f.R_rows[step]: direct.add(row['C_raw'],row['R_reconstructed'])
            self.assertEqual(result['pooled'],direct.report())
            import numpy as np
            C=[r['C_raw'] for s in bridge.SELECTED_STEPS for r in f.R_rows[s]]
            R=[r['R_reconstructed'] for s in bridge.SELECTED_STEPS for r in f.R_rows[s]]
            self.assertAlmostEqual(result['pooled']['Pearson']['value'],float(np.corrcoef(C,R)[0,1]),places=13)

    def test_selected_single_step_source1_reusable_despite_old_dense_scope(self):
        with tempfile.TemporaryDirectory() as root:
            f=SyntheticFixture(root,[1]); result=f.analyze()
            self.assertEqual(result['total_pairs'],2560)
            self.assertFalse((f.R_dirs[1]/'accepted-summary.json').exists())
            self.assertFalse(result['historical_original_online_R_claimed'])


class RejectionFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f=SyntheticFixture(self.temp.name,[1,20])

    def hold(self):
        with self.assertRaises((bridge.Hold,pearson.Hold,c_gate.Hold)): self.f.analyze()

    def test_missing_selected_source(self): self.f.config['R_sources'].pop(); self.hold()
    def test_duplicate_selected_source(self): self.f.config['R_sources'].append(copy.deepcopy(self.f.config['R_sources'][0])); self.hold()
    def test_extra_selected_source(self): self.f.config['R_sources'][0]['selected_steps']=[1,2]; self.hold()
    def test_unsorted_selected_scope(self): self.f.config['selected_steps']=[20,1]; self.hold()
    def test_duplicate_scope(self): self.f.config['selected_steps']=[1,20,20]; self.hold()
    def test_native_run_manifest_SHA(self): self.f.config['run_manifest_sha256']='0'*64; self.hold()
    def test_source_config_SHA(self): self.f.config['R_sources'][1]['config_sha256']='0'*64; self.hold()
    def test_source_manifest_SHA(self): self.f.config['R_sources'][1]['source_manifest_sha256']='0'*64; self.hold()
    def test_source_file_hotchange(self): (self.f.D/'source-20/worker.py').write_text('# changed'); self.hold()
    def test_source_inputs_SHA(self): self.f.config['R_sources'][1]['inputs_sha256']='0'*64; self.hold()
    def test_C_inputs_SHA(self): self.f.config['C_inputs_sha256']='0'*64; self.hold()
    def test_partial_native_ledger(self):
        path=self.f.native/'logs/actual-updates.jsonl'; path.write_bytes(path.read_bytes()[:-1]); self.hold()
    def test_native_ledger_duplicate(self):
        self.f.updates.append(self.f.updates[-1]); jsonl(self.f.native/'logs/actual-updates.jsonl',self.f.updates); self.hold()
    def test_partial_R_record(self):
        path=self.f.R_dirs[20]/'TRAIN-20-pairs-R-reconstructed.jsonl'; path.write_bytes(path.read_bytes()[:-1])
        self.f.mutate_R_summary(lambda x:x.update(output_sha256=sha(path))); self.hold()
    def test_missing_R_row(self): self.f.mutate_R(lambda rows:rows.pop()); self.hold()
    def test_duplicate_R_row(self):
        self.f.mutate_R(lambda rows:rows.__setitem__(-1,dict(rows[-2],balanced_request_index=2559))); self.hold()
    def test_unknown_R_trajectory(self): self.f.mutate_R(lambda rows:rows[0].update(trajectory_id='unknown')); self.hold()
    def test_original_row_index(self): self.f.mutate_R(lambda rows:rows[0].update(original_sample_index=0)); self.hold()
    def test_balance_position(self): self.f.mutate_R(lambda rows:rows[0].update(balanced_request_index=4)); self.hold()
    def test_C_record_SHA(self): self.f.mutate_R(lambda rows:rows[0].update(C_record_sha256='0'*64)); self.hold()
    def test_R_wrong_run(self): self.f.mutate_R(lambda rows:rows[0].update(run_id='b'*32)); self.hold()
    def test_R_wrong_attempt(self): self.f.mutate_R(lambda rows:rows[0].update(attempt_id='b'*32)); self.hold()
    def test_R_wrong_probe_version(self): self.f.mutate_R(lambda rows:rows[0].update(probe_version='future')); self.hold()
    def test_R_claim_online(self): self.f.mutate_R(lambda rows:rows[0].update(original_online_R_available=True)); self.hold()
    def test_R_C_mismatch(self): self.f.mutate_R(lambda rows:rows[0].update(C_raw=1-rows[0]['C_raw'])); self.hold()
    def test_R_prob_outside(self): self.f.mutate_R(lambda rows:rows[0].update(b_reconstructed=1.1)); self.hold()
    def test_nonfinite_rejected_without_serialize(self):
        row=dict(self.f.R_rows[20][0],R_reconstructed=math.nan)
        update=self.f.updates[19]; gate=json.loads((self.f.native/'logs'/f'behavior-probability-attempt-{update["attempt_id"]}.json').read_bytes())
        Csummary=json.loads((self.f.C/'TRAIN-20-summary.json').read_bytes())
        indexed=bridge.index_C(self.f.C/'TRAIN-20-raw-C.jsonl',Csummary,gate,update,self.f.run_id,pearson)
        score=bridge.load_native_score(self.f.native/'repo/verl/trainer/main_ppo_eco.py',self.f.run['source_sha256']['verl/trainer/main_ppo_eco.py'],bridge.CONSTANTS)
        with self.assertRaises(bridge.Hold): bridge.validate_R_row(row,0,indexed,update,self.f.run_id,score)
    def test_wrong_python_reward(self): self.f.mutate_R(lambda rows:rows[0].update(R_python_before_FP32=4.)); self.hold()
    def test_not_native_final_FP32(self): self.f.mutate_R(lambda rows:rows[0].update(R_reconstructed=rows[0]['R_reconstructed']+1e-14)); self.hold()
    def test_R_summary_partial(self): self.f.mutate_R_summary(lambda x:x.update(actual_probe_rows=2559)); self.hold()
    def test_R_summary_unaccepted(self): self.f.mutate_R_summary(lambda x:x.update(status='RUNNING')); self.hold()
    def test_R_summary_changed_permutation(self): self.f.mutate_R_summary(lambda x:x.update(post_balance_permutation_sha256='0'*64)); self.hold()
    def test_R_summary_fake_moment(self): self.f.mutate_R_summary(lambda x:x.update(sum_CR=x['sum_CR']+1.)); self.hold()
    def test_native_precise_mean_mismatch(self):
        self.f.updates[19]['metrics']['critic/rewards/mean']+=.01; jsonl(self.f.native/'logs/actual-updates.jsonl',self.f.updates); self.hold()
    def test_native_extrema_gate(self):
        self.f.mutate_R_summary(lambda x:x['native_scalar_comparison']['native_printed'].update({'critic/score/max':9.})); self.hold()
    def test_missing_bitwise_caveat(self):
        self.f.mutate_R_summary(lambda x:x['native_scalar_comparison'].update(native_scalar_agreement_does_not_prove_online_per_row_bitwise_identity=False)); self.hold()


class CPUImportFixtures(unittest.TestCase):
    def test_production_rejects_Mac_or_wrong_node_and_never_imports_Torch(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=Path(root).resolve()/'wrong-node.json'; cfgsha=put(cfg,dict(expected_node='impossible-synthetic-node'))
            stdout=io.StringIO()
            with patch.object(sys,'argv',['selected_pair_statistics.py','--config',str(cfg),
                    '--config-sha256',cfgsha,'--helper-sha256',sha(bridge.__file__)]),contextlib.redirect_stdout(stdout):
                self.assertEqual(bridge.main(),2)
            result=json.loads(stdout.getvalue()); self.assertEqual(result['status'],'HOLD')
            self.assertEqual(result['reason'],'not_bound_remote_CPU_process')
            self.assertTrue(result['raw_exception_not_exported'])
            self.assertEqual(sorted(x.name for x in Path(root).iterdir()),['wrong-node.json'])
        self.assertNotIn('torch',sys.modules)
    def test_sparse_zero_variance_undefined(self):
        run='a'*32; step=20; attempt='b'*32
        ids=[f'{run}-{step}-attempt-{attempt}-{n//5}-{h(n//5)}-sample-{n%5}' for n in range(2560)]
        acc=bridge.selected_accumulator(pearson,dict(run_id=run,selected_steps=[20],updates=[dict(step=20,attempt_id=attempt,
            trajectory_ids_sha256=pearson.trajectory_ids_sha256(ids))]))
        for identity in ids: acc.add(dict(run_id=run,step=step,attempt_id=attempt,trajectory_id=identity,C_raw=0,R_reconstructed=.05))
        result=acc.finish(); self.assertIsNone(result['pooled']['Pearson']['value']); self.assertFalse(result['pooled']['Pearson']['defined']); self.assertEqual(result['total_pairs'],2560)


if __name__ == '__main__': unittest.main()
