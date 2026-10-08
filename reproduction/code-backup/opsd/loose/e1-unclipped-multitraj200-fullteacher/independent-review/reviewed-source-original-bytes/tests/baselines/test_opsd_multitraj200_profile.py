"""New-profile CPU contract/consumer tests; none constitutes GPU acceptance."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from search_r1.baselines import checkpoint_gate as gate
from search_r1.baselines.e1_profile import compose_profile, profile_contract, validate_profile
from tests.baselines.test_opsd_e1_checkpoint_gate import metrics, rsa, signed_gate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    for key, value in {
        'OPSD_E1_OFFICIAL_MODEL': str(tmp_path/'model'),
        'OPSD_E1_COMMON_MANIFEST': str(tmp_path/'manifest.json'),
        'OPSD_E1_COMMON_DATA_ROOT': str(tmp_path/'data'),
        'SEARCH_R1_BASELINE_RUN_ROOT': str(tmp_path/'run'),
        'OPSD_E1_REVIEW_PUBLIC_KEY': str(tmp_path/'public.json'),
        'OPSD_E1_REVIEW_PUBLIC_KEY_SHA256': 'a'*64,
        'OPSD_E1_BINDING_MANIFEST': str(tmp_path/'binding.json'),
        'OPSD_E1_BINDING_MANIFEST_SHA256': 'b'*64,
    }.items():
        monkeypatch.setenv(key, value)
    return compose_profile('multitraj200')


def multi_metrics():
    result = metrics()
    for rank in range(8):
        result[f'actor/optimizer_steps/rank_{rank}'] = 10
    return result


def test_exact_new_budget_and_preserved_old_contracts(cfg):
    result = validate_profile(cfg)
    actor = result['actor_rollout_ref']['actor']
    rollout = result['actor_rollout_ref']['rollout']
    assert result['baselines']['opsd_e1']['profile'] == 'multitraj200'
    assert 512*rollout['n_agent'] == 2560
    assert 2560//actor['ppo_mini_batch_size'] == 10
    assert actor['ppo_mini_batch_size']//8 == 32
    assert actor['ppo_micro_batch_size']//8 == 4
    assert actor['ppo_mini_batch_size']//actor['ppo_micro_batch_size'] == 8
    assert rollout['log_prob_micro_batch_size']//8 == 8
    assert result['actor_rollout_ref']['ref']['log_prob_micro_batch_size'] == 64
    assert (rollout['temperature'], rollout['top_p'], rollout['top_k']) == (1, 1, -1)
    assert result['baselines']['distillation']['component_clip'] is None
    assert result['baselines']['distillation']['response_only_logits'] is False
    assert result['baselines']['distillation']['teacher_forward_micro_batch'] == 4
    assert not actor['use_kl_loss'] and actor['kl_loss_coef'] == actor['entropy_coeff'] == 0
    assert actor['optim']['lr'] == 1e-6 and actor['optim']['lr_warmup_steps'] == 57
    assert actor['grad_clip'] == 1 and actor['ppo_epochs'] == 1
    old = validate_profile(compose_profile())
    author = validate_profile(compose_profile('author200'))
    assert old['trainer']['total_training_steps'] == 300
    assert old['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 128
    assert old['actor_rollout_ref']['actor']['ppo_micro_batch_size'] == 16
    assert author['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 32
    assert author['actor_rollout_ref']['actor']['optim']['lr'] == 5e-6
    assert author['baselines']['distillation']['component_clip'] == .05
    assert profile_contract()['optimizer_steps'] == 20
    assert profile_contract('author200')['optimizer_steps'] == 16


@pytest.mark.parametrize('key, value', [
    ('actor_rollout_ref.actor.ppo_mini_batch_size', 128),
    ('actor_rollout_ref.actor.ppo_mini_batch_size', 32),
    ('actor_rollout_ref.actor.ppo_micro_batch_size', 16),
    ('actor_rollout_ref.actor.grad_clip', .1),
    ('actor_rollout_ref.actor.optim.lr', 5e-6),
    ('actor_rollout_ref.actor.optim.lr_warmup_steps', 0),
    ('actor_rollout_ref.actor.optim.lr_warmup_steps_ratio', .1),
    ('actor_rollout_ref.rollout.n_agent', 1),
    ('actor_rollout_ref.rollout.n', 5),
    ('actor_rollout_ref.rollout.temperature', 1.1),
    ('actor_rollout_ref.rollout.top_p', .95),
    ('actor_rollout_ref.rollout.top_k', 20),
    ('actor_rollout_ref.rollout.log_prob_micro_batch_size', 32),
    ('actor_rollout_ref.ref.log_prob_micro_batch_size', 32),
    ('baselines.distillation.component_clip', .05),
    ('baselines.distillation.response_only_logits', True),
    ('baselines.distillation.response_only_logits', 0),
    ('baselines.distillation.response_only_logits', None),
    ('baselines.distillation.teacher_forward_micro_batch', True),
    ('baselines.distillation.teacher_forward_micro_batch', 1),
    ('baselines.distillation.teacher_forward_micro_batch', 2),
    ('baselines.distillation.teacher_forward_micro_batch', -1),
    ('baselines.distillation.teacher_forward_micro_batch', 2.0),
    ('baselines.distillation.teacher_forward_micro_batch', 4.0),
    ('baselines.distillation.temperature', 1.1),
    ('baselines.distillation.beta', .1),
    ('actor_rollout_ref.actor.use_kl_loss', True),
    ('algorithm.kl_ctrl.kl_coef', .001),
    ('actor_rollout_ref.actor.entropy_coeff', .001),
    ('trainer.total_training_steps', 201),
    ('trainer.save_freq', 25),
    ('data.train_batch_size', 256),
    ('baselines.opsd_e1.profile', 'e1_300'),
    ('baselines.opsd_e1.profile', 'author200'),
])
def test_new_profile_drift_refused(cfg, key, value):
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_profile(cfg)


def test_exact_eight_rank_calls_and_samples_in_standalone_frozen_consumer():
    # Production external reviewers compile these actual bytes outside a package.
    import types
    path = ROOT/'search_r1/baselines/checkpoint_gate.py'
    external = types.ModuleType('first5_frozen_gate')
    external.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), external.__dict__)
    assert external.__package__ is None
    external.validate_outer(multi_metrics(), profile='multitraj200')
    external.validate_outer(metrics())
    for profile in ('e1_300', 'author200'):
        with pytest.raises(RuntimeError):
            external.validate_outer(multi_metrics(), profile=profile)
    for key, value in [('actor/optimizer_steps/rank_7', 9),
                       ('actor/optimizer_steps/rank_0', 20),
                       ('actor/samples/rank_3', 64),
                       ('actor/optimizer_steps/rank_0', True),
                       ('actor/skipped_empty_minibatches/rank_2', 1),
                       ('actor/opsd_loss_finite/rank_6', 0)]:
        with pytest.raises(RuntimeError):
            external.validate_outer(dict(multi_metrics(), **{key: value}), profile='multitraj200')


def test_signed_five_gate_rejects_old_clocks_then_accepts_new_clocks(signed_gate):
    g, path, manifest, payload, sign = signed_gate
    g.profile = 'multitraj200'
    manifest.update(horizon=200, warmup_steps=57)
    path.write_bytes(gate.canonical(manifest))
    payload.update(horizon=200, warmup_steps=57,
                   checkpoint_manifest_sha256=gate.sha256(path))
    g.receipt.write_bytes(gate.canonical(sign(payload)))
    with pytest.raises(RuntimeError, match='rank update audit'):
        g.verify_receipt(g.receipt, path, manifest)
    rows = [dict(actual_outer_updates=i, global_step=i, driver_pid=g.pid,
        job_id=g.job, source_commit=g.source, config_sha256=g.config_sha,
        metrics=multi_metrics()) for i in range(1, 6)]
    raw = ''.join('OPSD_E1_OUTER_JSON='+json.dumps(row)+'\n' for row in rows).encode()
    prefix = payload['train_log_prefix']
    snapshot = Path(prefix['snapshot_path']); snapshot.write_bytes(raw)
    Path(prefix['live_path']).write_bytes(raw)
    prefix.update(sha256=gate.sha256(snapshot), bytes=len(raw))
    g.receipt.write_bytes(gate.canonical(sign(payload)))
    g.verify_receipt(g.receipt, path, manifest)
    assert g.horizon == 200 and g.contract['optimizer_steps']*5 == 50


def test_precise_save_points_and_no201():
    assert [i for i in range(1, 202) if gate.should_save(i, 200, 'multitraj200')] == [5, 50, 100, 150, 200]
    assert [i for i in range(1, 301) if gate.should_save(i)] == [5]+list(range(25, 301, 25))
    assert [i for i in range(1, 202) if gate.should_save(i, 200, 'author200')] == [5]+list(range(25, 201, 25))
    for step, horizon in ((True, 200), (5, 300)):
        with pytest.raises(ValueError):
            gate.should_save(step, horizon, 'multitraj200')


def test_exact_adam50_scheduler_outer57_and_actual_used_lr(cfg):
    import torch
    from search_r1.baselines.full_state import resolve_warmup_steps
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    assert resolve_warmup_steps(cfg.actor_rollout_ref.actor.optim) == 57
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=57)
    assert optimizer.param_groups[0]['lr'] == 0
    initial = model.weight.detach().clone()
    for outer in range(1, 59):
        used_lr = 1e-6*min(1, (outer-1)/57)
        for _ in range(10):
            assert optimizer.param_groups[0]['lr'] == used_lr
            optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
        scheduler.step()
        assert scheduler.last_epoch == outer
        assert optimizer.param_groups[0]['lr'] == 1e-6*min(1, outer/57)
        if outer == 1:
            assert torch.equal(model.weight, initial)  # Adam clocks advance at LR0.
        if outer == 5:
            assert all(int(state['step']) == 50 for state in optimizer.state.values())
            assert not torch.equal(model.weight, initial)


def test_fixed_launcher_and_real_hydra_cpu_entry(cfg):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1',
               PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
    launcher = ROOT/'reproduction/baselines/run_opsd_e1_multitraj200.sh'
    result = subprocess.run(['bash', str(launcher), '--cfg'], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    d = json.loads(result.stdout)
    assert d['baselines']['opsd_e1']['profile'] == 'multitraj200'
    assert d['actor_rollout_ref']['actor']['ppo_micro_batch_size'] == 32
    assert d['actor_rollout_ref']['rollout']['log_prob_micro_batch_size'] == 64
    code = '''import contextlib,io,json,sys,torch
from omegaconf import OmegaConf
from search_r1.baselines.main import main
from search_r1.baselines.e1_profile import validate_profile
assert not torch.cuda.is_initialized()
sys.argv=['main','--config-name','ppo_opsd_e1_multitraj200','--cfg','job','--resolve']
out=io.StringIO()
with contextlib.redirect_stdout(out):main()
d=validate_profile(OmegaConf.create(out.getvalue()))
assert d['baselines']['opsd_e1']['profile']=='multitraj200'
assert d['actor_rollout_ref']['actor']['ppo_mini_batch_size']==256
assert not torch.cuda.is_initialized()
print('ACTUAL_IMPORTED_MULTITRAJ200_HYDRA_CPU_ONLY_PASS')
'''
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'ACTUAL_IMPORTED_MULTITRAJ200_HYDRA_CPU_ONLY_PASS' in result.stdout
    for args in (['--typo'], ['--run', 'trainer.total_training_steps=201']):
        result = subprocess.run(['bash', str(launcher), *args], env=env, text=True, capture_output=True)
        assert result.returncode == 2 and 'Only --cfg' in result.stderr


def test_actual_fit_tail_first5_before6_and_final200_no201():
    # Exercise the real trainer control-flow tail; fixtures are CPU schema only.
    source = ROOT/'verl/trainer/ppo/ray_trainer.py'
    cls = next(x for x in ast.parse(source.read_text()).body
               if isinstance(x, ast.ClassDef) and x.name == 'RayPPOTrainer')
    fit = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'fit')
    epoch = next(x for x in fit.body if isinstance(x, ast.For))
    batch = next(x for x in epoch.body if isinstance(x, ast.For))
    start = next(i for i, x in enumerate(batch.body) if isinstance(x, ast.If)
        and isinstance(x.test, ast.Name) and x.test.id == 'e1'
        and any(isinstance(y, ast.Constant) and y.value == 'OPSD_E1_OUTER_JSON='
                for y in ast.walk(x)))
    loop = ast.parse('''for consumed in iterator:
    self.actual_outer_updates += 1
    assert self.global_steps == consumed
''').body[0]
    loop.body.extend(batch.body[start:])
    fn = ast.FunctionDef(name='drive', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=x) for x in ('self', 'metrics', 'e1', 'logger', 'iterator')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[loop], decorator_list=[])
    namespace = dict(os=os, json=json, validate_evaluation=gate.validate_evaluation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(source), 'exec'), namespace)
    fetched=[]; pauses=[]; finals=[]
    def iterator():
        for i in range(1, 202):
            fetched.append(i); yield i
    obj = SimpleNamespace(global_steps=1, actual_outer_updates=0, total_training_steps=200,
        e1_pause_manifest=('CPU_FIXTURE', {}), val_reward_fn=object(),
        _prepare_final_validation=lambda: finals.append(obj.global_steps),
        _validate=lambda: {'val/qa_count/all':51713, 'val/evaluated_count/all':51713})
    def release(*args):
        assert fetched == list(range(1, 6))
        pauses.append(obj.global_steps)
    obj.e1_gate = SimpleNamespace(job='CPU_FIXTURE', source='source', config_sha='config',
                                 horizon=200, profile='multitraj200', wait=release)
    namespace['drive'](obj, {}, True, SimpleNamespace(log=lambda **kwargs: None), iterator())
    assert fetched == list(range(1, 201)) and pauses == [5] and finals == [200]
    assert obj.global_steps == obj.actual_outer_updates == 200
