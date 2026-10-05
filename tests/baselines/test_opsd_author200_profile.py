"""CPU-only production profile, signed gate and real trainer-tail regressions.

Fixtures here are explicitly schema/control-flow evidence, never GPU acceptance.
"""
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
    return compose_profile('author200')


def author_metrics():
    result = metrics()
    for rank in range(8):
        result['actor/optimizer_steps/rank_'+str(rank)] = 16
        result['actor/samples/rank_'+str(rank)] = 64
    return result


def test_author200_exact_budget_and_old_profile_unchanged(cfg):
    result = validate_profile(cfg)
    actor = result['actor_rollout_ref']['actor']
    rollout = result['actor_rollout_ref']['rollout']
    assert result['baselines']['opsd_e1']['profile'] == 'author200'
    assert 512*rollout['n_agent'] == 512
    assert 512//actor['ppo_mini_batch_size'] == 16
    assert actor['ppo_mini_batch_size']//8 == 4
    assert actor['ppo_micro_batch_size']//8 == 2
    assert actor['ppo_mini_batch_size']//actor['ppo_micro_batch_size'] == 2
    assert actor['optim']['lr'] == 5e-6 and actor['optim']['lr_warmup_steps'] == 0
    assert result['baselines']['distillation']['temperature'] == 1
    assert result['baselines']['distillation']['component_clip'] == .05
    old = validate_profile(compose_profile())
    assert old['trainer']['total_training_steps'] == 300
    assert old['actor_rollout_ref']['actor']['optim']['lr_warmup_steps'] == 57
    assert old['baselines']['distillation']['component_clip'] is None
    assert old['actor_rollout_ref']['rollout']['n_agent'] == 5


@pytest.mark.parametrize('key, value', [
    ('actor_rollout_ref.actor.ppo_mini_batch_size', 128),
    ('actor_rollout_ref.actor.ppo_micro_batch_size', 32),
    ('actor_rollout_ref.actor.grad_clip', 1),
    ('actor_rollout_ref.actor.optim.lr', 1e-6),
    ('actor_rollout_ref.actor.optim.lr_warmup_steps', 57),
    ('actor_rollout_ref.actor.optim.lr_warmup_steps_ratio', .1),
    ('actor_rollout_ref.rollout.n_agent', 5),
    ('actor_rollout_ref.rollout.n', 5),
    ('actor_rollout_ref.rollout.temperature', 1),
    ('actor_rollout_ref.rollout.top_p', 1),
    ('actor_rollout_ref.rollout.top_k', -1),
    ('actor_rollout_ref.rollout.log_prob_micro_batch_size', 16),
    ('baselines.distillation.component_clip', None),
    ('baselines.distillation.temperature', 1.1),
    ('baselines.distillation.beta', .1),
    ('actor_rollout_ref.actor.use_kl_loss', True),
    ('algorithm.kl_ctrl.kl_coef', .001),
    ('trainer.total_training_steps', 201),
    ('data.train_batch_size', 32),
    ('baselines.opsd_e1.profile', 'e1_300'),
    ('baselines.opsd_e1.profile', 'unreviewed'),
])
def test_author_profile_drift_is_rejected(cfg, key, value):
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_profile(cfg)


def test_each_rank_actual16_and64_samples_required():
    good = author_metrics()
    gate.validate_outer(good, profile='author200')
    with pytest.raises(RuntimeError):
        gate.validate_outer(good)  # The old profile cannot accept author200 clocks.
    with pytest.raises(RuntimeError):
        gate.validate_outer(metrics(), profile='author200')
    for key, value in [('actor/optimizer_steps/rank_7', 15),
                       ('actor/samples/rank_3', 320),
                       ('actor/optimizer_steps/rank_0', True),
                       ('actor/opsd_loss_finite/rank_6', 0)]:
        mutated = dict(good, **{key: value})
        with pytest.raises(RuntimeError):
            gate.validate_outer(mutated, profile='author200')
    for bad in ('unknown', None, True):
        with pytest.raises(ValueError):
            profile_contract(bad)


def test_actual_external_standalone_gate_bytes_accept_new_budget():
    import importlib.util
    path = ROOT/'search_r1/baselines/checkpoint_gate.py'
    spec = importlib.util.spec_from_file_location('first5_frozen_gate', path)
    external = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external)
    assert external.__package__ == ''
    external.validate_outer(author_metrics(), profile='author200')
    external.validate_outer(metrics())
    with pytest.raises(RuntimeError):
        external.validate_outer(metrics(), profile='author200')


def test_author200_signed_gate_requires_new_budget_before_sixth(signed_gate):
    g, path, manifest, payload, sign = signed_gate
    g.profile = 'author200'
    manifest.update(horizon=200, warmup_steps=0)
    path.write_bytes(gate.canonical(manifest))
    payload.update(horizon=200, warmup_steps=0,
                   checkpoint_manifest_sha256=gate.sha256(path))
    # Even a signed envelope and manifest are insufficient with old 20/320 rows.
    g.receipt.write_bytes(gate.canonical(sign(payload)))
    with pytest.raises(RuntimeError, match='rank update audit'):
        g.verify_receipt(g.receipt, path, manifest)
    rows = [dict(actual_outer_updates=i, global_step=i, driver_pid=g.pid,
        job_id=g.job, source_commit=g.source, config_sha256=g.config_sha,
        metrics=author_metrics()) for i in range(1, 6)]
    raw = ''.join('OPSD_E1_OUTER_JSON='+json.dumps(row)+'\n' for row in rows).encode()
    prefix = payload['train_log_prefix']
    snapshot = Path(prefix['snapshot_path']); snapshot.write_bytes(raw)
    Path(prefix['live_path']).write_bytes(raw)
    prefix.update(sha256=gate.sha256(snapshot), bytes=len(raw))
    g.receipt.write_bytes(gate.canonical(sign(payload)))
    g.verify_receipt(g.receipt, path, manifest)
    assert g.horizon == 200 and g.contract['warmup_steps'] == 0


def test_production_fit_tail_stops_at200_before201_fetch():
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
    prefix = ast.parse('''for consumed in iterator:
    self.actual_outer_updates += 1
    assert self.global_steps == consumed
''').body[0]
    prefix.body.extend(batch.body[start:])
    fn = ast.FunctionDef(name='drive', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=x) for x in ('self', 'metrics', 'e1', 'logger', 'iterator')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[prefix], decorator_list=[])
    namespace = dict(os=os, json=json, validate_evaluation=gate.validate_evaluation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(source), 'exec'), namespace)
    fetched=[]; pauses=[]; finals=[]; logs=[]
    def iterator():
        for i in range(1, 202):
            fetched.append(i)
            yield i
    obj = SimpleNamespace(global_steps=1, actual_outer_updates=0, total_training_steps=200,
        e1_pause_manifest=('fixture', {}), val_reward_fn=object(),
        _prepare_final_validation=lambda: finals.append(obj.global_steps),
        _validate=lambda: {'val/qa_count/all':51713, 'val/evaluated_count/all':51713})
    def release(*args):
        assert fetched == list(range(1, 6))
        pauses.append(obj.global_steps)
    obj.e1_gate = SimpleNamespace(job='CPU_FIXTURE', source='source', config_sha='config',
                                 horizon=200, wait=release)
    namespace['drive'](obj, {}, True, SimpleNamespace(log=lambda data, step: logs.append(step)), iterator())
    assert fetched == list(range(1, 201))
    assert pauses == [5] and finals == [200] and logs == [200]
    assert obj.global_steps == obj.actual_outer_updates == 200
    assert [i for i in range(1, 202) if gate.should_save(i, horizon=200)] == [5]+list(range(25, 201, 25))


def test_no_warmup_scheduler_constant_from_first_update(cfg):
    import torch
    from search_r1.baselines.full_state import resolve_warmup_steps
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    assert resolve_warmup_steps(cfg.actor_rollout_ref.actor.optim) == 0
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
    for outer in range(1, 6):
        for mini in range(16):
            assert optimizer.param_groups[0]['lr'] == 5e-6
            optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
        scheduler.step()
        assert scheduler.last_epoch == outer
    assert all(int(state['step']) == 80 for state in optimizer.state.values())
    original = torch.optim.AdamW(model.parameters(), lr=1e-6)
    original_scheduler = get_constant_schedule_with_warmup(original, num_warmup_steps=57)
    assert original.param_groups[0]['lr'] == 0
    for outer in range(1, 59):
        original.step(); original_scheduler.step()
        assert original.param_groups[0]['lr'] == 1e-6*min(1, outer/57)


def test_fixed_launcher_and_actual_imported_hydra_select_author200(cfg):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1',
               PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
    launcher = ROOT/'reproduction/baselines/run_opsd_e1_author200.sh'
    result = subprocess.run(['bash', str(launcher), '--cfg'], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = json.loads(result.stdout)
    assert result['baselines']['opsd_e1']['profile'] == 'author200'
    assert result['trainer']['total_training_steps'] == 200
    code = '''import contextlib,io,json,sys,torch
from omegaconf import OmegaConf
from search_r1.baselines.main import main
from search_r1.baselines.e1_profile import validate_profile
assert not torch.cuda.is_initialized()
sys.argv=['main','--config-name','ppo_opsd_e1_author200','--cfg','job','--resolve']
out=io.StringIO()
with contextlib.redirect_stdout(out):main()
d=validate_profile(OmegaConf.create(out.getvalue()))
assert d['baselines']['opsd_e1']['profile']=='author200'
assert d['actor_rollout_ref']['actor']['ppo_mini_batch_size']==32
assert not torch.cuda.is_initialized()
print('ACTUAL_IMPORTED_AUTHOR200_HYDRA_CPU_ONLY_PASS')
'''
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'ACTUAL_IMPORTED_AUTHOR200_HYDRA_CPU_ONLY_PASS' in result.stdout
    for args in (['--typo'], ['--run', 'trainer.total_training_steps=201']):
        result = subprocess.run(['bash', str(launcher)]+args, env=env,
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 2 and 'Only --cfg' in result.stderr
