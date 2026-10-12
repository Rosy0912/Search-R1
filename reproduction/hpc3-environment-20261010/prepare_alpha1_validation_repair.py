"""Prepare isolated repair; never submit a job or choose a rollback point."""
import ast
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True

BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
ROOT = BASE / 'runtime'
OUT = BASE / 'outputs/training-preparation'
OLD_SOURCE = ROOT / 'tools/topd-alpha1-source-20261012'
OLD_LAUNCH = ROOT / 'tools/topd-alpha1-launch-20261012'
NEW_SOURCE = ROOT / 'tools/topd-alpha1-source-validation-repair-20261013'
NEW_LAUNCH = ROOT / 'tools/topd-alpha1-launch-validation-repair-20261013'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_order(source):
    tree = ast.parse(source)
    fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    lines = {}
    for n in ast.walk(fit):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            lines.setdefault(n.func.attr, []).append(n.lineno)
    save = lines['_save_checkpoint']
    stage = lines['stage_validation']
    assert len(save) == len(stage) == 1 and save[0] < stage[0]
    # The regular in-loop validation must follow persistence too.
    assert all(n > save[0] for n in lines['_validate'] if n > 800)
    return {'save_line': save[0], 'stage_validation_line': stage[0]}


def validate_failure_order(source):
    """Execute the exact reordered block against fake trainer objects."""
    start = source.index('                    # Persist scheduled full recovery state before inference.')
    index = source.index('# collect metrics', start)
    end = source.rfind('\n', 0, index) + 1
    import textwrap
    executable = '\n'.join(line for line in source[start:end].splitlines()
                           if line.strip() and not line.lstrip().startswith('#'))
    block = textwrap.dedent(executable)
    events = []
    class C:
        def __init__(self, **values): self.__dict__.update(values)
    class T:
        global_steps = 50
        val_reward_fn = True
        config = C(trainer=C(save_freq=50, acceptance_step=5, test_freq=0))
        total_training_steps = 201
        def _save_checkpoint(self): events.append('durable_checkpoint')
        def _validate(self): events.append('normal_validation'); raise RuntimeError('injected validation failure')
    class Controller:
        def stage_validation(self, step):
            events.append('stage_validation'); raise RuntimeError('injected validation failure')
    class Timer:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    t = T(); t.baseline_controller = Controller()
    for regular in (False, True):
        events.clear(); t.config.trainer.test_freq = 50 if regular else 0
        try:
            exec(block, {'self': t, '_timer': lambda *a: Timer(), 'timing_raw': {}, 'metrics': {'actor/optimizer_steps': 20}})
        except RuntimeError as exc:
            assert str(exc) == 'injected validation failure'
        else: raise AssertionError('Injected failure not reached')
        assert events == ['durable_checkpoint', 'normal_validation' if regular else 'stage_validation'], events
    return {'injected_regular_and_stage_validation_failure_after_save': True}


def main():
    accounting = subprocess.check_output(['sacct', '-j', '712204', '-n', '-P', '--format=JobID,State,ExitCode'], text=True)
    assert '712204|FAILED|1:0' in accounting
    queued = subprocess.check_output(['squeue', '-u', 'mou176', '-h', '-o', '%i'], text=True).splitlines()
    assert '712204' not in queued
    if NEW_SOURCE.exists() or NEW_LAUNCH.exists():
        # A preparation-only failure may have left pristine copies; reuse only
        # after confirming both edited inputs still equal their historical source.
        current = (NEW_SOURCE/'verl/trainer/ppo/ray_trainer.py').read_text()
        if '# Persist scheduled full recovery state before inference.' in current:
            validate_order(current); validate_failure_order(current)
        else:
            assert sha(NEW_SOURCE/'verl/trainer/ppo/ray_trainer.py') == sha(OLD_SOURCE/'verl/trainer/ppo/ray_trainer.py')
            assert sha(NEW_LAUNCH/'runtime/student_phase.py') == sha(OLD_LAUNCH/'runtime/student_phase.py')
    else:
        shutil.copytree(OLD_SOURCE, NEW_SOURCE, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(OLD_LAUNCH, NEW_LAUNCH, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    p = NEW_SOURCE / 'verl/trainer/ppo/ray_trainer.py'
    source = p.read_text()
    if '# Persist scheduled full recovery state before inference.' not in source:
        validate_start = source.index('                # validate\n')
        index = source.index('if self.config.trainer.save_freq > 0', validate_start)
        save_start = source.rfind('\n', 0, index) + 1
        index = source.index('# collect metrics', save_start)
        save_end = source.rfind('\n', 0, index) + 1
        save_block = source[save_start:save_end]
        validation_block = source[validate_start:save_start]
        source = source[:validate_start] + '                    # Persist scheduled full recovery state before inference.\n' + save_block + validation_block + source[save_end:]
    order = validate_order(source)
    injected = validate_failure_order(source)
    p.write_text(source)
    phase = NEW_LAUNCH / 'runtime/student_phase.py'
    text = phase.read_text()
    assert text.count('batch_size=512,') + text.count('batch_size=32,') == 1
    text = text.replace('batch_size=512,', 'batch_size=32,')
    text = text.replace("'rows':512,'metrics':metrics", "'rows':512,'validation_batch_size':32,'metrics':metrics")
    ast.parse(text); phase.write_text(text)
    assert 'batch_size=32,' in text and 'len(dataset)!=512' in text
    # The evaluator concatenates every row across batches before calculating metrics.
    evaluator = source[source.index('    def _validate('):source.index('\n    def ', source.index('    def _validate(') + 10)]
    assert 'qa_rows.extend(batch_answer_metrics' in evaluator and 'aggregate_answer_metrics(qa_rows)' in evaluator
    assert 'torch.cat([rw.sum(-1) for rw in reward_tensor_lst]' in evaluator
    for p in NEW_LAUNCH.rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.json', '.sbatch', '.sh'):
            text = p.read_text().replace(str(OLD_SOURCE), str(NEW_SOURCE)).replace(str(OLD_LAUNCH), str(NEW_LAUNCH))
            text = text.replace('$TASK_ROOT/tools/topd-alpha1-launch-20261012', '$TASK_ROOT/tools/topd-alpha1-launch-validation-repair-20261013')
            if p.suffix == '.py': ast.parse(text)
            p.write_text(text)
    c = json.loads((OUT / 'hpc3-topd200-alpha1.json').read_text())
    c = json.loads(json.dumps(c).replace(str(OLD_SOURCE), str(NEW_SOURCE)).replace(str(OLD_LAUNCH), str(NEW_LAUNCH)))
    # Remove only import caches generated by this preparation attempt.
    for root in (NEW_SOURCE, NEW_LAUNCH):
        for cache in root.rglob('__pycache__'):
            shutil.rmtree(cache)
    for key in ['source_manifest', 'tools_manifest', 'launcher_manifest']:
        p = pathlib.Path(c[key]['path']); record = json.loads(p.read_text())
        record['files'] = {str(f.relative_to(p.parent)): {'size': f.stat().st_size, 'sha256': sha(f)} for f in p.parent.rglob('*') if f.is_file() and f != p and '__pycache__' not in f.parts}
        p.write_text(json.dumps(record, sort_keys=True, indent=2)); c[key]['sha256'] = sha(p)
    config = OUT / 'hpc3-topd200-alpha1-validation-repair-PREPARED-NOT-SUBMITTED.json'
    config.write_text(json.dumps(c, indent=2))
    sys.path.insert(0, str(NEW_LAUNCH / 'runtime'))
    from supervise import verify_recipe
    from student_binding import check
    assert verify_recipe(c) == 200; check(c)
    report = {'epoch': time.time(), 'status': 'ISOLATED_REPAIR_PREPARED_CPU_VALIDATED_NOT_SUBMITTED', 'source': str(NEW_SOURCE), 'launcher': str(NEW_LAUNCH), 'config': str(config), 'changes': ['scheduled full recovery save before regular and stage validation', 'same fixed512 rows in batches32 with row-level aggregation'], 'order_test': order, 'failure_injection_test': injected, 'changed_files': {str(p): sha(p) for p in [NEW_SOURCE/'verl/trainer/ppo/ray_trainer.py', phase]}, 'GPU_validation_tested': False, 'CUDA_originating_kernel_proved': False, 'rollback_or_initialization_selected': False, 'GPU_job_submitted': False, 'existing_final_job_untouched': 712266}
    (OUT / 'alpha1-validation-repair-preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
