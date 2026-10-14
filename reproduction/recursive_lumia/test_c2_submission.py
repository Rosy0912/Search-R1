import json
from pathlib import Path
import shlex
import subprocess
import unittest

from .c2_submission import GROUP, script
from .prepare_ablation import condition_for
from release.recursive_bran_a.ablation150_v1.contracts import planned_conditions


class C2Submission(unittest.TestCase):
    def test_generated_script_resolves_executable_condition_and_keeps_c1_gate(self):
        body = script('/study with spaces', 'a' * 40, 138654)
        subprocess.run(['bash', '-n'], input=body, text=True, check=True)
        exports = {}
        for line in body.splitlines():
            if line.startswith('export '):
                exports.update(token.split('=', 1) for token in shlex.split(line)[1:])
        self.assertEqual(exports['RECURSIVE_GROUP'], GROUP)
        self.assertEqual(condition_for(exports['RECURSIVE_GROUP'])['variant'], 'random_final')
        gate = next(line for line in body.splitlines() if line.startswith('python3 '))
        arguments = shlex.split(gate)
        self.assertEqual(arguments[arguments.index('--job') + 1], '138654')
        self.assertEqual(arguments[arguments.index('--run') + 1], '/study with spaces/runs/C1-root-only')
        self.assertIn('gate_c1_complete', gate)
        self.assertLess(body.index('gate_c1_complete'), body.index('exec bash'))

    def test_study_labels_and_reuse_match_executable_registry(self):
        study = json.loads(Path(__file__).with_name('expanded-study.json').read_text())
        executable = {r['group']: r for r in planned_conditions()}
        self.assertEqual({r['label'] for r in study['ablation_order']}, set(executable))
        for row in study['ablation_order']:
            self.assertEqual(bool(row.get('reuse')), bool(executable[row['label']].get('reuse')))
            if not row.get('reuse'):
                self.assertEqual(condition_for(row['label']), executable[row['label']]['options'])

    def test_mistyped_condition_or_unbound_source_rejected_before_submission(self):
        with self.assertRaises(ValueError):
            condition_for('C2-uniform-final')
        for base, source, job in [('relative', 'a' * 40, 123), ('/study', 'HEAD', 123),
                                  ('/study', 'a' * 40, '123;false')]:
            with self.assertRaises(ValueError):
                script(base, source, job)


if __name__ == '__main__':
    unittest.main()
