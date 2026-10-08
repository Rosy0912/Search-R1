"""Run unchanged legacy parser/identity probes against the new candidate module."""
import importlib.util,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('candidate',Path(__file__).with_name('resource_watch.py'));candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
path=root/'outputs/bapo-resource-watch-tests.py';ns={'__name__':'legacy_regression','__file__':str(path)};exec(compile(path.read_text(),str(path),'exec'),ns);ns['w']=candidate
suite=unittest.TestSuite()
excluded={'test_wrapper_returns_pause_never_signals_and_preserves_fds','test_unsupported_monitor_does_not_start_phase','test_high_whole_gpu_fresh_own_supports_foreign_absence'}
for name in unittest.defaultTestLoader.getTestCaseNames(ns['WatchTests']):
 if name not in excluded:suite.addTest(ns['WatchTests'](name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
