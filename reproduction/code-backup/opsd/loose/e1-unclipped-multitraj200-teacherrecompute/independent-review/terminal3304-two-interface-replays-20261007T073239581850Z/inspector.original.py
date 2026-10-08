from pathlib import Path
import hashlib,types,unittest
c=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute/terminal3304-observation-candidate')
p=c/'collect_actual3304_final_originals.py';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='00934aab798f1abac7bdf8529603023722c5e3e812b25197894fed9536b4b0b7'
m=types.ModuleType('independent_exact_3304');m.__file__=str(p);exec(compile(raw,str(p),'exec'),m.__dict__)
t=c/'test_terminal3304_delta.py';tr=t.read_bytes();assert hashlib.sha256(tr).hexdigest()=='a71f528ef8dbd5f3455ca0c71ec9edbd0e46b42847a6191c39e3bd448807d8a4'
n=types.ModuleType('independent_fixture_3304');n.__file__=str(t);exec(compile(tr,str(t),'exec'),n.__dict__);n.c=m
suite=unittest.TestSuite(n.Terminal3304Delta(name) for name in ('test_actual18_origin_map_and_replacement_cannot_close','test_physical8_record_detects_rewrite_and_literal_child0_wrapper1'))
r=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if r.wasSuccessful() else 1)
