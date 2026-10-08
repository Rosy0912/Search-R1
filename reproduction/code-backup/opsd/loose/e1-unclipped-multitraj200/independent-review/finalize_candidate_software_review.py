"""Independent local evidence join only. No deployment, GPU, job or signature."""
import ast
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import types

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent / 'opsd-unclipped-multitraj200'
OUT = TASK / 'independent-review'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def emit(path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with path.open('xb') as f:
        f.write(raw)
    return {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def pinned(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def main():
    finalizer = TASK / 'finalize_runtime_tools.py'
    tree = ast.parse(finalizer.read_bytes())
    source_names = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES'
                                for target in node.targets))
    assert len(source_names) == 19 and len(set(source_names)) == 19
    before = {name: (REPO / name).read_bytes() for name in source_names}
    source_pins = {name: sha(raw) for name, raw in before.items()}
    for name, raw in before.items():
        if name.endswith('.py'):
            compile(raw, str(REPO / name), 'exec', dont_inherit=True)

    execution = json.loads((OUT / 'source-and-first5-final-independent-cpu/execution.original.json').read_bytes())
    assert execution['returncode'] == 0 and execution['before_sha256'] == execution['after_sha256']
    for name, digest in execution['after_sha256'].items():
        assert sha(Path(name).read_bytes()) == digest, name
    assert b'328 passed' in (OUT / 'source-and-first5-final-independent-cpu/stdout.original').read_bytes()
    evidence = []
    for directory, count in [('source-and-first5-final-independent-cpu', 328),
                             ('trajectory-v4-independent-cpu', 21),
                             ('final-numerics-state-independent-cpu', 17),
                             ('resource-component-independent-cpu', 12)]:
        base = OUT / directory
        receipt = json.loads((base / 'execution.original.json').read_bytes())
        assert receipt['returncode'] == 0
        evidence.append({'independently_executed': True, 'test_count': count,
                         'scope': receipt.get('scope', 'Local CPU component checks only'),
                         'originals': [pinned(base / n) for n in ('execution.original.json', 'stdout.original', 'stderr.original')]})

    # Re-run the actual production writer -> repaired consumer at exact reviewed
    # bytes using the old explicitly synthetic local manifest, never a GPU save.
    sys.path.insert(0, str(REPO))
    from search_r1.baselines.trajectory_archive import TrajectoryArchive
    old = json.loads((OUT / 'archive-scope-writer-consumer-counterexample/counterexample.original.json').read_bytes())
    assert source_pins['search_r1/baselines/trajectory_archive.py'] == old['producer_sha256']
    writer = TrajectoryArchive.__new__(TrajectoryArchive)
    writer.completed_outer = 5
    writer.last_manifest = copy.deepcopy(old['actual_production_method_output']['last_manifest'])
    state = writer.state_dict(5)
    native_path = TASK / 'first5-native-candidate/inspect_payloads.py'
    native_raw = native_path.read_bytes()
    assert sha(native_raw) == '25d5ef8659fbf22c466ed7da92881b5cde276523726875f7a64f1925376a88e5'
    module = types.ModuleType('independent_exact_native_scope_reader')
    module.__file__ = str(native_path)
    exec(compile(native_raw, str(native_path), 'exec', dont_inherit=True), module.__dict__)
    manifest = {'files': {'trajectory_outer_000005/paired-manifest.json':
                         {k: writer.last_manifest[k] for k in ('path', 'sha256')}}}
    assert module.archive_progress({'trajectory_archive': state}, manifest) == state
    rejected = []
    for label, scope in [('wrong', 'wrong-scope'), ('none', None), ('bool', True)]:
        bad = copy.deepcopy(state)
        bad['scope'] = scope
        try:
            module.archive_progress({'trajectory_archive': bad}, manifest)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    for label in ('missing', 'unknown'):
        bad = copy.deepcopy(state)
        if label == 'missing':
            del bad['scope']
        else:
            bad['unknown'] = 1
        try:
            module.archive_progress({'trajectory_archive': bad}, manifest)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    repair = emit(OUT / 'ACTUAL_WRITER_REPAIRED_READER_INDEPENDENT_LOCAL_JOIN.json', {
        'status': 'PASS_INDEPENDENT_LOCAL_ACTUAL_PRODUCTION_METHOD_SCHEMA_REPAIR_ONLY',
        'scope': 'Production state_dict method and exact repaired native consumer with explicitly synthetic local manifest; not actual first5 checkpoint',
        'producer_sha256': old['producer_sha256'], 'consumer': pinned(native_path),
        'actual_production_method_output': state, 'strict_invalid_schema_cases_rejected': rejected,
        'old_rejected_counterexample': pinned(OUT / 'archive-scope-writer-consumer-counterexample/counterexample.original.json'),
    })

    tool_names = [f'formal-tools-candidate/{name}' for name in (
        'formal_job.sh', 'formal_supervisor.py', 'owned_runtime.py', 'resource_sample.py',
        'retriever_entry.py', 'ray_cpu_check.py', 'hardware_rank.py', 'cpu_import_check.py')]
    tool_names += ['first5-native-candidate/inspect_payloads.py',
                   'first5-receipt-candidate/hf_cpu_reload.py',
                   'first5-receipt-candidate/inspect_first5_cpu.py',
                   'first5-receipt-candidate/receipt_payload.py', 'finalize_runtime_tools.py']
    tool_pins = {name: sha((TASK / name).read_bytes()) for name in tool_names}
    for name in tool_names:
        if name.endswith('.py'):
            compile((TASK / name).read_bytes(), str(TASK / name), 'exec', dont_inherit=True)
    additional_names = [
        'docs/superpowers/specs/2026-10-06-opsd-unclipped-multitraj200-design.md',
        'docs/superpowers/specs/2026-10-06-opsd-trajectory-archive-interface.md',
        'tests/baselines/test_opsd_multitraj200_profile.py',
        'tests/baselines/test_opsd_owned_log_framing.py',
        'tests/baselines/test_opsd_response_only_logits.py',
        'tests/baselines/test_opsd_trajectory_archive.py',
    ]
    additional = {name: sha((REPO / name).read_bytes()) for name in additional_names}
    snapshot = OUT / 'reviewed-source-original-bytes'
    snapshot.mkdir()
    for name, raw in before.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(raw)
    assert before == {name: (REPO / name).read_bytes() for name in source_names}, 'Source drift during independent final join'
    assert tool_pins == {name: sha((TASK / name).read_bytes()) for name in tool_names}, 'Tool drift during independent final join'
    report = {
        'version': 1,
        'status': 'PASS_INDEPENDENT_SOURCE_AND_TOOL_CANDIDATE_BYTES_SOFTWARE_ONLY',
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Independent design, current exact source/candidate-tool static review and local CPU checks. Eligible for normal local source commit and subsequent exact committed-source/tool binding review. Not GPU, job, capacity, first5, effect, checkpoint200, final51713 or terminal acceptance.',
        'repository': str(REPO),
        'existing_base_head': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'reviewed_source_files': source_pins,
        'additional_reviewed_docs_and_tests': additional,
        'reviewed_candidate_tool_files': tool_pins,
        'source_commit_identity_pending_actual_commit': True,
        'remaining_proven_implementation_blockers': [],
        'independent_cpu_evidence': evidence,
        'author_or_root_additional_cpu_outputs_not_independent_reexecution': [
            dict(pinned(TASK / 'root-multitraj-source-cpu.stdout.original'), result='67 passed; root execution'),
            dict(pinned(TASK / 'formal-resource-v3-cpu.stdout.original'), result='38 passed; root execution')],
        'repaired_counterexamples': [
            {'problem': 'Actual production archive state has six keys; prior first5 native reader rejected its scope field', 'repair_evidence': repair},
            {'problem': 'Prior parser rejected old actual owned FSDP689 transport warning and could fail terminal wrapper',
             'original_counterexample': pinned(OUT / 'NEW_FORMAL_OLD_ACTUAL_FSDP_FRAME_COUNTEREXAMPLE.json'),
             'repair': 'Current exact gate accepts only separately pinned native FSDP689 source/path/literal and owned PID; actual complete old OUTER74/driver comparison and owned AMP+exact green Ray native tag regressions plus negative unknown frames passed in independent 328 replay'},
            {'problem': 'Resource query/read failure could drop unresolved initial NVML apps',
             'repair_evidence': pinned(OUT / 'RESOURCE_RACE_COMPONENT_INDEPENDENT_REVIEW.json')}],
        'verified_contract': {
            'pure_opsd': 'Full-vocabulary forward KL q_teacher||p_student, component_clip None, beta0, no PG/reference KL/entropy; original global valid-token denominator retained',
            'outer_budget': '512 questions x 5 grouped student responses =2560; mini256/global actor micro32/local4; 10 synchronized Adam calls/rank/outer; old LP global64/local8',
            'schedule': 'Original fixed57 outer warmup preserved: outer1 actual usedLR0, postouter1 LR1e-6/57; outer57 uses56/57e-6; outer58 first usesfull1e-6; usedLR and postLR separately logged',
            'numerical_memory_changes': 'Native response width+1 LM-head projection through normal model/FSDP forward and independent frozen teacher row2 segmentation; CPU full loss/all named-gradient equivalence verified with padding/tool masks and checkpointing',
            'teacher_diagnostics': '512 original unique questions/rank64 privileged short greedy max128 answers, independently paired with2560 students; training RNG/state restored, q/student/evaluation never consume diagnostic output; raw model/tool tensor history retained and strict official tokenizer/identity replay',
            'continuity': 'First5 saved HF/fullstate+100 archive members before fetch6; signed independent gate; sameprocess continuation to200, no201; checkpoints5/50/100/150/200; dev512+stage64/final51713 conditions remain fixed',
        },
        'not_proven_and_required_next': [
            'Actual native Torch2.4 GPU imports/official assets/commondata/source resolved profile and actual8 H100 path',
            'Actual eight-rank BF16/FSDP full-vs-response-only/teacher2 comparison: finite nonzero grads, relativeL2<=0.02, cosine>=0.9999, relativeLoss<=0.002, teacher qmeanL1<=1e-4; tolerances are not bitwise equality',
            'Actual worst requested actor local4/teacher2 and packed oldLP local8 capacity must pass without changing requested sizes. LP probe covers initial Adam state, production oldLP optimizer state is offloaded; vLLM coexistence and later first5 actual memory are separate evidence',
            'Actual fresh5: all8 rank parameters/finite loss gradients/50 Adamcalls/scheduler5/RNG/data cursor2560/HF and fullstate stored reload/official teacher/100 archive members plus GPU provenance; no CPU-only test substitutes',
            'Actual source commit and exact finalized immutable bundle review before deployment or signing',
            'Training/effect200/final51713/true job terminal and full resource release remain future actual evidence'
        ],
        'scope_limits': ['Local CPU Torch2.8/Transformers4.46.3, not native Torch2.4 BF16 GPU',
                         'This multi-factor fresh experiment cannot uniquely attribute effects to clip removal',
                         'Teacher diagnostic outputs are short privileged answers, not full independently searched teacher trajectories',
                         'Negative behaviour/effect must be reported separately from structural implementation acceptance',
                         'Original old188/job3261 negative evidence and real failed terminal remain unchanged'],
    }
    record = emit(OUT / 'SOURCE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.json', report)
    markdown = ('当前候选源码和验收工具的软件独审通过；没有剩余已证实的实现阻断。\n\n'
                '独立实际重放：328项配置/门禁/首5工具检查、21项轨迹检查、17项数学/教师状态检查、12项资源退出反例。各套件原始输出与文件SHA已绑定；测试数分别报告，不冒称彼此独立的总覆盖数。\n\n'
                '明确修复了两个真实反例：首5状态读者漏scope字段；旧真实FSDP689警告的日志帧无法通过终态解析。资源退出只能在严格ENOENT及确认NVML已退出、先前确为本作业进程时例外，未知或外国进程仍拒绝。\n\n'
                '纯OPSD完整词表KL、不裁剪、原global token归约、10次Adam/外层、原57轮scheduler语义和独立教师诊断均已按当前字节审查。响应logits与teacher按行拆分的CPU全梯度等价检查通过。\n\n'
                '此结论允许正常本地commit；真实commit与不可变工具仍需绑定复核。GPU比较与容量、实际首5保存/读回及独立放行尚未发生。LP8容量探针只覆盖初始Adam状态，不能声称已验证训练阶段vLLM共存。\n\n'
                f'完整记录：{record["path"]}\nSHA256：{record["sha256"]}\n')
    md = OUT / 'SOURCE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.md'
    with md.open('x') as f:
        f.write(markdown)
    print(json.dumps({'review': record, 'markdown': pinned(md), 'source_files': len(source_pins), 'candidate_tools': len(tool_pins)}))


if __name__ == '__main__':
    main()
