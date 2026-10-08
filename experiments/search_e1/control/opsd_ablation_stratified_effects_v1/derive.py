"""Local SHA-bound native512 summaries; no inference, rescoring or raw trace read."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import time

LABELS = ('bf16_zero', 'fp16_zero', 'C0', 'C1', 'M', 'R', 'W', 'P', 'J', 'P_fp16', 'J_fp16')
STRATA = ('overall', 'nq', 'hotpotqa')
PAIRS = tuple([('bf16_zero', v) for v in ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')]
    + [('fp16_zero', 'P_fp16'), ('fp16_zero', 'J_fp16')]
    + [('C0', 'C1'), ('C1', 'M'), ('C1', 'R'), ('R', 'W'), ('C1', 'P'),
       ('C1', 'J'), ('P', 'P_fp16'), ('J', 'J_fp16'), ('bf16_zero', 'fp16_zero')])
PURE_SHA = '609e2e3379d2a26d3af0f9ddad522dabf97df0bbbe63cf7a23d0937d52f48877'
SHA = re.compile(r'[0-9a-f]{64}\Z')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def normalize(value):
    # Native counts has integer histogram keys; accepted JSON has string keys.
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def version_for(label):
    return label if label in ('bf16_zero', 'fp16_zero') else 'raw10_fp16' if label.endswith('_fp16') else 'raw10_bf16'


class Reader:
    """One fixed-handle SHA read per path/SHA; decode those same bytes."""
    def __init__(self):
        self.cache = {}

    def read(self, ref):
        require(isinstance(ref, dict) and isinstance(ref.get('path'), str)
            and isinstance(ref.get('sha256'), str) and SHA.fullmatch(ref['sha256']), 'Explicit path/SHA required')
        path = Path(ref['path'])
        require(path.is_absolute() and '..' not in path.parts, 'Canonical absolute local input required')
        key = (str(path), ref['sha256'])
        if key not in self.cache:
            current = Path(path.anchor)
            for part in path.parts[1:]:
                current /= part
                require(not current.is_symlink(), 'Symlink input refused')
            def stamp(s):
                return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
            before = stamp(path.stat())
            require(stat.S_ISREG(before[-1]), 'Regular input required')
            with path.open('rb') as handle:
                require(stamp(os.fstat(handle.fileno())) == before, 'Input changed before read')
                raw = handle.read()
                require(stamp(os.fstat(handle.fileno())) == before, 'Input changed during read')
            require(stamp(path.stat()) == before, 'Input path changed during read')
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == ref['sha256'], 'Input SHA mismatch')
            self.cache[key] = (raw, dict(sha256=digest, bytes=len(raw)))
        raw, proof = self.cache[key]
        require('bytes' not in ref or ref['bytes'] == len(raw), 'Declared input byte count differs')
        return raw, proof

    def json(self, ref):
        raw, proof = self.read(ref)
        return json.loads(raw), proof


def pure_functions(reader, ref):
    require(ref['sha256'] == PURE_SHA, 'Different native pure-function source revision')
    raw, proof = reader.read(ref)
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('counts', 'bootstrap')]
    require(len(nodes) == 2 and {n.name for n in nodes} == {'counts', 'bootstrap'}
        and not any(n.decorator_list for n in nodes), 'Two undecorated native pure functions required')
    namespace = dict(require=require, Counter=Counter, random=random)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<SHA-bound native counts/bootstrap AST>', 'exec'), namespace)
    return namespace['counts'], namespace['bootstrap'], proof


def validate_source(summary, raw, record_proof, counts):
    require(summary.get('schema') == 'opsd-ablation-native512-summary-v1' and summary.get('accepted') is True,
        'Native accepted summary required')
    native_ref = summary['private_primary_records']
    require(native_ref['sha256'] == record_proof['sha256'] and native_ref['bytes'] == record_proof['bytes'],
        'Local records do not match native private_primary_records SHA/bytes')
    require(raw.endswith(b'\n'), 'Partial JSONL record refused')
    items = [json.loads(line) for line in raw.splitlines()]
    versions = set(summary['models'])
    expected_versions = {'bf16_zero', 'fp16_zero'} if summary['evaluation_role'] == 'common_zero' else (
        {'raw10_bf16', 'raw10_fp16'} if summary['variant'] in ('P', 'J') else {'raw10_bf16'})
    require(summary['evaluation_role'] in ('group', 'common_zero') and versions == expected_versions
        and (summary['evaluation_role'] != 'common_zero' or summary['variant'] == 'C0')
        and summary['variant'] in ('C0', 'C1', 'M', 'R', 'W', 'P', 'J'), 'Native role/variant/version differs')
    require(len(items) == summary['primary_slots'] == 512 * len(versions), 'Native whole-source denominator differs')
    index = {}
    for row in items:
        i, version = row['validation_index'], row['version']
        require(type(i) is int and 0 <= i < 512 and version in versions and (version, i) not in index,
            'Unexpected or duplicate native model/question slot')
        require(row['primary'] is True and row['status'] == 'complete' and row['run_id'] == summary['run_id']
            and type(row['rank']) is int and row['rank'] == i % 8 and type(row['slot']) is int and row['slot'] == i // 8,
            'Native primary/run/rank/slot identity differs')
        require(row['dataset'] in ('nq', 'hotpotqa') and isinstance(row['question_id'], str) and row['question_id']
            and isinstance(row['row_semantic_sha256'], str) and SHA.fullmatch(row['row_semantic_sha256']),
            'Native question/dataset identity absent')
        em, valid = row['protocol_em'], row['protocol_valid_final_action']
        require(type(em) is int and em in (0, 1) and type(valid) is bool, 'Strict scoring types differ')
        require(row['failure_class'] == (None if em else 'final_invalid' if not valid else 'valid_gold_mismatch')
            and (not em or valid), 'Failure classes are not mutually exclusive')
        require(type(row['search_calls']) is int and 0 <= row['search_calls'] <= 4
            and isinstance(row['termination'], str) and row['termination'], 'Native search/termination types differ')
        if row['slot'] == 0:
            require(row['hook_control']['accepted'] is True and row['hook_control']['excluded_from_primary_denominator'] is True,
                'Native extra hook control absent')
            if version == 'bf16_zero':
                require(row['zero_effect_control']['accepted'] is True, 'Native zero-effect extra control absent')
        index[version, i] = row
    result = {}
    for version in sorted(versions):
        selected = [index[version, i] for i in range(512)]
        require(len({r['question_id'] for r in selected}) == 512, 'Duplicate native question IDs')
        datasets = {d: [r for r in selected if r['dataset'] == d] for d in ('nq', 'hotpotqa')}
        require(all(len(rs) == 256 for rs in datasets.values()), 'Native NQ/Hotpot denominator must be256 each')
        reproduced = dict(overall=normalize(counts(selected)), datasets={d: normalize(counts(rs)) for d, rs in datasets.items()})
        require(reproduced == summary['models'][version], 'Recorded native overall/dataset counts differ')
        result[version] = selected
    return result


def paired(left, right, bootstrap):
    n = len(left)
    require(n in (256, 512) and len(right) == n and all(
        (a['validation_index'], a['question_id'], a['dataset'], a['row_semantic_sha256']) ==
        (b['validation_index'], b['question_id'], b['dataset'], b['row_semantic_sha256']) for a, b in zip(left, right)),
        'Paired cohort identity/order differs')
    cells = Counter((a['protocol_em'], b['protocol_em']) for a, b in zip(left, right))
    lost = [b for a, b in zip(left, right) if a['protocol_em'] == 1 and b['protocol_em'] == 0]
    gained = [b for a, b in zip(left, right) if a['protocol_em'] == 0 and b['protocol_em'] == 1]
    return dict(questions=n, left_hits=sum(r['protocol_em'] for r in left), right_hits=sum(r['protocol_em'] for r in right),
        new_hits=cells[0, 1], lost_hits=cells[1, 0], preserved_hits=cells[1, 1], still_wrong=cells[0, 0],
        net_hits=cells[0, 1] - cells[1, 0], net_percentage_points=100 * (cells[0, 1] - cells[1, 0]) / n,
        lost_final_invalid=sum(r['failure_class'] == 'final_invalid' for r in lost),
        lost_valid_gold_mismatch=sum(r['failure_class'] == 'valid_gold_mismatch' for r in lost),
        gained_no_search=sum(r['search_calls'] == 0 for r in gained), lost_no_search=sum(r['search_calls'] == 0 for r in lost),
        bootstrap=bootstrap([b['protocol_em'] - a['protocol_em'] for a, b in zip(left, right)], replicates=10000, seed=42))


def derive(manifest, reader, counts, bootstrap):
    require(manifest.get('schema') == 'opsd-ablation-stratified-native-inputs-v1' and isinstance(manifest.get('models'), dict)
        and manifest['models'] and set(manifest['models']) <= set(LABELS), 'Expected explicit eleven-label subset manifest')
    models, records, inputs, source_cache = {}, {}, {}, {}
    common_identity = common_study = common_cohort = None
    for label in LABELS:
        if label not in manifest['models']:
            continue
        item = manifest['models'][label]
        require(item['version'] == version_for(label), 'Label/version mapping differs')
        native, sp = reader.json(item['summary'])
        raw, rp = reader.read(item['records'])
        key = (sp['sha256'], rp['sha256'])
        if key not in source_cache:
            source_cache[key] = validate_source(native, raw, rp, counts)
        selected = source_cache[key][item['version']]
        require(native['evaluation_role'] == ('common_zero' if label in ('bf16_zero', 'fp16_zero') else 'group')
            and native['variant'] == ('C0' if label in ('bf16_zero', 'fp16_zero') else label.split('_')[0]),
            'Label/native role or variant differs')
        identity, study = native['identities'], native['study_id']
        require(set(identity) == {'source_model_identity', 'runtime_identity', 'common_initial_adapter_tensor_sha256',
            'validation_sha256', 'greedy_contract'} and all(isinstance(identity[k], str) and SHA.fullmatch(identity[k])
            for k in identity if k != 'greedy_contract') and isinstance(study, str) and study,
            'Shared native identity schema differs')
        cohort = [(r['validation_index'], r['question_id'], r['dataset'], r['row_semantic_sha256']) for r in selected]
        if common_identity is None:
            common_identity, common_study, common_cohort = identity, study, cohort
        require(identity == common_identity and study == common_study and cohort == common_cohort,
            'Different source/runtime/data/initial/greedy/study or paired cohort')
        models[label] = dict(status='COMPLETE', version=item['version'], run_id=native['run_id'],
            overall=normalize(counts(selected)), datasets={d: normalize(counts([r for r in selected if r['dataset'] == d]))
                for d in ('nq', 'hotpotqa')})
        records[label] = selected
        inputs[label] = dict(summary=sp, records=rp, version=item['version'], native_counts_reproduced=True,
            native_private_primary_records_SHA_match=True)
    comparisons, csvrows = {}, []
    for a, b in PAIRS:
        name = a + '->' + b
        missing = [v for v in (a, b) if v not in records]
        comparisons[name] = dict(status='PENDING' if missing else 'COMPLETE', missing_labels=missing, strata={})
        for stratum in STRATA:
            row = dict(comparison=name, left=a, right=b, stratum=stratum,
                status='PENDING' if missing else 'COMPLETE', missing_labels=','.join(missing))
            if not missing:
                left = records[a] if stratum == 'overall' else [r for r in records[a] if r['dataset'] == stratum]
                right = records[b] if stratum == 'overall' else [r for r in records[b] if r['dataset'] == stratum]
                stats = paired(left, right, bootstrap)
                comparisons[name]['strata'][stratum] = stats
                row.update({k: v for k, v in stats.items() if k != 'bootstrap'})
                ci = stats['bootstrap']
                row.update(bootstrap_lower_pp=100 * ci['lower'], bootstrap_upper_pp=100 * ci['upper'],
                    confidence=ci['confidence'], bootstrap_replicates=ci['replicates'], bootstrap_seed=ci['seed'])
            csvrows.append(row)
    return dict(schema='opsd-ablation-stratified-recorded-summary-v1', accepted=True, study_id=common_study,
        coverage=dict(complete_models=len(models), expected_models=11, complete_primary_slots=512 * len(models),
            expected_primary_slots=5632, pending_models=[v for v in LABELS if v not in models],
            complete_comparisons=sum(v['status'] == 'COMPLETE' for v in comparisons.values()), expected_comparisons=18,
            expected_comparison_strata=54, full5632_model_records_complete=len(models) == 11),
        identities=common_identity, models=models, comparisons=comparisons, inputs=inputs,
        statistics=dict(question_units={'overall': 512, 'nq': 256, 'hotpotqa': 256}, bootstrap_replicates=10000,
            seed=42, confidence=.95, multiplicity_adjusted=False,
            interval_scope='Descriptive marginal percentile intervals; no multiple-comparison correction or familywise claim.'),
        scope='SHA-bound recorded native strict scores, no re-score/trace/model/tensor read. No causal or goal-completion acceptance.'), csvrows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'manifest-sha256', 'pure-source', 'pure-source-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    reader = Reader()
    manifest, mp = reader.json(dict(path=a.manifest, sha256=a.manifest_sha256))
    counts, bootstrap, pure_proof = pure_functions(reader, dict(path=a.pure_source, sha256=a.pure_source_sha256))
    summary, csvrows = derive(manifest, reader, counts, bootstrap)
    summary['input_manifest'] = mp
    summary['pure_function_source'] = dict(**pure_proof, ast_only_functions=['counts', 'bootstrap'], no_module_import=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    summary_path = out / 'stratified-paired-summary.json'
    write_json(summary_path, summary)
    csv_path = out / 'stratified-paired.csv'
    fields = list(dict.fromkeys(k for r in csvrows for k in r))
    with csv_path.open('x', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader(); writer.writerows(csvrows)
    note = out / 'stratified-paired-notes.zh.md'
    coverage = summary['coverage']
    note.write_text('# 已记录 native512 的分层配对统计\n\n'
        f"已完成 {coverage['complete_models']}/11 模型、{coverage['complete_primary_slots']}/5632 正式记录；"
        f"{coverage['complete_comparisons']}/18 个预定对照可计算，其余明确 pending，不补零。\n\n"
        '整体每模型512题，NQ和Hotpot各256题；身份、唯一槽、native原摘要所有计数逐项复现，'
        '使用已记录严格最终行动EM，不重新评分或推理。两个zero可共用1024行文件，按version各筛512。'
        '生成控制及teacher诊断均不计正式分母。\n\n'
        '只AST提取原已绑定源码的counts/bootstrap两个纯函数；每个源文件一次SHA读取与同字节解析，'
        '不import整个summary、worker、Torch或原模型。\n\n'
        '18个对照分别输出overall/NQ/Hotpot配对保留、丢失、新增、仍错；'
        '丢失分final-invalid和合法答案不匹配，另列零搜索。'
        '按问题单位独立重采样，seed42/10000次/95% percentile区间；'
        '这些是未作多重比较校正的描述性边际区间，不能称总体家族显著性或算法因果。'
        'J是组合分支；实际K/Adam/曝光及在线采样差异须在最终报告另列。\n\n'
        'FP16部署必须比较FP16zero；BF16zero→FP16zero是固定初始模型部署精度对照，不是训练收益。'
        '统计公开产物只含聚合、内容digest和模型身份；没有问题正文、question ID、行ID或私有输入路径。'
        '全11模型记录齐全也不替代70轮、执行终态、数值、debug与持久资产验收。\n')
    def artifact(path):
        raw = path.read_bytes()
        return dict(name=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    code = Path(__file__).read_bytes()
    proof_path = out / 'derivation-manifest.json'
    write_json(proof_path, dict(schema='opsd-ablation-stratified-derivation-v1', accepted=True,
        source_code=dict(bytes=len(code), sha256=hashlib.sha256(code).hexdigest()), input_manifest=mp,
        pure_function_source=pure_proof, inputs=summary['inputs'], coverage=coverage,
        artifacts=[artifact(summary_path), artifact(csv_path), artifact(note)],
        no_remote_IO=True, no_new_scientific_test=True, no_rescoring=True, no_question_or_row_identifiers=True,
        observed_unix=time.time()))
    print(json.dumps(dict(accepted=True, coverage=coverage, manifest=artifact(proof_path))))


if __name__ == '__main__':
    main()
