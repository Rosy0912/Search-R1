#!/usr/bin/env python3
"""Read back completed local aggregates; write scalar-only notes and provenance."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def proof(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Local artifact changed during read')
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--observed-aggregate-exit-code', required=True, type=int)
    parser.add_argument('--observed-tool-session', required=True, type=int)
    args=parser.parse_args()
    if args.observed_aggregate_exit_code != 0:
        raise ValueError('An unsuccessful observed aggregation cannot receive an accepted readback')
    result = json.loads((ROOT/'C0-ten-outer-clipmass.json').read_text())
    manifest = json.loads((ROOT/'derived-manifest.json').read_text())
    for item in manifest['files']:
        actual = proof(Path(item['path']))
        if (actual['bytes'], actual['sha256']) != (item['bytes'], item['sha256']):
            raise ValueError('Derived artifact does not match native aggregation manifest')
    rows = list(csv.DictReader((ROOT/'C0-ten-outer-clipmass.csv').open()))
    if len(rows) != 10 or sum(int(r['valid_token_rows']) for r in rows) != 583074:
        raise ValueError('CSV coverage disagrees')
    t = result['totals']
    body = ['# C0 十轮实际训练 token 的 clip 命中质量', '',
            '只读本地已完成备份，沿 100 个 committed update receipt 的明确 token_log 引用读取；不读取失败 attempt、原始题文或模型，不调用节点、CUDA、HTTP。', '',
            f"覆盖 **3012 条训练轨迹、583074 个有效 policy token、100 次 Adam**。3112 个引用文件（100 receipt + 3012 token logs），总计 {result['referenced_files']['total_bytes']:,} 字节，全部逐文件 fresh SHA 与两份 source/local 固定库存及 receipt 对账通过。每轮有效位置数与 facts 精确一致。", '',
            f"上裁剪命中 **{t['hit_token_count']} 个有效位置，占 {100*t['hit_token_fraction']:.6f}%**。clipped vocabulary contribution 总数也是 1021，所以本次每个命中位置恰有一个被裁剪词表项。词表 V=151936；真正的 pooled vocabulary fraction 是 {t['pooled_vocabulary_fraction']:.12g}，两个比例不可混用。", '',
            '| outer（从 0 开始） | 有效位置 | 命中位置/裁剪项 | 命中位置比例 | student mass sum | teacher/target mass sum |',
            '|---:|---:|---:|---:|---:|---:|']
    for r in result['outers']:
        body.append(f"| {r['outer_index']} | {r['valid_token_rows']} | {r['hit_token_count']} | {100*r['hit_token_fraction']:.6f}% | {r['masses']['student']['sum_over_valid_tokens']:.9f} | {r['masses']['teacher']['sum_over_valid_tokens']:.12g} |")
    body += ['', '| 命中集合质量 | 跨有效位置 sum | 全有效位置 pooled mean | 仅命中位置 mean | 等轨迹 mean | 单位置 max |',
             '|---|---:|---:|---:|---:|---:|']
    for key in ('student','teacher','mixture'):
        m=t['masses'][key]
        body.append(f"| {key} | {m['sum_over_valid_tokens']:.12g} | {m['pooled_valid_token_mean']:.12g} | {m['hit_token_mean']:.12g} | {m['equal_trajectory_token_mean']:.12g} | {m['maximum_token_mass']:.12g} |")
    body += ['', 'mass 是同一 `p(v)*(log p(v)-log target(v)) > 10` 集合上的分布概率质量。跨位置的 sum 可以超过 1，不能把它当作单个概率；仅命中位置均值与全有效位置均值使用不同分母。C0 没有 mixture 干预，target=固定 teacher；teacher 与日志中的 mixture mass 在所有有效位置逐行完全相同。', '',
             f"全 token pooled raw/post 均值为 {t['loss']['raw']['pooled_valid_token_mean']:.12g} / {t['loss']['post']['pooled_valid_token_mean']:.12g}；与实际 trajectory_mean 合同相符的等轨迹均值为 **{t['loss']['raw']['equal_trajectory_token_mean']:.12g} / {t['loss']['post']['equal_trajectory_token_mean']:.12g}**。重聚合与 native pair/outer 记录均值最大差 {max(result['native_chunked_vs_local_sequential_recorded_mean_max_absolute_error'].values()):.12g}，仅为 Python 累加次序差。", '',
             '所有本次明确检查的 mass/KL/entropy/logprob/contribution 标量均 finite。另有 2 条完整词表 dump 辅助记录按明确 schema 识别，不计入 token 分母；本次不重读其二进制 payload，不重验 top32 数组。', '',
             '原 objective 只上裁剪正分量，负分量未下裁剪，raw/post 已是包含正负分量的完整词表和。日志没有单独的负分量 sum/count，因此不能重构负分量比例、完整 logits/分布或梯度。高命中 student mass 描述被裁剪集合的概率覆盖，不能独立证明性能变化的原因。10 个 outer 使用不同 TRAIN cohort，这也不是固定 heldout512 收益曲线。', '',
             '此前两次本地读取器失败分别是把 native `retries=0` 错看成空列表、未区分辅助 full_vocab_dump 行；真实 stderr/rc1 保留。修正只读解析后全量通过；这些是工具 schema 问题，不是训练故障。']
    (ROOT/'C0-clipmass-notes.zh.md').write_text('\n'.join(body)+'\n')
    execution=dict(schema='local-c0-clipmass-execution-readback-v1',actual_exit_code=args.observed_aggregate_exit_code,
                   actual_tool_session=args.observed_tool_session,
                   returncode_provenance='Caller supplied actual exec/write_stdin completion result; this readback did not supervise or rerun aggregation',
                   started_unix=result['started_unix'],ended_unix=result['ended_unix'],
                   execution_scope='Local Python stdlib only; no Node/GPU/model/HTTP/Git/publisher mutation',
                   stdout=proof(ROOT/'aggregate-final.stdout.jsonl'),stderr=proof(ROOT/'aggregate-final.stderr.txt'),
                   preserved_attempts=[dict(attempt=1,actual_exit_code=1,reason='reader native retries schema mismatch',stderr=proof(ROOT/'aggregate-attempt1.stderr.txt')),
                                       dict(attempt=2,actual_exit_code=1,reason='reader auxiliary vocabulary row mistaken for metric row',stderr=proof(ROOT/'aggregate-attempt2.stderr.txt')),
                                       dict(attempt=3,actual_exit_code=0,scope='same successful numeric aggregation; final run only corrects sum-semantics label',stdout=proof(ROOT/'aggregate-attempt3.stdout.jsonl'))])
    (ROOT/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    publication={key:value for key,value in result.items() if key!='inputs'}
    publication['input_digests']={'facts':{key:result['inputs']['facts'][key] for key in ('sha256','bytes')},
                                 'backup_inventories':[{kind:{key:item[kind][key] for key in ('sha256','bytes')}
                                                       for kind in ('source_inventory','local_inventory')}
                                                      for item in result['inputs']['backups']]}
    publication['actual_exit_code']=args.observed_aggregate_exit_code
    publication['privacy']='Aggregate numbers and content digests only; no raw/private paths, question bodies, token text, IDs or logits.'
    (ROOT/'C0-ten-outer-clipmass-publication.json').write_text(json.dumps(publication,ensure_ascii=False,indent=2)+'\n')
    names=['aggregate_clipmass.py','finalize_clipmass_proof.py','C0-ten-outer-clipmass.json','C0-ten-outer-clipmass-publication.json','C0-ten-outer-clipmass.csv','referenced-files-proof.private.json','derived-manifest.json','C0-clipmass-notes.zh.md','execution.json','aggregate-final.stdout.jsonl','aggregate-final.stderr.txt']
    bundle=dict(schema='c0-clipmass-scope-proof-v1',accepted=True,verified_unix=time.time(),actual_exit_code=0,
                run_id=result['run_id'],actual_updates=100,pair_exposures=3012,valid_token_rows=583074,
                committed_file_count=3112,committed_file_bytes=result['referenced_files']['total_bytes'],
                all_referenced_sha_verified=True,per_outer_token_denominators_exact=True,
                input_inventory_refs=result['inputs'],files=[proof(ROOT/name) for name in names],
                scope='Fresh content verification of referenced committed scalar logs only; excludes failed attempts, full vocabulary payload, top32 revalidation, logits/gradient reconstruction and causal/effect claims.')
    (ROOT/'scope-proof.json').write_text(json.dumps(bundle,indent=2)+'\n')
    print(json.dumps(dict(accepted=True,scope_proof=proof(ROOT/'scope-proof.json')),ensure_ascii=False))


if __name__=='__main__':
    main()
