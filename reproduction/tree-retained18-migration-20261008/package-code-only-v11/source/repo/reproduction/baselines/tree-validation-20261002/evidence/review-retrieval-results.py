# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,math
chat=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2');source=chat/'outputs/evidence'
paths=[source/f for f in ['retrieval-cpu-2854.json','retrieval-gpu-encoder-cpu-index.json','retrieval-gpu.json']]
a,c,b=[json.loads(p.read_text()) for p in paths]
assert a['recipe']==b['recipe']==c['recipe']
assert a['asset_identity']==b['asset_identity']==c['asset_identity']
assert all(d['asset_identity_verified'] for d in [a,b,c])
assert all(len(d['resources']['measurements'])==2 and all(v['query_count']==14 for v in d['resources']['measurements']) for d in [a,b,c])
assert a['queries']==b['queries']==c['queries'] and len(a['queries'])==14
assert all(d['queries_sha256']==hashlib.sha256(json.dumps(d['queries'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest() for d in [a,b,c])
assert all(d['source']['slurm_job_id']=='2854' and d['source']['code_commit']==a['source']['code_commit'] and d['source']['retrieval_server_sha256']==a['source']['retrieval_server_sha256'] for d in [a,b,c])
assert b['embeddings_sha256']==c['embeddings_sha256'] and a['embeddings_sha256']!=b['embeddings_sha256']
rows=[]
for i,q in enumerate(a['queries']):
 row={'query_index':i+1,'query':q,'top3_ids':a['results'][i]['3']['ids'],'modes':{}}
 for mode,d in [('gpu_encoder_cpu_index',c),('gpu',b)]:
  stats={}
  for k in (3,50):
   left=a['results'][i][str(k)];right=d['results'][i][str(k)]
   assert len(set(left['ids']))==len(set(right['ids']))==k and set(left['ids'])==set(right['ids'])
   lm=dict(zip(left['ids'],zip(left['scores'],left['contents_sha256'])));rm=dict(zip(right['ids'],zip(right['scores'],right['contents_sha256'])))
   assert all(math.isfinite(v[0]) for v in [*lm.values(),*rm.values()])
   assert all(lm[v][1]==rm[v][1] for v in lm)
   moved=[j+1 for j,(x,y) in enumerate(zip(left['ids'],right['ids'])) if x!=y]
   stats[str(k)]={'id_set_equal':True,'content_hash_equal':True,'rank_order_equal':not moved,'changed_rank_positions':moved,'max_matched_id_score_abs_delta':max(abs(lm[v][0]-rm[v][0]) for v in lm)}
  assert stats['3']['rank_order_equal']
  row['modes'][mode]=stats
 row['gpu_cpu_index_vs_gpu_index_order_equal']=c['results'][i]['50']['ids']==b['results'][i]['50']['ids']
 rows.append(row)
comparison_paths=[]
for mode,name in [('gpu_encoder_cpu_index','retrieval-cpu-vs-gpu-encoder-cpu-index.json'),('gpu','retrieval-cpu-vs-gpu.json')]:
 cp=source/name;comparison_paths.append(cp);summary=json.loads(cp.read_text())['summary']
 for k in ('3','50'):
  assert summary[k]['exact_rank_queries']==sum(r['modes'][mode][k]['rank_order_equal'] for r in rows)
  assert summary[k]['queries']==14 and summary[k]['mean_overlap_fraction']==1.0
  assert summary[k]['max_matched_id_score_abs_delta']==max(r['modes'][mode][k]['max_matched_id_score_abs_delta'] for r in rows)
paths+=comparison_paths
report={'decision':'SUFFICIENT_FOR_LIMITED_TREE_TOP3_SANITY_TO_PROCEED_TO_TWO_UPDATE_SMOKE_REVIEW','not_acceptance_record':True,'training_started_by_reviewer':False,'full_query_equivalence':False,'cosearch_top50_order_parity':False,'ablation_pending':True,'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'recipe':a['recipe'],'source_code_sha':a['source']['code_commit'],'retrieval_server_sha256':a['source']['retrieval_server_sha256'],'asset_identity_equal_and_verified':True,'rows':rows}
out=chat/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/evidence/retrieval-independent-result-review.json';out.write_text(json.dumps(report,indent=2))
lines=['# Tree-GRPO 检索结果独立复核：Slurm 2854','', '结论：**足以作为Tree top3 CPU部署的有限sanity证据，支持进入两次实际更新smoke的下一阶段审查。** 本文不是retrieval-acceptance记录，不启动训练，不认定全部query等价或CoSearch top50排序等价。CPU embedding/GPU index ablation尚待完整结果，可补充误差归因，不是已取得证据。','', '审查者重新计算三个原始measurement JSON的ID集合、rank顺序、内容hash、matched-ID score delta和来源身份，未仅照抄comparison summary。14个query每模式两次真实测量通过repeat稳定性检查；所有top3文档ID/顺序/内容hash完全相同，top3最大score差3.975033760070801e-4。top50集合与对应内容hash全部相同，9/14 query顺序不同，最大matched-ID score差4.2307376861572266e-4。','', '| # | 固定query | top3 IDs（同顺序） | top3最大Δ CPU→GPU | top50变动位置 GPU encoder/CPU index | top50变动位置 GPU index |', '|---|---|---|---:|---|---|']
for row in rows:
 ge=row['modes']['gpu_encoder_cpu_index'];gpu=row['modes']['gpu']
 positions=lambda v:','.join(map(str,v)) or '无'
 lines.append(f"| {row['query_index']} | {row['query']} | {','.join(map(str,row['top3_ids']))} | {gpu['3']['max_matched_id_score_abs_delta']:.8g} | {positions(ge['50']['changed_rank_positions'])} | {positions(gpu['50']['changed_rank_positions'])} |")
lines+=['', '每一query top50集合均50/50相同，共享文档内容hash全相同；顺序完全相同的是4、6、7、9、12，其余9条均需披露重排。Apollo query（8）两种GPU路径的具体top50顺序并不完全一样；两路径虽然均呈9/14变化，不能说所有变动位置一致。完整逐query两模式最大score delta及输入文件SHA见独立JSON。','', '三个配方均为E5 FP16、mean pooling、query cap256、top3/top50、batch-size上限512、同21015324文档的768维index和相同corpus。CPU：生产FAISS1.8.0 IndexFlatIP FP32 index、CUDA不可见；GPU encoder/CPU index：GPU FP16 E5，overlay FAISS1.8.0但CPU IndexFlatIP；GPU：相同GPU embeddings，overlay FAISS八卡sharded GPU index/useFloat16。FAISS CPU/GPU不同构建及backend仍需保留解释。GPU两个模式embedding hash相同，CPU hash不同；故encoder数值差异已存在，9/14重排也在GPU encoder/CPU index中出现。GPU backend对Apollo还改变细部排序，不能把全部变化只归因encoder或只归因FP16 quantization。','', 'source均a545a0d2349be9f4d591576f56f62b7d35d1bed8，retrieval_server hash b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052，Slurm2854。实际FAISS导入路径分别是Tree ownroot/env/.../faiss及Tree ownroot/retriever-gpu-lib/faiss；GPU UUID lease集合对应八卡，CPU child无GPU/无overlay。三个报告的index/corpus/model/manifest内容身份相同且verified；index SHA69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166、corpus SHA85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9。模型identity还包含未用于加载的partial/aria2文件，这些属于快照provenance，不能视作E5输入权重；实际model.safetensors SHA固定相同。','', 'Tree部署topk3，当前Search-R1 observation格式化读取文档contents、不读取retrieval score，所以本组实测top3得分微差没有改变actor看到的三个文档及其顺序。top50重排未改变此Tree top3输出，但它明确阻止“top50顺序等价”结论；本判断不能外推CoSearch ranker。','', 'CPU测量14query而非512query，只记录batch-size上限。CPU两轮search36.746/22.541秒同时含top3和top50请求，load73.952秒、自身峰值RAM67.48GB；不能把它当生产512吞吐。两更新smoke仍应检查真实CPU检索延迟/RAM、search请求和超时、六叶来源、finite loss/gradient、非零参数更新、actual update计数、GPU峰值、保存与重载。若出现真实query top3差异/资源问题，停止并记录，不能由这14条sanity保证继续。','', '结果范围：少量手工固定query、两个repeat、一个固定资产/版本/节点配置；不是统计保证、七集效果结果或训练成功。只建议有限smoke阶段继续，不自动写接受或派发文件。']
(chat/'outputs/TREE_RETRIEVAL_REVIEW_2854.md').write_text('\n'.join(lines)+'\n')
print('independent retrieval review written')
