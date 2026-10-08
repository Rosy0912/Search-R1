# Tree-GRPO 检索结果独立复核：Slurm 2854

结论：**足以作为Tree top3 CPU部署的有限sanity证据，支持进入两次实际更新smoke的下一阶段审查。** 本文不是retrieval-acceptance记录，不启动训练，不认定全部query等价或CoSearch top50排序等价。CPU embedding/GPU index ablation已取得并独立复核，见末尾补充。

审查者重新计算三个原始measurement JSON的ID集合、rank顺序、内容hash、matched-ID score delta和来源身份，未仅照抄comparison summary。14个query每模式两次真实测量通过repeat稳定性检查；所有top3文档ID/顺序/内容hash完全相同，top3最大score差3.975033760070801e-4。top50集合与对应内容hash全部相同，9/14 query顺序不同，最大matched-ID score差4.2307376861572266e-4。

| # | 固定query | top3 IDs（同顺序） | top3最大Δ CPU→GPU | top50变动位置 GPU encoder/CPU index | top50变动位置 GPU index |
|---|---|---|---:|---|---|
| 1 | What is the capital of France? | 1039175,9447685,2543511 | 6.1869621e-05 | 27,28,49,50 | 27,28,49,50 |
| 2 | Who wrote Pride and Prejudice? | 4950710,592735,13328669 | 0.0001449585 | 33,34 | 33,34 |
| 3 | What is the largest planet in the Solar System? | 11549858,18934865,13379999 | 4.2498112e-05 | 38,39,42,43,46,47 | 38,39,42,43,46,47 |
| 4 | Where is Mount Everest located? | 5733239,11694402,9996982 | 5.5670738e-05 | 无 | 无 |
| 5 | Who painted the Mona Lisa? | 12232081,2203859,2203852 | 0.00012969971 | 46,47 | 46,47 |
| 6 | What is the chemical symbol for gold? | 8537275,7548030,5486094 | 7.3432922e-05 | 无 | 无 |
| 7 | Which country contains the ancient city of Petra? | 17785757,10726707,10726706 | 4.3988228e-05 | 无 | 无 |
| 8 | When did the Apollo 11 mission land on the Moon? | 6517991,12324085,3893448 | 0.00031834841 | 37,38,40,41 | 11,12,37,38,40,41 |
| 9 | What language is primarily spoken in Brazil? | 14300822,10255973,19367753 | 0.00025373697 | 无 | 无 |
| 10 | Who developed the theory of general relativity? | 9170063,7721330,12036758 | 6.8664551e-05 | 44,45 | 44,45 |
| 11 | Which river flows through Cairo? | 860631,19151556,19124882 | 7.9154968e-05 | 34,35 | 34,35 |
| 12 | What is the tallest building in the world? | 16864745,7009180,7009195 | 0.00039750338 | 无 | 无 |
| 13 | Who was the first person to win two Nobel Prizes? | 3417551,11376096,2026541 | 0.00011366606 | 45,46,47,48 | 45,46,47,48 |
| 14 | Which ocean lies between Africa and Australia? | 20339771,285785,2272179 | 0.00017881393 | 15,16 | 15,16 |

每一query top50集合均50/50相同，共享文档内容hash全相同；顺序完全相同的是4、6、7、9、12，其余9条均需披露重排。Apollo query（8）两种GPU路径的具体top50顺序并不完全一样；两路径虽然均呈9/14变化，不能说所有变动位置一致。完整逐query两模式最大score delta及输入文件SHA见独立JSON。

三个配方均为E5 FP16、mean pooling、query cap256、top3/top50、batch-size上限512、同21015324文档的768维index和相同corpus。CPU：生产FAISS1.8.0 IndexFlatIP FP32 index、CUDA不可见；GPU encoder/CPU index：GPU FP16 E5，overlay FAISS1.8.0但CPU IndexFlatIP；GPU：相同GPU embeddings，overlay FAISS八卡sharded GPU index/useFloat16。FAISS CPU/GPU不同构建及backend仍需保留解释。GPU两个模式embedding hash相同，CPU hash不同；故encoder数值差异已存在，9/14重排也在GPU encoder/CPU index中出现。GPU backend对Apollo还改变细部排序，不能把全部变化只归因encoder或只归因FP16 quantization。

source均a545a0d2349be9f4d591576f56f62b7d35d1bed8，retrieval_server hash b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052，Slurm2854。实际FAISS导入路径分别是Tree ownroot/env/.../faiss及Tree ownroot/retriever-gpu-lib/faiss；GPU UUID lease集合对应八卡，CPU child无GPU/无overlay。三个报告的index/corpus/model/manifest内容身份相同且verified；index SHA69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166、corpus SHA85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9。模型identity还包含未用于加载的partial/aria2文件，这些属于快照provenance，不能视作E5输入权重；实际model.safetensors SHA固定相同。

Tree部署topk3，当前Search-R1 observation格式化读取文档contents、不读取retrieval score，所以本组实测top3得分微差没有改变actor看到的三个文档及其顺序。top50重排未改变此Tree top3输出，但它明确阻止“top50顺序等价”结论；本判断不能外推CoSearch ranker。

CPU测量14query而非512query，只记录batch-size上限。CPU两轮search36.746/22.541秒同时含top3和top50请求，load73.952秒、自身峰值RAM67.48GB；不能把它当生产512吞吐。两更新smoke仍应检查真实CPU检索延迟/RAM、search请求和超时、六叶来源、finite loss/gradient、非零参数更新、actual update计数、GPU峰值、保存与重载。若出现真实query top3差异/资源问题，停止并记录，不能由这14条sanity保证继续。

结果范围：少量手工固定query、两个repeat、一个固定资产/版本/节点配置；不是统计保证、七集效果结果或训练成功。只建议有限smoke阶段继续，不自动写接受或派发文件。

## 已完成的索引对照补充

同Slurm2854、相同资产和配方，真实GPU index明确复用CPU embeddings（hash与CPU完全相同）。独立重算结果：top3共14/14 ID/顺序/内容一致，最大得分差1.1920928955078125e-7；top50共13/14排序相同，集合/内容全部一致，最大得分差1.7881393432617188e-7。唯一变化是Apollo查询排名11/12互换：文档11729878、11729861在CPU同分0.8659123182296753，在GPU同分0.8659123778343201，是并列分数的排列变化。其他13查询top50顺序不变。

这补足了索引路径的有限误差归因：本14查询中，主要9条重排已出现在GPU encoder/CPU index路径，固定CPU embeddings后GPU index仅出现上述一处同分互换。只能描述本组实测，不能宣称所有index量化误差均如此。有限Tree top3 smoke阶段判断保持；尚无真实生产训练/全query等价证据。

独立日志复核：console唯一改动为数值格式.3f→.17g。执行原函数，用Python float、numpy float32/float64/int64、1e-12梯度、负小数及整数500回读，值与float(value)完全相等且输入字典不变。1e-12显示为9.9999999999999998e-13，不再显示0。此格式调整未修改metrics、梯度、loss或W&B backend；整数500在本预算范围可准确保留。offline W&B旧datastore读取接口不可用不等于离线日志失败，本次不声称已解析真实训练日志。
