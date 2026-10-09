"""Aggregate-only CSV, standard Agg plots and Chinese interpretation.

No statistical estimate is recomputed. Percentages are display conversions.
"""
import csv
import hashlib
import json
from pathlib import Path
try:
    from .common import VARIANTS, MAX_AGGREGATE, encode, require, finite
except ImportError:
    from common import VARIANTS, MAX_AGGREGATE, encode, require, finite

TRAIN_FEATURES = ('answer_gap','student_entropy','raw_teacher_kl','post_loss','weighted_post_loss',
                  'selected_weight','answer_student_logp','answer_teacher_logp')
HELDOUT_FEATURES = ('answer_gap','student_entropy','raw_teacher_reverse_kl','post_clip_teacher_loss',
                    'source100_ordinary_reverse_kl','answer_student_logp','answer_teacher_logp',
                    'policy_student_logp','policy_teacher_logp')
LABELS = {'answer_gap':'答案 student−teacher logprob 差','student_entropy':'全 policy 学生 entropy',
          'raw_teacher_kl':'原始 teacher reverse KL','post_loss':'clip10 后 loss',
          'weighted_post_loss':'实际加权 loss','selected_weight':'所选轨迹权重',
          'answer_student_logp':'答案 student 平均 logprob','answer_teacher_logp':'答案 teacher 平均 logprob',
          'raw_teacher_reverse_kl':'原始 teacher reverse KL','post_clip_teacher_loss':'clip10 后 loss',
          'source100_ordinary_reverse_kl':'student→source100 普通策略 KL',
          'policy_student_logp':'policy student 平均 logprob','policy_teacher_logp':'policy teacher 平均 logprob'}
COLORS = {'G_POS':'#165DAB','G_NEG':'#68A6D8','H_POS':'#A64B20','H_NEG':'#E69F71'}
PRIMARY = ('G_NEG/25->G_POS/25','H_NEG/25->H_POS/25')


def aggregate_bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()+b'\n'


def num(value, digits=3):
    return 'NA' if value is None else ('{:+.'+str(digits)+'f}').format(value)


def pct(value):
    return '{:.2f}%'.format(100*value)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x).replace('|','／') for x in row)+' |' for row in rows])


def small_aggregate(data):
    """Explicit whitelists: never copy raw qid/text/trace/weights or full reports."""
    analyses = data['analyses']; effects = analyses['effects']; held = analyses['heldout_relationship']; train = analyses['train_relationship']
    train_rel = {}; held_rel = {}; mechanisms = {}
    for v in VARIANTS:
        group = train['groups'][v]
        train_rel[v] = {f: {label:{k:group['selected_D_relationships'][f][label].get(k) for k in
            ('measured_questions','population_questions','missing_questions','pearson','spearman','raw_direction_auc','pearson_question_block')}
            for label in ('selected_reward','sampling_s_over_5')} for f in TRAIN_FEATURES}
        mechanisms[v] = {k:group.get(k) for k in ('population_questions','population_sampled_trajectories','sampling_strict_correct',
            'sampling_train_em','all5_wrong_noB','eligible_with_B','selected_D_encoding_skips','actual_trained_questions',
            'CC_CW_exposure','self_pair_exposure','CC_CW_weighted_exposure','selected_weight','actual_batches','clipped_batches','mean_preclip_norm','regions')}
    for name, model in held['models'].items():
        held_rel[name] = dict(bank_coverage=model['bank_coverage'], bank_subset=model['bank_subset'],
            features={f:{k:model['features'][f].get(k) for k in ('questions','metric_available','metric_unavailable','missing_reasons',
            'correct_available','wrong_available','subset_em','pearson','spearman','original_increasing_direction_EM_AUC',
            'association_NA_reason','adjusted_partial_pearson')} for f in HELDOUT_FEATURES}, regions=model['regions'])
    displacement = [{k:c[k] for k in ('variant','source','destination','source_updates','destination_updates','destination_pair_exposure')}
        | dict(metrics={part:c['metrics'][part] for part in ('aggregate','by_factor')}) for c in data['movement']['comparisons']]
    output = dict(schema='opsd-signal-final-report-small-aggregate-v1', study_id=data['study_id'],
        formal_report=not data['synthetic'], synthetic_fixture=data['synthetic'], curves=data['curves'], models=data['models'],
        paired=effects['paired'], primary_family_size=2, primary_time=25, bootstrap=effects['bootstrap'],
        train_relationships=train_rel, heldout_relationships=held_rel, training_mechanisms=mechanisms,
        parameter_movement=displacement,
        debug={k:data['debug'].get(k) for k in ('complete_outers','sampling_workers','teacher_parents','teacher_cases',
                  'teacher_not_applicable_parents','referenced_inventory')},
        scope=dict(raw_content_or_question_ids_exported=False, parameter_movement_not_effective_model_weight_delta=True,
                   region_contribution_not_gradient_or_causality=True, no_E25_no_multiseed_no_new_pass5=True))
    def forbid_private_keys(value):
        if isinstance(value,dict):
            require(not set(value)&{'question_id','question','answers','response','trace','token_rows','token_log',
                    'private_question_features','adapter_model','payload','weights'},'Raw content field in bounded aggregate export')
            for child in value.values():forbid_private_keys(child)
        elif isinstance(value,list):
            for child in value:forbid_private_keys(child)
    forbid_private_keys(output)
    require(len(aggregate_bytes(output)) <= MAX_AGGREGATE, 'Aggregate exceeds256KiB; never truncate scientific coverage silently')
    return output


def write_csv(path, fields, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='raise');writer.writeheader();writer.writerows(rows)


def csvs(aggregate, output):
    rows=[]
    for v in VARIANTS:
        for r in aggregate['curves'][v]:
            rows.append(dict(variant=v,outer=r['outer'],strict_correct=r['strict_correct'],denominator=2560,em=r['em'],
                true_adam_updates=r['optimizer_updates'],K=r['K'],unweighted_post_loss=r['unweighted_post_loss'],
                weighted_post_loss=r['weighted_post_loss'],raw_loss=r['raw_loss'],selected_weight_mean=r['weight']['mean']))
    write_csv(output/'train-preupdate-em.csv',tuple(rows[0]),rows)
    rows=[]
    for name, model in aggregate['models'].items():
        for dataset, row in dict(overall=model['overall'],**model['datasets']).items():
            fields={k:row[k] for k in ('hits','em','final_invalid','valid_gold_mismatch','no_executed_search')}
            rows.append(dict(endpoint=name,dataset=dataset,denominator=512 if dataset=='overall' else 256,**fields))
    write_csv(output/'heldout-em.csv',tuple(rows[0]),rows)
    rows=[]
    for key,p in aggregate['paired'].items():
        ci=p['ci95'];test=p.get('exact_mcnemar_holm',{})
        rows.append(dict(comparison=key,questions=p['questions'],gained=p['gained'],lost=p['lost'],preserved=p['preserved'],
            still_wrong=p['still_wrong'],delta_pp=p['delta_percentage_points'],ci95_low=ci['low'],ci95_high=ci['high'],
            primary=p['primary_comparison'],raw_exact_mcnemar_p=test.get('raw_p'),holm_p=test.get('holm_p')))
    write_csv(output/'paired-endpoint-effects.csv',tuple(rows[0]),rows)
    rows=[]
    for v in VARIANTS:
        for f,labels in aggregate['train_relationships'][v].items():
            for label,r in labels.items():
                ci=r['pearson_question_block'];bounds=ci.get('interval95')
                rows.append(dict(variant=v,feature=f,label=label,measured=r['measured_questions'],population=r['population_questions'],
                    missing=r['missing_questions'],pearson=r['pearson'],spearman=r['spearman'],raw_direction_auc=r['raw_direction_auc'],
                    pearson_ci95_low=None if bounds is None else bounds[0],pearson_ci95_high=None if bounds is None else bounds[1],
                    interval_NA_reason=ci.get('NA_reason')))
    write_csv(output/'train-selected-relations.csv',tuple(rows[0]),rows)
    rows=[]
    for name,model in aggregate['heldout_relationships'].items():
        for f,r in model['features'].items():
            rows.append(dict(endpoint=name,feature=f,metric_available=r['metric_available'],metric_unavailable=r['metric_unavailable'],
                subset_em=r['subset_em'],pearson=r['pearson'],spearman=r['spearman'],
                increasing_EM_auc=r['original_increasing_direction_EM_AUC'],adjusted_partial_pearson=r['adjusted_partial_pearson'],
                NA_reason=r['association_NA_reason']))
    write_csv(output/'heldout-relations.csv',tuple(rows[0]),rows)
    rows=[]
    for c in aggregate['parameter_movement']:
        for factor,values in dict(all=c['metrics']['aggregate'],**c['metrics']['by_factor']).items():
            rows.append(dict(variant=c['variant'],source=c['source'],destination=c['destination'],factor=factor,
                source_updates=c['source_updates'],destination_updates=c['destination_updates'],delta_l2=values['delta_l2'],
                source_l2=values['source_l2'],destination_l2=values['destination_l2'],relative_l2=values['relative_l2'],
                relative_l2_NA_reason=values['relative_l2_na_reason'],changed_tensors=values['changed_tensors']))
    write_csv(output/'parameter-displacement.csv',tuple(rows[0]),rows)


def plots(aggregate, output):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'svg.fonttype':'none'})
    tag='SYNTHETIC FIXTURE - NOT EXPERIMENT RESULTS\n' if aggregate['synthetic_fixture'] else ''
    def save(fig,name):
        fig.savefig(output/(name+'.png'),bbox_inches='tight');fig.savefig(output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.8,5.5))
    for v in VARIANTS:
        rr=aggregate['curves'][v];ax.plot([r['outer'] for r in rr],[100*r['em'] for r in rr],label=v,color=COLORS[v],marker='o',markersize=3,lw=1.5)
    ax.set(xlabel='Completed sampling/training outer (1-25)',ylabel='Preupdate sampled TRAIN strict EM (%)',
           title=tag+'All 512 questions x 5 trajectories per outer (denominator 2,560)')
    ax.set_xticks([1,5,10,15,20,25]);ax.grid(alpha=.2)
    # Reserve separate figure-coordinate rows: axes labels, legend, then caption.
    # Axes-relative negative anchors previously placed legend and caption together.
    fig.subplots_adjust(bottom=.27,top=.87)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.105))
    fig.text(.5,.025,'TRAIN cohorts differ across outer; this curve is not fixed-heldout or postupdate EM.',ha='center',fontsize=9)
    save(fig,'train-preupdate-em')
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.4),sharex=True)
    for ax,dataset,label in zip(axes,('overall','nq','hotpotqa'),('Overall (512)','NQ (256)','HotpotQA (256)')):
        for v in VARIANTS:
            vals=[aggregate['models']['source100/0']['overall' if dataset=='overall' else 'datasets'][dataset] if dataset!='overall' else aggregate['models']['source100/0']['overall']]
            vals += [aggregate['models'][v+'/'+str(t)]['overall'] if dataset=='overall' else aggregate['models'][v+'/'+str(t)]['datasets'][dataset] for t in (15,25)]
            ax.plot([0,15,25],[100*r['em'] for r in vals],marker='o',color=COLORS[v],label=v,lw=1.8)
        ax.set(title=label,xlabel='Saved raw endpoint (outer)',xticks=[0,15,25]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Fixed native greedy strict EM (%)')
    fig.suptitle(tag+'Same 512 validation questions; primary endpoint = outer 25',y=1.06)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.1))
    fig.tight_layout();save(fig,'heldout-zero-15-25')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.1,1]})
    finals=[100*aggregate['models'][v+'/25']['overall']['em'] for v in VARIANTS]
    axes[0].bar(VARIANTS,finals,color=[COLORS[v] for v in VARIANTS],width=.62)
    baseline=100*aggregate['models']['source100/0']['overall']['em'];axes[0].axhline(baseline,color='#666666',ls='--',label='Common zero')
    for i,value in enumerate(finals):axes[0].text(i,value+.6,format(value,'.2f')+'%',ha='center',fontsize=9)
    axes[0].set(ylabel='Strict EM (%)',title='Final outer 25; common-zero reference');axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    ymax=max(finals+[baseline]);axes[0].set_ylim(0,min(100,ymax+9))
    for i,key in enumerate(PRIMARY):
        p=aggregate['paired'][key];point=p['delta_percentage_points'];ci=p['ci95']
        axes[1].plot([ci['low'],ci['high']],[i,i],color='#333333',lw=2)
        axes[1].plot(point,i,'o',color=COLORS['G_POS' if i==0 else 'H_POS'])
    axes[1].axvline(0,color='#777777',ls='--');axes[1].set(yticks=[0,1],yticklabels=['G_POS - G_NEG','H_POS - H_NEG'],
        xlabel='Paired strict EM difference (percentage points)',title='Predeclared family of two; native 95% intervals',ylim=(-.6,1.6));axes[1].grid(axis='x',alpha=.2)
    fig.suptitle(tag+'Final primary comparisons (no best-of-15/25 selection)',y=1.06);fig.tight_layout();save(fig,'final-primary-comparisons')


def markdown(aggregate, data):
    effects=data['analyses']['effects'];held=data['analyses']['heldout_relationship'];train=data['analyses']['train_relationship']
    lines=['# 四组动态权重消融最终报告','']
    if aggregate['synthetic_fixture']:
        lines += ['**合成排版样例：所有数值均为合成数据，不是实验结果，不可作为正式验收。**','']
    lines += ['四组各完成 25 个在线采样／训练 outer。25 outer 不是 25 次 Adam 更新。正式主终点为第 25 轮末 raw checkpoint；第 15 轮用于预先指定的阶段观察，不以较高值替代第 25 轮。',
        '共同起点是 Search-R1 checkpoint100 与相同初始 LoRA，各组 fresh Adam。四卡 micro1×accumulation8 保持 global32。G_POS/G_NEG 对答案 student−teacher logprob 差正／负加权，H_POS/H_NEG 对学生全 policy entropy 正／负加权；幅度 0.9，题内五兄弟中心化／最大绝对偏差归一，实际优化分母为本批真实轨迹数。','',
        '## 正式测试 EM','',
        '以下指标来自相同 512 题的普通提示 native greedy 测试；没有新增 pass5。总体、NQ 和 HotpotQA 的分母分别为 512、256、256。只展示原统计入口已经计算的区间和检验。','']
    rows=[]
    for name in ['source100/0']+[v+'/'+str(t) for v in VARIANTS for t in (15,25)]:
        m=aggregate['models'][name];o=m['overall'];ds=m['datasets']
        rows.append([name,str(o['hits'])+'/512',pct(o['em']),pct(ds['nq']['em']),pct(ds['hotpotqa']['em']),o['final_invalid'],o['valid_gold_mismatch'],o['no_executed_search']])
    lines += [table(['权重／时点','正确数','总体 EM','NQ EM','HotpotQA EM','最终行动无效','有效但答案错','未执行搜索'],rows),'',
              '![同题测试曲线](heldout-zero-15-25.png)','', '## 第 25 轮主比较','']
    rows=[]
    for key in PRIMARY:
        p=effects['paired'][key];ci=p['ci95'];test=p['exact_mcnemar_holm']
        rows.append([p['delta_direction'],num(p['delta_percentage_points'])+' pp',
            '['+num(ci['low'])+', '+num(ci['high'])+']',p['gained'],p['lost'],format(test['raw_p'],'.6g'),format(test['holm_p'],'.6g'),
            '是' if test['reject_at_family_alpha'] else '否'])
    lines += [table(['右减左','EM 差','原配对 95% CI（pp）','新增正确','丢失正确','exact McNemar p','Holm p','family α=.05 拒绝'],rows),'',
        '区间为原入口的按题、NQ／HotpotQA 分层配对 percentile bootstrap；不是跨 seed 的算法不确定性。主比较家族仅 G_POS−G_NEG 与 H_POS−H_NEG 两项。G↔H 交叉比较为描述性结果，不新增显著性家族。','',
        '![最终比较](final-primary-comparisons.png)','', '## 每轮训练采样表现','',
        '每一点包含本轮全部 512×5=2560 条真实采样，含全错题及未被训练选中的轨迹。它是本轮采样模型更新前的严格 EM；各轮 TRAIN 题集不同，因此不是固定测试曲线，也不能据点间下降断言同题性能退化。','',
        '![训练前采样 EM](train-preupdate-em.png)','', '## EM 与 loss／概率／权重的关系','',
        'TRAIN 表是已选 D、存在成功 B 且可编码的真实训练轨迹子集的观测关联。标签来自 outer 起始采样，loss／概率来自该响应后来实际更新前 forward；它们不是预测未来成功的概率。未成功取得 B 的题仍进入训练 EM 分母，但其 teacher 数值为 NA，不填 0。','']
    rows=[]
    for v in VARIANTS:
        for f in TRAIN_FEATURES:
            for label,display in (('selected_reward','D 严格 EM'),('sampling_s_over_5','同题 s/5')):
                r=aggregate['train_relationships'][v][f][label];ci=r['pearson_question_block'].get('interval95')
                rows.append([v,LABELS[f],display,str(r['measured_questions'])+'/'+str(r['population_questions']),num(r['pearson']),
                             num(r['spearman']),'NA' if ci is None else '['+num(ci[0])+', '+num(ci[1])+']',num(r['raw_direction_auc'])])
    lines += [table(['组','指标','标签','数值覆盖','Pearson r','Spearman ρ','原 r 的题块 95% CI','原递增方向 AUC'],rows),'',
        '正 r 表示该指标越大、该子集的成功标签通常越高；负 r 表示通常越低。接近 0 表示线性关系弱，不说明一定没有非线性关系。原递增方向 AUC<0.5 表示较大的指标偏向失败，不自动翻转成成功概率；NA 表示样本不足、常量或单一标签等原定义条件，不表示 0 相关。','',
        '固定测试响应事后重放的 gap／teacher KL 使用共同 zero 成功 B bank；该 bank 为 gold 选择的约 239 题子集（以实际覆盖为准），因此包含答案信息和选择偏差／collider 风险。entropy／普通策略 drift 的可用性单独记录，不因没有 B 删去测试 EM。不同模型自己的响应历史是内生变化，不能把相同 token 位置视作同一事件。','']
    rows=[]
    for name in ['source100/0']+[v+'/'+str(t) for v in VARIANTS for t in (15,25)]:
        model=aggregate['heldout_relationships'][name]
        for f in ('answer_gap','student_entropy','post_clip_teacher_loss','source100_ordinary_reverse_kl'):
            r=model['features'][f]
            rows.append([name,LABELS[f],str(r['metric_available'])+'/512',pct(r['subset_em']) if r['subset_em'] is not None else 'NA',
                         num(r['pearson']),num(r['spearman']),num(r['original_increasing_direction_EM_AUC']),
                         r['association_NA_reason'] or '—'])
    lines += [table(['端点','事后指标','覆盖','指标子集 EM','r','ρ','原方向 AUC','NA 原因'],rows),'',
        '本表不添加未计算的关联置信区间或 p 值。完整 CSV 还保存原入口的探索性 partial Pearson，但它不证明控制变量消除了混杂。','',
        '## 区域 loss 与参数位移','',
        'think／search／最终答案／结构标签等区域保存的是 loss、entropy 的标量贡献与覆盖。区域贡献可定位分歧集中位置，不能当作逐 token 参数梯度，也不能证明这些位置导致性能下降。完整区域聚合保存在 small_aggregate.json，原逐 token 和轨迹留远端。','']
    rows=[]
    for c in aggregate['parameter_movement']:
        m=c['metrics'];rows.append([c['variant'],c['source']+'→'+c['destination'],num(m['aggregate']['delta_l2'],6),
            num(m['by_factor']['A']['delta_l2'],6),num(m['by_factor']['B']['delta_l2'],6),
            num(m['aggregate']['relative_l2'],6),m['aggregate']['changed_tensors']])
    lines += [table(['组','保存区间','LoRA 参数 ΔL2','A ΔL2','B ΔL2','ΔL2／源参数 L2','改变张量数'],rows),'',
        '这是保存的 LoRA A/B 坐标空间位移，受因子分解尺度影响；没有计算有效模型 BA 权重变化、梯度或单步更新向量。B 的初始范数为 0 时相对位移为 NA，不填 0／无穷大。只有 zero／15／25 保存点，不能插值出精确退化发生时刻。','',
        '## 证据与结论边界','',
        '正式覆盖验收要求四组共 100 outer、256000 条 TRAIN 采样、common zero 加八次测试共 4608 个正式测试槽、真实四个最终 raw checkpoint、完整实际退出／释放、数值状态、远端备份及原分析。debug retention 另验 800 个采样 worker 与 100 个 teacher parent；teacher 实际案例数按可用 P 计算，不凑每轮 8 例。','',
        '本消融只有一个 seed，没有未加权 E 的第 25 轮同条件对照；不能声称优于 E25、跨 seed 收益、校准 pass 概率或逐 token 因果效应。现有统计描述权重方向的受控干预差异及观测机制，不用相关性替代最终 EM。','',
        '小型导出：六个聚合 CSV、三幅 PNG／SVG、small_aggregate.json（最多 256 KiB）、本报告与 complete.json。没有原题、question_id、完整轨迹、逐题预测、权重或 tensor；这些原件均留 HKUST。','']
    if not data['synthetic']:
        lines += ['输入原件路径与 SHA：','',table(['证据','绝对路径','SHA256'],
            [[name,ref['path'],ref['sha256']] for name,ref in
             [('final coverage',data['request']['coverage']['output'])]+
             [(k,p['output']) for k,p in data['request']['analysis'].items()]+
             [('debug retention',data['request']['debug_retention']),('parameter movement',data['request']['parameter_movement'])]]),'']
    return '\n'.join(lines)+'\n'


def render(data, output_dir):
    output=Path(output_dir).resolve();require(not output.exists(),'Exclusive new report output directory required')
    aggregate=small_aggregate(data)
    if not data['synthetic']:data['reader'].stable_end()
    output.mkdir(parents=True)
    (output/'small_aggregate.json').write_bytes(aggregate_bytes(aggregate))
    csvs(aggregate,output);plots(aggregate,output)
    (output/'report-zh.md').write_text(markdown(aggregate,data),encoding='utf-8')
    if not data['synthetic']:data['reader'].stable_end()
    refs={p.name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
    complete=dict(schema='opsd-signal-final-report-complete-v1',accepted=not data['synthetic'],
        formal_report=not data['synthetic'],synthetic_fixture=data['synthetic'],synthetic_rendered=data['synthetic'],
        study_id=data['study_id'],outputs=refs,small_aggregate_bytes=refs['small_aggregate.json']['bytes'],
        original_statistics_recomputed=False,new_training_generation_model_tensor_reads_or_network=False,
        full_study_complete=not data['synthetic'],request=data.get('request_ref'),
        final_coverage=None if data['synthetic'] else data['request']['coverage']['output'],
        inputs=[] if data['synthetic'] else list(data['reader'].proofs.values()))
    path=output/'complete.json';path.write_bytes(encode(complete))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
