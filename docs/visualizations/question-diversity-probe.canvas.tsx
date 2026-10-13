import React, { useEffect, useRef, useState } from "react";
import { questionDiversityStyles } from "./question-diversity-probe.styles";

/** An illustrative animation. No model, network, or training actions are performed. */
const PROBABILITIES = [0.05, 0.10, 0.25, 0.40, 0.15, 0.05];
const STAGES = ["候选题", "问题编码", "预测差异", "选 512 题", "真实轨迹", "双路更新"];
const TITLES = ["大候选池，只读问题", "编码问题，复用缓存", "预测五条轨迹会有多大差异", "随机探索 + 高差异选题", "只给选中的问题生成轨迹", "同一批轨迹，更新两条学习链"];
const DETAILS = ["5,120 个候选问题，此时尚未生成任何训练轨迹。", "固定 encoder 不随 actor 更新；新的选题 head 每轮更新。", "概率分布同时给出 mean 与 std；准确率接近 0.5 不等于组内差异大。", "随机题先入选，再从剩余候选按预测 std 排序，合计始终为 512 题。", "预测和实测分开记录；全对、全错组也保留，不补采。", "标签来自更新前的 actor；新 head 和新 actor 在下一轮共同使用。"];
const sampleStd = (k: number) => Math.sqrt(k * (5 - k) / 20);
const predictedMean = PROBABILITIES.reduce((sum, p, k) => sum + p * k / 5, 0);
const predictedStd = PROBABILITIES.reduce((sum, p, k) => sum + p * sampleStd(k), 0);

function QuestionCloud({ count, exploratory = -1 }: { count: number; exploratory?: number }) {
  return <>{Array.from({ length: count }, (_, i) => <div key={i} className="qd-question" style={{ "--i": i, ...(exploratory >= 0 ? { background: `color-mix(in srgb,var(--viz-series-${i < exploratory ? 2 : 1}) 65%,transparent)` } : {}) } as React.CSSProperties} />)}</>;
}

function Distribution() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    const update = () => setWidth(Math.max(280, Math.round(host.getBoundingClientRect().width)));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(host);
    return () => observer.disconnect();
  }, []);
  const baseline = 134;
  const left = 30;
  const dx = (width - 42) / 6;
  const barWidth = Math.min(48, dx * 0.58);
  return <div ref={ref} className="qd-distribution"><svg viewBox={`0 0 ${width} 170`} height={170} role="img" aria-label="示例预测，五条轨迹答对零至五条的概率为 5%、10%、25%、40%、15%、5%">
    <title>示例预测：五条轨迹的答对条数</title><desc>预测 mean 0.530，预测样本 std 0.468。</desc>
    <line x1={left} y1={baseline} x2={width - 12} y2={baseline} stroke="var(--border)" />
    {PROBABILITIES.map((p, k) => { const x = left + dx * (k + 0.5); const height = p / 0.45 * 112; return <g key={k}>
      <rect x={x - barWidth / 2} y={baseline - height} width={barWidth} height={height} fill="var(--viz-series-1)" opacity={k === 0 || k === 5 ? 0.25 : 0.8} />
      <text x={x} y={baseline - height - 7} textAnchor="middle">{Math.round(p * 100)}%</text><text x={x} y={baseline + 19} textAnchor="middle">{k}</text>
    </g>; })}
    <text x={width - 12} y={169} textAnchor="end">答对条数 K</text>
  </svg></div>;
}

export default function QuestionDiversityProbe() {
  const [step, setStep] = useState(0);
  const [exploration, setExploration] = useState(64);
  const [correct, setCorrect] = useState(2);
  const [playing, setPlaying] = useState(false);
  const [animated, setAnimated] = useState(false);
  const chooseStep = (next: number) => { setPlaying(false); setAnimated(true); setStep(next); };
  useEffect(() => {
    if (!playing || step >= 5) return;
    const timer = window.setTimeout(() => { setAnimated(true); setStep(step + 1); if (step + 1 === 5) setPlaying(false); }, 3200);
    return () => window.clearTimeout(timer);
  }, [playing, step]);
  useEffect(() => {
    const pauseHidden = () => { if (document.hidden) setPlaying(false); };
    document.addEventListener("visibilitychange", pauseHidden);
    return () => document.removeEventListener("visibilitychange", pauseHidden);
  }, []);
  const sceneClass = `qd-scene${animated ? " qd-enter" : ""}`;
  return <div id="qd-probe-20261013" className="qd-root">
    <style>{questionDiversityStyles}</style>
    <div className="qd-heading"><div><h2>先选问题，再生成轨迹</h2><span className="text-small text-muted">Question diversity probe · 原理演示，非训练结果</span></div></div>
    <div className="viz-controls qd-controls">
      <button className="btn btn-primary" type="button" onClick={() => { setAnimated(true); if (step === 5) setStep(0); setPlaying(!playing); }}>{playing ? "暂停" : step === 5 ? "重新播放" : "播放流程"}</button>
      <button className="btn" type="button" disabled={step === 5} onClick={() => chooseStep(Math.min(5, step + 1))}>下一步</button>
      <button className="btn btn-ghost" type="button" onClick={() => chooseStep(0)}>重播</button>
      <label className="form-label" htmlFor="qd-canvas-explore">随机探索题数 <select id="qd-canvas-explore" className="form-select" value={exploration} onChange={event => { setExploration(Number(event.target.value)); chooseStep(3); }}>{[0, 64, 128, 256].map(n => <option key={n} value={n}>{n}</option>)}</select></label>
    </div>
    <div className="qd-steps" aria-label="训练流程">{STAGES.map((label, index) => <button key={label} className="btn btn-ghost" type="button" aria-pressed={step === index} onClick={() => chooseStep(index)}><span className="qd-index">{index + 1}</span>{label}</button>)}</div>
    <div className="qd-stage-heading"><h3>{TITLES[step]}</h3><span className="text-small text-muted tabular-nums">{step + 1} / 6</span></div>
    <div className="qd-main" data-phase={step}>
      {step === 0 && <section key={0} className={sceneClass} aria-label="候选问题池"><div className="qd-pool-label"><span>候选问题池</span><span className="tabular-nums">5,120 题</span></div><div className="qd-question-cloud" aria-hidden="true"><QuestionCloud count={80} /></div><div className="qd-flow-line"><span>只有问题文本</span><span>→</span><span>下一步：固定编码器</span></div></section>}
      {step === 1 && <section key={1} className={sceneClass} aria-label="只对问题编码，不做检索"><div className="qd-encoder-layout">
        <div className="qd-input-stack" aria-hidden="true">{["问题 q₁", "问题 q₂", "问题 q₃", "…"].map(q => <div key={q}>{q}</div>)}</div>
        <div className="qd-stream" aria-hidden="true"><span /><span /><span /></div><div className="qd-encoder"><div className="qd-encoder-mark" aria-hidden="true" /><span>固定问题 encoder</span><span className="text-small text-muted">首次编码 · 后续读缓存</span></div>
        <div className="qd-stream" aria-hidden="true"><span /><span /><span /></div><div className="qd-vectors">{[0, 1, 2].map(j => <div key={j} className="qd-vector" aria-hidden="true">{Array.from({ length: 8 }, (_, i) => <i key={i} style={{ height: `${6 + (i * 7 + j * 3) % 17}px` }} />)}</div>)}<span className="text-small text-muted">问题表征缓存</span></div>
      </div><div className="qd-flow-line"><span>无 CoT</span><span>·</span><span>无搜索</span><span>·</span><span>无答案生成</span></div></section>}
      {step === 2 && <section key={2} className={sceneClass} aria-label="预测五条轨迹答对条数分布"><div className="qd-distribution-heading"><span>同一个问题 q · 预测答对条数 K</span><span className="text-small text-muted">轻量探针</span></div><Distribution /><div className="qd-moments"><span>预测 mean <strong className="tabular-nums">{predictedMean.toFixed(3)}</strong></span><span>预测 std <strong className="tabular-nums">{predictedStd.toFixed(3)}</strong></span><span className="text-small text-muted">std = 未来五条 EM 的样本标准差期望</span></div></section>}
      {step === 3 && <section key={3} className={sceneClass} aria-label="随机探索与高差异选题"><div className="qd-pool-label"><span>从 5,120 题中选择</span><span className="tabular-nums">512 题</span></div><div className="qd-selection-strip" role="img" aria-label={`随机 ${exploration} 题，预测差异最高 ${512 - exploration} 题`}><div className="qd-exploration-fill" style={{ width: `${100 * exploration / 512}%` }} /><div className="qd-top-fill" style={{ width: `${100 * (512 - exploration) / 512}%` }} /></div><div className="qd-selection-labels"><span><i className="qd-legend-dot qd-explore-color" />均匀随机 <strong className="tabular-nums">{exploration}</strong> 题</span><span><i className="qd-legend-dot qd-focus-color" />预测 std 最高 <strong className="tabular-nums">{512 - exploration}</strong> 题</span></div><div className="qd-selected-cloud" aria-hidden="true"><QuestionCloud count={64} exploratory={exploration / 8} /></div><div className="qd-flow-line"><span>先随机，再从剩余候选取 Top</span><span>·</span><span>每题只选一次</span></div></section>}
      {step === 4 && <section key={4} className={sceneClass} aria-label="生成真实五条轨迹，得到严格 EM"><div className="qd-rollout-heading"><span>512 题 × 5 条 = <strong className="tabular-nums">2,560 条真实轨迹</strong></span><label className="form-label" htmlFor="qd-canvas-correct">示例答对数 <select id="qd-canvas-correct" className="form-select" value={correct} onChange={event => setCorrect(Number(event.target.value))}>{[0, 1, 2, 3, 4, 5].map(k => <option key={k} value={k}>{k} / 5</option>)}</select></label></div><div className="qd-rollout"><div className="qd-rollout-q">问题 q</div><div className="qd-trajectories">{Array.from({ length: 5 }, (_, i) => <div key={i} className="qd-trajectory" style={{ "--i": i } as React.CSSProperties}><div className="qd-trace" aria-label="思考、搜索和答案"><i /><i /><i /><i /><i /></div><div className={`qd-em${i < correct ? " is-correct" : ""}`}>{i < correct ? "✓" : "×"}</div><span className="text-small tabular-nums">EM {i < correct ? 1 : 0}</span></div>)}</div></div><div className="qd-moments"><span>实测 mean <strong className="tabular-nums">{(correct / 5).toFixed(3)}</strong></span><span>实测 std <strong className="tabular-nums">{sampleStd(correct).toFixed(3)}</strong></span><span className="text-small text-muted">strict EM · ddof = 1</span></div></section>}
      {step === 5 && <section key={5} className={sceneClass} aria-label="actor 与选题探针双路更新"><div className="qd-feedback-source">同一批真实轨迹</div><div className="qd-update-lanes"><div className="qd-update-lane"><div className="qd-lane-link" aria-hidden="true" /><span>原 Search-G1 奖励</span><div className="qd-update-mark qd-actor-mark" aria-hidden="true" /><strong>Actor / PPO 更新</strong><span className="text-small text-muted">原奖励与三项防护</span></div><div className="qd-update-lane"><div className="qd-lane-link" aria-hidden="true" /><span>严格 EM → K / mean / std</span><div className="qd-update-mark qd-head-mark" aria-hidden="true" /><strong>选题 head 在线更新</strong><span className="text-small text-muted">每个成功 iteration 更新一次</span></div></div><div className="qd-next-loop"><span className="qd-loop-mark" aria-hidden="true">↶</span><span>下一轮：新 actor + 新选题 head → 再选 512 题</span></div><div className="text-small text-muted qd-reward-note">原奖励 probe 的 50 轮刷新周期独立保留</div></section>}
    </div>
    <div className="qd-detail text-small" aria-live="polite">{DETAILS[step]}</div>
    <div className="qd-budget" aria-label="完整轨迹生成预算对比"><div className="qd-budget-row"><span>原 mixed-first 上限</span><div className="qd-budget-track"><div className="qd-budget-old" /></div><span className="tabular-nums">10,240 条</span></div><div className="qd-budget-row"><span>探针选题</span><div className="qd-budget-track"><div className="qd-budget-new" /></div><span className="tabular-nums">2,560 条</span></div><div className="text-small text-muted">完整轨迹生成量比较，不代表总训练耗时缩短 4 倍</div></div>
  </div>;
}
