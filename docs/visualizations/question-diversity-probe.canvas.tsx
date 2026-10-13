import React, { useEffect, useState } from "react";
import { questionDiversityStyles } from "./question-diversity-probe.styles";

/** Synthetic explanatory examples. Controls never start training or network work. */
const DATA = {
  stages: ["候选题", "缓存", "预测", "选题", "轨迹", "更新"],
  titles: ["先看问题，不急着生成", "一次编码，之后轻量读取", "预测：五条回答会有多大差异？", "把预算留给高差异问题", "现在，才生成真实搜索轨迹", "同一批轨迹，两路学习"],
  details: ["5,120 个候选，此时没有 CoT、检索或答案生成。", "官方固定 encoder → 问题表征缓存；每轮只运行轻量 head。", "排序依据是五条严格 EM 的期望样本 std，而非平均准确率。", "先均匀随机探索，再从剩余候选取 Top；不重复、不按实测 EM 补采。", "真实 EM 再作为探针标签；全对、全错组也完整保留。", "标签对应更新前的 actor；新 actor 与新 head 用于下一轮选题。"],
  probabilities: [0.05, 0.10, 0.25, 0.40, 0.15, 0.05],
  phaseMs: [3900, 4200, 5200, 4600, 5400, 4200],
};
const sampleStd = (k: number) => Math.sqrt(k * (5 - k) / 20);
const predictedMean = DATA.probabilities.reduce((v, p, k) => v + p * k / 5, 0);
const predictedStd = DATA.probabilities.reduce((v, p, k) => v + p * sampleStd(k), 0);
const CX = ["候选", "缓存", "预测", "选择", "生成", "更新"];

function Pool({ selected = false, exploration = 64 }: { selected?: boolean; exploration?: number }) {
  const count = selected ? 64 : 80;
  return <div className={`qd-pool${selected ? " qd-pool-selected" : ""}`} role="img" aria-label={selected ? `${exploration}题随机探索，${512-exploration}题按预测差异选择，方块为示意` : "5,120个候选问题，方块为示意"}>
    {Array.from({length: count}, (_,i) => <span key={i} data-pool-index={i} data-question-index={i} className={`qd-ticket${selected ? i < exploration/8 ? " is-explore" : " is-focus" : ""}`} style={{"--i":i} as React.CSSProperties}><i/><i/><i/></span>)}
  </div>;
}

function Moments({actual = false, correct = 2}: {actual?: boolean; correct?: number}) {
  return <div className="qd-moments">
    <div><span className="text-small text-muted">{actual ? "实测" : "预测"} mean</span><h2 className="tabular-nums" data-field={actual ? "true-mean" : "pred-mean"}>{(actual ? correct/5 : predictedMean).toFixed(3)}</h2></div>
    <div><span className="text-small text-muted">{actual ? "实测" : "预测"} std</span><h2 className="tabular-nums" data-field={actual ? "true-std" : "pred-std"}>{(actual ? sampleStd(correct) : predictedStd).toFixed(3)}</h2></div>
    <span className="text-small text-muted qd-moment-note">{actual ? "严格 EM · 样本 std · ddof = 1" : "五条 EM 的样本标准差期望"}</span>
  </div>;
}

function Distribution() {
  return <div className="qd-distribution" role="img" aria-label="示例五条轨迹答对数量的概率：5%、10%、25%、40%、15%、5%">
    <div className="qd-chart-label text-small text-muted">概率 P(K)</div>
    <div className="qd-bars">{DATA.probabilities.map((p,k) => <div key={k} className={`qd-bar-column${k===0||k===5 ? " is-endpoint" : ""}`}>
      <div className="qd-bar-space"><div className="qd-bar-measure" style={{height:`${p/0.45*100}%`}} data-bar-k={k}><span className="text-small tabular-nums">{Math.round(p*100)}%</span><div className="qd-bar"/></div></div><span className="text-small tabular-nums">{k}</span>
    </div>)}</div>
    <div className="qd-chart-axis text-small text-muted">五条轨迹中答对的数量 K</div>
  </div>;
}

function Trajectories({correct}: {correct: number}) {
  return <div className="qd-rollout" aria-label="同一个问题分叉成五条思考、检索、答案轨迹">
    <div className="qd-q-origin"><span className="text-small text-muted">同一个问题</span><h2>q</h2></div>
    <div className="qd-traces">
      <div className="qd-trace-labels text-small text-muted"><span>思考</span><span>检索</span><span>答案</span><span>EM</span></div>
      {Array.from({length:5},(_,i)=><div key={i} className="qd-trace-row" data-trace-index={i} data-trajectory={i} data-correct={i<correct} style={{"--i":i} as React.CSSProperties}>
        <div className="qd-trace-thought"><i/><i style={{width:`${38+i*9}%`}}/></div>
        <div className="qd-trace-search"><span/><i/></div>
        <div className="qd-trace-answer"><i style={{width:`${62-i*7}%`}}/></div>
        <span className={`qd-em${i<correct ? " is-correct" : ""}`} data-em-index={i} data-em={i} aria-label={`轨迹${i+1}，EM ${i<correct?1:0}`}>{i<correct ? "1" : "0"}</span>
      </div>)}
    </div>
  </div>;
}

export default function QuestionDiversityProbe() {
  const [step,setStep]=useState(0),[exploration,setExploration]=useState(64),[correct,setCorrect]=useState(2);
  const [playing,setPlaying]=useState(false),[animated,setAnimated]=useState(false);
  const choose=(next:number)=>{setPlaying(false);setAnimated(true);setStep(next);};
  useEffect(()=>{
    if(!playing||step>=5)return;
    const timer=window.setTimeout(()=>{setAnimated(true);setStep(step+1);if(step+1===5)setPlaying(false);},DATA.phaseMs[step]);
    return()=>window.clearTimeout(timer);
  },[playing,step]);
  useEffect(()=>{const pause=()=>{if(document.hidden)setPlaying(false);};document.addEventListener("visibilitychange",pause);return()=>document.removeEventListener("visibilitychange",pause);},[]);
  return <div id="qd-probe-20261013" className="qd-root" data-model={JSON.stringify(DATA)} data-phase={step} data-animated={animated ? "true" : "false"}>
    <style dangerouslySetInnerHTML={{__html:questionDiversityStyles}}/>
    <header className="qd-header"><span className="qd-eyebrow text-small text-muted">SEARCH-G1 / QUESTION DIVERSITY PROBE</span><h1><span>先选好问题，</span><span>再生成轨迹。</span></h1><span className="text-small text-muted">算法原理 · 示意预测与轨迹，非训练结果</span></header>
    <div className="qd-player">
      <div className="viz-controls"><button type="button" className="btn btn-primary" data-action="play" aria-pressed={playing} onClick={()=>{setAnimated(true);if(step===5)setStep(0);setPlaying(!playing);}}>{playing ? "暂停" : step===5 ? "重新播放" : "播放流程"}</button><button type="button" className="btn btn-ghost" data-action="next" disabled={step===5} onClick={()=>choose(Math.min(5,step+1))}>下一步 →</button></div>
      <span className="qd-player-phase text-small text-muted tabular-nums" data-field="stage-count">{step+1} / 6</span>
    </div>
    <nav className="qd-navigation" aria-label="六个阶段">{DATA.stages.map((name,i)=><div key={name} className={`qd-nav-item${step===i ? " is-active" : ""}${step>i ? " is-done" : ""}`} data-nav-index={i}>
      <button type="button" className="btn btn-ghost" data-step={i} aria-label={`${i+1} ${CX[i]}`} aria-pressed={step===i} onClick={()=>choose(i)}>{String(i+1).padStart(2,"0")}</button><span className="text-small">{name}</span>
    </div>)}</nav>
    <div className="qd-stage-heading"><span className="qd-stage-rule"/><h3 data-field="stage-title">{DATA.titles[step]}</h3></div>
    <main className="qd-main">
      <section className="qd-scene" data-scene="0" hidden={step!==0}>
        <div className="qd-candidate-top"><div><span className="text-small text-muted">本轮候选问题池</span><h2 className="tabular-nums">5,120 <span className="text-small text-muted">题</span></h2></div><span className="text-small text-muted">仅问题文本 · 尚未 rollout</span></div>
        <Pool/>
        <div className="qd-flow-caption"><div className="qd-mini-paper"><i/><i/><i/></div><span>问题 q</span><span className="qd-arrow">→</span><span>轻量预测</span><span className="qd-arrow">→</span><span>选择 512 题</span></div>
      </section>
      <section className="qd-scene" data-scene="1" hidden={step!==1}>
        <div className="qd-cache-flow">
          <div className="qd-flow-unit qd-input-unit"><div className="qd-paper-stack"><div><i/><i/><i/></div><div><i/><i/><i/></div><div><i/><i/><i/></div></div><span>只有问题文本</span></div>
          <div className="qd-transport"><span/><span/><span/></div>
          <div className="qd-flow-unit"><div className="qd-encoder-art"><span/><span/><span/><span/><span/><span/><span/><span/><span/></div><span>固定 encoder</span><span className="text-small text-muted">官方 Qwen · 不随 actor 更新</span></div>
          <div className="qd-transport"><span/><span/><span/></div>
          <div className="qd-flow-unit"><div className="qd-vector-cache">{Array.from({length:4},(_,r)=><div key={r}>{Array.from({length:9},(_,c)=><i key={c} style={{opacity:.2+((r*5+c*3)%8)/12}}/>)}</div>)}</div><span>问题表征缓存</span><span className="text-small text-muted">首次写入 · 之后直接读取</span></div>
        </div>
        <div className="qd-cache-read"><span className="qd-cache-line"/><span className="text-small text-muted">每轮读取缓存</span><span className="qd-arrow">→</span><span>轻量选题 head</span></div>
      </section>
      <section className="qd-scene" data-scene="2" hidden={step!==2}>
        <div className="qd-prediction-context"><span>同一个问题 q</span><span className="text-small text-muted">预测五条轨迹的答对数量</span></div>
        <Distribution/><Moments/>
      </section>
      <section className="qd-scene" data-scene="3" hidden={step!==3}>
        <div className="qd-select-head"><div><span className="text-small text-muted">5,120 →</span><h2 className="tabular-nums">512 <span className="text-small text-muted">题</span></h2></div><div className="qd-select-setting"><label className="form-label" htmlFor="qd-canvas-explore">随机探索题数</label><select id="qd-canvas-explore" data-action="exploration" className="form-select" value={exploration} onChange={e=>{setExploration(Number(e.target.value));setAnimated(true);setPlaying(false);}}>{[0,64,128,256].map(n=><option key={n} value={n}>{n}</option>)}</select></div></div>
        <div className="qd-selection-strip" role="img" aria-label={`均匀随机${exploration}题，预测std最高${512-exploration}题`}><div className="qd-exploration-fill" data-allocation="random" style={{width:`${exploration/512*100}%`}}/><div className="qd-top-fill" data-allocation="top" style={{width:`${(512-exploration)/512*100}%`}}/></div>
        <div className="qd-selection-labels"><span><i className="qd-legend-dot is-explore"/>均匀随机 <strong className="tabular-nums" data-field="explore-count">{exploration}</strong></span><span><i className="qd-legend-dot is-focus"/>高预测 std <strong className="tabular-nums" data-field="focus-count">{512-exploration}</strong></span></div>
        <Pool selected exploration={exploration}/>
        <div className="qd-selection-order text-small text-muted"><span>01 先随机</span><span>→</span><span>02 再取剩余 Top</span><span>→</span><span>03 合计 512</span></div>
      </section>
      <section className="qd-scene" data-scene="4" hidden={step!==4}>
        <div className="qd-rollout-heading"><span>512 题 × 5 条 <span className="qd-arrow">→</span> <strong className="tabular-nums">2,560 条真实轨迹</strong></span><div className="qd-rollout-setting"><label className="form-label" htmlFor="qd-canvas-correct">示例答对数</label><select id="qd-canvas-correct" className="form-select" data-action="correct" value={correct} onChange={e=>{setCorrect(Number(e.target.value));setPlaying(false);}}>{[0,1,2,3,4,5].map(k=><option key={k} value={k}>{k} / 5</option>)}</select></div></div>
        <Trajectories correct={correct}/><Moments actual correct={correct}/>
      </section>
      <section className="qd-scene" data-scene="5" hidden={step!==5}>
        <div className="qd-update-source"><div className="qd-mini-traces">{[0,1,2,3,4].map(i=><i key={i}/>)}</div><span>同一批真实轨迹</span></div>
        <div className="qd-fork"><span/><span/></div>
        <div className="qd-learning-lanes"><div className="qd-learning-lane"><span className="text-small text-muted">原 Search-G1 奖励</span><div className="qd-actor-art"><i/><i/><i/><i/></div><h3>Actor / PPO</h3><span className="text-small text-muted">原奖励与三项防护</span></div><div className="qd-learning-lane"><span className="text-small text-muted">严格 EM → K / mean / std</span><div className="qd-probe-art"><i/><i/><i/><i/></div><h3>选题 head</h3><span className="text-small text-muted">每个成功 iteration 更新</span></div></div>
        <div className="qd-return-flow"><span>新 actor</span><span>+</span><span>新 head</span><span className="qd-arrow">→</span><span>下一轮选题</span></div>
        <div className="qd-old-probe-note text-small text-muted">原奖励 probe 独立保留 50 轮刷新周期</div>
      </section>
    </main>
    <div className="qd-detail text-small text-muted" data-field="stage-detail" aria-live="polite">{DATA.details[step]}</div>
    <footer className="qd-footer"><span className="text-small text-muted">完整轨迹生成预算</span><span className="text-small">原上限 <span className="tabular-nums">10,240</span> <span className="qd-arrow">→</span> 本方案 <strong className="tabular-nums">2,560</strong></span><span className="text-small text-muted">不等同于总耗时缩短 4 倍</span></footer>
  </div>;
}
