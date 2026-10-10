"""Emit a standalone React Canvas containing only compact aggregate data."""
import json
from pathlib import Path

p = Path(__file__).parent
data = json.loads((p/'chart-data.json').read_text())
raw=json.loads((p/'raw-reward-distributions.json').read_text())
data['raw']={'common61':raw['calibration_common61'],
             'archive':[[r['iteration'],r['p_reward_1']] for r in raw['actual_training_archive'] if r['iteration']>0],
             'g1_correct':[[r['iteration'],r['correct_fraction_console']] for r in raw['actual_training_g1'] if not r['protocol_invalid_present']]}
source = r'''import React, {useState} from "react";
const DATA = __DATA__;
const h=React.createElement;
const COLORS=["var(--viz-series-1,#2563eb)","var(--viz-series-2,#ea580c)","var(--viz-series-3,#0f766e)"];
const card={border:"1px solid var(--border,#dce0e5)",borderRadius:12,padding:18,background:"var(--card,#fff)"};
const muted={color:"var(--muted-foreground,#64748b)",fontSize:13,lineHeight:1.6};
function Plot({lines,title,maxX=150,maxY=1,hist=false,cursor}) {
  const W=620,H=245,L=53,R=20,T=15,B=36;
  const X=x=>L+x/maxX*(W-L-R), Y=y=>T+(maxY-y)/maxY*(H-T-B);
  const draw=[];
  for(let i=0;i<=4;i++) {const v=maxY*i/4;draw.push(h("line",{key:"g"+i,x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:"var(--border,#dce0e5)"}),h("text",{key:"y"+i,x:L-8,y:Y(v)+4,textAnchor:"end",fontSize:11,fill:"currentColor"},hist?(v*100).toFixed(0)+"%":v.toFixed(2)));}
  for(let i=0;i<=4;i++) draw.push(h("text",{key:"x"+i,x:X(maxX*i/4),y:H-12,textAnchor:"middle",fontSize:11,fill:"currentColor"},hist?(maxX*i/4).toFixed(2):Math.round(maxX*i/4)));
  if(!hist) for(const x of [51,101])draw.push(h("line",{key:"v"+x,x1:X(x),x2:X(x),y1:T,y2:H-B,stroke:"#64748b",strokeDasharray:"2 4",opacity:.6}));
  lines.forEach((line,j)=>{const ps=line.points;const path=ps.map((q,i)=>(i?"L":"M")+X(q[0])+","+Y(q[1])).join(" ");if(line.markOnly)ps.forEach((q,i)=>draw.push(h("circle",{key:`s${j}-${i}`,cx:X(q[0]),cy:Y(q[1]),r:2.5,fill:COLORS[line.color??j]})));else draw.push(h("path",{key:"p"+j,d:path,fill:"none",stroke:COLORS[line.color??j],strokeWidth:2.5,strokeDasharray:line.dash}));if(!hist&&cursor){const q=ps.find(q=>q[0]===cursor);if(q)draw.push(h("circle",{key:"dot"+j,cx:X(q[0]),cy:Y(q[1]),r:4,fill:COLORS[line.color??j]}));}});
  return h("section",{style:card},h("div",{style:{fontWeight:650,marginBottom:12}},title),h("svg",{viewBox:`0 0 ${W} ${H}`,style:{width:"100%",display:"block"},role:"img","aria-label":title},...draw),h("div",{style:{display:"flex",gap:15,flexWrap:"wrap",fontSize:12}},...lines.map((l,j)=>h("span",{key:l.label,style:{color:COLORS[l.color??j]}},"● "+l.label))));
}
function histogram(key) {return DATA.rounds.map((r,j)=>{const d=r[key],points=[];d.histogram_counts.forEach((n,i)=>{points.push([d.histogram_edges[i],n/d.count],[d.histogram_edges[i+1],n/d.count]);});return {label:`第${r.round}轮，n=${d.count}`,points,color:j,dash:[undefined,"8 4","2 4"][j]};});}
function smooth(key,on){const a=DATA.series[key];return !on?a:a.slice(4).map((q,i)=>[q[0],a.slice(i,i+5).reduce((s,v)=>s+v[1],0)/5]);}
function RawBars(){
  const W=760,H=255,L=45,B=65,T=12,gh=H-B-T,bw=25;
  const bars=[];
  for(let i=0;i<=3;i++){const y=T+gh*(1-i/3);bars.push(h("line",{key:"g"+i,x1:L,x2:W-15,y1:y,y2:y,stroke:"var(--border,#dce0e5)"}),h("text",{key:"y"+i,x:L-5,y:y+4,textAnchor:"end",fontSize:10,fill:"currentColor"},i*25+"%"));}
  DATA.raw.common61[0].categories.forEach((label,i)=>{const x=L+(i+.5)*(W-L-15)/8;bars.push(h("text",{key:"l"+i,x,y:H-40,textAnchor:"middle",fontSize:10,fill:"currentColor"},label));DATA.raw.common61.forEach((r,j)=>{const v=r.counts[i]/305,y=T+gh*(1-v/.75);bars.push(h("rect",{key:`b${i}-${j}`,x:x+(j-1)*bw-bw/2,y,width:bw-2,height:T+gh-y,fill:COLORS[j]}));});});
  return h("section",{style:card},h("div",{style:{fontWeight:650}},"校准重构：同61道完整检索题，归一化前R直方图"),h("svg",{viewBox:`0 0 ${W} ${H}`,style:{width:"100%"},role:"img","aria-label":"同61校准题原始reward直方图；不是历史TRAIN"},...bars),h("div",{style:{display:"flex",gap:20}},...DATA.raw.common61.map((r,j)=>h("span",{key:r.round,style:{color:COLORS[j],fontSize:13}},`● ${r.round}轮：均值${r.raw_mean.toFixed(3)}，正确${Math.round(r.correct_fraction*305)}/305`))),h("p",{style:muted},"这是跨0/50/100三个保存点均完整合法检索的61题筛选子集，每点重新生成305条。匹配拟合head的CPU重构；不是每轮训练2560条的历史奖励。"));
}
export default function ProbeEvolution(){
  const [sm,setSm]=useState(true),[step,setStep]=useState(100);
  const row=DATA.series.g1_reward.find(x=>x[0]===step),em=DATA.series.g1_em.find(x=>x[0]===step);
  const cols=["轮次","b 均值 / 中位数","d 均值 / 中位数","d(1-b) 均值","实际闭卷答对"];
  return h("main",{style:{fontFamily:"system-ui,sans-serif",color:"var(--foreground,#18212f)",padding:24,maxWidth:1400,margin:"auto"}},
    h("h1",{style:{fontSize:26,margin:"0 0 10px"}},"Search-G1：0–150轮归一化前奖励与探针"),
    h("p",{style:muted},"固定快照 "+DATA.stamp+"。上图是同300题校准样本的拟合内预测；不是全训练过程逐条 b/d。每50轮同时替换 encoder 和 head，新轨迹数量也变化。"),
    h("div",{style:{...card,marginBottom:16}},h("strong",null,"历史细分分布未保存，不能从均值逆推"),h("p",{style:muted},"该G1 run实际512×5=2560条/轮，141个完整轮。仅存全局reward均值/min/max及行为概率，未存每条原始R。72轮可确认无−0.1非法结果，其有效错答都为0.05，正确落[0.98,1.6]；其它轮错误/非法质量拆分不可恢复。Archive原R为0/1，日志均值可恢复边际占比，但不能恢复同题五条排列。")),
    h("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(min(100%,400px),1fr))",gap:16,marginBottom:16}},
      h(Plot,{title:"Archive实际TRAIN：R=1占比（其余R=0）",maxY:1,lines:[{label:"归一化前0/1质量；三位小数精度",color:2,points:DATA.raw.archive}]}),
      h(Plot,{title:"G1实际TRAIN：72轮可恢复的粗分布",maxY:1,lines:[{label:"正确区间[0.98,1.6]质量",color:0,markOnly:true,points:DATA.raw.g1_correct},{label:"R=0.05质量",color:1,markOnly:true,points:DATA.raw.g1_correct.map(q=>[q[0],1-q[1]])}]})),
    h(RawBars),
    h("p",{style:muted},"重点：共同61题50→100轮正确数均121/305，但正确条的平均原始reward从1.3663升至1.3877。同题组内R平均标准差0→50→100为0.1761→0.1559→0.2010，不是单调扩大。"),
    h("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(290px,1fr))",gap:16}},
      h(Plot,{lines:histogram("b"),title:"b：闭卷能力读出分数",maxX:1,maxY:.8,hist:true}),
      h(Plot,{lines:histogram("d"),title:"d：删除证据后的答案敏感度",maxX:1,maxY:.8,hist:true}),
      h(Plot,{lines:histogram("ground"),title:"g=d(1-b)：搜索加成的分数",maxX:1,maxY:.8,hist:true})),
    h("div",{style:{...card,overflowX:"auto",marginTop:16}},h("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14}},
      h("thead",null,h("tr",null,...cols.map(c=>h("th",{key:c,style:{textAlign:"left",padding:10}},c)))),
      h("tbody",null,...DATA.rounds.map(r=>h("tr",{key:r.round},...[
        r.round,`${r.b.mean.toFixed(3)} / ${r.b.quantiles[3].toFixed(3)}`,`${r.d.mean.toFixed(3)} / ${r.d.quantiles[3].toFixed(3)}`,r.ground.mean.toFixed(3),`${r.cohort.closedbook_correct_question_labels}/300`
      ].map((v,i)=>h("td",{key:i,style:{borderTop:"1px solid var(--border,#dce0e5)",padding:10}},v)))))),
      h("p",{style:muted},"b 预测下降，实际闭卷答对36→40；不能把换版后的 b 当稳定能力刻度。d 不检查答案正确性：第100轮正确/错误轨迹 g 均值分别为0.610/0.801，错误分支仍没有搜索加成。")),
    h("div",{style:{display:"flex",gap:20,alignItems:"center",flexWrap:"wrap",margin:"22px 0"}},
      h("label",null,h("input",{type:"checkbox",checked:sm,onChange:e=>setSm(e.target.checked)})," 5轮移动均值"),
      h("label",null,"查看G1轮次 ",h("input",{type:"range",min:1,max:140,value:step,onChange:e=>setStep(Number(e.target.value))})," "+step),
      h("span",{style:muted},row?`该轮原值：reward=${row[1].toFixed(4)}，TRAIN EM≈${(em[1]*100).toFixed(1)}%`:"无观测")),
    h("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(min(100%,400px),1fr))",gap:16}},
      h(Plot,{cursor:step,title:"训练奖励和EM：刻度不同",maxY:.8,lines:[{label:"G1 shaped reward",color:0,points:smooth("g1_reward",sm)},{label:"G1 TRAIN EM",color:1,dash:"8 4",points:smooth("g1_em",sm)},{label:"Archive reward / EM",color:2,points:smooth("archive_reward",sm)}]}),
      h(Plot,{cursor:step,title:"实际搜索次数（各运行日志口径）",maxY:4,lines:[{label:"G1",color:0,points:smooth("g1_search",sm)},{label:"Archive",color:2,points:smooth("archive_search",sm)}]}),
      h(Plot,{cursor:step,title:"策略熵：两组均逐渐集中",maxY:1,lines:[{label:"G1",color:0,points:smooth("g1_entropy",sm)},{label:"Archive",color:2,points:smooth("archive_entropy",sm)}]}),
      h(Plot,{title:"各自原生512验证：不能直接归因算法",maxY:.55,lines:[{label:"G1：202→205/512",color:0,points:DATA.fixed512.g1},{label:"Archive：162→229→230/512",color:2,points:DATA.fixed512.archive}]})),
    h("p",{style:muted},"只看0–150轮；G1尚未产出的轮次留空。灰线51/101：新探针开始用于训练。Archive没有 b/d 探针。两组的KL、损失归约、PPO保护、硬件与评测条件并不完全相同。trust@correct≈0.7是独立时序监控分数，不是d。")
  );
}
'''
(p/'probe-reward-evolution.canvas.tsx').write_text(source.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))))
