export const questionDiversityStyles = String.raw`
#qd-probe-20261013 { --qd-ink:var(--foreground); --qd-soft:var(--muted); --qd-focus:var(--viz-series-1); --qd-explore:var(--viz-series-2); color:var(--foreground); font-family:inherit; font-size:var(--font-size-base); width:100%; box-sizing:border-box; }
#qd-probe-20261013 * { box-sizing:border-box; }
#qd-probe-20261013 h2,#qd-probe-20261013 h3 { margin:0; font-weight:500; }
#qd-probe-20261013 strong { font-weight:500; }
#qd-probe-20261013 [hidden] { display:none!important; }
#qd-probe-20261013 .qd-heading { display:flex; padding:4px 96px 18px 0; }
#qd-probe-20261013 .qd-controls { margin-bottom:18px; }
#qd-probe-20261013 .qd-controls .form-label { display:flex; align-items:center; gap:8px; margin-left:auto; }
#qd-probe-20261013 .qd-steps { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:6px; padding-bottom:12px; border-bottom:1px solid var(--border); }
#qd-probe-20261013 .qd-index { display:block; opacity:.55; }
#qd-probe-20261013 .qd-stage-heading { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:20px 0 12px; }
#qd-probe-20261013 .qd-main { min-height:270px; display:grid; align-items:center; }
#qd-probe-20261013 .qd-scene { width:100%; padding:10px 0; }
#qd-probe-20261013 .qd-pool-label { display:flex; justify-content:space-between; margin-bottom:16px; }
#qd-probe-20261013 .qd-question-cloud { display:grid; grid-template-columns:repeat(16,minmax(0,1fr)); gap:6px; }
#qd-probe-20261013 .qd-question { height:23px; background:color-mix(in srgb,var(--foreground) 10%,transparent); border-radius:3px; position:relative; transform:translateY(0); }
#qd-probe-20261013 .qd-question::after { content:""; position:absolute; top:8px; bottom:8px; left:8px; right:8px; background:color-mix(in srgb,var(--foreground) 16%,transparent); }
#qd-probe-20261013 .qd-flow-line { display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:12px; padding:22px 0 0; color:var(--muted-foreground); }
#qd-probe-20261013 .qd-encoder-layout { display:grid; grid-template-columns:1fr 50px 1.35fr 50px 1fr; gap:14px; align-items:center; }
#qd-probe-20261013 .qd-input-stack { display:grid; gap:9px; }
#qd-probe-20261013 .qd-input-stack div { padding:8px 12px; background:color-mix(in srgb,var(--foreground) 7%,transparent); border-radius:3px; }
#qd-probe-20261013 .qd-stream { position:relative; height:2px; background:var(--border); }
#qd-probe-20261013 .qd-stream span { position:absolute; left:0; top:-3px; width:8px; height:8px; border-radius:50%; background:var(--qd-focus); opacity:0; }
#qd-probe-20261013 .qd-encoder { display:grid; justify-items:center; gap:12px; text-align:center; }
#qd-probe-20261013 .qd-encoder-mark { width:72px; height:72px; border-radius:50%; background:color-mix(in srgb,var(--qd-focus) 14%,transparent); position:relative; }
#qd-probe-20261013 .qd-encoder-mark::before,#qd-probe-20261013 .qd-encoder-mark::after { content:""; position:absolute; inset:20px; background:var(--qd-focus); border-radius:3px; opacity:.75; transform:rotate(45deg); }
#qd-probe-20261013 .qd-encoder-mark::after { inset:29px; background:var(--background); }
#qd-probe-20261013 .qd-vectors { display:grid; gap:12px; justify-items:center; }
#qd-probe-20261013 .qd-vector { display:flex; gap:4px; height:22px; align-items:center; }
#qd-probe-20261013 .qd-vector i { display:block; width:7px; background:var(--qd-focus); opacity:.65; border-radius:1px; }
#qd-probe-20261013 .qd-distribution-heading { display:flex; justify-content:space-between; gap:12px; margin-bottom:8px; }
#qd-probe-20261013 .qd-distribution { width:100%; min-height:170px; }
#qd-probe-20261013 .qd-distribution svg { display:block; width:100%; }
#qd-probe-20261013 .qd-distribution svg text { fill:var(--foreground); font-family:inherit; font-size:12px; font-weight:400; }
#qd-probe-20261013 .qd-moments { display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:18px; padding-top:18px; }
#qd-probe-20261013 .qd-moments strong { margin-left:6px; }
#qd-probe-20261013 .qd-selection-strip { display:flex; height:44px; gap:3px; margin:8px 0 14px; }
#qd-probe-20261013 .qd-exploration-fill { background:var(--qd-explore); width:12.5%; transition:width .55s cubic-bezier(.22,1,.36,1); }
#qd-probe-20261013 .qd-top-fill { background:var(--qd-focus); width:87.5%; transition:width .55s cubic-bezier(.22,1,.36,1); }
#qd-probe-20261013 .qd-selection-labels { display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; }
#qd-probe-20261013 .qd-legend-dot { width:9px; height:9px; display:inline-block; border-radius:50%; margin-right:7px; }
#qd-probe-20261013 .qd-explore-color { background:var(--qd-explore); }
#qd-probe-20261013 .qd-focus-color { background:var(--qd-focus); }
#qd-probe-20261013 .qd-selected-cloud { display:grid; grid-template-columns:repeat(16,minmax(0,1fr)); gap:5px; padding-top:16px; }
#qd-probe-20261013 .qd-selected-cloud .qd-question { height:16px; }
#qd-probe-20261013 .qd-selected-cloud .qd-question::after { display:none; }
#qd-probe-20261013 .qd-rollout-heading { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; }
#qd-probe-20261013 .qd-rollout-heading .form-label { display:flex; align-items:center; gap:8px; }
#qd-probe-20261013 .qd-rollout { display:grid; grid-template-columns:85px 1fr; align-items:center; gap:25px; padding-top:18px; }
#qd-probe-20261013 .qd-rollout-q { padding:16px 8px; text-align:center; background:color-mix(in srgb,var(--foreground) 8%,transparent); border-radius:3px; }
#qd-probe-20261013 .qd-trajectories { display:grid; gap:10px; }
#qd-probe-20261013 .qd-trajectory { display:grid; grid-template-columns:1fr 30px 48px; align-items:center; gap:10px; }
#qd-probe-20261013 .qd-trace { display:flex; align-items:center; gap:5px; height:21px; position:relative; }
#qd-probe-20261013 .qd-trace::before { content:""; position:absolute; inset:50% 0 auto; border-top:1px solid var(--border); }
#qd-probe-20261013 .qd-trace i { height:8px; border-radius:1px; background:color-mix(in srgb,var(--qd-focus) 58%,var(--background)); z-index:1; width:18%; transform-origin:left; }
#qd-probe-20261013 .qd-em { width:26px; height:26px; border-radius:50%; display:grid; place-items:center; color:var(--foreground); background:color-mix(in srgb,var(--foreground) 10%,transparent); }
#qd-probe-20261013 .qd-em.is-correct { background:color-mix(in srgb,var(--qd-focus) 23%,transparent); }
#qd-probe-20261013 .qd-feedback-source { text-align:center; }
#qd-probe-20261013 .qd-update-lanes { display:grid; grid-template-columns:1fr 1fr; gap:28px; padding:0 10px; }
#qd-probe-20261013 .qd-update-lane { display:grid; gap:12px; justify-items:center; text-align:center; }
#qd-probe-20261013 .qd-lane-link { height:28px; width:1px; background:var(--border); }
#qd-probe-20261013 .qd-update-mark { width:66px; height:22px; border-radius:3px; background:var(--qd-focus); opacity:.85; }
#qd-probe-20261013 .qd-actor-mark { background:color-mix(in srgb,var(--foreground) 22%,transparent); }
#qd-probe-20261013 .qd-next-loop { display:flex; justify-content:center; align-items:center; gap:10px; padding-top:20px; }
#qd-probe-20261013 .qd-loop-mark { color:var(--muted-foreground); }
#qd-probe-20261013 .qd-reward-note { text-align:center; padding-top:8px; }
#qd-probe-20261013 .qd-detail { min-height:32px; display:flex; align-items:center; padding:10px 0 15px; color:var(--muted-foreground); border-bottom:1px solid var(--border); }
#qd-probe-20261013 .qd-budget { padding-top:17px; display:grid; gap:10px; }
#qd-probe-20261013 .qd-budget-row { display:grid; grid-template-columns:150px 1fr 92px; gap:12px; align-items:center; }
#qd-probe-20261013 .qd-budget-track { height:11px; background:color-mix(in srgb,var(--foreground) 5%,transparent); }
#qd-probe-20261013 .qd-budget-old { width:100%; height:100%; background:color-mix(in srgb,var(--foreground) 25%,transparent); }
#qd-probe-20261013 .qd-budget-new { width:25%; height:100%; background:var(--qd-focus); }
#qd-probe-20261013 .qd-budget-row>span:last-child { text-align:right; }
#qd-probe-20261013 .qd-enter .qd-question { animation:qd-rise .7s cubic-bezier(.22,1,.36,1) both; animation-delay:calc(var(--i,0)*7ms); }
#qd-probe-20261013 .qd-enter .qd-stream span { animation:qd-stream 1.3s ease both; }
#qd-probe-20261013 .qd-enter .qd-stream span:nth-child(2) { animation-delay:.22s; }
#qd-probe-20261013 .qd-enter .qd-stream span:nth-child(3) { animation-delay:.44s; }
#qd-probe-20261013 .qd-enter .qd-trace i { animation:qd-grow .7s cubic-bezier(.22,1,.36,1) both; animation-delay:calc(var(--i,0)*150ms); }
#qd-probe-20261013 .qd-enter .qd-update-mark { animation:qd-rise .8s cubic-bezier(.22,1,.36,1) both; }
@keyframes qd-rise { from { transform:translateY(18px); } to { transform:translateY(0); } }
@keyframes qd-stream { 0% { transform:translateX(0); opacity:0; } 20% { opacity:1; } 80% { opacity:1; } 100% { transform:translateX(44px); opacity:0; } }
@keyframes qd-grow { from { transform:scaleX(.02); } to { transform:scaleX(1); } }
@media (max-width:620px) { #qd-probe-20261013 .qd-steps { grid-template-columns:repeat(3,minmax(0,1fr)); } #qd-probe-20261013 .qd-controls .form-label { margin-left:0; } #qd-probe-20261013 .qd-encoder-layout { grid-template-columns:1fr 20px 1.3fr; } #qd-probe-20261013 .qd-encoder-layout>.qd-stream:nth-of-type(4),#qd-probe-20261013 .qd-vectors { display:none; } #qd-probe-20261013 .qd-question-cloud { grid-template-columns:repeat(10,minmax(0,1fr)); } #qd-probe-20261013 .qd-selected-cloud { grid-template-columns:repeat(8,minmax(0,1fr)); } #qd-probe-20261013 .qd-budget-row { grid-template-columns:128px 1fr 85px; gap:8px; } #qd-probe-20261013 .qd-update-lanes { gap:12px; padding:0; } #qd-probe-20261013 .qd-heading { padding-right:90px; } }
@media (max-width:380px) { #qd-probe-20261013 .qd-budget-row { grid-template-columns:1fr 80px; } #qd-probe-20261013 .qd-budget-track { grid-row:2; grid-column:1/-1; } #qd-probe-20261013 .qd-rollout { grid-template-columns:60px 1fr; gap:12px; } #qd-probe-20261013 .qd-moments { gap:12px; } }
@media (prefers-reduced-motion:reduce) { #qd-probe-20261013 *,#qd-probe-20261013 *::before,#qd-probe-20261013 *::after { animation:none!important; transition:none!important; } }
`;
