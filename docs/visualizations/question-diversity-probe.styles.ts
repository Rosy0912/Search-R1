export const questionDiversityStyles = String.raw`
#qd-probe-20261013 { --qd-focus:var(--viz-series-1); --qd-explore:var(--viz-series-2); --qd-neutral:color-mix(in srgb,var(--foreground) 12%,transparent); width:100%; color:var(--foreground); font-family:inherit; }
#qd-probe-20261013 * { box-sizing:border-box; }
#qd-probe-20261013 [hidden] { display:none!important; }
#qd-probe-20261013 h1,#qd-probe-20261013 h2,#qd-probe-20261013 h3 { margin:0; font-weight:500; }
#qd-probe-20261013 strong { font-weight:500; }
#qd-probe-20261013 .qd-header { display:grid; gap:10px; padding:4px 92px 24px 0; }
#qd-probe-20261013 .qd-eyebrow { letter-spacing:.12em; }
#qd-probe-20261013 .qd-player { display:flex; justify-content:space-between; align-items:center; gap:12px; padding-bottom:14px; }
#qd-probe-20261013 .qd-navigation { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px; padding-bottom:20px; position:relative; }
#qd-probe-20261013 .qd-navigation::before { content:""; position:absolute; height:1px; background:var(--border); left:8%; right:8%; top:19px; z-index:0; }
#qd-probe-20261013 .qd-nav-item { display:grid; gap:6px; justify-items:center; position:relative; z-index:1; color:var(--muted-foreground); }
#qd-probe-20261013 .qd-nav-item.is-active { color:var(--foreground); }
#qd-probe-20261013 .qd-stage-heading { display:flex; align-items:center; gap:10px; padding:18px 0 20px; border-top:1px solid var(--border); }
#qd-probe-20261013 .qd-stage-rule { width:4px; height:18px; background:var(--foreground); border-radius:2px; flex:none; }
#qd-probe-20261013 .qd-main { min-height:320px; display:grid; align-items:center; }
#qd-probe-20261013 .qd-scene { padding:0 2px 12px; width:100%; }
#qd-probe-20261013 .qd-candidate-top,#qd-probe-20261013 .qd-select-head { display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:20px; }
#qd-probe-20261013 .qd-candidate-top>div,#qd-probe-20261013 .qd-select-head>div { display:grid; gap:5px; }
#qd-probe-20261013 .qd-pool { display:grid; grid-template-columns:repeat(16,minmax(0,1fr)); gap:8px; }
#qd-probe-20261013 .qd-ticket { position:relative; display:grid; align-content:center; gap:3px; height:31px; padding:6px; background:var(--qd-neutral); border-radius:3px; transform:translate(0,0); }
#qd-probe-20261013 .qd-ticket i { display:block; height:2px; background:color-mix(in srgb,var(--foreground) 22%,transparent); border-radius:1px; }
#qd-probe-20261013 .qd-ticket i:nth-child(2) { width:78%; }
#qd-probe-20261013 .qd-ticket i:nth-child(3) { width:48%; }
#qd-probe-20261013 .qd-pool-selected { padding-top:20px; }
#qd-probe-20261013 .qd-ticket.is-focus { background:color-mix(in srgb,var(--qd-focus) 18%,transparent); }
#qd-probe-20261013 .qd-ticket.is-explore { background:color-mix(in srgb,var(--qd-explore) 20%,transparent); }
#qd-probe-20261013 .qd-ticket.is-focus i { background:color-mix(in srgb,var(--qd-focus) 66%,transparent); }
#qd-probe-20261013 .qd-ticket.is-explore i { background:color-mix(in srgb,var(--qd-explore) 66%,transparent); }
#qd-probe-20261013 .qd-flow-caption { display:flex; align-items:center; justify-content:center; gap:16px; padding-top:24px; }
#qd-probe-20261013 .qd-arrow { color:var(--muted-foreground); }
#qd-probe-20261013 .qd-mini-paper { width:23px; height:28px; display:grid; align-content:center; gap:3px; padding:5px; background:var(--qd-neutral); border-radius:2px; }
#qd-probe-20261013 .qd-mini-paper i { height:2px; background:color-mix(in srgb,var(--foreground) 36%,transparent); }
#qd-probe-20261013 .qd-cache-flow { display:grid; grid-template-columns:1fr 42px 1.15fr 42px 1.1fr; align-items:center; gap:16px; padding:26px 0 24px; }
#qd-probe-20261013 .qd-flow-unit { display:grid; justify-items:center; text-align:center; gap:12px; }
#qd-probe-20261013 .qd-paper-stack { position:relative; width:82px; height:92px; margin-left:6px; margin-bottom:8px; }
#qd-probe-20261013 .qd-paper-stack div { position:absolute; width:56px; height:72px; padding:16px 10px; display:grid; align-content:start; gap:7px; background:color-mix(in srgb,var(--foreground) 10%,var(--background)); border-radius:4px; }
#qd-probe-20261013 .qd-paper-stack div:first-child { transform:translate(0,0) rotate(-8deg); }
#qd-probe-20261013 .qd-paper-stack div:nth-child(2) { transform:translate(12px,8px) rotate(2deg); }
#qd-probe-20261013 .qd-paper-stack div:nth-child(3) { transform:translate(22px,17px) rotate(9deg); }
#qd-probe-20261013 .qd-paper-stack i { height:3px; border-radius:2px; background:color-mix(in srgb,var(--foreground) 27%,transparent); }
#qd-probe-20261013 .qd-paper-stack i:nth-child(2) { width:80%; }
#qd-probe-20261013 .qd-encoder-art { width:100px; height:100px; display:grid; grid-template-columns:repeat(3,1fr); padding:21px; gap:7px; background:color-mix(in srgb,var(--qd-focus) 9%,transparent); border-radius:20px; }
#qd-probe-20261013 .qd-encoder-art span { border-radius:3px; background:color-mix(in srgb,var(--qd-focus) 35%,transparent); }
#qd-probe-20261013 .qd-encoder-art span:nth-child(5) { background:var(--qd-focus); }
#qd-probe-20261013 .qd-transport { height:1px; background:var(--border); position:relative; }
#qd-probe-20261013 .qd-transport span { position:absolute; left:0; top:-3px; width:7px; height:7px; border-radius:50%; background:var(--qd-focus); opacity:0; }
#qd-probe-20261013 .qd-vector-cache { display:grid; gap:6px; height:100px; align-content:center; }
#qd-probe-20261013 .qd-vector-cache>div { display:flex; gap:4px; }
#qd-probe-20261013 .qd-vector-cache i { display:block; width:8px; height:14px; border-radius:2px; background:var(--qd-focus); }
#qd-probe-20261013 .qd-cache-read { display:flex; justify-content:center; align-items:center; gap:12px; padding-top:20px; }
#qd-probe-20261013 .qd-cache-line { height:1px; width:35px; background:var(--border); }
#qd-probe-20261013 .qd-prediction-context { display:flex; justify-content:space-between; align-items:center; gap:15px; padding-bottom:15px; }
#qd-probe-20261013 .qd-chart-label { padding-bottom:10px; }
#qd-probe-20261013 .qd-bars { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:18px; padding:0 18px; }
#qd-probe-20261013 .qd-bar-column { display:grid; gap:10px; justify-items:center; }
#qd-probe-20261013 .qd-bar-space { height:144px; width:100%; position:relative; display:flex; justify-content:center; align-items:flex-end; }
#qd-probe-20261013 .qd-bar-measure { width:65%; max-width:54px; display:flex; flex-direction:column; position:relative; }
#qd-probe-20261013 .qd-bar-measure>span { position:absolute; top:-23px; width:100%; text-align:center; color:var(--foreground); }
#qd-probe-20261013 .qd-bar { flex:1; background:color-mix(in srgb,var(--qd-focus) 78%,transparent); border-radius:3px 3px 0 0; transform-origin:bottom; }
#qd-probe-20261013 .qd-bar-column.is-endpoint .qd-bar { background:var(--qd-neutral); }
#qd-probe-20261013 .qd-chart-axis { text-align:right; padding:12px 0 0; }
#qd-probe-20261013 .qd-moments { display:grid; grid-template-columns:auto auto 1fr; gap:38px; align-items:end; margin-top:20px; padding-top:17px; border-top:1px solid var(--border); }
#qd-probe-20261013 .qd-moments>div { display:grid; gap:5px; }
#qd-probe-20261013 .qd-moment-note { text-align:right; padding-bottom:3px; }
#qd-probe-20261013 .qd-select-setting { display:grid; gap:4px; min-width:118px; }
#qd-probe-20261013 .qd-rollout-setting { display:grid; grid-template-columns:max-content 110px; align-items:center; gap:10px; white-space:nowrap; }
#qd-probe-20261013 .qd-selection-strip { display:flex; gap:3px; height:11px; }
#qd-probe-20261013 .qd-exploration-fill { background:var(--qd-explore); border-radius:2px; transition:width .7s cubic-bezier(.22,1,.36,1); }
#qd-probe-20261013 .qd-top-fill { background:var(--qd-focus); border-radius:2px; transition:width .7s cubic-bezier(.22,1,.36,1); }
#qd-probe-20261013 .qd-selection-labels { display:flex; justify-content:space-between; gap:14px; padding-top:14px; }
#qd-probe-20261013 .qd-selection-labels strong { margin-left:5px; }
#qd-probe-20261013 .qd-legend-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; }
#qd-probe-20261013 .qd-legend-dot.is-focus { background:var(--qd-focus); }
#qd-probe-20261013 .qd-legend-dot.is-explore { background:var(--qd-explore); }
#qd-probe-20261013 .qd-selection-order { display:flex; align-items:center; justify-content:center; gap:14px; padding-top:22px; }
#qd-probe-20261013 .qd-rollout-heading { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:16px; }
#qd-probe-20261013 .qd-rollout { display:grid; grid-template-columns:90px 1fr; gap:50px; align-items:center; padding-top:24px; }
#qd-probe-20261013 .qd-q-origin { display:grid; gap:10px; justify-items:center; position:relative; text-align:center; }
#qd-probe-20261013 .qd-q-origin h2 { display:grid; place-items:center; width:58px; height:58px; border-radius:14px; background:var(--qd-neutral); }
#qd-probe-20261013 .qd-q-origin::after { content:""; position:absolute; right:-30px; top:46px; width:28px; height:1px; background:var(--border); }
#qd-probe-20261013 .qd-traces { display:grid; gap:13px; position:relative; }
#qd-probe-20261013 .qd-traces::before { content:""; position:absolute; left:-21px; top:42px; bottom:16px; border-left:1px solid var(--border); border-radius:8px; }
#qd-probe-20261013 .qd-trace-labels,#qd-probe-20261013 .qd-trace-row { display:grid; grid-template-columns:1fr 1fr 1fr 28px; gap:18px; align-items:center; }
#qd-probe-20261013 .qd-trace-row { min-height:28px; position:relative; }
#qd-probe-20261013 .qd-trace-row::before { content:""; position:absolute; width:18px; left:-21px; top:14px; height:1px; background:var(--border); }
#qd-probe-20261013 .qd-trace-row>div { height:27px; position:relative; display:grid; align-content:center; gap:4px; padding:5px 7px; border-radius:3px; background:color-mix(in srgb,var(--qd-focus) 11%,transparent); transform-origin:left; }
#qd-probe-20261013 .qd-trace-row>div::after { content:""; position:absolute; top:13px; right:-18px; width:18px; height:1px; background:var(--border); }
#qd-probe-20261013 .qd-trace-row i { height:3px; border-radius:2px; background:color-mix(in srgb,var(--qd-focus) 44%,transparent); width:86%; }
#qd-probe-20261013 .qd-trace-search span { position:absolute; right:7px; top:7px; width:4px; height:4px; border-radius:50%; background:var(--qd-focus); }
#qd-probe-20261013 .qd-em { display:grid; place-items:center; width:28px; height:28px; border-radius:50%; background:var(--qd-neutral); font-variant-numeric:tabular-nums; }
#qd-probe-20261013 .qd-em.is-correct { background:color-mix(in srgb,var(--qd-focus) 24%,transparent); }
#qd-probe-20261013 .qd-update-source { display:grid; justify-items:center; gap:12px; }
#qd-probe-20261013 .qd-mini-traces { display:flex; gap:5px; height:24px; align-items:center; }
#qd-probe-20261013 .qd-mini-traces i { height:5px; width:40px; border-radius:2px; background:color-mix(in srgb,var(--qd-focus) 45%,transparent); }
#qd-probe-20261013 .qd-fork { display:flex; justify-content:center; height:40px; }
#qd-probe-20261013 .qd-fork span { display:block; width:25%; height:24px; margin-top:13px; border-top:1px solid var(--border); }
#qd-probe-20261013 .qd-fork span:first-child { border-left:1px solid var(--border); border-radius:9px 0 0; }
#qd-probe-20261013 .qd-fork span:last-child { border-right:1px solid var(--border); border-radius:0 9px 0 0; }
#qd-probe-20261013 .qd-learning-lanes { display:grid; grid-template-columns:1fr 1fr; gap:36px; }
#qd-probe-20261013 .qd-learning-lane { display:grid; justify-items:center; gap:10px; text-align:center; }
#qd-probe-20261013 .qd-actor-art,#qd-probe-20261013 .qd-probe-art { display:flex; gap:5px; align-items:end; height:42px; padding-bottom:3px; }
#qd-probe-20261013 .qd-actor-art i,#qd-probe-20261013 .qd-probe-art i { width:12px; height:21px; border-radius:2px; background:color-mix(in srgb,var(--foreground) 24%,transparent); }
#qd-probe-20261013 .qd-actor-art i:nth-child(2),#qd-probe-20261013 .qd-probe-art i:nth-child(2) { height:31px; }
#qd-probe-20261013 .qd-actor-art i:nth-child(3),#qd-probe-20261013 .qd-probe-art i:nth-child(3) { height:26px; }
#qd-probe-20261013 .qd-probe-art i { background:color-mix(in srgb,var(--qd-focus) 58%,transparent); }
#qd-probe-20261013 .qd-return-flow { display:flex; align-items:center; justify-content:center; gap:12px; padding-top:28px; }
#qd-probe-20261013 .qd-old-probe-note { text-align:center; padding-top:12px; }
#qd-probe-20261013 .qd-detail { padding:16px 0 20px; min-height:45px; }
#qd-probe-20261013 .qd-footer { display:flex; flex-wrap:wrap; align-items:center; gap:8px 22px; padding-top:17px; border-top:1px solid var(--border); }
#qd-probe-20261013 .qd-footer>span:last-child { margin-left:auto; }
#qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-ticket { animation:qd-ticket-arrive .85s cubic-bezier(.22,1,.36,1) both; animation-delay:calc(var(--i)*9ms); }
#qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-transport span { animation:qd-vector-pass 1.4s ease both; animation-delay:.2s; }
#qd-probe-20261013[data-animated="true"] .qd-transport span:nth-child(2) { animation-delay:.55s; }
#qd-probe-20261013[data-animated="true"] .qd-transport span:nth-child(3) { animation-delay:.9s; }
#qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-bar { animation:qd-bar-grow .9s cubic-bezier(.22,1,.36,1) both; }
#qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-trace-row>div { animation:qd-bar-grow .7s cubic-bezier(.22,1,.36,1) both; animation-delay:calc(var(--i)*.14s); }
#qd-probe-20261013[data-animated="true"] .qd-trace-row>div:nth-child(2) { animation-delay:calc(var(--i)*.14s + .35s); }
#qd-probe-20261013[data-animated="true"] .qd-trace-row>div:nth-child(3) { animation-delay:calc(var(--i)*.14s + .7s); }
#qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-probe-art i,#qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-actor-art i { animation:qd-ticket-arrive .8s cubic-bezier(.22,1,.36,1) both; }
@keyframes qd-ticket-arrive { from { transform:translateY(15px) scale(.88); } to { transform:translateY(0) scale(1); } }
@keyframes qd-vector-pass { 0% { transform:translateX(0); opacity:0; } 20%,75% { opacity:1; } 100% { transform:translateX(38px); opacity:0; } }
@keyframes qd-vertical-vector-pass { 0% { transform:translateY(0); opacity:0; } 20%,75% { opacity:1; } 100% { transform:translateY(16px); opacity:0; } }
@keyframes qd-bar-grow { from { transform:scaleY(.06); } to { transform:scaleY(1); } }
@media(max-width:620px) {
 #qd-probe-20261013 .qd-header { padding-right:90px; }
 #qd-probe-20261013 .qd-header h1>span { display:block; }
 #qd-probe-20261013[data-animated="true"] .qd-scene:not([hidden]) .qd-transport span { animation-name:qd-vertical-vector-pass; }
 #qd-probe-20261013 .qd-pool { gap:5px; grid-template-columns:repeat(10,minmax(0,1fr)); }
 #qd-probe-20261013 .qd-pool-selected { grid-template-columns:repeat(8,minmax(0,1fr)); }
 #qd-probe-20261013 .qd-ticket { height:25px; padding:4px; }
 #qd-probe-20261013 .qd-cache-flow { grid-template-columns:1fr; padding:6px 0; gap:15px; }
 #qd-probe-20261013 .qd-flow-unit { grid-template-columns:100px 1fr; justify-items:start; text-align:left; align-items:center; gap:4px 17px; }
 #qd-probe-20261013 .qd-flow-unit>:first-child { grid-row:1/3; }
 #qd-probe-20261013 .qd-transport { width:1px; height:20px; margin-left:49px; }
 #qd-probe-20261013 .qd-transport span { left:-3px; top:0; }
 #qd-probe-20261013 .qd-paper-stack { height:90px; margin-bottom:0; }
 #qd-probe-20261013 .qd-cache-read { padding-top:16px; gap:9px; }
 #qd-probe-20261013 .qd-cache-line { display:none; }
 #qd-probe-20261013 .qd-prediction-context { align-items:start; }
 #qd-probe-20261013 .qd-moments { gap:24px; grid-template-columns:auto auto; }
 #qd-probe-20261013 .qd-moment-note { text-align:left; grid-column:1/-1; padding:0; }
 #qd-probe-20261013 .qd-bars { gap:10px; padding:0 5px; }
 #qd-probe-20261013 .qd-candidate-top { align-items:start; }
 #qd-probe-20261013 .qd-candidate-top>span { max-width:132px; text-align:right; }
 #qd-probe-20261013 .qd-select-head { align-items:end; gap:12px; }

 #qd-probe-20261013 .qd-rollout-heading { flex-wrap:wrap; }
 #qd-probe-20261013 .qd-rollout { grid-template-columns:45px 1fr; gap:27px; }
 #qd-probe-20261013 .qd-q-origin h2 { width:42px; height:48px; border-radius:9px; }
 #qd-probe-20261013 .qd-q-origin>span { display:none; }
 #qd-probe-20261013 .qd-q-origin::after { width:13px; right:-15px; top:23px; }
 #qd-probe-20261013 .qd-traces::before { left:-12px; }
 #qd-probe-20261013 .qd-trace-row::before { left:-12px; width:9px; }
 #qd-probe-20261013 .qd-trace-labels,#qd-probe-20261013 .qd-trace-row { gap:9px; grid-template-columns:1fr 1fr 1fr 24px; }
 #qd-probe-20261013 .qd-trace-row>div { padding:5px; }
 #qd-probe-20261013 .qd-trace-row>div::after { right:-9px; width:9px; }
 #qd-probe-20261013 .qd-em { width:24px; height:24px; }
 #qd-probe-20261013 .qd-learning-lanes { gap:18px; }
 #qd-probe-20261013 .qd-return-flow { flex-wrap:wrap; gap:10px; }
 #qd-probe-20261013 .qd-footer>span:last-child { margin-left:0; width:100%; }
}
@media(max-width:380px) {
 #qd-probe-20261013 .qd-navigation { gap:3px; }
 #qd-probe-20261013 .qd-header { gap:8px; }
 #qd-probe-20261013 .qd-eyebrow { letter-spacing:.06em; }
 #qd-probe-20261013 .qd-flow-caption { gap:10px; }
 #qd-probe-20261013 .qd-mini-paper { display:none; }
 #qd-probe-20261013 .qd-selection-labels { gap:10px; flex-wrap:wrap; }
 #qd-probe-20261013 .qd-selection-order { gap:7px; }
 #qd-probe-20261013 .qd-chart-axis { text-align:left; }
}
@media(prefers-reduced-motion:reduce) { #qd-probe-20261013 *,#qd-probe-20261013 *::before,#qd-probe-20261013 *::after { animation:none!important; transition:none!important; } }
`;
