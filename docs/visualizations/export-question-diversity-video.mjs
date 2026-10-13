/** Export the existing six-scene explainer; synthetic diagrams, no training work. */
import { browserForVisualization, directory, fs, path } from './visualization-test-utils.mjs';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';

const output = path.resolve(process.argv[2] || 'question-diversity-probe.mp4');
const previewOnly = process.argv.includes('--preview');
const option = name => { const index=process.argv.indexOf(name); return index<0?undefined:process.argv[index+1]; };
const pace = option('--pace') || 'standard';
if(!['standard','brisk'].includes(pace)) throw new Error('--pace must be standard or brisk');
const musicPath = option('--music') ? path.resolve(option('--music')) : null;
if(musicPath&&!fs.existsSync(musicPath)) throw new Error('Music input does not exist');
const fps = 30, width = 1920, height = 1080;
const durations = pace==='brisk' ? [3.5,4,5.5,5.5,5.5,6] : [6, 7, 9, 10, 10, 10];
const totalSeconds = durations.reduce((a,b)=>a+b,0);
const motionRate = pace==='brisk' ? 2 : 1;
const transitionMs = pace==='brisk' ? 250 : 550;
const captions = pace==='brisk' ? [
  ['5,120 个候选，只读取问题。', '先预测再选题；此时不生成思考、检索或答案。'],
  ['固定编码器，缓存问题表征。', '官方 Qwen 首次编码；每轮读缓存，运行轻量 head。'],
  ['预测五条严格 EM 的组内差异。', '预测 K = 0…5 的分布 → mean / 期望样本 std；按 std 排序。'],
  ['先随机探索，再取高预测 std。', '示例 64 随机 + 448 高差异；探索题数可选 0、64、128、256。'],
  ['选中 512 题，生成 2,560 条真实轨迹。', '每题 5 条，以严格 EM 标注；全对、全错组也保留。'],
  ['同一批轨迹，两路学习。', '选题 head 每个成功 iteration 更新；原奖励 probe 仍按 50 轮刷新。'],
] : [
  ['5,120 个候选，只读取问题。', '此时不生成思考、检索或答案；完整 rollout 预算留给最终选中的 512 题。'],
  ['固定编码器，缓存问题表征。', '官方 Qwen 首次编码后，每轮只读取缓存并运行轻量选题 head。'],
  ['预测五条严格 EM 的组内差异。', '六个概率对应 K = 0…5；同时得到 mean 与期望样本 std，以后者排序。'],
  ['先随机探索，再取高预测 std。', '示例 E = 64：64 题均匀随机 + 448 题高差异；E 可消融为 0、64、128、256。'],
  ['只有选中的 512 题，才生成真实轨迹。', '每题 5 条，共 2,560 条；用严格 EM 标注，全对、全错组也保留。'],
  ['同一批轨迹，同时训练 actor 与选题 head。', '每个成功 iteration 更新选题 head，供下一轮使用；原奖励 probe 仍按 50 轮刷新。'],
];
fs.mkdirSync(path.dirname(output), { recursive: true });
const prefix = output.replace(/\.mp4$/i, '');
const browser = await browserForVisualization();
const page = await browser.newPage({ viewport: { width, height }, colorScheme: 'light', reducedMotion: 'no-preference', deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
let encoder;
try {
  await page.goto('file://' + path.join(directory, 'question-diversity-probe-demo.html'));
  const frame = page.frames().find(item => item !== page.mainFrame());
  await frame.locator('#qd-probe-20261013').waitFor();
  await page.addStyleTag({ content: `
    html,body{padding:0!important;margin:0!important;overflow:hidden!important;background:#fff!important;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#202329}
    #codex-visualization{position:absolute;left:344px;top:104px;width:1120px;height:685px!important;transform:scale(1.1);transform-origin:top left}
    .video-brand{position:absolute;left:260px;top:46px;font-size:15px;letter-spacing:3px;color:#797f89}
    .video-phase{position:absolute;right:260px;top:42px;font-size:20px;letter-spacing:2px;color:#797f89;font-variant-numeric:tabular-nums}
    .video-caption{position:absolute;left:260px;right:260px;top:874px;border-top:1px solid #e8eaee;padding-top:27px}
    .video-caption h2{margin:0 0 13px;font-size:28px;font-weight:500;line-height:1.4}
    .video-caption p{margin:0;font-size:19px;color:#6c7380;line-height:1.55}
    .video-track{position:absolute;left:260px;right:260px;bottom:40px;height:3px;display:grid;grid-template-columns:${durations.map(n=>n+'fr').join(' ')};gap:7px}
    .video-track span{background:#e8eaee;position:relative;overflow:hidden}
    .video-track i{height:100%;display:block;transform-origin:left;background:#485daa;transform:scaleX(0)}
  `});
  await page.evaluate(() => {
    const items = [ ['div','video-brand','SEARCH-G1 · MIXED-FIRST'], ['div','video-phase','01 / 06'], ['div','video-caption',''], ['div','video-track',''] ];
    for (const [tag, className, content] of items) { const node = document.createElement(tag); node.className = className; node.textContent = content; document.body.append(node); }
    document.querySelector('.video-caption').innerHTML='<h2></h2><p></p>';
    document.querySelector('.video-track').innerHTML='<span><i></i></span>'.repeat(6);
  });
  await frame.addStyleTag({ content: `
    html,body{padding:0!important;margin:0!important;overflow:hidden!important}
    #qd-probe-20261013{--font-size-base:17px!important}
    #qd-probe-20261013 .qd-player{display:none!important}
    #qd-probe-20261013 .qd-header{padding-bottom:24px!important}
    #qd-probe-20261013 .qd-main{min-height:340px!important}
    #qd-probe-20261013 .qd-select-setting,#qd-probe-20261013 .qd-rollout-setting{display:none!important}
    #qd-probe-20261013 .qd-navigation{padding-bottom:24px!important}
    #qd-probe-20261013 .qd-footer{display:none!important}
    #qd-probe-20261013 .qd-detail{display:none!important}
    #qd-probe-20261013 .qd-stage-heading{padding-bottom:25px!important}
  `});
  await frame.evaluate(() => document.fonts.ready);
  const frames = fps * totalSeconds;
  const sceneProofs = [];
  if (!previewOnly) {
    const audioArgs = musicPath ? ['-i',musicPath,'-map','0:v:0','-map','1:a:0','-af','loudnorm=I=-24:TP=-2:LRA=7','-c:a','aac','-b:a','160k','-ar','48000','-t',String(totalSeconds)] : ['-an'];
    encoder = spawn(process.env.FFMPEG || 'ffmpeg', ['-hide_banner','-loglevel','warning','-y','-f','image2pipe','-vcodec','mjpeg','-framerate',String(fps),'-i','pipe:0',...audioArgs,'-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart','-metadata','title=Search-G1 question diversity probe','-metadata','comment=Synthetic explanatory animation; not measured training results',output], { stdio: ['pipe','ignore','pipe'] });
    encoder.stderr.on('data', chunk => process.stderr.write(chunk));
  }
  let completed = 0;
  for (let step=0; step<6; step++) {
    await frame.evaluate(step=>document.querySelector(`[data-step="${step}"]`).click(),step);
    await frame.evaluate(() => {
      const root=document.getElementById('qd-probe-20261013');
      root.dataset.animated='true';
      void root.offsetHeight;
      for (const animation of document.getAnimations()) animation.pause();
    });
    await page.evaluate(({step,caption}) => {
      document.querySelector('.video-phase').textContent=`${String(step+1).padStart(2,'0')} / 06`;
      document.querySelector('.video-caption h2').textContent=caption[0];
      document.querySelector('.video-caption p').textContent=caption[1];
    }, {step,caption:captions[step]});
    const proof = await frame.evaluate(() => {
      const root=document.getElementById('qd-probe-20261013'), bounds=root.getBoundingClientRect();
      return { phase:Number(root.dataset.phase), height:bounds.height, width:bounds.width,
        scenes:[...root.querySelectorAll('[data-scene]')].filter(n=>!n.hidden).map(n=>Number(n.dataset.scene)),
        animations:document.getAnimations().length,
        selected:[...root.querySelectorAll('[data-field="explore-count"],[data-field="focus-count"]')].map(n=>n.textContent),
        actual:[...root.querySelectorAll('[data-field="true-mean"],[data-field="true-std"]')].map(n=>n.textContent) };
    });
    if(proof.height>685||proof.scenes.join()!==String(step)) throw new Error(`Clipped or wrong scene: ${JSON.stringify(proof)}`);
    sceneProofs.push(proof);
    const count=durations[step]*fps;
    for(let f=0; f<(previewOnly?1:count); f++) {
      const t=previewOnly ? 3000 : f/fps*1000;
      await frame.evaluate(({t,motionRate,transitionMs}) => {
        for(const animation of document.getAnimations()) {animation.pause();animation.currentTime=t*motionRate;}
        const phase=Number(document.getElementById('qd-probe-20261013').dataset.phase);
        const scene=document.querySelector(`[data-scene="${phase}"]`);
        const fraction=Math.min(1,t/transitionMs), eased=1-Math.pow(1-fraction,3);
        scene.style.opacity=String(eased);
        scene.style.transform=`translateY(${(1-eased)*12}px)`;
        for(const animation of document.getAnimations()) if(animation.effect?.target) void getComputedStyle(animation.effect.target).transform;
      },{t,motionRate,transitionMs});
      await page.evaluate(({step,f,count,t,transitionMs})=>{
        const fraction=Math.min(1,t/transitionMs),eased=1-Math.pow(1-fraction,3),caption=document.querySelector('.video-caption');
        caption.style.opacity=String(eased);caption.style.transform=`translateY(${(1-eased)*8}px)`;
        document.querySelectorAll('.video-track i').forEach((n,i)=>{n.style.transform=`scaleX(${i<step?1:i===step?f/count:0})`;});
      },{step,f,count,t,transitionMs});
      const data=await page.screenshot({type:'jpeg',quality:94,animations:'allow'});
      if(previewOnly) fs.writeFileSync(`${prefix}-scene-${step+1}.jpg`,data);
      else if(!encoder.stdin.write(data)) await once(encoder.stdin,'drain');
      completed++;
      if(!previewOnly&&completed%90===0) console.log(JSON.stringify({frames:completed,total_frames:frames,percent:Math.round(completed/frames*100),output_bytes:fs.existsSync(output)?fs.statSync(output).size:0}));
    }
  }
  if(errors.length) throw new Error(errors.join('\n'));
  if(encoder) {
    encoder.stdin.end();
    const [code]=await once(encoder,'close');
    if(code!==0) throw new Error(`FFmpeg exited ${code}`);
    const probe=JSON.parse(execFileSync(process.env.FFPROBE||'ffprobe',['-v','error','-count_frames','-show_entries','stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames,sample_rate,channels,duration:format=duration,size','-of','json',output],{encoding:'utf8'}));
    const stream=probe.streams.find(item=>item.codec_type==='video');
    const audioStream=probe.streams.find(item=>item.codec_type==='audio');
    if(stream.width!==width||stream.height!==height||Number(stream.nb_read_frames)!==frames||Math.abs(Number(probe.format.duration)-totalSeconds)>.06||Boolean(audioStream)!==Boolean(musicPath)) throw new Error('Video verification mismatch');
    const receipt={created_at:new Date().toISOString(),source:'question-diversity-probe-demo.html',source_sha256:createHash('sha256').update(fs.readFileSync(path.join(directory,'question-diversity-probe-demo.html'))).digest('hex'),output,width,height,fps,frames,durations,total_seconds:totalSeconds,pace,motion_rate:motionRate,transition_ms:transitionMs,sceneProofs,probe,sha256:createHash('sha256').update(fs.readFileSync(output)).digest('hex'),synthetic:true,audio:Boolean(audioStream),music:musicPath?{input_sha256:createHash('sha256').update(fs.readFileSync(musicPath)).digest('hex'),target_lufs:-24}:null,page_errors:errors};
    fs.writeFileSync(`${prefix}-video-verification.json`,JSON.stringify(receipt,null,2)+'\n');
    console.log(JSON.stringify({completed:true,output,bytes:fs.statSync(output).size,sha256:receipt.sha256}));
  } else console.log(JSON.stringify({preview:true,sceneProofs,errors}));
} finally { if(encoder&&!encoder.killed&&encoder.exitCode===null)encoder.kill();await browser.close(); }
