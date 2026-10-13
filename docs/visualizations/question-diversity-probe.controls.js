/* Local interaction only. Markup, SVG and CSS come directly from the Canvas SSR. */
(() => {
  const root = document.getElementById('qd-probe-20261013');
  if (!root || root.dataset.initialized) return;
  root.dataset.initialized = 'true';
  const model = JSON.parse(root.dataset.model || '{}');
  const phases = model.titles || [];
  const details = model.details || [];
  const probabilities = model.probabilities || [.05, .10, .25, .40, .15, .05];
  const sampleStd = k => Math.sqrt(k * (5 - k) / 20);
  const fields = name => [...root.querySelectorAll('[data-field="' + name + '"]')];
  const put = (name, value) => fields(name).forEach(el => { el.textContent = String(value); });
  const action = name => root.querySelector('[data-action="' + name + '"]');
  let state = { step: 0, exploration: 64, correct: 2, playing: false };
  let timer = null;
  function restore(saved) {
    const meaningful = saved?.modelContent || {};
    const privateState = saved?.privateContent || {};
    if ([0, 64, 128, 256].includes(meaningful.exploration_questions)) state.exploration = meaningful.exploration_questions;
    if (Number.isInteger(privateState.step) && privateState.step >= 0 && privateState.step < 6) state.step = privateState.step;
    if (Number.isInteger(privateState.correct) && privateState.correct >= 0 && privateState.correct <= 5) state.correct = privateState.correct;
  }
  restore(window.openai?.widgetState);
  function save() {
    window.openai?.setWidgetState?.({
      modelContent: { exploration_questions: state.exploration, focus_questions: 512 - state.exploration, selection_score: 'expected_sample_std', illustrative: true },
      privateContent: { step: state.step, correct: state.correct },
    })?.catch(() => {});
  }
  function stop() { state.playing = false; clearTimeout(timer); timer = null; }
  function render(animate = false) {
    const playLabel = state.playing ? '暂停' : state.step === 5 ? '重新播放' : '播放流程';
    if (fields('play-label').length) put('play-label', playLabel);
    else action('play').textContent = playLabel;
    action('play').setAttribute('aria-pressed', String(state.playing));
    action('next').disabled = state.step === 5;
    action('exploration').value = String(state.exploration);
    action('correct').value = String(state.correct);
    put('stage-count', `${state.step + 1} / 6`);
    put('stage-title', phases[state.step] || '');
    put('stage-detail', details[state.step] || '');
    put('detail', details[state.step] || '');
    put('explore-count', state.exploration);
    put('focus-count', 512 - state.exploration);
    put('pred-mean', probabilities.reduce((sum, p, k) => sum + p * k / 5, 0).toFixed(3));
    put('pred-std', probabilities.reduce((sum, p, k) => sum + p * sampleStd(k), 0).toFixed(3));
    put('true-mean', (state.correct / 5).toFixed(3));
    put('true-std', sampleStd(state.correct).toFixed(3));
    root.dataset.phase = String(state.step);
    root.dataset.animated = String(animate);
    root.dataset.playing = String(state.playing);
    root.querySelectorAll('[data-step]').forEach(button => {
      const index = Number(button.dataset.step);
      button.setAttribute('aria-pressed', String(index === state.step));
      button.dataset.state = index < state.step ? 'complete' : index === state.step ? 'active' : 'upcoming';
    });
    root.querySelectorAll('[data-scene]').forEach(scene => {
      scene.hidden = Number(scene.dataset.scene) !== state.step;
      scene.classList.remove('qd-enter');
    });
    const scene = root.querySelector('[data-scene="' + state.step + '"]');
    if (animate && scene) { void scene.offsetHeight; scene.classList.add('qd-enter'); }
    root.querySelectorAll('[data-allocation]').forEach(el => {
      const amount = el.dataset.allocation === 'random' ? state.exploration : 512 - state.exploration;
      el.style.width = amount / 512 * 100 + '%';
      el.dataset.empty = String(amount === 0);
    });
    root.querySelectorAll('[data-nav-index]').forEach(el => {
      const index = Number(el.dataset.navIndex);
      el.classList.toggle('is-active', index === state.step);
      el.classList.toggle('is-done', index < state.step);
    });
    root.querySelectorAll('.qd-pool-selected [data-question-index]').forEach(el => {
      const isRandom = Number(el.dataset.questionIndex) < state.exploration / 8;
      el.dataset.kind = isRandom ? 'random' : 'top';
      el.classList.toggle('is-explore', isRandom);
      el.classList.toggle('is-focus', !isRandom);
    });
    root.querySelectorAll('[data-trajectory]').forEach(el => {
      const correct = Number(el.dataset.trajectory) < state.correct;
      el.dataset.correct = String(correct);
      el.classList.toggle('is-correct', correct);
      el.querySelectorAll('[data-em]').forEach(em => {
        em.textContent = String(Number(correct));
        em.classList.toggle('is-correct', correct);
        em.setAttribute('aria-label', `轨迹${Number(el.dataset.trajectory) + 1}，EM ${Number(correct)}`);
      });
      el.querySelectorAll('[data-em-mark]').forEach(mark => { mark.textContent = correct ? '✓' : '×'; mark.classList.toggle('is-correct', correct); });
    });
  }
  function schedule() {
    clearTimeout(timer);
    if (!state.playing || state.step >= 5) return;
    timer = setTimeout(() => {
      state.step += 1;
      if (state.step === 5) stop();
      render(true); save(); schedule();
    }, model.phaseMs?.[state.step] || Number(model.durationMs) || 3200);
  }
  action('play').addEventListener('click', () => {
    if (state.playing) stop();
    else { if (state.step === 5) state.step = 0; state.playing = true; }
    render(true); save(); schedule();
  });
  action('next').addEventListener('click', () => { stop(); state.step = Math.min(5, state.step + 1); render(true); save(); });
  action('reset')?.addEventListener('click', () => { stop(); state.step = 0; render(true); save(); });
  root.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => { stop(); state.step = Number(button.dataset.step); render(true); save(); }));
  action('exploration').addEventListener('change', event => { stop(); state.exploration = Number(event.target.value); state.step = 3; render(true); save(); });
  action('correct').addEventListener('change', event => { stop(); state.correct = Number(event.target.value); render(false); save(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { stop(); render(false); } });
  window.addEventListener('openai:set_globals', event => {
    const saved = event.detail?.globals?.widgetState;
    if (!saved) return;
    const previous = [state.step, state.exploration, state.correct].join(':');
    restore(saved);
    if (previous !== [state.step, state.exploration, state.correct].join(':')) { stop(); render(false); }
  });
  render(false);
})();
