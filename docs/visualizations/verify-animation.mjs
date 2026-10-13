import { browserForVisualization, directory, assertInteractiveStates, layoutSnapshot, graphicSnapshot, graphicsPass, openFrame, fs, path } from './visualization-test-utils.mjs';

const browser = await browserForVisualization();
const report = { illustrative: true, implementation: 'SSR Canvas + local controls', errors: [], layouts: [], controls: [], passed: false };
try {
  for (const theme of ['light', 'dark']) for (const width of [320, 360, 736, 1024]) {
    const { page, frame } = await openFrame(browser, { width, theme, reducedMotion: 'reduce' });
    page.on('pageerror', error => report.errors.push(error.message));
    report.controls.push({ width, theme, ...(await assertInteractiveStates(frame)) });
    for (let step = 0; step < 6; step++) {
      await frame.locator(`[data-step="${step}"]`).click();
      const graphics = await graphicSnapshot(frame, step);
      report.layouts.push({ width, theme, step, ...(await layoutSnapshot(frame)), graphics, graphicsPass: graphicsPass(step, graphics) });
      if (width === 736 || (width === 360 && [1, 3, 4].includes(step))) await frame.locator('#qd-probe-20261013').screenshot({ path: path.join(directory, `question-diversity-probe-redesign-${theme}-${width}-step-${step}.png`) });
    }
    await page.close();
  }
  const { page, frame } = await openFrame(browser, { width: 736, theme: 'light', reducedMotion: 'no-preference' });
  await frame.locator('[data-step="4"]').click();
  report.actualMotion = await frame.evaluate(() => document.getAnimations().length > 0);
  await frame.locator('[data-action="play"]').click();
  await frame.locator('[data-scene="5"]').waitFor({ state: 'visible', timeout: 7000 });
  await page.waitForTimeout(3500);
  report.autoStops = !(await frame.locator('[data-action="play"]').getAttribute('aria-pressed') === 'true');
  report.finiteAnimations = await frame.evaluate(() => document.getAnimations().every(animation => animation.effect.getTiming().iterations !== Infinity));
  await page.close();
  report.passed = !report.errors.length && report.layouts.every(item => !item.horizontalOverflow && !item.overflow.length && !item.nonFinite.length && item.graphicsPass) && report.autoStops && report.finiteAnimations && report.actualMotion;
} catch (error) { report.failure = String(error); }
finally {
  await browser.close();
  fs.writeFileSync(path.join(directory, 'question-diversity-probe-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, errors: report.errors, failure: report.failure, layoutCases: report.layouts.length, failedCases: report.layouts.filter(item => item.horizontalOverflow || item.overflow.length || item.nonFinite.length || !item.graphicsPass), autoStops: report.autoStops, finiteAnimations: report.finiteAnimations, actualMotion: report.actualMotion }));
}
if (!report.passed) process.exitCode = 1;
