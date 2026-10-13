import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright'); } catch { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const { chromium } = playwright;
const directory = path.dirname(fileURLToPath(import.meta.url));
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH || ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', path.join(os.homedir(), 'Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')].find(candidate => fs.existsSync(candidate));
const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
const report = { illustrative: true, errors: [], layouts: [], controls: {}, maths: {} };
try {
  for (const width of [320, 360, 736, 1024]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, colorScheme: width === 360 ? 'dark' : 'light', reducedMotion: 'reduce' });
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto('file://' + path.join(directory, 'question-diversity-probe-demo.html'));
    const frame = page.frames().find(frame => frame !== page.mainFrame());
    await frame.locator('#qd-probe-20261013').waitFor();
    for (let step = 0; step < 6; step++) {
      await frame.locator(`[data-step="${step}"]`).click();
      const result = await frame.evaluate(() => {
        const root = document.getElementById('qd-probe-20261013');
        const bounds = root.getBoundingClientRect();
        const overflow = [...root.querySelectorAll('*')].filter(el => !el.closest('[hidden]')).filter(el => {
          const r = el.getBoundingClientRect();
          return r.width > 0 && (r.left < bounds.left - 1 || r.right > bounds.right + 1);
        }).map(el => ({ tag: el.tagName, class: el.className?.baseVal ?? el.className }));
        return { rootWidth: bounds.width, horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth, overflow };
      });
      report.layouts.push({ width, step, ...result });
    }
    for (const exploration of [0, 64, 128, 256]) {
      await frame.locator('#qd-explore').selectOption(String(exploration));
      const observed = await frame.locator('[data-field="focus-count"]').textContent();
      if (Number(observed) !== 512 - exploration) throw new Error('Exploration total mismatch');
    }
    await frame.locator('[data-step="4"]').click();
    for (let correct = 0; correct <= 5; correct++) {
      await frame.locator('#qd-correct').selectOption(String(correct));
      const observed = Number(await frame.locator('[data-field="true-std"]').textContent());
      const expected = Math.sqrt(correct * (5 - correct) / 20);
      if (Math.abs(observed - expected) > .00051) throw new Error('Sample std mismatch');
      report.maths[correct] = { mean: correct / 5, sampleStd: expected, displayedStd: observed };
    }
    await frame.locator('[data-step="2"]').click();
    report.controls.predictedMean = await frame.locator('[data-field="pred-mean"]').textContent();
    report.controls.predictedStd = await frame.locator('[data-field="pred-std"]').textContent();
    if (width === 736 || width === 360) await page.screenshot({ path: path.join(directory, `question-diversity-probe-${width}.png`), fullPage: true });
    await frame.locator('[data-action="reset"]').click();
    await frame.locator('[data-action="play"]').click();
    await frame.locator('[data-action="play"]').click();
    if (await frame.locator('[data-action="play"]').textContent() !== '播放流程') throw new Error('Pause failed');
    report.controls.playPause = true;
    await page.close();
  }
  const page = await browser.newPage({ viewport: { width: 736, height: 1000 } });
  await page.goto('file://' + path.join(directory, 'question-diversity-probe-demo.html'));
  const frame = page.frames().find(frame => frame !== page.mainFrame());
  await frame.locator('[data-step="4"]').click();
  await frame.locator('[data-action="play"]').click();
  await frame.locator('[data-field="stage-count"]').filter({ hasText: '6 / 6' }).waitFor({ timeout: 6000 });
  await page.waitForTimeout(3500);
  report.controls.autoStopsAtEnd = (await frame.locator('[data-action="play"]').textContent()) === '重新播放';
  await page.close();
} finally {
  await browser.close();
}
report.passed = report.errors.length === 0 && report.layouts.every(x => !x.horizontalOverflow && x.overflow.length === 0) && report.controls.autoStopsAtEnd;
fs.writeFileSync(path.join(directory, 'question-diversity-probe-verification.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed: report.passed, errors: report.errors, layoutCases: report.layouts.length, overflowCases: report.layouts.filter(x => x.horizontalOverflow || x.overflow.length), controls: report.controls }));
if (!report.passed) process.exitCode = 1;
