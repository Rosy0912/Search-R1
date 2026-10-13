import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright'); }
catch { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const directory = path.dirname(fileURLToPath(import.meta.url));
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH || ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', path.join(os.homedir(), 'Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')].find(candidate => fs.existsSync(candidate));
export async function browserForVisualization() { return playwright.chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) }); }
export async function openFrame(browser, { width, theme, reducedMotion = 'reduce' }) {
  const page = await browser.newPage({ viewport: { width, height: 1800 }, colorScheme: theme, reducedMotion });
  await page.goto('file://' + path.join(directory, 'question-diversity-probe-demo.html'));
  const frame = page.frames().find(frame => frame !== page.mainFrame());
  await frame.locator('#qd-probe-20261013').waitFor();
  return { page, frame };
}
export async function assertInteractiveStates(frame) {
  const result = { exploration: [], labels: [], playPause: false, stepNavigation: false };
  await frame.locator('[data-step="3"]').click();
  for (const exploration of [0, 64, 128, 256]) {
    await frame.locator('[data-action="exploration"]').selectOption(String(exploration));
    const focus = Number(await frame.locator('[data-field="focus-count"]').first().textContent());
    const random = Number(await frame.locator('[data-field="explore-count"]').first().textContent());
    if (focus !== 512 - exploration || random !== exploration) throw new Error(`Selection mismatch E=${exploration}: ${random}+${focus}`);
    result.exploration.push({ exploration, focus });
  }
  await frame.locator('[data-step="4"]').click();
  for (let correct = 0; correct <= 5; correct++) {
    await frame.locator('[data-action="correct"]').selectOption(String(correct));
    const mean = Number(await frame.locator('[data-field="true-mean"]').first().textContent());
    const std = Number(await frame.locator('[data-field="true-std"]').first().textContent());
    const expectedStd = Math.sqrt(correct * (5 - correct) / 20);
    const trajectoryMarks = await frame.locator('[data-trajectory][data-correct="true"]').count();
    if (Math.abs(mean - correct / 5) > .00051 || Math.abs(std - expectedStd) > .00051 || trajectoryMarks !== correct) throw new Error(`Five EM labels mismatch K=${correct}: ${mean}/${std}/${trajectoryMarks}`);
    result.labels.push({ correct, mean, std });
  }
  await frame.locator('[data-step="2"]').click();
  const predictedMean = Number(await frame.locator('[data-field="pred-mean"]').first().textContent());
  const predictedStd = Number(await frame.locator('[data-field="pred-std"]').first().textContent());
  const expectedPredictedStd = [.05, .10, .25, .40, .15, .05].reduce((sum, probability, k) => sum + probability * Math.sqrt(k * (5 - k) / 20), 0);
  if (Math.abs(predictedMean - .53) > .00051 || Math.abs(predictedStd - expectedPredictedStd) > .00051) throw new Error('Predicted moments mismatch');
  await frame.locator('[data-step="0"]').click();
  await frame.locator('[data-action="next"]').click();
  if (!(await frame.locator('[data-scene="1"]').isVisible())) throw new Error('Next failed');
  result.stepNavigation = true;
  await frame.locator('[data-action="play"]').click();
  await frame.locator('[data-action="play"]').click();
  if (await frame.locator('[data-action="play"]').getAttribute('aria-pressed') === 'true') throw new Error('Pause failed');
  result.playPause = true;
  await frame.locator('[data-step="3"]').click();
  await frame.locator('[data-action="exploration"]').selectOption('64');
  await frame.locator('[data-step="4"]').click();
  await frame.locator('[data-action="correct"]').selectOption('2');
  return result;
}
export async function layoutSnapshot(frame) {
  return frame.evaluate(() => {
    const root = document.getElementById('qd-probe-20261013');
    const bounds = root.getBoundingClientRect();
    const visible = [...root.querySelectorAll('*')].filter(el => !el.closest('[hidden]'));
    const nonFinite = visible.filter(el => [...Object.values(el.getBoundingClientRect().toJSON())].some(value => !Number.isFinite(value))).map(el => el.tagName);
    const overflow = visible.filter(el => {
      const rect = el.getBoundingClientRect();
      return rect.width > 0 && (rect.left < bounds.left - 1 || rect.right > bounds.right + 1);
    }).map(el => ({ tag: el.tagName, class: el.className?.baseVal ?? el.className }));
    const chartLabels = visible.filter(el => el.tagName === 'text' && el.closest('svg')).map(el => ({ text: el.textContent, fontSize: getComputedStyle(el).fontSize, rect: el.getBoundingClientRect().toJSON() }));
    return { rootWidth: bounds.width, rootHeight: bounds.height, horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth, overflow, nonFinite, chartLabels };
  });
}
export async function graphicSnapshot(frame, step) {
  return frame.evaluate(step => {
    const root = document.getElementById('qd-probe-20261013');
    const selector = step === 1 ? '.qd-vector-cache > div' : step === 2 ? '.qd-bar' : step === 4 ? '.qd-trace-row > div' : null;
    if (!selector) return null;
    return [...root.querySelectorAll(selector)].map(el => {
      const rect = el.getBoundingClientRect();
      const css = getComputedStyle(el);
      return { class: el.className, width: rect.width, height: rect.height, display: css.display, background: css.backgroundColor };
    });
  }, step);
}
export function graphicsPass(step, elements) {
  if (!elements) return true;
  if (step === 1) return elements.length === 4 && elements.every(el => el.display === 'flex' && el.width > 50 && el.height >= 14 && el.height <= 20);
  if (step === 2) return elements.length === 6 && elements.every(el => el.width > 8 && el.height > 12 && el.background !== 'rgba(0, 0, 0, 0)');
  if (step === 4) return elements.length === 15 && elements.every(el => el.display === 'grid' && el.width > 12 && el.height >= 25 && el.background !== 'rgba(0, 0, 0, 0)');
  return true;
}
export { directory, fs, path };
