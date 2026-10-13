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
const report = { illustrative: true, compiler: 'TypeScript 5.8.3 / React 18.3.1', syntaxDiagnostics: [], runtimeErrors: [], passed: false };
try {
  const page = await browser.newPage({ viewport: { width: 736, height: 1000 } });
  page.on('pageerror', error => report.runtimeErrors.push(error.message));
  await page.goto('file://' + path.join(directory, 'question-diversity-probe-demo.html'));
  const frame = page.frames().find(frame => frame !== page.mainFrame());
  await frame.locator('#qd-probe-20261013').waitFor();
  for (const url of ['https://cdn.jsdelivr.net/npm/typescript@5.8.3/lib/typescript.js', 'https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js', 'https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom.production.min.js']) await frame.addScriptTag({ url });
  const source = fs.readFileSync(path.join(directory, 'question-diversity-probe.canvas.tsx'), 'utf8').replace(/^import React[^\n]*\n/, 'const { useEffect, useRef, useState } = React;\n').replace(/^import \{ questionDiversityStyles \}[^\n]*\n/m, 'const questionDiversityStyles = window.__qdStyles;\n').replace('export default function', 'function');
  const css = fs.readFileSync(path.join(directory, 'question-diversity-probe.fragment.html'), 'utf8').split('<style>\n')[1].split('</style>')[0];
  report.syntaxDiagnostics = await frame.evaluate(({ source, css }) => {
    window.__qdStyles = css;
    const compiled = window.ts.transpileModule(source, { compilerOptions: { target: window.ts.ScriptTarget.ES2020, module: window.ts.ModuleKind.None, jsx: window.ts.JsxEmit.React }, fileName: 'question-diversity-probe.canvas.tsx', reportDiagnostics: true });
    const diagnostics = compiled.diagnostics.map(d => window.ts.flattenDiagnosticMessageText(d.messageText, '\n'));
    if (diagnostics.length === 0) {
      const mount = document.createElement('div');
      document.body.replaceChildren(mount);
      new Function('mount', compiled.outputText + '\nReactDOM.createRoot(mount).render(React.createElement(QuestionDiversityProbe));')(mount);
    }
    return diagnostics;
  }, { source, css });
  if (report.syntaxDiagnostics.length) throw new Error('Canvas syntax diagnostics');
  await frame.locator('#qd-probe-20261013').waitFor();
  await frame.locator('#qd-canvas-explore').selectOption('128');
  const selectedText = await frame.locator('.qd-selection-labels').textContent();
  if (!selectedText.includes('128') || !selectedText.includes('384')) throw new Error('Canvas exploration selection mismatch');
  await frame.getByRole('button', { name: /5\s*真实轨迹/ }).click();
  await frame.locator('#qd-canvas-correct').selectOption('2');
  if (!(await frame.locator('.qd-moments').textContent()).includes('0.548')) throw new Error('Canvas sample std mismatch');
  await frame.getByRole('button', { name: '播放流程', exact: true }).click();
  await frame.locator('.qd-stage-heading').filter({ hasText: '6 / 6' }).waitFor({ timeout: 7000 });
  report.passed = report.runtimeErrors.length === 0;
} catch (error) {
  report.failure = String(error);
} finally {
  await browser.close();
  fs.writeFileSync(path.join(directory, 'question-diversity-probe-canvas-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
}
if (!report.passed) process.exitCode = 1;
