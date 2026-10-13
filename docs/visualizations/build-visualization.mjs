/** Export the exact Canvas JSX as a dependency-free, literal HTML fragment. */
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright'); }
catch { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const directory = path.dirname(fileURLToPath(import.meta.url));
const name = 'question-diversity-probe';
const files = {
  canvas: path.join(directory, name + '.canvas.tsx'),
  styles: path.join(directory, name + '.styles.ts'),
  controller: path.join(directory, name + '.controls.js'),
  fragment: path.join(directory, name + '.fragment.html'),
};
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH || [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  path.join(os.homedir(), 'Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
].find(candidate => fs.existsSync(candidate));
const browser = await playwright.chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
let result;
try {
  const page = await browser.newPage();
  for (const url of [
    'https://cdn.jsdelivr.net/npm/typescript@5.8.3/lib/typescript.js',
    'https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js',
    'https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom-server-legacy.browser.production.min.js',
  ]) await page.addScriptTag({ url });
  result = await page.evaluate(({ canvas, styles }) => {
    const diagnostics = [];
    const compile = (source, filename) => {
      const output = ts.transpileModule(source, {
        fileName: filename, reportDiagnostics: true,
        compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, esModuleInterop: true },
      });
      diagnostics.push(...output.diagnostics.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')));
      const module = { exports: {} };
      const localRequire = name => {
        if (name === 'react') return React;
        if (name === './question-diversity-probe.styles') return styleModule;
        throw new Error('Unexpected Canvas dependency: ' + name);
      };
      new Function('require', 'module', 'exports', output.outputText)(localRequire, module, module.exports);
      return module.exports;
    };
    const styleModule = compile(styles, 'question-diversity-probe.styles.ts');
    const canvasModule = compile(canvas, 'question-diversity-probe.canvas.tsx');
    if (diagnostics.length) throw new Error(diagnostics.join('\n'));
    // React SSR HTML-escapes text children of style, including CSS child selectors.
    // Native React inserts a text node; preserve that exact CSS in literal HTML too.
    const rendered = ReactDOMServer.renderToStaticMarkup(React.createElement(canvasModule.default));
    const css = styleModule.questionDiversityStyles;
    if (typeof css !== 'string' || css.includes('</style')) throw new Error('Invalid scoped stylesheet');
    const html = rendered.replace(/<style>[\s\S]*?<\/style>/, () => '<style>\n' + css + '\n</style>');
    return { html, diagnostics, compiler: 'TypeScript 5.8.3 / React 18.3.1 SSR' };
  }, { canvas: fs.readFileSync(files.canvas, 'utf8'), styles: fs.readFileSync(files.styles, 'utf8') });
} finally { await browser.close(); }

const controller = fs.readFileSync(files.controller, 'utf8');
const fragment = result.html + '\n<script>\n' + controller.replaceAll('</script', '<\\/script') + '\n</script>\n';
if (Buffer.byteLength(fragment) > 1_000_000) throw new Error('Fragment exceeds 1 MB');
if (/<!doctype|<html|<head>|<body>/i.test(fragment)) throw new Error('Expected a fragment');
fs.writeFileSync(files.fragment, fragment);
const inlinePath = process.env.VISUALIZATION_INLINE_PATH;
if (inlinePath) fs.writeFileSync(inlinePath, fragment);
const renderScript = process.env.VISUALIZE_RENDER_SCRIPT || path.join(os.homedir(), '.codex/plugins/cache/openai-bundled/visualize/1.0.49/skills/visualize/scripts/render.py');
if (!fs.existsSync(renderScript)) throw new Error('Set VISUALIZE_RENDER_SCRIPT to the installed visualize renderer');
const rendered = spawnSync(process.env.PYTHON || 'python3', [renderScript, files.fragment, path.join(directory, name + '-demo.html'), '--force'], { encoding: 'utf8' });
if (rendered.status !== 0) throw new Error(rendered.stderr || rendered.stdout);
const receipt = {
  compiler: result.compiler, diagnostics: result.diagnostics,
  source: ['canvas', 'styles', 'controller'].map(key => ({ file: path.basename(files[key]), sha256: crypto.createHash('sha256').update(fs.readFileSync(files[key])).digest('hex') })),
  fragmentBytes: Buffer.byteLength(fragment),
  fragmentSha256: crypto.createHash('sha256').update(fragment).digest('hex'),
  rendered: path.basename(path.join(directory, name + '-demo.html')),
};
fs.writeFileSync(path.join(directory, name + '-build.json'), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
