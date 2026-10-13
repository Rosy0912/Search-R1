import { browserForVisualization, directory, assertInteractiveStates, layoutSnapshot, graphicSnapshot, graphicsPass, openFrame, fs, path } from './visualization-test-utils.mjs';

const browser = await browserForVisualization();
const report = { illustrative: true, compiler: 'TypeScript 5.8.3 / React 18.3.1', syntaxDiagnostics: [], errors: [], layouts: [], controls: [], passed: false };
try {
  for (const theme of ['light', 'dark']) for (const width of [320, 360, 736, 1024]) {
    const { page, frame } = await openFrame(browser, { width, theme });
    page.on('pageerror', error => report.errors.push(error.message));
    for (const url of ['https://cdn.jsdelivr.net/npm/typescript@5.8.3/lib/typescript.js', 'https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js', 'https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom.production.min.js']) await frame.addScriptTag({ url });
    const diagnostics = await frame.evaluate(({ canvas, styles }) => {
      const diagnostics = [];
      const compile = (source, filename) => {
        const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, esModuleInterop: true }, fileName: filename, reportDiagnostics: true });
        diagnostics.push(...compiled.diagnostics.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')));
        const module = { exports: {} };
        new Function('require', 'module', 'exports', compiled.outputText)(name => {
          if (name === 'react') return React;
          if (name === './question-diversity-probe.styles') return styleModule;
          throw new Error('Unexpected dependency ' + name);
        }, module, module.exports);
        return module.exports;
      };
      const styleModule = compile(styles, 'question-diversity-probe.styles.ts');
      const canvasModule = compile(canvas, 'question-diversity-probe.canvas.tsx');
      const mount = document.createElement('div');
      document.body.replaceChildren(mount);
      ReactDOM.createRoot(mount).render(React.createElement(canvasModule.default));
      return diagnostics;
    }, { canvas: fs.readFileSync(path.join(directory, 'question-diversity-probe.canvas.tsx'), 'utf8'), styles: fs.readFileSync(path.join(directory, 'question-diversity-probe.styles.ts'), 'utf8') });
    report.syntaxDiagnostics.push(...diagnostics);
    if (diagnostics.length) throw new Error('Canvas syntax diagnostics');
    await frame.locator('#qd-probe-20261013').waitFor();
    report.controls.push({ width, theme, ...(await assertInteractiveStates(frame)) });
    for (let step = 0; step < 6; step++) {
      await frame.locator(`[data-step="${step}"]`).click();
      const graphics = await graphicSnapshot(frame, step);
      report.layouts.push({ width, theme, step, ...(await layoutSnapshot(frame)), graphics, graphicsPass: graphicsPass(step, graphics) });
    }
    if (width === 736 && theme === 'light') await frame.locator('#qd-probe-20261013').screenshot({ path: path.join(directory, 'question-diversity-probe-redesign-native-736.png') });
    await page.close();
  }
  report.passed = !report.errors.length && !report.syntaxDiagnostics.length && report.layouts.every(item => !item.horizontalOverflow && !item.overflow.length && !item.nonFinite.length && item.graphicsPass);
} catch (error) { report.failure = String(error); }
finally {
  await browser.close();
  fs.writeFileSync(path.join(directory, 'question-diversity-probe-canvas-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, errors: report.errors, syntaxDiagnostics: report.syntaxDiagnostics, failure: report.failure, layoutCases: report.layouts.length, failedCases: report.layouts.filter(item => item.horizontalOverflow || item.overflow.length || item.nonFinite.length || !item.graphicsPass) }));
}
if (!report.passed) process.exitCode = 1;
