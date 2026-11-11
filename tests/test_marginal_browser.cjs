"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  { spawn, spawnSync } = require("node:child_process");
const { waitForDebugger } = require("./chrome_support.js");
const controls = fs
  .readdirSync(path.join(__dirname, "../transportlab/artifacts"))
  .filter((p) => p.startsWith("marginal-"))
  .sort()
  .map((p) =>
    JSON.parse(
      fs.readFileSync(path.join(__dirname, "../transportlab/artifacts", p)),
    ),
  );
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
test(
  "real Chrome all12controls, every update/cell, normalization, keyboard, downloads and36layouts",
  { timeout: 120000 },
  async () => {
    const profile = fs.mkdtempSync(
        path.join(os.tmpdir(), "transport-marginal-"),
      ),
      downloads = path.join(profile, "downloads");
    fs.mkdirSync(downloads);
    const browser = spawn(
      process.env.CHROME_BIN || "google-chrome",
      [
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-dev-shm-usage",
        "--remote-debugging-port=0",
        `--user-data-dir=${profile}`,
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    let ws,
      error,
      stderr = "";
    browser.on("error", (e) => (error = e));
    browser.stderr.on("data", (d) => (stderr = (stderr + d).slice(-4000)));
    try {
      for (
        let i = 0;
        i < 1200 &&
        !error &&
        browser.exitCode === null &&
        !fs.existsSync(path.join(profile, "DevToolsActivePort"));
        i++
      )
        await delay(25);
      assert.ok(
        fs.existsSync(path.join(profile, "DevToolsActivePort")),
        `${error?.message || stderr}`,
      );
      const port = fs
          .readFileSync(path.join(profile, "DevToolsActivePort"), "utf8")
          .split("\n")[0],
        tab = await waitForDebugger(`http://127.0.0.1:${port}/json/list`, {
          alive: () => browser.exitCode === null && !error,
        });
      ws = new WebSocket(tab.webSocketDebuggerUrl);
      await new Promise((res, rej) => {
        ws.onopen = res;
        ws.onerror = rej;
      });
      let seq = 0;
      const pending = new Map();
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (pending.has(m.id)) {
          const { res, rej } = pending.get(m.id);
          pending.delete(m.id);
          m.error ? rej(Error(JSON.stringify(m.error))) : res(m.result);
        }
      };
      const call = (method, params = {}) =>
          new Promise((res, rej) => {
            const id = ++seq;
            pending.set(id, { res, rej });
            ws.send(JSON.stringify({ id, method, params }));
          }),
        js = async (expression) => {
          const r = await call("Runtime.evaluate", {
            expression,
            returnByValue: true,
            awaitPromise: true,
          });
          if (r.exceptionDetails)
            throw Error(JSON.stringify(r.exceptionDetails));
          return r.result.value;
        };
      await call("Page.enable");
      await call("Page.navigate", {
        url:
          process.env.TRANSPORT_MARGINAL_URL ||
          "file://" + path.resolve(__dirname, "../docs/index.html"),
      });
      for (let i = 0; i < 300; i++) {
        if (await js("Boolean(globalThis.MARGINAL_APP?.getReport())")) break;
        await delay(20);
      }
      assert.equal(
        await js(`document.getElementById('error').textContent`),
        "",
      );
      const near = (a, b) =>
        assert.ok(
          Number.isFinite(a) && Math.abs(a - b) <= 2e-8 * (1 + Math.abs(b)),
          `${a} vs ${b}`,
        );
      let updates = 0,
        cells = 0,
        layouts = 0;
      for (const c of controls) {
        await js(
          `document.getElementById('control').value=${JSON.stringify(c.name)};document.getElementById('control').dispatchEvent(new Event('change'))`,
        );
        const full = await js("MARGINAL_APP.getReport()");
        near(
          full.source_tangent.reduce((a, b) => a + b, 0),
          0,
        );
        assert.equal(full.history.length, c.report.history.length);
        for (let t = 1; t <= full.history.length; t++) {
          await js(
            `document.getElementById('step').value='${t}';document.getElementById('step').dispatchEvent(new Event('input'))`,
          );
          const e = await js("MARGINAL_SELECTED"),
            ref = c.report.history[t - 1];
          near(e.full_tangent, ref.full_tangent);
          near(e.envelope_candidate, ref.envelope_candidate);
          near(e.residual_tangent_terms, ref.residual_tangent_terms);
          near(
            Number(await js(`document.getElementById('full').textContent`)),
            ref.full_tangent,
          );
          const displayed = await js(
            `Array.from(document.querySelectorAll('#cell-table tbody tr')).map(tr=>Array.from(tr.cells).map(td=>td.textContent))`,
          );
          let k = 0;
          for (let i = 0; i < ref.plan.length; i++)
            for (let j = 0; j < ref.plan[i].length; j++) {
              near(e.plan_tangent[i][j], ref.plan_tangent[i][j]);
              near(Number(displayed[k][3]), ref.plan[i][j]);
              near(Number(displayed[k][4]), ref.plan_tangent[i][j]);
              if (ref.log_plan[i][j] === null)
                assert.equal(displayed[k][5], "excluded");
              else near(Number(displayed[k][5]), ref.log_plan[i][j]);
              k++;
              cells++;
            }
          updates++;
        }
        for (const width of [320, 390, 1280]) {
          await call("Emulation.setDeviceMetricsOverride", {
            width,
            height: 1100,
            deviceScaleFactor: 1,
            mobile: width < 700,
          });
          assert.ok(
            await js("document.documentElement.scrollWidth<=innerWidth"),
            "overflow " + c.name + width,
          );
          assert.ok(
            await js(
              `Array.from(document.querySelectorAll('button,input,select,summary,textarea')).every(e=>e.getBoundingClientRect().height===0||e.getBoundingClientRect().height>=44)`,
            ),
            "touch " + c.name + width,
          );
          layouts++;
        }
      }
      await js(
        `document.getElementById('control').value='joint-direction';MARGINAL_APP.restore();document.getElementById('step').value='1';document.getElementById('step').focus()`,
      );
      await call("Input.dispatchKeyEvent", {
        type: "keyDown",
        key: "ArrowRight",
        code: "ArrowRight",
        windowsVirtualKeyCode: 39,
      });
      await call("Input.dispatchKeyEvent", {
        type: "keyUp",
        key: "ArrowRight",
        code: "ArrowRight",
        windowsVirtualKeyCode: 39,
      });
      assert.equal(await js("MARGINAL_SELECTED.iteration"), 2);
      await js(
        `document.querySelector('.input-panel details').open=true;document.getElementById('input').focus()`,
      );
      const replace = async (text) => {
        await call("Input.dispatchKeyEvent", {
          type: "keyDown",
          key: "a",
          code: "KeyA",
          windowsVirtualKeyCode: 65,
          modifiers: 2,
        });
        await call("Input.dispatchKeyEvent", {
          type: "keyUp",
          key: "a",
          code: "KeyA",
          windowsVirtualKeyCode: 65,
          modifiers: 2,
        });
        await call("Input.insertText", { text });
      };
      await replace('{"epsilon":1,"epsilon":2}');
      assert.equal(await js("MARGINAL_APP.getReport()"), null);
      assert.match(
        await js(`document.getElementById('error').textContent`),
        /Duplicate/,
      );
      assert.equal(
        await js(`document.getElementById('download').disabled`),
        true,
      );
      assert.equal(
        await js(`document.getElementById('full').textContent`),
        "-",
      );
      const edited = structuredClone(
        controls.find((c) => c.name === "zero-cost-product").input,
      );
      edited.source_weights = [3, 4];
      edited.source_direction = [0.3, -0.2];
      edited.iterations = 2;
      await replace(JSON.stringify(edited));
      assert.deepEqual(await js("MARGINAL_APP.getReport().input"), edited);
      await js(
        `Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`,
      );
      await delay(10);
      assert.match(
        await js(`document.getElementById('status').textContent`),
        /Copy was blocked/,
      );
      await js(
        `Object.defineProperty(navigator,'clipboard',{value:{writeText:async s=>globalThis.copied=s},configurable:true});document.getElementById('copy').click()`,
      );
      await delay(10);
      assert.deepEqual(await js("JSON.parse(copied).input"), edited);
      await call("Browser.setDownloadBehavior", {
        behavior: "allow",
        downloadPath: downloads,
      });
      for (const name of ["joint-direction", "normalization-only"]) {
        await js(
          `document.getElementById('control').value='${name}';MARGINAL_APP.restore();document.getElementById('download').click()`,
        );
        const f = path.join(downloads, "transportlab-marginal-report.json");
        for (let i = 0; i < 200 && !fs.existsSync(f); i++) await delay(25);
        assert.ok(fs.existsSync(f));
        const py = spawnSync(
          "python",
          [
            "-c",
            'import sys,json,math\nfrom transportlab.marginal import audit\nr=json.load(open(sys.argv[1]));expected=audit(r["input"])\ndef same(a,b):\n if type(b) in (float,int):assert type(a) in (float,int) and abs(a-b)<=2e-9*(1+abs(b))\n elif isinstance(b,dict):\n  assert a.keys()==b.keys()\n  for k in b:same(a[k],b[k])\n elif isinstance(b,list):\n  assert len(a)==len(b)\n  for x,y in zip(a,b):same(x,y)\n else:assert a==b\nsame(r,expected)',
            f,
          ],
          { cwd: path.resolve(__dirname, ".."), encoding: "utf8" },
        );
        assert.equal(py.status, 0, py.stderr);
        if (process.env.TRANSPORT_MARGINAL_ARCHIVES) {
          fs.mkdirSync(process.env.TRANSPORT_MARGINAL_ARCHIVES, {
            recursive: true,
          });
          fs.copyFileSync(
            f,
            path.join(process.env.TRANSPORT_MARGINAL_ARCHIVES, name + ".json"),
          );
        }
        fs.unlinkSync(f);
      }
      console.log(
        JSON.stringify({
          controls: 12,
          updates,
          cells,
          layouts,
          completedNativeReplayedDownloads: 2,
        }),
      );
      if (process.env.TRANSPORT_MARGINAL_SCREENSHOTS) {
        fs.mkdirSync(process.env.TRANSPORT_MARGINAL_SCREENSHOTS, {
          recursive: true,
        });
        for (const name of [
          "early-weight-edit",
          "normalization-only",
          "rectangular",
        ])
          for (const width of [390, 1280]) {
            await js(
              `document.getElementById('control').value='${name}';MARGINAL_APP.restore();document.querySelector('.input-panel details').open=false;document.activeElement.blur()`,
            );
            await call("Emulation.setDeviceMetricsOverride", {
              width,
              height: 1200,
              deviceScaleFactor: 1,
              mobile: width < 700,
            });
            for (const [part, y] of [
              ["top", 0],
              ["terms", 750],
              ["cells", 1600],
            ]) {
              await js(`scrollTo(0,${y})`);
              const image = await call("Page.captureScreenshot", {
                format: "png",
              });
              fs.writeFileSync(
                path.join(
                  process.env.TRANSPORT_MARGINAL_SCREENSHOTS,
                  name + "-" + width + "-" + part + ".png",
                ),
                Buffer.from(image.data, "base64"),
              );
            }
          }
      }
    } finally {
      if (ws) ws.close();
      if (browser.pid && browser.exitCode === null) {
        browser.kill("SIGTERM");
        await new Promise((res) => browser.once("exit", res));
      }
      await fs.promises.rm(profile, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    }
  },
);
