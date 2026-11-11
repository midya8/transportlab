"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  { spawnSync } = require("node:child_process");
const E = require("../docs/marginal-engine.js"),
  J = require("../docs/marginal-json.js");
const controls = fs
  .readdirSync(path.join(__dirname, "../transportlab/artifacts"))
  .filter((p) => p.startsWith("marginal-"))
  .sort()
  .map((p) =>
    JSON.parse(
      fs.readFileSync(path.join(__dirname, "../transportlab/artifacts", p)),
    ),
  );
function same(a, b, p = "report") {
  if (typeof b === "number") {
    assert.equal(typeof a, "number", p);
    assert.ok(
      Number.isFinite(a) && Math.abs(a - b) <= 2e-9 * (1 + Math.abs(b)),
      p + ": " + a + " vs " + b,
    );
  } else if (Array.isArray(b)) {
    assert.ok(Array.isArray(a), p);
    assert.equal(a.length, b.length, p);
    b.forEach((v, i) => same(a[i], v, p + "." + i));
  } else if (b && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), p);
    for (const k of Object.keys(b)) same(a[k], b[k], p + "." + k);
  } else assert.equal(a, b, p);
}
test("12 complete native versus independent dual-number reports at every update", () => {
  assert.equal(controls.length, 12);
  for (const c of controls) same(E.audit(c.input), c.report, c.name);
});
test("normalization null direction, finite residual and underflow controls", () => {
  const by = Object.fromEntries(
    controls.map((c) => [c.name, E.audit(c.input)]),
  );
  assert.deepEqual(by["normalization-only"].source_tangent, [0, 0]);
  assert.equal(by["normalization-only"].final.full_tangent, 0);
  assert.ok(
    Math.abs(
      by["early-weight-edit"].final.full_tangent -
        by["early-weight-edit"].final.envelope_candidate,
    ) > 0.02,
  );
  assert.ok(
    Math.abs(
      by["later-weight-edit"].final.full_tangent -
        by["later-weight-edit"].final.envelope_candidate,
    ) < 1e-10,
  );
  assert.equal(by.underflow.final.underflowed_active_entries, 2);
});
test("strict parsing, support and schema restrictions", () => {
  for (const text of [
    '{"a":1,"a":2}',
    "[1,]",
    "{} trailing",
    '{"x":1}\u00a0',
    " ".repeat(65537),
  ])
    assert.throws(() => J.parse(text));
  for (const [key, value] of [
    ["source_weights", [0, 0]],
    ["source_direction", [true, 1]],
    ["target_weights", [1e-5, 1]],
    ["iterations", 501],
    ["iterations", 1.2],
    ["epsilon", 0],
    ["cost", [[0]]],
    ["epsilon_direction", 101],
    ["extra", 1],
  ]) {
    const inp = structuredClone(controls[0].input);
    inp[key] = value;
    assert.throws(() => E.audit(inp));
  }
  const inp = structuredClone(
    controls.find((c) => c.name === "early-weight-edit").input,
  );
  inp.source_weights = [0, 1];
  assert.throws(() => E.audit(inp));
});
test(
  "80 complete generated native/dual-number cases",
  { timeout: 120000 },
  () => {
    let seed = 10101;
    const rand = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    const cases = Array.from({ length: 80 }, () => {
      const n = 1 + Math.floor(rand() * 4),
        m = 1 + Math.floor(rand() * 4);
      return {
        cost: Array.from({ length: n }, () =>
          Array.from({ length: m }, () => rand() * 5),
        ),
        cost_direction: Array.from({ length: n }, () =>
          Array.from({ length: m }, () => rand() - 0.5),
        ),
        source_weights: Array.from({ length: n }, () => rand() + 0.1),
        source_direction: Array.from({ length: n }, () => rand() - 0.5),
        target_weights: Array.from({ length: m }, () => rand() + 0.1),
        target_direction: Array.from({ length: m }, () => rand() - 0.5),
        epsilon: 0.3 + rand(),
        epsilon_direction: rand() - 0.5,
        iterations: 1 + Math.floor(rand() * 15),
      };
    });
    const py = spawnSync(
      "python",
      [
        "-c",
        "import sys,json\nfrom transportlab.marginal import audit\nprint(json.dumps([audit(x) for x in json.load(sys.stdin)],allow_nan=False))",
      ],
      {
        cwd: path.resolve(__dirname, ".."),
        input: JSON.stringify(cases),
        encoding: "utf8",
        maxBuffer: 16 * 1024 * 1024,
      },
    );
    assert.equal(py.status, 0, py.stderr);
    const refs = JSON.parse(py.stdout);
    cases.forEach((x, i) => same(E.audit(x), refs[i], "random" + i));
  },
);

test("full4x4matrix and500updates remain inspectable", () => {
  const x = {
    cost: Array.from({ length: 4 }, (_, i) =>
      Array.from({ length: 4 }, (_, j) => 0.2 + i + j),
    ),
    cost_direction: Array.from({ length: 4 }, () => Array(4).fill(0.1)),
    source_weights: [1, 2, 3, 4],
    source_direction: [0.1, -0.2, 0.3, -0.4],
    target_weights: [4, 3, 2, 1],
    target_direction: [0.4, -0.3, 0.2, -0.1],
    epsilon: 0.5,
    epsilon_direction: 0.1,
    iterations: 500,
  };
  const r = E.audit(x);
  assert.equal(r.history.length, 500);
  assert.ok(Math.abs(r.final.envelope_identity_signed_residual) < 1e-10);
  assert.ok(r.final.row_tangent_residual < 1e-10);
});
