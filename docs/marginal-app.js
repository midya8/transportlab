"use strict";
(function () {
  const $ = (id) => document.getElementById(id),
    make = (tag, text) => {
      const el = document.createElement(tag);
      if (text !== undefined) el.textContent = text;
      return el;
    },
    fmt = (x) => (x === null ? "excluded" : Number(x).toPrecision(9));
  let result = null,
    url = null;
  const notes = {
    "early-weight-edit":
      "One update: actual rows and their tangents differ from the requested normalized marginals. The envelope-like candidate omits a visible residual-tangent term.",
    "later-weight-edit":
      "Same local raw-weight direction after50 updates. Numerical residuals and the candidate difference are small here. This is a disclosed control, not a convergence guarantee.",
    "joint-direction":
      "Raw source/target weights, costs and epsilon all move together. The full derivative includes normalization and every finite scaling update.",
    "normalization-only":
      "Each raw direction is proportional to its own raw vector. Normalization removes the change completely, even though this finite plan is not yet feasible.",
    "zero-cost-product":
      "Zero costs give the product plan after one complete update. A closed-form normalized-weight reference checks every cell derivative.",
    underflow:
      "Extreme but accepted cost/epsilon ratio: off-diagonal masses and tangents underflow. Log masses remain finite. Floating zeros do not prove zero real mass.",
    "zero-support":
      "Zero raw weights have zero directions and remain excluded. Potentials/log masses on excluded support are undefined, shown as excluded.",
  };
  for (const c of MARGINAL_DATA) {
    const o = make("option", c.name.replaceAll("-", " "));
    o.value = c.name;
    $("control").append(o);
  }
  $("control").value = "early-weight-edit";
  function table(id, headers, rows) {
    const t = make("table"),
      head = make("thead"),
      tr = make("tr");
    for (const h of headers) {
      const th = make("th", h);
      th.scope = "col";
      tr.append(th);
    }
    head.append(tr);
    t.append(head);
    const body = make("tbody");
    for (const row of rows) {
      const tr = make("tr");
      row.forEach((x, i) => {
        const td = make("td");
        td.dataset.label = headers[i];
        td.append(x instanceof Node ? x : document.createTextNode(String(x)));
        tr.append(td);
      });
      body.append(tr);
    }
    t.append(body);
    $(id).replaceChildren(t);
  }
  function show() {
    if (!result) return;
    const k = Number($("step").value) - 1,
      e = result.history[k],
      r = result;
    globalThis.MARGINAL_SELECTED = e;
    $("step-label").textContent =
      "Update " + e.iteration + " of " + r.input.iterations;
    for (const [id, key] of [
      ["full", "full_tangent"],
      ["candidate", "envelope_candidate"],
      ["missing", "residual_tangent_terms"],
    ])
      $(id).textContent = e[key].toPrecision(9);
    $("identity").textContent =
      "Signed identity residual: full − candidate − residual term = " +
      e.envelope_identity_signed_residual.toExponential(3) +
      ". This is a floating diagnostic, not a certified real-arithmetic bound.";
    table(
      "terms",
      ["Term", "Value"],
      [
        ["Objective F", fmt(e.regularized_objective)],
        ["Direct cost/epsilon terms", fmt(e.direct_terms)],
        [
          "Requested normalized-marginal terms",
          fmt(e.requested_marginal_terms),
        ],
        ["Actual mass-tangent terms", fmt(e.actual_mass_terms)],
        [
          "Full derivative minus candidate",
          fmt(e.full_tangent - e.envelope_candidate),
        ],
        ["Underflowed active cells", e.underflowed_active_entries],
      ],
    );
    const weights = [],
      mass = [],
      pot = [];
    for (const side of ["source", "target"]) {
      for (let i = 0; i < r[side].length; i++) {
        weights.push([
          side + " " + i,
          fmt(r.input[side + "_weights"][i]),
          fmt(r.input[side + "_direction"][i]),
          fmt(r[side][i]),
          fmt(r[side + "_tangent"][i]),
        ]);
        mass.push([
          side + " " + i,
          fmt(r[side][i]),
          fmt(e["actual_" + side][i]),
          fmt(r[side + "_tangent"][i]),
          fmt(e["actual_" + side + "_tangent"][i]),
        ]);
        pot.push([
          side + " " + i,
          fmt(e[side + "_potential"][i]),
          fmt(e[side + "_potential_tangent"][i]),
        ]);
      }
    }
    table(
      "weights",
      ["Weight", "Raw w", "Raw dw", "Normalized mass", "Normalized tangent"],
      weights,
    );
    table(
      "mass-table",
      [
        "Constraint",
        "Requested mass",
        "Actual mass",
        "Requested tangent",
        "Actual tangent",
      ],
      mass,
    );
    table("potentials", ["Potential", "Value", "Derivative"], pot);
    const cells = [];
    for (let i = 0; i < e.plan.length; i++)
      for (let j = 0; j < e.plan[i].length; j++)
        cells.push([
          i + " → " + j,
          fmt(r.input.cost[i][j]),
          fmt(r.input.cost_direction[i][j]),
          fmt(e.plan[i][j]),
          fmt(e.plan_tangent[i][j]),
          fmt(e.log_plan[i][j]),
          fmt(e.log_plan_tangent[i][j]),
        ]);
    table(
      "cell-table",
      [
        "Cell",
        "Cost",
        "Cost direction",
        "Plan mass",
        "Plan tangent",
        "Log mass",
        "Log tangent",
      ],
      cells,
    );
    table(
      "residuals",
      ["Constraint diagnostic", "Maximum absolute residual"],
      [
        ["Row mass", fmt(e.row_residual)],
        ["Column mass", fmt(e.column_residual)],
        ["Row tangent", fmt(e.row_tangent_residual)],
        ["Column tangent", fmt(e.column_tangent_residual)],
      ],
    );
    document
      .querySelectorAll(".step-button")
      .forEach((b) =>
        b.setAttribute(
          "aria-pressed",
          Number(b.dataset.step) === e.iteration ? "true" : "false",
        ),
      );
  }
  function invalidate(message) {
    result = null;
    globalThis.MARGINAL_SELECTED = null;
    if (url) {
      URL.revokeObjectURL(url);
      url = null;
    }
    $("error").textContent = message;
    $("report").value = "";
    $("status").textContent = "";
    $("identity").textContent = "";
    $("step").disabled = true;
    $("step-label").textContent = "-";
    for (const id of ["full", "candidate", "missing"]) $(id).textContent = "-";
    for (const id of [
      "terms",
      "weights",
      "mass-table",
      "potentials",
      "cell-table",
      "residuals",
      "history",
    ])
      $(id).replaceChildren();
    $("copy").disabled = true;
    $("download").disabled = true;
  }
  function update() {
    invalidate("");
    try {
      result = MARGINAL_ENGINE.audit(MARGINAL_JSON.parse($("input").value));
    } catch (err) {
      invalidate(err.message);
      return;
    }
    $("step").disabled = false;
    $("step").max = String(result.history.length);
    $("step").value = String(result.history.length);
    $("copy").disabled = false;
    $("download").disabled = false;
    table(
      "history",
      [
        "Update",
        "Row error",
        "Row tangent error",
        "Column tangent error",
        "Full derivative",
        "Candidate",
        "Difference",
      ],
      result.history.map((e) => {
        const b = make("button", String(e.iteration));
        b.type = "button";
        b.className = "step-button";
        b.dataset.step = String(e.iteration);
        b.onclick = () => {
          $("step").value = String(e.iteration);
          show();
        };
        return [
          b,
          ...[
            "row_residual",
            "row_tangent_residual",
            "column_tangent_residual",
            "full_tangent",
            "envelope_candidate",
          ].map((k) => fmt(e[k])),
          fmt(e.full_tangent - e.envelope_candidate),
        ];
      }),
    );
    if ($("report-details").open)
      $("report").value = JSON.stringify(result, null, 2);
    show();
  }
  function restore() {
    const c = MARGINAL_DATA.find((c) => c.name === $("control").value);
    $("input").value = JSON.stringify(c.input, null, 2);
    $("note").textContent =
      notes[c.name] ||
      "Synthetic numerical control with explicitly supplied local directions. No learned matching quality, causal influence or general numerical guarantee.";
    update();
  }
  $("input").addEventListener("input", update);
  $("control").addEventListener("change", restore);
  $("restore").onclick = restore;
  $("step").addEventListener("input", show);
  $("report-details").addEventListener("toggle", () => {
    if (result && $("report-details").open)
      $("report").value = JSON.stringify(result, null, 2);
  });
  $("copy").onclick = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      $("status").textContent = "Copied complete input and history.";
    } catch {
      $("status").textContent = "Copy was blocked. Use the download instead.";
    }
  };
  $("download").onclick = () => {
    if (!result) return;
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2) + "\n"], {
        type: "application/json",
      }),
    );
    const a = make("a");
    a.href = url;
    a.download = "transportlab-marginal-report.json";
    document.body.append(a);
    a.click();
    a.remove();
    $("status").textContent = "Downloaded complete input and history.";
  };
  globalThis.MARGINAL_APP = { getReport: () => result, update, restore };
  restore();
})();
