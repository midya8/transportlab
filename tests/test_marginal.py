import copy, importlib.util, itertools, json, math, random, subprocess, sys, unittest
from decimal import Decimal as D, localcontext
from pathlib import Path
from transportlab.marginal import audit, parse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "controls", ROOT / "examples/marginal_controls.py"
)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def decimal_primal(inp, step):
    # Independent ordinary positive-kernel matrix scaling. No tangent code,
    # no producer parser, no log scalings, no gauge and no early stopping.
    with localcontext() as ctx:
        ctx.prec = 80
        h = D(str(step))
        eps = D(str(inp["epsilon"])) + h * D(str(inp["epsilon_direction"]))

        def norm(w, d):
            xs = [D(str(x)) + h * D(str(v)) for x, v in zip(w, d)]
            total = sum(xs)
            return [x / total for x in xs]

        a = norm(inp["source_weights"], inp["source_direction"])
        b = norm(inp["target_weights"], inp["target_direction"])
        c = [
            [D(str(x)) + h * D(str(v)) for x, v in zip(row, drow)]
            for row, drow in zip(inp["cost"], inp["cost_direction"])
        ]
        k = [[(-x / eps).exp() for x in row] for row in c]
        n = len(a)
        m = len(b)
        v = [D(1)] * m
        for _ in range(inp["iterations"]):
            u = [
                a[i] / sum(k[i][j] * v[j] for j in range(m)) if a[i] else D(0)
                for i in range(n)
            ]
            v = [
                b[j] / sum(k[i][j] * u[i] for i in range(n)) if b[j] else D(0)
                for j in range(m)
            ]
        p = [[u[i] * k[i][j] * v[j] for j in range(m)] for i in range(n)]
        transport = sum(p[i][j] * c[i][j] for i in range(n) for j in range(m))
        F = transport + eps * sum((x * x.ln() for row in p for x in row if x), D(0))
        return p, F, transport


class MarginalTests(unittest.TestCase):
    def test_all12_saved_reports_reproduce(self):
        self.assertEqual(len(controls.controls()), 12)
        for c in controls.controls():
            saved = json.loads(
                (
                    ROOT / ("transportlab/artifacts/marginal-" + c["name"] + ".json")
                ).read_text()
            )
            self.assertEqual(saved, dict(c, report=audit(c["input"])))

    def test_all_control_updates_against_decimal80_primal_and_derivatives(self):
        checked = 0
        for c in controls.controls():
            if c["name"] == "underflow":
                continue
            inp = copy.deepcopy(c["input"])
            r = audit(inp)
            for t in [1, min(3, inp["iterations"]), inp["iterations"]]:
                inp["iterations"] = t
                entry = r["history"][t - 1]
                p, F, transport = decimal_primal(inp, "0")
                lo = decimal_primal(inp, "-1e-20")
                hi = decimal_primal(inp, "1e-20")
                with localcontext() as ctx:
                    ctx.prec = 80
                    for i, row in enumerate(p):
                        for j, x in enumerate(row):
                            self.assertAlmostEqual(
                                entry["plan"][i][j], float(x), places=10
                            )
                            self.assertAlmostEqual(
                                entry["plan_tangent"][i][j],
                                float((hi[0][i][j] - lo[0][i][j]) / D("2e-20")),
                                places=9,
                            )
                    self.assertAlmostEqual(
                        entry["regularized_objective"], float(F), places=10
                    )
                    self.assertAlmostEqual(
                        entry["full_tangent"],
                        float((hi[1] - lo[1]) / D("2e-20")),
                        places=9,
                    )
                    self.assertAlmostEqual(
                        entry["transport_tangent"],
                        float((hi[2] - lo[2]) / D("2e-20")),
                        places=9,
                    )
                checked += 1
        self.assertEqual(checked, 33)

    def test_60_random_rectangular_decimal_references(self):
        rng = random.Random(20261111)
        for _ in range(60):
            n = rng.randrange(1, 5)
            m = rng.randrange(1, 5)
            inp = dict(
                cost=[[rng.random() * 3 for _ in range(m)] for _ in range(n)],
                cost_direction=[
                    [rng.random() - 0.5 for _ in range(m)] for _ in range(n)
                ],
                source_weights=[rng.random() + 0.1 for _ in range(n)],
                target_weights=[rng.random() + 0.1 for _ in range(m)],
                source_direction=[rng.random() - 0.5 for _ in range(n)],
                target_direction=[rng.random() - 0.5 for _ in range(m)],
                epsilon=0.6,
                epsilon_direction=0.1,
                iterations=rng.randrange(1, 8),
            )
            r = audit(inp)["final"]
            lo = decimal_primal(inp, "-1e-18")
            hi = decimal_primal(inp, "1e-18")
            with localcontext() as ctx:
                ctx.prec = 80
                self.assertAlmostEqual(
                    r["full_tangent"], float((hi[1] - lo[1]) / D("2e-18")), places=9
                )
                for i in range(n):
                    for j in range(m):
                        self.assertAlmostEqual(
                            r["plan_tangent"][i][j],
                            float((hi[0][i][j] - lo[0][i][j]) / D("2e-18")),
                            places=9,
                        )

    def test_normalization_quotient_and_scale_null_direction(self):
        inp = controls.controls()[3]["input"]
        r = audit(inp)
        self.assertEqual(r["source_tangent"], [0.0, 0.0])
        self.assertEqual(r["target_tangent"], [0.0, 0.0])
        for e in r["history"]:
            self.assertEqual(e["plan_tangent"], [[0.0, 0.0], [0.0, 0.0]])
            self.assertEqual(e["full_tangent"], 0.0)
        inp = controls.controls()[0]["input"]
        r = audit(inp)
        self.assertEqual(r["source_tangent"], [0.1, -0.1])
        self.assertEqual(r["target_tangent"], [0.1, -0.1])

    def test_zero_cost_product_closed_form(self):
        r = audit(controls.controls()[4]["input"])
        a = r["source"]
        b = r["target"]
        da = r["source_tangent"]
        db = r["target_tangent"]
        e = r["final"]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(e["plan"][i][j], a[i] * b[j], 14)
                self.assertAlmostEqual(
                    e["plan_tangent"][i][j], da[i] * b[j] + a[i] * db[j], 14
                )
        expected = 0.5 * (
            sum(x * math.log(y) for x, y in zip(da, a))
            + sum(x * math.log(y) for x, y in zip(db, b))
        )
        self.assertAlmostEqual(e["full_tangent"], expected, 14)

    def test_finite_residual_chain_terms_not_optimum_claim(self):
        r = audit(controls.controls()[0]["input"])["final"]
        self.assertGreater(r["row_residual"], 0.3)
        self.assertGreater(abs(r["full_tangent"] - r["envelope_candidate"]), 0.02)
        self.assertLess(abs(r["envelope_identity_signed_residual"]), 1e-12)
        inp = controls.controls()[0]["input"]
        inp["iterations"] = 10
        r = audit(inp)["final"]
        self.assertLess(r["row_residual"], 2e-8)
        self.assertGreater(r["row_tangent_residual"], 1e-7)

    def test_all_iteration_chain_mass_identities(self):
        for c in controls.controls():
            r = audit(c["input"])
            self.assertAlmostEqual(sum(r["source_tangent"]), 0.0, 12)
            self.assertAlmostEqual(sum(r["target_tangent"]), 0.0, 12)
            for e in r["history"]:
                self.assertLess(abs(e["chain_identity_signed_residual"]), 1e-10)
                self.assertLess(abs(e["envelope_identity_signed_residual"]), 1e-10)
                self.assertAlmostEqual(sum(e["actual_source"]), 1, 10)
                self.assertAlmostEqual(sum(e["actual_source_tangent"]), 0, 10)
                self.assertLess(e["column_tangent_residual"], 1e-10)

    def test_zero_support_and_underflow_are_explicit(self):
        r = audit(controls.controls()[7]["input"])
        self.assertEqual(r["active_rows"], [1])
        self.assertEqual(r["active_columns"], [0, 2])
        self.assertIsNone(r["final"]["source_potential"][0])
        self.assertEqual(r["final"]["plan_tangent"][0], [0.0, 0.0, 0.0])
        r = audit(controls.controls()[8]["input"])
        self.assertEqual(r["final"]["underflowed_active_entries"], 2)
        self.assertEqual(r["final"]["plan_tangent"][0][1], 0)

    def test_cost_shift_and_joint_scale_controls(self):
        for k in [10, 11]:
            r = audit(controls.controls()[k]["input"])["final"]
            self.assertLess(
                max(abs(v) for row in r["plan_tangent"] for v in row), 1e-12
            )
            self.assertAlmostEqual(
                r["full_tangent"], 1 if k == 10 else r["regularized_objective"], 12
            )

    def test_original_fixed_marginal_sensitivity_agrees(self):
        from transportlab.sensitivity import differentiate_iterations

        for c in controls.controls():
            x = copy.deepcopy(c["input"])
            x["source_direction"] = [0] * len(x["source_weights"])
            x["target_direction"] = [0] * len(x["target_weights"])
            r = audit(x)
            a = r["source"]
            b = r["target"]
            old = differentiate_iterations(
                x["cost"],
                a,
                b,
                x["cost_direction"],
                x["epsilon"],
                x["epsilon_direction"],
                x["iterations"],
            )
            e = r["final"]
            self.assertAlmostEqual(
                e["full_tangent"],
                old["regularized_objective_directional_derivative"],
                10,
            )
            for i, row in enumerate(e["plan_tangent"]):
                for j, v in enumerate(row):
                    self.assertAlmostEqual(
                        v, old["plan_directional_derivative"][i][j], 10
                    )

    def test_row_column_permutation(self):
        x = copy.deepcopy(controls.controls()[2]["input"])
        r = audit(x)["final"]
        for key in [
            "source_weights",
            "source_direction",
            "target_weights",
            "target_direction",
        ]:
            x[key] = x[key][::-1]
        for key in ["cost", "cost_direction"]:
            x[key] = [row[::-1] for row in x[key][::-1]]
        q = audit(x)["final"]
        self.assertAlmostEqual(r["full_tangent"], q["full_tangent"], 12)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    r["plan_tangent"][i][j], q["plan_tangent"][1 - i][1 - j], 12
                )

    def test_direction_linearity(self):
        x = copy.deepcopy(controls.controls()[2]["input"])
        r = audit(x)["final"]
        y = copy.deepcopy(x)
        for key in ["source_direction", "target_direction"]:
            y[key] = [2 * v for v in y[key]]
        y["cost_direction"] = [[2 * v for v in row] for row in y["cost_direction"]]
        y["epsilon_direction"] *= 2
        q = audit(y)["final"]
        self.assertAlmostEqual(q["full_tangent"], 2 * r["full_tangent"], 12)

    def test_shapes_bounds_fixed_support_and_no_mutation(self):
        x = copy.deepcopy(controls.controls()[0]["input"])
        old = copy.deepcopy(x)
        audit(x)
        self.assertEqual(x, old)
        for key, val in [
            ("source_weights", [0, 0]),
            ("source_weights", [1e-5, 1]),
            ("source_direction", [True, 0]),
            ("source_direction", [101, 0]),
            ("target_weights", [float("nan"), 1]),
            ("cost", [[1]]),
            ("cost_direction", [[float("inf")] * 2] * 2),
            ("epsilon", 0),
            ("epsilon_direction", 101),
            ("iterations", True),
            ("iterations", 0),
            ("iterations", 501),
            ("extra", 1),
        ]:
            bad = copy.deepcopy(x)
            bad[key] = val
            with self.assertRaises(ValueError, msg=key):
                audit(bad)
        bad = copy.deepcopy(x)
        bad["source_weights"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "fixed support"):
            audit(bad)

    def test_extreme_boundaries_and_large_integers_reject_cleanly(self):
        x = copy.deepcopy(controls.controls()[0]["input"])
        for key, value in [
            ("epsilon", 10**10000),
            ("source_direction", [-(10**10000), 0]),
            ("source_weights", [10**10000, 1]),
        ]:
            bad = copy.deepcopy(x)
            bad[key] = value
            with self.assertRaises(ValueError):
                audit(bad)
        # Formal directions are allowed at cost/epsilon bounds. They need not
        # define a two-sided admissible finite step in the public domain.
        x["cost"] = [[0, 100], [100, 0]]
        x["cost_direction"] = [[-100, 100], [-100, 100]]
        x["source_weights"] = [1e-4, 100]
        x["source_direction"] = [100, -100]
        x["epsilon"] = 0.01
        x["epsilon_direction"] = -100
        r = audit(x)
        self.assertEqual(r["final"]["underflowed_active_entries"], 2)
        self.assertTrue(math.isfinite(r["final"]["full_tangent"]))

    def test_full_bounded_work_and_input_totals(self):
        x = dict(
            cost=[[0.2 + i + j for j in range(4)] for i in range(4)],
            cost_direction=[[0.1] * 4 for _ in range(4)],
            source_weights=[1.0, 2.0, 3.0, 4.0],
            source_direction=[0.1, -0.2, 0.3, -0.4],
            target_weights=[4.0, 3.0, 2.0, 1.0],
            target_direction=[0.4, -0.3, 0.2, -0.1],
            epsilon=0.5,
            epsilon_direction=0.1,
            iterations=500,
        )
        r = audit(x)
        self.assertEqual(len(r["history"]), 500)
        self.assertEqual(r["raw_totals"], [10.0, 10.0])
        self.assertAlmostEqual(sum(r["source_tangent"]), 0, 12)
        self.assertLess(abs(r["final"]["chain_identity_signed_residual"]), 1e-10)
        self.assertLess(r["final"]["row_tangent_residual"], 1e-10)

    def test_cli_roundtrip_duplicate_size_and_nonfinite(self):
        cmd = [sys.executable, "-m", "transportlab.marginal", "-"]
        x = controls.controls()[0]["input"]
        r = subprocess.run(
            cmd, input=json.dumps(x), capture_output=True, text=True, cwd=ROOT
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), audit(x))
        for text in ['{"epsilon":1,"epsilon":2}', " " * 65537, '{"epsilon":NaN}']:
            r = subprocess.run(
                cmd, input=text, capture_output=True, text=True, cwd=ROOT
            )
            self.assertEqual(r.returncode, 2)
            self.assertEqual(r.stdout, "")
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
