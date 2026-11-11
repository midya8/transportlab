"""Rebuild the finite controls, without changing any earlier transport artifact."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transportlab.marginal import audit


def controls():
    def item(name, c, sw, tw, sd, td, dc=None, eps=0.5, de=0.0, N=10):
        return dict(
            name=name,
            input=dict(
                cost=c,
                source_weights=sw,
                target_weights=tw,
                source_direction=sd,
                target_direction=td,
                cost_direction=dc if dc is not None else [[0.0] * len(tw) for _ in sw],
                epsilon=eps,
                epsilon_direction=de,
                iterations=N,
            ),
        )

    c = [[0.2, 3.0], [1.0, 0.4]]
    dc = [[0.1, 0.8], [-0.4, 0.3]]
    return [
        item(
            "early-weight-edit",
            c,
            [9.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.2, -0.2],
            N=1,
        ),
        item(
            "later-weight-edit",
            c,
            [9.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.2, -0.2],
            N=50,
        ),
        item(
            "joint-direction",
            c,
            [9.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.2, -0.2],
            dc,
            de=0.2,
            N=3,
        ),
        item(
            "normalization-only", c, [9.0, 1.0], [1.0, 1.0], [9.0, 1.0], [2.0, 2.0], N=3
        ),
        item(
            "zero-cost-product",
            [[0.0, 0.0], [0.0, 0.0]],
            [2.0, 3.0],
            [1.0, 4.0],
            [0.1, 0.4],
            [-0.3, 0.2],
            N=1,
        ),
        item(
            "one-row",
            [[0.2, 1.0, 0.8]],
            [2.0],
            [1.0, 2.0, 3.0],
            [7.0],
            [0.2, -0.1, 0.7],
            [[0.1, -0.2, 0.3]],
            de=0.1,
            N=3,
        ),
        item(
            "one-column",
            [[0.2], [1.0], [0.8]],
            [1.0, 2.0, 3.0],
            [2.0],
            [0.2, -0.1, 0.7],
            [7.0],
            [[0.1], [-0.2], [0.3]],
            de=0.1,
            N=3,
        ),
        item(
            "zero-support",
            [[0.2, 3.0, 1.0], [1.0, 0.4, 0.7]],
            [0.0, 3.0],
            [1.0, 0.0, 4.0],
            [0.0, 0.2],
            [0.2, 0.0, -0.1],
            N=5,
        ),
        item(
            "underflow",
            [[0.0, 100.0], [100.0, 0.0]],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [[0.0, 1.0], [1.0, 0.0]],
            eps=0.01,
            N=3,
        ),
        item(
            "rectangular",
            [[0.4, 3.0, 2.0], [4.0, 1.0, 0.2]],
            [6.0, 4.0],
            [1.0, 3.0, 6.0],
            [0.2, -0.3],
            [0.1, -0.2, 0.7],
            [[0.3, -0.4, 0.5], [0.2, 0.6, -0.8]],
            eps=0.7,
            de=0.2,
            N=7,
        ),
        item(
            "uniform-cost-shift",
            c,
            [7.0, 3.0],
            [4.0, 6.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [[1.0, 1.0], [1.0, 1.0]],
            N=5,
        ),
        item(
            "cost-epsilon-scale",
            c,
            [7.0, 3.0],
            [4.0, 6.0],
            [0.0, 0.0],
            [0.0, 0.0],
            c,
            de=0.5,
            N=3,
        ),
    ]


def build(root):
    root = Path(root)
    out = [dict(c, report=audit(c["input"])) for c in controls()]
    (root / "transportlab/artifacts").mkdir(exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    for c in out:
        (root / ("transportlab/artifacts/marginal-" + c["name"] + ".json")).write_text(
            json.dumps(c, separators=(",", ":"), allow_nan=False) + "\n"
        )
    (root / "docs/marginal-data.js").write_text(
        "globalThis.MARGINAL_DATA="
        + json.dumps(controls(), separators=(",", ":"), allow_nan=False)
        + ";\n"
    )
    return out


if __name__ == "__main__":
    for c in build(Path(__file__).resolve().parents[1]):
        r = c["report"]["final"]
        print(
            c["name"],
            r["full_tangent"],
            r["envelope_candidate"],
            r["row_residual"],
            r["row_tangent_residual"],
        )
