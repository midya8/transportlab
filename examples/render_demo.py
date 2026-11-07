"""All plan cells plus point-mean mapping; no edge threshold hides plan mass."""
import json
import sys
from pathlib import Path


def render(r):
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="605" viewBox="0 0 960 605" role="img" aria-label="TransportLab entropic transport plans and convergence audit">',
       '<rect width="960" height="605" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="33" font-size="24">TransportLab | cost, entropy and feasibility are different</text>',
       '<text x="24" y="59" font-size="13">Six uniform source/target points. Heatmaps show every plan cell numerically, shared scale0 to1/6.</text>']
    for k,arm in enumerate(r['arms']):
        ox,oy=40+310*k,104;s=arm['result'];size=34
        p.append(f'<text x="{ox}" y="86" font-size="16" font-weight="600">epsilon = {arm["epsilon"]:g}</text>')
        for i,row in enumerate(s['plan']):
            for j,value in enumerate(row):
                # White to teal, same mass scale for all panels.
                t=min(1.,value*6);color='rgb('+','.join(str(round(255+(c-255)*t)) for c in [115,168,154])+')'
                p.extend([f'<rect x="{ox+j*size}" y="{oy+i*size}" width="34" height="34" fill="{color}" stroke="#eae9ea"/>',f'<text x="{ox+j*size+17}" y="{oy+i*size+20}" text-anchor="middle" font-size="9">{value:.3f}</text>'])
        for j in range(6):p.append(f'<text x="{ox+j*size+17}" y="{oy-5}" text-anchor="middle" font-size="10">t{j}</text>')
        for i in range(6):p.append(f'<text x="{ox-6}" y="{oy+i*size+20}" text-anchor="end" font-size="10">s{i}</text>')
        lines=[f'Converged: {str(s["converged"]).lower()} | iterations: {s["iterations"]}',f'Transport cost: {s["transport_cost"]:.6f}',f'Entropy: {s["entropy_nats"]:.6f} nats',f'Regularized objective: {s["regularized_objective"]:.6f}',f'Max row residual: {s["row_max_abs_residual"]:.2e}']
        for j,line in enumerate(lines):p.append(f'<text x="{ox}" y="{332+20*j}" font-size="12">{line}</text>')
    exact=r['exact_uniform_assignment'];ind=r['independence']
    p.extend([f'<text x="24" y="455" font-size="13">Unregularized uniform assignment: {exact["transport_cost"]:.6f}, all720 permutations. Independence cost: {ind["transport_cost"]:.6f}.</text>',
              '<text x="24" y="480" font-size="13">epsilon0.1 reached the 5,000-iteration cap. Its slightly infeasible cost is not an optimality certificate.</text>',
              '<text x="24" y="505" font-size="13">Objective = cost - epsilon*entropy. Different epsilons have different objectives; do not rank those scalars.</text>',
              f'<text x="24" y="530" font-size="13">Self-transport cost at epsilon2: {r["self_transport_epsilon2"]["transport_cost"]:.6f}, not zero. This is not a debiased Sinkhorn divergence.</text>',
              '<text x="24" y="555" font-size="13">Squared Euclidean ground cost is not a metric. No learned correspondence, downstream task or wall-clock speed claim.</text>',
              '<text x="24" y="580" font-size="13">Full plans, marginal history and barycentric target means in JSON. Means are not target-distributed samples.</text>','</g></svg>'])
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    text=render(json.loads(Path(sys.argv[1]).read_text()));dest=Path('assets/transport-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==text,'stale transport-audit.svg'
    else:dest.write_text(text)
