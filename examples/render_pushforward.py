"""Weighted atoms and conditional means are distinct geometric objects."""
import json
import math
import sys
from pathlib import Path


def render(r):
    audit=next(a for a in r['pushforward_audit'] if a['epsilon']==2.)
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500" role="img" aria-label="TransportLab joint atom pushforward versus barycentric mean collapse">',
       '<rect width="960" height="500" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="32" font-size="23">TransportLab | moving mass is not moving its mean</text>',
       '<text x="24" y="57" font-size="13">Same converged epsilon2 coupling. Every pair keeps its mass at (1-t)*source + t*target.</text>']
    for j,s in enumerate(audit['slices']):
        ox,oy=40+310*j,104
        p.extend([f'<text x="{ox}" y="86" font-size="16" font-weight="600">t = {s["time"]:g}</text>',f'<rect x="{ox}" y="{oy}" width="260" height="220" fill="white" stroke="#ddd"/>'])
        def xy(v):return ox+(v[0]+1.5)/4.5*260,oy+(1.4-v[1])/3*220
        for x in [-1,0,1,2]:
            xx,_=xy([x,0]);p.append(f'<path d="M{xx},{oy}v220" stroke="#eae9ea"/><text x="{xx}" y="{oy+237}" text-anchor="middle" font-size="11">{x}</text>')
        for y in [-1,0,1]:
            _,yy=xy([0,y]);p.append(f'<path d="M{ox},{yy}h260" stroke="#eae9ea"/><text x="{ox-5}" y="{yy+4}" text-anchor="end" font-size="10">{y}</text>')
        # Aggregate exact coincident coordinates only for area-correct plotting.
        groups={}
        for a in s['atoms']:
            v=tuple(a['point']);groups[v]=groups.get(v,0.)+a['mass']
        for point,mass in sorted(groups.items()):
            if mass==0:continue
            xx,yy=xy(point);radius=20*math.sqrt(mass/s['total_mass'])
            p.append(f'<circle cx="{xx:.3f}" cy="{yy:.3f}" r="{radius:.3f}" fill="#585254" opacity=".55"/>')
        for a in s['barycentric_atoms']:
            if not a['mass']:continue
            xx,yy=xy(a['point']);p.append(f'<path d="M{xx-4:.3f},{yy:.3f}h8 M{xx:.3f},{yy-4:.3f}v8" stroke="#387667" stroke-width="2"/>')
        p.extend([f'<text x="{ox}" y="364" font-size="12">Full atom variance trace: {s["full_moments"]["variance_trace"]:.6f}</text>',f'<text x="{ox}" y="385" font-size="12">Mean-map variance trace: {s["barycentric_moments"]["variance_trace"]:.6f}</text>'])
    p.extend(['<text x="24" y="419" font-size="13">Gray circle area follows joint mass. Exact coincident atoms aggregated for display. Teal +: one mean per source.</text>',
              '<text x="24" y="442" font-size="13">Same axes: x[-1.5,3], y[-1.6,1.4]. All36 pair atoms and six mean rows remain in JSON; no mass threshold.</text>',
              '<text x="24" y="465" font-size="13">At t=1 full atoms reproduce actual target column masses; mean-map variance is smaller. Neither is a sampled path.</text>',
              '<text x="24" y="488" font-size="13">Moment decomposition is an arithmetic identity, not coupling optimality, geometry recovery or a causal claim.</text>','</g></svg>'])
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    text=render(json.loads(Path(sys.argv[1]).read_text()));dest=Path('assets/pushforward-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==text,'stale pushforward-audit.svg'
    else:dest.write_text(text)
