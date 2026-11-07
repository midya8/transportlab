import argparse
import json
from .core import sinkhorn,squared_cost,barycentric,exact_uniform_assignment,pushforward
from .demo import demo
from .audit import audit_transport
from .sensitivity import differentiate_iterations


def read(path):
    with open(path,encoding='utf8') as f:return json.load(f)


def main():
    p=argparse.ArgumentParser(description='Auditable offline log-domain entropic transport')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('demo')
    q=sub.add_parser('sensitivity');q.add_argument('input');q.add_argument('--epsilon',type=float,default=.5);q.add_argument('--iterations',type=int,default=100)
    q=sub.add_parser('solve');q.add_argument('input');q.add_argument('--epsilon',type=float,default=.5)
    q.add_argument('--max-iterations',type=int,default=5000);q.add_argument('--tolerance',type=float,default=1e-9)
    q=sub.add_parser('audit');q.add_argument('input');q.add_argument('--epsilon',type=float,default=.5);q.add_argument('--max-iterations',type=int,default=5000);q.add_argument('--tolerance',type=float,default=1e-9)
    q=sub.add_parser('assignment');q.add_argument('input')
    q=sub.add_parser('push');q.add_argument('input');q.add_argument('--epsilon',type=float,default=.5);q.add_argument('--time',type=float,default=.5)
    a=p.parse_args()
    try:
        if a.command=='demo':r=demo()
        elif a.command=='sensitivity':
            v=read(a.input)
            if not isinstance(v,dict) or set(v)!={'cost','cost_direction','source_weights','target_weights','epsilon_direction'}:raise ValueError('sensitivity input requires exact cost/cost_direction/source_weights/target_weights/epsilon_direction fields')
            r=differentiate_iterations(v['cost'],v['source_weights'],v['target_weights'],v['cost_direction'],a.epsilon,v['epsilon_direction'],a.iterations)
        else:
            v=read(a.input)
            if not isinstance(v,dict) or set(v)!={'source_points','target_points','source_weights','target_weights'}:raise ValueError('input requires exact source_points/target_points/source_weights/target_weights fields')
            c=squared_cost(v['source_points'],v['target_points'])
            if a.command=='assignment':
                # The reference explicitly needs equal uniform weights, not arbitrary input weights.
                n=len(c)
                from .core import weights
                x,_=weights(v['source_weights'],'source');y,_=weights(v['target_weights'],'target')
                if len(x)!=n or len(y)!=n or any(abs(z-1/n)>1e-12 for z in x+y):raise ValueError('assignment requires equal-size uniform weights')
                r=exact_uniform_assignment(c)
            elif a.command=='audit':
                r=audit_transport(c,v['source_weights'],v['target_weights'],a.epsilon,a.max_iterations,a.tolerance)
            elif a.command=='push':
                r=sinkhorn(c,v['source_weights'],v['target_weights'],a.epsilon)
                if not r['converged']:raise ValueError('push CLI requires a converged plan; use solve to inspect residuals')
                r={'solver':r,'pushforward':pushforward(r['plan'],v['source_points'],v['target_points'],a.time)}
            else:
                r=sinkhorn(c,v['source_weights'],v['target_weights'],a.epsilon,a.max_iterations,a.tolerance)
                r['barycentric']=barycentric(r['plan'],v['target_points'])
        print(json.dumps(r,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,f'transportlab: {e}\n')


if __name__=='__main__':main()
