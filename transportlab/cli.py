import argparse
import json
from .core import sinkhorn,squared_cost,barycentric,exact_uniform_assignment
from .demo import demo


def read(path):
    with open(path,encoding='utf8') as f:return json.load(f)


def main():
    p=argparse.ArgumentParser(description='Auditable offline log-domain entropic transport')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('demo')
    q=sub.add_parser('solve');q.add_argument('input');q.add_argument('--epsilon',type=float,default=.5)
    q.add_argument('--max-iterations',type=int,default=5000);q.add_argument('--tolerance',type=float,default=1e-9)
    q=sub.add_parser('assignment');q.add_argument('input')
    a=p.parse_args()
    try:
        if a.command=='demo':r=demo()
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
            else:
                r=sinkhorn(c,v['source_weights'],v['target_weights'],a.epsilon,a.max_iterations,a.tolerance)
                r['barycentric']=barycentric(r['plan'],v['target_points'])
        print(json.dumps(r,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,f'transportlab: {e}\n')


if __name__=='__main__':main()
