import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'research/bess_followup/scripts')
from pricemaker import Scripted
from easy_marl.examples.bidding import bess_experiment as E
T=24
R=Path('outputs/bess_sweep/full_carryover')
def run(seed,power,dur):
    cfg=E.SweepConfig(power,dur,seed,'frozen',episode_days=7)
    gens=E.get_pretrained(seed,E.PRESETS['full_carryover'],R)
    params=cfg.param_func()(N=4,T=T)
    bp=E.make_default_params(N=3,T=T,day_ahead=True,episode_days=7)
    base=E.simulate_episodes(gens,bp,num_episodes=10,seed=E.EVAL_SEED)
    hp=base['market_prices'].mean(0)
    ppo=json.load(open(R/f'p{power:g}_d{dur:g}_s{seed}/metrics.json'))['metrics']
    best=None
    for k,f,dp in itertools.product([1,2,3,4,6,8],[0.25,0.5,1.0],[-20,40]):
        d=E.simulate_episodes(gens+[Scripted(k,f,dp,hp)],params,num_episodes=10,seed=E.EVAL_SEED)
        P,dis,chg=d['market_prices'],d['bess_discharge'],d['bess_charge']
        rev=float((P*(dis-chg)).sum(1).mean())
        r=dict(k=k,f=f,dp=dp,rev=rev,fph=float(dis.sum(1).mean()/power),price=float(P.mean()))
        if best is None or rev>best['rev']: best=r
    best.update(seed=seed,power=power,dur=dur,ppo_profit=ppo['bess_profit_mean'],ppo_cycles=ppo['bess_equivalent_cycles_mean'],base_price=float(base['market_prices'].mean()))
    print(json.dumps(best),flush=True)
if __name__=='__main__':
    run(int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]))
