import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/tmp/claude-1000/-home-eugene/1d0aecc7-4a99-4a2b-8ef8-ecd9d6705e85/scratchpad")
from pm_opt import Fixed, T
from easy_marl.examples.bidding import bess_experiment as E
seed, power, dur = 43, float(sys.argv[1]), float(sys.argv[2])
best = np.load(f"{sys.argv[3]}")
cfg = E.SweepConfig(power, dur, seed, "frozen")
gens = E.get_pretrained(seed, E.PRESETS["full"], Path("outputs/bess_sweep/full"))
params = cfg.param_func()(N=4, T=T)
base = E.simulate_episodes(gens, E.make_default_params(N=3, T=T, day_ahead=True), 20, E.EVAL_SEED)["market_prices"]
print("baseline mean price", base.mean().round(2))
q, p = best[:T], best[T:]
print("optimised q (MW frac):", np.round(np.clip(q, -1, 1), 2).tolist())
print("optimised bid price:", np.round(30 + 50*np.tanh(p), 0).tolist())
for s in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
    a = np.concatenate([q * s, p])
    d = E.simulate_episodes(gens + [Fixed(a)], params, 20, E.EVAL_SEED)
    P, dis, chg, soc = d["market_prices"], d["bess_discharge"], d["bess_charge"], d["soc"]
    en = power * dur
    rev = (P * (dis - chg)).sum(1); pen = np.maximum(0, .5*en - soc[:, -1]) * 80
    buy = (P*chg).sum()/max(chg.sum(),1e-9); sell = (P*dis).sum()/max(dis.sum(),1e-9)
    print(f"scale {s:4.2f} cyc {dis.sum(1).mean()/en:.2f} chg {chg.sum(1).mean():5.0f} dis {dis.sum(1).mean():5.0f} buy@{buy:5.1f} sell@{sell:5.1f} rev {rev.mean():6.0f} pen {pen.mean():5.0f} net {(rev-pen).mean():6.0f}")
