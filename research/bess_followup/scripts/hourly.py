import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/tmp/claude-1000/-home-eugene/1d0aecc7-4a99-4a2b-8ef8-ecd9d6705e85/scratchpad")
from pm_opt import Fixed, T
from easy_marl.examples.bidding import bess_experiment as E
seed, power, dur = 43, 25.0, 8.0
best = np.load(f"{sys.argv[1]}")
cfg = E.SweepConfig(power, dur, seed, "frozen")
gens = E.get_pretrained(seed, E.PRESETS["full"], Path("outputs/bess_sweep/full"))
params = cfg.param_func()(N=4, T=T)
print({k: v for k, v in params.items() if "bess" in k or k in ("power_mw","duration_h")})
base = E.simulate_episodes(gens, E.make_default_params(N=3, T=T, day_ahead=True), 20, E.EVAL_SEED)["market_prices"].mean(0)
for s in (1.0, 0.5):
    a = np.concatenate([best[:T]*s, best[T:]])
    d = E.simulate_episodes(gens + [Fixed(a)], params, 20, E.EVAL_SEED)
    print("scale", s)
    print("hr  q    bid  basePrice  price  chg   dis   soc")
    for h in range(T):
        print(f"{h:2d} {np.clip(a[h],-1,1):5.2f} {30+50*np.tanh(a[T+h]):5.0f} {base[h]:7.1f} {d['market_prices'][:,h].mean():7.1f} {d['bess_charge'][:,h].mean():5.1f} {d['bess_discharge'][:,h].mean():5.1f} {d['soc'][:,h].mean():6.1f}")
