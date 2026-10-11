"""Open-loop price-maker optimum: CMA-ES over the battery's 48-dim day-ahead action, cleared in the real env."""
import json, sys, time
from pathlib import Path
import numpy as np
import cma
from easy_marl.examples.bidding import bess_experiment as E

T = 24
OUT = Path("/tmp/claude-1000/-home-eugene/1d0aecc7-4a99-4a2b-8ef8-ecd9d6705e85/scratchpad/pm")

class Fixed:
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float32)
    def fixed_act_function(self, deterministic=True): return lambda obs: self.a

def score(gens, params, power, dur, a, n, seed):
    d = E.simulate_episodes(gens + [Fixed(a)], params, num_episodes=n, seed=seed)
    P, dis, chg, soc = d["market_prices"], d["bess_discharge"], d["bess_charge"], d["soc"]
    energy = power * dur
    rev = (P * (dis - chg)).sum(1)
    pen = np.maximum(0, 0.5 * energy - soc[:, -1]) * 80
    return float((rev - pen).mean()), float(dis.sum(1).mean() / energy), float(dis.sum(1).mean()), float(rev.mean()), float(pen.mean())

def main():
    seed, power, dur = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    cfg = E.SweepConfig(power, dur, seed, "frozen")
    gens = E.get_pretrained(seed, E.PRESETS["full"], Path("outputs/bess_sweep/full"))
    params = cfg.param_func()(N=4, T=T)
    es = cma.CMAEvolutionStrategy(np.zeros(2 * T), 1.0, {"popsize": 24, "seed": seed, "verbose": -9})
    t0 = time.time()
    for it in range(iters):
        X = es.ask()
        # training seeds 1000+ are disjoint from EVAL_SEED
        es.tell(X, [-score(gens, params, power, dur, x, 6, 1000 + it)[0] for x in X])
    best = es.result.xbest
    net, cyc, dis, rev, pen = score(gens, params, power, dur, best, 20, E.EVAL_SEED)
    res = dict(power=power, dur=dur, seed=seed, cycles=cyc, dis=dis, rev=rev, pen=pen, net=net, secs=time.time() - t0, iters=iters)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"best_p{power:g}_d{dur:g}_s{seed}.npy", best)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
