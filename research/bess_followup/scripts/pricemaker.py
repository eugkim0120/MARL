"""Price-maker benchmark: scripted battery cleared in the real env (price impact included)."""
import itertools, json, sys
from pathlib import Path
import numpy as np
from easy_marl.examples.bidding import bess_experiment as E

T = 24

class Scripted:
    def __init__(self, k, f_dis, dis_price, hour_price, power_eff=0.95):
        self.k, self.f_dis, self.dis_price = k, f_dis, dis_price
        self.order = np.argsort(hour_price)
        self.f_chg = min(1.0, f_dis / (power_eff ** 2))
    def fixed_act_function(self, deterministic=True):
        def act(obs):
            order = self.order
            chg, dis = order[: self.k], order[-self.k:]
            q = np.zeros(T); p = np.zeros(T)
            q[dis] = self.f_dis; q[chg] = -self.f_chg
            p[dis] = np.arctanh(np.clip((self.dis_price - 30) / 50, -0.999, 0.999))
            p[chg] = 3.0
            return np.concatenate([q, p]).astype(np.float32)
        return act

def evaluate(gens, params, power, dur, agent):
    d = E.simulate_episodes(gens + [agent], params, num_episodes=20, seed=E.EVAL_SEED)
    P, dis, chg, soc = d["market_prices"], d["bess_discharge"], d["bess_charge"], d["soc"]
    energy = power * dur
    rev = (P * (dis - chg)).sum(1)
    pen = np.maximum(0, 0.5 * energy - soc[:, -1]) * 80
    return dict(cycles=float(dis.sum(1).mean() / energy), dis=float(dis.sum(1).mean()),
                rev=float(rev.mean()), pen=float(pen.mean()), net=float((rev - pen).mean()),
                price=float(P.mean()))

def main():
    seed, power, dur = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    cfg = E.SweepConfig(power, dur, seed, "frozen")
    gens = E.get_pretrained(seed, E.PRESETS["full"], Path("outputs/bess_sweep/full"))
    params = cfg.param_func()(N=4, T=T)
    base_params = E.make_default_params(N=3, T=T, day_ahead=True)
    base = E.simulate_episodes(gens, base_params, num_episodes=20, seed=E.EVAL_SEED)
    hour_price = base["market_prices"].mean(0)
    best = None
    for k, f, dp in itertools.product([1, 2, 3, 4, 6, 8, 10], [0.25, 0.5, 1.0], [-20, 40]):
        r = evaluate(gens, params, power, dur, Scripted(k, f, dp, hour_price))
        r.update(k=k, f=f, dp=dp)
        if best is None or r["net"] > best["net"]:
            best = r
    best.update(power=power, dur=dur, seed=seed)
    print(json.dumps(best))

if __name__ == "__main__":
    main()
