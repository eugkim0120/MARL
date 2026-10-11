"""Probe: retrain one battery config with a variant of the PPO setup and report cycling and revenue."""
import json, sys, time
from pathlib import Path
import numpy as np
from easy_marl.src.agents import PPOAgent
from easy_marl.src.environment import MARLElectricityMarketEnv
from easy_marl.examples.bidding import training as T
from easy_marl.examples.bidding import bess_experiment as E

def main():
    seed, power, dur, variant = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    out = Path("/tmp/claude-1000/-home-eugene/1d0aecc7-4a99-4a2b-8ef8-ecd9d6705e85/scratchpad/probe") / f"{variant}_p{power:g}_d{dur:g}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)

    base = dict(T.ppo_params)
    VARIANTS = {
        "control": dict(),
        "bigroll": dict(ppo=dict(n_steps=256, batch_size=64)),
        "bigroll_lowstd": dict(ppo=dict(n_steps=256, batch_size=64), log_std=-1.0),
        "noent": dict(ppo=dict(ent_coef=0.0)),
        "long5x": dict(steps_mult=5),
        "all": dict(ppo=dict(n_steps=256, batch_size=64, ent_coef=0.0), log_std=-1.0),
        "all_long": dict(ppo=dict(n_steps=256, batch_size=64, ent_coef=0.0), log_std=-1.0, steps_mult=3),
        "bigroll_lowstd_long": dict(ppo=dict(n_steps=256, batch_size=64), log_std=-1.0, steps_mult=3),
    }
    v = VARIANTS[variant]
    preset = E.PRESETS["full"]
    cfg = E.SweepConfig(power, dur, seed, "frozen")
    gens = E.get_pretrained(seed, preset, Path("outputs/bess_sweep/full"))
    params = cfg.param_func()(N=4, T=24)
    env = MARLElectricityMarketEnv(agents=[], params=params, seed=seed, agent_index=3, observer_name=T.DEFAULT_OBS)
    kw = {**base, **v.get("ppo", {})}
    battery = PPOAgent(env=env, seed=seed + 3, **kw)
    if "log_std" in v:
        battery.model.policy.log_std.data.fill_(v["log_std"])
    t0 = time.time()
    agents, _ = T.parallel_train(
        N=4, num_rounds=preset["num_rounds"],
        timesteps_per_agent=preset["timesteps_per_agent"] * v.get("steps_mult", 1),
        seed=seed, save_dir=str(out / "training"), verbose=False,
        param_func=cfg.param_func(), update_probability=1.0,
        initial_agents=gens + [battery], frozen_agents=range(3),
    )
    d = E.simulate_episodes(agents, params, num_episodes=20, seed=E.EVAL_SEED)
    P = d["market_prices"]; dis = d["bess_discharge"]; chg = d["bess_charge"]; soc = d["soc"]
    energy = power * dur
    rev = (P * (dis - chg)).sum(1)
    pen = np.maximum(0, 0.5 * energy - soc[:, -1]) * 80
    res = dict(variant=variant, power=power, dur=dur, seed=seed,
               cycles=float(dis.sum(1).mean() / energy), dis_mwh=float(dis.sum(1).mean()),
               rev=float(rev.mean()), pen=float(pen.mean()), net=float((rev - pen).mean()),
               mean_price=float(P.mean()), secs=time.time() - t0)
    (out / "result.json").write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
