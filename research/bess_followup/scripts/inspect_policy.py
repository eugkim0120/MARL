import sys
from functools import partial

if __name__ == "__main__":
    import numpy as np
    import torch
    from easy_marl.examples.bidding.training import make_bess_params, parallel_train
    from easy_marl.examples.bidding.bess_experiment import EVAL_SEED
    from easy_marl.src.environment import MARLElectricityMarketEnv

    rounds, steps = int(sys.argv[1]), int(sys.argv[2])
    pf = partial(make_bess_params, power_mw=50, duration_h=4)
    agents, _ = parallel_train(N=4, num_rounds=rounds, timesteps_per_agent=steps, seed=42,
                               save_dir=None, verbose=False, update_probability=1.0, param_func=pf)
    bat = agents[3]
    model = bat.model
    print("gamma", model.gamma, "gae_lambda", model.gae_lambda, "ent_coef", model.ent_coef)
    print("log_std", model.policy.log_std.detach().numpy())

    params = pf(N=4, T=24)
    env = MARLElectricityMarketEnv(agents=agents, params=params, seed=EVAL_SEED, observer_name="simple_v3")
    env.reset(seed=EVAL_SEED)
    print(" h   raw_mean_a0  raw_mean_a1  value")
    for t in range(24):
        obs = env._get_obs(3).copy()
        with torch.no_grad():
            ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = model.policy.get_distribution(ot)
            mean = dist.distribution.mean.numpy()[0]
            value = model.policy.predict_values(ot).item()
        print(f"{t:2d} {mean[0]:11.2f} {mean[1]:12.2f} {value:8.3f}")
        env.step(None, fixed_evaluation=True)
