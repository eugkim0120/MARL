"""
Unit tests for training module.
"""

import multiprocessing
import subprocess
import sys
import numpy as np
import pytest
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from easy_marl.src.agents import PPOAgent
from easy_marl.src.environment import MARLElectricityMarketEnv
from easy_marl.examples.bidding.training import (
    init_agents,
    parallel_train,
    set_all_seeds,
    train_single_agent_worker,
)


# Minimal test config
N_AGENTS = 2
TIMESTEPS = 48  # Just 2 PPO updates (n_steps=24)
SEED = 42

TEST_PARAMS = {
    "N_generators": N_AGENTS,
    "T": 24,
    "demand_profile": [50.0] * 24,
    "capacities": [30.0] * N_AGENTS,
    "costs": [20.0, 25.0],
    "max_bid_delta": 50.0,
    "lambda_bid_penalty": 0.01,
}

TEST_PPO_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 24,
    "batch_size": 24,
    "hidden_sizes": (8, 8),
    "ent_coef": 0.01,
}


def create_test_agents(observer_name="simple"):
    """Create agents for testing."""
    set_all_seeds(SEED)
    agents = []
    for i in range(N_AGENTS):
        env = MARLElectricityMarketEnv(
            agents=[], params=TEST_PARAMS, seed=SEED, agent_index=i, observer_name=observer_name
        )
        agents.append(PPOAgent(env=env, seed=SEED + i, **TEST_PPO_PARAMS))
    return agents


def get_weights(agent):
    """Extract model weights as flat array."""
    return {
        n: p.detach().cpu().numpy().copy()
        for n, p in agent.model.policy.named_parameters()
    }


PARALLEL_TRAIN_TIMEOUT_S = 180

PARALLEL_TRAIN_SCRIPT = """
import sys
from easy_marl.examples.bidding.training import parallel_train

if __name__ == "__main__":
    parallel_train(
        N={n}, num_rounds=1, timesteps_per_agent={timesteps}, n_workers={n},
        update_probability=1.0, save_dir=sys.argv[1], verbose=False,
    )
"""


class TestParallelTrainDoesNotDeadlock:
    """parallel_train runs torch in the parent before starting its workers."""

    def test_that_parallel_train_completes_instead_of_deadlocking(self, tmp_path):
        """A fresh interpreter is used so a fork-after-torch hang fails as a timeout."""
        script = PARALLEL_TRAIN_SCRIPT.format(n=N_AGENTS, timesteps=TIMESTEPS)
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            timeout=PARALLEL_TRAIN_TIMEOUT_S,
            check=True,
        )
        assert (tmp_path / "training_summary.json").exists()


class TestTrainingEquivalence:
    """Tests that sequential and parallel training produce equivalent results."""

    def test_that_one_step_in_sequential_train_is_equivalent_to_one_step_in_parallel_train(
        self,
    ):
        """Verify one round of SBR produces identical weights in both modes."""
        # === SEQUENTIAL ===
        agents_seq = create_test_agents()
        snapshots = [a.save_to_bytes() for a in agents_seq]
        trained_seq = {}

        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if j != i:
                    agents_seq[j].load_from_bytes(snapshots[j])

            set_all_seeds(SEED + i)
            env = MARLElectricityMarketEnv(
                agents=agents_seq, params=TEST_PARAMS, seed=SEED + i, agent_index=i, observer_name="simple"
            )
            agents_seq[i].model.set_env(env)
            agents_seq[i].train(total_timesteps=TIMESTEPS)
            trained_seq[i] = agents_seq[i].save_to_bytes()

        for i in range(N_AGENTS):
            agents_seq[i].load_from_bytes(trained_seq[i])

        # === PARALLEL ===
        agents_par = create_test_agents()
        snapshots_par = [a.save_to_bytes() for a in agents_par]
        trained_par = {}

        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=N_AGENTS, mp_context=spawn_context
        ) as executor:
            futures = [
                executor.submit(
                    train_single_agent_worker,
                    agent_index=i,
                    round_idx=0,
                    agent_state=snapshots_par[i],
                    agents_states=snapshots_par,
                    params=TEST_PARAMS,
                    timesteps_per_agent=TIMESTEPS,
                    save_dir=None,
                    verbose=False,
                    N=N_AGENTS,
                    seed=SEED,
                    observer_name="simple",
                )
                for i in range(N_AGENTS)
            ]
            for f in as_completed(futures):
                idx, state, _ = f.result()
                trained_par[idx] = state

        for i in range(N_AGENTS):
            agents_par[i].load_from_bytes(trained_par[i])

        # === COMPARE ===
        for i in range(N_AGENTS):
            # 1. Compare model weights
            w_seq = get_weights(agents_seq[i])
            w_par = get_weights(agents_par[i])
            for name in w_seq:
                np.testing.assert_allclose(
                    w_seq[name],
                    w_par[name],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Agent {i} weight '{name}' differs",
                )

            # 2. Compare optimizer state (Adam momentum/variance)
            opt_seq = agents_seq[i].model.policy.optimizer.state_dict()
            opt_par = agents_par[i].model.policy.optimizer.state_dict()

            for key in opt_seq["state"]:
                for state_key in opt_seq["state"][key]:
                    val_seq = opt_seq["state"][key][state_key]
                    val_par = opt_par["state"][key][state_key]
                    if torch.is_tensor(val_seq):
                        np.testing.assert_allclose(
                            val_seq.numpy(),
                            val_par.numpy(),
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg=f"Agent {i} optimizer state '{state_key}' differs",
                        )

            # 3. Compare training counters
            assert (
                agents_seq[i].model.num_timesteps == agents_par[i].model.num_timesteps
            ), f"Agent {i} num_timesteps differs"

            # 4. Compare log_std (exploration parameter)
            log_std_seq = agents_seq[i].model.policy.log_std.detach().cpu().numpy()
            log_std_par = agents_par[i].model.policy.log_std.detach().cpu().numpy()
            np.testing.assert_allclose(
                log_std_seq,
                log_std_par,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Agent {i} log_std differs",
            )


DAY_N = 2
DAY_T = 4
DAY_TIMESTEPS = 48


def day_ahead_params(N=DAY_N, T=DAY_T):
    return {
        "N_generators": N,
        "T": T,
        "day_ahead": True,
        "demand_profile": [50.0] * T,
        "capacities": [30.0] * N,
        "costs": [20.0 + 5.0 * i for i in range(N)],
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }


def train_day_ahead(initial_agents=None, frozen_agents=(), num_rounds=1):
    return parallel_train(
        N=DAY_N,
        T=DAY_T,
        num_rounds=num_rounds,
        timesteps_per_agent=DAY_TIMESTEPS,
        seed=SEED,
        save_dir=None,
        verbose=False,
        update_probability=1.0,
        param_func=day_ahead_params,
        n_workers=DAY_N,
        initial_agents=initial_agents,
        frozen_agents=frozen_agents,
    )


def make_initial_agents():
    return init_agents(DAY_N, day_ahead_params(), seed=SEED + 100)


def assert_weights_equal(a, b):
    wa, wb = get_weights(a), get_weights(b)
    for name in wa:
        np.testing.assert_array_equal(wa[name], wb[name], err_msg=name)


def weights_differ(a, b):
    wa, wb = get_weights(a), get_weights(b)
    return any(not np.array_equal(wa[name], wb[name]) for name in wa)


class TestParallelTrainFrozenAndInitialAgents:
    def test_that_frozen_agents_keep_their_initial_policy_while_others_train(self):
        initial = make_initial_agents()
        reference = make_initial_agents()

        agents, _ = train_day_ahead(initial_agents=initial, frozen_agents={0}, num_rounds=2)

        assert_weights_equal(agents[0], reference[0])
        assert weights_differ(agents[1], reference[1])

    def test_that_a_frozen_agent_still_acts_as_an_opponent_of_the_trained_agent(self):
        frozen_a = make_initial_agents()
        frozen_b = make_initial_agents()
        frozen_b[0] = init_agents(DAY_N, day_ahead_params(), seed=SEED + 200)[0]

        trained_a, _ = train_day_ahead(initial_agents=frozen_a, frozen_agents={0})
        trained_b, _ = train_day_ahead(initial_agents=frozen_b, frozen_agents={0})

        # Same starting agent 1 and seed, different frozen opponent: training must differ.
        assert weights_differ(trained_a[1], trained_b[1])

    def test_that_the_callers_initial_agents_are_not_modified_by_training(self):
        initial = make_initial_agents()
        reference = make_initial_agents()

        train_day_ahead(initial_agents=initial, frozen_agents={0})

        assert_weights_equal(initial[0], reference[0])
        assert_weights_equal(initial[1], reference[1])

    def test_that_wrong_number_of_initial_agents_is_rejected(self):
        with pytest.raises(ValueError, match="initial_agents"):
            train_day_ahead(initial_agents=make_initial_agents()[:1])

    def test_that_frozen_index_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match="frozen_agents"):
            train_day_ahead(frozen_agents={DAY_N})
