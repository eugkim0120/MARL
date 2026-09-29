"""
Multi-Agent Reinforcement Learning (MARL) Training Module
=========================================================

This module implements two distinct training regimes for multi-agent reinforcement
learning in electricity market environments:

1. ITERATED BEST RESPONSE (IBR) - implemented as `sequential_train()`
   - Also known as: Gauss-Seidel updates, Fictitious Play variant
   - Agents train one at a time, each optimizing against the CURRENT state of all
     other agents (including updates made earlier in the same round)
   - Within a round: Agent 0 trains → Agent 1 sees Agent 0's new policy → etc.
   - Convergence: May converge faster in potential games; can cycle in zero-sum games
   - Reference: Brown (1951), Fudenberg & Levine (1998)

2. SIMULTANEOUS BEST RESPONSE (SBR) - implemented as `parallel_train()`
   - Also known as: Jacobi updates, Self-play with frozen opponents
   - All agents train concurrently against a FROZEN SNAPSHOT of opponent policies
     taken at the start of each round
   - Within a round: All agents see the same "stale" opponent policies
   - Convergence: More stable in adversarial settings; used in OpenAI Five, AlphaStar
   - Reference: Heinrich et al. (2015), Lanctot et al. (2017)

KEY DIFFERENCE:
--------------
The fundamental distinction is WHEN opponent policy updates become visible:

    Sequential (IBR):    Agent i sees Agent j's update if j < i (same round)
    Parallel (SBR):      Agent i sees Agent j's update only in the NEXT round

For electricity markets with multiple equilibria, the choice of regime can lead to
convergence to DIFFERENT Nash equilibria.

USAGE:
------
    # Iterated Best Response (sequential updates within rounds)
    agents, info = sequential_train(N=5, num_rounds=3, timesteps_per_agent=5000, update_schedule="IBR")

    # Simultaneous Best Response (sequential updates, frozen opponents)
    agents, info = sequential_train(N=5, num_rounds=3, timesteps_per_agent=5000, update_schedule="SBR")

    # Simultaneous Best Response (parallel updates, frozen opponents)
    agents, info = parallel_train(N=5, num_rounds=3, timesteps_per_agent=5000)

    # Auto-select based on problem size and resources
    agents, info = auto_train(N=5, num_rounds=3, timesteps_per_agent=5000)

REPRODUCIBILITY:
----------------
To ensure consistent results between sequential and parallel training modes:
1. We use explicit seeding for all RNGs (Python, NumPy, PyTorch).
2. Seeds are set *immediately before* each agent's training loop.
3. Agents are serialized/deserialized using full state preservation (including optimizer).
"""

import multiprocessing
import os
import random
import numpy as np
import torch
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import psutil

from easy_marl.src.agents import PPOAgent, BaseAgent
from easy_marl.src.environment import MARLElectricityMarketEnv

MAX_ENV_PROCS = psutil.cpu_count(logical=False) or os.cpu_count()
SAFE_MAX_ENV_PROCS = max(1, MAX_ENV_PROCS - 1)
DEFAULT_OBS = "simple_v3"  # keep as just "simple" if unsure ("simple" is used for tools/runtime_analysis.py)
DEFAULT_UPDATE_PROB = 0.75  # keep at 1 if unsure


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility across processes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ppo_params = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 24,  # Match episode length for stable updates, this is good as our environment is deterministic over episodes
    "batch_size": 24,  # Match episode length for stable updates, this is good as our environment is deterministic over episodes
    "hidden_sizes": (32, 32, 32, 32),  # default is (64, 64)
    "ent_coef": 0.01,
}


def make_competitive_params(N: int = 8, T: int = 24, competition_rate=2) -> Dict:
    """
    Create competitive environment parameters.

    Args:
        N: Number of generators
        T: Time horizon (number of timesteps)

    Returns:
        Dictionary of environment parameters
    """
    # Sinusoidal demand profile (mimics daily pattern)
    demand_profile = np.sin(np.linspace(0, 2 * np.pi, T)) + 1.5

    # Uniform capacities
    capacities = np.full(N, 30.0 * 5 / N)

    # Scale demand profile
    scaled_demand_profile = demand_profile / demand_profile.max()
    demand_profile = scaled_demand_profile * (np.sum(capacities[:-1]) - 1)

    # Heterogeneous costs (increasing)
    costs = np.linspace(20.0, 40.0, int(np.ceil(N / competition_rate))).tolist()
    costs = costs * competition_rate  # Duplicate to create pairs
    costs = np.array(costs[:N])  # Trim to N if odd
    costs = np.sort(costs)

    params = {
        "N_generators": N,
        "T": T,
        "demand_profile": demand_profile.tolist(),
        "capacities": capacities.tolist(),
        "costs": costs.tolist(),
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }
    return params


def make_default_params(N: int = 5, T: int = 24) -> Dict:
    """
    Create default environment parameters.

    Args:
        N: Number of generators
        T: Time horizon (number of timesteps)

    Returns:
        Dictionary of environment parameters
    """
    # Sinusoidal demand profile (mimics daily pattern)
    demand_profile = (np.sin(np.linspace(0, 2 * np.pi, T)) + 1.5) * 50.0

    # Uniform capacities
    capacities = np.full(N, 30.0 * 5 / N)

    # Heterogeneous costs (increasing)
    costs = np.linspace(20.0, 40.0, N)

    params = {
        "N_generators": N,
        "T": T,
        "demand_profile": demand_profile.tolist(),
        "capacities": capacities.tolist(),
        "costs": costs.tolist(),
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }
    return params


def make_bess_params(
    N: int = 4,
    T: int = 24,
    power_mw: float = 25.0,
    duration_h: float = 4.0,
    efficiency_rt: float = 0.9,
    initial_soc_frac: float = 0.5,
    bid_ref: float = 30.0,
) -> Dict:
    """
    Default market with N - 1 generators plus one battery as the last agent.

    Generators and demand are identical to ``make_default_params(N - 1)`` so a
    battery run can be compared directly against the no-battery baseline.
    """
    params = make_default_params(N=N - 1, T=T)
    params["bess"] = {
        "power_mw": power_mw,
        "duration_h": duration_h,
        "efficiency_rt": efficiency_rt,
        "initial_soc_frac": initial_soc_frac,
        "bid_ref": bid_ref,
    }
    return params


def make_env_for_agent(
    agent_index: int,
    agents: List[BaseAgent],
    params: Dict,
    observer_name: str = DEFAULT_OBS,
    seed: int = None,
) -> MARLElectricityMarketEnv:
    """
    Create an environment instance for a specific agent with other agents' policies fixed.

    Args:
        agent_index: Index of the agent to train
        agents: List of all agent objects
        params: Environment parameters
        observer_name: Type of observation function
        seed: Random seed

    Returns:
        Environment instance configured for the specified agent
    """
    env = MARLElectricityMarketEnv(
        agents=agents,
        params=params,
        seed=seed,
        agent_index=agent_index,
        observer_name=observer_name,
    )

    # Create dictionary of other agents' action functions
    other_fns = {}
    for j, agent in enumerate(agents):
        if j == agent_index:
            continue

        # Get appropriate action function based on agent type
        if isinstance(agent, PPOAgent):
            other_fns[j] = agent.fixed_act_function(deterministic=True)
        else:
            raise ValueError(
                f"PPO agent not being used, found agent of type {type(agent)}"
            )

    return env


def init_agents(
    N: int, params: Dict, seed: int, observer_name: str = DEFAULT_OBS
) -> List[PPOAgent]:
    """Initialize N agents with default PPO configuration."""
    agents = []
    for i in range(N):
        env = MARLElectricityMarketEnv(
            agents=[],
            params=params,
            seed=seed,
            agent_index=i,
            observer_name=observer_name,
        )
        agents.append(
            PPOAgent(
                env=env,
                seed=seed + i,
                **ppo_params,
            )
        )
    return agents


def convert_np_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_np_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def evaluate_agents(
    agents: List[BaseAgent],
    params: Dict,
    num_episodes: int = 1,
    seed: int = None,
    observer_name: str = DEFAULT_OBS,
) -> Dict:
    """
    Evaluate trained agents over multiple episodes.

    Args:
        agents: List of trained agent objects
        params: Environment parameters
        num_episodes: Number of evaluation episodes
        seed: Random seed, only appropriate for stochastic environments
        observer_name: Type of observation function

    Returns:
        Dictionary containing evaluation metrics
    """
    N = len(agents)

    # Storage for evaluation metrics
    episode_rewards = np.zeros((num_episodes, N))
    episode_profits = np.zeros((num_episodes, N))
    episode_quantities = np.zeros((num_episodes, N))
    market_prices = []
    all_outputs = []

    for ep in range(num_episodes):
        # Create evaluation environment with all agents frozen
        eval_env = MARLElectricityMarketEnv(
            agents=agents,
            params=params,
            seed=seed + ep if seed else None,
            observer_name=observer_name,
        )

        obs, _ = eval_env.reset()
        done = False

        while not done:
            # Collect actions from all agents' fixed policies
            obs, reward, terminated, truncated, _ = eval_env.step(
                None, fixed_evaluation=True
            )
            done = terminated or truncated

        # Extract metrics from environment output
        output = eval_env.output

        for i in range(N):
            total_reward = np.sum(output["rewards"][:, i])
            if eval_env.has_bess and i == eval_env.bess_index:
                total_profit = np.sum(
                    output["market_prices"]
                    * (output["bess_discharge"] - output["bess_charge"])
                )
            else:
                total_profit = np.sum(
                    (output["market_prices"] - params["costs"][i])
                    * output["q_cleared"][:, i]
                )
            total_quantity = np.sum(output["q_cleared"][:, i])

            episode_rewards[ep, i] = total_reward
            episode_profits[ep, i] = total_profit
            episode_quantities[ep, i] = total_quantity

        market_prices.append(output["market_prices"])

    # Aggregate results
    results = {
        "mean_rewards": np.mean(episode_rewards, axis=0),
        "std_rewards": np.std(episode_rewards, axis=0),
        "mean_profits": np.mean(episode_profits, axis=0),
        "std_profits": np.std(episode_profits, axis=0),
        "mean_quantities": np.mean(episode_quantities, axis=0),
        "mean_market_price": np.mean(market_prices),
        "std_market_price": np.std(market_prices),
        "mean_market_prices_over_time": np.mean(market_prices, axis=0),
    }

    return results


def _train_agent_core(
    agent_index: int,
    round_idx: int,
    agents: List[PPOAgent],
    params: Dict,
    timesteps_per_agent: int,
    save_dir: str,
    verbose: bool,
    N: int,
    seed: int,
    observer_name: str = DEFAULT_OBS,
) -> Dict:
    """
    Core training logic shared between sequential and parallel modes.

    Args:
        agent_index: Index of the agent to train
        round_idx: Current round index
        agents: List of PPO agents
        params: Environment parameters
        timesteps_per_agent: Number of training timesteps
        save_dir: Directory for saving outputs
        verbose: Verbosity flag
        N: Number of agents
        seed: Base random seed
        observer_name: Type of observation function

    Returns:
        eval_metrics: Evaluation metrics after training this agent
    """
    if verbose:
        mode_str = (
            "[Process]" if hasattr(os, "getpid") and os.getpid() != os.getppid() else ""
        )
        print(
            f"\n-> Training Agent {agent_index} ({timesteps_per_agent} timesteps) {mode_str}"
        )

    # Set all seeds immediately before training for reproducibility
    training_seed = seed + round_idx * N + agent_index
    set_all_seeds(training_seed)

    env = make_env_for_agent(
        agent_index=agent_index,
        agents=agents,
        params=params,
        seed=training_seed,
        observer_name=observer_name,
    )

    agents[agent_index].model.set_env(env)
    agents[agent_index].train(total_timesteps=timesteps_per_agent)

    # Save checkpoint
    if save_dir is not None:
        round_dir = os.path.join(save_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        save_path = os.path.join(round_dir, f"agent_{agent_index}")
        agents[agent_index].save(save_path)
        if verbose:
            print(f"  [OK] Saved to {save_path}")

    # Evaluate all agents after training this agent
    eval_metrics = evaluate_agents(
        agents, params, num_episodes=1, seed=seed, observer_name=observer_name
    )
    eval_metrics = convert_np_to_python(eval_metrics)

    # Save evaluation metrics as JSON
    if save_dir is not None:
        eval_path = os.path.join(
            round_dir, f"eval_agent_{agent_index}_round_{round_idx + 1}.json"
        )
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=4)

        if verbose:
            print(f"  [OK] Evaluation saved to {eval_path}")

    return eval_metrics


def sequential_train(
    N: int = 5,
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    param_func=make_default_params,
    update_schedule: str = "SBR",
    T: int = 24,
    observer_name: str = DEFAULT_OBS,
    update_probability: float = DEFAULT_UPDATE_PROB,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Single-process training with configurable update schedule (IBR or SBR).

    Trains agents sequentially (one at a time) within each round. The update_schedule
    parameter controls when opponent policy updates become visible to other agents.

    Update Schedules:
        - "SBR" (Simultaneous Best Response, default):
            All agents train against a FROZEN SNAPSHOT of opponent policies from
            the start of the round. Updates are synchronized at round boundaries.
            This is Jacobi-style iteration.

        - "IBR" (Iterated Best Response):
            Each agent sees the CURRENT state of all other agents, meaning Agent i
            will see the updated policies of Agents 0..i-1 trained earlier in the
            same round. This is Gauss-Seidel-style iteration.

    Game-Theoretic Properties:
        SBR:
            - More stable in adversarial/zero-sum settings
            - Avoids "chasing" dynamics
            - Same semantics as parallel_train() but single-process
        IBR:
            - Updates propagate immediately within a round
            - May converge faster in potential games
            - Can exhibit cycling behavior in some adversarial games

    Round Structure:
        SBR: Snapshot policies → Train Agent 0..N-1 (all see snapshot) → Sync
        IBR: Train Agent 0 → Train Agent 1 (sees Agent 0's update) → ... → Train Agent N-1

    References:
        - Brown, G. W. (1951). Iterative solution of games by fictitious play.
        - Fudenberg, D., & Levine, D. K. (1998). The Theory of Learning in Games.
        - Heinrich, J., et al. (2015). Fictitious Self-Play in Extensive-Form Games.

    Args:
        N: Number of agents (generators in the market)
        num_rounds: Number of complete training rounds (outer loop)
        timesteps_per_agent: PPO training timesteps per agent per round
        seed: Random seed for reproducibility
        save_dir: Directory to save trained models and evaluation metrics
        verbose: Whether to print training progress
        param_func: Function to generate environment parameters
        update_schedule: "SBR" (default) for simultaneous best response or
                        "IBR" for iterated best response
        observer_name: Type of observation function ("simple", "basic", or "simple_v2")
        update_probability: Probability (0.0 to 1.0) that an agent updates its policy in a round.
                          Values < 1.0 introduce "inertia" which helps convergence in cyclic games.
                          Only applies to SBR.

    Returns:
        agents: List of trained PPOAgent objects
        training_info: Dictionary containing training metadata and configuration

    Raises:
        ValueError: If update_schedule is not "IBR" or "SBR"
    """
    # Validate update_schedule
    update_schedule = update_schedule.upper()
    if update_schedule not in ("IBR", "SBR"):
        raise ValueError(
            f"update_schedule must be 'IBR' or 'SBR', got '{update_schedule}'"
        )

    # Generate environment parameters
    params = param_func(N=N, T=T)
    agents = init_agents(N, params, seed, observer_name)

    # Training loop
    training_info = {
        "N": N,
        "num_rounds": num_rounds,
        "timesteps_per_agent": timesteps_per_agent,
        "total_timesteps": N * num_rounds * timesteps_per_agent,
        "seed": seed,
        "parallel": False,
        "update_schedule": update_schedule,
        "update_probability": update_probability,
    }

    # Ensure inertia decisions are reproducible
    random.seed(seed)

    all_results = []

    for round_idx in range(num_rounds):
        schedule_label = f"[{update_schedule}]"
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Round {round_idx + 1}/{num_rounds} {schedule_label}")
            print(f"{'=' * 60}")

        # SBR: Snapshot all agent policies at the start of the round
        if update_schedule == "SBR":
            agents_snapshots = [agent.save_to_bytes() for agent in agents]

        trained_states = {}
        for i in range(N):
            # SBR: Restore all agents to round-start snapshot before each training
            # This ensures each agent trains against the frozen policies
            if update_schedule == "SBR":
                for j, agent in enumerate(agents):
                    if j != i:  # Don't restore the agent we're about to train
                        agent.load_from_bytes(agents_snapshots[j])

            # Inertia check for SBR
            # Use a local RNG seeded with (seed, round, agent) to ensure:
            # 1. Reproducibility independent of global random state (which is reset by training)
            # 2. Different agents are skipped in different rounds (avoiding patterns)
            rng = random.Random(seed + round_idx * 100 + i)
            if (
                update_schedule == "SBR"
                and rng.random() > update_probability
                and round_idx > 0
            ):
                if verbose:
                    print(f"  [Inertia] Skipping update for Agent {i}")
                # Keep the old state (snapshot)
                trained_states[i] = agents_snapshots[i]
                continue

            eval_metrics = _train_agent_core(
                agent_index=i,
                round_idx=round_idx,
                agents=agents,
                params=params,
                timesteps_per_agent=timesteps_per_agent,
                save_dir=save_dir,
                verbose=verbose,
                N=N,
                seed=seed,
                observer_name=observer_name,
            )

            # SBR: Save the trained state to apply at round end
            if update_schedule == "SBR":
                trained_states[i] = agents[i].save_to_bytes()

            all_results.append(
                {
                    "round": round_idx + 1,
                    "agent": i,
                    "metrics": eval_metrics,
                }
            )

        # SBR: Synchronize all trained policies at round end
        if update_schedule == "SBR":
            for i in range(N):
                agents[i].load_from_bytes(trained_states[i])
            if verbose:
                print(
                    f"\n[OK] Round {round_idx + 1} complete - all {N} agents synchronized"
                )

    # Save aggregated training info and all results
    if save_dir is not None:
        summary_path = os.path.join(save_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {"training_info": training_info, "all_evaluations": all_results},
                f,
                indent=4,
            )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Total timesteps: {training_info['total_timesteps']:,}")
        print(f"Models and evaluations saved to: {save_dir}")

    return agents, training_info


# ============================================================================
# PARALLEL TRAINING IMPLEMENTATION
# ============================================================================


def train_single_agent_worker(
    agent_index: int,
    round_idx: int,
    agent_state: bytes,
    agents_states: List[bytes],
    params: Dict,
    timesteps_per_agent: int,
    save_dir: str,
    verbose: bool,
    N: int,
    seed: int,
    observer_name: str = DEFAULT_OBS,
) -> Tuple[int, bytes, Dict]:
    """
    Worker function to train a single agent in a separate process.

    Args:
        agent_index: Index of the agent to train
        round_idx: Current round index
        agent_state: Serialized state of the agent to train
        agents_states: Serialized states of all agents (for environment setup)
        params: Environment parameters
        timesteps_per_agent: Number of training timesteps
        save_dir: Directory for saving outputs
        verbose: Verbosity flag
        N: Number of agents
        seed: Base random seed
        observer_name: Type of observation function

    Returns:
        Tuple of (agent_index, trained_agent_state, eval_metrics)
    """
    # Deserialize agents
    agents = []
    for i, state in enumerate(agents_states):
        # Create environment for this agent
        env = MARLElectricityMarketEnv(
            agents=[],
            params=params,
            seed=seed,
            agent_index=i,
            observer_name=observer_name,
        )

        # Use factory method to create agent directly from bytes (consistent with sequential)
        agent = PPOAgent.from_bytes(state, env)
        agents.append(agent)

    # Now train the specific agent using the core logic
    eval_metrics = _train_agent_core(
        agent_index=agent_index,
        round_idx=round_idx,
        agents=agents,
        params=params,
        timesteps_per_agent=timesteps_per_agent,
        save_dir=save_dir,
        verbose=verbose,
        N=N,
        seed=seed,
        observer_name=observer_name,
    )

    # Serialize the trained agent state
    trained_state = agents[agent_index].save_to_bytes()

    return agent_index, trained_state, eval_metrics


def parallel_train(
    N: int = 5,
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    param_func=make_default_params,
    n_workers: int = SAFE_MAX_ENV_PROCS,
    T: int = 24,
    observer_name: str = DEFAULT_OBS,
    update_probability: float = DEFAULT_UPDATE_PROB,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Simultaneous Best Response (SBR) training with Jacobi-style updates.

    Trains all agents in parallel within each round. Each agent optimizes its
    policy against a FROZEN SNAPSHOT of opponent policies taken at the start
    of the round. Updates are synchronized at round boundaries.

    Game-Theoretic Properties:
        - Implements simultaneous best response dynamics
        - Updates are batched and synchronized (Jacobi-style)
        - More stable in adversarial/zero-sum settings
        - Avoids "chasing" dynamics where agents chase each other's updates
        - Used in large-scale systems: OpenAI Five, AlphaStar population training

    Round Structure:
        Round r: Snapshot all policies → Train all agents in parallel → Sync updates

    Note on Semantic Difference from sequential_train:
        In sequential training, Agent i sees Agent j's round-r update if j < i.
        In parallel training, Agent i sees Agent j's round-(r-1) policy for ALL j.
        This can lead to convergence to DIFFERENT equilibria in games with multiple
        Nash equilibria (common in electricity markets).

    References:
        - Heinrich, J., et al. (2015). Fictitious Self-Play in Extensive-Form Games.
        - Lanctot, M., et al. (2017). A Unified Game-Theoretic Approach to MARL.

    Args:
        N: Number of agents (generators in the market)
        num_rounds: Number of complete training rounds (outer loop)
        timesteps_per_agent: PPO training timesteps per agent per round
        seed: Random seed for reproducibility
        save_dir: Directory to save trained models and evaluation metrics
        verbose: Whether to print training progress
        param_func: Function to generate environment parameters
        n_workers: Number of parallel worker processes (defaults to CPU count - 1)
        observer_name: Type of observation function ("simple", "basic", or "simple_v2")
        update_probability: Probability (0.0 to 1.0) that an agent updates its policy in a round.
                          Values < 1.0 introduce "inertia" which helps convergence in cyclic games.

    Returns:
        agents: List of trained PPOAgent objects
        training_info: Dictionary containing training metadata and configuration
    """
    # Limit workers to number of agents
    n_workers = min(n_workers, N)

    # Generate environment parameters
    params = param_func(N=N, T=T)
    agents = init_agents(N, params, seed, observer_name)

    # Training loop
    training_info = {
        "N": N,
        "num_rounds": num_rounds,
        "timesteps_per_agent": timesteps_per_agent,
        "total_timesteps": N * num_rounds * timesteps_per_agent,
        "seed": seed,
        "parallel": True,
        "n_workers": n_workers,
        "update_probability": update_probability,
    }

    # Ensure inertia decisions are reproducible
    random.seed(seed)

    all_results = []

    # Submit all training jobs for this round
    # "fork" deadlocks: torch has already started its thread pool in this process.
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=spawn_context) as executor:
        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n{'=' * 60}")
                print(
                    f"Round {round_idx + 1}/{num_rounds} [PARALLEL MODE - {n_workers} workers]"
                )
                print(f"{'=' * 60}")

            # Serialize all agent states at the start of the round
            agents_states = [agent.save_to_bytes() for agent in agents]

            futures = []
            active_agent_indices = []

            for i in range(N):
                # Inertia check
                # Use a local RNG seeded with (seed, round, agent) for consistency with sequential_train
                rng = random.Random(seed + round_idx * 100 + i)
                if rng.random() > update_probability and round_idx > 0:
                    if verbose:
                        print(f"  [Inertia] Skipping update for Agent {i}")
                    continue

                active_agent_indices.append(i)
                future = executor.submit(
                    train_single_agent_worker,
                    agent_index=i,
                    round_idx=round_idx,
                    agent_state=agents_states[i],
                    agents_states=agents_states,
                    params=params,
                    timesteps_per_agent=timesteps_per_agent,
                    save_dir=save_dir,
                    verbose=verbose,
                    N=N,
                    seed=seed,
                    observer_name=observer_name,
                )
                futures.append(future)

            # Collect results as they complete
            results_this_round = {}
            for future in as_completed(futures):
                agent_index, trained_state, eval_metrics = future.result()
                results_this_round[agent_index] = (trained_state, eval_metrics)

                all_results.append(
                    {
                        "round": round_idx + 1,
                        "agent": agent_index,
                        "metrics": eval_metrics,
                    }
                )

                if verbose:
                    print(f"  [OK] Agent {agent_index} training complete")

            # Update all agents with their trained states
            # Only update agents that actually trained
            for i in active_agent_indices:
                trained_state, _ = results_this_round[i]
                agents[i].load_from_bytes(trained_state)

            if verbose:
                print(
                    f"\n[OK] Round {round_idx + 1} complete - all {N} agents trained in parallel"
                )

    # Save aggregated training info and all results
    if save_dir is not None:
        summary_path = os.path.join(save_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {"training_info": training_info, "all_evaluations": all_results},
                f,
                indent=4,
            )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Total timesteps: {training_info['total_timesteps']:,}")
        print(f"Parallel workers used: {n_workers}")
        print(f"Models and evaluations saved to: {save_dir}")

    return agents, training_info


def auto_train(
    N: int = 5,
    num_rounds: int = 3,
    timesteps_per_agent: int = 24*4,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    param_func=make_default_params,
    T: int = 24,
    observer_name: str = DEFAULT_OBS,
    update_probability: float = DEFAULT_UPDATE_PROB,
    **kwargs,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Automatically choose between sequential and parallel training based on system resources.
    """

    if N < 2:
        print("Auto training: N < 2, using sequential training")
        return sequential_train(
            N,
            num_rounds,
            timesteps_per_agent,
            seed,
            save_dir,
            verbose,
            param_func,
            T=T,
            observer_name=observer_name,
            update_probability=update_probability,
        )

    if timesteps_per_agent < 4000:
        print("Auto training: timesteps_per_agent < 4000, using sequential training")
        return sequential_train(
            N,
            num_rounds,
            timesteps_per_agent,
            seed,
            save_dir,
            verbose,
            param_func,
            T=T,
            observer_name=observer_name,
            update_probability=update_probability,
        )
    else:
        print("Auto training: timesteps_per_agent >= 4000, using parallel training")
        return parallel_train(
            N,
            num_rounds,
            timesteps_per_agent,
            seed,
            save_dir,
            verbose,
            param_func,
            T=T,
            observer_name=observer_name,
            update_probability=update_probability,
            **kwargs,
        )


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
USAGE:
    # Sequential training (original)
    agents, info = sequential_train(N=5, num_rounds=3, timesteps_per_agent=5000)
    
    # Parallel training (new)
    agents, info = parallel_train(N=5, num_rounds=3, timesteps_per_agent=5000, n_workers=5)

    # Auto training (new)
    agents, info = auto_train(N=5, num_rounds=3, timesteps_per_agent=5000)
"""
