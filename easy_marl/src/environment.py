"""Gymnasium-compatible multi-agent electricity market environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Callable, Dict, Optional, Tuple

from easy_marl.examples.bidding.market import clear_with_storage, market_clearing
from easy_marl.src.observators import OBSERVERS

# penalty per unit of unmet demand, this acts as a system stabilizer
# reduce this to encourage risk-taking agents
UNIT_LOL_PENALTY = 1


class MARLElectricityMarketEnv(gym.Env):
    """Simulates proportional bidding for multiple generators in a day-ahead market."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        agents,
        params=None,
        seed=None,
        agent_index: int = 0,
        observer_name: str = "simple",
    ) -> None:
        super().__init__()
        self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.agents = agents
        self.fixed_policies = [
            agent.fixed_act_function(deterministic=True) for agent in self.agents
        ]

        # Parameters
        params = params or {}
        self.agent_index = agent_index
        self.N_generators = params["N_generators"]
        self.T = params["T"]
        self.max_bid_delta = params.get("max_bid_delta", 50.0)
        self.lambda_bid_penalty = params.get("lambda_bid_penalty", 0.01)
        # Day-ahead: every agent commits to all T hours at once and one step is one day.
        self.day_ahead = bool(params.get("day_ahead", False))

        # Demand & generator parameters
        self.D_profile = np.array(params["demand_profile"], dtype=np.float32)
        self.base_D_profile = self.D_profile.copy()
        self.K = np.array(params["capacities"], dtype=np.float32)
        self.c = np.array(params["costs"], dtype=np.float32)
        assert len(self.D_profile) == self.T, (
            f"Demand profile length does not match T: {len(self.D_profile)} != {self.T}"
        )
        assert np.all(self.base_D_profile > 0)

        # Scaling
        self.demand_scale = (
            np.max(self.D_profile) if np.max(self.D_profile) > 0 else 1.0
        )
        self.cost_scale = np.max(self.c) if np.max(self.c) > 0 else 1.0
        self.capacity_scale = np.max(self.K) if np.max(self.K) > 0 else 1.0
        self.system_capacity = np.sum(self.K)

        # Optional battery: always the last agent, after the generators.
        # Scales above use generators only so generator observations do not depend on it.
        bess = params.get("bess")
        self.has_bess = bess is not None
        self.N = self.N_generators + int(self.has_bess)
        self.bess_index = self.N_generators if self.has_bess else None
        if self.has_bess:
            self.bess_power = float(bess["power_mw"])
            self.bess_energy = self.bess_power * float(bess["duration_h"])
            self.bess_leg_efficiency = float(np.sqrt(bess["efficiency_rt"]))
            self.bess_initial_soc = float(bess["initial_soc_frac"]) * self.bess_energy
            self.bess_bid_ref = float(bess["bid_ref"])
            self.K_all = np.append(self.K, self.bess_power).astype(np.float32)
            self.c_all = np.append(self.c, self.bess_bid_ref).astype(np.float32)
        else:
            self.K_all = self.K
            self.c_all = self.c

        # Observer setup (day-ahead agents use the fixed layout of _get_day_obs instead)
        obs_dim_fn, obs_fn = OBSERVERS[observer_name]
        if self.day_ahead:
            # Whole demand profile plus own capacity and cost (battery: power, energy, reference price).
            self.obs_dim = self.T + 2
            bess_obs_dim = self.T + 3
        else:
            self.obs_dim = obs_dim_fn(self.N_generators)
            # Battery sees the market features plus its state of charge.
            bess_obs_dim = obs_dim_fn(self.N) + 1
        self._obs_fn = obs_fn  # bind directly (no dict lookup later)
        self.other_mask = np.arange(self.N_generators) != self.agent_index
        self.obs_buf = np.empty(self.obs_dim, dtype=np.float32)
        if self.has_bess:
            self.bess_obs_buf = np.empty(bess_obs_dim, dtype=np.float32)

        is_bess_agent = self.has_bess and self.agent_index == self.bess_index
        own_obs_dim = len(self.bess_obs_buf) if is_bess_agent else self.obs_dim
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(own_obs_dim,), dtype=np.float32
        )

        # Action space; the battery's first component is signed (<0 charge, >0 discharge)
        reasonable_bound = 10
        quantity_low = -1.0 if is_bess_agent else 0.0
        if self.day_ahead:
            # Flat layout: T quantity parameters, then T price parameters.
            low = np.concatenate(
                [np.full(self.T, quantity_low), np.full(self.T, -reasonable_bound)]
            )
            high = np.concatenate(
                [np.full(self.T, 1.0), np.full(self.T, reasonable_bound)]
            )
        else:
            low = np.array([quantity_low, -reasonable_bound])
            high = np.array([1.0, reasonable_bound])
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32)
        )

        # misc
        self.system_capacity = np.sum(self.K)
        self.lower_stochastic_bound = -0.5 * np.min(self.D_profile)
        self.upper_stochastic_bound = 0.5 * (
            self.system_capacity - np.max(self.D_profile)
        )

        # place to store other agents' fixed action functions:
        # mapping agent_index -> function(obs) -> action (2-vector)
        self.other_action_fns: Dict[int, Callable] = {}

        # outputs / internal state
        self.reset(seed=seed)

    def run_stochastics(self) -> None:
        """Apply stochastic perturbations to the demand profile in-place."""

        # generate demand perturbation
        demand_peturbation = self.rng.normal(
            loc=0.0, scale=0.05 * self.demand_scale, size=self.T
        ).astype(np.float32)

        # clip to avoid negative demand
        demand_peturbation = np.clip(
            demand_peturbation,
            a_min=self.lower_stochastic_bound,
            a_max=self.upper_stochastic_bound,
        )

        # update demand profile with stochastic perturbation
        self.D_profile = self.base_D_profile + demand_peturbation

    def reset(self, seed=None):
        """Reset environment state and return the initial observation tuple."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.run_stochastics()

        self.t = 0

        self.output = {
            "bids": np.zeros((self.T, self.N), dtype=np.float32),
            "q_offered": np.zeros((self.T, self.N), dtype=np.float32),
            "q_cleared": np.zeros((self.T, self.N), dtype=np.float32),
            "market_prices": np.zeros(self.T, dtype=np.float32),
            "rewards": np.zeros((self.T, self.N), dtype=np.float32),
            "penalty": np.zeros((self.T, self.N), dtype=np.float32),
            "demand": np.zeros(self.T, dtype=np.float32),
        }
        if self.has_bess:
            self.output["bess_charge"] = np.zeros(self.T, dtype=np.float32)
            self.output["bess_discharge"] = np.zeros(self.T, dtype=np.float32)
            self.output["soc"] = np.zeros(self.T, dtype=np.float32)
            self.soc = self.bess_initial_soc
            self.bess_buy_bid = 0.0
            self.bess_buy_quantity = 0.0

        self.b_all = np.zeros(self.N, dtype=np.float32)
        self.q_all = np.zeros(self.N, dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """Return the normalized observation vector for ``agent_index``."""

        if agent_index is None:
            agent_index = self.agent_index

        if self.day_ahead:
            return self._get_day_obs(agent_index)

        if self.has_bess and agent_index == self.bess_index:
            self._obs_fn(
                self.D_profile,
                self.K_all,
                self.c_all,
                agent_index,
                self.demand_scale,
                self.capacity_scale,
                self.cost_scale,
                np.arange(self.N) != agent_index,
                self.bess_obs_buf[:-1],
                self.t,
            )
            self.bess_obs_buf[-1] = self.soc / self.bess_energy
            return self.bess_obs_buf

        obs = self._obs_fn(
            self.D_profile,
            self.K,
            self.c,
            agent_index,
            self.demand_scale,
            self.capacity_scale,
            self.cost_scale,
            np.arange(self.N_generators) != agent_index,
            self.obs_buf,
            self.t,
        )
        return obs

    def _get_day_obs(self, agent_index: int) -> np.ndarray:
        """Day-ahead observation: the whole demand profile plus the agent's own static features."""

        T = self.T
        if self.has_bess and agent_index == self.bess_index:
            buf = self.bess_obs_buf
            buf[T] = self.bess_power / self.capacity_scale
            buf[T + 1] = self.bess_energy / (self.capacity_scale * T)
            buf[T + 2] = self.bess_bid_ref / self.cost_scale
        else:
            buf = self.obs_buf
            buf[T] = self.K[agent_index] / self.capacity_scale
            buf[T + 1] = self.c[agent_index] / self.cost_scale
        buf[:T] = self.D_profile / self.demand_scale
        return buf

    def _as_schedule(self, action, agent_index: int) -> np.ndarray:
        """Return a (T, 2) schedule of [quantity, price] parameters for a day-ahead agent."""

        action = np.asarray(action, dtype=np.float32)
        if action.size == 2:
            # A constant action such as the null bid applies to every hour.
            return np.tile(action, (self.T, 1))
        if action.size != 2 * self.T:
            raise ValueError(
                f"Action for agent {agent_index} has size {action.size}, "
                f"expected 2 or {2 * self.T}."
            )
        return np.stack([action[: self.T], action[self.T :]], axis=1)

    def update_agent_bid(self, action: np.ndarray, agent_idx: int) -> None:
        """Project an agent's action into quantity and price bids."""

        if self.has_bess and agent_idx == self.bess_index:
            self._update_bess_bid(action)
            return

        q_t = float(1 - action[0]) * self.K[agent_idx]
        b_t = float(self.c[agent_idx]) + float(np.tanh(action[1])) * self.max_bid_delta

        self.q_all[agent_idx] = q_t
        self.b_all[agent_idx] = b_t

    def _update_bess_bid(self, action: np.ndarray) -> None:
        """Turn the battery action into either a sell offer or a buy bid."""

        power_frac = float(np.clip(action[0], -1.0, 1.0))
        price = self.bess_bid_ref + float(np.tanh(action[1])) * self.max_bid_delta
        self.b_all[self.bess_index] = price
        if power_frac >= 0:
            deliverable = self.soc * self.bess_leg_efficiency
            self.q_all[self.bess_index] = min(power_frac * self.bess_power, deliverable)
            self.bess_buy_quantity = 0.0
        else:
            headroom = (self.bess_energy - self.soc) / self.bess_leg_efficiency
            self.q_all[self.bess_index] = 0.0
            self.bess_buy_quantity = min(-power_frac * self.bess_power, headroom)
        self.bess_buy_bid = price

    def _clear_market(self, demand: float) -> Tuple[float, np.ndarray, float]:
        """Clear the market; returns price, sell dispatch per agent, battery charge."""

        if not self.has_bess:
            P_t, q_cleared = market_clearing(self.b_all, self.q_all, demand)
            return P_t, q_cleared, 0.0

        if self.q_all[self.bess_index] > 0:
            P_t, q_cleared = market_clearing(self.b_all, self.q_all, demand)
            return P_t, q_cleared, 0.0

        # A battery that offers nothing must not set the price, so it is left out as a seller.
        gens = slice(0, self.N_generators)
        P_t, q_generators, charged = clear_with_storage(
            self.b_all[gens],
            self.q_all[gens],
            demand,
            self.bess_buy_bid,
            self.bess_buy_quantity,
        )
        q_cleared = np.zeros(self.N, dtype=q_generators.dtype)
        q_cleared[gens] = q_generators
        return P_t, q_cleared, float(charged)

    def update_all_bids(self, exclude_agent_index: bool = True) -> None:
        """Populate bids for every agent using their fixed policies when available."""

        for j in range(self.N):
            if exclude_agent_index and j == self.agent_index:
                continue
            fn = self.fixed_policies[j]
            if fn is not None:
                obs_j = self._get_obs(agent_index=j)
                act_j = fn(obs_j)
                act_j = np.asarray(act_j, dtype=np.float32)
                if act_j.size >= 2:
                    self.update_agent_bid(act_j, j)
                else:
                    raise ValueError(
                        f"Action function for agent {j} returned invalid action of size {act_j.size}."
                    )

    def step(
        self, action, fixed_evaluation: bool = False
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict[str, float]]:
        """Advance the environment by one hour, or by the whole day in day-ahead mode."""
        if self.day_ahead:
            return self._step_day(action, fixed_evaluation)

        # Reset bids/q to default baseline
        # self.q_all[:] = self.K  # default: offer max capacity
        # self.b_all[:] = self.c  # default: bid at cost
        self.q_all[:] = np.full(self.N, np.nan)  # default: error if not set
        self.b_all[:] = np.full(self.N, np.nan)  # default: error if not set

        if fixed_evaluation:
            action = None  # ignore input action, use fixed policies for all agents
            self.update_all_bids(exclude_agent_index=False)
        else:
            # First, fill other agents using provided fixed policies (if any)
            self.update_all_bids(exclude_agent_index=True)

            # Update with current agent action
            self.update_agent_bid(action, self.agent_index)

        r = self._run_hour()

        # Step time
        self.t += 1
        done = self.t >= self.T
        obs = self._get_obs() if not done else None
        # gymnasium step returns (obs, reward, terminated, truncated, info)
        # keep compatibility: return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        return obs, r[self.agent_index], terminated, truncated, {}

    def _step_day(
        self, action, fixed_evaluation: bool
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict[str, float]]:
        """Collect every agent's full-day schedule, then clear the T hours in order."""

        schedules = np.empty((self.N, self.T, 2), dtype=np.float32)
        for j in range(self.N):
            if j == self.agent_index and not fixed_evaluation:
                agent_action = action
            else:
                agent_action = self.fixed_policies[j](self._get_obs(agent_index=j))
            schedules[j] = self._as_schedule(agent_action, j)

        day_reward = np.zeros(self.N)
        for t in range(self.T):
            self.t = t
            for j in range(self.N):
                self.update_agent_bid(schedules[j, t], j)
            day_reward += self._run_hour()

        self.t = self.T
        return None, day_reward[self.agent_index], True, False, {}

    def _run_hour(self) -> np.ndarray:
        """Clear hour ``self.t`` from the bids already in ``b_all``/``q_all``; returns scaled rewards."""

        # Run market clearing
        demand = self.D_profile[self.t]
        P_t, q_cleared, charged = self._clear_market(demand)

        # Rewards
        base_rewards = (P_t - self.c_all) * q_cleared

        # Bid regulariser
        bid_penalties = self.lambda_bid_penalty * (self.b_all - self.c_all) ** 2

        # Penalty for loss of load (energy bought by the battery is not served load)
        total_cleared = np.sum(q_cleared) - charged
        loss_of_load_penalty = UNIT_LOL_PENALTY * max(0, demand - total_cleared)

        r = base_rewards - bid_penalties - loss_of_load_penalty

        if self.has_bess:
            discharged = float(q_cleared[self.bess_index])
            self.soc += (
                charged * self.bess_leg_efficiency
                - discharged / self.bess_leg_efficiency
            )
            self.soc = min(max(self.soc, 0.0), self.bess_energy)
            r[self.bess_index] = (
                P_t * (discharged - charged)
                - bid_penalties[self.bess_index]
                - loss_of_load_penalty
            )
            # Stops the battery from selling its starting energy for free.
            if self.t == self.T - 1:
                terminal_penalty = max(0.0, self.bess_initial_soc - self.soc) * (
                    self.bess_bid_ref + self.max_bid_delta
                )
                r[self.bess_index] -= terminal_penalty
                self.output["penalty"][self.t, self.bess_index] = terminal_penalty
            self.output["bess_charge"][self.t] = charged
            self.output["bess_discharge"][self.t] = discharged
            self.output["soc"][self.t] = self.soc

        # Scale reward
        r /= self.demand_scale * self.cost_scale * max(1, self.T)
        r *= 20  # scale to reasonable range

        # Store outputs
        t_idx = self.t
        self.output["bids"][t_idx] = self.b_all.copy()
        self.output["q_offered"][t_idx] = self.q_all.copy()
        self.output["q_cleared"][t_idx] = q_cleared
        self.output["market_prices"][t_idx] = P_t
        self.output["rewards"][t_idx] = r
        self.output["demand"][t_idx] = demand
        return r

    def render(self) -> None:
        """Print latest timestep data (for basic debugging only)."""

        t_idx = min(self.t - 1, self.T - 1)
        print(
            f"t={self.t}, Demand={self.D_profile[t_idx]:.2f}, Bid={self.output['bids'][t_idx, self.agent_index]:.2f}"
        )

    def seed(self, seed=None):
        """Seed Gym's RNG utility and return the resulting seed list."""

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_metadata(self) -> Dict[str, float]:
        """Return a JSON-serializable dictionary describing the environment."""

        metadata = {
            "N_generators": self.N_generators,
            "T": self.T,
            "capacities": self.K.tolist(),
            "costs": self.c.tolist(),
            "demand_profile": self.D_profile.tolist(),
            "max_bid_delta": self.max_bid_delta,
            "lambda_bid_penalty": self.lambda_bid_penalty,
            "day_ahead": self.day_ahead,
        }
        if self.has_bess:
            metadata["bess"] = {
                "power_mw": self.bess_power,
                "energy_mwh": self.bess_energy,
                "leg_efficiency": self.bess_leg_efficiency,
                "initial_soc": self.bess_initial_soc,
                "bid_ref": self.bess_bid_ref,
            }
        return metadata
