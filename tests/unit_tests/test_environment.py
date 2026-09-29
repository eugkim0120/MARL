import numpy as np
import gymnasium as gym
from easy_marl.src.environment import MARLElectricityMarketEnv


# --- Mocks ---
class MockAgent:
    def __init__(self, action_to_return):
        self.action = action_to_return

    def fixed_act_function(self, deterministic=True):
        return lambda obs: self.action


# --- Test Data ---
PARAMS = {
    "N_generators": 3,
    "T": 5,
    "demand_profile": [100.0] * 5,
    "capacities": [50.0, 50.0, 50.0],
    "costs": [10.0, 20.0, 30.0],
    "max_bid_delta": 10.0,
    "lambda_bid_penalty": 0.0,  # Simplify reward check by removing penalty
}


class TestMARLEnv:
    """Tests for MARLElectricityMarketEnv."""

    def test_initialization(self):
        """Test environment initializes with correct parameters."""
        agents = [MockAgent([0.0, 0.0]) for _ in range(3)]
        env = MARLElectricityMarketEnv(agents=agents, params=PARAMS)

        assert env.N == 3
        assert env.T == 5
        assert env.observation_space.shape[0] > 0
        assert isinstance(env.action_space, gym.spaces.Box)

    def test_reset(self):
        """Test reset returns correct initial observation."""
        agents = [MockAgent([0.0, 0.0]) for _ in range(3)]
        env = MARLElectricityMarketEnv(agents=agents, params=PARAMS)

        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert env.t == 0
        # Check internal tracking reset
        assert np.all(env.output["market_prices"] == 0)

    def test_step_logic(self):
        """Test step function clears market and returns rewards."""
        # Setup:
        # Agent 0 (Learning): Cost 10. Action: offer max cap (0.0), price delta 0 (0.0) -> Bid ~10
        # Agent 1 (Fixed): Cost 20. Action: null -> Bid 20
        # Agent 2 (Fixed): Cost 30. Action: null -> Bid 30
        # Demand: 100
        # Expected:
        # Ag 0 (50 MW) + Ag 1 (50 MW) meet demand.
        # Clearing price: 20 (marginal unit from Ag 1)
        # Agent 0 reward should be positive: (20 - 10) * 50 = 500 (unscaled)

        agents = [
            None,  # Agent 0 is self
            MockAgent(np.array([0.0, 0.0])),  # Agent 1
            MockAgent(np.array([0.0, 0.0])),  # Agent 2
        ]

        # We need to fill the list with something for agent 0 too
        # because environment init iterates over all provided agents
        # even though it skips self.agent_index in update_all_bids logic EXCEPT for fixed_policies list creation
        # Wait, the code says:
        # self.fixed_policies = [agent.fixed_act_function(...) for agent in self.agents]
        # so self.agents MUST contain objects for all indices including self.

        agents[0] = MockAgent(np.array([0.0, 0.0]))

        env = MARLElectricityMarketEnv(agents=agents, params=PARAMS, agent_index=0)
        env.reset()

        # Action for agent 0: [0.0 (max qty), 0.0 (0 delta)]
        # sigmoid(0) -> bid delta calculation... wait
        # env logic: b_t = c + tanh(action[1]) * max_bid_delta
        # tanh(0) = 0 -> bid = cost

        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Check dynamics
        assert env.t == 1
        assert not terminated
        assert not truncated

        # Check market result used for reward
        # We can inspect env.output for t=0
        market_price = env.output["market_prices"][0]
        q_cleared = env.output["q_cleared"][0]

        assert market_price == 20.0, f"Expected clearing price 20.0, got {market_price}"
        assert q_cleared[0] == 50.0  # Full capacity cleared for cheapest

        # Reward check
        # Reward is scaled, so just check it's positive
        assert reward > 0

    def test_episode_end(self):
        """Test that environment terminates after T steps."""
        agents = [MockAgent([0.0, 0.0]) for _ in range(3)]
        env = MARLElectricityMarketEnv(agents=agents, params=PARAMS)
        env.reset()

        done = False
        steps = 0
        while not done:
            obs, r, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            done = terminated or truncated
            steps += 1

        assert steps == PARAMS["T"]
        assert env.t == PARAMS["T"]

    def test_constraints(self):
        """Verify capacity constraints."""
        # Demand > Total Capacity
        # Total Cap = 150. Demand = 200.
        params_high_demand = PARAMS.copy()
        params_high_demand["demand_profile"] = [200.0] * 5

        agents = [MockAgent([0.0, 0.0]) for _ in range(3)]
        env = MARLElectricityMarketEnv(agents=agents, params=params_high_demand)
        env.reset()

        _, _, _, _, _ = env.step(np.array([0.0, 0.0]))

        q_cleared = env.output["q_cleared"][0]
        assert np.sum(q_cleared) <= np.sum(PARAMS["capacities"])
        assert np.allclose(
            q_cleared, PARAMS["capacities"], atol=1e-5
        )  # Should run full out


# --- Storage (BESS) ---
BESS_POWER = 20.0
BESS_DURATION = 2.0
BESS_EFFICIENCY = 0.81  # sqrt = 0.9 per leg
BESS_PARAMS = {
    **PARAMS,
    # Below 100 so a discharge offer at ~20 is inside the merit order.
    "demand_profile": [60.0] * 5,
    "bess": {
        "power_mw": BESS_POWER,
        "duration_h": BESS_DURATION,
        "efficiency_rt": BESS_EFFICIENCY,
        "initial_soc_frac": 0.5,
        "bid_ref": 25.0,
    },
}
BESS_INDEX = PARAMS["N_generators"]
GENERATOR_NULL = np.array([0.0, 0.0])
IDLE = np.array([0.0, 0.0])
FULL_CHARGE = np.array([-1.0, 10.0])  # buy at up to bid_ref + max_bid_delta
FULL_DISCHARGE = np.array([1.0, -10.0])  # offer at bid_ref - max_bid_delta = 15


def make_bess_env(bess_action, agent_index=0, params=BESS_PARAMS):
    agents = [MockAgent(GENERATOR_NULL) for _ in range(3)] + [MockAgent(bess_action)]
    return MARLElectricityMarketEnv(agents=agents, params=params, agent_index=agent_index)


def run_fixed_episode(env):
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(None, fixed_evaluation=True)
        done = terminated or truncated
    return env.output


class TestMARLEnvWithStorage:
    """Tests for the optional battery (BESS) agent."""

    def test_that_idle_battery_leaves_generator_outcomes_unchanged(self):
        plain_params = {k: v for k, v in BESS_PARAMS.items() if k != "bess"}
        without = run_fixed_episode(
            MARLElectricityMarketEnv(
                agents=[MockAgent(GENERATOR_NULL) for _ in range(3)], params=plain_params
            )
        )
        with_idle = run_fixed_episode(make_bess_env(IDLE))

        np.testing.assert_allclose(with_idle["market_prices"], without["market_prices"])
        np.testing.assert_allclose(with_idle["q_cleared"][:, :3], without["q_cleared"])
        assert np.all(with_idle["q_cleared"][:, BESS_INDEX] == 0)
        assert np.all(with_idle["bess_charge"] == 0)

    def test_that_generator_observations_do_not_depend_on_battery(self):
        plain = MARLElectricityMarketEnv(
            agents=[MockAgent(GENERATOR_NULL) for _ in range(3)],
            params={k: v for k, v in BESS_PARAMS.items() if k != "bess"},
            observer_name="simple_v3",
        )
        with_bess = MARLElectricityMarketEnv(
            agents=[MockAgent(GENERATOR_NULL) for _ in range(3)] + [MockAgent(IDLE)],
            params=BESS_PARAMS,
            observer_name="simple_v3",
        )
        for j in range(3):
            np.testing.assert_allclose(
                with_bess._get_obs(j).copy(), plain._get_obs(j).copy()
            )

    def test_that_battery_agent_has_signed_action_space_and_soc_observation(self):
        env = make_bess_env(IDLE, agent_index=BESS_INDEX)
        assert env.action_space.low[0] == -1.0
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs[-1] == 0.5

    def test_that_charging_respects_power_energy_and_efficiency(self):
        energy = BESS_POWER * BESS_DURATION
        leg = np.sqrt(BESS_EFFICIENCY)
        output = run_fixed_episode(make_bess_env(FULL_CHARGE))

        assert np.all(output["bess_charge"] <= BESS_POWER + 1e-6)
        assert np.all(output["soc"] <= energy + 1e-6)
        # First hour: full power, stored energy is scaled by one efficiency leg.
        assert output["bess_charge"][0] == BESS_POWER
        np.testing.assert_allclose(output["soc"][0], 0.5 * energy + BESS_POWER * leg, rtol=1e-5)
        # Battery fills and stops charging.
        np.testing.assert_allclose(output["soc"][-1], energy, rtol=1e-5)
        assert output["bess_charge"][-1] == 0.0

    def test_that_discharging_sells_energy_and_drains_soc_through_efficiency(self):
        energy = BESS_POWER * BESS_DURATION
        leg = np.sqrt(BESS_EFFICIENCY)
        output = run_fixed_episode(make_bess_env(FULL_DISCHARGE))

        sold = output["bess_discharge"][0]
        assert sold == output["q_cleared"][0, BESS_INDEX]
        assert sold > 0
        np.testing.assert_allclose(output["soc"][0], 0.5 * energy - sold / leg, rtol=1e-5)
        assert np.all(output["soc"] >= -1e-6)

    def test_that_ending_below_initial_soc_is_penalised_at_the_last_step(self):
        output = run_fixed_episode(make_bess_env(FULL_DISCHARGE))
        params = BESS_PARAMS["bess"]
        soc0 = params["initial_soc_frac"] * BESS_POWER * BESS_DURATION
        expected = (soc0 - output["soc"][-1]) * (params["bid_ref"] + PARAMS["max_bid_delta"])

        assert expected > 0
        np.testing.assert_allclose(output["penalty"][-1, BESS_INDEX], expected, rtol=1e-5)
        assert np.all(output["penalty"][:-1, BESS_INDEX] == 0)
