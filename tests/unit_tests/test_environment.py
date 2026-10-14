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


# --- Day-ahead mode: each agent commits to all T hours in a single step ---
DAY_PARAMS = {**PARAMS, "demand_profile": [60.0] * 5, "day_ahead": True}
DAY_BESS_PARAMS = {**BESS_PARAMS, "day_ahead": True}
T_DAY = PARAMS["T"]


def day_action(quantity_params, price_params):
    return np.concatenate([quantity_params, price_params]).astype(np.float32)


def make_day_env(agent_action=None, agent_index=0, params=DAY_PARAMS, n_generators=3):
    agents = [MockAgent(GENERATOR_NULL) for _ in range(n_generators)]
    if "bess" in params:
        agents.append(MockAgent(agent_action if agent_action is not None else IDLE))
    return MARLElectricityMarketEnv(agents=agents, params=params, agent_index=agent_index)


class TestMARLEnvDayAhead:
    def test_that_one_step_plays_the_whole_day_and_terminates(self):
        env = make_day_env()
        env.reset()
        action = day_action(np.zeros(T_DAY), np.zeros(T_DAY))

        obs, reward, terminated, truncated, _ = env.step(action)

        assert terminated and not truncated
        assert env.t == T_DAY
        assert np.all(env.output["demand"] == 60.0)
        assert np.all(env.output["market_prices"] > 0)

    def test_that_spaces_cover_the_full_day(self):
        env = make_day_env()
        obs, _ = env.reset()

        assert env.action_space.shape == (2 * T_DAY,)
        assert obs.shape == env.observation_space.shape
        # The agent sees the whole demand profile, scaled by the peak.
        np.testing.assert_allclose(obs[:T_DAY], 1.0)

    def test_that_null_actions_match_the_hourly_environment(self):
        hourly_params = {k: v for k, v in DAY_PARAMS.items() if k != "day_ahead"}
        agents = [MockAgent(GENERATOR_NULL) for _ in range(3)]
        hourly = run_fixed_episode(MARLElectricityMarketEnv(agents=agents, params=hourly_params))

        day_env = MARLElectricityMarketEnv(agents=agents, params=DAY_PARAMS)
        day_env.reset()
        day_env.step(None, fixed_evaluation=True)

        np.testing.assert_allclose(day_env.output["market_prices"], hourly["market_prices"])
        np.testing.assert_allclose(day_env.output["q_cleared"], hourly["q_cleared"])
        np.testing.assert_allclose(day_env.output["rewards"], hourly["rewards"], rtol=1e-6)

    def test_that_each_hour_uses_its_own_entry_of_the_action(self):
        env = make_day_env()
        env.reset()
        quantity_params = np.zeros(T_DAY)
        quantity_params[2] = 1.0  # withhold everything in hour 2 only
        env.step(day_action(quantity_params, np.zeros(T_DAY)))

        offered = env.output["q_offered"][:, 0]
        assert offered[2] == 0.0
        assert np.all(np.delete(offered, 2) == PARAMS["capacities"][0])

    def test_that_the_step_reward_is_the_sum_of_the_hourly_rewards(self):
        env = make_day_env()
        env.reset()
        _, reward, _, _, _ = env.step(day_action(np.zeros(T_DAY), np.zeros(T_DAY)))

        assert reward > 0
        assert reward == np.float32(env.output["rewards"][:, 0].sum())

    def test_that_a_wrong_sized_fixed_policy_action_is_rejected(self):
        agents = [MockAgent(np.zeros(3)) for _ in range(3)]
        env = MARLElectricityMarketEnv(agents=agents, params=DAY_PARAMS)
        env.reset()
        try:
            env.step(None, fixed_evaluation=True)
        except ValueError as error:
            assert "size 3" in str(error)
        else:
            raise AssertionError("expected ValueError for a 3-element action")

    def test_that_a_battery_schedule_charges_then_discharges_within_its_limits(self):
        energy = BESS_POWER * BESS_DURATION
        leg = np.sqrt(BESS_EFFICIENCY)
        env = make_day_env(agent_index=BESS_INDEX, params=DAY_BESS_PARAMS)
        env.reset()
        power = np.zeros(T_DAY)
        power[0], power[2] = -1.0, 1.0  # charge in hour 0, discharge in hour 2
        price = np.zeros(T_DAY)
        price[0], price[2] = 10.0, -10.0  # buy high, offer low so both clear
        env.step(day_action(power, price))

        out = env.output
        assert out["bess_charge"][0] == BESS_POWER
        np.testing.assert_allclose(out["soc"][0], 0.5 * energy + BESS_POWER * leg, rtol=1e-5)
        assert out["bess_discharge"][2] > 0
        assert out["bess_charge"][1] == 0 and out["bess_discharge"][1] == 0
        assert np.all(out["soc"] >= -1e-6) and np.all(out["soc"] <= energy + 1e-6)

    def test_that_battery_observation_shows_demand_and_its_own_size(self):
        env = make_day_env(agent_index=BESS_INDEX, params=DAY_BESS_PARAMS)
        obs, _ = env.reset()

        assert obs.shape == env.observation_space.shape
        assert env.action_space.low[0] == -1.0
        assert env.action_space.shape == (2 * T_DAY,)


# --- Multi-day episodes: charge carries over and agents see yesterday's prices ---
DAYS = 3
MULTI_PARAMS = {**DAY_PARAMS, "episode_days": DAYS}
MULTI_BESS_PARAMS = {**DAY_BESS_PARAMS, "episode_days": DAYS}


def make_multi_env(battery_action=None, agent_index=0, params=MULTI_BESS_PARAMS, seed=None):
    agents = [MockAgent(GENERATOR_NULL) for _ in range(3)]
    if "bess" in params:
        agents.append(MockAgent(battery_action if battery_action is not None else IDLE))
    return MARLElectricityMarketEnv(
        agents=agents, params=params, agent_index=agent_index, seed=seed
    )


class TestMARLEnvMultiDay:
    def test_that_multi_day_needs_day_ahead_mode(self):
        hourly = {k: v for k, v in MULTI_PARAMS.items() if k != "day_ahead"}
        try:
            MARLElectricityMarketEnv(agents=[MockAgent(GENERATOR_NULL)] * 3, params=hourly)
        except ValueError as error:
            assert "day_ahead" in str(error)
        else:
            raise AssertionError("expected ValueError for episode_days without day_ahead")

    def test_that_episode_days_below_one_is_rejected(self):
        try:
            MARLElectricityMarketEnv(
                agents=[MockAgent(GENERATOR_NULL)] * 3, params={**DAY_PARAMS, "episode_days": 0}
            )
        except ValueError as error:
            assert "episode_days" in str(error)
        else:
            raise AssertionError("expected ValueError for episode_days=0")

    def test_that_a_non_integer_episode_days_is_rejected(self):
        try:
            MARLElectricityMarketEnv(
                agents=[MockAgent(GENERATOR_NULL)] * 3, params={**DAY_PARAMS, "episode_days": 2.9}
            )
        except ValueError as error:
            assert "episode_days" in str(error)
        else:
            raise AssertionError("expected ValueError for episode_days=2.9")

    def test_that_a_bool_episode_days_is_rejected(self):
        try:
            MARLElectricityMarketEnv(
                agents=[MockAgent(GENERATOR_NULL)] * 3, params={**DAY_PARAMS, "episode_days": True}
            )
        except ValueError as error:
            assert "episode_days" in str(error)
        else:
            raise AssertionError("expected ValueError for episode_days=True")

    def test_that_a_numpy_integer_episode_days_is_accepted(self):
        env = MARLElectricityMarketEnv(
            agents=[MockAgent(GENERATOR_NULL)] * 3,
            params={**DAY_PARAMS, "episode_days": np.int64(3)},
        )
        assert env.episode_days == 3

    def test_that_generators_see_demand_own_features_and_yesterdays_prices(self):
        env = make_multi_env(params=MULTI_PARAMS)
        obs, _ = env.reset()

        assert obs.shape == (2 * T_DAY + 2,) == env.observation_space.shape
        # Multi-day mode adds daily noise, so demand is no longer exactly the flat profile.
        np.testing.assert_allclose(obs[:T_DAY], env.D_profile / env.demand_scale)
        np.testing.assert_allclose(obs[T_DAY + 2 :], env.prev_prices / env.cost_scale)

    def test_that_battery_also_sees_its_charge(self):
        env = make_multi_env(agent_index=BESS_INDEX)
        obs, _ = env.reset()

        assert obs.shape == (2 * T_DAY + 4,) == env.observation_space.shape
        np.testing.assert_allclose(obs[T_DAY + 3 : 2 * T_DAY + 3], env.prev_prices / env.cost_scale)
        assert obs[2 * T_DAY + 3] == 0.5  # starts half full

    def test_that_first_day_prices_are_the_at_cost_clearing_of_that_days_demand(self):
        env = make_multi_env(params=MULTI_PARAMS)
        env.reset()
        # Demand 60 against 50 MW blocks costing 10, 20, 30: the second block sets the price.
        np.testing.assert_allclose(env.prev_prices, 20.0)

    def test_that_multi_day_observation_space_allows_negative_prices(self):
        env = make_multi_env(params=MULTI_PARAMS)
        assert np.all(env.observation_space.low == -np.inf)

    def test_that_single_day_observations_are_unchanged(self):
        env = make_day_env()
        obs, _ = env.reset()
        assert obs.shape == (T_DAY + 2,)
        assert np.all(env.observation_space.low == 0)

    def test_that_the_episode_ends_only_after_the_last_day(self):
        env = make_multi_env(params=MULTI_PARAMS)
        env.reset()
        for day in range(DAYS):
            obs, _, terminated, truncated, _ = env.step(None, fixed_evaluation=True)
            assert terminated == (day == DAYS - 1) and not truncated
            assert (obs is None) == terminated
        assert env.output["market_prices"].shape == (DAYS * T_DAY,)
        assert np.all(env.output["market_prices"] > 0)

    def test_that_tomorrows_observation_shows_todays_prices(self):
        env = make_multi_env(params=MULTI_PARAMS)
        env.reset()
        obs, _, _, _, _ = env.step(None, fixed_evaluation=True)
        today = env.output["market_prices"][:T_DAY]
        np.testing.assert_allclose(env.prev_prices, today)
        np.testing.assert_allclose(obs[T_DAY + 2 :], today / env.cost_scale)

    def test_that_charge_carries_over_to_the_next_day(self):
        energy = BESS_POWER * BESS_DURATION
        leg = np.sqrt(BESS_EFFICIENCY)
        env = make_multi_env(FULL_CHARGE)
        env.reset()
        env.step(None, fixed_evaluation=True)
        end_of_day_one = env.output["soc"][T_DAY - 1]
        assert end_of_day_one > 0.5 * energy
        env.step(None, fixed_evaluation=True)
        # Day two starts from where day one ended, not from half full.
        first_hour_day_two = env.output["soc"][T_DAY]
        expected = min(end_of_day_one + env.output["bess_charge"][T_DAY] * leg, energy)
        np.testing.assert_allclose(first_hour_day_two, expected, rtol=1e-5)

    def test_that_the_battery_observation_shows_its_charge_each_day(self):
        env = make_multi_env(FULL_CHARGE, agent_index=BESS_INDEX)
        env.reset()
        obs, _, _, _, _ = env.step(None, fixed_evaluation=True)
        energy = BESS_POWER * BESS_DURATION
        np.testing.assert_allclose(obs[2 * T_DAY + 3], env.output["soc"][T_DAY - 1] / energy, rtol=1e-6)

    def test_that_multi_day_markets_never_charge_a_penalty(self):
        env = make_multi_env(FULL_DISCHARGE)
        run_fixed_episode(env)
        soc0 = MULTI_BESS_PARAMS["bess"]["initial_soc_frac"] * BESS_POWER * BESS_DURATION

        # The battery ends below where it started, which the single-day market would penalise.
        assert env.output["soc"][-1] < soc0
        assert np.all(env.output["penalty"] == 0)
        # Its reward is its trading profit and nothing else.
        hourly = env.output["market_prices"] * (env.output["bess_discharge"] - env.output["bess_charge"])
        scale = 20 / (env.demand_scale * env.cost_scale * T_DAY)
        np.testing.assert_allclose(env.output["rewards"][:, BESS_INDEX], hourly * scale, rtol=1e-5)

    def test_that_charge_stays_within_limits_across_days(self):
        energy = BESS_POWER * BESS_DURATION
        for action in (FULL_CHARGE, FULL_DISCHARGE):
            env = make_multi_env(action)
            for _ in range(3):  # several episodes, so the charge also crosses episode boundaries
                run_fixed_episode(env)
                assert np.all(env.output["soc"] >= -1e-6)
                assert np.all(env.output["soc"] <= energy + 1e-6)

    def test_that_demand_is_redrawn_every_day_and_reproducible_by_seed(self):
        def demands(seed):
            env = make_multi_env(params=MULTI_PARAMS, seed=seed)
            run_fixed_episode(env)
            return env.output["demand"].reshape(DAYS, T_DAY)

        first = demands(seed=3)
        assert not np.allclose(first[0], first[1])
        np.testing.assert_array_equal(first, demands(seed=3))

    def test_that_day_one_stand_in_matches_that_days_own_demand(self):
        params = {**MULTI_PARAMS, "demand_profile": [40.0, 55.0, 70.0, 90.0, 110.0]}
        env = make_multi_env(params=params, seed=3)

        env.reset()

        np.testing.assert_allclose(env.prev_prices, env._at_cost_prices())

    def test_that_reset_keeps_the_charge_and_prices_but_draws_new_demand(self):
        env = make_multi_env(FULL_DISCHARGE, seed=3)
        run_fixed_episode(env)
        soc_at_end = env.soc
        last_prices = env.output["market_prices"][-T_DAY:].copy()
        last_demand = env.D_profile.copy()

        obs, _ = env.reset()

        assert env.day == 0
        assert soc_at_end < env.bess_initial_soc
        assert env.soc == soc_at_end
        np.testing.assert_allclose(env.prev_prices, last_prices)
        assert not np.allclose(env.D_profile, last_demand)
        assert obs is not None
