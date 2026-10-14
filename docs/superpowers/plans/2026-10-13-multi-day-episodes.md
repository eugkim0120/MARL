# Multi-Day Episodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the market run as a continuous sequence of days: the battery's charge is never reset (so no end-of-charge penalty is needed), and every agent sees the previous day's prices.

**Architecture:** One new opt-in parameter, `params["episode_days"]` (default 1). With the default, every code path behaves exactly as today, so the finished `results/bess_full/` sweep still replays bit for bit. With `episode_days > 1` (day-ahead mode only), one environment step is still one day and an episode is `episode_days` days, but episode boundaries only exist for PPO bookkeeping: the battery's charge and the previous-day prices carry straight across them. Observations gain the previous day's 24 prices (all agents) plus the charge level (battery only). There is no charge penalty. Demand noise is redrawn every day. The sweep code splits each multi-day episode back into days before computing metrics, so every metric keeps its per-day meaning.

**Tech Stack:** Python 3.13, numpy, gymnasium, stable-baselines3 PPO (wrapped by `easy_marl.src.agents.PPOAgent`), pytest.

**Spec:** The "Design" section below is the spec. It was agreed in chat on 2026-10-13 and extends open decision 3 in `research/bess_followup/NOTES.md`.

## Design (the spec)

The user agreed to bundle two changes into one redesign:

1. **Generators and the battery see the previous day's outturn prices.** Real day-ahead bidders see yesterday's prices. This means an episode needs more than one day, because today each day is its own episode with no memory (`easy_marl/src/environment.py:382` always returns `terminated=True`).
2. **The battery's charge carries over, with no penalty.** Today it resets to 50% every day and pays `(bid_ref + max_bid_delta)` per MWh for ending a day below 50%; the penalty exists only to stop it selling its free starting charge. The user decided (2026-10-14) there should be no penalty at all. With this change, the charge is set to the initial level once, when the environment is constructed, and is never reset afterwards, not even by `reset()` at an episode boundary. The free starting charge is then spent once per environment: in training that is once per `timesteps_per_agent` days (20,000 in the full preset), and in evaluation an unrecorded warm-up episode spends it before anything is measured.

Decisions made here, with reasons:

- **Episode length 7 days**, as a preset value (`episode_days: 7`), not hard-coded.
- **Observation layout** (day-ahead, `episode_days > 1`):
  - Generator, `2T + 2` = 50 numbers: `[demand (T) / demand_scale, capacity / capacity_scale, cost / cost_scale, previous-day prices (T) / cost_scale]`.
  - Battery, `2T + 4` = 52 numbers: `[demand (T) / demand_scale, power / capacity_scale, energy / (capacity_scale * T), bid_ref / cost_scale, previous-day prices (T) / cost_scale, soc / energy]`.
  - No "days left" feature: nothing happens at an episode end, so the battery has no reason to know where it is in the episode.
  - With `episode_days > 1`, the observation space's lower bound is `-inf`, because prices can be negative (generators can bid down to `cost - max_bid_delta`).
- **"Previous-day prices" on the very first day** after the environment is constructed: the price each hour would clear at if every generator offered its full capacity at cost, computed from that day's demand. After that, the previous-day prices always come from the day before, across episode boundaries too.
- **What `reset()` does in multi-day mode:** sets `day` back to 0, redraws demand, and re-sizes the output arrays. It does not touch `soc` or `prev_prices`. `reset(seed=...)` reseeds the demand random number generator and otherwise behaves the same.
- **Demand noise is redrawn every day** in multi-day mode. Today, noise is drawn only by `reset(seed=...)`, and `reset()` without a seed reuses the same demand. Verified on 2026-10-13: within one training call, every simulated day has identical demand. In multi-day mode that would make the previous-day prices carry no information, so it must change there. Single-day mode keeps the current behaviour so old results still replay.
- **Evaluation runs one continuous environment.** In multi-day mode, `simulate_episodes` builds one environment seeded with `seed`, plays one unrecorded warm-up episode (which spends the free starting charge and replaces the at-cost first-day prices with real ones), then records `num_episodes` consecutive episodes. Single-day mode keeps its one-environment-per-episode loop with `seed + ep`, unchanged.
- **Metrics stay per day.** `simulate_episodes` reshapes `(episodes, days * T)` arrays into `(episodes * days, T)`. `summarise_market` and `bess_analysis.py` need no change.

What this changes about the experiment, which the report must state:

- In the frozen arm, generator policies still never update, but they now react to yesterday's prices. A battery changes those prices, so the frozen generators bid differently with and without the battery. The paired difference then measures the battery plus the generators' immediate reaction to it, without retraining.

Out of scope: wear cost, weekly load shape, solar, Nash benchmark, best-round policy selection, and running the full sweep. The plan ends with a smoke run and a report to the user. The full sweep needs the user's go-ahead.

## Global Constraints

- Default behaviour must be unchanged: with `episode_days` absent or 1, every existing test passes unmodified, and the stored `results/bess_full` metrics still replay.
- Multi-day mode is only valid with `day_ahead=True`. Anything else raises `ValueError`.
- Fail loudly. No silent fallbacks, no broad `except`, no defaults that hide a missing value. The one allowed default is `preset.get("episode_days", 1)`, because presets stored in old `metrics.json` files predate the key and did run single days. Comment that reason at each use.
- Use the virtualenv interpreter: `/home/eugene/projects/MARL/.venv/bin/python`. `python` and `python3` are not on PATH.
- Test command: `.venv/bin/python -m pytest tests/unit_tests -q --deselect tests/unit_tests/test_agents.py::TestFixedActionFunction`. The deselected class has 3 known, unrelated failures. Baseline on 2026-10-13: `128 passed, 6 deselected` in about 2 minutes.
- Work on a new branch `feat/multi-day-episodes`, created from `feat/day-ahead-bidding`. Commit after each task. Never push to `upstream` (wd7512/MARL). Push to `origin` only when the user asks.
- Match the surrounding style: docstrings in plain sentences, comments only for *why*, test names `test_that_<behaviour>`.
- Do not start the full sweep. Task 6 runs one probe config and stops.

## Review Focus

1. **SB3 auto-reset.** After the last day, stable-baselines3 calls `reset()` with no seed. That reset must start a new episode at day 0 with new demand noise, while keeping the battery's charge and the previous-day prices exactly as the last day left them. Pinned in Task 2 (`test_that_reset_keeps_the_charge_and_prices_but_draws_new_demand`).
2. **Old 26-number generator checkpoints loaded into a 50-number environment.** A cached pretrain from the single-day sweep must fail loudly, never load silently. Pinned in Task 4 (`test_that_single_day_generators_cannot_load_into_a_multi_day_market`).
3. **Negative prices in observations.** The observation space must allow them in multi-day mode. Pinned in Task 1 (`test_that_multi_day_observation_space_allows_negative_prices`).
4. **The charge level stays within `[0, energy]` over many days** of full discharge or full charge with carry-over. Pinned in Task 2 (`test_that_charge_stays_within_limits_across_days`).
5. **Old results still replay.** `episode_days` defaults must leave `simulate_episodes` output identical for single-day markets. Pinned in Task 4 (`test_that_single_day_dispatch_shape_is_unchanged`) and checked against a real stored config in Task 6, step 1.

---

## File Structure

- Modify `easy_marl/src/environment.py`: the `episode_days` parameter, observation layout, multi-day stepping, charge carry-over, no penalty in multi-day mode, daily demand redraw, and first-day prices.
- Modify `easy_marl/examples/bidding/training.py`: pass `episode_days` through `make_default_params` and `make_bess_params`.
- Modify `easy_marl/examples/bidding/bess_experiment.py`: `SweepConfig.episode_days`, preset keys, day-splitting in `simulate_episodes`, wiring through `get_pretrained`, `build_configs` and `main`, plus a report note.
- Modify `tests/unit_tests/test_environment.py`: new `TestMARLEnvMultiDay` class.
- Modify `tests/unit_tests/test_bidding/test_training.py` and `tests/unit_tests/test_bidding/test_bess_experiment.py`: new tests.
- Modify `research/bess_followup/NOTES.md`: mark decision 3 as implemented and record the probe result.

---

### Task 1: Parameter validation and observation layout

**Files:**
- Modify: `easy_marl/src/environment.py:44-103` (`__init__`), `:186-239` (`_get_obs`, `_get_day_obs`), `:155-184` (`reset`), `:454-475` (`get_metadata`)
- Test: `tests/unit_tests/test_environment.py` (append a new class at the end)

**Interfaces:**
- Produces: `env.episode_days: int`, `env.multi_day: bool`, `env.day: int` (0-based day within the current episode, set to 0 in `reset`), `env.prev_prices: np.ndarray` of shape `(T,)` float32 (only meaningful when `multi_day`; `None` until the first `reset` computes the at-cost prices), and `env._at_cost_prices() -> np.ndarray` of shape `(T,)`. Observation sizes: generator `2T+2`, battery `2T+4` when `multi_day`.

- [ ] **Step 1: Confirm the branch**

The branch `feat/multi-day-episodes` already exists, created from `feat/day-ahead-bidding` with this plan committed on it. Confirm:

```bash
cd /home/eugene/projects/MARL
git branch --show-current
```

Expected: `feat/multi-day-episodes`. If not, stop and report.

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit_tests/test_environment.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_environment.py::TestMARLEnvMultiDay -v`
Expected: the validation tests FAIL ("expected ValueError"), the observation tests FAIL with a shape mismatch or `AttributeError: ... 'prev_prices'`. `test_that_single_day_observations_are_unchanged` PASSES.

- [ ] **Step 4: Implement**

In `__init__`, directly after the `self.day_ahead = ...` line (currently line 45), add:

```python
        # Multi-day episodes: battery charge carries across days and every agent sees yesterday's prices.
        self.episode_days = int(params.get("episode_days", 1))
        if self.episode_days < 1:
            raise ValueError(f"episode_days must be at least 1, got {self.episode_days}.")
        self.multi_day = self.episode_days > 1
        if self.multi_day and not self.day_ahead:
            raise ValueError("episode_days > 1 requires day_ahead=True.")
```

Replace the day-ahead branch of the observer setup (currently lines 85-88):

```python
        if self.day_ahead:
            # Whole demand profile plus own capacity and cost (battery: power, energy, reference price).
            self.obs_dim = self.T + 2
            bess_obs_dim = self.T + 3
            if self.multi_day:
                # Yesterday's T prices for everyone; the battery also sees its charge.
                self.obs_dim += self.T
                bess_obs_dim += self.T + 1
```

Change the `observation_space` construction (currently lines 101-103) so the lower bound allows negative prices:

```python
        self.observation_space = spaces.Box(
            low=-np.inf if self.multi_day else 0,
            high=np.inf,
            shape=(own_obs_dim,),
            dtype=np.float32,
        )
```

In `reset`, after `self.t = 0`, add:

```python
        self.day = 0
```

and at the end of `reset`, directly before `return self._get_obs(), {}`, add:

```python
        # Only the first day after construction has no real yesterday; later resets keep the last day's prices.
        if self.multi_day and self.prev_prices is None:
            self.prev_prices = self._at_cost_prices()
```

In `__init__`, directly before the final `self.reset(seed=seed)` call, add:

```python
        self.prev_prices = None
```

Add this method directly after `reset`:

```python
    def _at_cost_prices(self) -> np.ndarray:
        """Hourly prices if every generator offered full capacity at cost; stands in for yesterday on day one."""
        return np.array(
            [market_clearing(self.c, self.K, demand)[0] for demand in self.D_profile],
            dtype=np.float32,
        )
```

Replace `_get_day_obs` with:

```python
    def _get_day_obs(self, agent_index: int) -> np.ndarray:
        """Day-ahead observation: the whole demand profile plus the agent's own static features."""

        T = self.T
        is_bess = self.has_bess and agent_index == self.bess_index
        if is_bess:
            buf = self.bess_obs_buf
            buf[T] = self.bess_power / self.capacity_scale
            buf[T + 1] = self.bess_energy / (self.capacity_scale * T)
            buf[T + 2] = self.bess_bid_ref / self.cost_scale
            n_static = 3
        else:
            buf = self.obs_buf
            buf[T] = self.K[agent_index] / self.capacity_scale
            buf[T + 1] = self.c[agent_index] / self.cost_scale
            n_static = 2
        buf[:T] = self.D_profile / self.demand_scale
        if self.multi_day:
            start = T + n_static
            buf[start : start + T] = self.prev_prices / self.cost_scale
            if is_bess:
                buf[start + T] = self.soc / self.bess_energy
        return buf
```

`reset` already sets `self.soc` before calling `_get_obs`, and `prev_prices` must be set before `_get_obs` runs, which is why it goes right before the `return`.

In `get_metadata`, add `"episode_days": self.episode_days,` after `"day_ahead": self.day_ahead,`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_environment.py -v`
Expected: all PASS, including every existing test in the file.

- [ ] **Step 6: Commit**

```bash
git add easy_marl/src/environment.py tests/unit_tests/test_environment.py
git commit -m "Add episode_days parameter and multi-day observation layout"
```

---

### Task 2: Multi-day stepping, continuous charge, no penalty, daily demand redraw

**Files:**
- Modify: `easy_marl/src/environment.py` (`__init__` battery block, `reset`, `_step_day`, `_run_hour`)
- Test: `tests/unit_tests/test_environment.py` (add to `TestMARLEnvMultiDay`)

**Interfaces:**
- Consumes: `env.multi_day`, `env.day`, `env.prev_prices`, `env._at_cost_prices()` from Task 1.
- Produces: in multi-day mode, `step` returns `(obs, reward, False, False, {})` for every day except the last and `(None, reward, True, False, {})` on the last. `env.output[key]` has `episode_days * T` rows, where row `day * T + hour` is that hour. `env.prev_prices` holds the previous day's cleared prices after each day. `env.soc` is set once in `__init__` and is never reset in multi-day mode. `env.output["penalty"]` stays all zero in multi-day mode.

- [ ] **Step 1: Write the failing tests**

Add these methods to `TestMARLEnvMultiDay`:

```python
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
```

The bid regulariser is zero in these tests because `PARAMS` sets `lambda_bid_penalty` to 0.0, and no load goes unserved at demand 60, so the battery's scaled reward is exactly price times net energy sold. If the reward check fails because of a loss-of-load term, print `env.output["rewards"][:, BESS_INDEX] - hourly * scale` and find the cause; do not loosen the tolerance.

`run_fixed_episode` calls `env.reset()` and then loops until `terminated`, so it plays all `DAYS` days. Calling it again on the same environment plays the next episode with the charge carried over.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_environment.py::TestMARLEnvMultiDay -v`
Expected: the new tests FAIL. For example, `test_that_the_episode_ends_only_after_the_last_day` fails because day 0 returns `terminated=True`, and the output shape is `(5,)` instead of `(15,)`.

- [ ] **Step 3: Implement**

In `__init__`, in the `if self.has_bess:` block (currently lines 71-78), add after `self.bess_initial_soc = ...`:

```python
            # Multi-day markets set the charge once here and never reset it, so no penalty is needed.
            self.soc = self.bess_initial_soc
```

In `reset`, replace:

```python
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.run_stochastics()
```

with:

```python
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.run_stochastics()
        elif self.multi_day:
            # Single-day mode keeps one draw per seed so finished sweeps replay exactly.
            self.run_stochastics()
```

In `reset`, size every output array for the whole episode. Add `hours = self.T * self.episode_days` before `self.output = {`, then replace each `self.T` in the `np.zeros((self.T, ...))` and `np.zeros(self.T, ...)` calls inside the output dicts (both the main dict and the battery block) with `hours`.

In `reset`'s battery block, replace `self.soc = self.bess_initial_soc` with:

```python
            if not self.multi_day:
                self.soc = self.bess_initial_soc
```

Leave `self.bess_buy_bid = 0.0` and `self.bess_buy_quantity = 0.0` as they are.

Replace the tail of `_step_day` (from `day_reward = np.zeros(self.N)` to the end of the method) with:

```python
        day_reward = np.zeros(self.N)
        for t in range(self.T):
            self.t = t
            for j in range(self.N):
                self.update_agent_bid(schedules[j, t], j)
            day_reward += self._run_hour()

        self.t = self.T
        if self.multi_day:
            start = self.day * self.T
            self.prev_prices = self.output["market_prices"][start : start + self.T].copy()
        self.day += 1
        if self.day < self.episode_days:
            self.run_stochastics()
            return self._get_obs(), day_reward[self.agent_index], False, False, {}
        return None, day_reward[self.agent_index], True, False, {}
```

In `_run_hour`:

1. Leave the demand line as it is: `demand = self.D_profile[self.t]`. Demand is indexed by hour of the current day.
2. Directly after that line add `hour = self.day * self.T + self.t`.
3. Replace the terminal penalty condition `if self.t == self.T - 1:` with:

```python
            if not self.multi_day and self.t == self.T - 1:
```

   and update the comment above it to: `# Stops a single-day battery from selling its starting energy for free; multi-day markets never reset the charge, so they need no penalty.`
4. Index the penalty's output write as `self.output["penalty"][hour, self.bess_index] = terminal_penalty`, replace every other `self.output[...][self.t]` write in the battery block with `[hour]`, and set `t_idx = hour` in the "Store outputs" block (currently `t_idx = self.t`).

The hourly (non-day-ahead) path also calls `_run_hour`. There `self.day` is always 0 and `episode_days` is 1, so `hour == self.t` and nothing changes.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_environment.py -v`
Expected: all PASS, old and new. In particular `test_that_ending_below_initial_soc_is_penalised_at_the_last_step` still passes: single-day markets keep their penalty.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit_tests -q --deselect tests/unit_tests/test_agents.py::TestFixedActionFunction`
Expected: `128 + 15 = 143 passed` (7 tests from Task 1, 8 from Task 2), 6 deselected. If any pre-existing test fails, stop and find the cause. Do not edit the old test.

- [ ] **Step 6: Commit**

```bash
git add easy_marl/src/environment.py tests/unit_tests/test_environment.py
git commit -m "Run multi-day markets continuously: charge and prices carry over, no penalty"
```

---

### Task 3: Parameter builders pass episode_days through

**Files:**
- Modify: `easy_marl/examples/bidding/training.py:136-196` (`make_default_params`, `make_bess_params`)
- Test: `tests/unit_tests/test_bidding/test_training.py`

**Interfaces:**
- Produces: `make_default_params(N=5, T=24, day_ahead=False, episode_days=1)` and `make_bess_params(..., day_ahead=False, episode_days=1)`. Both always put `"episode_days"` in the returned dict.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit_tests/test_bidding/test_training.py`. Check its existing imports first; add `make_bess_params`, `make_default_params`, `init_agents` and `parallel_train` from `easy_marl.examples.bidding.training` if they are not already imported, plus `from functools import partial`.

```python
def test_that_param_builders_pass_episode_days_through():
    assert make_default_params(N=3, day_ahead=True)["episode_days"] == 1
    assert make_default_params(N=3, day_ahead=True, episode_days=7)["episode_days"] == 7
    assert make_bess_params(N=4, day_ahead=True, episode_days=7)["episode_days"] == 7


def test_that_agents_built_for_multi_day_markets_have_the_wider_observations():
    params = make_bess_params(N=4, T=24, day_ahead=True, episode_days=7)
    agents = init_agents(4, params, seed=0)
    assert agents[0].env.observation_space.shape == (50,)
    assert agents[3].env.observation_space.shape == (52,)


def test_that_parallel_training_runs_on_multi_day_episodes(tmp_path):
    agents, info = parallel_train(
        N=4,
        num_rounds=1,
        timesteps_per_agent=48,
        seed=0,
        save_dir=str(tmp_path),
        verbose=False,
        param_func=partial(make_bess_params, day_ahead=True, episode_days=3),
        n_workers=1,
        frozen_agents=(0, 1, 2),
    )
    assert info["frozen_agents"] == [0, 1, 2]
    assert agents[3].env.observation_space.shape == (52,)
```

`PPOAgent` keeps its environment as `agent.env`. Check `easy_marl/src/agents.py` for the exact attribute name; `parallel_train` itself uses `agent.env` at `training.py:795`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_bidding/test_training.py -v -k "episode_days or multi_day"`
Expected: FAIL with `TypeError: make_default_params() got an unexpected keyword argument 'episode_days'`.

- [ ] **Step 3: Implement**

In `make_default_params`, change the signature to `def make_default_params(N: int = 5, T: int = 24, day_ahead: bool = False, episode_days: int = 1) -> Dict:`. Add the docstring line `episode_days: Days per episode; above 1, battery charge and prices carry across days` under `day_ahead`, and add `"episode_days": episode_days,` to the dict after `"day_ahead": day_ahead,`.

In `make_bess_params`, add the parameter `episode_days: int = 1` after `day_ahead: bool = False`, and change the first body line to:

```python
    params = make_default_params(N=N - 1, T=T, day_ahead=day_ahead, episode_days=episode_days)
```

Update the `make_bess_params` docstring's last sentence to: `The battery starts each episode half full and is charged for ending the episode below that, so it can only profit by shifting energy between hours, not by selling its starting charge.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_bidding/test_training.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add easy_marl/examples/bidding/training.py tests/unit_tests/test_bidding/test_training.py
git commit -m "Pass episode_days through the market parameter builders"
```

---

### Task 4: Sweep wiring, day-split metrics, presets and report note

**Files:**
- Modify: `easy_marl/examples/bidding/bess_experiment.py`: `PRESETS` (:51-78), `SweepConfig` (:81-117), `build_configs` (:120-128), `simulate_episodes` (:131-167), `get_pretrained` (:286-327), `write_report` (:765-836), `main`
- Test: `tests/unit_tests/test_bidding/test_bess_experiment.py`

**Interfaces:**
- Consumes: `make_default_params(..., episode_days=...)` and `make_bess_params(..., episode_days=...)` from Task 3, and `env.episode_days` from Task 1.
- Produces:
  - `SweepConfig(power_mw, duration_h, seed, arm="frozen", episode_days=1)`. `config_id` is unchanged by `episode_days`, because each preset writes to its own output directory.
  - `build_configs(powers, durations, seeds, arms=("frozen",), episode_days=1)`.
  - Presets `"smoke_carryover"` and `"full_carryover"`, each with `"episode_days": 7`.
  - `simulate_episodes` returns per-day arrays: `market_prices` has shape `(num_episodes * episode_days, T)`.

- [ ] **Step 1: Write the failing tests**

Read the top of `tests/unit_tests/test_bidding/test_bess_experiment.py` first and reuse its imports and helpers. Merge the imports below into the file's existing import block at the top (skip any already there), and append the class at the end:

```python
from easy_marl.examples.bidding.bess_experiment import (
    PRESETS,
    SweepConfig,
    build_configs,
    simulate_episodes,
)
from easy_marl.examples.bidding.training import init_agents, make_bess_params, make_default_params
from easy_marl.src.agents import PPOAgent
from easy_marl.src.environment import MARLElectricityMarketEnv


class TestMultiDaySweep:
    def test_that_carryover_presets_run_seven_day_episodes(self):
        assert PRESETS["smoke_carryover"]["episode_days"] == 7
        assert PRESETS["full_carryover"]["episode_days"] == 7
        assert "episode_days" not in PRESETS["full"]

    def test_that_configs_build_multi_day_markets_without_changing_their_ids(self):
        config = SweepConfig(50, 4, 42, episode_days=7)
        assert config.config_id == "p50_d4_s42"
        assert config.param_func()(N=4, T=24)["episode_days"] == 7
        baseline = SweepConfig(None, None, 42, episode_days=7)
        assert baseline.param_func()(N=3, T=24)["episode_days"] == 7

    def test_that_build_configs_passes_episode_days_to_every_config(self):
        configs = build_configs([10], [1], [42], episode_days=7)
        assert {c.episode_days for c in configs} == {7}

    def test_that_old_stored_configs_without_episode_days_still_load(self):
        stored = {"power_mw": 10, "duration_h": 1, "seed": 42, "arm": "frozen"}
        assert SweepConfig(**stored).episode_days == 1

    def test_that_multi_day_dispatch_is_split_into_days(self):
        params = make_bess_params(N=4, T=24, day_ahead=True, episode_days=3)
        agents = init_agents(4, params, seed=0)
        dispatch = simulate_episodes(agents, params, num_episodes=2, seed=10_000)
        assert dispatch["market_prices"].shape == (6, 24)
        assert dispatch["q_cleared"].shape == (6, 24, 4)
        assert dispatch["soc"].shape == (6, 24)

    def test_that_the_warm_up_episode_spends_the_free_starting_charge(self):
        class Fixed:
            def __init__(self, action):
                self.action = np.asarray(action, dtype=np.float32)

            def fixed_act_function(self, deterministic=True):
                return lambda obs: self.action

        # A battery that always sells empties its free 50 MWh within the warm-up episode,
        # so nothing it sells can show up in the recorded days.
        params = make_bess_params(N=4, T=24, power_mw=25, duration_h=4, day_ahead=True, episode_days=3)
        agents = [Fixed([0.0, 0.0]) for _ in range(3)] + [Fixed([1.0, -10.0])]
        dispatch = simulate_episodes(agents, params, num_episodes=2, seed=10_000)
        assert dispatch["soc"].max() < 1e-6
        assert dispatch["bess_discharge"].sum() == 0

    def test_that_single_day_dispatch_shape_is_unchanged(self):
        params = make_default_params(N=3, T=24, day_ahead=True)
        agents = init_agents(3, params, seed=0)
        dispatch = simulate_episodes(agents, params, num_episodes=2, seed=10_000)
        assert dispatch["market_prices"].shape == (2, 24)

    def test_that_single_day_generators_cannot_load_into_a_multi_day_market(self):
        single = make_default_params(N=3, T=24, day_ahead=True)
        multi = make_default_params(N=3, T=24, day_ahead=True, episode_days=7)
        saved = init_agents(3, single, seed=0)[0].save_to_bytes()
        env = MARLElectricityMarketEnv(agents=[], params=multi, seed=0, agent_index=0)
        try:
            PPOAgent.from_bytes(saved, env)
        except ValueError as error:
            assert "bservation" in str(error)
        else:
            raise AssertionError("a 26-number policy loaded into a 50-number market")
```

If `PPOAgent.from_bytes` does not raise on a mismatched observation space, do not loosen the test. That means the stale-cache risk is real. Add an explicit check at the top of `PPOAgent.from_bytes` in `easy_marl/src/agents.py` that compares the loaded model's `observation_space.shape` with `env.observation_space.shape` and raises `ValueError(f"Observation space mismatch: saved {saved_shape}, environment {env_shape}.")`. Include that file in this task's commit.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_bidding/test_bess_experiment.py::TestMultiDaySweep -v`
Expected: FAIL. The presets raise `KeyError: 'smoke_carryover'`, `SweepConfig` raises `TypeError` on `episode_days`, and the multi-day shape is `(2, 72)` instead of `(6, 24)`.

- [ ] **Step 3: Implement**

After the `PRESETS = {...}` dict, add:

```python
# Seven-day episodes: battery charge and yesterday's prices carry across days.
PRESETS["smoke_carryover"] = {**PRESETS["smoke"], "episode_days": 7}
PRESETS["full_carryover"] = {**PRESETS["full"], "episode_days": 7}
```

In `SweepConfig`, add the field `episode_days: int = 1` after `arm: str = "frozen"`. In `param_func`, pass it to both builders:

```python
    def param_func(self):
        # Every agent, baseline generators included, commits to a full day at once.
        if self.is_baseline:
            return partial(make_default_params, day_ahead=True, episode_days=self.episode_days)
        return partial(
            make_bess_params,
            power_mw=self.power_mw,
            duration_h=self.duration_h,
            day_ahead=True,
            episode_days=self.episode_days,
        )
```

Change `build_configs` to accept and pass `episode_days`:

```python
def build_configs(powers, durations, seeds, arms=("frozen",), episode_days=1) -> List[SweepConfig]:
    configs = []
    for seed in seeds:
        for arm in arms:
            configs.append(SweepConfig(None, None, seed, arm, episode_days))
            for power in powers:
                for duration in durations:
                    configs.append(SweepConfig(power, duration, seed, arm, episode_days))
    return configs
```

Replace the episode loop at the top of `simulate_episodes` (from `episodes: List[Dict[str, np.ndarray]] = []` up to, not including, `keys = [...]`) with:

```python
    if params.get("episode_days", 1) > 1:
        # One continuous market: charge and prices carry from episode to episode. The unrecorded
        # warm-up episode spends the free starting charge and replaces the at-cost first-day prices.
        env = MARLElectricityMarketEnv(
            agents=agents, params=params, seed=seed, observer_name=observer_name
        )
        _play_episode(env)
        episodes = [_play_episode(env) for _ in range(num_episodes)]
    else:
        episodes = []
        for ep in range(num_episodes):
            env = MARLElectricityMarketEnv(
                agents=agents,
                params=params,
                seed=None if seed is None else seed + ep,
                observer_name=observer_name,
            )
            episodes.append(_play_episode(env))
```

and add this helper directly above `simulate_episodes`:

```python
def _play_episode(env: MARLElectricityMarketEnv) -> Dict[str, np.ndarray]:
    """Reset, play one episode with every agent on its fixed policy, and return its output."""
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(None, fixed_evaluation=True)
        done = terminated or truncated
    return env.output
```

`reset` builds a new `env.output` dict each time, so the returned dicts are never overwritten by later episodes. `params.get("episode_days", 1)` is the same old-params default as everywhere else: parameter dicts from before this change have no key.

Then replace the `dispatch = {...}` comprehension with:

```python
    days = env.episode_days
    dispatch = {}
    for key in keys:
        stacked = np.stack([out[key] for out in episodes]).astype(np.float32)
        # Split multi-day episodes into days so every metric stays per day.
        dispatch[key] = stacked.reshape(num_episodes * days, env.T, *stacked.shape[2:])
```

Update the docstring's second sentence to: `Every array is float32 with the day first: prices and demand are (day, hour), per-agent arrays are (day, hour, agent), multi-day episodes are split into their days, and the battery arrays exist only when the market has a battery. Multi-day markets are simulated as one continuous run after an unrecorded warm-up episode.`

In `get_pretrained`, replace the two `make_default_params` uses:

```python
    # Presets stored before multi-day episodes have no key; they ran single days.
    episode_days = preset.get("episode_days", 1)
    params = make_default_params(
        N=N_GENERATORS, T=HOURS, day_ahead=True, episode_days=episode_days
    )
```

Inside the `parallel_train` call, change `param_func`:

```python
            param_func=partial(make_default_params, day_ahead=True, episode_days=episode_days),
```

In `main`, pass it to `build_configs`:

```python
    configs = build_configs(
        preset["powers"],
        preset["durations"],
        preset["seeds"],
        preset["arms"],
        # Presets stored before multi-day episodes have no key; they ran single days.
        episode_days=preset.get("episode_days", 1),
    )
```

In `write_report`, after the opening `lines = [...]` block and before `if convergence:`, add:

```python
    episode_days = {r.get("episode_days", 1) for r in rows}
    if episode_days != {1}:
        lines += [
            f"Days run continuously in {max(episode_days)}-day episodes. Battery charge is never "
            "reset and carries no penalty, and every agent sees the previous day's prices. Frozen "
            "generators therefore react to the battery's effect on yesterday's prices, so paired "
            "effects include that immediate response. Evaluation discards one warm-up episode. "
            "Metrics are per day.",
            "",
        ]
```

Check that `relative_rows` (`bess_experiment.py:522`) keeps the `episode_days` key on each row. If it builds new dicts with only selected keys, compute `episode_days` from the `rows` argument before the `rows = relative_rows(rows)` line instead.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_bidding/test_bess_experiment.py -v`
Expected: all PASS, including the existing replay tests.

- [ ] **Step 5: Add a report test**

Append to `TestMultiDaySweep`. Use the same fake-row helper that `test_that_aggregate_writes_paired_effects_and_convergence` (around line 442) uses, and adapt it so every row has `"episode_days": 7`:

```python
    def test_that_the_report_says_when_episodes_span_several_days(self, tmp_path):
        # Build fake finished configs exactly as test_that_aggregate_writes_paired_effects_and_convergence
        # does, with "episode_days": 7 in every stored config, then:
        report = aggregate(tmp_path).read_text()
        assert "Days run continuously in 7-day episodes" in report
```

Replace the comment with the real setup code copied from that existing test, adding `"episode_days": 7` to each stored `config` dict. Import `aggregate` from `bess_experiment`. Run: `.venv/bin/python -m pytest tests/unit_tests/test_bidding/test_bess_experiment.py -v`. Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit_tests -q --deselect tests/unit_tests/test_agents.py::TestFixedActionFunction`
Expected: all pass, 6 deselected, no pre-existing test changed.

- [ ] **Step 7: Commit**

```bash
git add easy_marl/examples/bidding/bess_experiment.py tests/unit_tests/test_bidding/test_bess_experiment.py
git commit -m "Wire multi-day episodes through the BESS sweep and report"
```

---

### Task 5: Comment and notes cleanup

**Files:**
- Modify: `easy_marl/examples/bidding/training.py:90-91` (stale `n_steps` comment)
- Modify: `research/bess_followup/NOTES.md`

- [ ] **Step 1: Fix the stale PPO comment**

The `n_steps` and `batch_size` comments at `training.py:90-91` say "Match episode length" and "deterministic over episodes". Neither is true for day-ahead runs, where one step is one day. Replace both comments with:

```python
    "n_steps": 24,  # Rollout length in env steps; day-ahead steps are whole days. 256 collapsed a battery to idle in probes.
    "batch_size": 24,
```

Do not change the values.

- [ ] **Step 2: Update NOTES.md**

In `research/bess_followup/NOTES.md`, replace open decision 3 with:

```markdown
3. SoC carry-over: implemented on branch `feat/multi-day-episodes` as `params["episode_days"]` (presets `smoke_carryover`, `full_carryover`, 7 days). Agents see the previous day's prices; the battery also sees its charge; the charge is never reset, so there is no penalty; evaluation discards one warm-up episode; demand noise is redrawn daily. The full sweep has not been run.
```

Add under "Environment notes":

```markdown
- In single-day mode, demand noise is drawn once per environment seed, so every day inside one training call has identical demand. Multi-day mode redraws it daily. Single-day mode was left as is so `results/bess_full` still replays.
```

- [ ] **Step 3: Commit**

```bash
git add easy_marl/examples/bidding/training.py research/bess_followup/NOTES.md
git commit -m "Update PPO rollout comment and BESS notes for multi-day episodes"
```

---

### Task 6: Verify old results still replay, then run the probe and stop

This task produces evidence, not code. Do not commit outputs (`outputs/` is gitignored).

- [ ] **Step 1: Check one stored single-day config still reproduces**

The finished sweep lives in `outputs/bess_sweep/full/`. Run this from the repo root:

```bash
.venv/bin/python - <<'EOF'
import json
from pathlib import Path
from easy_marl.examples.bidding.bess_experiment import (
    HOURS, SweepConfig, _metrics_match, _saved_agents, simulate_episodes, summarise_market,
)
out = Path("outputs/bess_sweep/full")
stored = json.loads((out / "p50_d4_s42" / "metrics.json").read_text())
config = SweepConfig(**stored["config"])
params = config.param_func()(N=config.n_agents, T=HOURS)
agents = _saved_agents(config, stored["preset"], out, params)
dispatch = simulate_episodes(agents, params, stored["preset"]["eval_episodes"], seed=10_000)
print("reproduces:", _metrics_match(summarise_market(dispatch, params), stored["metrics"]))
EOF
```

Expected: `reproduces: True`. If it prints `False`, or the directory is missing, stop and report the exact output to the user. Do not continue.

- [ ] **Step 2: Run the probe**

The probe is one seed, generators pretrained with 7-day episodes, plus one 50 MW / 4 h battery in the frozen arm:

```bash
.venv/bin/python -m easy_marl.examples.bidding.bess_experiment run \
  --preset smoke_carryover --seeds 42 --powers 50 --durations 4 \
  --out outputs/bess_sweep/probe_carryover
```

Run it in the background and note its wall-clock time.

- [ ] **Step 3: Check the probe gates from the saved dispatch**

```bash
.venv/bin/python - <<'EOF'
import json
import numpy as np
from pathlib import Path
out = Path("outputs/bess_sweep/probe_carryover")
conv = json.loads((out / "pretrain" / "s42" / "convergence.json").read_text())
d = np.load(out / "p50_d4_s42" / "dispatch.npz")
m = json.loads((out / "p50_d4_s42" / "metrics.json").read_text())["metrics"]
soc = d["soc"].reshape(-1, 7, 24)          # episode, day, hour
day_end = soc[:, :, -1] / 200.0            # 50 MW x 4 h = 200 MWh
print("pretrain last change:", conv["policy_change_per_round"][-1], "converged:", conv["converged"])
print("charge MWh/day:", m["bess_charge_mwh_mean"], "discharge MWh/day:", m["bess_discharge_mwh_mean"])
print("full-power hours/day:", m["bess_discharge_mwh_mean"] / 50.0)
print("buy price < sell price:", m["bess_mean_charge_price"], "<", m["bess_mean_discharge_price"])
print("end-of-day charge fraction, first episode:", np.round(day_end[0], 2))
print("end-of-day charge spread across days:", float(day_end.std()))
EOF
```

Pass gates:
- Pretraining `converged` is `True`.
- The battery charges more than 0 MWh per day.
- Mean buy price is below mean sell price.
- The end-of-day charge is not pinned at 0.5 on every day (spread above 0.02). That shows the battery uses carry-over.

For comparison, the old single-day 50 MW / 4 h configs showed 1.9 full-power hours per day (`results/bess_full/report.md`: 0.48 cycles × 4 h).

- [ ] **Step 4: Stop and report**

Report to the user:
- Test counts.
- The replay check result.
- Probe wall-clock time.
- Every printed probe number.
- Which gates passed.

If any gate fails, say which one and show the numbers; do not tune anything. Then ask whether to run `--preset full_carryover`. Give a time estimate scaled from the probe: the probe ran 1 pretrain plus 1 battery config, and the full preset is 10 pretrains plus 210 configs.

---

## Self-review notes

- Spec coverage:
  - Previous-day prices: Tasks 1 and 2.
  - Charge carry-over and charge level in the observation: Tasks 1 and 2.
  - No charge penalty in multi-day markets: Task 2.
  - Daily demand redraw: Task 2.
  - Per-day metrics: Task 4.
  - Presets: Task 4.
  - Report wording on what the frozen arm now measures: Task 4.
  - Old results unchanged: Global Constraints, Task 4 tests, Task 6 step 1.
  - Full sweep: out of scope by design.
- Types:
  - `episode_days` is an `int` everywhere.
  - `prev_prices` is a float32 array of shape `(T,)`.
  - Observation sizes are 50 and 52 for T=24, consistent across Tasks 1, 3 and 4.
- Known limitation: PPO's rollout of 24 steps covers 3.4 seven-day episodes, so rollouts cut episodes mid-way. Stable-baselines3 bootstraps the cut with the value estimate, which is standard. Changing `n_steps` is out of scope.
