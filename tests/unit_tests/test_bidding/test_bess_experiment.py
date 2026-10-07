"""
Unit tests for the battery (BESS) experiment pipeline.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from easy_marl.src.agents import SimpleAgent
from easy_marl.src.environment import MARLElectricityMarketEnv
from easy_marl.examples.bidding import bess_experiment
from easy_marl.examples.bidding.bess_experiment import (
    CSV_METRICS,
    SweepConfig,
    aggregate,
    bootstrap_ci,
    build_configs,
    evaluate_market_metrics,
    get_pretrained,
    paired_effects,
    replay_dispatch,
    run_config,
    run_sweep,
    simulate_episodes,
)
from easy_marl.examples.bidding.training import (
    DEFAULT_OBS,
    evaluate_agents,
    init_agents,
    make_bess_params,
    make_default_params,
)

T = 4
POWER = 20.0
DURATION = 2.0
BESS_INDEX = 3

PARAMS = {
    "N_generators": 3,
    "T": T,
    "demand_profile": [60.0] * T,
    "capacities": [50.0, 50.0, 50.0],
    "costs": [10.0, 20.0, 30.0],
    "max_bid_delta": 10.0,
    "lambda_bid_penalty": 0.0,
    "bess": {
        "power_mw": POWER,
        "duration_h": DURATION,
        "efficiency_rt": 0.81,
        "initial_soc_frac": 0.5,
        "bid_ref": 25.0,
    },
}


class FixedAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def fixed_act_function(self, deterministic=True):
        return lambda obs: self.action


def fixed_agents(bess_action):
    return [SimpleAgent() for _ in range(3)] + [FixedAgent(bess_action)]


class TestMakeBessParams:
    def test_that_generators_and_demand_match_the_no_battery_baseline(self):
        baseline = make_default_params(N=3, T=24)
        params = make_bess_params(N=4, T=24, power_mw=POWER, duration_h=DURATION)

        for key in ("N_generators", "demand_profile", "capacities", "costs"):
            assert params[key] == baseline[key]
        assert params["bess"]["power_mw"] == POWER
        assert params["bess"]["duration_h"] == DURATION

    def test_that_default_battery_starts_half_full_for_cyclic_days(self):
        params = make_bess_params(N=4, T=24, power_mw=POWER, duration_h=DURATION)
        assert params["bess"]["initial_soc_frac"] == 0.5

    def test_that_sweep_configs_run_every_agent_day_ahead(self):
        baseline = SweepConfig(None, None, 42).param_func()(N=3, T=24)
        with_battery = SweepConfig(25, 4, 42).param_func()(N=4, T=24)

        assert baseline["day_ahead"] is True
        assert with_battery["day_ahead"] is True
        assert with_battery["bess"]["power_mw"] == 25


class TestEvaluateWithBattery:
    def test_that_battery_profit_is_price_times_net_energy_sold(self):
        results = evaluate_agents(fixed_agents([1.0, -10.0]), PARAMS)

        # Hour 0: offer at 15 is marginal after the 10 generator, sells 10 at 15.
        # Hour 1: remaining 8 MWh deliverable sells at 20 (20 generator marginal).
        assert results["mean_quantities"][BESS_INDEX] == pytest.approx(18.0, rel=1e-5)
        assert results["mean_profits"][BESS_INDEX] == pytest.approx(
            15.0 * 10.0 + 20.0 * 8.0, rel=1e-5
        )


class TestEvaluateMarketMetrics:
    def test_that_constant_prices_give_zero_volatility_and_known_costs(self):
        metrics = evaluate_market_metrics(
            [SimpleAgent() for _ in range(3)],
            {k: v for k, v in PARAMS.items() if k != "bess"},
            num_episodes=2,
            seed=None,
        )
        # Demand 60 is met by the 10 generator (50) and 10 from the 20 generator.
        assert metrics["mean_price"] == pytest.approx(20.0)
        assert metrics["price_std"] == pytest.approx(0.0)
        assert metrics["daily_spread_mean"] == pytest.approx(0.0)
        assert metrics["consumer_cost_mean"] == pytest.approx(20.0 * 60.0 * T)
        assert metrics["loss_of_load_mwh_mean"] == pytest.approx(0.0)
        assert metrics["generator_profit_total_mean"] == pytest.approx(10.0 * 50.0 * T)
        assert metrics["bess_profit_mean"] is None

    def test_that_charging_battery_reports_cycles_and_purchase_price(self):
        metrics = evaluate_market_metrics(
            fixed_agents([-1.0, 10.0]), PARAMS, num_episodes=1, seed=None
        )
        # Charges 20 MW in hour 0 (store 20 -> 38), then 2/0.9 more in hour 1 (full).
        charged = 20.0 + (40.0 - 38.0) / 0.9
        assert metrics["bess_charge_mwh_mean"] == pytest.approx(charged, rel=1e-5)
        assert metrics["bess_discharge_mwh_mean"] == pytest.approx(0.0)
        assert metrics["bess_equivalent_cycles_mean"] == pytest.approx(0.0)
        assert metrics["bess_mean_charge_price"] == pytest.approx(20.0)
        assert metrics["bess_profit_mean"] == pytest.approx(-20.0 * charged, rel=1e-5)


class TestSweep:
    def test_that_configs_are_baseline_plus_power_by_duration_grid_per_seed(self):
        configs = build_configs(powers=[10, 25], durations=[1, 4], seeds=[42, 43])

        assert len(configs) == (1 + 2 * 2) * 2
        assert SweepConfig(None, None, 42) in configs
        assert SweepConfig(25, 4, 43) in configs
        assert len({c.config_id for c in configs}) == len(configs)

    def test_that_finished_configs_are_skipped_on_resume(self, tmp_path, monkeypatch):
        configs = build_configs(powers=[10], durations=[1], seeds=[42])
        for config in configs:
            config_dir = tmp_path / config.config_id
            config_dir.mkdir()
            (config_dir / "metrics.json").write_text(json.dumps({"done": True}))

        def fail_if_called(*args, **kwargs):
            raise AssertionError("training ran for a finished config")

        monkeypatch.setattr(bess_experiment, "parallel_train", fail_if_called)
        run_sweep(configs, bess_experiment.PRESETS["smoke"], tmp_path)

    @pytest.mark.parametrize("preset_name", sorted(bess_experiment.PRESETS))
    def test_that_the_battery_trains_every_round_against_frozen_generators(
        self, preset_name, tmp_path, monkeypatch
    ):
        preset = bess_experiment.PRESETS[preset_name]
        assert preset["update_probability"] == 1.0

        captured = {}

        class StopAfterCapture(Exception):
            pass

        def capture(**kwargs):
            captured.update(kwargs)
            raise StopAfterCapture

        generators = init_agents(3, make_default_params(N=3, day_ahead=True), seed=42)
        monkeypatch.setattr(bess_experiment, "get_pretrained", lambda *a, **k: generators)
        monkeypatch.setattr(bess_experiment, "parallel_train", capture)
        with pytest.raises(StopAfterCapture):
            run_config(SweepConfig(10, 1, 42), preset, tmp_path)
        assert captured["update_probability"] == 1.0
        assert set(captured["frozen_agents"]) == {0, 1, 2}
        assert len(captured["initial_agents"]) == 4

    @pytest.mark.parametrize("preset_name", sorted(bess_experiment.PRESETS))
    def test_that_presets_pretrain_for_at_least_two_rounds_to_measure_convergence(
        self, preset_name
    ):
        assert bess_experiment.PRESETS[preset_name]["pretrain_rounds"] >= 2


TINY_PRESET = {
    "powers": [10],
    "durations": [1],
    "seeds": [42],
    "arms": ["frozen"],
    "pretrain_rounds": 2,
    "pretrain_timesteps_per_agent": 48,
    "pretrain_change_tol": 0.05,
    "num_rounds": 1,
    "timesteps_per_agent": 48,
    "eval_episodes": 2,
    "update_probability": 1.0,
}


class TestSweepConfigArms:
    def test_that_adaptive_arm_configs_get_their_own_ids(self):
        assert SweepConfig(25, 4, 42).config_id == "p25_d4_s42"
        assert SweepConfig(25, 4, 42, "adaptive").config_id == "p25_d4_s42_adaptive"
        assert SweepConfig(None, None, 42, "adaptive").config_id == "baseline_s42_adaptive"

    def test_that_an_unknown_arm_is_rejected(self):
        with pytest.raises(ValueError, match="arm"):
            SweepConfig(25, 4, 42, "sideways")

    def test_that_each_arm_gets_its_own_baseline(self):
        configs = build_configs(
            powers=[10], durations=[1], seeds=[42], arms=("frozen", "adaptive")
        )

        assert len(configs) == 4
        assert SweepConfig(None, None, 42, "adaptive") in configs
        assert SweepConfig(10, 1, 42, "adaptive") in configs


class TestPretrainedGenerators:
    def test_that_generators_are_trained_once_and_reused(self, tmp_path, monkeypatch):
        calls = []
        real_train = bess_experiment.parallel_train

        def spy(**kwargs):
            calls.append(kwargs)
            return real_train(**kwargs)

        monkeypatch.setattr(bess_experiment, "parallel_train", spy)
        first = get_pretrained(42, TINY_PRESET, tmp_path)
        second = get_pretrained(42, TINY_PRESET, tmp_path)

        assert len(calls) == 1
        assert calls[0]["N"] == 3
        assert calls[0]["update_probability"] == 1.0
        assert calls[0]["num_rounds"] == TINY_PRESET["pretrain_rounds"]
        assert len(first) == len(second) == 3
        obs = np.zeros(first[0].model.observation_space.shape, dtype=np.float32)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.act(obs), b.act(obs))

    def test_that_convergence_is_recorded_as_policy_change_between_rounds(self, tmp_path):
        get_pretrained(42, TINY_PRESET, tmp_path)

        info = json.loads((tmp_path / "pretrain" / "s42" / "convergence.json").read_text())
        assert len(info["policy_change_per_round"]) == TINY_PRESET["pretrain_rounds"] - 1
        assert info["policy_change_per_round"][-1] >= 0.0
        assert info["converged"] in (True, False)

    def test_that_a_single_pretraining_round_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="pretrain_rounds"):
            get_pretrained(42, {**TINY_PRESET, "pretrain_rounds": 1}, tmp_path)


class TestPretrainThenAddBattery:
    def test_that_generators_bid_identically_with_and_without_the_battery(self, tmp_path):
        generators = get_pretrained(42, TINY_PRESET, tmp_path)
        params3 = SweepConfig(None, None, 42).param_func()(N=3, T=24)
        params4 = SweepConfig(10, 1, 42).param_func()(N=4, T=24)
        battery = init_agents(4, params4, seed=42)[-1]

        bids = {}
        for name, agents, params in (
            ("without", generators, params3),
            ("with", generators + [battery], params4),
        ):
            env = MARLElectricityMarketEnv(
                agents=agents, params=params, seed=7, observer_name=DEFAULT_OBS
            )
            env.reset()
            env.step(None, fixed_evaluation=True)
            bids[name] = env.output["bids"][:, :3].copy()

        np.testing.assert_array_equal(bids["without"], bids["with"])

    def test_that_the_frozen_arm_trains_only_the_battery(self, tmp_path):
        run_config(SweepConfig(10, 1, 42), TINY_PRESET, tmp_path)

        round_dir = tmp_path / "p10_d1_s42" / "training" / "round_1"
        assert (round_dir / "agent_3.zip").exists()
        for i in range(3):
            assert not (round_dir / f"agent_{i}.zip").exists()
        assert (tmp_path / "p10_d1_s42" / "metrics.json").exists()

    def test_that_the_frozen_baseline_evaluates_the_pretrained_generators_without_training(
        self, tmp_path, monkeypatch
    ):
        get_pretrained(42, TINY_PRESET, tmp_path)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("the frozen baseline must not train")

        monkeypatch.setattr(bess_experiment, "parallel_train", fail_if_called)
        result = run_config(SweepConfig(None, None, 42), TINY_PRESET, tmp_path)

        assert result["metrics"]["bess_profit_mean"] is None

    def test_that_the_adaptive_arm_keeps_training_every_agent_with_and_without_the_battery(
        self, tmp_path
    ):
        run_config(SweepConfig(10, 1, 42, "adaptive"), TINY_PRESET, tmp_path)
        run_config(SweepConfig(None, None, 42, "adaptive"), TINY_PRESET, tmp_path)

        with_battery = tmp_path / "p10_d1_s42_adaptive" / "training" / "round_1"
        without_battery = tmp_path / "baseline_s42_adaptive" / "training" / "round_1"
        for i in range(4):
            assert (with_battery / f"agent_{i}.zip").exists()
        for i in range(3):
            assert (without_battery / f"agent_{i}.zip").exists()
        assert not (without_battery / "agent_3.zip").exists()


def make_row(power, duration, seed, arm="frozen", **overrides):
    row = {
        "config_id": f"p{power}_d{duration}_s{seed}_{arm}",
        "power_mw": power,
        "duration_h": duration,
        "seed": seed,
        "arm": arm,
    }
    row.update({key: 0.0 for key in bess_experiment.CSV_METRICS})
    row["bess_equivalent_cycles_mean"] = None if power is None else 1.0
    row.update(overrides)
    return row


class TestPairedEffects:
    def test_that_effects_are_the_per_seed_difference_from_the_same_seed_baseline(self):
        rows = [
            make_row(None, None, 1, mean_price=100.0),
            make_row(None, None, 2, mean_price=50.0),
            make_row(10, 1, 1, mean_price=90.0),
            make_row(10, 1, 2, mean_price=41.0),
        ]

        effect = next(
            e for e in paired_effects(rows) if e["metric"] == "mean_price"
        )

        assert effect["n"] == 2
        assert effect["mean_delta"] == pytest.approx(-9.5)
        assert (effect["power_mw"], effect["duration_h"], effect["arm"]) == (10, 1, "frozen")

    def test_that_a_consistent_shift_is_significant_and_scatter_around_zero_is_not(self):
        seeds = range(10)
        rows = [make_row(None, None, s, mean_price=70.0, price_std=10.0) for s in seeds]
        rows += [
            make_row(
                10, 1, s,
                mean_price=65.0 + 0.1 * s,
                price_std=10.0 + (3.0 if s % 2 else -3.0),
            )
            for s in seeds
        ]

        by_metric = {e["metric"]: e for e in paired_effects(rows)}

        assert by_metric["mean_price"]["significant"] is True
        assert by_metric["mean_price"]["ci_high"] < 0
        assert by_metric["price_std"]["significant"] is False
        assert by_metric["price_std"]["ci_low"] < 0 < by_metric["price_std"]["ci_high"]

    def test_that_batteries_that_barely_cycle_can_be_excluded(self):
        rows = [make_row(None, None, s, mean_price=70.0) for s in (1, 2, 3)]
        rows += [
            make_row(10, 1, 1, mean_price=60.0, bess_equivalent_cycles_mean=1.0),
            make_row(10, 1, 2, mean_price=60.0, bess_equivalent_cycles_mean=1.0),
            make_row(10, 1, 3, mean_price=70.0, bess_equivalent_cycles_mean=0.47),
        ]

        def mean_price_effect(**kwargs):
            return next(e for e in paired_effects(rows, **kwargs) if e["metric"] == "mean_price")

        assert mean_price_effect()["n"] == 3
        assert mean_price_effect(exclude_flagged=True)["n"] == 2
        assert mean_price_effect(exclude_flagged=True)["mean_delta"] == pytest.approx(-10.0)

    def test_that_a_battery_row_without_a_baseline_for_its_seed_and_arm_is_an_error(self):
        rows = [make_row(None, None, 1), make_row(10, 1, 2)]

        with pytest.raises(ValueError, match="baseline"):
            paired_effects(rows)

    def test_that_bootstrap_interval_is_deterministic_and_contains_the_mean(self):
        values = [1.0, 2.0, 4.0, 3.0, 5.0]

        low, high = bootstrap_ci(values)

        assert (low, high) == bootstrap_ci(values)
        assert low < np.mean(values) < high


class TestAggregate:
    def test_that_aggregate_writes_paired_effects_and_convergence(self, tmp_path):
        for seed in (1, 2):
            pre = tmp_path / "pretrain" / f"s{seed}"
            pre.mkdir(parents=True)
            (pre / "convergence.json").write_text(
                json.dumps({"policy_change_per_round": [0.2, 0.01], "converged": True})
            )
            for power, duration in ((None, None), (10, 1)):
                cfg = SweepConfig(power, duration, seed)
                metrics = {k: 1.0 for k in CSV_METRICS}
                if power is None:
                    for k in CSV_METRICS:
                        if k.startswith("bess_"):
                            metrics[k] = None
                (tmp_path / cfg.config_id).mkdir()
                (tmp_path / cfg.config_id / "metrics.json").write_text(
                    json.dumps({"config": asdict(cfg), "metrics": metrics})
                )

        report = aggregate(tmp_path)

        assert (tmp_path / "paired_effects.csv").exists()
        text = Path(report).read_text()
        assert "Pretraining" in text
        assert "Paired effects" in text


class TestDispatchRecording:
    def test_that_simulation_records_every_agent_per_hour_in_float32(self):
        dispatch = simulate_episodes(
            fixed_agents([1.0, 10.0]), PARAMS, num_episodes=3, seed=5
        )

        n = 4
        for key in ("bids", "q_offered", "q_cleared", "rewards"):
            assert dispatch[key].shape == (3, T, n)
        for key in ("market_prices", "demand", "bess_charge", "bess_discharge", "soc"):
            assert dispatch[key].shape == (3, T)
        for key, value in dispatch.items():
            assert value.dtype == np.float32, key
        np.testing.assert_array_equal(dispatch["generator_cost"], PARAMS["costs"])
        assert dispatch["generator_capacity"].shape == (3,)

    def test_that_a_market_without_a_battery_has_no_battery_arrays(self):
        dispatch = simulate_episodes(
            [SimpleAgent() for _ in range(3)],
            {k: v for k, v in PARAMS.items() if k != "bess"},
            num_episodes=2,
            seed=None,
        )

        assert dispatch["bids"].shape == (2, T, 3)
        assert not {"bess_charge", "bess_discharge", "soc"} & set(dispatch)

    def test_that_run_config_saves_the_dispatch_next_to_the_metrics(self, tmp_path):
        run_config(SweepConfig(10, 1, 42), TINY_PRESET, tmp_path)

        with np.load(tmp_path / "p10_d1_s42" / "dispatch.npz") as saved:
            assert saved["bids"].shape == (TINY_PRESET["eval_episodes"], 24, 4)
            assert saved["bess_discharge"].shape == (TINY_PRESET["eval_episodes"], 24)

    def test_that_the_baseline_saves_dispatch_for_the_generators_only(self, tmp_path):
        run_config(SweepConfig(None, None, 42), TINY_PRESET, tmp_path)

        with np.load(tmp_path / "baseline_s42" / "dispatch.npz") as saved:
            assert saved["q_cleared"].shape == (TINY_PRESET["eval_episodes"], 24, 3)
            assert "soc" not in saved.files


class TestReplayDispatch:
    @pytest.mark.parametrize("arm", ["frozen", "adaptive"])
    def test_that_replay_restores_the_dispatch_a_run_would_have_saved(self, tmp_path, arm):
        configs = [SweepConfig(None, None, 42, arm), SweepConfig(10, 1, 42, arm)]
        run_sweep(configs, TINY_PRESET, tmp_path)
        original = {}
        for config in configs:
            path = tmp_path / config.config_id / "dispatch.npz"
            with np.load(path) as saved:
                original[config.config_id] = {k: saved[k] for k in saved.files}
            path.unlink()

        replayed = replay_dispatch(tmp_path)

        assert sorted(replayed) == sorted(original)
        for config_id, arrays in original.items():
            with np.load(tmp_path / config_id / "dispatch.npz") as saved:
                assert sorted(saved.files) == sorted(arrays)
                for key, value in arrays.items():
                    np.testing.assert_array_equal(saved[key], value, err_msg=key)

    def test_that_replay_skips_configs_that_already_have_dispatch(self, tmp_path):
        run_config(SweepConfig(10, 1, 42), TINY_PRESET, tmp_path)

        assert replay_dispatch(tmp_path) == []

    def test_that_replay_refuses_metrics_it_cannot_reproduce(self, tmp_path):
        run_config(SweepConfig(10, 1, 42), TINY_PRESET, tmp_path)
        config_dir = tmp_path / "p10_d1_s42"
        (config_dir / "dispatch.npz").unlink()
        result = json.loads((config_dir / "metrics.json").read_text())
        result["metrics"]["mean_price"] += 1.0
        (config_dir / "metrics.json").write_text(json.dumps(result))

        with pytest.raises(ValueError, match="reproduce"):
            replay_dispatch(tmp_path)
        assert not (config_dir / "dispatch.npz").exists()

    def test_that_replay_fails_loudly_when_saved_agents_are_missing(self, tmp_path):
        run_config(SweepConfig(10, 1, 42), TINY_PRESET, tmp_path)
        config_dir = tmp_path / "p10_d1_s42"
        (config_dir / "dispatch.npz").unlink()
        (config_dir / "training" / "round_1" / "agent_3.zip").unlink()

        with pytest.raises(FileNotFoundError, match="agent_3"):
            replay_dispatch(tmp_path)
