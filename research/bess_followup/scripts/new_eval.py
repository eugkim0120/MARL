def simulate_episodes(
    agents,
    params: Dict,
    num_episodes: int,
    seed: Optional[int],
    observer_name: str = DEFAULT_OBS,
) -> Dict[str, np.ndarray]:
    """Run frozen agents over seeded demand episodes and record the full dispatch.

    Every array is float32 with the episode first: prices and demand are (episode, hour),
    per-agent arrays are (episode, hour, agent), and the battery arrays exist only when
    the market has a battery.
    """
    episodes: List[Dict[str, np.ndarray]] = []
    for ep in range(num_episodes):
        env = MARLElectricityMarketEnv(
            agents=agents,
            params=params,
            seed=None if seed is None else seed + ep,
            observer_name=observer_name,
        )
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(None, fixed_evaluation=True)
            done = terminated or truncated
        episodes.append(env.output)

    keys = ["market_prices", "demand", "bids", "q_offered", "q_cleared", "rewards"]
    if env.has_bess:
        keys += ["bess_charge", "bess_discharge", "soc"]
    dispatch = {
        key: np.stack([out[key] for out in episodes]).astype(np.float32) for key in keys
    }
    dispatch["generator_cost"] = np.asarray(env.c, dtype=np.float32)
    dispatch["generator_capacity"] = np.asarray(env.K, dtype=np.float32)
    return dispatch


def summarise_market(dispatch: Dict[str, np.ndarray], params: Dict) -> Dict:
    """Market outcome metrics from a recorded dispatch."""
    prices = dispatch["market_prices"].astype(np.float64)
    demands = dispatch["demand"].astype(np.float64)
    cost = dispatch["generator_cost"].astype(np.float64)
    q_gen = dispatch["q_cleared"][:, :, : len(cost)].astype(np.float64)
    gen_profits = ((prices[:, :, None] - cost[None, None, :]) * q_gen).sum(axis=1)
    has_bess = "bess" in params
    if has_bess:
        charges = dispatch["bess_charge"].astype(np.float64)
        discharges = dispatch["bess_discharge"].astype(np.float64)
    else:
        charges = discharges = np.zeros_like(prices)
    served = q_gen.sum(axis=2) + discharges - charges
    loss_of_load = np.maximum(demands - served, 0.0).sum(axis=1)

    metrics = {
        "mean_price": float(prices.mean()),
        "price_std": float(prices.std()),
        "intraday_std_mean": float(prices.std(axis=1).mean()),
        "daily_spread_mean": float((prices.max(axis=1) - prices.min(axis=1)).mean()),
        "price_p5": float(np.percentile(prices, 5)),
        "price_p95": float(np.percentile(prices, 95)),
        "price_max": float(prices.max()),
        "mean_hourly_price": prices.mean(axis=0).tolist(),
        "consumer_cost_mean": float((prices * demands).sum(axis=1).mean()),
        "loss_of_load_mwh_mean": float(loss_of_load.mean()),
        "generator_profit_mean": gen_profits.mean(axis=0).tolist(),
        "generator_profit_total_mean": float(gen_profits.sum(axis=1).mean()),
        "bess_profit_mean": None,
        "bess_charge_mwh_mean": None,
        "bess_discharge_mwh_mean": None,
        "bess_equivalent_cycles_mean": None,
        "bess_mean_charge_price": None,
        "bess_mean_discharge_price": None,
    }

    if has_bess:
        energy = params["bess"]["power_mw"] * params["bess"]["duration_h"]
        charged_total = charges.sum()
        discharged_total = discharges.sum()
        metrics.update(
            {
                "bess_profit_mean": float(
                    (prices * (discharges - charges)).sum(axis=1).mean()
                ),
                "bess_charge_mwh_mean": float(charges.sum(axis=1).mean()),
                "bess_discharge_mwh_mean": float(discharges.sum(axis=1).mean()),
                "bess_equivalent_cycles_mean": float(
                    discharges.sum(axis=1).mean() / energy
                ),
                "bess_mean_charge_price": (
                    float((prices * charges).sum() / charged_total)
                    if charged_total > 0
                    else None
                ),
                "bess_mean_discharge_price": (
                    float((prices * discharges).sum() / discharged_total)
                    if discharged_total > 0
                    else None
                ),
            }
        )
    return metrics


def evaluate_market_metrics(
    agents,
    params: Dict,
    num_episodes: int,
    seed: Optional[int],
    observer_name: str = DEFAULT_OBS,
) -> Dict:
    """Run frozen agents over seeded demand episodes and summarise market outcomes."""
    dispatch = simulate_episodes(agents, params, num_episodes, seed, observer_name)
    return summarise_market(dispatch, params)


