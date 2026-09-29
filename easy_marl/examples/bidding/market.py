"""
Helper functions for RL electricity market environment.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit(cache=True)
def market_clearing(bids: np.ndarray, quantities: np.ndarray, demand: float):
    """
    Compute market clearing price and accepted quantities.

    Args:
        bids (np.ndarray): Array of bid prices, shape (N,)
        quantities (np.ndarray): Array of offered quantities, shape (N,)
        demand (float): Market demand

    Returns:
        P_t (float): Clearing price
        q_cleared (np.ndarray): Accepted quantities, shape (N,)
    """
    bids = np.asarray(bids)
    quantities = np.asarray(quantities)
    N = len(bids)

    order = np.argsort(bids)
    bids_sorted = bids[order]
    q_sorted = quantities[order]

    cum_supply = np.cumsum(q_sorted)
    # Find the first index where cumulative supply meets/exceeds demand
    m = np.searchsorted(cum_supply, demand, side="left")

    q_cleared = np.zeros_like(q_sorted)
    if m >= N:  # demand exceeds total supply
        q_cleared[:] = q_sorted
        P_t = bids_sorted[-1]
    else:
        q_cleared[:m] = q_sorted[:m]
        q_cleared[m] = demand - cum_supply[m - 1] if m > 0 else demand
        P_t = bids_sorted[m]

    # Reorder q_cleared to original order
    q_cleared_final = np.zeros_like(q_cleared)
    q_cleared_final[order] = q_cleared

    return P_t, q_cleared_final


@njit(cache=True)
def clear_with_storage(
    bids: np.ndarray,
    quantities: np.ndarray,
    demand: float,
    buy_bid: float,
    buy_quantity: float,
):
    """
    Clear the market with one price-sensitive storage buyer on top of inelastic demand.

    The buyer takes the most energy it can without the clearing price rising above
    its bid: the supply offered at or below ``buy_bid`` left over after inelastic demand.

    Returns:
        P_t (float): Clearing price
        q_cleared (np.ndarray): Accepted sell quantities, shape (N,)
        charged (float): Energy bought by the storage buyer
    """
    supply_at_or_below_bid = 0.0
    for i in range(len(bids)):
        if bids[i] <= buy_bid:
            supply_at_or_below_bid += quantities[i]
    charged = min(max(supply_at_or_below_bid - demand, 0.0), buy_quantity)
    P_t, q_cleared = market_clearing(bids, quantities, demand + charged)
    return P_t, q_cleared, charged


@njit(cache=True)
def linear_sf_market_clearing(bids: np.ndarray, quantities: np.ndarray, demand: float):
    """Compute market clearing using supply functions"""
    pass
