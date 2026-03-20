"""
Open-only long/short time-series momentum strategy for the MIG competition.

Chosen parameters from strategy2.ipynb:
- lookback = 120 trading days
- volatility window = 40 trading days
- max shares =400

Signal:
- long  when lagged 120-day cumulative log return > 0
- short when lagged 120-day cumulative log return < 0

Sizing:
- inverse-volatility scaled
- normalized by the cross-sectional median inverse volatility each day
- capped at 100 shares per ticker
"""

import numpy as np


LOOKBACK = 120
VOL_WINDOW = 40
MAX_SHARES = 100
VOL_FLOOR = 1e-6


def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Convert open-price history into integer trade actions.

    Parameters
    ----------
    prices : np.ndarray
        Open prices with shape (num_stocks, num_days).

    Returns
    -------
    np.ndarray
        Trade actions with shape (num_stocks, num_days).
    """
    num_stocks, num_days = prices.shape
    positions = np.zeros((num_stocks, num_days), dtype=int)

    log_prices = np.log(prices.astype(float))
    returns = np.empty((num_stocks, num_days), dtype=float)
    returns[:, 0] = np.nan
    returns[:, 1:] = np.diff(log_prices, axis=1)

    start_day = max(LOOKBACK, VOL_WINDOW) + 1

    for day_idx in range(start_day, num_days):
        trend = np.sum(returns[:, day_idx - LOOKBACK : day_idx], axis=1)
        vol = np.std(returns[:, day_idx - VOL_WINDOW : day_idx], axis=1, ddof=1)
        vol = np.maximum(vol, VOL_FLOOR)

        signal = np.sign(trend).astype(int)
        active = signal != 0

        if not np.any(active):
            continue

        inv_vol = np.zeros(num_stocks, dtype=float)
        inv_vol[active] = 1.0 / vol[active]

        median_inv_vol = np.median(inv_vol[active])
        if not np.isfinite(median_inv_vol) or median_inv_vol <= 0:
            continue

        relative_inv_vol = np.zeros(num_stocks, dtype=float)
        relative_inv_vol[active] = inv_vol[active] / median_inv_vol

        shares = np.zeros(num_stocks, dtype=int)
        shares[active] = np.ceil((MAX_SHARES / 3.0) * relative_inv_vol[active]).astype(int)
        shares = np.clip(shares, 0, MAX_SHARES)

        positions[:, day_idx] = signal * shares

    actions = np.zeros((num_stocks, num_days), dtype=int)
    actions[:, 0] = positions[:, 0]
    actions[:, 1:] = positions[:, 1:] - positions[:, :-1]
    return actions
