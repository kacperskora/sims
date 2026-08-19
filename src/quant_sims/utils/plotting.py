"""Shared plotting helpers for quant_sims notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_equity_curves(paths: np.ndarray, title: str = "Equity Curves", log_scale: bool = True, max_paths: int = 200, ax=None):
    """
    Plot a sample of simulated capital paths.

    paths : array of shape (n_simulations, n_steps)
    max_paths : cap on number of individual paths drawn (for readability/perf)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    n_to_plot = min(max_paths, paths.shape[0])
    idx = np.random.choice(paths.shape[0], size=n_to_plot, replace=False)

    for i in idx:
        ax.plot(paths[i], alpha=0.15, color="steelblue", linewidth=0.8)

    median_path = np.median(paths, axis=0)
    ax.plot(median_path, color="darkorange", linewidth=2, label="Median path")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Capital")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_final_capital_distribution(paths: np.ndarray, title: str = "Distribution of Final Capital", bins: int = 50, ax=None):
    """Histogram of final capital across simulations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    final_values = paths[:, -1]
    ax.hist(final_values, bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(np.median(final_values), color="darkorange", linewidth=2, label=f"Median = {np.median(final_values):.1f}")
    ax.axvline(np.mean(final_values), color="green", linewidth=2, linestyle="--", label=f"Mean = {np.mean(final_values):.1f}")
    ax.set_xlabel("Final capital")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_strategy_comparison(paths_by_strategy: dict, title: str = "Strategy Comparison (median paths)", log_scale: bool = True, ax=None):
    """
    Overlay median equity curves for multiple strategies.

    paths_by_strategy : dict mapping strategy name -> paths array (n_sims, n_steps)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for name, paths in paths_by_strategy.items():
        median_path = np.median(paths, axis=0)
        ax.plot(median_path, linewidth=2, label=name)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Capital (median across simulations)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_paths(time_grid: np.ndarray, paths: np.ndarray, title: str = "Simulated Paths", max_paths: int = 200, ylabel: str = "Value", ax=None):
    """
    Plot a sample of simulated paths against a shared time grid (used for
    random walk / Brownian motion / GBM path plots).

    paths : array of shape (n_paths, len(time_grid))
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    n_to_plot = min(max_paths, paths.shape[0])
    idx = np.random.choice(paths.shape[0], size=n_to_plot, replace=False)

    for i in idx:
        ax.plot(time_grid, paths[i], alpha=0.15, color="steelblue", linewidth=0.8)

    mean_path = np.mean(paths, axis=0)
    ax.plot(time_grid, mean_path, color="darkorange", linewidth=2, label="Mean path")

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_terminal_distribution_vs_theory(values: np.ndarray, theoretical_pdf_fn=None, title: str = "Terminal Value Distribution", bins: int = 60, ax=None):
    """
    Histogram of terminal values with an optional theoretical density curve
    overlaid (e.g. a normal or lognormal pdf) for visual comparison against
    the empirical Monte Carlo distribution.

    theoretical_pdf_fn : callable taking an array of x-values and returning
        pdf values, or None to skip overlay.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.hist(values, bins=bins, density=True, color="steelblue", edgecolor="white", alpha=0.7, label="Simulated")

    if theoretical_pdf_fn is not None:
        x = np.linspace(values.min(), values.max(), 400)
        ax.plot(x, theoretical_pdf_fn(x), color="darkorange", linewidth=2, label="Theoretical")

    ax.set_xlabel("Terminal value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_growth_rate_curve(f_values: np.ndarray, growth_rates: np.ndarray, f_star: float, title: str = "Expected Log Growth Rate vs. Bet Fraction", ax=None):
    """Plot expected log growth rate as a function of bet fraction, marking the Kelly-optimal point."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(f_values, growth_rates, color="steelblue", linewidth=2)
    ax.axvline(f_star, color="darkorange", linestyle="--", label=f"f* = {f_star:.3f}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Bet fraction f")
    ax.set_ylabel("Expected log growth per bet")
    ax.set_title(title)
    ax.legend()
    return ax
