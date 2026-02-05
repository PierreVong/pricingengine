"""
Geometric Brownian Motion (GBM) Model

Implements the stochastic differential equation:
    dS_t = μ S_t dt + σ S_t dW_t

Under the risk-neutral measure (via Girsanov theorem):
    dS_t = r S_t dt + σ S_t dW_t^Q

where:
    S_t: Asset price at time t
    μ: Drift (expected return)
    r: Risk-free rate
    σ: Volatility
    W_t: Brownian motion
"""

import numpy as np
from typing import Optional, Tuple


class GBMModel:
    """
    Geometric Brownian Motion asset price model.

    The GBM model assumes log-normal distribution of asset prices:
        S_T = S_0 * exp((r - σ²/2)T + σ√T * Z)
    where Z ~ N(0,1)

    Attributes:
        S0: Initial asset price
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        q: Dividend yield (default: 0)
    """

    def __init__(self, S0: float, r: float, sigma: float, q: float = 0.0):
        """
        Initialize GBM model parameters.

        Args:
            S0: Initial asset price (must be positive)
            r: Risk-free rate (annualized, e.g., 0.05 for 5%)
            sigma: Volatility (annualized, e.g., 0.2 for 20%)
            q: Dividend yield (annualized, default 0)

        Raises:
            ValueError: If S0 <= 0 or sigma < 0
        """
        if S0 <= 0:
            raise ValueError(f"Initial price S0 must be positive, got {S0}")
        if sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")

        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.q = q

    def simulate_paths(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate asset price paths using exact solution of GBM SDE.

        Uses the analytical solution:
            S(t+Δt) = S(t) * exp((r - q - σ²/2)Δt + σ√Δt * Z)

        Args:
            T: Time to maturity (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility
            antithetic: If True, use antithetic variates (doubles paths)

        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated prices
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps

        # Generate random normal variates
        if antithetic:
            Z = np.random.standard_normal((n_paths // 2, n_steps))
            Z = np.vstack([Z, -Z])  # Antithetic pairs
        else:
            Z = np.random.standard_normal((n_paths, n_steps))

        # Drift and diffusion components
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Calculate log returns
        log_returns = drift + diffusion * Z

        # Initialize paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        # Calculate cumulative log returns and exponentiate
        paths[:, 1:] = self.S0 * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def simulate_terminal_prices(
        self,
        T: float,
        n_paths: int,
        seed: Optional[int] = None,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate terminal asset prices at maturity T (single time step).

        More efficient than simulate_paths when only terminal values are needed.

        Args:
            T: Time to maturity (years)
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility
            antithetic: If True, use antithetic variates

        Returns:
            Array of shape (n_paths,) containing terminal prices
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random normal variates
        if antithetic:
            Z = np.random.standard_normal(n_paths // 2)
            Z = np.concatenate([Z, -Z])
        else:
            Z = np.random.standard_normal(n_paths)

        # Calculate terminal prices using exact solution
        drift = (self.r - self.q - 0.5 * self.sigma**2) * T
        diffusion = self.sigma * np.sqrt(T)

        S_T = self.S0 * np.exp(drift + diffusion * Z)

        return S_T

    def expected_value(self, T: float) -> float:
        """
        Expected asset price at time T under risk-neutral measure.

        E^Q[S_T] = S_0 * exp((r-q)T)

        Args:
            T: Time horizon (years)

        Returns:
            Expected price at time T
        """
        return self.S0 * np.exp((self.r - self.q) * T)

    def variance(self, T: float) -> float:
        """
        Variance of asset price at time T under risk-neutral measure.

        Var^Q[S_T] = S_0² * exp(2(r-q)T) * (exp(σ²T) - 1)

        Args:
            T: Time horizon (years)

        Returns:
            Variance of price at time T
        """
        exp_mean = self.expected_value(T)
        return exp_mean**2 * (np.exp(self.sigma**2 * T) - 1)

    def __repr__(self) -> str:
        return (f"GBMModel(S0={self.S0:.2f}, r={self.r:.4f}, "
                f"sigma={self.sigma:.4f}, q={self.q:.4f})")
