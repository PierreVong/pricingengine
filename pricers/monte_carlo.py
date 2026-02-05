"""
Monte Carlo Option Pricing

Prices options by simulating asset paths under risk-neutral measure:
    V(S,t) = e^(-r(T-t)) E^Q[Payoff(S_T) | S_t = S]

Methods:
    1. Standard Monte Carlo
    2. Antithetic Variates: Use pairs (Z, -Z) to reduce variance
    3. Control Variates: Use correlation with known-price instrument

Variance reduction significantly improves convergence rate and reduces
computational cost for target accuracy.

Standard Error: SE = σ/√N
With variance reduction: SE_reduced < SE_standard
"""

import numpy as np
from typing import Literal, Dict, Optional, Callable
from scipy.stats import norm


class MonteCarloPricer:
    """
    Monte Carlo option pricer with variance reduction techniques.

    Simulates risk-neutral paths and calculates discounted expected payoff.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        n_simulations: int = 100000,
        seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo pricer.

        Args:
            S: Current asset price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
            n_simulations: Number of Monte Carlo paths
            seed: Random seed for reproducibility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_simulations = n_simulations
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def _simulate_terminal_prices(self, n_paths: int, antithetic: bool = False) -> np.ndarray:
        """
        Simulate terminal asset prices at maturity.

        Uses exact solution: S_T = S_0 * exp((r - q - σ²/2)T + σ√T * Z)

        Args:
            n_paths: Number of paths to simulate
            antithetic: Use antithetic variates if True

        Returns:
            Array of terminal prices
        """
        if antithetic:
            # Generate half the required paths
            Z = np.random.standard_normal(n_paths // 2)
            # Create antithetic pairs
            Z = np.concatenate([Z, -Z])
        else:
            Z = np.random.standard_normal(n_paths)

        # Calculate terminal prices using exact GBM solution
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T)

        S_T = self.S * np.exp(drift + diffusion * Z)

        return S_T

    def _payoff(self, S_T: np.ndarray, option_type: Literal['call', 'put']) -> np.ndarray:
        """
        Calculate option payoff at maturity.

        Args:
            S_T: Terminal asset prices
            option_type: 'call' or 'put'

        Returns:
            Array of payoffs
        """
        if option_type == 'call':
            return np.maximum(S_T - self.K, 0)
        elif option_type == 'put':
            return np.maximum(self.K - S_T, 0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def price_standard(self, option_type: Literal['call', 'put']) -> Dict[str, float]:
        """
        Standard Monte Carlo pricing (no variance reduction).

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary with price, standard error, and confidence interval
        """
        # Simulate terminal prices
        S_T = self._simulate_terminal_prices(self.n_simulations, antithetic=False)

        # Calculate payoffs
        payoffs = self._payoff(S_T, option_type)

        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_simulations)

        # 95% confidence interval
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }

    def price_antithetic(self, option_type: Literal['call', 'put']) -> Dict[str, float]:
        """
        Monte Carlo pricing with antithetic variates.

        Uses pairs (Z, -Z) to reduce variance by exploiting negative correlation.
        Variance reduction depends on payoff function monotonicity.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary with price, standard error, and confidence interval
        """
        # Simulate terminal prices with antithetic variates
        S_T = self._simulate_terminal_prices(self.n_simulations, antithetic=True)

        # Calculate payoffs
        payoffs = self._payoff(S_T, option_type)

        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_simulations)

        # 95% confidence interval
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }

    def price_control_variate(self, option_type: Literal['call', 'put']) -> Dict[str, float]:
        """
        Monte Carlo pricing with control variates.

        Uses geometric average Asian option as control (has analytical solution).
        Reduction: Var[Y*] = Var[Y](1 - ρ²)

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary with price, standard error, and confidence interval
        """
        # Simulate terminal prices
        S_T = self._simulate_terminal_prices(self.n_simulations, antithetic=False)

        # Calculate payoffs (target estimator)
        payoffs = self._payoff(S_T, option_type)
        discount_factor = np.exp(-self.r * self.T)
        Y = discount_factor * payoffs

        # Control variate: Use terminal price itself (known expectation)
        # E[S_T] = S_0 * exp((r-q)T)
        X = S_T
        expected_X = self.S * np.exp((self.r - self.q) * self.T)

        # Calculate optimal coefficient
        # c* = -Cov[Y,X] / Var[X]
        covariance = np.cov(Y, X)[0, 1]
        variance_X = np.var(X, ddof=1)
        c = -covariance / variance_X

        # Control variate estimator
        Y_cv = Y + c * (X - expected_X)

        # Calculate statistics
        price = np.mean(Y_cv)
        std_error = np.std(Y_cv, ddof=1) / np.sqrt(self.n_simulations)

        # 95% confidence interval
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        # Variance reduction ratio
        var_reduction = np.var(Y_cv, ddof=1) / np.var(Y, ddof=1)

        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'variance_reduction_ratio': var_reduction,
            'optimal_c': c
        }

    def price(
        self,
        option_type: Literal['call', 'put'],
        method: Literal['standard', 'antithetic', 'control_variate'] = 'antithetic'
    ) -> Dict[str, float]:
        """
        Price option using specified Monte Carlo method.

        Args:
            option_type: 'call' or 'put'
            method: Variance reduction technique (default: 'antithetic')

        Returns:
            Dictionary with pricing results

        Raises:
            ValueError: If method is not recognized
        """
        if method == 'standard':
            return self.price_standard(option_type)
        elif method == 'antithetic':
            return self.price_antithetic(option_type)
        elif method == 'control_variate':
            return self.price_control_variate(option_type)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'standard', 'antithetic', or 'control_variate'")

    def price_path_dependent(
        self,
        option_type: str,
        payoff_func: Callable[[np.ndarray], np.ndarray],
        n_steps: int = 252,
        antithetic: bool = True
    ) -> Dict[str, float]:
        """
        Price path-dependent options using Monte Carlo.

        For exotic options like Asian, Barrier, Lookback options.

        Args:
            option_type: Description of option type (for documentation)
            payoff_func: Function that takes path array (n_paths, n_steps+1) and returns payoffs
            n_steps: Number of time steps per path
            antithetic: Use antithetic variates

        Returns:
            Dictionary with pricing results
        """
        dt = self.T / n_steps

        # Generate random increments
        if antithetic:
            Z = np.random.standard_normal((self.n_simulations // 2, n_steps))
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.standard_normal((self.n_simulations, n_steps))

        # Simulate paths
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        log_returns = drift + diffusion * Z
        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = self.S
        paths[:, 1:] = self.S * np.exp(np.cumsum(log_returns, axis=1))

        # Calculate payoffs using provided function
        payoffs = payoff_func(paths)

        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_simulations)

        # 95% confidence interval
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }

    def delta(
        self,
        option_type: Literal['call', 'put'],
        method: Literal['standard', 'antithetic'] = 'antithetic',
        bump_size: float = 0.01
    ) -> float:
        """
        Estimate Delta using finite difference (pathwise method).

        Δ ≈ [V(S+ΔS) - V(S-ΔS)] / (2ΔS)

        Args:
            option_type: 'call' or 'put'
            method: Monte Carlo method
            bump_size: Relative bump size (default 1%)

        Returns:
            Delta estimate
        """
        # Price at S
        V_0 = self.price(option_type, method)['price']

        # Price at S + bump
        S_original = self.S
        self.S = S_original * (1 + bump_size)
        V_up = self.price(option_type, method)['price']

        # Price at S - bump
        self.S = S_original * (1 - bump_size)
        V_down = self.price(option_type, method)['price']

        # Restore original S
        self.S = S_original

        # Central difference
        delta = (V_up - V_down) / (2 * S_original * bump_size)

        return delta

    def __repr__(self) -> str:
        return (f"MonteCarloPricer(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, sigma={self.sigma:.4f}, n_sim={self.n_simulations})")
