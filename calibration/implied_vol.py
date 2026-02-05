"""
Implied Volatility Calculation

Given observed market price, solve for volatility σ:
    V_market = V_BS(S, K, T, r, σ, q)

Methods:
    1. Newton-Raphson: Fast convergence using Vega
       σ_{n+1} = σ_n - [V(σ_n) - V_market] / Vega(σ_n)

    2. Bisection: Robust but slower
       Guaranteed convergence if root exists in bracket

    3. Brent's Method: Hybrid approach (scipy.optimize)

Numerical considerations:
    - Volatility bounds: typically [0.001, 5.0]
    - Deep ITM/OTM options may have numerical issues
    - Check for arbitrage violations
"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Literal, Optional, Dict
from pricers.black_scholes import BlackScholesPricer


class ImpliedVolatility:
    """
    Implied volatility solver using multiple numerical methods.

    Inverts Black-Scholes formula to recover volatility from market price.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        vol_min: float = 0.001,
        vol_max: float = 5.0
    ):
        """
        Initialize implied volatility solver.

        Args:
            S: Current asset price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            q: Dividend yield (default 0)
            vol_min: Minimum volatility for search (default 0.1%)
            vol_max: Maximum volatility for search (default 500%)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.vol_min = vol_min
        self.vol_max = vol_max

        # Intrinsic values for arbitrage checks
        discount = np.exp(-r * T)
        forward_discount = np.exp(-q * T)
        self.call_intrinsic = max(S * forward_discount - K * discount, 0)
        self.put_intrinsic = max(K * discount - S * forward_discount, 0)

    def _objective_function(
        self,
        sigma: float,
        option_type: Literal['call', 'put'],
        market_price: float
    ) -> float:
        """
        Objective function: BS_price(σ) - market_price

        Args:
            sigma: Volatility to evaluate
            option_type: 'call' or 'put'
            market_price: Observed market price

        Returns:
            Difference between theoretical and market price
        """
        pricer = BlackScholesPricer(self.S, self.K, self.T, self.r, sigma, self.q)
        return pricer.price(option_type) - market_price

    def _vega(self, sigma: float) -> float:
        """
        Calculate vega for Newton-Raphson method.

        Args:
            sigma: Volatility

        Returns:
            Vega value (per unit, not per 1%)
        """
        pricer = BlackScholesPricer(self.S, self.K, self.T, self.r, sigma, self.q)
        return pricer.vega() * 100  # Convert back to per unit

    def newton_raphson(
        self,
        option_type: Literal['call', 'put'],
        market_price: float,
        initial_guess: float = 0.3,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Solve for implied volatility using Newton-Raphson method.

        Iteration: σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)
        where f(σ) = V_BS(σ) - V_market

        Args:
            option_type: 'call' or 'put'
            market_price: Observed market price
            initial_guess: Starting volatility (default 30%)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Dictionary with implied_vol, iterations, and error

        Raises:
            ValueError: If no convergence or price is invalid
        """
        # Arbitrage check
        self._check_arbitrage(option_type, market_price)

        sigma = initial_guess

        for i in range(max_iter):
            # Calculate price difference and vega
            price_diff = self._objective_function(sigma, option_type, market_price)
            vega = self._vega(sigma)

            # Check convergence
            if abs(price_diff) < tol:
                return {
                    'implied_vol': sigma,
                    'iterations': i + 1,
                    'error': abs(price_diff),
                    'converged': True
                }

            # Newton update
            if abs(vega) < 1e-10:
                raise ValueError("Vega too small, Newton-Raphson failed")

            sigma_new = sigma - price_diff / vega

            # Enforce bounds
            sigma_new = np.clip(sigma_new, self.vol_min, self.vol_max)

            sigma = sigma_new

        # Did not converge
        return {
            'implied_vol': sigma,
            'iterations': max_iter,
            'error': abs(self._objective_function(sigma, option_type, market_price)),
            'converged': False
        }

    def bisection(
        self,
        option_type: Literal['call', 'put'],
        market_price: float,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Solve for implied volatility using bisection method.

        Guaranteed to converge if root exists in bracket.
        Slower than Newton-Raphson but more robust.

        Args:
            option_type: 'call' or 'put'
            market_price: Observed market price
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Dictionary with implied_vol, iterations, and error
        """
        # Arbitrage check
        self._check_arbitrage(option_type, market_price)

        # Check if root exists in bracket
        f_min = self._objective_function(self.vol_min, option_type, market_price)
        f_max = self._objective_function(self.vol_max, option_type, market_price)

        if f_min * f_max > 0:
            raise ValueError(
                f"Root not bracketed: f({self.vol_min}) = {f_min:.6f}, "
                f"f({self.vol_max}) = {f_max:.6f}"
            )

        # Bisection iterations
        sigma_low = self.vol_min
        sigma_high = self.vol_max

        for i in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            f_mid = self._objective_function(sigma_mid, option_type, market_price)

            # Check convergence
            if abs(f_mid) < tol or abs(sigma_high - sigma_low) < tol:
                return {
                    'implied_vol': sigma_mid,
                    'iterations': i + 1,
                    'error': abs(f_mid),
                    'converged': True
                }

            # Update bracket
            f_low = self._objective_function(sigma_low, option_type, market_price)
            if f_low * f_mid < 0:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid

        # Did not converge
        return {
            'implied_vol': sigma_mid,
            'iterations': max_iter,
            'error': abs(f_mid),
            'converged': False
        }

    def brent(
        self,
        option_type: Literal['call', 'put'],
        market_price: float,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Solve for implied volatility using Brent's method (scipy).

        Hybrid root-finding combining bisection, secant, and inverse quadratic interpolation.
        Best of both worlds: robustness of bisection with speed of Newton.

        Args:
            option_type: 'call' or 'put'
            market_price: Observed market price
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Dictionary with implied_vol and error
        """
        # Arbitrage check
        self._check_arbitrage(option_type, market_price)

        try:
            sigma = brentq(
                lambda s: self._objective_function(s, option_type, market_price),
                self.vol_min,
                self.vol_max,
                maxiter=max_iter,
                xtol=tol
            )

            error = abs(self._objective_function(sigma, option_type, market_price))

            return {
                'implied_vol': sigma,
                'error': error,
                'converged': True
            }

        except ValueError as e:
            raise ValueError(f"Brent's method failed: {str(e)}")

    def solve(
        self,
        option_type: Literal['call', 'put'],
        market_price: float,
        method: Literal['newton', 'bisection', 'brent'] = 'newton',
        **kwargs
    ) -> float:
        """
        Solve for implied volatility using specified method.

        Args:
            option_type: 'call' or 'put'
            market_price: Observed market price
            method: Numerical method (default: 'newton')
            **kwargs: Additional arguments for chosen method

        Returns:
            Implied volatility

        Raises:
            ValueError: If method is unknown or solver fails
        """
        if method == 'newton':
            result = self.newton_raphson(option_type, market_price, **kwargs)
        elif method == 'bisection':
            result = self.bisection(option_type, market_price, **kwargs)
        elif method == 'brent':
            result = self.brent(option_type, market_price, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'newton', 'bisection', or 'brent'")

        if not result.get('converged', True):
            import warnings
            warnings.warn(f"Solver did not converge. Final error: {result['error']:.2e}")

        return result['implied_vol']

    def _check_arbitrage(self, option_type: Literal['call', 'put'], market_price: float):
        """
        Check if market price violates no-arbitrage bounds.

        Args:
            option_type: 'call' or 'put'
            market_price: Observed market price

        Raises:
            ValueError: If arbitrage violations detected
        """
        if market_price <= 0:
            raise ValueError(f"Market price must be positive, got {market_price}")

        if option_type == 'call':
            if market_price < self.call_intrinsic:
                raise ValueError(
                    f"Call price {market_price:.4f} below intrinsic value "
                    f"{self.call_intrinsic:.4f} (arbitrage)"
                )
            # Call price cannot exceed spot
            if market_price > self.S:
                raise ValueError(
                    f"Call price {market_price:.4f} exceeds spot {self.S:.4f} (arbitrage)"
                )
        elif option_type == 'put':
            if market_price < self.put_intrinsic:
                raise ValueError(
                    f"Put price {market_price:.4f} below intrinsic value "
                    f"{self.put_intrinsic:.4f} (arbitrage)"
                )
            # Put price cannot exceed strike
            discount = np.exp(-self.r * self.T)
            if market_price > self.K * discount:
                raise ValueError(
                    f"Put price {market_price:.4f} exceeds discounted strike "
                    f"{self.K * discount:.4f} (arbitrage)"
                )

    def volatility_smile(
        self,
        option_type: Literal['call', 'put'],
        strikes: np.ndarray,
        market_prices: np.ndarray,
        method: str = 'newton'
    ) -> np.ndarray:
        """
        Calculate implied volatility smile across strikes.

        Args:
            option_type: 'call' or 'put'
            strikes: Array of strike prices
            market_prices: Array of corresponding market prices
            method: Numerical method

        Returns:
            Array of implied volatilities
        """
        implied_vols = []

        for K, price in zip(strikes, market_prices):
            # Create new solver for this strike
            solver = ImpliedVolatility(self.S, K, self.T, self.r, self.q)

            try:
                iv = solver.solve(option_type, price, method=method)
                implied_vols.append(iv)
            except (ValueError, RuntimeError):
                # If solver fails, append NaN
                implied_vols.append(np.nan)

        return np.array(implied_vols)

    def __repr__(self) -> str:
        return (f"ImpliedVolatility(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f})")
