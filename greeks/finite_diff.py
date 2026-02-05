"""
Greeks Calculation via Finite Differences

Computes option sensitivities numerically for any pricing method.

First-order Greeks (central difference):
    Delta:  ∂V/∂S ≈ [V(S+h) - V(S-h)] / (2h)
    Vega:   ∂V/∂σ ≈ [V(σ+h) - V(σ-h)] / (2h)
    Theta:  ∂V/∂t ≈ [V(t+h) - V(t)] / h  (forward, as time decreases)
    Rho:    ∂V/∂r ≈ [V(r+h) - V(r-h)] / (2h)

Second-order Greeks:
    Gamma:  ∂²V/∂S² ≈ [V(S+h) - 2V(S) + V(S-h)] / h²

Useful for:
    - Validating analytical Greeks
    - Computing Greeks for PDE/MC methods
    - Exotic options without closed-form Greeks
"""

import numpy as np
from typing import Callable, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class GreeksResult:
    """Container for Greeks calculation results."""
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    price: Optional[float] = None


class GreeksCalculator:
    """
    Numerical Greeks calculator using finite differences.

    Works with any pricing function that returns option value.
    """

    def __init__(
        self,
        pricing_func: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        **pricing_kwargs
    ):
        """
        Initialize Greeks calculator.

        Args:
            pricing_func: Function that prices option (must accept S, K, T, r, sigma, q)
            S: Current asset price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            **pricing_kwargs: Additional arguments to pass to pricing_func
        """
        self.pricing_func = pricing_func
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.pricing_kwargs = pricing_kwargs

        # Default bump sizes (relative for S, absolute for others)
        self.h_S = 0.01  # 1% of S
        self.h_sigma = 0.0001  # 1 bp
        self.h_T = 1 / 365  # 1 day
        self.h_r = 0.0001  # 1 bp

    def _price(self, **overrides) -> float:
        """
        Price option with parameter overrides.

        Args:
            **overrides: Parameter values to override

        Returns:
            Option price
        """
        params = {
            'S': self.S,
            'K': self.K,
            'T': self.T,
            'r': self.r,
            'sigma': self.sigma,
            'q': self.q,
            **self.pricing_kwargs
        }
        params.update(overrides)

        return self.pricing_func(**params)

    def delta(self, method: str = 'central') -> float:
        """
        Calculate Delta: ∂V/∂S

        Args:
            method: 'central', 'forward', or 'backward'

        Returns:
            Delta value
        """
        h = self.S * self.h_S

        if method == 'central':
            V_up = self._price(S=self.S + h)
            V_down = self._price(S=self.S - h)
            return (V_up - V_down) / (2 * h)
        elif method == 'forward':
            V_0 = self._price()
            V_up = self._price(S=self.S + h)
            return (V_up - V_0) / h
        elif method == 'backward':
            V_0 = self._price()
            V_down = self._price(S=self.S - h)
            return (V_0 - V_down) / h
        else:
            raise ValueError(f"Unknown method: {method}")

    def gamma(self) -> float:
        """
        Calculate Gamma: ∂²V/∂S²

        Uses central difference for second derivative.

        Returns:
            Gamma value
        """
        h = self.S * self.h_S

        V_up = self._price(S=self.S + h)
        V_0 = self._price()
        V_down = self._price(S=self.S - h)

        return (V_up - 2 * V_0 + V_down) / (h**2)

    def vega(self) -> float:
        """
        Calculate Vega: ∂V/∂σ

        Returns value per 1% change in volatility (divide by 100 for per unit).

        Returns:
            Vega value
        """
        h = self.h_sigma

        V_up = self._price(sigma=self.sigma + h)
        V_down = self._price(sigma=self.sigma - h)

        vega = (V_up - V_down) / (2 * h)

        return vega / 100  # Per 1% volatility change

    def theta(self) -> float:
        """
        Calculate Theta: ∂V/∂t = -∂V/∂T

        Note: Uses forward difference since time only moves forward.
        Returns value per day (divide by 365 for per year).

        Returns:
            Theta value (negative for long positions)
        """
        if self.T <= self.h_T:
            # Near expiration, use backward difference
            V_0 = self._price()
            V_past = self._price(T=self.T + self.h_T)
            theta = -(V_0 - V_past) / self.h_T
        else:
            # Normal case: forward difference
            V_0 = self._price()
            V_future = self._price(T=self.T - self.h_T)
            theta = -(V_future - V_0) / self.h_T

        return theta / 365  # Per day

    def rho(self) -> float:
        """
        Calculate Rho: ∂V/∂r

        Returns value per 1% change in rate (divide by 100 for per unit).

        Returns:
            Rho value
        """
        h = self.h_r

        V_up = self._price(r=self.r + h)
        V_down = self._price(r=self.r - h)

        rho = (V_up - V_down) / (2 * h)

        return rho / 100  # Per 1% rate change

    def all_greeks(self) -> GreeksResult:
        """
        Calculate all Greeks and price.

        More efficient than calling each method separately as it reuses
        the base price calculation.

        Returns:
            GreeksResult object with all Greeks
        """
        # Calculate base price
        price = self._price()

        # Calculate Greeks
        delta = self.delta()
        gamma = self.gamma()
        vega = self.vega()
        theta = self.theta()
        rho = self.rho()

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            price=price
        )

    def delta_gamma_approximation(self, S_new: float) -> float:
        """
        Approximate option value change using Delta-Gamma approximation.

        ΔV ≈ Δ·ΔS + ½Γ·(ΔS)²

        Args:
            S_new: New asset price

        Returns:
            Approximate new option value
        """
        V_0 = self._price()
        delta = self.delta()
        gamma = self.gamma()

        dS = S_new - self.S
        dV = delta * dS + 0.5 * gamma * dS**2

        return V_0 + dV

    def verify_against_analytical(
        self,
        analytical_greeks: Dict[str, float],
        tolerance: float = 0.01
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare numerical Greeks with analytical values.

        Args:
            analytical_greeks: Dictionary of analytical Greek values
            tolerance: Acceptable relative error

        Returns:
            Dictionary with comparison results
        """
        numerical = self.all_greeks()

        results = {}
        for greek_name in ['delta', 'gamma', 'vega', 'theta', 'rho']:
            if greek_name in analytical_greeks:
                analytical_value = analytical_greeks[greek_name]
                numerical_value = getattr(numerical, greek_name)

                abs_error = abs(numerical_value - analytical_value)
                rel_error = abs_error / abs(analytical_value) if analytical_value != 0 else abs_error

                results[greek_name] = {
                    'analytical': analytical_value,
                    'numerical': numerical_value,
                    'abs_error': abs_error,
                    'rel_error': rel_error,
                    'within_tolerance': rel_error <= tolerance
                }

        return results

    def __repr__(self) -> str:
        return (f"GreeksCalculator(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, sigma={self.sigma:.4f})")
