"""
Black-Scholes Analytical Pricing

Implements closed-form solutions for European options based on:
    - Risk-neutral valuation: V(S,t) = e^(-rτ) E^Q[Payoff(S_T) | S_t = S]
    - Black-Scholes PDE: ∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² - rV = 0

Formulas:
    Call: C = S₀e^(-qτ)Φ(d₁) - Ke^(-rτ)Φ(d₂)
    Put:  P = Ke^(-rτ)Φ(-d₂) - S₀e^(-qτ)Φ(-d₁)

where:
    d₁ = [ln(S₀/K) + (r - q + σ²/2)τ] / (σ√τ)
    d₂ = d₁ - σ√τ
    Φ: Standard normal CDF
"""

import numpy as np
from scipy.stats import norm
from typing import Literal, Dict, Tuple


class BlackScholesPricer:
    """
    Analytical Black-Scholes pricer for European options.

    Provides closed-form pricing and Greeks computation based on the
    Black-Scholes-Merton framework.
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        Initialize pricer with option parameters.

        Args:
            S: Current asset price (spot)
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            q: Dividend yield (annualized, default 0)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

        # Calculate d1 and d2
        self._d1, self._d2 = self._calculate_d1_d2()

    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.

        Returns:
            Tuple of (d1, d2)
        """
        if self.T <= 0:
            # At expiration
            return 0.0, 0.0

        sqrt_T = np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T

        return d1, d2

    def price(self, option_type: Literal['call', 'put']) -> float:
        """
        Calculate option price using Black-Scholes formula.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Option price

        Raises:
            ValueError: If option_type is not 'call' or 'put'
        """
        if self.T <= 0:
            # Intrinsic value at expiration
            if option_type == 'call':
                return max(self.S - self.K, 0)
            elif option_type == 'put':
                return max(self.K - self.S, 0)
            else:
                raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        discount = np.exp(-self.r * self.T)
        forward_discount = np.exp(-self.q * self.T)

        if option_type == 'call':
            price = (self.S * forward_discount * norm.cdf(self._d1) -
                    self.K * discount * norm.cdf(self._d2))
        elif option_type == 'put':
            price = (self.K * discount * norm.cdf(-self._d2) -
                    self.S * forward_discount * norm.cdf(-self._d1))
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        return price

    def delta(self, option_type: Literal['call', 'put']) -> float:
        """
        Calculate Delta: ∂V/∂S

        Measures sensitivity of option price to changes in underlying price.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Delta value
        """
        if self.T <= 0:
            if option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0

        forward_discount = np.exp(-self.q * self.T)

        if option_type == 'call':
            return forward_discount * norm.cdf(self._d1)
        elif option_type == 'put':
            return -forward_discount * norm.cdf(-self._d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def gamma(self) -> float:
        """
        Calculate Gamma: ∂²V/∂S²

        Measures rate of change of Delta with respect to underlying price.
        Gamma is the same for calls and puts.

        Returns:
            Gamma value
        """
        if self.T <= 0:
            return 0.0

        forward_discount = np.exp(-self.q * self.T)
        sqrt_T = np.sqrt(self.T)

        gamma = (forward_discount * norm.pdf(self._d1)) / (self.S * self.sigma * sqrt_T)
        return gamma

    def vega(self) -> float:
        """
        Calculate Vega: ∂V/∂σ

        Measures sensitivity of option price to changes in volatility.
        Vega is the same for calls and puts.

        Note: Returned value is per 1% change in volatility (divide by 100 for per unit)

        Returns:
            Vega value
        """
        if self.T <= 0:
            return 0.0

        forward_discount = np.exp(-self.q * self.T)
        sqrt_T = np.sqrt(self.T)

        vega = self.S * forward_discount * norm.pdf(self._d1) * sqrt_T
        return vega / 100  # Per 1% volatility change

    def theta(self, option_type: Literal['call', 'put']) -> float:
        """
        Calculate Theta: ∂V/∂t

        Measures sensitivity of option price to passage of time (time decay).

        Note: Returned value is per day (divide by 365)

        Args:
            option_type: 'call' or 'put'

        Returns:
            Theta value (negative for long positions)
        """
        if self.T <= 0:
            return 0.0

        sqrt_T = np.sqrt(self.T)
        discount = np.exp(-self.r * self.T)
        forward_discount = np.exp(-self.q * self.T)

        term1 = -(self.S * forward_discount * norm.pdf(self._d1) * self.sigma) / (2 * sqrt_T)

        if option_type == 'call':
            term2 = self.q * self.S * forward_discount * norm.cdf(self._d1)
            term3 = -self.r * self.K * discount * norm.cdf(self._d2)
            theta = term1 - term2 + term3
        elif option_type == 'put':
            term2 = self.q * self.S * forward_discount * norm.cdf(-self._d1)
            term3 = self.r * self.K * discount * norm.cdf(-self._d2)
            theta = term1 + term2 - term3
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        return theta / 365  # Per day

    def rho(self, option_type: Literal['call', 'put']) -> float:
        """
        Calculate Rho: ∂V/∂r

        Measures sensitivity of option price to changes in risk-free rate.

        Note: Returned value is per 1% change in rate

        Args:
            option_type: 'call' or 'put'

        Returns:
            Rho value
        """
        if self.T <= 0:
            return 0.0

        discount = np.exp(-self.r * self.T)

        if option_type == 'call':
            rho = self.K * self.T * discount * norm.cdf(self._d2)
        elif option_type == 'put':
            rho = -self.K * self.T * discount * norm.cdf(-self._d2)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        return rho / 100  # Per 1% rate change

    def all_greeks(self, option_type: Literal['call', 'put']) -> Dict[str, float]:
        """
        Calculate all Greeks at once.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary containing price and all Greeks
        """
        return {
            'price': self.price(option_type),
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type)
        }

    def put_call_parity_check(self) -> Dict[str, float]:
        """
        Verify put-call parity relationship.

        Put-call parity: C - P = S₀e^(-qτ) - Ke^(-rτ)

        Returns:
            Dictionary with call, put, and parity difference
        """
        call_price = self.price('call')
        put_price = self.price('put')

        parity_lhs = call_price - put_price
        parity_rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        difference = abs(parity_lhs - parity_rhs)

        return {
            'call': call_price,
            'put': put_price,
            'parity_lhs': parity_lhs,
            'parity_rhs': parity_rhs,
            'difference': difference
        }

    def __repr__(self) -> str:
        return (f"BlackScholesPricer(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, sigma={self.sigma:.4f}, q={self.q:.4f})")
