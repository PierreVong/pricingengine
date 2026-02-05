"""
Basic tests to verify pricing engine functionality.

Run with: python -m pytest tests/test_basic.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pricers.black_scholes import BlackScholesPricer
from pricers.pde_pricer import PDEPricer
from pricers.monte_carlo import MonteCarloPricer
from models.gbm import GBMModel
from calibration.implied_vol import ImpliedVolatility


class TestBlackScholes:
    """Test Black-Scholes analytical pricer."""

    def setup_method(self):
        """Setup test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        self.q = 0.0

    def test_call_price_positive(self):
        """Call price should be positive."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        price = bs.price('call')
        assert price > 0

    def test_put_call_parity(self):
        """Verify put-call parity relationship."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        call_price = bs.price('call')
        put_price = bs.price('put')

        # C - P = S*e^(-qT) - K*e^(-rT)
        lhs = call_price - put_price
        rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)

        assert abs(lhs - rhs) < 1e-10

    def test_call_delta_bounds(self):
        """Call delta should be in (0, 1)."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        delta = bs.delta('call')
        assert 0 < delta < 1

    def test_gamma_positive(self):
        """Gamma should be positive for long options."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        gamma = bs.gamma()
        assert gamma > 0

    def test_vega_positive(self):
        """Vega should be positive."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        vega = bs.vega()
        assert vega > 0


class TestPDEPricer:
    """Test PDE finite difference methods."""

    def setup_method(self):
        """Setup test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        self.q = 0.0

    def test_pde_vs_bs(self):
        """PDE price should match Black-Scholes within tolerance."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        bs_price = bs.price('call')

        pde = PDEPricer(self.S, self.K, self.T, self.r, self.sigma, self.q, M=100, N=1000)
        pde_price = pde.price('call', method='crank-nicolson')

        relative_error = abs(pde_price - bs_price) / bs_price
        assert relative_error < 0.01  # 1% tolerance

    def test_crank_nicolson_most_accurate(self):
        """Crank-Nicolson should be most accurate among PDE methods."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        bs_price = bs.price('call')

        pde = PDEPricer(self.S, self.K, self.T, self.r, self.sigma, self.q, M=100, N=1000)

        cn_error = abs(pde.price('call', method='crank-nicolson') - bs_price)
        impl_error = abs(pde.price('call', method='implicit') - bs_price)

        # Crank-Nicolson should have smallest error
        assert cn_error <= impl_error


class TestMonteCarlo:
    """Test Monte Carlo pricer."""

    def setup_method(self):
        """Setup test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        self.q = 0.0

    def test_mc_converges_to_bs(self):
        """Monte Carlo should converge to Black-Scholes with enough simulations."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        bs_price = bs.price('call')

        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            n_simulations=100000, seed=42
        )
        mc_result = mc.price('call', method='antithetic')

        # Should be within 3 standard errors
        assert abs(mc_result['price'] - bs_price) < 3 * mc_result['std_error']

    def test_variance_reduction_helps(self):
        """Variance reduction should reduce standard error."""
        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            n_simulations=50000, seed=42
        )

        standard = mc.price('call', method='standard')
        antithetic = mc.price('call', method='antithetic')

        # Antithetic should have smaller standard error
        assert antithetic['std_error'] < standard['std_error']


class TestGBMModel:
    """Test Geometric Brownian Motion model."""

    def test_terminal_price_distribution(self):
        """Terminal prices should be log-normal."""
        S0 = 100.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        T = 1.0

        model = GBMModel(S0, r, sigma, q)
        S_T = model.simulate_terminal_prices(T, n_paths=10000, seed=42)

        # Log returns should be approximately normal
        log_returns = np.log(S_T / S0)

        # Mean should be close to (r - q - sigma^2/2) * T
        expected_mean = (r - q - 0.5 * sigma**2) * T
        assert abs(np.mean(log_returns) - expected_mean) < 0.05

        # Std should be close to sigma * sqrt(T)
        expected_std = sigma * np.sqrt(T)
        assert abs(np.std(log_returns) - expected_std) < 0.02

    def test_expected_value_under_risk_neutral(self):
        """Expected terminal price under Q should be S0 * exp((r-q)T)."""
        S0 = 100.0
        r = 0.05
        sigma = 0.2
        q = 0.02
        T = 1.0

        model = GBMModel(S0, r, sigma, q)
        S_T = model.simulate_terminal_prices(T, n_paths=100000, seed=42)

        expected = S0 * np.exp((r - q) * T)
        assert abs(np.mean(S_T) - expected) < 1.0  # Within $1


class TestImpliedVolatility:
    """Test implied volatility solver."""

    def setup_method(self):
        """Setup test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.q = 0.0
        self.sigma_true = 0.2

    def test_recover_true_volatility(self):
        """Should recover true volatility from Black-Scholes price."""
        # Generate "market" price using true vol
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma_true, self.q)
        market_price = bs.price('call')

        # Solve for implied vol
        iv_solver = ImpliedVolatility(self.S, self.K, self.T, self.r, self.q)
        implied_vol = iv_solver.solve('call', market_price, method='newton')

        # Should match true volatility
        assert abs(implied_vol - self.sigma_true) < 1e-6

    def test_all_methods_agree(self):
        """All solver methods should give same result."""
        bs = BlackScholesPricer(self.S, self.K, self.T, self.r, self.sigma_true, self.q)
        market_price = bs.price('call')

        iv_solver = ImpliedVolatility(self.S, self.K, self.T, self.r, self.q)

        iv_newton = iv_solver.solve('call', market_price, method='newton')
        iv_bisection = iv_solver.solve('call', market_price, method='bisection')
        iv_brent = iv_solver.solve('call', market_price, method='brent')

        assert abs(iv_newton - iv_bisection) < 1e-4
        assert abs(iv_newton - iv_brent) < 1e-4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
