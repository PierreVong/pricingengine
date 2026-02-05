"""
Risk Metrics: VaR and CVaR

Value at Risk (VaR):
    VaR_α = -inf{x : P(L ≤ x) ≥ α}
    α-quantile of loss distribution (e.g., α = 0.05 for 95% VaR)

Conditional Value at Risk (CVaR) / Expected Shortfall (ES):
    CVaR_α = E[L | L ≥ VaR_α]
    Expected loss given that loss exceeds VaR

Properties:
    - CVaR is a coherent risk measure (VaR is not)
    - CVaR ≥ VaR
    - CVaR captures tail risk better than VaR

Applications:
    - Portfolio risk management
    - Regulatory capital requirements (Basel III uses ES)
    - Backtesting and model validation
"""

import numpy as np
from typing import Dict, Optional, Literal
from scipy import stats


class RiskMetrics:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR) for option positions.

    Methods:
        - Historical simulation
        - Parametric (variance-covariance)
        - Monte Carlo simulation
    """

    def __init__(
        self,
        position_value: float,
        returns: Optional[np.ndarray] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize risk metrics calculator.

        Args:
            position_value: Current position value
            returns: Historical or simulated returns (optional)
            confidence_level: Confidence level for VaR/CVaR (default 95%)
        """
        self.position_value = position_value
        self.returns = returns
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # Tail probability

    def var_historical(self, returns: Optional[np.ndarray] = None) -> float:
        """
        Calculate VaR using historical simulation.

        Non-parametric method using empirical distribution of returns.

        Args:
            returns: Array of historical returns (use self.returns if None)

        Returns:
            VaR value (positive = loss)

        Raises:
            ValueError: If no returns data available
        """
        if returns is None:
            returns = self.returns

        if returns is None or len(returns) == 0:
            raise ValueError("No returns data available")

        # Calculate P&L distribution
        pnl = self.position_value * returns

        # VaR is the α-quantile of losses (negative returns)
        var = -np.percentile(pnl, self.alpha * 100)

        return var

    def cvar_historical(self, returns: Optional[np.ndarray] = None) -> float:
        """
        Calculate CVaR (Expected Shortfall) using historical simulation.

        Average of losses exceeding VaR.

        Args:
            returns: Array of historical returns (use self.returns if None)

        Returns:
            CVaR value (positive = loss)
        """
        if returns is None:
            returns = self.returns

        if returns is None or len(returns) == 0:
            raise ValueError("No returns data available")

        # Calculate P&L distribution
        pnl = self.position_value * returns

        # Find VaR threshold
        var = -np.percentile(pnl, self.alpha * 100)

        # CVaR is the expected loss beyond VaR
        losses = -pnl[pnl <= -var]
        cvar = np.mean(losses) if len(losses) > 0 else var

        return cvar

    def var_parametric(
        self,
        mean_return: float,
        std_return: float,
        distribution: Literal['normal', 't'] = 'normal',
        df: Optional[int] = None
    ) -> float:
        """
        Calculate VaR using parametric method (variance-covariance).

        Assumes specific distribution for returns.

        Args:
            mean_return: Expected return (e.g., 0 for single period)
            std_return: Standard deviation of returns
            distribution: 'normal' or 't' (Student's t)
            df: Degrees of freedom for t-distribution

        Returns:
            VaR value (positive = loss)
        """
        if distribution == 'normal':
            # VaR_α = -(μ - z_α * σ) * V
            z_alpha = stats.norm.ppf(self.alpha)
            var = -(mean_return + z_alpha * std_return) * self.position_value

        elif distribution == 't':
            if df is None:
                raise ValueError("Degrees of freedom required for t-distribution")
            # Use t-distribution for fat tails
            t_alpha = stats.t.ppf(self.alpha, df)
            var = -(mean_return + t_alpha * std_return) * self.position_value

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return var

    def cvar_parametric(
        self,
        mean_return: float,
        std_return: float,
        distribution: Literal['normal', 't'] = 'normal',
        df: Optional[int] = None
    ) -> float:
        """
        Calculate CVaR using parametric method.

        For normal distribution:
            CVaR_α = μ - σ * φ(z_α) / α

        Args:
            mean_return: Expected return
            std_return: Standard deviation of returns
            distribution: 'normal' or 't'
            df: Degrees of freedom for t-distribution

        Returns:
            CVaR value (positive = loss)
        """
        if distribution == 'normal':
            z_alpha = stats.norm.ppf(self.alpha)
            phi_z = stats.norm.pdf(z_alpha)

            # Expected shortfall formula
            es_return = mean_return - std_return * phi_z / self.alpha
            cvar = -es_return * self.position_value

        elif distribution == 't':
            if df is None:
                raise ValueError("Degrees of freedom required for t-distribution")

            t_alpha = stats.t.ppf(self.alpha, df)
            pdf_t = stats.t.pdf(t_alpha, df)

            # ES for t-distribution
            adjustment = (df + t_alpha**2) / (df - 1)
            es_return = mean_return - std_return * pdf_t * adjustment / self.alpha
            cvar = -es_return * self.position_value

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return cvar

    def var_monte_carlo(
        self,
        simulated_values: np.ndarray,
        current_value: Optional[float] = None
    ) -> float:
        """
        Calculate VaR from Monte Carlo simulation results.

        Args:
            simulated_values: Array of simulated future position values
            current_value: Current position value (use self.position_value if None)

        Returns:
            VaR value (positive = loss)
        """
        if current_value is None:
            current_value = self.position_value

        # Calculate P&L distribution
        pnl = simulated_values - current_value

        # VaR is the α-quantile of losses
        var = -np.percentile(pnl, self.alpha * 100)

        return var

    def cvar_monte_carlo(
        self,
        simulated_values: np.ndarray,
        current_value: Optional[float] = None
    ) -> float:
        """
        Calculate CVaR from Monte Carlo simulation results.

        Args:
            simulated_values: Array of simulated future position values
            current_value: Current position value (use self.position_value if None)

        Returns:
            CVaR value (positive = loss)
        """
        if current_value is None:
            current_value = self.position_value

        # Calculate P&L distribution
        pnl = simulated_values - current_value

        # Find VaR threshold
        var = -np.percentile(pnl, self.alpha * 100)

        # CVaR is the expected loss beyond VaR
        losses = -pnl[pnl <= -var]
        cvar = np.mean(losses) if len(losses) > 0 else var

        return cvar

    def risk_summary(
        self,
        method: Literal['historical', 'parametric', 'monte_carlo'],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.

        Args:
            method: Risk calculation method
            **kwargs: Additional arguments for chosen method

        Returns:
            Dictionary with VaR, CVaR, and other metrics
        """
        if method == 'historical':
            var = self.var_historical(**kwargs)
            cvar = self.cvar_historical(**kwargs)

            returns = kwargs.get('returns', self.returns)
            pnl = self.position_value * returns
            max_loss = -np.min(pnl)
            avg_loss = -np.mean(pnl[pnl < 0]) if np.any(pnl < 0) else 0

        elif method == 'parametric':
            var = self.var_parametric(**kwargs)
            cvar = self.cvar_parametric(**kwargs)
            max_loss = None  # Not defined for parametric
            avg_loss = None

        elif method == 'monte_carlo':
            simulated_values = kwargs.get('simulated_values')
            current_value = kwargs.get('current_value', self.position_value)

            var = self.var_monte_carlo(simulated_values, current_value)
            cvar = self.cvar_monte_carlo(simulated_values, current_value)

            pnl = simulated_values - current_value
            max_loss = -np.min(pnl)
            avg_loss = -np.mean(pnl[pnl < 0]) if np.any(pnl < 0) else 0

        else:
            raise ValueError(f"Unknown method: {method}")

        result = {
            'position_value': self.position_value,
            'confidence_level': self.confidence_level,
            'var': var,
            'cvar': cvar,
            'var_pct': var / self.position_value * 100,
            'cvar_pct': cvar / self.position_value * 100,
        }

        if max_loss is not None:
            result['max_loss'] = max_loss
            result['avg_loss'] = avg_loss

        return result

    def portfolio_var(
        self,
        position_values: np.ndarray,
        returns_matrix: np.ndarray,
        method: Literal['historical', 'parametric'] = 'historical'
    ) -> Dict[str, float]:
        """
        Calculate portfolio VaR considering correlations.

        Args:
            position_values: Array of position values for each asset
            returns_matrix: Matrix of returns (n_observations, n_assets)
            method: Calculation method

        Returns:
            Dictionary with portfolio VaR and component VaR
        """
        n_assets = len(position_values)

        if method == 'historical':
            # Calculate portfolio P&L for each scenario
            portfolio_pnl = returns_matrix @ position_values

            # Portfolio VaR
            portfolio_var = -np.percentile(portfolio_pnl, self.alpha * 100)

            # Component VaR (incremental VaR)
            component_vars = []
            for i in range(n_assets):
                # Remove asset i
                mask = np.ones(n_assets, dtype=bool)
                mask[i] = False

                if np.sum(mask) > 0:
                    reduced_pnl = returns_matrix[:, mask] @ position_values[mask]
                    reduced_var = -np.percentile(reduced_pnl, self.alpha * 100)
                    component_vars.append(portfolio_var - reduced_var)
                else:
                    component_vars.append(portfolio_var)

        elif method == 'parametric':
            # Covariance matrix
            cov_matrix = np.cov(returns_matrix.T)

            # Portfolio variance
            portfolio_variance = position_values @ cov_matrix @ position_values
            portfolio_std = np.sqrt(portfolio_variance)

            # Portfolio VaR (assuming normal distribution)
            z_alpha = stats.norm.ppf(self.alpha)
            portfolio_var = -z_alpha * portfolio_std

            # Component VaR using marginal VaR
            marginal_vars = cov_matrix @ position_values / portfolio_std
            component_vars = -z_alpha * marginal_vars

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'portfolio_var': portfolio_var,
            'component_vars': component_vars,
            'total_position': np.sum(position_values)
        }

    def backtest_var(
        self,
        realized_returns: np.ndarray,
        var_forecasts: np.ndarray
    ) -> Dict[str, float]:
        """
        Backtest VaR model using realized returns.

        Tests:
            1. Coverage test: Are violations = expected frequency?
            2. Independence test: Are violations clustered?

        Args:
            realized_returns: Actual historical returns
            var_forecasts: VaR forecasts for each period

        Returns:
            Dictionary with backtest statistics
        """
        # Calculate realized losses
        realized_losses = -self.position_value * realized_returns

        # Count VaR violations (losses exceeding VaR)
        violations = realized_losses > var_forecasts
        n_violations = np.sum(violations)
        n_periods = len(realized_returns)

        # Expected violations under correct model
        expected_violations = n_periods * self.alpha

        # Violation rate
        violation_rate = n_violations / n_periods

        # Kupiec test (likelihood ratio test for correct coverage)
        if n_violations > 0 and n_violations < n_periods:
            p = n_violations / n_periods
            lr_stat = -2 * (
                n_violations * np.log(self.alpha / p) +
                (n_periods - n_violations) * np.log((1 - self.alpha) / (1 - p))
            )
            # Under null, LR ~ χ²(1)
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        else:
            lr_stat = None
            p_value = None

        return {
            'n_periods': n_periods,
            'n_violations': n_violations,
            'expected_violations': expected_violations,
            'violation_rate': violation_rate,
            'expected_rate': self.alpha,
            'kupiec_lr_stat': lr_stat,
            'kupiec_p_value': p_value
        }

    def __repr__(self) -> str:
        return (f"RiskMetrics(position_value={self.position_value:.2f}, "
                f"confidence_level={self.confidence_level:.2%})")
