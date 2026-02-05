"""
Multi-Method Option Pricing & Risk Engine

A production-quality option pricing engine implementing:
- Stochastic calculus (GBM simulation)
- PDE methods (finite differences)
- Monte Carlo methods (with variance reduction)
- Greeks computation
- Risk metrics (VaR, CVaR)

Author: MFE Capstone Project
"""

__version__ = "1.0.0"

from .models import GBMModel
from .pricers import BlackScholesPricer, PDEPricer, MonteCarloPricer
from .greeks import GreeksCalculator
from .calibration import ImpliedVolatility
from .risk import RiskMetrics

__all__ = [
    'GBMModel',
    'BlackScholesPricer',
    'PDEPricer',
    'MonteCarloPricer',
    'GreeksCalculator',
    'ImpliedVolatility',
    'RiskMetrics'
]
