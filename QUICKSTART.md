# Quick Start Guide

Get up and running with the Option Pricing Engine in 5 minutes.

## Installation

```bash
cd pricing_engine
pip install -r requirements.txt
```

## Run the Example

```bash
python example.py
```

This will demonstrate:
- All three pricing methods (BS, PDE, MC)
- Greeks calculation
- Implied volatility calibration
- Risk metrics (VaR/CVaR)

## Interactive Jupyter Notebooks

```bash
cd notebooks
jupyter notebook
```

Open and run:
1. `01_pricing_comparison.ipynb` - Compare pricing methods
2. `02_greeks_and_risk.ipynb` - Greeks and risk analysis

## Quick Code Examples

### 1. Price a Call Option

```python
from pricers.black_scholes import BlackScholesPricer

bs = BlackScholesPricer(
    S=100,      # Spot price
    K=100,      # Strike price
    T=1.0,      # 1 year to maturity
    r=0.05,     # 5% risk-free rate
    sigma=0.2   # 20% volatility
)

price = bs.price('call')
print(f"Call Price: ${price:.2f}")
# Output: Call Price: $10.45
```

### 2. Calculate Greeks

```python
greeks = bs.all_greeks('call')

print(f"Delta: {greeks['delta']:.4f}")   # ~0.6368
print(f"Gamma: {greeks['gamma']:.6f}")   # ~0.018605
print(f"Vega:  {greeks['vega']:.4f}")    # ~0.3969
```

### 3. PDE Pricing (Crank-Nicolson)

```python
from pricers.pde_pricer import PDEPricer

pde = PDEPricer(S=100, K=100, T=1.0, r=0.05, sigma=0.2, M=100, N=1000)
price = pde.price('call', method='crank-nicolson')

print(f"PDE Price: ${price:.4f}")
# Should match BS within ~0.01%
```

### 4. Monte Carlo with Variance Reduction

```python
from pricers.monte_carlo import MonteCarloPricer

mc = MonteCarloPricer(
    S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    n_simulations=100000,
    seed=42
)

result = mc.price('call', method='antithetic')

print(f"MC Price: ${result['price']:.4f} ± ${result['std_error']:.4f}")
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
```

### 5. Implied Volatility

```python
from calibration.implied_vol import ImpliedVolatility

# Suppose we observe market price of $10.45
market_price = 10.45

iv_solver = ImpliedVolatility(S=100, K=100, T=1.0, r=0.05)
implied_vol = iv_solver.solve('call', market_price, method='newton')

print(f"Implied Volatility: {implied_vol:.2%}")
# Output: Implied Volatility: 20.00%
```

### 6. Risk Metrics (VaR & CVaR)

```python
from risk.var_cvar import RiskMetrics
import numpy as np

# Position: $1,000 option portfolio
position_value = 1000

# Simulate future values (simplified)
returns = np.random.normal(0, 0.02, 10000)  # 2% daily volatility

risk = RiskMetrics(position_value, confidence_level=0.95)
var = risk.var_historical(returns)
cvar = risk.cvar_historical(returns)

print(f"95% VaR:  ${var:.2f}")
print(f"95% CVaR: ${cvar:.2f}")
```

## Understanding the Results

### Pricing Method Comparison

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Black-Scholes** | Exact, ultra-fast | European only | Vanilla options |
| **PDE** | Very accurate, stable | Moderate speed | American, barriers |
| **Monte Carlo** | Most flexible | Slowest | Path-dependent |

### Typical Errors

With standard parameters (M=100, N=1000 for PDE; N=100k for MC):
- PDE (Crank-Nicolson): < 0.01% error
- Monte Carlo (antithetic): < 0.05% error

### Greeks Interpretation

- **Delta (Δ)**: Change in option price per $1 change in stock
  - Call: 0 to 1
  - Put: -1 to 0

- **Gamma (Γ)**: Change in delta per $1 change in stock
  - Highest for ATM options near expiry

- **Vega (ν)**: Change in option price per 1% change in volatility
  - Highest for ATM options with longer maturity

- **Theta (Θ)**: Time decay per day
  - Usually negative (options lose value over time)

## Next Steps

1. **Explore Notebooks**: Run Jupyter notebooks for detailed analysis
2. **Read Report**: See `report/capstone.pdf` for mathematical derivations
3. **Experiment**: Modify parameters and observe behavior
4. **Extend**: Add new features (see README for ideas)

## Common Issues

### Import Errors
Make sure you're in the correct directory:
```bash
cd pricing_engine
python example.py
```

### Slow Monte Carlo
Reduce simulations for testing:
```python
mc = MonteCarloPricer(..., n_simulations=10000)  # Faster
```

### PDE Stability Warning
Increase grid size or decrease time steps:
```python
pde = PDEPricer(..., M=200, N=2000)  # More stable
```

## Getting Help

- Check `README.md` for detailed documentation
- Review code docstrings (they include math!)
- Examine test cases in `tests/`
- Read LaTeX report in `report/`

## Key Formulas

**Black-Scholes Call Price:**
```
C = S·e^(-qT)·Φ(d₁) - K·e^(-rT)·Φ(d₂)

where:
d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Monte Carlo Estimator:**
```
V ≈ e^(-rT) · (1/N) · Σ Payoff(S_T)

Standard Error ∝ 1/√N
```

**VaR Definition:**
```
VaR_α = -inf{x : P(Loss ≤ x) ≥ α}

95% VaR: 5% chance of exceeding this loss
```

---

**Happy pricing! 📈**
