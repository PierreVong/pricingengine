# Multi-Method Option Pricing & Risk Engine

A production-quality option pricing engine implementing stochastic calculus, PDEs, and Monte Carlo methods for derivatives valuation and risk analysis.

## 🎯 Project Overview

This project addresses the fundamental question: **How do different mathematical pricing methods compare in accuracy, speed, and risk sensitivity?**

### Key Features

- **Three Independent Pricing Frameworks:**
  - Black-Scholes analytical (closed-form solutions)
  - PDE finite differences (explicit, implicit, Crank-Nicolson)
  - Monte Carlo simulation (with variance reduction)

- **Comprehensive Greeks:**
  - Delta, Gamma, Vega, Theta, Rho
  - Analytical and numerical computation
  - Validation and comparison tools

- **Risk Analytics:**
  - Value at Risk (VaR)
  - Conditional VaR (CVaR/Expected Shortfall)
  - Delta hedging simulation
  - P&L distribution analysis

- **Model Calibration:**
  - Implied volatility solver (Newton-Raphson, Bisection, Brent)
  - Volatility smile/surface analysis

## 📁 Project Structure

```
pricing_engine/
│
├── models/                 # Asset price models
│   ├── __init__.py
│   └── gbm.py             # Geometric Brownian Motion
│
├── pricers/               # Option pricing methods
│   ├── __init__.py
│   ├── black_scholes.py   # Analytical Black-Scholes
│   ├── pde_pricer.py      # Finite difference PDE solver
│   └── monte_carlo.py     # Monte Carlo with variance reduction
│
├── greeks/                # Sensitivity analysis
│   ├── __init__.py
│   └── finite_diff.py     # Numerical Greeks calculator
│
├── calibration/           # Model calibration
│   ├── __init__.py
│   └── implied_vol.py     # Implied volatility solver
│
├── risk/                  # Risk metrics
│   ├── __init__.py
│   └── var_cvar.py        # VaR and CVaR calculation
│
├── notebooks/             # Jupyter analysis notebooks
│   ├── 01_pricing_comparison.ipynb
│   └── 02_greeks_and_risk.ipynb
│
├── report/                # LaTeX report
│   └── capstone.tex
│
├── tests/                 # Unit tests
│
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── __init__.py           # Package initialization
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pricing_engine

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from pricing_engine import (
    BlackScholesPricer,
    PDEPricer,
    MonteCarloPricer
)

# Option parameters
S = 100.0    # Spot price
K = 100.0    # Strike (ATM)
T = 1.0      # 1 year to maturity
r = 0.05     # 5% risk-free rate
sigma = 0.2  # 20% volatility

# Black-Scholes (analytical)
bs = BlackScholesPricer(S, K, T, r, sigma)
bs_price = bs.price('call')
bs_greeks = bs.all_greeks('call')

print(f"BS Price: ${bs_price:.4f}")
print(f"Delta: {bs_greeks['delta']:.4f}")
print(f"Gamma: {bs_greeks['gamma']:.6f}")

# PDE Method (Crank-Nicolson)
pde = PDEPricer(S, K, T, r, sigma, M=100, N=1000)
pde_price = pde.price('call', method='crank-nicolson')

print(f"PDE Price: ${pde_price:.4f}")
print(f"Error: {abs(pde_price - bs_price):.6f}")

# Monte Carlo (antithetic variates)
mc = MonteCarloPricer(S, K, T, r, sigma, n_simulations=100000)
mc_result = mc.price('call', method='antithetic')

print(f"MC Price: ${mc_result['price']:.4f} ± {mc_result['std_error']:.4f}")
print(f"95% CI: [{mc_result['ci_lower']:.4f}, {mc_result['ci_upper']:.4f}]")
```

## 📊 Jupyter Notebooks

### 1. Pricing Comparison (`01_pricing_comparison.ipynb`)
- Compare all three pricing methods
- Convergence analysis
- Speed vs accuracy trade-offs
- Moneyness sensitivity

### 2. Greeks and Risk Analysis (`02_greeks_and_risk.ipynb`)
- Greeks surface visualization
- Delta hedging simulation
- P&L distribution
- VaR and CVaR calculation

Run notebooks:
```bash
cd notebooks
jupyter notebook
```

## 🔬 Key Results

### Pricing Accuracy
| Method | Price | Error (%) | Time (ms) |
|--------|-------|-----------|-----------|
| Black-Scholes | 10.4506 | --- | 0.05 |
| PDE (CN) | 10.4503 | 0.003% | 15.8 |
| MC (Antithetic) | 10.4498 | 0.008% | 118.7 |

*Parameters: S=100, K=100, T=1y, r=5%, σ=20%*

### Variance Reduction Effectiveness
- **Antithetic variates:** 30-40% variance reduction
- **Control variates:** 50-70% variance reduction

### Greeks Validation
- Delta: < 0.1% error vs analytical
- Gamma: < 0.5% error (second derivative)
- Vega: < 0.2% error

## 📈 Mathematical Framework

### Black-Scholes PDE
```
∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0
```

### Call Option Price
```
C = S₀e^(-qτ)Φ(d₁) - Ke^(-rτ)Φ(d₂)

where:
d₁ = [ln(S₀/K) + (r - q + σ²/2)τ] / (σ√τ)
d₂ = d₁ - σ√τ
```

### Monte Carlo Estimator
```
V ≈ e^(-rT) × (1/N) × Σ Payoff(S_T^(i))

Standard Error ∝ 1/√N
```

## 🛠️ Technical Highlights

### Software Engineering
- **Clean architecture:** Modular design with separation of concerns
- **Type safety:** Full type hints throughout codebase
- **Documentation:** Comprehensive docstrings with mathematical context
- **Vectorization:** NumPy for computational efficiency
- **Sparse matrices:** SciPy for PDE linear systems

### Numerical Stability
- PDE stability warnings (explicit scheme)
- Arbitrage checks in implied volatility solver
- Input validation and error handling
- Convergence tolerance monitoring

## 📚 LaTeX Report

Comprehensive 20+ page report covering:
1. Mathematical derivations from first principles
2. Numerical methods analysis
3. Implementation details
4. Experimental results
5. Model risk and limitations
6. Extensions and future work

Compile report:
```bash
cd report
pdflatex capstone.tex
```

## 🎓 Use Cases

### For Students
- Master option pricing theory
- Learn production-quality code practices
- Understand numerical method trade-offs
- Prepare for quant finance interviews

### For Practitioners
- Benchmark proprietary pricing systems
- Validate Greeks computation
- Analyze model risk
- Prototype new pricing strategies

## 🚧 Extensions

### Immediate Improvements
1. **Local volatility:** Dupire equation implementation
2. **Stochastic volatility:** Heston model
3. **American options:** Free boundary PDE solver
4. **Exotic payoffs:** Barrier, Asian, lookback options

### Advanced Features
1. Market data integration (real-time pricing)
2. Volatility surface calibration
3. Multi-asset correlation
4. Credit risk adjustments (CVA/DVA)

## 📖 References

1. Hull, J.C. (2018). *Options, Futures, and Other Derivatives*
2. Shreve, S.E. (2004). *Stochastic Calculus for Finance II*
3. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*
4. Wilmott, P. et al. (1995). *The Mathematics of Financial Derivatives*

## 📄 License

This project is for educational purposes. See LICENSE for details.

## 👤 Author

Pierre Vong

---

**Keywords:** quantitative finance, option pricing, Black-Scholes, PDE, Monte Carlo, Greeks, VaR, CVaR, risk management, derivatives, numerical methods, stochastic calculus

