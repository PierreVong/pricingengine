#!/usr/bin/env python3
"""
Verification script to ensure the pricing engine is properly installed.

Run this script after installation to verify all components work correctly.
"""

import sys
import importlib.util

def check_module(module_name):
    """Check if a module can be imported."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def main():
    print("=" * 70)
    print(" PRICING ENGINE INSTALLATION VERIFICATION")
    print("=" * 70)

    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print("✓ Python version OK")

    # Check required packages
    print("\nChecking required packages...")
    required_packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
    }

    all_ok = True
    for module, name in required_packages.items():
        if check_module(module):
            print(f"✓ {name:15s} installed")
        else:
            print(f"❌ {name:15s} NOT FOUND - run: pip install {module}")
            all_ok = False

    if not all_ok:
        print("\n❌ Missing dependencies. Install with: pip install -r requirements.txt")
        return False

    # Test imports from pricing engine
    print("\nTesting pricing engine modules...")
    try:
        from models.gbm import GBMModel
        print("✓ models.gbm")

        from pricers.black_scholes import BlackScholesPricer
        print("✓ pricers.black_scholes")

        from pricers.pde_pricer import PDEPricer
        print("✓ pricers.pde_pricer")

        from pricers.monte_carlo import MonteCarloPricer
        print("✓ pricers.monte_carlo")

        from greeks.finite_diff import GreeksCalculator
        print("✓ greeks.finite_diff")

        from calibration.implied_vol import ImpliedVolatility
        print("✓ calibration.implied_vol")

        from risk.var_cvar import RiskMetrics
        print("✓ risk.var_cvar")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Quick functionality test
    print("\nRunning quick functionality test...")
    try:
        # Test Black-Scholes
        bs = BlackScholesPricer(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        price = bs.price('call')
        assert 10 < price < 11, f"Unexpected BS price: {price}"
        print(f"✓ Black-Scholes pricing: ${price:.4f}")

        # Test Greeks
        delta = bs.delta('call')
        assert 0 < delta < 1, f"Unexpected delta: {delta}"
        print(f"✓ Greeks calculation: Δ={delta:.4f}")

        # Test PDE
        pde = PDEPricer(S=100, K=100, T=1.0, r=0.05, sigma=0.2, M=50, N=500)
        pde_price = pde.price('call', method='crank-nicolson')
        error = abs(pde_price - price) / price
        assert error < 0.05, f"PDE error too large: {error:.2%}"
        print(f"✓ PDE pricing: ${pde_price:.4f} (error: {error:.2%})")

        # Test Monte Carlo
        mc = MonteCarloPricer(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                             n_simulations=10000, seed=42)
        mc_result = mc.price('call', method='antithetic')
        mc_error = abs(mc_result['price'] - price) / price
        assert mc_error < 0.10, f"MC error too large: {mc_error:.2%}"
        print(f"✓ Monte Carlo pricing: ${mc_result['price']:.4f} ± ${mc_result['std_error']:.4f}")

        # Test implied vol
        iv_solver = ImpliedVolatility(S=100, K=100, T=1.0, r=0.05)
        implied_vol = iv_solver.solve('call', price, method='newton')
        iv_error = abs(implied_vol - 0.2)
        assert iv_error < 0.001, f"IV recovery error: {iv_error}"
        print(f"✓ Implied volatility: {implied_vol:.4f} (true: 0.2000)")

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Success
    print("\n" + "=" * 70)
    print(" ✓ ALL CHECKS PASSED!")
    print("=" * 70)
    print("\nThe pricing engine is correctly installed and functional.")
    print("\nNext steps:")
    print("  1. Run example:        python example.py")
    print("  2. Run tests:          python -m pytest tests/ -v")
    print("  3. Open notebooks:     jupyter notebook notebooks/")
    print("  4. Read documentation: cat README.md")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
