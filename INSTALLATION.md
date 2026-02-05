# Installation Guide

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Step 1: Navigate to Project Directory

```bash
cd /Users/pierrevong/pricing_engine
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy scipy pandas matplotlib seaborn jupyter notebook
```

## Step 3: Verify Installation

```bash
python3 verify_installation.py
```

You should see:
```
✓ ALL CHECKS PASSED!
```

## Step 4: Run Example

```bash
python3 example.py
```

This will demonstrate all features of the pricing engine.

## Step 5: Explore Notebooks

```bash
cd notebooks
jupyter notebook
```

Open and run:
- `01_pricing_comparison.ipynb`
- `02_greeks_and_risk.ipynb`

## Step 6: Run Tests (Optional)

```bash
pip install pytest  # if not already installed
python3 -m pytest tests/test_basic.py -v
```

## Step 7: Compile LaTeX Report (Optional)

```bash
cd report
pdflatex capstone.tex
pdflatex capstone.tex  # Run twice for references
```

Requires LaTeX distribution (e.g., TeXLive, MiKTeX).

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd /Users/pierrevong/pricing_engine
python3 example.py
```

### Missing Packages

Install any missing package individually:

```bash
pip install <package-name>
```

### Permission Errors

Use `--user` flag:

```bash
pip install --user -r requirements.txt
```

### Virtual Environment (Recommended)

Create isolated environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Test

After installation, try this in Python:

```python
from pricers.black_scholes import BlackScholesPricer

bs = BlackScholesPricer(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
print(f"Call price: ${bs.price('call'):.4f}")
# Should output: Call price: $10.4506
```

If this works, you're ready to go!

## Next Steps

1. Read [QUICKSTART.md](QUICKSTART.md) for quick examples
2. Read [README.md](README.md) for detailed documentation
3. Run `example.py` for comprehensive demonstration
4. Explore Jupyter notebooks for interactive analysis
5. Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for project overview

## Support

For issues or questions:
1. Check code docstrings (they include mathematical explanations)
2. Review example.py for usage patterns
3. Examine test cases in tests/test_basic.py
4. Read the LaTeX report in report/capstone.tex
