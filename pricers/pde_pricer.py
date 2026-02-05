"""
PDE-Based Option Pricing using Finite Difference Methods

Solves the Black-Scholes PDE:
    ∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0

Backward in time (τ = T - t):
    ∂V/∂τ = ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV

Three numerical schemes implemented:
    1. Explicit: Forward difference in time, central in space
    2. Implicit: Backward difference in time, central in space
    3. Crank-Nicolson: Average of explicit and implicit (more stable)

Boundary conditions:
    Call:
        - S = 0: V = 0
        - S → ∞: V ≈ S - K*e^(-r*τ)
        - τ = 0: V = max(S - K, 0)
    Put:
        - S = 0: V = K*e^(-r*τ)
        - S → ∞: V ≈ 0
        - τ = 0: V = max(K - S, 0)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Literal, Tuple, Optional


class PDEPricer:
    """
    Finite difference solver for Black-Scholes PDE.

    Implements three numerical schemes with varying stability and accuracy properties.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        S_max: Optional[float] = None,
        M: int = 100,
        N: int = 1000
    ):
        """
        Initialize PDE pricer with grid parameters.

        Args:
            S: Current asset price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
            S_max: Maximum asset price for grid (default: 3*max(S, K))
            M: Number of space steps (grid points in S)
            N: Number of time steps
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

        # Grid parameters
        self.S_max = S_max if S_max is not None else 3 * max(S, K)
        self.M = M
        self.N = N

        # Grid setup
        self.dS = self.S_max / M
        self.dt = T / N

        # Asset price grid
        self.S_grid = np.linspace(0, self.S_max, M + 1)

        # Stability check for explicit method
        self.max_sigma_dt_dS2 = self.sigma**2 * self.dt / self.dS**2
        if self.max_sigma_dt_dS2 > 0.5:
            import warnings
            warnings.warn(
                f"Explicit scheme may be unstable: σ²Δt/ΔS² = {self.max_sigma_dt_dS2:.4f} > 0.5. "
                f"Consider increasing M or decreasing N."
            )

    def _payoff(self, option_type: Literal['call', 'put']) -> np.ndarray:
        """
        Calculate terminal payoff at maturity.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Array of payoff values on the grid
        """
        if option_type == 'call':
            return np.maximum(self.S_grid - self.K, 0)
        elif option_type == 'put':
            return np.maximum(self.K - self.S_grid, 0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def _boundary_conditions(
        self,
        option_type: Literal['call', 'put'],
        tau: float
    ) -> Tuple[float, float]:
        """
        Calculate boundary conditions at S=0 and S=S_max.

        Args:
            option_type: 'call' or 'put'
            tau: Time from expiration (T - t)

        Returns:
            Tuple of (V_lower, V_upper) boundary values
        """
        discount = np.exp(-self.r * tau)

        if option_type == 'call':
            V_lower = 0.0  # S = 0
            V_upper = self.S_max - self.K * discount  # S → ∞
        elif option_type == 'put':
            V_lower = self.K * discount  # S = 0
            V_upper = 0.0  # S → ∞
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        return V_lower, V_upper

    def price_explicit(self, option_type: Literal['call', 'put']) -> float:
        """
        Price option using explicit finite difference scheme.

        Scheme:
            V[i,n+1] = V[i,n] + Δt * (½σ²S²V''[i,n] + (r-q)SV'[i,n] - rV[i,n])

        Stability condition: σ²Δt/ΔS² ≤ 0.5

        Args:
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        # Initialize with terminal payoff
        V = self._payoff(option_type)

        # Coefficients for interior points
        j = np.arange(1, self.M)
        S_j = self.S_grid[j]

        a_j = 0.5 * self.dt * ((self.r - self.q) * S_j / self.dS - self.sigma**2 * S_j**2 / self.dS**2)
        b_j = 1 - self.dt * (self.sigma**2 * S_j**2 / self.dS**2 + self.r)
        c_j = 0.5 * self.dt * ((self.r - self.q) * S_j / self.dS + self.sigma**2 * S_j**2 / self.dS**2)

        # Step backward in time
        for n in range(self.N):
            tau = (n + 1) * self.dt
            V_lower, V_upper = self._boundary_conditions(option_type, tau)

            V_new = np.zeros(self.M + 1)
            V_new[0] = V_lower
            V_new[self.M] = V_upper

            # Update interior points
            V_new[j] = a_j * V[j-1] + b_j * V[j] + c_j * V[j+1]

            V = V_new

        # Interpolate to get value at current spot price
        return np.interp(self.S, self.S_grid, V)

    def price_implicit(self, option_type: Literal['call', 'put']) -> float:
        """
        Price option using implicit finite difference scheme.

        Scheme (solving system A*V[n+1] = V[n]):
            Unconditionally stable, more computationally expensive than explicit.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        # Initialize with terminal payoff
        V = self._payoff(option_type)

        # Build coefficient matrix for interior points
        j = np.arange(1, self.M)
        S_j = self.S_grid[j]

        a_j = -0.5 * self.dt * ((self.r - self.q) * S_j / self.dS - self.sigma**2 * S_j**2 / self.dS**2)
        b_j = 1 + self.dt * (self.sigma**2 * S_j**2 / self.dS**2 + self.r)
        c_j = -0.5 * self.dt * ((self.r - self.q) * S_j / self.dS + self.sigma**2 * S_j**2 / self.dS**2)

        # Create tridiagonal matrix
        diagonals = [a_j[1:], b_j, c_j[:-1]]
        A = diags(diagonals, [-1, 0, 1], shape=(self.M - 1, self.M - 1), format='csr')

        # Step backward in time
        for n in range(self.N):
            tau = (n + 1) * self.dt
            V_lower, V_upper = self._boundary_conditions(option_type, tau)

            # Right-hand side
            rhs = V[1:self.M].copy()
            rhs[0] -= a_j[0] * V_lower
            rhs[-1] -= c_j[-1] * V_upper

            # Solve linear system
            V_interior = spsolve(A, rhs)

            # Update grid
            V = np.zeros(self.M + 1)
            V[0] = V_lower
            V[1:self.M] = V_interior
            V[self.M] = V_upper

        # Interpolate to get value at current spot price
        return np.interp(self.S, self.S_grid, V)

    def price_crank_nicolson(self, option_type: Literal['call', 'put']) -> float:
        """
        Price option using Crank-Nicolson scheme.

        Scheme (θ = 0.5 weighted average of explicit and implicit):
            Superior stability and accuracy (O(Δt², ΔS²))

        Args:
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        # Initialize with terminal payoff
        V = self._payoff(option_type)

        # Coefficients
        j = np.arange(1, self.M)
        S_j = self.S_grid[j]

        alpha = 0.25 * self.dt * (self.sigma**2 * S_j**2 / self.dS**2 - (self.r - self.q) * S_j / self.dS)
        beta = -0.5 * self.dt * (self.sigma**2 * S_j**2 / self.dS**2 + self.r)
        gamma = 0.25 * self.dt * (self.sigma**2 * S_j**2 / self.dS**2 + (self.r - self.q) * S_j / self.dS)

        # Left-hand side matrix (implicit part)
        a_lhs = -alpha
        b_lhs = 1 - beta
        c_lhs = -gamma

        M1 = diags([a_lhs[1:], b_lhs, c_lhs[:-1]], [-1, 0, 1], shape=(self.M - 1, self.M - 1), format='csr')

        # Right-hand side matrix (explicit part)
        a_rhs = alpha
        b_rhs = 1 + beta
        c_rhs = gamma

        M2 = diags([a_rhs[1:], b_rhs, c_rhs[:-1]], [-1, 0, 1], shape=(self.M - 1, self.M - 1), format='csr')

        # Step backward in time
        for n in range(self.N):
            tau = (n + 1) * self.dt
            V_lower, V_upper = self._boundary_conditions(option_type, tau)

            # Right-hand side
            rhs = M2.dot(V[1:self.M])
            rhs[0] += alpha[0] * (V_lower + V[0])
            rhs[-1] += gamma[-1] * (V_upper + V[self.M])

            # Solve linear system
            V_interior = spsolve(M1, rhs)

            # Update grid
            V = np.zeros(self.M + 1)
            V[0] = V_lower
            V[1:self.M] = V_interior
            V[self.M] = V_upper

        # Interpolate to get value at current spot price
        return np.interp(self.S, self.S_grid, V)

    def price(
        self,
        option_type: Literal['call', 'put'],
        method: Literal['explicit', 'implicit', 'crank-nicolson'] = 'crank-nicolson'
    ) -> float:
        """
        Price option using specified finite difference method.

        Args:
            option_type: 'call' or 'put'
            method: Numerical scheme to use (default: 'crank-nicolson')

        Returns:
            Option price

        Raises:
            ValueError: If method is not recognized
        """
        if method == 'explicit':
            return self.price_explicit(option_type)
        elif method == 'implicit':
            return self.price_implicit(option_type)
        elif method == 'crank-nicolson':
            return self.price_crank_nicolson(option_type)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'explicit', 'implicit', or 'crank-nicolson'")

    def get_full_grid(
        self,
        option_type: Literal['call', 'put'],
        method: Literal['explicit', 'implicit', 'crank-nicolson'] = 'crank-nicolson'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get full solution grid for visualization.

        Args:
            option_type: 'call' or 'put'
            method: Numerical scheme to use

        Returns:
            Tuple of (S_grid, time_grid, V_grid)
                - S_grid: Asset prices (M+1,)
                - time_grid: Time points (N+1,)
                - V_grid: Option values (M+1, N+1)
        """
        # This is a simplified version - full implementation would store all time steps
        raise NotImplementedError("Full grid output not yet implemented")

    def __repr__(self) -> str:
        return (f"PDEPricer(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, sigma={self.sigma:.4f}, M={self.M}, N={self.N})")
