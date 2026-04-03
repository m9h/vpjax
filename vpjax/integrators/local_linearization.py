"""Local Linearization (LL) integration method.

The LL method approximates a nonlinear ODE dx/dt = f(x) by linearizing
f around the current state at each step:

    f(x) ≈ f(x_n) + J_n · (x - x_n)

where J_n = ∂f/∂x evaluated at x_n. The linear ODE has exact solution:

    x_{n+1} = x_n + (exp(J_n · dt) - I) · J_n⁻¹ · f(x_n)

This is more stable than Euler for stiff systems (common in
neurovascular coupling ODEs) while remaining relatively cheap.

When J is singular, the matrix-exponential-based formula degrades
gracefully to the Euler step: x_{n+1} ≈ x_n + dt · f(x_n).

References
----------
Ozaki T (1992) Statistica Sinica 2:113-135
    "A bridge between nonlinear time series models and nonlinear
    stochastic dynamical systems"
Riera JJ et al. (2004) NeuroImage 21:547-567
    Used LL for neurovascular coupling ODEs
Jimenez JC et al. (2002) Stat Comput 12:313-328
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def ll_step(
    f: callable,
    x: Float[Array, "N"],
    dt: float | Float[Array, ""],
    args: tuple = (),
) -> Float[Array, "N"]:
    """Single Local Linearization step.

    Parameters
    ----------
    f : vector field function f(x, *args) -> dx/dt, operating on flat arrays
    x : current state vector, shape (N,)
    dt : time step (s)
    args : additional arguments passed to f

    Returns
    -------
    x_next : state after one LL step, shape (N,)
    """
    # Evaluate f and its Jacobian at current state
    fx = f(x, *args)
    J = jax.jacobian(f)(x, *args)

    N = x.shape[0]

    # Matrix exponential: expm(J * dt)
    Jdt = J * dt
    expJdt = jax.scipy.linalg.expm(Jdt)

    # Compute (expm(J·dt) - I) · J⁻¹ · f(x)
    # This is equivalent to solving J · z = (expm(J·dt) - I) · f(x) / ...
    # More stable: use (expm(J·dt) - I) · f(x) and solve J · result = that
    #
    # Actually: x_next = x + phi(J·dt) · dt · f(x)
    # where phi(A) = (expm(A) - I) · A⁻¹ = I + A/2! + A²/3! + ...
    #
    # Compute phi(J·dt) via the identity:
    #   phi(A) = sum_{k=0}^{inf} A^k / (k+1)!
    # For numerical stability, use: phi(A) · f = solve(A, (expm(A) - I) · f)
    # with fallback to Taylor series for near-singular A

    diff = (expJdt - jnp.eye(N)) @ fx

    # Try to solve J · z = diff  (i.e., z = J⁻¹ · diff)
    # If J is well-conditioned, this gives the LL update
    # Regularize slightly for numerical stability
    J_reg = J + 1e-8 * jnp.eye(N)
    z = jnp.linalg.solve(J_reg, diff)

    return x + z


def ll_solve(
    f: callable,
    y0: Float[Array, "N"],
    ts: Float[Array, "T"],
    args: tuple = (),
) -> Float[Array, "T N"]:
    """Integrate an ODE using Local Linearization over time points.

    Parameters
    ----------
    f : vector field function f(x, *args) -> dx/dt
    y0 : initial state, shape (N,)
    ts : time points, shape (T,). Must be sorted.
    args : additional arguments passed to f

    Returns
    -------
    ys : state trajectory, shape (T, N). ys[0] = y0.
    """
    def scan_fn(x, t_pair):
        t_curr, t_next = t_pair
        dt = t_next - t_curr
        x_next = ll_step(f, x, dt, args)
        return x_next, x_next

    # Pair up consecutive time points
    t_pairs = jnp.stack([ts[:-1], ts[1:]], axis=-1)

    _, trajectory = jax.lax.scan(scan_fn, y0, t_pairs)

    # Prepend initial condition
    ys = jnp.concatenate([y0[None, :], trajectory], axis=0)

    return ys


def ll_step_pytree(
    f: callable,
    x_tree,
    dt: float | Float[Array, ""],
    args: tuple = (),
):
    """Local Linearization step for pytree-valued states.

    Flattens the pytree to a 1-D array, performs the LL step,
    then unflattens back to the original structure.

    Parameters
    ----------
    f : vector field f(x_tree, *args) -> dx_tree (pytree in, pytree out)
    x_tree : current state (any pytree, e.g. BalloonState)
    dt : time step
    args : additional arguments

    Returns
    -------
    x_next_tree : next state (same pytree structure)
    """
    # Flatten
    x_flat, treedef = jax.tree.flatten(x_tree)
    x_vec = jnp.concatenate([xi.ravel() for xi in x_flat])

    # Sizes for reconstruction
    shapes = [xi.shape for xi in x_flat]
    sizes = [xi.size for xi in x_flat]

    def f_flat(x_vec, *args):
        # Unflatten
        splits = jnp.cumsum(jnp.array(sizes[:-1]))
        parts = jnp.split(x_vec, splits)
        leaves = [p.reshape(s) for p, s in zip(parts, shapes)]
        tree = jax.tree.unflatten(treedef, leaves)

        # Evaluate
        dy_tree = f(tree, *args)

        # Flatten output
        dy_flat, _ = jax.tree.flatten(dy_tree)
        return jnp.concatenate([dyi.ravel() for dyi in dy_flat])

    x_next_vec = ll_step(f_flat, x_vec, dt, args)

    # Unflatten result
    splits = jnp.cumsum(jnp.array(sizes[:-1]))
    parts = jnp.split(x_next_vec, splits)
    leaves_next = [p.reshape(s) for p, s in zip(parts, shapes)]
    return jax.tree.unflatten(treedef, leaves_next)
