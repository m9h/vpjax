# Contributing to vpjax

Thank you for your interest in contributing to vpjax! This project provides
differentiable cerebrovascular models in JAX, bridging computational
neuroscience and medical imaging.

## 1. Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/ins-amu/vpjax.git
cd vpjax

# Create a virtual environment and install dev dependencies
uv venv
uv pip install -e ".[dev,full]"

# Verify the installation
uv run pytest
```

### Optional dependency groups

| Group | Contents |
|-------|----------|
| `full` | lineax, optimistix |
| `angiography` | scikit-image, scipy |
| `validation` | mne, nibabel, scipy, matplotlib |
| `gpu` | JAX CUDA 12 support (Linux only) |
| `dev` | pytest, pytest-xdist, and all validation deps |
| `doc` | Sphinx, furo, myst-parser, sphinx-copybutton |

## 2. Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with tests (see Testing below).

3. **Run the test suite**:
   ```bash
   uv run pytest
   # Or in parallel:
   uv run pytest -n auto
   ```

4. **Build and check docs** (if you touched docstrings or tutorials):
   ```bash
   uv pip install -e ".[doc]"
   cd docs && make html
   ```

5. **Open a pull request** against `main`.

### Commit Messages

Use conventional commit style:

```
Add multi-compartment Riera model with NO/adenosine pathways
Fix oxygen extraction at low flow values
Update BOLD observation for multi-echo TE array
```

## 3. Code Style

### Docstrings

All public functions and classes use **NumPy-style docstrings**:

```python
def observe_bold(state, params=None):
    """Compute the BOLD signal change from a Balloon state.

    Maps venous volume and deoxyhemoglobin content to the
    T2*-weighted signal change using the Obata (2004) model.

    Parameters
    ----------
    state : BalloonState
        Current hemodynamic state (v, q must be populated).
    params : BOLDParams, optional
        BOLD observation parameters. Uses 3T defaults if None.

    Returns
    -------
    Float[Array, "..."]
        Fractional BOLD signal change (0 = baseline).

    References
    ----------
    Obata T et al. (2004) NeuroImage 21:144-153.
    """
```

### Type Annotations

Use `jaxtyping` for array shapes:

```python
from jaxtyping import Array, Float

def my_function(x: Float[Array, "N"]) -> Float[Array, "N"]:
    ...
```

### Module Structure

Each subpackage (`hemodynamics/`, `cardiac/`, etc.) should have:

- An `__init__.py` that re-exports public names
- One module per model or concept
- Docstrings on the module, all classes, and all public functions

### Equinox Modules

Models are Equinox modules (immutable pytrees). Parameters are stored
as module fields so the entire system is compatible with `jax.grad`,
`jax.jit`, and `jax.vmap`:

```python
class MyModel(eqx.Module):
    params: MyParams

    def __call__(self, t, y, args):
        ...
```

## 4. Testing

### Structure

Tests live in `vpjax/tests/` and follow this naming convention:

- `test_<module>.py` for unit tests of a specific module
- `test_integration.py` for cross-module pipeline tests
- `test_validation_*.py` for tests against experimental data

### What to Test

Every model should test at minimum:

1. **Steady state**: RHS is zero at baseline (no stimulus)
2. **Stimulus response**: qualitative direction of response is correct
3. **Differentiability**: `jax.grad` produces finite, nonzero gradients
4. **Return to baseline**: system recovers after transient stimulus

Example:

```python
class TestMyModel:
    def test_steady_state(self):
        model = MyModel(params=MyParams())
        y0 = MyState.steady_state()
        dy = model(jnp.array(0.0), y0, jnp.array(0.0))
        assert jnp.allclose(dy.x, 0.0, atol=1e-6)

    def test_differentiable(self):
        def loss(param):
            ...
        g = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(g)
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest vpjax/tests/test_balloon.py

# Parallel execution
uv run pytest -n auto

# With verbose output
uv run pytest -v
```

## 5. Documentation

### Building Docs Locally

```bash
uv pip install -e ".[doc]"
cd docs
make html
# Open _build/html/index.html in your browser
```

### Writing Tutorials

Tutorials live in `docs/tutorials/` as MyST Markdown (`.md`) files.
They should:

- Start from a test case and build up to a complete worked example
- Include the mathematical background with LaTeX equations
- Show code that can be copy-pasted and run
- Reference the original papers

### API Documentation

API docs are auto-generated from docstrings via `sphinx-apidoc`.
To regenerate after adding new modules:

```bash
sphinx-apidoc -o docs/reference vpjax --separate --module-first --force
```

## 6. AI-Assisted Development

vpjax was developed with AI assistance (Claude). When using AI tools:

- **Always verify** generated code against the referenced papers
- **Test physiological plausibility**: does the model produce the right
  sign, magnitude, and time course?
- **Check units**: cerebrovascular models mix seconds, mL/100g/min,
  mmHg, and fractional changes -- unit errors are the most common bug
- **Cite sources**: every model equation should reference a paper

## 7. Scientific Standards

### Physiological Plausibility

All models should produce results in physiologically plausible ranges:

| Quantity | Typical range | Units |
|----------|--------------|-------|
| CBF | 20-80 | mL/100g/min |
| OEF | 0.20-0.50 | fraction |
| CMRO2 | 80-200 | umol/100g/min |
| BOLD change | 0.5-5 | % |
| CBV | 2-6 | mL/100g |
| Arterial BP | 60-140 | mmHg |
| Heart rate | 40-120 | bpm |

### References

Every model module should include a `References` section in its
module-level docstring citing the original papers. Use the format:

```
Author AB et al. (Year). Title. Journal Volume:Pages.
```

### Reproducibility

- Pin JAX and Equinox versions in CI
- Use `jax.random.PRNGKey` for any stochastic components
- Report solver settings (step size, tolerance) in test docstrings
