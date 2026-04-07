# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "vpjax"
author = "Morgan G Hough"
copyright = "2024-2026, Morgan G Hough"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings (NumPy-style docstrings) ------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock imports for packages that are heavy / unavailable at doc build time
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "equinox",
    "diffrax",
    "optax",
    "distrax",
    "lineax",
    "optimistix",
    "jaxtyping",
    "numpy",
    "scipy",
    "matplotlib",
    "nibabel",
    "nilearn",
    "h5py",
    "trimesh",
    "mne",
    "sklearn",
    "skimage",
]

# -- MyST settings -----------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "vpjax"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/ins-amu/vpjax",
    "source_branch": "main",
    "source_directory": "docs/",
}
