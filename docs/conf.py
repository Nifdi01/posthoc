from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "PostHoc"
author = "Nifdi Guliyev"
copyright = "2026, Nifdi Guliyev"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("posthoc")
    version = ".".join(release.split(".")[:2])
except Exception:
    release = version = "0.1.5"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_click",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc -----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["captum", "shap"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# -- HTML output ---------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
