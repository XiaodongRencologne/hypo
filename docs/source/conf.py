import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hypo'
copyright = '2026, xiaodong ren'
author = 'xiaodong ren'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
source_suffix =  {
    ".rst": "restructuredtext",
    ".md": "markdown"
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
]
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_theme_option = {
    "navigation_depth": -1,
    "collapse_navigation": False,
    "sticky_navigation":  True,
    "titles_only": False,
    "includehidden": True,
}

#theml_show_sourcelink = True
#html_copy_source =True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
