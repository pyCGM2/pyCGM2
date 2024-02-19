# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['btk']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
# -- Project information -----------------------------------------------------

project = 'pyCGM2-API'
copyright = '2022, Fabien Leboeuf'
author = 'Fabien Leboeuf'

# The full version, including alpha/beta/rc tags
release = '4.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_parser",
             "sphinx.ext.autosectionlabel",
             'sphinx.ext.napoleon',
             "sphinx.ext.autodoc",
             "sphinx.ext.autosummary",
             'sphinxarg.ext',
             'sphinxcontrib.mermaid' ]

source_suffix = [".rst",".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["templates"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
  "show_toc_level": 5
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

napoleon_custom_sections = [('Returns', 'params_style')]

html_logo = '_static/pyCGM2-logo2.png'


# 
autosectionlabel_prefix_document = True

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

