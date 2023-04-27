# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kilosort 4'
copyright = '2023, Marius Pachitariu & Carsen Stringer'
author = 'Jacob Pennington (documentation)'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'myst_parser',            # Use markdown files in addition to .rst
    'nbsphinx'                # Render notebooks
    ]

# Notebooks will be displayed even if they include errors
nbsphinx_allow_errors = True
# Don't auto-execute notebooks.
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
