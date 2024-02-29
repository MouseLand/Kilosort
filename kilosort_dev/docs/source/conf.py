# NOTE: To build locally, run the following from the top level of the Kilosort4
#       repository: `sphinx-build -b html docs/source docs/build/html`
#       For api: `sphinx-apidoc -f -o docs/source/api kilosort` (WIP)
#                For now, add to api.rst manually instead.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kilosort4'
copyright = '2023, Marius Pachitariu & Carsen Stringer'
author = 'Jacob Pennington (documentation)'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'myst_parser',            # Use markdown files in addition to .rst
    'nbsphinx',               # Render notebooks
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
    ]

autoapi_dirs = ['.../kilosort']
master_doc = 'index'

# Notebooks will be displayed even if they include errors
nbsphinx_allow_errors = True
# Don't auto-execute notebooks.
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_logo = 'http://www.kilosort.org/static/downloads/kilosort_logo_small.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'top',
    'style_external_links': False,
    'style_nav_header_background': 'black',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
html_static_path = ['_static']
