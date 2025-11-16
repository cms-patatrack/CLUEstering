# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../CLUEstering'))

project = 'CLUEstering'
copyright = '2025, Simone Balducci, Felice Pantaleo, Wahid Redjeb, Marco Rovere, Aurora Perego, Francesco Giacomini'
author = 'Simone Balducci'
release = '2.8.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_immaterial',
    'breathe'
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = [".rst"]

pygments_style = "sphinx"

# html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_immaterial"
html_static_path = ['_static']
# html_context = {
#         "default_mode": "dark"  # Options are "auto", "dark", or "light"
# }
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'CLUEstering',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://cms-patatrack.github.io/CLUEstering',

    # Set the color and the accent color
    'color_primary': 'blue',
    'color_accent': 'light-blue',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/cms-patatrack/CLUEstering/',
    'repo_name': 'CLUEstering',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 3,
    # If False, expand all TOC entries
    'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,

    "palette": [
        {
            "scheme": "slate",
            "primary": "blue",
            "accent": "light-blue",
            # "toggle": {
            #     "icon": "material/lightbulb",
            #     "name": "Switch to light mode",
            # }
        },
        # {
        #     "scheme": "default",
        #     "primary": "blue",
        #     "accent": "light-blue",
        #     "toggle": {
        #         "icon": "material/lightbulb-outline",
        #         "name": "Switch to dark mode",
        #     }
        # }
    ],

    "features": [
        "navigation.top",
        "toc.follow",
        "search.share",
        "header.autohide",
    ],
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


breathe_projects = {"CLUEstering": "../xml"}
breathe_default_project = "CLUEstering"
breathe_domain_by_extension = {"cpp": "cpp", "h": "cpp", "hpp": "cpp", "tpp": "cpp"}

cpp_id_attributes = [
    "ALPAKA_FN_ACC",
    "ALPAKA_FN_HOST",
    "ALPAKA_FN_HOST_ACC",
    "ALPAKA_FN_INLINE",
    "ALPAKA_NO_HOST_ACC_WARNING",
    "ALPAKA_STATIC_ACC_MEM_CONSTANT",
    "ALPAKA_STATIC_ACC_MEM_GLOBAL",
    "inline",
    "constexpr"
]
