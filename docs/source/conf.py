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

autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []

source_suffix = [".rst"]

pygments_style = "sphinx"

# html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_immaterial"
html_static_path = ['_static']
html_theme_options = {

    'repo_name': 'CLUEstering',

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

    "globaltoc_collapse": False,
    "globaltoc_includehidden": False,
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
