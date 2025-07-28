"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

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
sys.path.insert(0, os.path.abspath('../../src/qrisp'))

# -- Project information -----------------------------------------------------

project = ""
copyright = '2025, Qrisp developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
                # 'sphinx_fontawesome',
              'sphinx_toolbox.sidebar_links',
              #'sphinx_toolbox.github',
              'nbsphinx',
               'myst_parser',
              "sphinx.ext.autosummary",
              "sphinx.ext.autodoc",
              "sphinx.ext.coverage",
              "sphinx.ext.doctest",
              "sphinx.ext.intersphinx",
               "sphinx.ext.mathjax",
              "sphinx.ext.todo",
              "sphinx.ext.viewcode",
                # "sphinx.ext.imgmath",
              "texext",
                "numpydoc",
              "sphinx_sitemap",
               "sphinx_copybutton",
               "sphinx_design",
               "sphinx_thebe"
              #"nb2plots",
              ]

imgmath_latex_preamble = r'\usepackage{braket}\n\usepackage{xcolor}'

thebe_config = {
    "repository_url": "https://github.com/fraunhoferfokus/Qrisp",
    "repository_branch": "thebe_branch",
    "selector": "div.highlight",
    "selector_output": "span.go",
}


imgmath_use_preview = True
#github_username = 'positr0nium'
github_repository = 'https://github.com/eclipse-qrisp/Qrisp'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '**/.ipynb_checkpoints',
]
nbsphinx_timeout = 60

master_doc = "index"

html_baseurl = 'https://qrisp.eu'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#


html_context = {
   # ...
   "default_mode": "light"
}



html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/eclipse-qrisp/Qrisp",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "enable_search_shortcuts" : True,
    "search_bar_text": "Search the docs... ",
    # "navbar_align": "content",
    "pygment_light_style": "lovelace",
    "secondary_sidebar_items": ["page-toc.html", "slack_link.html", "dynamic_sidebar.html", "thebe_button.html"]
}
html_sidebars = {
    "**": ["sidebar-nav-bs"],
    "index": [],
    "install": [],
    #"tutorial": [],
    "tutorial/*": ["sidebar-nav-bs"],
    "auto_examples/index": [],
}

autodoc_default_options = {
    'members': False,
    'member-order': 'bysource',
    'special-members': None,
    'undoc-members': False,
    'exclude-members': None
}
autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Removes the sphinx integrated option to see the page source
html_show_sourcelink = False

html_favicon = '../../logo/qrisp_favicon.png'

# html_logo = "../../logo/logo_extended.png"
html_logo = "../../logo/qrisp_logo.png"


add_module_names = False

html_css_files = [
    'css/custom05.css',
]

source_suffix = ['.rst', '.md']
# Adds 'Edit on gitlab' in the upper right corner
# html_context = {
#     "display_gitlab": True, # Integrate Gitlab
#     "gitlab_repo": "Qrisp Compiler", # Repo name
#     "conf_py_path": "/source/", # Path in the checkout to the docs root
# }
