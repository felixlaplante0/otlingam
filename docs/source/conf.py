"""Configure the Sphinx documentation builder."""

project = "otlingam"
copyright = "2026, Félix Laplante"
author = "Félix Laplante"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False
napoleon_use_ivar = True

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/otlingam-logo.svg"
html_favicon = "_static/otlingam-logo.svg"
html_css_files = ["custom.css"]
html_title = "OTLiNGAM"
nb_execution_mode = "off"
html_theme_options = {
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/felixlaplante0/otlingam",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
html_sidebars = {"**": []}
