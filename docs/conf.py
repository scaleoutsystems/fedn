import os
import sys


# Insert paths
# sys.path.insert(0, os.path.abspath(".."))  # repo root
sys.path.insert(0, os.path.abspath("../scaleout-core"))
sys.path.insert(0, os.path.abspath("../scaleout-core/scaleoutcore/network/api/v1"))
sys.path.insert(0, os.path.abspath("../scaleout-client-python"))
sys.path.insert(0, os.path.abspath("../scaleout-util"))

# Project info
project = "Scaleout Edge"
author = "Scaleout Systems AB"

# The full version, including alpha/beta/rc tags
release = "0.33.0"

# Add any Sphinx extension module names here, as strings
extensions = [
    "sphinx_click.ext",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_code_tabs",
    "sphinx_copybutton",
]

autosummary_generate = True

# SEO configuration
html_title = "Scaleout Edge Documentation - Scalable Federated Learning Framework"
html_short_title = "Scaleout Edge Docs"

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", ".venv", "venv", "Thumbs.db", ".DS_Store"]

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

html_baseurl = "https://docs.scaleoutsystems.com/en/stable/"
html_theme_options = {
    "logo_only": True,
}

# SEO improvements
html_use_index = True
html_split_index = False

# mock imports
autodoc_mock_imports = [
    "click",
    "psutil",
    "grpc",
    "flask",
    "numpy",
    "pymongo",
    "jwt",
    "pydantic",
    "sqlalchemy",
    "psycopg2",
    "requests",
    "boto3",
    "minio",
    "redis",
    "yaml",
    "werkzeug",
    "fastapi",
    "uvicorn",
    "google",
    "alembic",
    "alembic.config",
    "opentelemetry",
    "opentelemetry.trace",
    "opentelemetry.instrumentation",
    "opentelemetry.sdk",
    "scaleoututil.grpc.scaleout_pb2",
    "scaleoututil.grpc.scaleout_pb2_grpc",
    "scaleoutcore.network.grpc.server_pb2",
    "scaleoutcore.network.grpc.server_pb2_grpc",
]

# Allow search engines to index the documentation
# Remove any robots restrictions
html_extra_path = ["robots.txt"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "scaleoutdocs"

# If defined shows an image instead of project name on page top-left (link to index page)
html_logo = "_static/images/scaleout_logo_flat_dark.svg"
# Scaleout Edge logo looks ugly on rtd theme

html_favicon = "favicon.png"

# Here we assume that the file is at _static/custom.css
html_css_files = [
    "css/elements.css",
    "css/text.css",
]

html_js_files = [
    (
        "https://scripts.simpleanalyticscdn.com/sri/v11.js",
        {
            "async": "async",
            "crossorigin": "anonymous",
            "integrity": (
                "sha256-hkUzQr3zWmSDnmhw95ZmQSZ949upqD+ML9ejiN0UIIE= "
                "sha384-rfv15RJy1bBYZ1Mf4xizO26jorXb2myipCvHXy4rkG0SuEET96S+m0sTzu5vfbSI "
                "sha512-lQzjzTbOxHLwkZGDVMf4V0sm8v2Mrqm73IvKcXBftJ/MSZKQC4/jwKFToxT+3IVAVWQzLplSNHH8gM5d7b1BSg=="
            ),
        },
    ),
]

# LaTeX elements
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "scaleout.tex", "Scaleout Edge Documentation", "Scaleout Systems AB", "manual"),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "scaleout", "Scaleout Edge Documentation", [author], 1)]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, "scaleout", "Scaleout Edge Documentation", author, "scaleout", "One line description of project.", "Miscellaneous"),
]

# Bibliographic Dublin Core info.
epub_title = project

epub_exclude_files = ["search.html"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}

pygments_style = "sphinx"

autodoc_default_options = {
    "members": True,  # Include all members
    "undoc-members": False,  # Include members without docstrings
    "private-members": False,  # Include private members (if any)
    "special-members": "__init__",  # Include special methods (if needed)
    "show-inheritance": True,  # Show class inheritance
}
