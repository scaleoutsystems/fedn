FEDn is using sphinx with reStructuredText.

sphinx-apidoc --ext-autodoc --module-first -o _source  ../fedn/fedn ../*tests* ../*exceptions* ../*common* ../ ../fedn/fedn/network/api/server.py ../fedn/fedn/network/controller/controlbase.py
sphinx-build . _build