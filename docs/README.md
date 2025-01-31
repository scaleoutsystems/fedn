FEDn is using sphinx with reStructuredText.

From repository root do
in bash:
sphinx-apidoc --ext-autodoc --module-first -o _source  ../fedn/fedn ../*tests* ../*exceptions* ../*common* ../ ../fedn/fedn/network/api/server.py ../fedn/fedn/network/controller/controlbase.py
in zsh:
sphinx-apidoc --ext-autodoc --module-first -o _source ../fedn/fedn ../\*tests\* ../\*exceptions\* ../\*common\* ../ ../fedn/fedn/network/api/server.py ../fedn/fedn/network/controller/controlbase.py

cd docs/
sphinx-build . _build

cd _build/
on mac:
open index.html
on linux:
xdg-open index.html
on windows powershell:
start index.html