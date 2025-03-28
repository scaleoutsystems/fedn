FEDn is using sphinx with reStructuredText.

From repository root do
in bash:
sphinx-apidoc --ext-autodoc --module-first -o api-reference  ../fedn ../*tests* ../*exceptions* ../*common* ../ ../fedn/network/api/server.py ../fedn/network/controller/controlbase.py --templatedir ./module.rst_t
in zsh:
sphinx-apidoc --ext-autodoc --module-first -o api-reference  ../fedn ../\*tests\* ../\*exceptions\* ../\*common\* ../ ../fedn/network/api/server.py ../fedn/network/controller/controlbase.py --templatedir ./module.rst_t


cd docs/
sphinx-build . _build

cd _build/
on mac:
open index.html
on linux:
xdg-open index.html
on windows powershell:
start index.html



# Updated build Script
make html