Scaleout Edge is using sphinx with reStructuredText.

# Install sphinx
pip install -r requirements.txt

# Updated build Script
cd docs/
make html

cd _build/
on mac:
open index.html
on linux:
xdg-open index.html
on windows powershell:
start index.html



