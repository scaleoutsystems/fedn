.PHONY: clean apidoc html

clean:
	rm -rf fedn.*.rst
	rm -rf fedn.rst
	rm -rf _build/

apidoc:
	sphinx-apidoc --ext-autodoc --module-first -o . ../fedn ../*tests* ../*exceptions* ../*common* ../*v1* ../*cli* ../ ../fedn/network/api/server.py ../fedn/network/controller/controlbase.py

html: clean apidoc
	sphinx-build . _build