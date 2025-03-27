.PHONY: clean apidoc html

clean:
	rm -rf api-reference/
	rm -rf _build/

apidoc:
	sphinx-apidoc --ext-autodoc --module-first -o api-reference ../fedn ../*tests* ../*exceptions* ../*common* ../*cli* ../ ../fedn/network/api/server.py ../fedn/network/controller/controlbase.py

html: clean apidoc
	sphinx-build . _build