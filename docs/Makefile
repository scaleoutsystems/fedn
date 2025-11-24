.PHONY: clean apidoc html

clean:
	rm -rf fedn.*.rst
	rm -rf fedn.rst
	rm -rf _build/

html: clean apidoc
	sphinx-build . _build