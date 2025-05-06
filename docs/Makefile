.PHONY: clean apidoc html

clean:
	rm -rf fedn.*.rst
	rm -rf fedn.rst
	rm -rf _build/

apidoc:
	sphinx-apidoc --ext-autodoc --module-first -o . ../fedn ../*tests* ../fedn/cli* ../fedn/common* ../fedn/network/api/v1* ../fedn/network/grpc/fedn_pb2.py ../fedn/network/grpc/fedn_pb2_grpc.py  ../fedn/network/api/server.py ../fedn/network/controller/controlbase.py

html: clean apidoc
	sphinx-build . _build