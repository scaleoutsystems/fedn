# fashion-mnist test project

Fashion-mnist is a common used dataset for machine learning. To test FEDn with fashion-mnist dataset, following steps
should be followed: 
1. modify ``EXAMPLE`` in ``.env`` file as ``fashion-mnist``
2. prepare data: run ``cd ./test/fashion-mnist/data && ./prepare_data``
3. run FEDn kernel: ``cd ../../.. && docker-compose up``
4. run combiners: ``docker-compose -f combiner.yaml up``
5. run clients: ``docker-compose -f fashion-mnist-clients.yaml up``
6. after training... (the same as other examples)
7. compare performance of FEDn and centralized learning: run notebook
``./test/fashion-mnist/after-party/Visualization_and_comparison.ipynb``
8. clean all temp data and files: ``cd ./test/fashion_mnist/data && ./clean``
