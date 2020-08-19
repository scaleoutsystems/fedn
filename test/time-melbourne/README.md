# time series on dataset minimum temperature in melbourne test project

Minimum temperature in Melbourne is a time series dataset for machine learning. To test FEDn with this dataset, following steps
should be followed: 
1. modify ``EXAMPLE`` in ``.env`` file as ``time-melbourne``
2. prepare data: run ``cd ./test/time-melbourne/data && ./prepare_data``
3. run FEDn kernel: ``cd ../../.. && docker-compose up``
4. run combiners: ``docker-compose -f combiner.yaml up``
5. run clients: ``docker-compose -f time-melbourne-clients.yaml up``
6. after training... (the same as other examples)
7. compare performance of FEDn and centralized learning: run notebook
``./test/time-melbourne/after-party/Visualization_and_comparison.ipynb``
8. clean all temp data and files: ``cd ./test/time-melbourne/data && ./clean``
