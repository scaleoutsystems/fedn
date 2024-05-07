import unittest
from unittest.mock import MagicMock

from fedn.network.combiner.aggregators.fedavg import FedAvg


class TestFedAvg(unittest.TestCase):
    """Test the FedAvg aggregator"""

    def SetUp(self):
        pass

    def test_fedavg_init(self, *args, **kwargs):
        """Test the FedAvg aggregator constructor"""
        aggregator = FedAvg("id", None, None, None, None)
        self.assertEqual(aggregator.name, "FedAvg")

    def test_fedavg_combine_models(self, *args, **kwargs):
        """Test the FedAvg aggregator combine_models method with mock classes and methods"""
        aggregator = FedAvg("id", None, None, None, None)
        aggregator.next_model_update = MagicMock(return_value=(None, None, None))
        aggregator.server = MagicMock()

        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0
        data['nr_aggregated_models'] = 0

        self.assertEqual(aggregator.combine_models(), (None, data))


if __name__ == '__main__':
    unittest.main()
