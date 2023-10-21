from fedn.network.combiner.server import Combiner

config = {'discover_host': 'localhost', 'discover_port': 8092, 'token': None, 'host': 'localhost',
          'port': 12080, 'fqdn': None, 'name': 'combiner', 'secure': False, 'verify': False, 'max_clients': 10,
          'init': None, 'aggregator': 'fedavg'}

fedn_combiner = Combiner(config)
