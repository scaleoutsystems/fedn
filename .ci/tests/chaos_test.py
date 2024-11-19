from toxiproxy import Toxiproxy
import unittest
import grpc
import time
from fedn.network.clients.grpc_handler import GrpcHandler
import fedn.network.grpc.fedn_pb2 as fedn


class TestGRPCWithToxiproxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client_name = 'test-client'
        client_id = 'test-client-id'
        host = 'localhost'
        port_proxy = 12081
        port_server = 12080
        token = ""
        combiner_name = 'combiner'

        cls.toxiproxy = Toxiproxy()
        if cls.toxiproxy.proxies():
            cls.toxiproxy.destroy_all()
    
    @classmethod
    def tearDownClass(cls):
        # Close the proxy and gRPC channel when done
        cls.toxiproxy.destroy_all()
    
    @unittest.skip("Not implemented")
    def test_normal_heartbeat(self):
        # Test the heartbeat without any toxic
        client_name = 'test-client'
        client_id = 'test-client-id'
        # Random proxy port
        grpc_handler = GrpcHandler(host='localhost', port=12080, name=client_name, token='', combiner_name='combiner')
        try:
            response = grpc_handler.heartbeat(client_name, client_id)
            self.assertIsInstance(response, fedn.Response)
        except grpc.RpcError as e:
            self.fail(f'gRPC error: {e.code()} {e.details()}')
        finally:
            grpc_handler.channel.close()

    @unittest.skip("Not implemented")
    def test_latency_2s_toxic_heartbeat(self):
         # Add latency of 1000ms
        client_name = 'test-client'
        client_id = 'test-client-id'

        proxy = self.toxiproxy.create(name='test_latency_toxic_heartbeat', listen='localhost:12082', upstream='localhost:12080')
        grpc_handler = GrpcHandler(host='localhost', port=12082, name=client_name, token='', combiner_name='combiner')
        proxy.add_toxic(name='latency', type='latency', attributes={'latency': 2000})
        
        start_time = time.time()
        try:
            response = grpc_handler.heartbeat(client_name, client_id)
        finally:
            grpc_handler.channel.close()
            proxy.destroy()
        end_time = time.time()
        
        # Check that the latency delay is present
        self.assertGreaterEqual(end_time - start_time, 2)  # Expect at least 1 second delay
        self.assertIsInstance(response, fedn.Response)
    
    def test_latency_long_toxic_heartbeat(self):
        """Test gRPC request with a simulated latency of 25s. Should timeout based on KEEPALIVE_TIMEOUT_MS (default set to 20000)."""
        client_name = 'test-client'
        client_id = 'test-client-id'
        latency = 20  # 15s latency

        proxy = self.toxiproxy.create(name='test_latency_toxic_heartbeat', listen='localhost:12083', upstream='localhost:12080')
        grpc_handler = GrpcHandler(host='localhost', port=12083, name=client_name, token='', combiner_name='combiner')
        proxy.add_toxic(name='latency', type='latency', attributes={'latency': latency * 1000})
        
        start_time = time.time()
        try:
            response = grpc_handler.heartbeat(client_name, client_id)
        except grpc.RpcError as e:
            response = e
        finally:
            grpc_handler.channel.close()
            proxy.destroy()
        end_time = time.time()

        response
        
        # Check that the latency delay is present
        self.assertGreaterEqual(end_time - start_time, latency)  # Expect at least 1 second delay
        self.assertIsInstance(response, grpc.RpcError)
        self.assertEqual(response.code(), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(response.details(), 'failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:12083: connection attempt timed out before receiving SETTINGS frame')

    def test_close_channel(self):
        """
        Test closing the gRPC channel and trying to send a heartbeat.
        Expect a ValueError to be raised.
        """

        client_name = 'test-client'
        client_id = 'test-client-id'

        grpc_handler = GrpcHandler(host='localhost', port=12080, name=client_name, token='', combiner_name='combiner')

        # Close the channel
        grpc_handler._disconnect()

        # Try to send heartbeat
        with self.assertRaises(ValueError) as context:
            response = grpc_handler.heartbeat(client_name, client_id)
        self.assertEqual(str(context.exception), 'Cannot invoke RPC on closed channel!')


    @unittest.skip("Not implemented")
    def test_disconnect_toxic_heartbeat(self):
        """Test gRPC request with a simulated disconnection."""
        # Add a timeout toxic to simulate network disconnection
        client_name = 'test-client'
        client_id = 'test-client-id'

        proxy = self.toxiproxy.create(name='test_disconnect_toxic_heartbeat', listen='localhost:12084', upstream='localhost:12080')
        grpc_handler = GrpcHandler(host='localhost', port=12084, name=client_name, token='', combiner_name='combiner')
        proxy.add_toxic(name='timeout', type='timeout', attributes={'timeout': 1000})

        try:
            response = grpc_handler.heartbeat(client_name, client_id)
        except grpc.RpcError as e:
            response = e
        finally:
            grpc_handler.channel.close()
            proxy.destroy()
        
        # Assert that the response is a gRPC error with status code UNAVAILABLE
        self.assertEqual(response.code(), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(response.details(), 'failed to connect to all addresses; last error: UNAVAILABLE: ipv4:127.0.0.1:12084: Socket closed')
        
    @unittest.skip("Not implemented")
    def test_timeout_toxic_heartbeat(self):
        """Stops all data from getting through, and closes the connection after timeout. timeout is 0, 
        the connection won't close, and data will be delayed until the toxic is removed.
        """
        # Add a timeout toxic to simulate network disconnection
        client_name = 'test-client'
        client_id = 'test-client-id'

        proxy = self.toxiproxy.create(name='test_timeout_toxic_heartbeat', listen='localhost:12085', upstream='localhost:12080')
        grpc_handler = GrpcHandler(host='localhost', port=12085, name=client_name, token='', combiner_name='combiner')
        proxy.add_toxic(name='timeout', type='timeout', attributes={'timeout': 0})

        try:
            response = grpc_handler.heartbeat(client_name, client_id)
        except grpc.RpcError as e:
            response = e
        finally:
            grpc_handler.channel.close()
            proxy.destroy()
        
        # Assert that the response is a gRPC error with status code UNAVAILABLE
        self.assertEqual(response.code(), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(response.details(), 'failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:12085: connection attempt timed out before receiving SETTINGS frame')

    @unittest.skip("Not implemented")
    def test_rate_limit_toxic_heartbeat(self):
        # Purpose: Limits the number of connections that can be established within a certain time frame.
        # Toxic: rate_limit
        # Use Case: Useful for testing how the client behaves under strict rate limits. For example, in Federated Learning, 
        # this could simulate constraints in networks with multiple clients trying to access the server.

        # Add a rate limit toxic to the proxy
        self.proxy.add_rate_limit(rate=1000)
    
    @unittest.skip("Not implemented")
    def test_bandwidth_toxic_heartbeat(self):
        # Purpose: Limits the bandwidth of the connection.
        # Toxic: bandwidth
        # Use Case: Useful for testing how the client behaves under limited bandwidth. For example, in Federated Learning, 
        # this could simulate a slow network connection between the client and the server.

        # Add a bandwidth toxic to the proxy
        self.proxy.add_bandwidth(rate=1000) # 1 KB/s
    
    @unittest.skip("Not implemented")
    def test_connection_reset(self):
        # Purpose: Immediately resets the connection, simulating an abrupt network drop.
        # Toxic: add_reset
        # Use Case: This is helpful for testing error-handling logic on sudden network failures, 
        # ensuring the client retries appropriately or fails gracefully

        # Add a connection_reset toxic to the proxy
        self.proxy.add_reset()
    
    @unittest.skip("Not implemented")
    def test_slow_close(self):
        # Purpose: Simulates a slow closing of the connection.
        # Toxic: slow_close
        # Use Case: Useful for testing how the client behaves when the server closes the connection slowly. 
        # This can help ensure that the client handles slow network disconnections gracefully.

        # Add a slow_close toxic to the proxy
        self.proxy.add_slow_close(delay=1000)  # Delay closing the connection by 1 second
    
    @unittest.skip("Not implemented")
    def test_slicer(self):
        # Purpose: Slices the data into smaller chunks.
        # Toxic: slicer
        # Use Case: Useful for testing how the client handles fragmented data. 
        # This can help ensure that the client can reassemble the data correctly and handle partial data gracefully.

        # Add a slicer toxic to the proxy
        self.proxy.add_slicer(average_size=1000, size_variation=100)  # Slice data into chunks of 1 KB with 100 bytes variation