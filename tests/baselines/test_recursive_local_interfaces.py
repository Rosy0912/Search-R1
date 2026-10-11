import socket,unittest
from unittest.mock import patch,MagicMock
from search_r1.baselines.local_runtime import local_addresses
class Interfaces(unittest.TestCase):
 def test_dns_loopback_still_accepts_actual_route_interface(self):
  import fcntl
  raw=b'\0'*20+socket.inet_aton('10.122.23.4')+b'\0'*8
  with patch('socket.getaddrinfo',return_value=[(0,0,0,0,('127.0.1.1',0))]),patch('socket.if_nameindex',return_value=[(2,'eth0')]),patch('fcntl.ioctl',return_value=raw):
   values=local_addresses();self.assertIn('10.122.23.4',values);self.assertNotIn('10.122.23.5',values)
 def test_unaddressed_interface_not_whitelisted(self):
  import fcntl
  with patch('socket.getaddrinfo',return_value=[]),patch('socket.if_nameindex',return_value=[(2,'eth0')]),patch('fcntl.ioctl',side_effect=OSError()):self.assertEqual(local_addresses(),{'127.0.0.1','::1'})
if __name__=='__main__':unittest.main()
