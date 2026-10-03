"""Constrain this pip process to IPv4, without editing host DNS or networking."""
import socket,sys
from pip._vendor.urllib3.util import connection
connection.allowed_gai_family=lambda: socket.AF_INET
from pip._internal.cli.main import main
raise SystemExit(main(sys.argv[1:]))
