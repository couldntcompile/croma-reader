import socket
import struct

def pad(s): return (s + ('\0' * (4 - len(s) % 4))).encode()

def build_osc_message(address, value):
    return pad(address) + pad(',i') + struct.pack('>i', value)

msg = build_osc_message('/hello', 123)

sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
sock.sendto(msg, ('::1', 9000))
sock.close()