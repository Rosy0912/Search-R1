"""Inspect one owned stale CPU-test session without spawning an HPC shell.

Run on the authorized HKUST3 bridge, with the JXU password on stdin in RAM.
SSH forwards the remote system-bus Unix socket under the authenticated JXU
identity. EXTERNAL uses the socket's real credentials (empty identity). No
alternate authentication, UID override, privilege escalation or quota change.
Default is read-only. The only mutation is a separately requested termination
of the exact old test session after a fresh MOU process/scope proof is supplied.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time


OWNER = 14090
SESSION = '680116'
SCOPE = 'session-' + SESSION + '.scope'
SESSION_TIMESTAMP = 1791964100509709  # Actual read-only D-Bus measurement.
TEST_PIDS = [2337454, 2348169, *range(2348170, 2348178)]
TEST_COMMAND = '-m unittest -v reproduction.hpc3_qprobe.test_launch_review_fixes reproduction.common8.test_production_safety_admission reproduction.common8.test_behavior_runtime reproduction.common8.test_model_smoke'


def pad(data, alignment):
    return data + b'\0' * (-len(data) % alignment)


def string(value, signature='s'):
    value = value.encode('utf-8')
    return (bytes([len(value)]) if signature == 'g' else struct.pack('<I', len(value))) + value + b'\0'


class Reader:
    """Small bounded D-Bus decoder for the read-only scalar/struct replies."""
    def __init__(self, data, endian='<'):
        self.data, self.pos, self.endian = data, 0, endian

    def take(self, count):
        result = self.data[self.pos:self.pos+count]
        if len(result) != count:
            raise ValueError('Truncated D-Bus message')
        self.pos += count
        return result

    def align(self, count):
        self.take(-self.pos % count)

    def value(self, signature, offset=0):
        kind = signature[offset]
        self.align({'y':1, 'g':1, 'v':1, 's':4, 'o':4, 'u':4, 'b':4, 't':8, '(':8}[kind])
        if kind in ('y', 'u', 'b', 't'):
            fmt = {'y':'B', 'u':'I', 'b':'I', 't':'Q'}[kind]
            return struct.unpack(self.endian+fmt, self.take(struct.calcsize(fmt)))[0], offset+1
        if kind in ('s', 'o', 'g'):
            count = self.take(1)[0] if kind == 'g' else struct.unpack(self.endian+'I', self.take(4))[0]
            result = self.take(count).decode('utf-8')
            if self.take(1) != b'\0':
                raise ValueError('Invalid D-Bus string')
            return result, offset+1
        if kind == 'v':
            inner, _ = self.value('g')
            result, end = self.value(inner)
            if end != len(inner):
                raise ValueError('Unsupported multi-value variant')
            return result, offset+1
        result, offset = [], offset+1
        while signature[offset] != ')':
            item, offset = self.value(signature, offset)
            result.append(item)
        return result, offset+1


class Bus:
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(8)
        self.socket.connect(path)
        self.serial = 0
        self.socket.sendall(b'\0AUTH EXTERNAL\r\n')
        reply = self.line()
        if reply.startswith(b'DATA'):
            self.socket.sendall(b'DATA\r\n')
            reply = self.line()
        if not reply.startswith(b'OK '):
            raise PermissionError('Real-peer EXTERNAL authentication rejected: '+repr(reply))
        self.socket.sendall(b'BEGIN\r\n')

    def line(self):
        result = b''
        while not result.endswith(b'\r\n'):
            result += self.recv(1)
            if len(result) > 1024:
                raise ValueError('Oversized D-Bus authentication reply')
        return result

    def recv(self, count):
        result = b''
        while len(result) < count:
            part = self.socket.recv(count-len(result))
            if not part:
                raise ConnectionError('D-Bus connection closed')
            result += part
        return result

    def call(self, destination, path, interface, member, args=()):
        self.serial += 1
        serial = self.serial
        body = b''
        for arg in args:
            body = pad(body, 4) + string(arg)
        fields = [(1,'o',path), (2,'s',interface), (3,'s',member), (6,'s',destination)]
        if args:
            fields.append((8,'g','s'*len(args)))
        header = b''
        for number, sig, value in fields:
            header = pad(header, 8) + bytes([number]) + string(sig, 'g')
            header = pad(header, 1 if sig == 'g' else 4) + string(value, sig)
        self.socket.sendall(pad(b'l\1\0\1'+struct.pack('<III',len(body),serial,len(header))+header,8)+body)
        for _ in range(64):
            fixed = self.recv(16)
            if fixed[0:1] not in (b'l',b'B') or fixed[3] != 1:
                raise ValueError('Unsupported D-Bus frame')
            endian = '<' if fixed[0:1] == b'l' else '>'
            length, _, head_length = struct.unpack(endian+'III', fixed[4:])
            if length+head_length > 1024*1024:
                raise ValueError('Oversized D-Bus frame')
            reader = Reader(self.recv(head_length), endian)
            result_fields = {}
            while reader.pos < head_length:
                reader.align(8)
                if reader.pos == head_length:
                    break
                number, _ = reader.value('y')
                value, _ = reader.value('v')
                result_fields[number] = value
            self.recv(-(16+head_length) % 8)
            reader = Reader(self.recv(length), endian)
            if result_fields.get(5) != serial:
                continue
            if fixed[1] == 3:
                raise PermissionError('D-Bus method rejected: '+str(result_fields.get(4)))
            if fixed[1] != 2:
                raise ValueError('Unexpected D-Bus response type')
            signature = result_fields.get(8, '')
            result, offset = [], 0
            while offset < len(signature):
                value, offset = reader.value(signature, offset)
                result.append(value)
            if reader.pos != length:
                raise ValueError('D-Bus response has unexpected trailing bytes')
            return result
        raise TimeoutError('No matching D-Bus response')

    def inspect(self):
        bus = ('org.freedesktop.DBus','/org/freedesktop/DBus','org.freedesktop.DBus')
        name = self.call(*bus, 'Hello')[0]
        uid = self.call(*bus, 'GetConnectionUnixUser', [name])[0]
        if uid != OWNER:
            raise PermissionError('Forwarded D-Bus connection is not the real JXU owner')
        sender_pid = self.call(*bus, 'GetConnectionUnixProcessID', [name])[0]
        manager = ('org.freedesktop.login1','/org/freedesktop/login1','org.freedesktop.login1.Manager')
        path = self.call(*manager, 'GetSession', [SESSION])[0]
        properties = {}
        for key in ('Id','User','Name','Service','Scope','State','Timestamp','Leader'):
            properties[key] = self.call('org.freedesktop.login1',path,'org.freedesktop.DBus.Properties','Get',['org.freedesktop.login1.Session',key])[0]
        expected = dict(Id=SESSION, Name='jxu957', Service='sshd', Scope=SCOPE, State='closing', Leader=0)
        if any(properties[k] != v for k,v in expected.items()) or properties['User'][0] != OWNER:
            raise ValueError('Target stale-test session identity or state differs')
        return dict(real_bus_uid=uid, bus_unique_name=name, sender_pid=sender_pid, session=properties, checked_unix=time.time())


def verify_proof(proof, inspected):
    if not 0 <= time.time()-proof['epoch'] <= 60:
        raise ValueError('Fresh observer scope proof required')
    if proof['session']['returncode'] != 0 or proof['processes']['returncode'] != 0:
        raise ValueError('Actual observer query failed')
    actual = proof['session']['stdout']
    for line in ('Id='+SESSION, 'User=14090', 'Name=jxu957', 'Timestamp=Wed 2026-10-14 15:48:20 CST', 'Service=sshd', 'Scope='+SCOPE, 'Leader=0', 'State=closing'):
        if line not in actual.splitlines():
            raise ValueError('Observer session birth or identity differs')
    if sorted(map(int,proof['scope_pids'].split())) != TEST_PIDS:
        raise ValueError('Unexpected process in stale test scope')
    lines = proof['processes']['stdout'].splitlines()[1:]
    relevant = [line.split(None, 10) for line in lines if int(line.split()[0]) in TEST_PIDS]
    if len(relevant) != len(TEST_PIDS):
        raise ValueError('Expected stale test process missing')
    for row in relevant:
        if len(row) != 11 or row[2:4] != ['2337454','2337454']:
            raise ValueError('Test process session/group differs')
        pid = int(row[0])
        expected_start = ['Wed','Oct','14','15:48:20','2026'] if pid == 2337454 else ['Wed','Oct','14','15:48:56','2026']
        if row[4:9] != expected_start or row[1] != ('1' if pid == 2337454 else '2337454'):
            raise ValueError('Stale test process birth or parent differs')
        if pid == 2337454 and not row[10].endswith(TEST_COMMAND):
            raise ValueError('Expected CPU unittest command absent')
        if pid != 2337454 and not any(s in row[10] for s in ('from multiprocessing.resource_tracker import main;', 'from multiprocessing.spawn import spawn_main;')):
            raise ValueError('Unexpected process command in stale test scope')
    if int(proof['pids.current']) != 499 or int(proof['pids.max']) != 500:
        raise ValueError('Observed saturation changed; re-evaluate scope before termination')
    if inspected['real_bus_uid'] != OWNER or inspected['session']['Timestamp'] != SESSION_TIMESTAMP:
        raise ValueError('D-Bus target session birth differs')


def run(terminate=False):
    password = input()
    proof = json.loads(input()) if terminate else None
    env = os.environ.copy()
    env['QPROBE_OWNER_PASSWORD'] = password
    password = None
    with tempfile.TemporaryDirectory(prefix='qprobe-owner-bus-') as directory:
        os.chmod(directory,0o700)
        path = str(Path(directory)/'system-bus')
        askpass = Path(directory)/'askpass'
        # This helper contains no credentials. Only the private in-memory child
        # environment supplies the password, and OpenSSH consumes its stdout.
        askpass.write_text('#!'+sys.executable+'\nimport os,sys\nsys.stdout.write(os.environ["QPROBE_OWNER_PASSWORD"]+"\\n")\n')
        askpass.chmod(0o700)
        env.update(SSH_ASKPASS=str(askpass),SSH_ASKPASS_REQUIRE='force',DISPLAY='qprobe-owner-bus')
        command = ['ssh','-N','-T','-oStrictHostKeyChecking=yes','-oExitOnForwardFailure=yes','-oControlMaster=no','-S','none','-oConnectTimeout=8','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-L',path+':/run/dbus/system_bus_socket','jxu957@10.120.81.30']
        child = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        env.pop('QPROBE_OWNER_PASSWORD',None)
        bus = None
        try:
            start = time.monotonic()
            while not Path(path).exists():
                if child.poll() is not None:
                    raise ConnectionError('Authenticated owner SSH forwarding failed')
                if time.monotonic()-start > 12:
                    raise TimeoutError('Authenticated owner forwarding did not start')
                time.sleep(.05)
            bus = Bus(path)
            result = bus.inspect()
            result.update(terminate_requested=terminate, terminated=False, local_bridge_uid=os.getuid(), held_local_ssh_pid=child.pid)
            if terminate:
                verify_proof(proof,result)
                bus.call('org.freedesktop.login1','/org/freedesktop/login1','org.freedesktop.login1.Manager','TerminateSession',[SESSION])
                result['terminated'] = True
                result['terminated_unix'] = time.time()
            print(json.dumps(result),flush=True)
        finally:
            if bus is not None:
                bus.socket.close()
            # Only the held, unreaped SSH child is signalled; no arbitrary PID.
            if child.poll() is None:
                child.terminate()
            try:
                child.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill();child.communicate(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminate-verified-stale-test',action='store_true')
    run(parser.parse_args().terminate_verified_stale_test)
