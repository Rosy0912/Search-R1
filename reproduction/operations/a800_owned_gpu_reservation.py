#!/data/search-g1/env/bin/python
"""Hold 60 GiB on each of A800 GPUs 0, 3 in one independently owned process.

This is a capacity reservation, never a trainer or retry launcher. Run only after
reviewing/publishing this source and independently confirming the selected GPUs are
free. The fresh namespace and immutable claim/source copy are retained on exit.

Example (SOURCE_SHA is the independently verified SHA-256 of this exact file):
  CUDA_VISIBLE_DEVICES=0,3 /data/search-g1/env/bin/python -I \
    /data/search-g1/reservation-tools/a800_owned_gpu_reservation.py \
    --namespace /data/search-g1/reservations/hold-YYYYMMDD-HHMMSS \
    --expected-source-sha256 "$SOURCE_SHA"

Release only this reservation's recorded PID after independently matching its
/proc start ticks, UID, SID and source/claim. SIGTERM/SIGINT release only this
process's tensors. There is no command for signalling any other process.
Admission checks cannot atomically exclude another process starting CUDA later;
such a process is detected by the periodic read-only check and causes own release.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time


ROLE = "only_capacity_reservation"
GPUS = (0, 3)
VISIBLE = "0,3"
BYTES_PER_GPU = 60 * 1024**3
HEADROOM_BYTES = 1024**3
ENV_PREFIX = Path("/data/search-g1/env")
BASE = Path("/data/search-g1/reservations")
STATE_FILE = "reservation.json"
ROLLBACK_SECONDS = 15
QUERY_TIMEOUT_SECONDS = 5
HOLD_POLL_SECONDS = 10


class ReservationError(RuntimeError):
    pass


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def process_identity():
    pid = os.getpid()
    # Field 22 (starttime) follows the comm field, which may contain spaces/').
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "sid": os.getsid(0),
            "pgid": os.getpgrp(), "uid": os.getuid(), "euid": os.geteuid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "hostname": os.uname().nodename}


def safe_directory(path, *, private=False):
    """Open each absolute component without following symlinks."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ReservationError("directory must be an absolute path without '..'")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            new_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=fd)
            os.close(fd)
            fd = new_fd
            info = os.fstat(fd)
            if info.st_uid not in (0, os.geteuid()) or stat.S_IMODE(info.st_mode) & 0o022:
                raise ReservationError(f"unsafe ownership/write permissions: {path}")
        info = os.fstat(fd)
        if private and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise ReservationError(f"directory must be owned by this UID with mode 0700: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_source(expected_sha):
    source = Path(__file__).absolute()
    parent_fd = safe_directory(source.parent)
    try:
        fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) & 0o022 or info.st_nlink != 1):
                raise ReservationError("source must be a singly linked owned regular file, not writable by others")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                contents = stream.read(1024 * 1024 + 1)
            if len(contents) > 1024 * 1024:
                raise ReservationError("source exceeds the 1 MiB audit limit")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    digest = hashlib.sha256(contents).hexdigest()
    if digest != expected_sha:
        raise ReservationError("source SHA-256 does not match the independently reviewed digest")
    return source, contents, digest


class DurableNamespace:
    def __init__(self, path):
        self.path = path
        self.counter = 0
        parent_fd = safe_directory(BASE.parent)
        try:
            try:
                os.mkdir(BASE.name, 0o700, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileExistsError:
                pass
        finally:
            os.close(parent_fd)
        base_fd = safe_directory(BASE, private=True)
        try:
            # A pre-existing name is always rejected; no claim or source is reused.
            os.mkdir(path.name, 0o700, dir_fd=base_fd)
            os.fsync(base_fd)
            self.fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=base_fd)
        finally:
            os.close(base_fd)
        self.inode = (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
        self.check()

    def check(self):
        info = os.lstat(self.path)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700
                or (info.st_dev, info.st_ino) != self.inode):
            raise ReservationError("reservation namespace ownership/path changed")

    def exclusive(self, name, data, mode=0o600):
        self.check()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     mode, dir_fd=self.fd)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(self.fd)

    def save(self, record):
        self.check()
        self.counter += 1
        name = f"state.tmp.{os.getpid()}.{self.counter}"
        data = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
        if len(data) > 64 * 1024:
            raise ReservationError("reservation metadata exceeds the 64 KiB audit limit")
        try:
            current = os.stat(STATE_FILE, dir_fd=self.fd, follow_symlinks=False)
            if (not stat.S_ISREG(current.st_mode) or current.st_uid != os.geteuid()
                    or stat.S_IMODE(current.st_mode) != 0o600 or current.st_nlink != 1):
                raise ReservationError("unsafe existing state file")
        except FileNotFoundError:
            pass
        self.exclusive(name, data)
        os.replace(name, STATE_FILE, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        os.fsync(self.fd)

    def source_backup_sha256(self):
        self.check()
        fd = os.open("source.py", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o400 or info.st_nlink != 1):
                raise ReservationError("unsafe source backup")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                return hashlib.sha256(stream.read(1024 * 1024 + 1)).hexdigest()
        finally:
            os.close(fd)


class Inventory:
    def __init__(self):
        self.program = shutil.which("nvidia-smi")
        if not self.program:
            raise ReservationError("nvidia-smi not found")
        self.program = str(Path(self.program).resolve(strict=True))
        parent_fd = safe_directory(Path(self.program).parent)
        try:
            info = os.stat(Path(self.program).name, dir_fd=parent_fd, follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
                    or stat.S_IMODE(info.st_mode) & 0o022):
                raise ReservationError("nvidia-smi must be a root-owned regular executable, not writable by others")
        finally:
            os.close(parent_fd)
        self.expected = None

    def rows(self, query):
        result = subprocess.run([self.program, query, "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True,
                                timeout=QUERY_TIMEOUT_SECONDS)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

    def snapshot(self, *, own_pid=None, require_free=False):
        selected = {}
        for row in self.rows("--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free,mig.mode.current"):
            if len(row) != 6:
                raise ReservationError("unrecognized GPU inventory row")
            index = int(row[0])
            if index not in GPUS:
                continue
            uuid = row[1].strip()
            if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
                raise ReservationError("physical GPU UUID required; MIG is not permitted")
            if row[5].strip().lower() not in ("disabled", "n/a", "[n/a]", "[not supported]"):
                raise ReservationError(f"MIG enabled or unknown on GPU {index}")
            selected[index] = {"index": index, "uuid": uuid, "pci_bus_id": row[2].strip(),
                               "total_bytes": int(row[3]) * 1024**2,
                               "free_bytes": int(row[4]) * 1024**2}
        if set(selected) != set(GPUS) or len({gpu["uuid"] for gpu in selected.values()}) != len(GPUS):
            raise ReservationError("the fixed selected physical GPUs are not uniquely present")
        identity = {i: (gpu["uuid"], gpu["pci_bus_id"]) for i, gpu in selected.items()}
        if self.expected is not None and identity != self.expected:
            raise ReservationError("physical GPU index/UUID mapping changed")
        apps = []
        uuids = {gpu["uuid"] for gpu in selected.values()}
        for row in self.rows("--query-compute-apps=gpu_uuid,pid,used_gpu_memory"):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != 3:
                raise ReservationError("unrecognized compute-app inventory row")
            if row[0].strip() not in uuids:
                continue
            app = {"uuid": row[0].strip(), "pid": int(row[1]),
                   "used_bytes": int(row[2]) * 1024**2}
            apps.append(app)
            if own_pid is None or app["pid"] != own_pid:
                raise ReservationError(f"foreign compute app on requested GPU: {app}")
        if require_free:
            for gpu in selected.values():
                if gpu["free_bytes"] < BYTES_PER_GPU + HEADROOM_BYTES:
                    raise ReservationError(f"GPU {gpu['index']} lacks 60 GiB plus 1 GiB headroom")
        self.expected = identity
        return {"at": utc_now(), "gpus": [selected[i] for i in GPUS], "compute_apps": apps}


def cuda_uuid_mapping(expected):
    """Verify driver ordinals under the fixed visibility mask before allocating."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    for function in (driver.cuInit, driver.cuDeviceGetCount, driver.cuDeviceGet, get_uuid):
        function.restype = ctypes.c_int
    def checked(code):
        if code != 0:
            raise ReservationError(f"CUDA identity query failed with driver code {code}")
    checked(driver.cuInit(0))
    count = ctypes.c_int()
    checked(driver.cuDeviceGetCount(ctypes.byref(count)))
    if count.value != len(GPUS):
        raise ReservationError("CUDA must expose exactly the selected two devices")
    mapping = []
    for ordinal, physical in enumerate(GPUS):
        device = ctypes.c_int()
        checked(driver.cuDeviceGet(ctypes.byref(device), ordinal))
        raw = (ctypes.c_ubyte * 16)()
        checked(get_uuid(raw, device.value))
        value = bytes(raw).hex()
        uuid = "GPU-" + "-".join((value[:8], value[8:12], value[12:16], value[16:20], value[20:]))
        if uuid.lower() != expected[physical][0].lower():
            raise ReservationError(f"CUDA ordinal {ordinal} does not map to physical GPU {physical}")
        mapping.append({"cuda_ordinal": ordinal, "physical_index": physical, "uuid": uuid,
                        "requested_bytes": BYTES_PER_GPU})
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--gpus", default=VISIBLE, choices=[VISIBLE])
    args = parser.parse_args()
    namespace = args.namespace
    if namespace.parent != BASE or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", namespace.name):
        parser.error(f"namespace must be a fresh direct child of {BASE}, with a simple unique name")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256):
        parser.error("expected-source-sha256 must be a lowercase 64-character SHA-256")
    if sys.platform != "linux" or os.getuid() != os.geteuid():
        raise ReservationError("Linux and matching real/effective UID are required")
    if Path(sys.prefix).resolve() != ENV_PREFIX.resolve() or not sys.flags.isolated:
        raise ReservationError("run with /data/search-g1/env/bin/python -I")
    env_fd = safe_directory(ENV_PREFIX)
    os.close(env_fd)
    os.umask(0o077)
    for name, required in (("CUDA_VISIBLE_DEVICES", VISIBLE), ("CUDA_DEVICE_ORDER", "PCI_BUS_ID")):
        if os.environ.get(name) not in (None, required):
            raise ReservationError(f"{name} must be unset or exactly {required}")
        os.environ[name] = required
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if os.environ.get(name) not in (None, "", "backend:native"):
            raise ReservationError(f"unexpected {name}; use a clean reservation environment")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
    source, contents, source_sha = read_source(args.expected_source_sha256)
    if os.getsid(0) != os.getpid():
        os.setsid()  # Only this process; failure does not fork or affect the caller.
    identity = process_identity()
    if identity["sid"] != identity["pid"] or identity["pgid"] != identity["pid"]:
        raise ReservationError("an independent session/process group is required")
    owned = DurableNamespace(namespace)
    record = {"schema_version": 1, "role": ROLE, "identity": identity,
              "namespace": str(namespace), "namespace_device": owned.inode[0],
              "namespace_inode": owned.inode[1], "source_path": str(source), "source_sha256": source_sha,
              "source_backup": "source.py", "claim": "claim.json",
              "cuda_visible_devices": VISIBLE, "cuda_device_order": "PCI_BUS_ID",
              "physical_indices": list(GPUS), "bytes_per_gpu": BYTES_PER_GPU,
              "status": "INITIALIZING", "created_at": utc_now(), "milestones": [],
              "allocated_devices": [], "visible_mapping": [], "preflight_samples": [],
              "ready_samples": [], "failure": None, "release": None}
    owned.exclusive("claim.json", (json.dumps({"role": ROLE, "identity": identity,
                    "namespace": str(namespace), "source_sha256": source_sha,
                    "physical_indices": list(GPUS)}, sort_keys=True) + "\n").encode())
    owned.exclusive("source.py", contents, mode=0o400)
    record["source_backup_sha256"] = owned.source_backup_sha256()
    if record["source_backup_sha256"] != source_sha:
        raise ReservationError("source backup hash differs from independently reviewed source")
    owned.save(record)
    os.fchdir(owned.fd)
    stop_signal = None
    tensors = []
    torch = None
    exit_code = 0

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum  # The main path releases only its own tensors.

    def save_status(status):
        record["status"] = status
        record["milestones"].append({"name": status, "at": utc_now()})
        owned.save(record)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
    try:
        inventory = Inventory()
        for _ in range(2):
            if stop_signal is not None:
                raise ReservationError("stop requested before CUDA initialization")
            record["preflight_samples"].append(inventory.snapshot(require_free=True))
            owned.save(record)
            time.sleep(0.2)
        save_status("ALLOCATING")
        # No torch import, CUDA library load, CUDA initialization or allocation precedes preflight.
        import torch as imported_torch
        torch = imported_torch
        record["torch_version"] = torch.__version__
        record["visible_mapping"] = cuda_uuid_mapping(inventory.expected)
        if torch.cuda.device_count() != len(GPUS):
            raise ReservationError("torch must expose exactly the selected two devices")
        for ordinal, physical in enumerate(GPUS):
            if stop_signal is not None:
                raise ReservationError("stop requested during allocation")
            inventory.snapshot(own_pid=identity["pid"])
            tensor = torch.empty(BYTES_PER_GPU, dtype=torch.uint8, device=f"cuda:{ordinal}")
            tensors.append(tensor)
            allocated_bytes = tensor.numel() * tensor.element_size()
            del tensor
            if allocated_bytes != BYTES_PER_GPU:
                raise ReservationError("tensor byte count differs from fixed 60 GiB request")
            record["allocated_devices"].append({"cuda_ordinal": ordinal, "physical_index": physical,
                                                 "tensor_bytes": allocated_bytes})
            owned.save(record)
        # Two independent measurements, spaced one second apart, gate READY.
        for sample_number in range(2):
            if stop_signal is not None:
                raise ReservationError("stop requested before readiness verification")
            snapshot = inventory.snapshot(own_pid=identity["pid"])
            usage = {}
            for app in snapshot["compute_apps"]:
                usage[app["uuid"].lower()] = usage.get(app["uuid"].lower(), 0) + app["used_bytes"]
            snapshot["torch_allocated_bytes"] = [torch.cuda.memory_allocated(i) for i in range(len(GPUS))]
            snapshot["torch_reserved_bytes"] = [torch.cuda.memory_reserved(i) for i in range(len(GPUS))]
            for item in record["visible_mapping"]:
                if (usage.get(item["uuid"].lower(), 0) < BYTES_PER_GPU
                        or snapshot["torch_allocated_bytes"][item["cuda_ordinal"]] < BYTES_PER_GPU):
                    raise ReservationError("nvidia-smi/torch did not confirm the full owned reservation")
            record["ready_samples"].append(snapshot)
            owned.save(record)
            if sample_number == 0:
                time.sleep(1)
        save_status("READY")
        print(json.dumps({"status": "READY", "role": ROLE, "identity": identity,
                          "state_path": str(namespace / STATE_FILE), "source_sha256": source_sha}), flush=True)
        # Sleeping and read-only inventory queries only; no kernels, model, optimizer, RNG or training.
        next_check = time.monotonic() + HOLD_POLL_SECONDS
        while stop_signal is None:
            time.sleep(0.25)
            if time.monotonic() >= next_check:
                inventory.snapshot(own_pid=identity["pid"])
                next_check = time.monotonic() + HOLD_POLL_SECONDS
    except BaseException as error:
        exit_code = 1
        record["failure"] = {"at": utc_now(), "type": type(error).__name__, "message": str(error)[:2000]}
        # The immediately following bounded release write includes this event.
        record["status"] = "FAILED"
        record["milestones"].append({"name": "FAILED", "at": utc_now()})
    finally:
        # A bounded cleanup affects only tensors/cache belonging to this process.
        record["release"] = {"at": utc_now(), "method": "own_tensors_and_cache_then_exit",
                             "signal": stop_signal, "allocation_bytes_verified_zero": False,
                             "rollback_timeout_seconds": ROLLBACK_SECONDS,
                             "timeout_action": "kernel_SIGALRM_terminates_only_this_process",
                             "timeout_requires_independent_process_exit_and_GPU_inventory_verification": True}
        # If cleanup or shutdown blocks, the kernel's default action ends only
        # this process, even if Python cannot run a handler. The timer remains
        # armed through interpreter shutdown and never targets another process.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL, ROLLBACK_SECONDS)
        try:
            save_status("RELEASING")
        except BaseException as error:
            print(f"could not persist RELEASING: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            exit_code = 1
        try:
            tensors.clear()
            release_errors = []
            after = []
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except BaseException as error:
                    release_errors.append(f"empty_cache: {type(error).__name__}: {error}"[:1000])
                for ordinal in range(len(record["visible_mapping"])):
                    try:
                        after.append({"cuda_ordinal": ordinal,
                                      "allocated_bytes": torch.cuda.memory_allocated(ordinal),
                                      "reserved_bytes": torch.cuda.memory_reserved(ordinal)})
                    except BaseException as error:
                        release_errors.append(f"device {ordinal}: {type(error).__name__}: {error}"[:1000])
            zero = not release_errors and all(item["allocated_bytes"] == 0 and item["reserved_bytes"] == 0
                                             for item in after)
            record["release"] = {"at": utc_now(), "method": "own_tensors_and_cache_then_exit",
                                 "signal": stop_signal, "allocation_bytes_verified_zero": zero,
                                 "torch_after": after, "errors": release_errors,
                                 "rollback_timeout_seconds": ROLLBACK_SECONDS,
                                 "driver_context_reclamation": "requires_independent_post_exit_verification"}
            if not zero:
                exit_code = 1
            save_status("RELEASED")
        finally:
            os.close(owned.fd)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        # Errors before private state creation have no CUDA allocations to roll back.
        print(json.dumps({"status": "REFUSED", "role": ROLE, "error_type": type(error).__name__,
                          "error": str(error)[:2000]}), file=sys.stderr, flush=True)
        sys.exit(2)
