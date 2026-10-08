#!/usr/bin/env python3
"""Record one already-created Slurm job; never submit or alter a job.

Launch in a new session with SEARCH_G1_BASE and SEARCH_G1_PROFILE unset:
    python3 native_state_recorder.py JOB_ID JOB_NAME /absolute/job/base
The launcher, not this recorder, must call setsid. Claims are never reclaimed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


INTERVAL = 30.0
QUERY_TIMEOUT = 15.0
CAPS = {"stdout": 64 * 1024, "stderr": 16 * 1024}
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}
NATIVE_KEYS = (
    "JobId", "JobName", "JobState", "UserId", "Partition", "SubmitTime",
    "EligibleTime", "StartTime", "EndTime", "Restarts", "ReqNodeList",
    "ExcNodeList", "NodeList", "NumNodes", "BatchHost", "RunTime",
    "TimeLimit", "TimeMin", "Deadline", "ExitCode", "Reason", "Command",
    "WorkDir",
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(directory, name, payload, exclusive=False):
    """Write complete JSON, then atomically publish; exclusive never replaces."""
    destination = directory / name
    temporary = directory / ("." + name + "." + str(os.getpid()) + ".tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def own_identity():
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    ticks = int(stat[19])  # /proc stat field 22, after comm and pid.
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    boot_seconds = next(
        int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    born = boot_seconds + ticks / os.sysconf("SC_CLK_TCK")
    return {
        "pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0),
        "uid": os.getuid(), "euid": os.geteuid(), "start_ticks": ticks,
        "boot_id": boot_id,
        "pid_birth_utc": datetime.fromtimestamp(born, timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "script_path": str(Path(__file__).absolute()),
    }


def query(command):
    result = {"started_at": now(), "host": socket.gethostname(), "command": command}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    failure = None
    process = None
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for name in buffers:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = time.monotonic() + QUERY_TIMEOUT
        while selector.get_map() and failure is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = "query_timeout"
                break
            for key, _ in selector.select(min(remaining, 0.5)):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                room = CAPS[name] - len(buffers[name])
                buffers[name].extend(chunk[:room])
                if len(chunk) > room:
                    failure = name + "_cap_exceeded"
                    break
        if failure is None:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                failure = "query_timeout"
    except OSError as error:
        failure = "query_os_error: " + str(error)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()  # Only this recorder's scontrol child, never a job.
            for name in buffers:
                stream = getattr(process, name)
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                failure = failure or "query_child_did_not_exit"
            result["returncode"] = process.poll()
        else:
            result["returncode"] = None
        selector.close()
    result.update({name: value.decode("utf-8", errors="replace")
                   for name, value in buffers.items()})
    result.update({"finished_at": now(), "query_error": failure})
    return result


def parse_fields(raw):
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", raw))
    fields = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise ValueError("duplicate native field: " + key)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        fields[key] = raw[match.end():end].strip()
    if not fields.get("JobState"):
        raise ValueError("missing native JobState")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("job_name")
    parser.add_argument("job_base")
    args = parser.parse_args()
    if not re.fullmatch(r"[1-9][0-9]*", args.job_id):
        parser.error("job_id must be one exact positive numeric Slurm job ID")
    if not args.job_name or any(ord(char) < 32 for char in args.job_name + args.job_base):
        parser.error("job_name and job_base must be nonempty plain text")
    if not os.path.isabs(args.job_base):
        parser.error("job_base must be absolute")
    if any(key in os.environ for key in ("SEARCH_G1_BASE", "SEARCH_G1_PROFILE")):
        parser.error("launcher must unset SEARCH_G1_BASE and SEARCH_G1_PROFILE")
    base = Path(os.path.normpath(args.job_base))
    if not base.is_dir():
        parser.error("job_base must be an existing directory")
    identity = own_identity()
    if identity["sid"] != identity["pid"]:
        parser.error("launcher must start this recorder with setsid")
    scontrol = shutil.which("scontrol")
    if scontrol is None:
        parser.error("scontrol is unavailable")
    expected = {
        "JobId": args.job_id, "JobName": args.job_name,
        "Command": str(base / "repo/reproduction/common8/launch_native.sh"),
        "WorkDir": str(base / "repo"),
    }
    os.umask(0o077)
    directory = base / "preparation/native-state-recorder" / args.job_id
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    claim = {"claimed_at": now(), "host": socket.gethostname(),
             "expected_job": expected, "recorder": identity}
    try:
        save(directory, "claim.json", claim, exclusive=True)
    except FileExistsError:
        print("Recorder claim already exists; refusing duplicate capture.", file=sys.stderr)
        return 2
    command = [scontrol, "show", "job", "-o", args.job_id]
    birth = None
    first_binding = None
    last_state = None
    invalid_count = 0
    while True:
        cycle_start = time.monotonic()
        observation = query(command)
        failure = observation["query_error"]
        fields = None
        if failure is None and observation["returncode"] == 0:
            try:
                fields = parse_fields(observation["stdout"])
            except ValueError as error:
                failure = "native_parse_error: " + str(error)
        elif failure is None:
            failure = "nonzero_scontrol_returncode"
        if fields is not None:
            mismatches = {key: {"expected": value, "actual": fields.get(key)}
                          for key, value in expected.items() if fields.get(key) != value}
            binding = {key: fields.get(key) for key in ("SubmitTime", "UserId")}
            if first_binding is not None and binding != first_binding:
                mismatches["submission_binding"] = {"expected": first_binding, "actual": binding}
            if mismatches:
                observation["identity_mismatch"] = mismatches
                observation["query_error"] = "identity_mismatch"
                save(directory, "latest-observation.json", observation)
                save(directory, "latest-error.json", observation)
                save(directory, "identity-mismatch-unknown.json",
                     {"outcome": "UNKNOWN", "claim": claim, "initial_birth": birth,
                      "observation": observation}, exclusive=True)
                return 3
            first_binding = first_binding or binding
            observation["native_fields"] = {key: fields.get(key) for key in NATIVE_KEYS}
        observation["query_error"] = failure
        save(directory, "latest-observation.json", observation)
        if failure is not None:
            save(directory, "latest-error.json", observation)
            invalid = (failure == "nonzero_scontrol_returncode" and
                       "invalid job id" in (observation["stdout"] + observation["stderr"]).lower())
            invalid_count = invalid_count + 1 if invalid else 0
            if invalid_count >= 3:
                save(directory, "purged-unknown.json",
                     {"outcome": "UNKNOWN", "reason": "scheduler_record_unavailable",
                      "invalid_job_id_observations": invalid_count, "claim": claim,
                      "initial_birth": birth, "submission_binding": first_binding,
                      "observation": observation}, exclusive=True)
                return 3
        else:
            invalid_count = 0
            state = fields["JobState"]
            if state == "RUNNING" and birth is None:
                birth = {"observed_at": observation["finished_at"], "claim": claim,
                         "submission_binding": first_binding, "observation": observation}
                save(directory, "birth.json", birth, exclusive=True)
            if state != last_state:
                with open(directory / "states.jsonl", "a", encoding="utf-8") as handle:
                    json.dump({"observed_at": observation["finished_at"], "state": state,
                               "restarts": fields.get("Restarts"),
                               "start_time": fields.get("StartTime")}, handle, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                last_state = state
            if state in TERMINAL:
                save(directory, "terminal.json",
                     {"capture_kind": "native_terminal", "native_job_state": state,
                      "claim": claim, "initial_birth": birth,
                      "first_running_observed": birth is not None,
                      "submission_binding": first_binding, "observation": observation},
                     exclusive=True)
                return 0
        time.sleep(max(0.0, INTERVAL - (time.monotonic() - cycle_start)))


if __name__ == "__main__":
    sys.exit(main())
