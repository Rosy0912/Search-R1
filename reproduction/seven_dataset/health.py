"""Bounded startup-only health retries; identity mismatches never retry."""
import json
import time
import urllib.error
import urllib.request
from .contracts import require


def wait_retriever_health(url, job, *, attempts=6, timeout=30, delay=5,
                          opener=None, sleeper=None):
    require(type(attempts) is int and 1 <= attempts <= 6, 'Invalid retry bound')
    require(0 < timeout <= 30 and 0 <= delay <= 5, 'Invalid health deadline')
    opener = opener or urllib.request.urlopen
    sleeper = sleeper or time.sleep
    for attempt in range(1, attempts + 1):
        try:
            with opener(url + '/health', timeout=timeout) as response:
                data = json.load(response)
        except (OSError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 502, 503, 504):
                raise
            if attempt == attempts:
                raise
            print('SEVEN_STARTUP_HEALTH_RETRY=' + json.dumps(dict(attempt=attempt,
                  error_type=type(exc).__name__, max_attempts=attempts)), flush=True)
            sleeper(delay)
            continue
        require(data.get('job') == job, 'Retriever job differs')
        return data
