#!/bin/bash
set -u
root=/dataset/shaoshitong/search-r1-rl/bapo
export PATH="$root/tools:$PATH" SEARCH_R1_BASE="$root" SEARCH_R1_PYTHON=/usr/bin/python3.10 UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 UV_CONCURRENT_DOWNLOADS=2 SEARCH_R1_PYPI_INDEX=https://pypi.org/simple/
export http_proxy=http://10.33.0.16:3128 https_proxy=http://10.33.0.16:3128 HTTP_PROXY=http://10.33.0.16:3128 HTTPS_PROXY=http://10.33.0.16:3128
bash "$root/repo/reproduction/h20/bootstrap.sh" > "$root/logs/bootstrap-pypi-20261003.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/logs/bootstrap-pypi-20261003.exit"
exit "$result"
