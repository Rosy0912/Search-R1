#!/bin/bash
set -euo pipefail
root=/dataset/shaoshitong/search-r1-rl/bapo
export http_proxy=http://10.33.0.16:3128 https_proxy=http://10.33.0.16:3128 HTTP_PROXY=http://10.33.0.16:3128 HTTPS_PROXY=http://10.33.0.16:3128
name=faiss_gpu_cu12-1.8.0.2-cp310-cp310-manylinux_2_28_x86_64.whl
sha=7ea5e6aa82897fd0c1af3b1decc9c8659a4398fed42de1d7746715b64bbf9bf9
mkdir -p "$root/wheelhouse"
url=$(/usr/bin/python3.10 -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("https://pypi.org/pypi/faiss-gpu-cu12/1.8.0.2/json",timeout=20)); print(next(f["url"] for f in d["urls"] if f["filename"]=="faiss_gpu_cu12-1.8.0.2-cp310-cp310-manylinux_2_28_x86_64.whl"))')
curl -fLsS --connect-timeout 15 --max-time 1800 --speed-time 60 --speed-limit 1024 -C - -o "$root/wheelhouse/$name.partial" "$url"
printf '%s  %s\n' "$sha" "$root/wheelhouse/$name.partial" | sha256sum -c -
mv "$root/wheelhouse/$name.partial" "$root/wheelhouse/$name"
