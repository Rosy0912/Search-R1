#!/bin/bash
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
mkdir -p "$R/lib/driver" "$R/logs/driver-link-check"
target=$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$target"
link="$R/lib/driver/libcuda.so"
if [ -L "$link" ]; then test "$(readlink -f "$link")" = "$target";else ln -s "$target" "$link";fi
export LIBRARY_PATH="$R/lib/driver${LIBRARY_PATH:+:$LIBRARY_PATH}"
printf '%s\n' 'extern int cuDriverGetVersion(int*); int main(void){int v=0;return cuDriverGetVersion(&v);}' > "$R/logs/driver-link-check/main.c"
gcc "$R/logs/driver-link-check/main.c" -lcuda -o "$R/logs/driver-link-check/probe"
readelf -d "$R/logs/driver-link-check/probe" | grep libcuda.so.1
sha256sum "$target"
printf 'ISOLATED_DRIVER_LINK_CPU_COMPILE_PASS %s\n' "$target"
