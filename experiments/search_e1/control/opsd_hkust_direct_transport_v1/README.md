# HKUST direct transport entry

`bin/ssh` is the unchanged 104-byte entry prepared for the authorized Mac-side HKUST backup queue. Prepend this directory's `bin` to `PATH` for that independent queue only. It executes `/usr/bin/ssh` with the fixed options `-B en0 -oProxyCommand=none -oProxyJump=none -oControlMaster=no -S none`, then forwards the caller's original arguments.

This entry does not change SSH configuration, published backup helpers, training configuration, checkpoints, or existing partial transfers. It takes no training or GPU lease. Root must publish and verify the exact source/readback before starting the queue; this source preparation does not claim an actual connection or completed backup.

Pinned entry SHA256: `9d62b0bba56fbc67331c6741f0c026033bdc5960a2abf7d93a9385156471f6a1`. The private queue command references this published source directory; its prior source/publication proof still binds backup helper commit `538ac99aac12ca9347f993556de22ad06391c132`. Root separately binds publication of this transport entry.
