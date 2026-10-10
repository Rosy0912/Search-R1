"""Repair only incongruent ELF load alignment in the isolated internal FAISS wheel."""
import hashlib
import json
import pathlib
import shutil
import struct

def repair(path):
    data = bytearray(path.read_bytes())
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected little-endian ELF64')
    offset = struct.unpack_from('<Q', data, 32)[0]
    size, count = struct.unpack_from('<HH', data, 54)
    changed = []
    for i in range(count):
        p = offset + i * size
        typ, flags, fo, va, pa, fs, ms, alignment = struct.unpack_from('<IIQQQQQQ', data, p)
        if typ == 1 and alignment > 4096 and (fo - va) % alignment:
            if (fo - va) % 4096:
                raise ValueError('Cannot safely repair non-page-congruent load segment')
            struct.pack_into('<Q', data, p + 48, 4096)
            changed.append({'segment': i, 'old_alignment': alignment, 'new_alignment': 4096})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    backup = path.with_name(path.name + '.before-elf-alignment')
    if changed:
        if backup.exists():
            raise ValueError('Existing original backup must be reviewed before another repair')
        shutil.copy2(path, backup)
        path.write_bytes(data)
    return {'file': str(path), 'before_sha256': before,
            'after_sha256': hashlib.sha256(data).hexdigest(), 'segments': changed}

if __name__ == '__main__':
    import sys
    print(json.dumps([repair(p) for p in pathlib.Path(sys.argv[1]).rglob('*.so')], indent=2))
