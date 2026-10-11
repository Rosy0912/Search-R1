"""Validate the current FAISS flat-IP serialization before a bounded read-only mmap.

No faiss.read_index, blind offset, full-vector heap allocation, or full-file SHA.
The runtime's tiny same-class serialized probes define the actual header layout.
"""
import hashlib
from pathlib import Path
import struct
import numpy as np


def file_metadata(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def derive_flat_layout(serialize_probe, dimension):
    if type(dimension) is not int or dimension < 2:
        raise ValueError('Distinct count/payload fields require dimension >= 2')
    headers = []
    for count in (0, 1, 2):
        serialized = bytes(serialize_probe(dimension, count))
        header_len = len(serialized)-count*dimension*4
        if header_len < 16:
            raise ValueError('Unsupported flat serialization')
        headers.append(serialized[:header_len])
    if len({len(header) for header in headers}) != 1:
        raise ValueError('Probe payload size does not follow exact FP32 flat vectors')
    if any(header[:4] != headers[0][:4] for header in headers):
        raise ValueError('Probe flat-IP magic changes with count')
    def matching_field(values):
        matches = [offset for offset in range(4, len(headers[0])-7)
            if [struct.unpack_from('<Q', header, offset)[0] for header in headers] == values]
        if len(matches) != 1:
            raise ValueError('Ambiguous or unsupported flat count field')
        return matches[0]
    total_offset = matching_field([0, 1, 2])
    vectors_offset = matching_field([0, dimension, 2*dimension])
    if not (total_offset+8 <= vectors_offset or vectors_offset+8 <= total_offset):
        raise ValueError('Overlapping count fields')
    normalized = []
    for header in headers:
        value = bytearray(header)
        struct.pack_into('<Q', value, total_offset, 0)
        struct.pack_into('<Q', value, vectors_offset, 0)
        normalized.append(bytes(value))
    if len(set(normalized)) != 1:
        raise ValueError('Header has unproven non-count variability')
    return dict(magic=headers[0][:4], dimension=dimension, header=normalized[0],
                header_bytes=len(headers[0]), ntotal_offset=total_offset,
                payload_count_offset=vectors_offset)


def runtime_layout(faiss, dimension):
    def serialize(d, count):
        index = faiss.IndexFlatIP(d)
        if count:
            # Distinct real FP32 payload, with no changes to the oracle's vectors.
            vectors = np.arange(count*d, dtype=np.float32).reshape(count,d)
            index.add(vectors)
        return faiss.serialize_index(index).tobytes()
    return derive_flat_layout(serialize, dimension)


def validated_prefix_mmap(path, layout, rows=None, hash_selected_rows=262144):
    path = Path(path).resolve(strict=True)
    before = file_metadata(path)
    with path.open('rb') as stream:
        header = stream.read(layout['header_bytes'])
    if len(header) != layout['header_bytes'] or header[:4] != layout['magic']:
        raise ValueError('Actual flat-IP header/magic differs from runtime probes')
    ntotal = struct.unpack_from('<Q', header, layout['ntotal_offset'])[0]
    expected = bytearray(layout['header'])
    struct.pack_into('<Q', expected, layout['ntotal_offset'], ntotal)
    struct.pack_into('<Q', expected, layout['payload_count_offset'], ntotal*layout['dimension'])
    if header != bytes(expected):
        raise ValueError('Actual FAISS class/dimension/metric/header differs from same-class probes')
    if ntotal < 1 or before['bytes'] != layout['header_bytes']+ntotal*layout['dimension']*4:
        raise ValueError('Actual file size does not match the proved flat FP32 payload')
    if rows is not None and (type(rows) is not int or rows < 1):
        raise ValueError('Requested mapped prefix rows must be positive')
    selected = ntotal if rows is None else min(ntotal, rows)
    host = np.memmap(path, dtype='<f4', mode='r', offset=layout['header_bytes'],
                     shape=(selected, layout['dimension']))
    digest = hashlib.sha256()
    # Full selected small prefix, but never an implicit full 64.6GB hash.
    if type(hash_selected_rows) is not int or hash_selected_rows < 0:
        raise ValueError('Selected vector hash budget must be a nonnegative integer')
    hashed_rows = min(selected, hash_selected_rows)
    for start in range(0, hashed_rows, 4096):
        digest.update(host[start:min(start+4096, hashed_rows)].tobytes())
    after = file_metadata(path)
    if after != before:
        raise ValueError('Index metadata changed during prefix validation')
    proof = dict(path=str(path), metadata_before=before, metadata_after=after,
        original_index_rows=ntotal, mapped_prefix_rows=selected, dimension=layout['dimension'],
        payload_offset=layout['header_bytes'], magic_hex=layout['magic'].hex(),
        actual_prefix_sha256=hashlib.sha256(header).hexdigest(),
        selected_vector_prefix_sha256=digest.hexdigest() if hashed_rows else None, selected_vector_hashed_rows=hashed_rows,
        selected_vector_hashed_bytes=hashed_rows*layout['dimension']*4,
        full_index_file_sha_checked=False, read_only_mmap=True)
    return host, proof


def verify_unchanged(proof):
    if file_metadata(proof['path']) != proof['metadata_before']:
        raise ValueError('Index metadata changed during actual benchmark')
