"""Extract every COMPLETE file from partially-downloaded zips.

The BBBC022 mirror collapsed to ~19 KB/s, with ~3.3 of 6.9 GB already on disk.
A truncated zip cannot be opened by `zipfile` (no central directory), but its
files are laid out sequentially from the start, so every entry whose compressed
data is entirely present can still be recovered.

Two storage conventions must be handled:
  * general-purpose flag bit 3 clear -> the local header carries comp_size, so
    the entry's byte range is known exactly;
  * bit 3 set -> comp_size lives in a trailing data descriptor, so the stream is
    decompressed incrementally and the consumed length comes from
    `decompressobj.unused_data`.

Anything truncated is skipped, never half-written.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

SIGNATURE = b'PK\x03\x04'
CHUNK = 1 << 20


def iter_entries(path: Path, limit_bytes: int | None = None):
    """Yield ``(name, payload)`` for every entry fully present in ``path``."""
    size = path.stat().st_size
    if limit_bytes is not None:
        size = min(size, limit_bytes)

    with path.open('rb') as handle:
        offset = 0
        while offset + 30 <= size:
            handle.seek(offset)
            header = handle.read(30)
            if len(header) < 30 or header[:4] != SIGNATURE:
                break
            (_, _version, flags, method, _t, _d, _crc, comp_size,
             _uncomp, name_len, extra_len) = struct.unpack('<4sHHHHHIIIHH', header)
            name_bytes = handle.read(name_len)
            handle.read(extra_len)
            name = name_bytes.decode('utf-8', 'replace')
            data_start = offset + 30 + name_len + extra_len

            if name.endswith('/'):
                offset = data_start
                continue

            if not (flags & 0x08) and comp_size:
                if data_start + comp_size > size:
                    break  # truncated: everything after is unusable
                handle.seek(data_start)
                raw = handle.read(comp_size)
                offset = data_start + comp_size
            else:
                # Unknown length: decompress until the deflate stream ends.
                handle.seek(data_start)
                decompressor = zlib.decompressobj(-15 if method == 8 else 15)
                produced = bytearray()
                consumed = 0
                truncated = False
                while not decompressor.eof:
                    block = handle.read(CHUNK)
                    if not block:
                        truncated = True
                        break
                    try:
                        produced += decompressor.decompress(block)
                    except zlib.error:
                        truncated = True
                        break
                    consumed += len(block)
                if truncated or not decompressor.eof:
                    break
                consumed -= len(decompressor.unused_data)
                raw = bytes(produced)
                offset = data_start + consumed

            if method == 0:
                payload = raw
            elif method == 8:
                try:
                    payload = zlib.decompress(raw, -15)
                except zlib.error:
                    offset = data_start + len(raw)
                    continue
            else:
                offset = max(offset, data_start + len(raw))
                continue

            yield name, payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--zips', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-per-zip', type=int, default=100000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {}
    for zip_path in args.zips:
        path = Path(zip_path)
        if not path.exists():
            continue
        stem = path.stem
        target = out / stem
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        for name, payload in iter_entries(path):
            if count >= args.max_per_zip:
                break
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            (target / Path(name).name).write_bytes(payload)
            count += 1
        report[stem] = count
        print(f'  {stem}: recovered {count} complete images '
              f'({path.stat().st_size / (1 << 30):.2f} GiB downloaded)')

    (out / 'salvage_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
