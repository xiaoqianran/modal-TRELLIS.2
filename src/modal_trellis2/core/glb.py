from __future__ import annotations

import json
import struct
from collections.abc import Sequence

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


class GlbError(ValueError):
    """Input is not a GLB we can hand to a viewer."""


def is_glb(data: bytes) -> bool:
    return (
        len(data) >= 12
        and data[:4] == GLB_MAGIC
        and struct.unpack_from("<I", data, 4)[0] == GLB_VERSION
    )


def validate_glb(data: bytes) -> None:
    if not is_glb(data):
        raise GlbError("not a GLB (missing glTF v2 header)")
    total = struct.unpack_from("<I", data, 8)[0]
    if total != len(data):
        raise GlbError(f"GLB length mismatch: header {total} vs actual {len(data)}")
    if len(data) < 20:
        raise GlbError("GLB too short to contain a JSON chunk")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != JSON_CHUNK:
        raise GlbError("first chunk is not JSON")
    if 20 + json_len > len(data):
        raise GlbError("JSON chunk overruns file")


def _pad4(data: bytes, pad: bytes) -> bytes:
    extra = (4 - (len(data) % 4)) % 4
    return data + pad * extra


def write_glb(document: dict, bin_chunk: bytes = b"") -> bytes:
    json_bytes = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = _pad4(bin_chunk, b"\x00")
    chunks = bytearray()
    chunks += struct.pack("<II", len(json_bytes), JSON_CHUNK)
    chunks += json_bytes
    if bin_bytes:
        chunks += struct.pack("<II", len(bin_bytes), BIN_CHUNK)
        chunks += bin_bytes
    header = struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, 12 + len(chunks))
    glb = header + chunks
    validate_glb(glb)
    return glb


def colored_cube_glb(
    color: Sequence[float],
    *,
    size: float = 0.72,
    generator: str = "modal-trellis2-mock",
) -> bytes:
    """A tiny PBR-looking cube so the web viewer works without a GPU."""
    r, g, b = (max(0.0, min(1.0, float(c))) for c in color[:3])
    s = size / 2
    faces = (
        ((0, 0, 1), ((-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s))),
        ((0, 0, -1), ((s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s))),
        ((0, 1, 0), ((-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s))),
        ((0, -1, 0), ((-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s))),
        ((1, 0, 0), ((s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s))),
        ((-1, 0, 0), ((-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s))),
    )
    positions: list[float] = []
    normals: list[float] = []
    colors: list[float] = []
    indices: list[int] = []
    for normal, corners in faces:
        base = len(positions) // 3
        for corner in corners:
            positions.extend(corner)
            normals.extend(normal)
            colors.extend((r, g, b))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    pos_b = struct.pack(f"<{len(positions)}f", *positions)
    nrm_b = struct.pack(f"<{len(normals)}f", *normals)
    col_b = struct.pack(f"<{len(colors)}f", *colors)
    idx_b = struct.pack(f"<{len(indices)}H", *indices)
    blob = pos_b + nrm_b + col_b + idx_b

    pos_off, nrm_off = 0, len(pos_b)
    col_off = nrm_off + len(nrm_b)
    idx_off = col_off + len(col_b)
    vertex_count = 24
    document = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "mock-cube"}],
        "meshes": [
            {
                "name": "mock-cube",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                        "indices": 3,
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "max": [s, s, s],
                "min": [-s, -s, -s],
            },
            {"bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "normalized": False,
            },
            {"bufferView": 3, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_b), "target": 34962},
            {"buffer": 0, "byteOffset": col_off, "byteLength": len(col_b), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_b), "target": 34963},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    return write_glb(document, blob)
