"""Lazy reads from explicitly selected indexed or single-file safetensors sources."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import struct
import torch
from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str


class ShardReader:
    """Resolve a safetensors source while keeping at most one file handle open."""

    def __init__(
        self,
        model_dir: str | Path,
        index_filename: str = "model.safetensors.index.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / index_filename).read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._reset_handle()

    @classmethod
    def from_index(cls, index_path: str | Path) -> ShardReader:
        path = Path(index_path)
        return cls(path.parent, path.name)

    @classmethod
    def from_file(cls, tensor_path: str | Path) -> ShardReader:
        path = Path(tensor_path)
        reader = cls.__new__(cls)
        reader.model_dir = path.parent
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            reader.weight_map = {name: path.name for name in handle.keys()}
        reader._reset_handle()
        return reader

    def _reset_handle(self) -> None:
        self._current_shard: str | None = None
        self._context = None
        self._handle = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _open_shard(self, shard: str):
        if shard == self._current_shard:
            return self._handle
        self.close()
        self._context = safe_open(
            str(self.model_dir / shard),
            framework="pt",
            device="cpu",
        )
        self._handle = self._context.__enter__()
        self._current_shard = shard
        return self._handle

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._open_shard(shard)
        return handle.get_tensor(name)

    def metadata(self, names: Iterable[str]) -> dict[str, TensorMetadata]:
        self.close()
        by_shard: dict[str, list[str]] = {}
        for name in names:
            shard = self.weight_map[name]
            by_shard.setdefault(shard, []).append(name)

        result: dict[str, TensorMetadata] = {}
        for shard, shard_names in by_shard.items():
            header = self._header(shard)
            for name in shard_names:
                entry = header[name]
                result[name] = TensorMetadata(
                    name=name,
                    shard=shard,
                    shape=tuple(entry["shape"]),
                    dtype=str(entry["dtype"]),
                )
        return result

    def _header(self, shard: str) -> dict[str, dict[str, object]]:
        """Parse the leading safetensors header (8-byte length + JSON) directly.

        Unlike safe_open, this does not require the full data span to be
        present, so metadata works on partial/stub shard downloads."""

        path = self.model_dir / shard
        with path.open("rb") as handle:
            packed = handle.read(8)
            if len(packed) < 8:
                raise ValueError(f"{shard}: file too short for safetensors header")
            (header_len,) = struct.unpack("<Q", packed)
            raw = handle.read(header_len)
        header = json.loads(raw)
        return {k: v for k, v in header.items() if k != "__metadata__"}

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._current_shard = None
        self._context = None
        self._handle = None

    def __enter__(self) -> ShardReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
