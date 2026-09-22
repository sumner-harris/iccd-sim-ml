"""Portable manifests for precomputed synthetic ICCD samples.

The training data layer deliberately knows nothing about plasma physics.  Each
manifest row points to an immutable ``.npz`` product created by the imaging
pipeline and carries only the identifiers needed for reproducibility and safe
dataset splitting.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SampleRecord:
    """Identity and location of one precomputed video sample."""

    sample_id: str
    path: Path
    element: str
    simulation_id: str
    family_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in ("sample_id", "element", "simulation_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self, *, path: Path | None = None) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "path": str(self.path if path is None else path),
            "element": self.element,
            "simulation_id": self.simulation_id,
            "family_id": self.family_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampleRecord:
        return cls(
            sample_id=str(value["sample_id"]),
            path=Path(value["path"]),
            element=str(value["element"]),
            simulation_id=str(value["simulation_id"]),
            family_id=None if value.get("family_id") is None else str(value["family_id"]),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class DatasetManifest:
    """Ordered sample records and the file they were loaded from, if any."""

    records: tuple[SampleRecord, ...]
    source: Path | None = field(default=None, compare=False)
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source).expanduser().resolve())
        if self.version != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported dataset manifest version {self.version}; expected {MANIFEST_VERSION}"
            )
        ids = [record.sample_id for record in self.records]
        if len(ids) != len(set(ids)):
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            raise ValueError(f"Duplicate sample IDs in manifest: {duplicates}")

    @property
    def base_dir(self) -> Path:
        return Path.cwd() if self.source is None else self.source.parent

    @property
    def by_id(self) -> dict[str, SampleRecord]:
        return {record.sample_id: record for record in self.records}

    def resolve_path(self, record: SampleRecord) -> Path:
        path = record.path.expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def select(self, sample_ids: Iterable[str]) -> tuple[SampleRecord, ...]:
        lookup = self.by_id
        requested = tuple(sample_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Requested sample IDs contain duplicates")
        missing = sorted(set(requested).difference(lookup))
        if missing:
            raise KeyError(f"Sample IDs are not present in the manifest: {missing}")
        return tuple(lookup[sample_id] for sample_id in requested)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "records": [record.to_dict() for record in self.records],
        }

    def save(self, path: str | Path, *, relative_paths: bool = True) -> Path:
        """Atomically save the manifest, using portable relative paths when possible."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for record in self.records:
            record_path = self.resolve_path(record)
            serialized_path = record_path
            if relative_paths:
                try:
                    serialized_path = Path(os.path.relpath(record_path, destination.parent))
                except ValueError:
                    # Different Windows drives cannot be represented by a relative path.
                    serialized_path = record_path
            rows.append(record.to_dict(path=serialized_path))
        payload = {"version": self.version, "records": rows}
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            records=tuple(SampleRecord.from_dict(row) for row in payload["records"]),
            source=source,
            version=int(payload.get("version", MANIFEST_VERSION)),
        )


def as_manifest(value: DatasetManifest | str | Path | Iterable[SampleRecord]) -> DatasetManifest:
    """Normalize common manifest inputs to :class:`DatasetManifest`."""

    if isinstance(value, DatasetManifest):
        return value
    if isinstance(value, (str, Path)):
        return DatasetManifest.load(value)
    return DatasetManifest(tuple(value))
