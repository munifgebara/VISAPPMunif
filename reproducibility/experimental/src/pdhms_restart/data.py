"""PaHaW indexing, loading, and audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import re

import numpy as np
import openpyxl
import pandas as pd


SVC_PATTERN = re.compile(r"^(?P<subject>\d+)__(?P<task>\d+)_(?P<repetition>\d+)\.svc$")
COLUMNS = ("y", "x", "timestamp", "state", "azimuth", "altitude", "pressure")


@dataclass(frozen=True)
class Sample:
    """One PaHaW recording and its subject-level class label."""

    path: Path
    subject_id: str
    task_id: int
    repetition_id: int
    label: str


@dataclass(frozen=True)
class AuditRow:
    """Structural checks for one recording."""

    subject_id: str
    task_id: int
    repetition_id: int
    label: str
    relative_path: str
    declared_points: int
    observed_points: int
    count_matches: bool
    finite: bool
    state_values_valid: bool
    negative_timestamp_steps: int
    zero_timestamp_steps: int
    surface_points: int
    surface_segments: int
    sha256: str


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labels(dataset_root: Path) -> dict[str, str]:
    """Read subject labels from the PaHaW workbook."""

    workbook_path = dataset_root / "PaHaW_files" / "corpus_PaHaW.xlsx"
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    rows = list(workbook.active.values)
    header = tuple(str(value) for value in rows[0])
    id_index = header.index("ID")
    label_index = header.index("Disease")
    labels: dict[str, str] = {}
    for row in rows[1:]:
        if row[id_index] is None:
            continue
        subject_id = f"{int(row[id_index]):05d}"
        label = str(row[label_index]).strip()
        if label not in {"H", "PD"}:
            raise ValueError(f"Unexpected label {label!r} for subject {subject_id}.")
        labels[subject_id] = label
    return labels


def index_samples(dataset_root: Path) -> list[Sample]:
    """Index every PaHaW SVC file and attach its label."""

    labels = read_labels(dataset_root)
    public_root = dataset_root / "PaHaW_public"
    samples: list[Sample] = []
    unparsed: list[Path] = []
    for path in public_root.rglob("*.svc"):
        match = SVC_PATTERN.fullmatch(path.name)
        if match is None:
            unparsed.append(path)
            continue
        subject_id = f"{int(match.group('subject')):05d}"
        if subject_id not in labels:
            raise ValueError(f"No metadata label for {path}.")
        samples.append(
            Sample(
                path=path,
                subject_id=subject_id,
                task_id=int(match.group("task")),
                repetition_id=int(match.group("repetition")),
                label=labels[subject_id],
            )
        )
    if unparsed:
        raise ValueError(f"Unrecognized SVC names: {[str(path) for path in unparsed]}")
    return sorted(samples, key=lambda item: (item.task_id, item.subject_id, item.repetition_id))


def load_svc(path: Path) -> tuple[int, np.ndarray]:
    """Load declared count and seven numeric columns from one SVC file."""

    with path.open("r", encoding="utf-8") as stream:
        first_line = stream.readline().strip()
    declared = int(first_line)
    points = np.loadtxt(path, dtype=np.float64, skiprows=1, ndmin=2)
    if points.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected seven columns, got {points.shape}.")
    return declared, points


def audit_samples(samples: list[Sample], dataset_root: Path) -> pd.DataFrame:
    """Audit each recording and return one row per file."""

    rows: list[dict[str, object]] = []
    for sample in samples:
        declared, points = load_svc(sample.path)
        timestamps = points[:, 2]
        states = points[:, 3]
        deltas = np.diff(timestamps)
        surface_points = states > 0
        surface_segments = surface_points[:-1] & surface_points[1:]
        row = AuditRow(
            subject_id=sample.subject_id,
            task_id=sample.task_id,
            repetition_id=sample.repetition_id,
            label=sample.label,
            relative_path=sample.path.relative_to(dataset_root).as_posix(),
            declared_points=declared,
            observed_points=int(points.shape[0]),
            count_matches=declared == int(points.shape[0]),
            finite=bool(np.all(np.isfinite(points))),
            state_values_valid=bool(np.all(np.isin(states, (0.0, 1.0)))),
            negative_timestamp_steps=int(np.count_nonzero(deltas < 0)),
            zero_timestamp_steps=int(np.count_nonzero(deltas == 0)),
            surface_points=int(np.count_nonzero(surface_points)),
            surface_segments=int(np.count_nonzero(surface_segments)),
            sha256=sha256_file(sample.path),
        )
        rows.append(asdict(row))
    return pd.DataFrame(rows).sort_values(["task_id", "subject_id", "repetition_id"])
