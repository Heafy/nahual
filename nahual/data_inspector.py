"""
nahual/data_inspector.py

Dataset inspection utilities for the LSM gesture dataset.

Scans the data/ directory tree and produces per-label summary rows
describing sample counts, gesture type, and array shape ranges.
All functions are pure (no I/O side effects) so they can be called
from inspect_data.py, train.py, or tests alike.

Usage::

    from pathlib import Path
    from nahual.data_inspector import collect_dataset_summary, format_dataset_table

    summaries = collect_dataset_summary(Path("data"))
    print(format_dataset_table(summaries))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

# Expected shape for a valid static gesture feature vector:
# 63 normalized coordinates + 10 finger angles + 8 inter-landmark distances.
STATIC_SAMPLE_SHAPE: tuple = (81,)


@dataclass
class LabelSummary:
    """Summary statistics for one gesture label.

    Attributes:
        label_name: Directory name used as the gesture label (e.g. "letra_a").
        gesture_type: "Static" or "Dynamic".
        shape_description: Human-readable shape string.
            Static: always "(81,)".
            Dynamic: "(min_frames-max_frames, 21, 3)" when frame counts vary,
                     "(N, 21, 3)" when all samples have the same frame count.
        sample_count: Number of .npy files found in the label directory.
    """

    label_name: str
    gesture_type: str
    shape_description: str
    sample_count: int


def _is_valid_label_name(directory_name: str) -> bool:
    """Return True if a directory name is a valid, printable gesture label.

    Rejects names that are empty or that begin with a non-printable character
    (ordinal < 32), which guards against filesystem artifacts created by
    ANSI escape sequences or other terminal control characters accidentally
    being interpreted as directory names.

    Args:
        directory_name: The bare directory name string (not a full path).

    Returns:
        True if the name is non-empty and starts with a printable character.
    """
    return bool(directory_name) and ord(directory_name[0]) >= 32


def collect_static_label_summary(label_directory: Path) -> LabelSummary:
    """Build a LabelSummary for one directory under data/static/.

    Counts .npy files. Shape is always "(81,)" for valid static samples —
    no file loading is needed because the shape is fixed by the feature
    extractor (63 coordinates + 10 angles + 8 distances = 81).

    Args:
        label_directory: Path object pointing to data/static/<label>/.

    Returns:
        LabelSummary with gesture_type="Static", shape_description="(81,)",
        and sample_count equal to the number of .npy files found.
    """
    sample_count = len(list(label_directory.glob("*.npy")))
    return LabelSummary(
        label_name=label_directory.name,
        gesture_type="Static",
        shape_description="(81,)",
        sample_count=sample_count,
    )


def collect_dynamic_label_summary(label_directory: Path) -> LabelSummary:
    """Build a LabelSummary for one directory under data/dynamic/.

    Counts .npy files and reads each file's shape[0] (frame count) using
    memory-mapped mode so only the numpy header is loaded — the full float
    array is not read into RAM. shape[1:] is always (21, 3) by construction
    (21 hand landmarks, each with x/y/z coordinates).

    Args:
        label_directory: Path object pointing to data/dynamic/<label>/.

    Returns:
        LabelSummary with gesture_type="Dynamic", a shape_description showing
        the frame-count range (e.g. "(32-73, 21, 3)"), and sample_count equal
        to the number of .npy files found.
    """
    npy_files = sorted(label_directory.glob("*.npy"))
    sample_count = len(npy_files)

    if sample_count == 0:
        shape_description = "(N, 21, 3)"
    else:
        frame_counts = [
            np.load(str(npy_file), mmap_mode="r").shape[0] for npy_file in npy_files
        ]
        minimum_frames = min(frame_counts)
        maximum_frames = max(frame_counts)

        if minimum_frames == maximum_frames:
            shape_description = f"({minimum_frames}, 21, 3)"
        else:
            shape_description = f"({minimum_frames}-{maximum_frames}, 21, 3)"

    return LabelSummary(
        label_name=label_directory.name,
        gesture_type="Dynamic",
        shape_description=shape_description,
        sample_count=sample_count,
    )


def collect_dataset_summary(data_root_directory: Path) -> list[LabelSummary]:
    """Scan the data/ directory tree and return one LabelSummary per label.

    Walks data/static/ and data/dynamic/ independently. Labels that appear
    in both subtrees produce two adjacent rows in the output (one Static,
    one Dynamic), sorted by label_name alphabetically. Labels with zero
    samples are omitted entirely to avoid cluttering the table with empty rows.

    Args:
        data_root_directory: Path to the project's data/ directory,
            typically Path("data") relative to the project root.

    Returns:
        List of LabelSummary objects sorted globally by label_name
        alphabetically. Returns an empty list if data_root_directory
        does not exist or contains no label subdirectories with samples.
    """
    summaries: list[LabelSummary] = []

    static_directory = data_root_directory / "static"
    if static_directory.exists() and static_directory.is_dir():
        for label_directory in sorted(static_directory.iterdir()):
            if not label_directory.is_dir():
                continue
            if not _is_valid_label_name(label_directory.name):
                continue
            summary = collect_static_label_summary(label_directory)
            if summary.sample_count > 0:
                summaries.append(summary)

    dynamic_directory = data_root_directory / "dynamic"
    if dynamic_directory.exists() and dynamic_directory.is_dir():
        for label_directory in sorted(dynamic_directory.iterdir()):
            if not label_directory.is_dir():
                continue
            if not _is_valid_label_name(label_directory.name):
                continue
            summary = collect_dynamic_label_summary(label_directory)
            if summary.sample_count > 0:
                summaries.append(summary)

    summaries.sort(key=lambda label_summary: label_summary.label_name)
    return summaries


def format_dataset_table(summaries: list[LabelSummary]) -> str:
    """Render a list of LabelSummary objects as a formatted PrettyTable.

    Builds a four-column table (Label, Type, Shape, Count) using PrettyTable.
    Label, Type and Shape are left-aligned; Count is right-aligned to make
    numbers easy to scan. A totals summary is appended below the table.

    Args:
        summaries: List of LabelSummary objects as returned by
            collect_dataset_summary().

    Returns:
        A multi-line string ready to be passed to print(). Returns an
        actionable message if summaries is empty.
    """
    if not summaries:
        return "No data found. Collect samples with:\n" "    uv run python collect.py"

    table = PrettyTable()
    table.field_names = ["Label", "Type", "Shape", "Count"]

    # Left-align text columns, right-align the numeric Count column.
    table.align["Label"] = "l"
    table.align["Type"] = "l"
    table.align["Shape"] = "l"
    table.align["Count"] = "r"

    for summary in summaries:
        table.add_row(
            [
                summary.label_name,
                summary.gesture_type,
                summary.shape_description,
                summary.sample_count,
            ]
        )

    total_static = sum(
        summary.sample_count
        for summary in summaries
        if summary.gesture_type == "Static"
    )
    total_dynamic = sum(
        summary.sample_count
        for summary in summaries
        if summary.gesture_type == "Dynamic"
    )
    grand_total = total_static + total_dynamic

    totals_lines = (
        f"\nStatic samples:  {total_static}"
        f"\nDynamic samples: {total_dynamic}"
        f"\nGrand total:     {grand_total}"
    )

    return str(table) + totals_lines
