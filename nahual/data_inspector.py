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

    Guards against filesystem artifacts created when ANSI escape sequences or
    other terminal control characters are accidentally interpreted as
    directory names.
    """
    return bool(directory_name) and ord(directory_name[0]) >= 32


def collect_static_label_summary(label_directory: Path) -> LabelSummary:
    """Build a LabelSummary for one directory under data/static/.

    No file is opened: the shape is fixed by the feature extractor at
    63 coordinates + 10 angles + 8 distances = 81.
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

    Frame counts vary per sample, so each file is opened memory-mapped to read
    only the numpy header rather than the full float array. shape[1:] is always
    (21, 3) by construction.
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

    A label present in both subtrees yields two adjacent rows (one Static, one
    Dynamic). Labels with zero samples are omitted so the table stays free of
    empty rows.

    Returns:
        Summaries sorted by label_name, or an empty list if the directory does
        not exist or holds no labels with samples.
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

    Returns:
        A multi-line string ready to print, or an actionable message telling
        the user how to collect samples if there are none.
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
