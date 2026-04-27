"""
inspect_data.py

Dataset inspection tool for the Nahual LSM gesture dataset.

Prints a formatted table of sample counts for every label found
under data/static/ and data/dynamic/.

Usage::

    uv run python inspect_data.py
"""

from pathlib import Path

from nahual.data_inspector import collect_dataset_summary, format_dataset_table


def main() -> None:
    """Load dataset summaries and print the inspection table.

    Scans the data/ directory relative to the project root and prints
    a formatted table with label, gesture type, shape, and sample count.
    """
    data_root_directory = Path("data")
    summaries = collect_dataset_summary(data_root_directory)
    print(format_dataset_table(summaries))


if __name__ == "__main__":
    main()
