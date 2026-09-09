"""
inspect_data.py

Dataset inspection tool for the Nahual gesture dataset.

Prints a formatted table of sample counts for every label found under the
selected sign language's static/ and dynamic/ directories.

Usage::

    uv run python inspect_data.py
    uv run python inspect_data.py -asl
"""

from nahual.data_inspector import collect_dataset_summary, format_dataset_table
from nahual.sign_language import data_directory, parse_sign_language_argument


def main() -> None:
    """Load dataset summaries and print the inspection table."""
    language = parse_sign_language_argument("Inspect the Nahual gesture dataset.")
    summaries = collect_dataset_summary(data_directory(language))
    print(format_dataset_table(summaries))


if __name__ == "__main__":
    main()
