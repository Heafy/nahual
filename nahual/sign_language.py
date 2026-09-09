"""
nahual/sign_language.py

Supported sign languages and the shared -lsm / -asl command-line flag.

Each language keeps its own dataset under data/<language>/ and its own trained
classifiers under models/<language>/, so collection, training, and inference
stay isolated between languages. The MediaPipe hand_landmarker.task asset is
language-independent and stays at the models/ top level.

Not imported by any module bundled into the browser demo (see
web/browser/build.py's NAHUAL_RUNTIME_MODULES).
"""

import argparse
from pathlib import Path

# Supported language codes. The first entry is the default.
SIGN_LANGUAGES = ("lsm", "asl")


def parse_sign_language_argument(description: str) -> str:
    """Parse the -lsm / -asl flag and return the selected language code.

    Used by all four root scripts (collect.py, inspect_data.py, main.py,
    train.py) so the flag behaves identically everywhere. The flags are
    mutually exclusive; passing both is rejected by argparse. Defaults to the
    first entry in SIGN_LANGUAGES when no flag is given.
    """
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group()
    for code in SIGN_LANGUAGES:
        group.add_argument(
            f"-{code}",
            dest="language",
            action="store_const",
            const=code,
            help=f"Use the {code.upper()} dataset and models.",
        )
    parser.set_defaults(language=SIGN_LANGUAGES[0])
    return parser.parse_args().language


def data_directory(language: str) -> Path:
    return Path("data") / language


def models_directory(language: str) -> Path:
    return Path("models") / language
