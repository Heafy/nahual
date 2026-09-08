"""
collect.py

Entrypoint for the Nahual gesture data collection tool.

Opens an interactive webcam window where you can label and capture
gesture samples for the selected sign language, to build its training
dataset.

Usage::

    uv run python collect.py
    uv run python collect.py -asl

Keyboard controls (also shown in the window):
    l  -- Enter a gesture label (uses terminal input)
    s  -- Capture one static sample
    d  -- Start / stop dynamic capture (auto-stops after 2 seconds)
    q  -- Quit
"""

from nahual.gesture_collector import CollectorConfig, GestureCollector
from nahual.sign_language import data_directory, parse_sign_language_argument


def main() -> None:
    """Run the interactive gesture data collector for the selected language."""
    language = parse_sign_language_argument("Collect Nahual gesture training samples.")
    config = CollectorConfig(
        model_asset_path="models/hand_landmarker.task",
        data_root_directory=data_directory(language),
        show_landmark_debug=False,
    )
    collector = GestureCollector(config)
    collector.run()


if __name__ == "__main__":
    main()
