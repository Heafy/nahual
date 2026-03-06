"""
collect.py

Entrypoint for the LSM gesture data collection tool.

Opens an interactive webcam window where you can label and capture
gesture samples to build the training dataset.

Usage::

    uv run python collect.py

Keyboard controls (also shown in the window):
    l  -- Enter a gesture label (uses terminal input)
    s  -- Capture one static sample
    d  -- Start / stop dynamic capture (auto-stops after 3 seconds)
    q  -- Quit
"""

from nahual.gesture_collector import CollectorConfig, GestureCollector


def main() -> None:
    """Run the interactive gesture data collector."""
    config = CollectorConfig(
        model_asset_path="models/hand_landmarker.task",
        show_landmark_debug=False,
    )
    collector = GestureCollector(config)
    collector.run()


if __name__ == "__main__":
    main()
