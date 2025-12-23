# -*- coding: utf-8 -*-
"""
Final inference entry for weapon sound recognition.

This script intentionally delegates to the runtime pipeline under `src.audio`
so it always uses the best available model version (e.g. ensemble_v2_5 -> v2.4 -> v2.3 ...).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.audio import ModelLoader, AudioRecognizer


class WeaponClassifier:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or PROJECT_DIR
        self.model_loader = ModelLoader(self.base_dir)
        ok, msg = self.model_loader.load_model()
        if not ok:
            raise RuntimeError(msg)
        self.recognizer = AudioRecognizer(self.model_loader)

    def predict_from_file(self, audio_path: str) -> tuple[str, float]:
        return self.recognizer.predict_from_file(audio_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Audio file path (*.mp3/*.wav/*.flac)")
    args = parser.parse_args()

    clf = WeaponClassifier(PROJECT_DIR)
    weapon, confidence = clf.predict_from_file(args.audio)
    print(f"weapon={weapon}")
    print(f"confidence={confidence:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
