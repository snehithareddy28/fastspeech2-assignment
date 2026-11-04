"""Train the Telugu FastSpeech 2 model.

    python Telugu/train.py                     # resumes from Telugu/checkpoints/latest.pth.tar if present
    python Telugu/train.py --max_hours 8.5     # stop cleanly before a session limit
    python Telugu/train.py --fresh             # ignore existing checkpoints

Training code: fastspeech2/train.py (modified ming024 train.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2"))

from train import cli  # noqa: E402

if __name__ == "__main__":
    cli(default_config=os.path.join(HERE, "config.yaml"))
