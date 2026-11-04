"""Train the English FastSpeech 2 model.

    python English/train.py                     # resumes from English/checkpoints/latest.pth.tar if present
    python English/train.py --max_hours 8.5     # stop cleanly before a session limit
    python English/train.py --fresh             # ignore existing checkpoints

Training code: fastspeech2/train.py (modified ming024 train.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2"))

from train import cli  # noqa: E402

if __name__ == "__main__":
    cli(default_config=os.path.join(HERE, "config.yaml"))
