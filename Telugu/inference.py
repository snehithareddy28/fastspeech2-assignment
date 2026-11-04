"""Synthesise the Telugu test sentences and plot variance-adaptor predictions.

    python Telugu/inference.py                             # best.pth.tar (else latest / fp16 export)
    python Telugu/inference.py --ckpt Telugu/checkpoints/X.pth.tar
    python Telugu/inference.py --text "any sentence"       # writes to Results/Telugu/custom/

Outputs: Results/Telugu/test_generated_audios/<ID>.wav,
         Results/Telugu/variance_adaptor_plots/{duration,pitch,energy,overview}/<ID>.png,
         Results/Telugu/variance_adaptor_plots/<ID>.json (all predicted values).
Inference code: fastspeech2/synthesize.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

from synthesize import cli  # noqa: E402

if __name__ == "__main__":
    cli(default_config=os.path.join(HERE, "config.yaml"),
        default_sentences=os.path.join(HERE, "test_sentences.tsv"),
        default_out=os.path.join(ROOT, "Results", "Telugu"))
