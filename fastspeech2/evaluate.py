"""Validation loss. Based on ming024 evaluate.py.

MODIFIED: returns (message, loss_means) for CSV logging, takes an explicit
device, caches the validation DataLoader, and drops the TensorBoard audio /
figure logging (inference.py produces figures instead). Can also be run
standalone to score a checkpoint on the validation set.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config  # noqa: E402
from dataset import Dataset  # noqa: E402
from model import FastSpeech2Loss  # noqa: E402
from utils.tools import to_device  # noqa: E402

_CACHE = {}


def evaluate(model, step, configs, logger=None, vocoder=None, device=None):
    preprocess_config, model_config, train_config = configs
    device = device or next(model.parameters()).device

    key = preprocess_config["path"]["preprocessed_path"]
    if key not in _CACHE:
        dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
        loader = DataLoader(
            dataset,
            batch_size=train_config["optimizer"]["batch_size"],
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=2,
        )
        _CACHE[key] = (dataset, loader, FastSpeech2Loss(preprocess_config, model_config).to(device))
    dataset, loader, Loss = _CACHE[key]

    loss_sums = [0.0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = ("Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, "
               "Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}").format(
        *([step] + loss_means))
    return message, loss_means


if __name__ == "__main__":
    from utils.model import get_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    _, configs = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.ckpt, configs, dev, train=False)
    print(evaluate(model, 0, configs, device=dev)[0])
