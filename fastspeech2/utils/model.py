import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


HIFIGAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hifigan")


def load_checkpoint(path, device="cpu"):
    # MODIFIED: torch>=2.6 defaults to weights_only=True, which rejects these pickles
    return torch.load(path, map_location=device, weights_only=False)


def get_model(ckpt_path, configs, device, train=False):
    """MODIFIED: takes an explicit checkpoint path (or None) instead of args.restore_step.

    Returns (model, optimizer, step) when train=True, else model.
    Accepts both full training checkpoints and fp16 inference exports.
    """
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt, step = None, 0
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, device)
        state = {k: v.float() if torch.is_floating_point(v) else v
                 for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        step = int(ckpt.get("step", 0))

    if train:
        scheduled_optim = ScheduledOptim(model, train_config, model_config, step)
        if ckpt is not None and "optimizer" in ckpt:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim, step

    model.eval()
    model.requires_grad_(False)  # MODIFIED: upstream assigned the attribute, a no-op
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        # MODIFIED: paths relative to this package, auto-unzip, map_location
        with open(os.path.join(HIFIGAN_DIR, "config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        fname = "generator_{}.pth.tar".format(speaker)
        path = os.path.join(HIFIGAN_DIR, fname)
        if not os.path.exists(path):
            import zipfile
            with zipfile.ZipFile(path + ".zip") as z:
                z.extractall(HIFIGAN_DIR)
        ckpt = load_checkpoint(path, device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
