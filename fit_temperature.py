#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit temperature (T) for post-hoc calibration using a held-out calibration set.

This code is intentionally aligned with your existing eval_whole_hubert_embed.py:
- Loads the same model + ckpt
- Iterates the same datalist format
- Calls model.evaluate(...) to get det_result (currently sigmoid probability)
- Converts prob -> logit, then learns a single scalar temperature T by minimizing
  BCEWithLogitsLoss(sigmoid(logit/T), y_release) on the calibration set.

Output:
- A .pt file containing {"temperature": float(T), "ckpt": ..., "config": ..., "datalist": ...}
"""

import argparse
import json
import os
import re
import yaml
import torch

from yamlinclude import YamlIncludeConstructor
from local.utils import read_list
from data.loader.data_utils import unfold_list
from model.TransformerKWSPhone_hubert_wenet_embed import TransformerKWSPhone_hubert_wenet_embed

# Keep consistent with your eval script
PATTERN = re.compile('^.*?LibriSpeech/')

def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert probability p in (0,1) to logit with clamping for numerical safety."""
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def fit_temperature_from_logits(logits: torch.Tensor, y: torch.Tensor, device: str = "cpu") -> float:
    """
    Learn temperature T > 0 by minimizing BCEWithLogitsLoss(logits / T, y).
    logits: (N,) float tensor
    y:      (N,) float tensor in {0,1}
    """
    logits = logits.detach().to(device).float().view(-1)
    y = y.detach().to(device).float().view(-1)

    logT = torch.zeros(1, device=device, requires_grad=True)  # T = exp(logT) > 0
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    crit = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = torch.exp(logT)
        loss = crit(logits / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = torch.exp(logT).detach().item()
    return float(T)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Same yaml used for training/eval")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (.pt). Must contain key 'model'.")
    parser.add_argument("--datalist", required=True, help="Calibration datalist file (json lines).")
    parser.add_argument("--out", required=True, help="Output .pt file to save temperature.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 etc. (Your eval uses cpu by default)")
    parser.add_argument("--n_word", type=int, default=4, help="Keep consistent with eval script sample_all_keywords usage")
    args = parser.parse_args()

    # Load config
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    #YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.dirname(args.config))
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_state_dict = ckpt["model"]
    model_config = config["model_config"]
    model = TransformerKWSPhone_hubert_wenet_embed(**model_config)
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(args.device)

    # We reuse the same sampling function from eval_whole_hubert_embed.py by importing it dynamically.
    # To keep changes minimal, we inline a small import trick here:
    from eval_whole_hubert_embed import sample_all_keywords  # noqa

    tr_list = read_list(args.datalist, split_cv=False, shuffle=False)

    all_logits = []
    all_y = []

    for i, one_test_obj in enumerate(tr_list):
        if i % 1000 == 0:
            print(f"[calib] processing {i}/{len(tr_list)}")
        one_test_obj = json.loads(one_test_obj)

        wav_path = one_test_obj["sph"]
        phn_label = one_test_obj["phn_label"]
        md_label = one_test_obj.get("md_label", None)
        negative_candidate = one_test_obj.get("negative_candidate", None)

        # Your eval loads pre-saved tensors as waveform/feature
        wav = torch.load(wav_path)
        speech = wav.unsqueeze(0).to("cpu")
        speech_len = torch.tensor([speech.size(1)], device="cpu")

        fbank_feats = speech
        fbank_len = speech_len
        if fbank_len < 7:
            continue

        # Sample multiple keywords like eval script
        aug_keywords, aug_keyword_lens, aug_md_labels, _ = sample_all_keywords(
            phn_label, md_label, negative_candidate, n_word=args.n_word
        )

        for j in range(len(aug_keywords)):
            aug_keyword = aug_keywords[j]
            aug_keyword_len = aug_keyword_lens[j]
            aug_md_label = aug_md_labels[j]  # this should be the "gold" md label on this sample

            input_data = (fbank_feats, fbank_len, aug_keyword, aug_keyword_len, aug_md_label)

            with torch.no_grad():
                det_prob, _ = model.evaluate(input_data)   # det_prob is already sigmoid output in current model
                det_prob = det_prob.view(-1).to(args.device)

            y = aug_md_label.view(-1).to(args.device).float()
            if y.numel() != det_prob.numel():
                # Safety: skip unexpected shape mismatch
                continue

            logits = safe_logit(det_prob)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())

    if len(all_logits) == 0:
        raise RuntimeError("No calibration samples collected. Check datalist / md_label field.")

    logits_calib = torch.cat(all_logits, dim=0)
    y_calib = torch.cat(all_y, dim=0)

    print(f"[calib] total tokens: {logits_calib.numel()}")

    T = fit_temperature_from_logits(logits_calib, y_calib, device=args.device)
    print(f"[calib] learned temperature T = {T:.6f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "temperature": T,
            "ckpt": args.ckpt,
            "config": args.config,
            "datalist": args.datalist,
            "note": "Temperature scaling on logits (logit(sigmoid_output)) using release/collective md_label as gold.",
        },
        args.out,
    )
    print(f"[calib] saved to {args.out}")

if __name__ == "__main__":
    main()
