#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit temperature (T) for post-hoc calibration using a held-out calibration set.

This code is intentionally aligned with your existing eval_whole_hubert_embed.py:
- Loads the same model + ckpt
- Iterates the same datalist format
- Calls model.evaluate_multi(...) (if available) to get 3 sigmoid posteriors (one per annotator head).
  Falls back to model.evaluate(...) for legacy single-head checkpoints.
- Converts prob -> logit, then learns temperature(s) by minimizing BCEWithLogitsLoss(logits/T, y) on a
  held-out calibration set.

Modes:
- heads (default): fit 3 temperatures for head1/head2/head3 against md_label1/2/3 (fallback: md_label)
- mean: fit a single temperature for the mean posterior against a chosen gold label key

Output:
- A .pt file containing {'temperature': float or [float,float,float], 'mode': str, ...}
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
from model.TransformerKWSPhone_hubert_wenet_embed_3md_adaptor import TransformerKWSPhone_hubert_wenet_embed_3md_adaptor

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
    parser.add_argument("--mode", default="heads", choices=["heads", "mean", "head1", "head2", "head3"],
                        help="Calibration target: fit per-head temperatures (heads) or a single temperature for mean posterior (mean).")
    parser.add_argument("--gold_key", default="md_label",
                        help="For --mode=mean: which label field to use as gold (e.g., md_label for voting/release).")
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
    model = TransformerKWSPhone_hubert_wenet_embed_3md_adaptor(**model_config)
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(args.device)

    # We reuse the same sampling function from eval_whole_hubert_embed.py by importing it dynamically.
    # To keep changes minimal, we inline a small import trick here:
    from eval_whole_hubert_embed import sample_all_keywords  # noqa

    tr_list = read_list(args.datalist, split_cv=False, shuffle=False)

    
    # Collect logits + labels for temperature fitting
    # - heads: fit 3 temps for md_label1/2/3 (fallback md_label)
    # - mean/headk: fit 1 temp for mean or a selected head
    if args.mode == "heads":
        all_logits = [[], [], []]
        all_y = [[], [], []]
    else:
        all_logits = []
        all_y = []

    for i, one_test_obj in enumerate(tr_list):
        if i % 1000 == 0:
            print(f"[calib] processing {i}/{len(tr_list)}")
        one_test_obj = json.loads(one_test_obj)
    
        wav_path = one_test_obj["sph"]
        phn_label = one_test_obj["phn_label"]
        negative_candidate = one_test_obj.get("negative_candidate", None)
    
        # md labels (1/2/3) for multi-head; fallback to md_label for backward compatibility
        md_label = one_test_obj.get("md_label", None)
        md_label1 = one_test_obj.get("md_label1", md_label)
        md_label2 = one_test_obj.get("md_label2", md_label)
        md_label3 = one_test_obj.get("md_label3", md_label)
    
        # Your eval loads pre-saved tensors as waveform/feature
        wav = torch.load(wav_path)
        speech = wav.unsqueeze(0).to("cpu")
        speech_len = torch.tensor([speech.size(1)], device="cpu")
    
        fbank_feats = speech
        fbank_len = speech_len
        if fbank_len < 7:
            continue

        # Sample multiple keywords like eval script (deterministic when md_label is provided)
        if args.mode == "heads":
            aug_keywords, aug_keyword_lens, aug_md_labels1, _ = sample_all_keywords(
                phn_label, md_label1, negative_candidate, n_word=args.n_word
            )
            _, _, aug_md_labels2, _ = sample_all_keywords(
                phn_label, md_label2, negative_candidate, n_word=args.n_word
            )
            _, _, aug_md_labels3, _ = sample_all_keywords(
                phn_label, md_label3, negative_candidate, n_word=args.n_word
            )
        else:
            gold_md_label = one_test_obj.get(args.gold_key, md_label)
            aug_keywords, aug_keyword_lens, aug_md_labels, _ = sample_all_keywords(
                phn_label, gold_md_label, negative_candidate, n_word=args.n_word
            )

        for j in range(len(aug_keywords)):
            aug_keyword = aug_keywords[j]
            aug_keyword_len = aug_keyword_lens[j]

            # feed a dummy md_label (not used in evaluate/evaluate_multi, but kept for API consistency)
            if args.mode == "heads":
                dummy_md = aug_md_labels1[j]
            else:
                dummy_md = aug_md_labels[j]

            input_data = (fbank_feats, fbank_len, aug_keyword, aug_keyword_len, dummy_md)

            with torch.no_grad():
                if hasattr(model, "evaluate_multi"):
                    det1, det2, det3, _ = model.evaluate_multi(input_data)
                    det_probs = [
                        det1.view(-1).to(args.device),
                        det2.view(-1).to(args.device),
                        det3.view(-1).to(args.device),
                    ]
                    det_mean = sum(det_probs) / 3.0
                else:
                    det_prob, _ = model.evaluate(input_data)
                    det_prob = det_prob.view(-1).to(args.device)
                    det_probs = [det_prob, det_prob, det_prob]
                    det_mean = det_prob

            if args.mode == "heads":
                y_list = [
                    aug_md_labels1[j].view(-1).to(args.device).float(),
                    aug_md_labels2[j].view(-1).to(args.device).float(),
                    aug_md_labels3[j].view(-1).to(args.device).float(),
                ]
                for k in range(3):
                    if y_list[k].numel() != det_probs[k].numel():
                        continue
                    logits = safe_logit(det_probs[k])
                    all_logits[k].append(logits.detach().cpu())
                    all_y[k].append(y_list[k].detach().cpu())
            else:
                if args.mode == "mean":
                    y = aug_md_labels[j].view(-1).to(args.device).float()
                    p = det_mean
                elif args.mode == "head1":
                    y = aug_md_labels[j].view(-1).to(args.device).float()
                    p = det_probs[0]
                elif args.mode == "head2":
                    y = aug_md_labels[j].view(-1).to(args.device).float()
                    p = det_probs[1]
                elif args.mode == "head3":
                    y = aug_md_labels[j].view(-1).to(args.device).float()
                    p = det_probs[2]
                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                if y.numel() != p.numel():
                    continue
                logits = safe_logit(p)
                all_logits.append(logits.detach().cpu())
                all_y.append(y.detach().cpu())

    if args.mode == "heads":
        if len(all_logits[0]) == 0 or len(all_logits[1]) == 0 or len(all_logits[2]) == 0:
            raise RuntimeError("No calibration samples collected for one or more heads. Check datalist md_label1/2/3 fields.")
    else:
    
        #0:
            raise RuntimeError("No calibration samples collected. Check datalist / md_label field.")
    
    
    if args.mode == "heads":
        temps = []
        for k in range(3):
            logits_calib = torch.cat(all_logits[k], dim=0)
            y_calib = torch.cat(all_y[k], dim=0)
            print(f"[calib] head{k+1}: total tokens: {logits_calib.numel()}")
            T = fit_temperature_from_logits(logits_calib, y_calib, device=args.device)
            temps.append(float(T))
            print(f"[calib] head{k+1}: learned temperature T = {float(T):.6f}")
        temperature_out = temps
    else:
        logits_calib = torch.cat(all_logits, dim=0)
        y_calib = torch.cat(all_y, dim=0)
        print(f"[calib] total tokens: {logits_calib.numel()}")
        T = fit_temperature_from_logits(logits_calib, y_calib, device=args.device)
        print(f"[calib] learned temperature T = {T:.6f}")
        temperature_out = float(T)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "temperature": temperature_out,
            "mode": args.mode,
            "gold_key": args.gold_key,
            "ckpt": args.ckpt,
            "config": args.config,
            "datalist": args.datalist,
            "note": "Temperature scaling on logits (logit(sigmoid_output)). For mode=heads uses md_label1/2/3 (fallback md_label); for mode=mean uses gold_key.",
        },
        args.out,
    )
    print(f"[calib] saved to {args.out}")

if __name__ == "__main__":
    main()
