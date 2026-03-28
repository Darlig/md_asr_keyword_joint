#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit temperature (T) for post-hoc calibration using a held-out calibration set.

This code is intentionally aligned with your existing eval_whole_hubert_embed.py:
- Loads the same model + ckpt
- Iterates the same datalist format
- Calls model.evaluate(...) from the unified model to get 1..N sigmoid posteriors.
- Converts prob -> logit, then learns temperature(s) by minimizing BCEWithLogitsLoss(logits/T, y) on a
  held-out calibration set.

Modes:
- heads (default): fit N temperatures for head1..headN against md_label1..md_labelN (fallback: md_label)
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
from model.TransformerKWSPhone_hubert_wenet import TransformerKWSPhone_hubert_wenet

# Keep consistent with your eval script
PATTERN = re.compile('^.*?LibriSpeech/')


def build_embed_model(model_config, model_state_dict):
    model_config = dict(model_config)
    model_config["input_mode"] = "embedding"
    model = TransformerKWSPhone_hubert_wenet(**model_config)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model


def unpack_eval_outputs(eval_outputs):
    if not isinstance(eval_outputs, tuple) or len(eval_outputs) < 2:
        raise ValueError("Unexpected evaluate() outputs")
    det_outputs = [det.view(-1) for det in eval_outputs[:-1]]
    hyp = eval_outputs[-1]
    return det_outputs, hyp


def collect_md_labels(sample):
    md_label = sample.get("md_label", None)
    md_label_items = []
    for key in sorted(sample.keys()):
        matched = re.fullmatch(r"md_label(\d+)", key)
        if matched:
            md_label_items.append((int(matched.group(1)), sample[key]))
    if not md_label_items and md_label is not None:
        md_label_items.append((1, md_label))
    return [label for _, label in md_label_items]

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
    parser.add_argument("--mode", default="heads",
                        help="Calibration target: heads / mean / headK (e.g. head1, head2, ...).")
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
    model = build_embed_model(model_config, model_state_dict)
    model.to(args.device)
    num_heads = int(getattr(model, "md_num_heads", 1))

    # We reuse the same sampling function from eval_whole_hubert_embed.py by importing it dynamically.
    # To keep changes minimal, we inline a small import trick here:
    from eval_whole_hubert_embed import sample_all_keywords  # noqa

    tr_list = read_list(args.datalist, split_cv=False, shuffle=False)

    if args.mode == "heads":
        all_logits = [[] for _ in range(num_heads)]
        all_y = [[] for _ in range(num_heads)]
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
        md_labels = collect_md_labels(one_test_obj)
        if not md_labels:
            raise RuntimeError("No md_label found in calibration sample.")
        if len(md_labels) < num_heads:
            md_labels.extend([md_labels[-1]] * (num_heads - len(md_labels)))
        md_labels = md_labels[:num_heads]
    
        # Your eval loads pre-saved tensors as waveform/feature
        wav = torch.load(wav_path)
        speech = wav.unsqueeze(0).to("cpu")
        speech_len = torch.tensor([speech.size(1)], device="cpu")
    
        fbank_feats = speech
        fbank_len = speech_len
        if fbank_len < 7:
            continue

        if args.mode == "heads":
            per_head_md_labels = []
            aug_keywords = aug_keyword_lens = None
            for md_label_one in md_labels:
                one_keywords, one_keyword_lens, one_md_labels, _ = sample_all_keywords(
                    phn_label, md_label_one, negative_candidate, n_word=args.n_word
                )
                if aug_keywords is None:
                    aug_keywords = one_keywords
                    aug_keyword_lens = one_keyword_lens
                per_head_md_labels.append(one_md_labels)
        else:
            gold_md_label = one_test_obj.get(args.gold_key, one_test_obj.get("md_label", md_labels[0]))
            aug_keywords, aug_keyword_lens, aug_md_labels, _ = sample_all_keywords(
                phn_label, gold_md_label, negative_candidate, n_word=args.n_word
            )

        for j in range(len(aug_keywords)):
            aug_keyword = aug_keywords[j]
            aug_keyword_len = aug_keyword_lens[j]

            if args.mode == "heads":
                dummy_md = per_head_md_labels[0][j]
            else:
                dummy_md = aug_md_labels[j]

            input_data = (fbank_feats, fbank_len, aug_keyword, aug_keyword_len, dummy_md)

            with torch.no_grad():
                det_probs, _ = unpack_eval_outputs(model.evaluate(input_data))
                det_probs = [det_prob.to(args.device) for det_prob in det_probs]
                det_mean = torch.stack(det_probs, dim=0).mean(dim=0)

            if args.mode == "heads":
                y_list = [per_head_md_labels[k][j].view(-1).to(args.device).float() for k in range(num_heads)]
                for k in range(num_heads):
                    if y_list[k].numel() != det_probs[k].numel():
                        continue
                    logits = safe_logit(det_probs[k])
                    all_logits[k].append(logits.detach().cpu())
                    all_y[k].append(y_list[k].detach().cpu())
            else:
                if args.mode == "mean":
                    y = aug_md_labels[j].view(-1).to(args.device).float()
                    p = det_mean
                elif args.mode.startswith("head"):
                    head_idx = int(args.mode[4:]) - 1
                    if head_idx < 0 or head_idx >= num_heads:
                        raise ValueError(f"Requested {args.mode}, but model has {num_heads} head(s).")
                    y = aug_md_labels[j].view(-1).to(args.device).float()
                    p = det_probs[head_idx]
                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                if y.numel() != p.numel():
                    continue
                logits = safe_logit(p)
                all_logits.append(logits.detach().cpu())
                all_y.append(y.detach().cpu())

    if args.mode == "heads":
        if any(len(one_head_logits) == 0 for one_head_logits in all_logits):
            raise RuntimeError("No calibration samples collected for one or more heads. Check datalist md_label1..md_labelN fields.")
    else:
        if len(all_logits) == 0:
            raise RuntimeError("No calibration samples collected. Check datalist / md_label field.")

    if args.mode == "heads":
        temps = []
        for k in range(num_heads):
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
            "note": "Temperature scaling on logits (logit(sigmoid_output)). For mode=heads uses md_label1..md_labelN (fallback md_label); for mode=mean uses gold_key.",
        },
        args.out,
    )
    print(f"[calib] saved to {args.out}")

if __name__ == "__main__":
    main()
