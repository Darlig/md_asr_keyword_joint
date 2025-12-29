#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path
from itertools import zip_longest


# -------------------- basic clean --------------------

def _clean(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\x00", "").replace("\ufeff", "").strip()


# -------------------- TextGrid robust parser --------------------

def parse_textgrid_tiers(path: Path):
    """
    Robust Praat ooTextFile TextGrid IntervalTier parser.
    Return: dict[tier_name] = [(xmin, xmax, text), ...]
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\x00", "").replace("\ufeff", "")

    tiers = {}

    # Match each IntervalTier block (until next item[k] or EOF)
    tier_pat = re.compile(
        r'item\s*\[\d+\]:\s*'
        r'class\s*=\s*"IntervalTier"\s*'
        r'name\s*=\s*"(?P<name>[^"]+)"\s*'
        r'(?P<body>.*?)(?=\n\s*item\s*\[\d+\]:|\Z)',
        flags=re.DOTALL
    )

    # Match each intervals[k] with xmin/xmax/text
    interval_pat = re.compile(
        r'intervals\s*\[\d+\]:\s*'
        r'xmin\s*=\s*(?P<xmin>[-\d\.eE]+)\s*'
        r'xmax\s*=\s*(?P<xmax>[-\d\.eE]+)\s*'
        r'text\s*=\s*"(?P<text>[^"]*)"\s*',
        flags=re.DOTALL
    )

    for tm in tier_pat.finditer(raw):
        name = tm.group("name").strip()
        body = tm.group("body")

        intervals = []
        for im in interval_pat.finditer(body):
            xmin = float(im.group("xmin"))
            xmax = float(im.group("xmax"))
            text = _clean(im.group("text"))
            intervals.append((xmin, xmax, text))

        tiers[name] = intervals

    return tiers


# -------------------- Canonical PY parsing --------------------

_INITIALS = sorted(
    ["zh", "ch", "sh",
     "b", "p", "m", "f",
     "d", "t", "n", "l",
     "g", "k", "h",
     "j", "q", "x",
     "r", "z", "c", "s",
     "y", "w"],
    key=len,
    reverse=True
)

def split_canonical_py_token(tok: str):
    """
    PY token format: "{init}{final}{tone}" like "dao1" "e2" "zhong1" ...
    Output ALWAYS length-2: [init, final+tone] where init may be "".
    """
    tok = _clean(tok).lower().replace("u:", "ü").replace("v", "ü")
    if not tok:
        return ["", ""]

    m = re.fullmatch(r"([a-zü]+)([0-5])", tok)
    if not m:
        # fallback: no tone etc. keep as final
        return ["", tok]

    core, tone = m.group(1), m.group(2)

    init = ""
    for ini in _INITIALS:
        if core.startswith(ini):
            init = ini
            break

    final = core[len(init):] + tone
    return [init, final]

def parse_py_interval(text: str):
    """
    Interval PY text: "{tok} {tok} ..."
    Each tok -> [init, final+tone], flattened.
    """
    text = _clean(text)
    if not text:
        return []
    phones = []
    for tok in re.split(r"\s+", text):
        phones.extend(split_canonical_py_token(tok))
    return phones


# -------------------- Actual PYBZ parsing --------------------

def _split_cano_actual(part: str):
    """
    part is one of the 3 pieces after splitting by ~ and ~~:
      - "d"        -> (cano="d", actual="d")
      - "d(t)"     -> (cano="d", actual="t")
      - "d(*)"     -> (cano="d", actual="*")
      - ""         -> ("","")
      - "(*)"      -> ("","*")  (rare but handle)
    """
    part = _clean(part)

    m = re.fullmatch(r'(?P<cano>[^()]*)\((?P<act>[^()]*)\)', part)
    if m:
        cano = _clean(m.group("cano"))
        act = _clean(m.group("act"))
        if act == "*":
            return cano, "*"
        return cano, act

    # no parentheses: actual = canonical
    return part, part

def parse_pybz_token(tok: str):
    """
    PYBZ token rule (your spec):
      split by "~" and "~~" into 3 parts:
        init_part ~ final_part ~~ tone_part
      each part may have parentheses: cano(act)
      if parentheses exist -> actual is inside parentheses
      else -> actual = cano

    phones_actual = [init_actual, final_actual + tone_actual]
    - init_actual can be "" and MUST be kept.
    - if final_actual == "*" OR tone_actual == "*" => merged final_with_tone = "*"
    """
    tok = _clean(tok)
    if not tok:
        return ["", ""]

    if "~~" not in tok or "~" not in tok:
        # cannot parse -> keep placeholders
        return ["", ""]

    left, tone_part = tok.split("~~", 1)
    # left might contain extra "~" in weird cases; take first split only
    init_part, final_part = left.split("~", 1)

    _, init_a = _split_cano_actual(init_part)
    _, fin_a  = _split_cano_actual(final_part)
    _, tone_a = _split_cano_actual(tone_part)

    init_a = _clean(init_a).lower().replace("u:", "ü").replace("v", "ü")
    fin_a  = _clean(fin_a).lower().replace("u:", "ü").replace("v", "ü")
    tone_a = _clean(tone_a)

    if fin_a == "*" or tone_a == "*":
        final_with_tone = "*"
    else:
        final_with_tone = f"{fin_a}{tone_a}"

    return [init_a, final_with_tone]

def parse_pybz_interval(text: str):
    """
    Interval PYBZ text: "{token} {token} ..."
    Each token -> [init_actual, final+tone], flattened.
    """
    text = _clean(text)
    if not text:
        return []
    phones = []
    for tok in re.split(r"\s+", text):
        phones.extend(parse_pybz_token(tok))
    return phones


# -------------------- build output --------------------

def phones_accuracy(canon_phones, actual_phones):
    """
    Compare per-phone position. '*' always treated as incorrect (0).
    If lengths differ -> missing positions are 0.
    """
    acc = []
    for c, a in zip_longest(canon_phones, actual_phones, fillvalue=None):
        if c is None or a is None:
            acc.append(0)
        elif c == "*" or a == "*":
            acc.append(0)
        else:
            acc.append(1 if c == a else 0)
    return acc

def build_entry_for_textgrid(path: Path, hz_tier="HZ", py_tier="PY", pybz_tier="PYBZ"):
    tiers = parse_textgrid_tiers(path)
    hz = tiers.get(hz_tier, [])
    py = tiers.get(py_tier, [])
    pybz = tiers.get(pybz_tier, [])

    n = min(len(hz), len(py), len(pybz))
    words = []
    full_text_parts = []

    for i in range(n):
        _, _, hz_text = hz[i]
        _, _, py_text = py[i]
        _, _, pybz_text = pybz[i]

        word_text = _clean(hz_text).replace("\x00", "")
        word_text = re.sub(r"\s+", "", word_text)

        canon_phones = parse_py_interval(py_text)
        actual_phones = parse_pybz_interval(pybz_text)

        acc = phones_accuracy(canon_phones, actual_phones)

        words.append({
            "text": word_text,
            "phones": canon_phones,
            "phones_actual": actual_phones,
            "phones-accuracy": acc
        })

        if word_text:
            full_text_parts.append(word_text)

    return {
        "text": "".join(full_text_parts),
        "words": words
    }

def batch_parse(root_dir: str, out_json: str, pattern_ext=(".TextGrid", ".textgrid")):
    root = Path(root_dir)
    tg_files = []
    for ext in pattern_ext:
        tg_files.extend(root.rglob(f"*{ext}"))
    tg_files = sorted(tg_files)

    results = {}
    for p in tg_files:
        uttid = p.stem
        try:
            results[uttid] = build_entry_for_textgrid(p)
        except Exception as e:
            results[uttid] = {"error": str(e), "path": str(p)}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(tg_files)} TextGrids -> {out_json}")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data", help="root directory, e.g. data/")
    ap.add_argument("--out", type=str, default="parsed_textgrid.json", help="output json path")
    args = ap.parse_args()
    batch_parse(args.root, args.out)

if __name__ == "__main__":
    main()
