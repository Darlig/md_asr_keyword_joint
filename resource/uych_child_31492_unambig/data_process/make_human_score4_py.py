#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path
from itertools import zip_longest
from collections import Counter


# -------------------- basic clean --------------------


def _clean(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\x00", "").replace("\ufeff", "").strip()



def _strip_dash_markers(s: str) -> str:
    """Strip common PYBZ marker dashes that should not affect canonical/actual comparisons."""
    s = _clean(s)
    # Many PYBZ strings use leading/trailing '-' as a marker.
    return s.strip("-")


# --- Canonicalize phone pieces by removing all whitespace and common invisible/separator chars ---
def _squeeze_piece(s: str) -> str:
    """Remove internal whitespace and common invisible separators from a phone piece."""
    s = _clean(s)
    if not s:
        return ""
    # Remove all unicode whitespace inside the string
    s = re.sub(r"\s+", "", s)
    # Remove common invisible chars that sometimes sneak in
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return s


# -------------------- canonical mismatch stats --------------------


# Key: (py_init, py_final, py_tone, pybz_init, pybz_final, pybz_tone)
CANON_PY_VS_PYBZ_MISMATCH = Counter()

# Store up to 3 example uttids for each canonical mismatch key.
CANON_PY_VS_PYBZ_MISMATCH_EXAMPLES = {}

# Collect raw examples for y/w (wan/yan-like) mismatches for debugging.
YW_MISMATCH_SAMPLES = []

# Context set by build_entry_for_textgrid
_CUR_UTTID = ""
_CUR_WORD_IDX = -1
_CUR_WORD_TEXT = ""


def _fmt_cano(init_: str, fin: str, tone: str) -> str:
    """Format a 3-part canonical phone as `init~final~~tone` (tone may be '')."""
    init_ = _clean(init_)
    fin = _clean(fin)
    tone = _clean(tone)
    return f"{init_}~{fin}~~{tone}"


# -------------------- normalization rules (PY/PYBZ) --------------------

def _split_final_tone_keep_empty(final_with_tone: str):
    """'ao1' -> ('ao','1'); if malformed return (final_with_tone, '')"""
    s = _clean(final_with_tone)
    m = re.fullmatch(r"(.+?)([0-5])", s)
    if m:
        return m.group(1), m.group(2)
    return s, ""


# ----------- Helper: conditional zero-initial completion to match other side -----------
def _maybe_complete_zero_initial_to_match(other_ini: str, other_finaltone: str, empty_ini: str, empty_finaltone: str):
    """Conditionally complete a zero-initial canonical to match the other side.

    Only used when exactly one side has empty initial and the other side has non-empty initial.

    Rules (apply only if completion makes (ini, final-without-tone) exactly match the other side):
      1) empty init + final startswith 'i': add init 'y' and drop leading 'i' (except final=='i' -> keep 'i')
      2) empty init + final startswith 'v': add init 'y' and replace leading 'v' with 'u'
      3) empty init + final in {'uo','ui'}: add init 'w' and map 'uo'->'o', 'ui'->'ei'
      4) empty init + final startswith 'u': add init 'w' and drop leading 'u' (except final=='u' -> keep 'u')

    Return: (new_ini, new_finaltone, changed_bool)
    """
    other_ini = _clean(other_ini).lower()
    empty_ini = _clean(empty_ini).lower()

    # Only attempt when empty side is truly empty
    if empty_ini != "":
        return empty_ini, empty_finaltone, False

    other_fin, _other_tone = _split_final_tone_keep_empty(_clean(other_finaltone).lower())
    empty_fin, empty_tone = _split_final_tone_keep_empty(_clean(empty_finaltone).lower())

    other_fin = _clean(other_fin).lower()
    empty_fin = _clean(empty_fin).lower()

    def _pack(fin: str, tone: str) -> str:
        fin = _clean(fin).lower()
        tone = _clean(tone)
        return f"{fin}{tone}" if tone else fin

    # 0) Direct initial fill: if finals (without tone) already match, just copy the other initial.
    # Apply only if it makes (ini, final-without-tone) match exactly.
    if empty_fin == other_fin and other_ini != "":
        cand_ini = other_ini
        cand_fin = empty_fin
        if cand_ini == other_ini and cand_fin == other_fin:
            return cand_ini, _pack(cand_fin, empty_tone), True

    # 1) i- series -> y + (drop leading i), except fin=='i'
    if empty_fin.startswith("i"):
        cand_ini = "y"
        cand_fin = empty_fin if empty_fin == "i" else empty_fin[1:]
        if cand_ini == other_ini and cand_fin == other_fin:
            return cand_ini, _pack(cand_fin, empty_tone), True

    # 2) v- series -> y + (replace leading v with u)
    if empty_fin.startswith("v"):
        cand_ini = "y"
        cand_fin = "u" + empty_fin[1:]
        if cand_ini == other_ini and cand_fin == other_fin:
            return cand_ini, _pack(cand_fin, empty_tone), True

    # 3) special: uo/ui -> w + (uo->o, ui->ei)
    if empty_fin in {"uo", "ui"}:
        cand_ini = "w"
        cand_fin = "o" if empty_fin == "uo" else "ei"
        if cand_ini == other_ini and cand_fin == other_fin:
            return cand_ini, _pack(cand_fin, empty_tone), True

    # 4) u- series -> w + (drop leading u), except fin=='u'
    if empty_fin.startswith("u"):
        cand_ini = "w"
        cand_fin = empty_fin if empty_fin == "u" else empty_fin[1:]
        if cand_ini == other_ini and cand_fin == other_fin:
            return cand_ini, _pack(cand_fin, empty_tone), True

    return empty_ini, empty_finaltone, False


def _rm_erhua_r(fin: str, ini: str) -> str:
    """Remove erhua trailing 'r' from finals.

    Keep 'er' ONLY for the special case where the initial is empty (zero-initial `~er`).
    For non-empty initials, treat `er` as erhua and drop the trailing `r` -> `e`.
    """
    fin = _clean(fin).lower()
    ini = _clean(ini).lower()

    # Keep the r in `~er` (zero initial) as requested.
    if fin == "er" and ini == "":
        return fin

    # Otherwise, remove erhua trailing r.
    if fin.endswith("r"):
        return fin[:-1]
    return fin


def _ii_iii_to_i(fin: str) -> str:
    fin = _clean(fin).lower()
    # replace iii first, then ii
    fin = fin.replace("iii", "i").replace("ii", "i")
    return fin


def _v_rule(fin: str, ini: str) -> str:
    """v -> u only when ini in {j,q,x,y}; otherwise keep v/u unchanged."""
    fin = _clean(fin).lower()
    ini = _clean(ini).lower()
    if ini in {"j", "q", "x", "y"}:
        return fin.replace("v", "u")
    return fin


def _zero_initial_to_yw(ini: str, fin: str, preferred: str = ""):
    """
    Map zero-initial (ini=='') to y/w-initial version.
    - 开头就是~ => ini==''，按零声母方案处理
    - a/e/o 开头也要处理，否则 ~ang vs y~ang 仍会 mismatch
    """
    ini = _clean(ini).lower()
    fin = _clean(fin).lower()
    if ini != "":
        return ini, fin
    
    preferred = _clean(preferred).lower()
    if preferred not in {"", "y", "w"}:
        preferred = ""

    # 特例：bare 'o' 更常映射成 wo
    if fin == "o":
        return "w", "o"

    # Special case: keep zero-initial `~er` as-is (do NOT map to y/w)
    if fin == "er":
        return "", "er"

    # ü-series (written as v...) -> y + v...
    if fin.startswith("v"):
        return "y", fin

    # a/e/o-series are ambiguous for zero-initial syllables in this dataset.
    # Prefer PY's initial (y/w) if provided; otherwise default to 'y'.
    if fin and fin[0] in {"a", "e", "o"}:
        if preferred in {"y", "w"}:
            return preferred, fin
        return "y", fin

    # i-series
    if fin == "i":
        return "y", "i"
    if fin.startswith("i") and len(fin) > 1:
        m = {
            "ia": "a",
            "ian": "an",
            "iang": "ang",
            "iao": "ao",
            "ie": "e",
            "iong": "ong",
            "iu": "ou",
        }
        if fin in m:
            return "y", m[fin]
        # fallback: drop leading i
        return "y", fin[1:]

    # u-series
    if fin == "u":
        return "w", "u"
    if fin.startswith("u") and len(fin) > 1:
        m = {
            "ua": "a",
            "uai": "ai",
            "uan": "an",
            "uang": "ang",
            "ue": "e",
            "ui": "ei",
            "un": "en",
            "uo": "o",
        }
        if fin in m:
            return "w", m[fin]
        # fallback: drop leading u
        return "w", fin[1:]

    return ini, fin


def normalize_pair(ini: str, finaltone: str, preferred_zero_initial: str = "", do_zero_initial_map: bool = True):
    """
    Normalize (ini, fin+tone) with the user's rules:
    - ~ / ~~ split already done outside; ini=='' is allowed
    - v: only after j q x y => v->u; else v stays v; u stays u
    - ii/iii (in final) -> i
    - erhua trailing r removed; keep 'er' ONLY for zero-initial (`~er`)
    - zero initial mapped to y/w version only when do_zero_initial_map=True
    - strip dash markers '-' everywhere
    - normalize ü -> v (then apply v_rule)
    """
    ini = _squeeze_piece(_strip_dash_markers(_clean(ini))).lower()
    finaltone = _squeeze_piece(_strip_dash_markers(_clean(finaltone))).lower()

    # Preserve '*' for the piece that is '*', but still normalize the other piece.
    ini_is_star = (ini == "*")
    ft_is_star = (finaltone == "*")

    if ft_is_star:
        # Only initial may be meaningful; keep '*' finaltone.
        return ini, "*"

    # For rules that depend on initial, keep '*' as non-empty (so `er` is treated as erhua and `r` is removed).
    ini_for_rules = ini

    fin, tone = _split_final_tone_keep_empty(finaltone)

    fin = _squeeze_piece(_strip_dash_markers(fin))
    tone = _squeeze_piece(_strip_dash_markers(tone))

    fin = fin.replace("ü", "v")  # unify ü into v system first

    fin = _ii_iii_to_i(fin)

    # Hard guard: if final is exactly 'er' and initial is non-empty, treat it as erhua and drop 'r' -> 'e'.
    # Keep 'er' only for zero-initial (~er).
    if fin == "er" and ini_for_rules != "":
        fin = "e"
    else:
        fin = _rm_erhua_r(fin, ini_for_rules)

    # Optional but safe: ensure any lingering trailing 'r' after the guard is removed for non-zero-initials.
    if ini_for_rules != "" and fin.endswith("r"):
        fin = fin[:-1]

    if do_zero_initial_map:
        ini2, fin2 = _zero_initial_to_yw(ini_for_rules, fin, preferred=preferred_zero_initial)
    else:
        ini2, fin2 = ini_for_rules, fin

    fin2 = _v_rule(fin2, ini2)

    if ini_is_star:
        ini2 = "*"

    if tone:
        return ini2, f"{fin2}{tone}"
    return ini2, fin2


# -------------------- TextGrid robust parser --------------------

def parse_textgrid_tiers(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\x00", "").replace("\ufeff", "")

    tiers = {}

    tier_pat = re.compile(
        r'item\s*\[\d+\]:\s*'
        r'class\s*=\s*"IntervalTier"\s*'
        r'name\s*=\s*"(?P<name>[^"]+)"\s*'
        r'(?P<body>.*?)(?=\n\s*item\s*\[\d+\]:|\Z)',
        flags=re.DOTALL
    )

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
    tok = _clean(tok).lower()
    if not tok:
        return ["", ""]

    m = re.fullmatch(r"([a-z]+)([0-5])", tok)
    if not m:
        # Fallback: token without tone digit (e.g. 'zi', 'tou').
        # Still split initial/final using the same initial list; keep tone empty.
        m2 = re.fullmatch(r"([a-z]+)", tok)
        if not m2:
            return ["", tok]

        core = m2.group(1)

        # Special-case: 'ng' should be treated as a final with empty initial.
        if core == "ng":
            return ["", "ng"]

        init = ""
        for ini in _INITIALS:
            if core.startswith(ini):
                init = ini
                break
        final = core[len(init):]  # no tone
        return [init, final]

    core, tone = m.group(1), m.group(2)

    # Special-case: 'ng' should be treated as a final with empty initial (e.g. 'ng2' -> ['', 'ng2']).
    if core == "ng":
        return ["", f"ng{tone}"]

    init = ""
    for ini in _INITIALS:
        if core.startswith(ini):
            init = ini
            break

    final = core[len(init):] + tone
    return [init, final]


# -------------------- build output --------------------

def phones_accuracy(canon_phones, actual_phones):
    acc = []
    for c, a in zip_longest(canon_phones, actual_phones, fillvalue=None):
        if c is None or a is None:
            acc.append(0)
        elif c == "*" or a == "*":
            acc.append(0)
        else:
            acc.append(1 if c == a else 0)
    return acc


def parse_py_tokens(text: str):
    """Return list of canonical (init, final_with_tone) pairs from PY interval text."""
    text = _clean(text)
    if not text:
        return []
    toks = re.split(r"\s+", text)
    pairs = []
    for tok in toks:
        init_, fin_tone = split_canonical_py_token(tok)
        init2, fin_tone2 = normalize_pair(init_, fin_tone, do_zero_initial_map=False)
        pairs.append((init2, fin_tone2))
    return pairs


def parse_pybz_tokens(text: str):
    """Return list of raw PYBZ tokens from PYBZ interval text."""
    text = _clean(text)
    if not text:
        return []
    return re.split(r"\s+", text)


def _parse_part_cano_act(part: str):
    """
    Parse one PYBZ part that may be `cano` or `cano(act)`.
    Return (cano, act, has_paren) where act is inside text (may be '' or '*') when has_paren=True.
    """
    part = _clean(part)
    m = re.fullmatch(r"(?P<cano>[^()]*)\((?P<act>[^()]*)\)", part)
    if m:
        return _clean(m.group('cano')), _clean(m.group('act')), True
    return part, "", False


def build_canonical_and_actual_from_py_pybz(py_pair, pybz_tok: str):
    """
    Canonical output follows PY (but normalized).
    Actual output follows PYBZ parentheses rule (then normalized).
    Also count mismatch between normalized PY canonical vs normalized PYBZ canonical.
    """
    py_init, py_finaltone = py_pair
    py_init = _clean(py_init).lower()
    py_finaltone = _clean(py_finaltone).lower()

    # canonical still from PY, normalized, no zero-initial completion
    py_init, py_finaltone = normalize_pair(py_init, py_finaltone, do_zero_initial_map=False)
    py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)

    canon = [py_init, py_finaltone]

    tok = _clean(pybz_tok)
    if not tok or "~~" not in tok or "~" not in tok:
        return canon, ["", ""], ["", ""]

    left, tone_part = tok.split("~~", 1)
    init_part, final_part = left.split("~", 1)

    init_c, init_a, init_has = _parse_part_cano_act(init_part)
    fin_c, fin_a, fin_has = _parse_part_cano_act(final_part)
    tone_c, tone_a, tone_has = _parse_part_cano_act(tone_part)

    def _norm_piece_keep_empty(x: str) -> str:
        x = _clean(x)
        return x.lower() if x else ""

    # PYBZ canonical parts (raw, strip '-' markers, squeeze)
    pybz_init_raw = _squeeze_piece(_strip_dash_markers(_norm_piece_keep_empty(init_c if init_has else init_part)))
    pybz_fin_raw  = _squeeze_piece(_strip_dash_markers(_norm_piece_keep_empty(fin_c  if fin_has  else final_part)))
    pybz_tone_raw = _squeeze_piece(_strip_dash_markers(_clean(tone_c if tone_has else tone_part)))

    # PY canonical triple (normalized already)
    py_init_c = _clean(py_init).lower()
    py_fin_c, py_tone_c = _split_final_tone_keep_empty(py_finaltone)
    py_fin_c = _clean(py_fin_c).lower()
    py_tone_c = _strip_dash_markers(_clean(py_tone_c))

    # PYBZ canonical triple normalized WITHOUT zero-initial completion
    bz_ini2, bz_finaltone2 = normalize_pair(
        pybz_init_raw,
        f"{pybz_fin_raw}{pybz_tone_raw}",
        do_zero_initial_map=False,
    )
    bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
    bz_fin2 = _clean(bz_fin2).lower()
    bz_tone2 = _strip_dash_markers(_clean(bz_tone2))

    # Canonical phones (PYBZ) as a 2-piece list aligned with PY canonical
    canon_pybz = [bz_ini2, bz_finaltone2]


    # -------- conditional zero-initial completion ONLY for mismatch cases --------
    # If PY and PYBZ canonicals differ AND exactly one side has empty initial while the other has non-empty initial,
    # try to complete the empty-initial side using the user's i/v/u rules to make (ini, final-without-tone) match.
    if (py_init != bz_ini2) or (py_finaltone != bz_finaltone2):
        # Case A: PY has empty initial, PYBZ has non-empty initial
        if py_init == "" and bz_ini2 != "":
            new_ini, new_ft, changed = _maybe_complete_zero_initial_to_match(bz_ini2, bz_finaltone2, py_init, py_finaltone)
            if changed:
                py_init, py_finaltone = new_ini, new_ft
                py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
                canon = [py_init, py_finaltone]

        # Case B: PYBZ has empty initial, PY has non-empty initial
        elif bz_ini2 == "" and py_init != "":
            new_ini, new_ft, changed = _maybe_complete_zero_initial_to_match(py_init, py_finaltone, bz_ini2, bz_finaltone2)
            if changed:
                bz_ini2, bz_finaltone2 = new_ini, new_ft
                bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
                bz_fin2 = _clean(bz_fin2).lower()
                bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
                canon_pybz = [bz_ini2, bz_finaltone2]

    # -------- unify known canonical alternations when PY vs PYBZ mismatch --------
    # Only apply when canonicals mismatch
    def _split_ft(ft: str):
        return _split_final_tone_keep_empty(_clean(ft).lower())

    def _maybe_apply_unify(side: str, ini: str, ft: str, target_ini: str, target_fin: str):
        """Return (new_ini, new_ft, changed) for the given side; keep original tone."""
        fin, tone = _split_ft(ft)
        if _clean(ini).lower() == _clean(target_ini).lower() and _clean(fin).lower() == _clean(target_fin).lower():
            # already in target
            return ini, ft, False
        return ini, ft, False

    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    # 1) ~e~~ <-> ~er~~ (tone may differ)
    if py_init == "" and py_fin_now == "e" and bz_ini2 == "" and bz_fin_now == "er":
        py_finaltone = f"er{py_tone_now}" if py_tone_now else "er"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]
        py_fin_now, py_tone_now = _split_ft(py_finaltone)
    elif bz_ini2 == "" and bz_fin_now == "e" and py_init == "" and py_fin_now == "er":
        bz_finaltone2 = f"er{bz_tone_now}" if bz_tone_now else "er"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]
        bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    # 2) ~iu~~ <-> y~ou~~ (tone may differ)
    if py_init == "" and py_fin_now == "iu" and bz_ini2 == "y" and bz_fin_now == "ou":
        py_init = "y"
        py_finaltone = f"ou{py_tone_now}" if py_tone_now else "ou"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]
        py_fin_now, py_tone_now = _split_ft(py_finaltone)
    elif bz_ini2 == "" and bz_fin_now == "iu" and py_init == "y" and py_fin_now == "ou":
        bz_ini2 = "y"
        bz_finaltone2 = f"ou{bz_tone_now}" if bz_tone_now else "ou"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]
        bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)


    # 3) ~un~~ <-> w~en~~ (tone may differ)
    if py_init == "" and py_fin_now == "un" and bz_ini2 == "w" and bz_fin_now == "en":
        py_init = "w"
        py_finaltone = f"en{py_tone_now}" if py_tone_now else "en"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]
    elif bz_ini2 == "" and bz_fin_now == "un" and py_init == "w" and py_fin_now == "en":
        bz_ini2 = "w"
        bz_finaltone2 = f"en{bz_tone_now}" if bz_tone_now else "en"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]

    # 4) ~ei~~ <-> ~ui~~  => unify to w~ei~~ (tone may differ)
    # Only when both sides are zero-initial variants and the pair is exactly {ei, ui}.
    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    if py_init == "" and bz_ini2 == "" and {py_fin_now, bz_fin_now} == {"ei", "ui"}:
        # Convert PY side to w~ei~~, preserving PY tone
        py_init = "w"
        py_finaltone = f"ei{py_tone_now}" if py_tone_now else "ei"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]

        # Convert PYBZ side to w~ei~~, preserving PYBZ tone
        bz_ini2 = "w"
        bz_finaltone2 = f"ei{bz_tone_now}" if bz_tone_now else "ei"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]

    # 5) z~i~~ <-> z~1~~ => unify to z~i~~ (tone may differ)
    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    if py_init == "z" and bz_ini2 == "z" and {py_fin_now, bz_fin_now} == {"i", "1"}:
        # Convert PY side if it is '1'
        if py_fin_now == "1":
            py_finaltone = f"i{py_tone_now}" if py_tone_now else "i"
            py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
            canon = [py_init, py_finaltone]

        # Convert PYBZ side if it is '1'
        if bz_fin_now == "1":
            bz_finaltone2 = f"i{bz_tone_now}" if bz_tone_now else "i"
            bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
            bz_fin2 = _clean(bz_fin2).lower()
            bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
            canon_pybz = [bz_ini2, bz_finaltone2]

    # 6) ~uang~~ <-> ~ang~~ => unify to w~ang~~ (tone may differ)
    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    if py_init == "" and bz_ini2 == "" and {py_fin_now, bz_fin_now} == {"uang", "ang"}:
        # Convert PY side to w~ang~~, preserving PY tone
        py_init = "w"
        py_finaltone = f"ang{py_tone_now}" if py_tone_now else "ang"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]

        # Convert PYBZ side to w~ang~~, preserving PYBZ tone
        bz_ini2 = "w"
        bz_finaltone2 = f"ang{bz_tone_now}" if bz_tone_now else "ang"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]


    # 7) ~lv~~ <-> l~v~~ => unify to l~v~~ (tone may differ)
    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    # If PY is ~lv and PYBZ is l~v, convert PY
    if py_init == "" and py_fin_now == "lv" and bz_ini2 == "l" and bz_fin_now == "v":
        py_init = "l"
        py_finaltone = f"v{py_tone_now}" if py_tone_now else "v"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]

    # If PYBZ is ~lv and PY is l~v, convert PYBZ
    elif bz_ini2 == "" and bz_fin_now == "lv" and py_init == "l" and py_fin_now == "v":
        bz_ini2 = "l"
        bz_finaltone2 = f"v{bz_tone_now}" if bz_tone_now else "v"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]

    # 8) w~a~~5 <-> ~aw~~  => unify ~aw~~ to w~a~~5 (force tone 5)
    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    # If PY is w~a~~5 and PYBZ is ~aw~~, convert PYBZ
    if py_init == "w" and py_fin_now == "a" and py_tone_now == "5" and bz_ini2 == "" and bz_fin_now == "aw":
        bz_ini2 = "w"
        bz_finaltone2 = "a5"
        bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
        bz_fin2 = _clean(bz_fin2).lower()
        bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
        canon_pybz = [bz_ini2, bz_finaltone2]

    # If PYBZ is w~a~~5 and PY is ~aw~~, convert PY
    elif bz_ini2 == "w" and bz_fin_now == "a" and bz_tone_now == "5" and py_init == "" and py_fin_now == "aw":
        py_init = "w"
        py_finaltone = "a5"
        py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
        canon = [py_init, py_finaltone]

    # -------- tone completion: if (ini, fin) match but one side lacks tone, copy the other's tone --------
    py_fin_now, py_tone_now = _split_ft(py_finaltone)
    bz_fin_now, bz_tone_now = _split_ft(bz_finaltone2)

    if _clean(py_init).lower() == _clean(bz_ini2).lower() and _clean(py_fin_now).lower() == _clean(bz_fin_now).lower():
        # PY missing tone, PYBZ has tone
        if (py_tone_now == "") and (bz_tone_now != ""):
            py_finaltone = f"{py_fin_now}{bz_tone_now}"
            py_final, py_tone = _split_final_tone_keep_empty(py_finaltone)
            canon = [py_init, py_finaltone]

        # PYBZ missing tone, PY has tone
        elif (bz_tone_now == "") and (py_tone_now != ""):
            bz_finaltone2 = f"{bz_fin_now}{py_tone_now}"
            bz_fin2, bz_tone2 = _split_final_tone_keep_empty(bz_finaltone2)
            bz_fin2 = _clean(bz_fin2).lower()
            bz_tone2 = _strip_dash_markers(_clean(bz_tone2))
            canon_pybz = [bz_ini2, bz_finaltone2]

    # Recompute comparison triples after all canonical adjustments
    py_init_c = _clean(py_init).lower()
    py_fin_c, py_tone_c = _split_final_tone_keep_empty(_clean(py_finaltone).lower())
    py_fin_c = _clean(py_fin_c).lower()
    py_tone_c = _strip_dash_markers(_clean(py_tone_c))

    bz_ini2 = _clean(bz_ini2).lower()
    bz_fin2, bz_tone2 = _split_final_tone_keep_empty(_clean(bz_finaltone2).lower())
    bz_fin2 = _clean(bz_fin2).lower()
    bz_tone2 = _strip_dash_markers(_clean(bz_tone2))

    # keep canon_pybz consistent if bz_finaltone2 changed above
    canon_pybz = [bz_ini2, bz_finaltone2]

    if (py_init_c, py_fin_c, py_tone_c) != (bz_ini2, bz_fin2, bz_tone2):
        k = (py_init_c, py_fin_c, py_tone_c, bz_ini2, bz_fin2, bz_tone2)
        CANON_PY_VS_PYBZ_MISMATCH[k] += 1

        # record up to 3 example uttids
        try:
            ex = CANON_PY_VS_PYBZ_MISMATCH_EXAMPLES.get(k)
            if ex is None:
                CANON_PY_VS_PYBZ_MISMATCH_EXAMPLES[k] = [_CUR_UTTID]
            else:
                if _CUR_UTTID and (_CUR_UTTID not in ex) and (len(ex) < 3):
                    ex.append(_CUR_UTTID)
        except Exception:
            pass
        
    # -------- collect y/w-only mismatch examples (wan/yan-like) --------
    if (py_fin_c == bz_fin2) and (py_tone_c == bz_tone2):
        if (py_init_c in {"y", "w"}) and (bz_ini2 in {"y", "w"}) and (py_init_c != bz_ini2):
            YW_MISMATCH_SAMPLES.append({
                "uttid": _CUR_UTTID,
                "word_idx": _CUR_WORD_IDX,
                "word_text": _CUR_WORD_TEXT,
                "pybz_token_raw": _clean(pybz_tok),
                "py_triple": _fmt_cano(py_init_c, py_fin_c, py_tone_c),
                "pybz_triple": _fmt_cano(bz_ini2, bz_fin2, bz_tone2),
                "pybz_raw_parts": {
                    "init_raw": pybz_init_raw,
                    "final_raw": pybz_fin_raw,
                    "tone_raw": pybz_tone_raw,
                },
            })

    # actual pieces (strip '-' markers too)
    def norm_piece(x: str):
        x = _squeeze_piece(_strip_dash_markers(_clean(x)))
        return x.lower()

    # If no parentheses => correct => use PY canonical piece
    if init_has:
        act_init = "*" if init_a == "*" else norm_piece(init_a)
    else:
        act_init = py_init

    if fin_has:
        act_fin = "*" if fin_a == "*" else norm_piece(fin_a)
    else:
        act_fin = py_final

    if tone_has:
        act_tone = "*" if tone_a == "*" else _squeeze_piece(_strip_dash_markers(_clean(tone_a)))
    else:
        act_tone = py_tone

    if act_fin == "*" or act_tone == "*":
        act_finaltone = "*"
    else:
        act_finaltone = f"{act_fin}{act_tone}"

    actual = [act_init, act_finaltone]

    # normalize actual for output (normalize even if one side is '*'), no zero-initial completion
    a0, a1 = normalize_pair(actual[0], actual[1], do_zero_initial_map=False)
    actual = [a0, a1]

    # ensure exact match uses PY canonical string
    if actual[0] != "*" and actual[0] == canon[0]:
        actual[0] = canon[0]
    if actual[1] != "*" and actual[1] == canon[1]:
        actual[1] = canon[1]

    return canon, canon_pybz, actual


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

        global _CUR_UTTID, _CUR_WORD_IDX, _CUR_WORD_TEXT
        _CUR_UTTID = str(path.stem)
        _CUR_WORD_IDX = i
        _CUR_WORD_TEXT = word_text

        py_pairs = parse_py_tokens(py_text)
        pybz_toks = parse_pybz_tokens(pybz_text)

        canon_phones = []
        canon_phones_pybz = []
        actual_phones = []

        for py_pair, pybz_tok in zip_longest(py_pairs, pybz_toks, fillvalue=None):
            if py_pair is None:
                canon_pair = ["", ""]
                canon_pair_pybz = ["", ""]
                actual_pair = ["", ""]
            else:
                if pybz_tok is None:
                    canon_pair = [py_pair[0], py_pair[1]]
                    canon_pair_pybz = ["", ""]
                    actual_pair = ["", ""]
                else:
                    canon_pair, canon_pair_pybz, actual_pair = build_canonical_and_actual_from_py_pybz(py_pair, pybz_tok)

            canon_phones.extend(canon_pair)
            canon_phones_pybz.extend(canon_pair_pybz)
            actual_phones.extend(actual_pair)

        acc = phones_accuracy(canon_phones, actual_phones)

        words.append({
            "text": word_text,
            "phones": canon_phones,
            "phones_pybz": canon_phones_pybz,
            "phones_actual": actual_phones,
            "phones-accuracy": acc
        })

        if word_text:
            full_text_parts.append(word_text)

    return {"text": "".join(full_text_parts), "words": words}


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

    if CANON_PY_VS_PYBZ_MISMATCH:
        mismatch_map = []
        for (py_i, py_f, py_t, bz_i, bz_f, bz_t), cnt in CANON_PY_VS_PYBZ_MISMATCH.most_common():
            k = (py_i, py_f, py_t, bz_i, bz_f, bz_t)
            mismatch_map.append({
                "py_canonical": _fmt_cano(py_i, py_f, py_t),
                "pybz_canonical": _fmt_cano(bz_i, bz_f, bz_t),
                "count": cnt,
                "uttid_examples": CANON_PY_VS_PYBZ_MISMATCH_EXAMPLES.get(k, [])[:3]
            })

        stat_path = Path(out_json).with_suffix(".py_vs_pybz_canonical_mismatch.json")
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(mismatch_map, f, ensure_ascii=False, indent=2)

        print(f"Canonical mismatch map saved to: {stat_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data", help="root directory, e.g. data/")
    ap.add_argument("--out", type=str, default="parsed_textgrid.json", help="output json path")
    args = ap.parse_args()
    batch_parse(args.root, args.out)


if __name__ == "__main__":
    main()