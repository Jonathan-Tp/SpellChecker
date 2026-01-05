from __future__ import annotations

import json
import math
import re
import string
import platform

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Tuple, Callable

import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ----------------------------
# Constants
# ----------------------------
PUNCT_CHARS = set(string.punctuation)
SPECIAL_MASKED_KEY: Tuple[str, int] = ("__MASKED__", -1)
_ALPHA_RE = re.compile(r"^[A-Za-z]+$")

SPELLING_KEY = ("spelling", -1)
CONTEXT_KEY = ("context", -2)
SUGGESTED_KEY = ("suggested", -3)

# ----------------------------
# NLTK setup
# ----------------------------
_NLTK_READY = False

def ensure_nltk(force: bool = False) -> None:
    """Ensure required NLTK resources are available.

    Hosted environments (e.g., Streamlit Community Cloud) frequently do not ship
    with NLTK data. Newer NLTK versions may require `punkt_tab` in addition to
    `punkt`. Universal POS tags may require `universal_tagset`.

    We download into a writable directory and add it to `nltk.data.path`.
    If downloads fail (e.g., no internet), callers should be prepared to fall
    back to non-NLTK tokenization/tagging (see helpers below).
    """
    global _NLTK_READY
    if _NLTK_READY and not force:
        return

    # Choose a writable NLTK data directory
    try:
        base_dir = Path.home() / "nltk_data"
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = Path("/tmp/nltk_data")
        base_dir.mkdir(parents=True, exist_ok=True)

    if str(base_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(base_dir))

    # Try to fetch everything we might need in this project.
    packages = [
        "punkt",
        "punkt_tab",  # required by newer PunktTokenizer
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",  # newer tagger name (safe to attempt)
        "universal_tagset",  # universal tagset mapping
    ]

    for pkg in packages:
        try:
            nltk.download(pkg, download_dir=str(base_dir), quiet=True)
        except Exception:
            # No hard fail here; we have fallbacks for tokenization/tagging
            pass

    _NLTK_READY = True

# ----------------------------
# Tokenization
# ----------------------------
def tokenize_paragraph(paragraph: str) -> list[str]:
    """Tokenize a paragraph with <s> ... </s> boundaries.

    Uses NLTK when available; falls back to a small regex-based tokenizer if
    required NLTK resources are unavailable (common on hosted deployments).
    """
    ensure_nltk()

    def _fallback_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p for p in parts if p]

    def _fallback_words(text: str) -> list[str]:
        # words / contractions, numbers, or single non-space punctuation chars
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", text)

    inner: list[str] = []
    try:
        for sent in sent_tokenize(paragraph):
            inner.extend(word_tokenize(sent))
    except LookupError:
        for sent in _fallback_sentences(paragraph):
            inner.extend(_fallback_words(sent))

    return ["<s>"] + inner + ["</s>"]

# # ----------------------------
# # Bigram LM
# # ----------------------------
# class BigramLM:
#     def __init__(self, k: float = 0.1) -> None:
#         self.k = k
#         self.unigram: Counter[str] = Counter()
#         self.bigram: Counter[tuple[str, str]] = Counter()
#         self.V: set[str] = set()
#         self.V_size: int = 0

#     @classmethod
#     def from_json(
#         cls,
#         unigram_path: str | Path,
#         bigram_path: str | Path,
#         *,
#         k: float = 0.1,
#         bigram_delim: str = " ",
#         lowercase_vocab: bool = True,
#     ) -> "BigramLM":
#         lm = cls(k=k)

#         unigram_data = _read_json_dict(unigram_path)
#         bigram_data = _read_json_dict(bigram_path)

#         for w, c in unigram_data.items():
#             w2 = w.lower() if lowercase_vocab else w
#             lm.unigram[w2] += int(c)

#         lm.V = {w for w in lm.unigram.keys() if w != "<s>"}

#         for key, c in bigram_data.items():
#             prev, w = _parse_bigram_key(key, delim=bigram_delim)
#             if lowercase_vocab:
#                 prev, w = prev.lower(), w.lower()

#             lm.bigram[(prev, w)] += int(c)
#             if w != "<s>":
#                 lm.V.add(w)

#         lm.V.add("</s>")
#         lm.V.discard("<s>")
#         lm.V_size = len(lm.V)
#         return lm

#     def prob(self, prev: str, w: str) -> float:
#         num = self.bigram[(prev, w)] + self.k
#         den = self.unigram[prev] + self.k * self.V_size
#         return num / den if den > 0 else 1.0 / max(self.V_size, 1)

#     def logprob(self, prev: str, w: str) -> float:
#         return math.log(self.prob(prev, w))


# # ----------------------------
# # JSON / bigram helpers
# # ----------------------------
# def _read_json_dict(path: str | Path) -> dict:
#     p = Path(path)
#     with p.open("r", encoding="utf-8") as f:
#         data = json.load(f)
#     if not isinstance(data, dict):
#         raise ValueError(
#             f"{p} must contain a JSON object/dict, got {type(data).__name__}"
#         )
#     return data


# def _parse_bigram_key(key: str, delim: str = " ") -> tuple[str, str]:
#     if not isinstance(key, str):
#         raise ValueError(f"Bigram key must be a string. Got: {type(key).__name__}")

#     parts = key.rsplit(delim, 1) if delim == " " else key.split(delim, 1)
#     if len(parts) != 2 or not parts[0] or not parts[1]:
#         raise ValueError(
#             f"Bigram key {key!r} did not split into 2 parts with delim={delim!r}"
#         )
#     return parts[0], parts[1]
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


class BigramLM:
    def __init__(self, k: float = 0.1) -> None:
        self.k = k
        self.unigram: Counter[str] = Counter()
        self.bigram: Counter[tuple[str, str]] = Counter()
        self.V: set[str] = set()
        self.V_size: int = 0

    @classmethod
    def from_json(
        cls,
        unigram_path: str | Path,
        bigram_path: str | Path,
        *,
        k: float = 0.1,
        bigram_delim: str = " ",
        lowercase_vocab: bool = True,
        bigram_format: str = "auto",  # "auto" | "flat" | "packed"
    ) -> "BigramLM":
        lm = cls(k=k)

        unigram_data = _read_json_dict(unigram_path)
        bigram_data = _read_json_dict(bigram_path)

        # --- load unigrams ---
        for w, c in unigram_data.items():
            if not isinstance(w, str):
                continue
            w2 = w.lower() if lowercase_vocab else w
            lm.unigram[w2] += int(c)

        lm.V = {w for w in lm.unigram.keys() if w != "<s>"}

        # --- detect bigram format ---
        if bigram_format not in {"auto", "flat", "packed"}:
            raise ValueError("bigram_format must be one of: auto, flat, packed")

        is_packed = (
            bigram_format == "packed"
            or (
                bigram_format == "auto"
                and isinstance(bigram_data, dict)
                and "vocab" in bigram_data
                and "bigrams" in bigram_data
            )
        )

        # --- load bigrams ---
        if is_packed:
            vocab = bigram_data.get("vocab")
            triples = bigram_data.get("bigrams")

            if not isinstance(vocab, list) or not isinstance(triples, list):
                raise ValueError(
                    "Packed bigram JSON must be like: "
                    '{"vocab":[...], "bigrams":[[prev_id,next_id,count], ...]}'
                )

            # Normalize vocab (optional lowercasing)
            if lowercase_vocab:
                vocab_norm = []
                for tok in vocab:
                    if not isinstance(tok, str):
                        raise ValueError("Packed vocab must be a list[str]")
                    vocab_norm.append(tok.lower())
                vocab = vocab_norm
            else:
                for tok in vocab:
                    if not isinstance(tok, str):
                        raise ValueError("Packed vocab must be a list[str]")

            # triples: [[i,j,c], ...]
            for t in triples:
                if not (isinstance(t, list) and len(t) == 3):
                    raise ValueError(f"Bad bigram triple entry: {t!r} (expected [i,j,c])")
                i, j, c = int(t[0]), int(t[1]), int(t[2])

                if i < 0 or i >= len(vocab) or j < 0 or j >= len(vocab):
                    # skip or raise; raising is safer for debugging
                    raise ValueError(f"Bigram id out of range: {(i, j)} with vocab_size={len(vocab)}")

                prev = vocab[i]
                w = vocab[j]

                lm.bigram[(prev, w)] += c
                if w != "<s>":
                    lm.V.add(w)

        else:
            # flat dict: {"prev w": count, ...}
            if not isinstance(bigram_data, dict):
                raise ValueError(f"{Path(bigram_path)} must contain a JSON object/dict.")

            for key, c in bigram_data.items():
                if not isinstance(key, str):
                    continue
                prev, w = _parse_bigram_key(key, delim=bigram_delim)
                if lowercase_vocab:
                    prev, w = prev.lower(), w.lower()

                lm.bigram[(prev, w)] += int(c)
                if w != "<s>":
                    lm.V.add(w)

        lm.V.add("</s>")
        lm.V.discard("<s>")
        lm.V_size = len(lm.V)
        return lm

    def prob(self, prev: str, w: str) -> float:
        num = self.bigram[(prev, w)] + self.k
        den = self.unigram[prev] + self.k * self.V_size
        return num / den if den > 0 else 1.0 / max(self.V_size, 1)

    def logprob(self, prev: str, w: str) -> float:
        return math.log(self.prob(prev, w))


# ----------------------------
# JSON / bigram helpers
# ----------------------------
def _read_json_dict(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object/dict, got {type(data).__name__}")
    return data


def _parse_bigram_key(key: str, delim: str = " ") -> tuple[str, str]:
    if not isinstance(key, str):
        raise ValueError(f"Bigram key must be a string. Got: {type(key).__name__}")

    parts = key.rsplit(delim, 1) if delim == " " else key.split(delim, 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Bigram key {key!r} did not split into 2 parts with delim={delim!r}")
    return parts[0], parts[1]


# ----------------------------
# Edit distance + vocab helpers
# ----------------------------
def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def words_within_edit_distance(token: str, vocab: set[str], x: int) -> list[str]:
    if x < 0:
        return []
    t = token.lower()
    out: list[str] = []
    for w in vocab:
        if edit_distance(t, w.lower()) <= x:
            out.append(w)
    return out


# ----------------------------
# POS + context helpers
# ----------------------------

_PTB_TO_UNIVERSAL_EXACT: dict[str, str] = {
    # Conjunctions / determiners / adpositions / particles
    "CC": "CONJ",
    "DT": "DET",
    "PDT": "DET",
    "WDT": "DET",
    "IN": "ADP",
    "TO": "PRT",
    "RP": "PRT",
    # Pronouns
    "PRP": "PRON",
    "PRP$": "PRON",
    "WP": "PRON",
    "WP$": "PRON",
    # Numbers
    "CD": "NUM",
}

def _ptb_to_universal(tag: str) -> str:
    """Best-effort PennTreebank -> Universal tag mapping without NLTK resources."""
    if not tag:
        return "X"

    if tag in _PTB_TO_UNIVERSAL_EXACT:
        return _PTB_TO_UNIVERSAL_EXACT[tag]

    # Prefix-based mapping
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"

    # Punctuation in PTB is often ., , , :, `` etc.
    if tag in {".", ",", ":", "``", "''", "-LRB-", "-RRB-"}:
        return "."

    return "X"


def _simple_universal(tok: str) -> str:
    """Tiny rule-based universal tagger fallback (when NLTK isn't usable)."""
    t = tok.strip()
    if not t:
        return "X"
    if is_punct_token(t):
        return "."
    low = t.lower()
    if low.isdigit():
        return "NUM"
    if low in {"a", "an", "the", "this", "that", "these", "those"}:
        return "DET"
    if low in {"and", "or", "but", "nor", "yet", "so"}:
        return "CONJ"
    if low in {"to"}:
        return "PRT"
    if low in {"in", "on", "at", "by", "for", "with", "from", "of", "as", "into", "over", "under", "between", "after", "before", "about"}:
        return "ADP"
    if low in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"}:
        return "PRON"
    if low.endswith("ly"):
        return "ADV"
    # Default to NOUN (works better than X for many downstream heuristics)
    return "NOUN"


def safe_pos_tag_universal(tokens: list[str]) -> list[tuple[str, str]]:
    """Return (token, universal_tag) for tokens, with robust fallbacks."""
    ensure_nltk()
    # First try the normal universal tagset path.
    try:
        return pos_tag(tokens, tagset="universal")
    except LookupError:
        pass

    # If universal mapping is missing, tag in PTB and map ourselves.
    try:
        ptb = pos_tag(tokens)  # defaults to Penn tags
        return [(tok, _ptb_to_universal(tag)) for tok, tag in ptb]
    except LookupError:
        # If even the tagger is missing, fall back to rules.
        return [(tok, _simple_universal(tok)) for tok in tokens]


def candidate_pos_tag(prev: str, cand: str, nxt: str) -> str:
    window: list[str] = []
    if prev not in {"<s>", "</s>"} and not is_punct_token(prev):
        window.append(prev)
    window.append(cand)
    if nxt not in {"<s>", "</s>"} and not is_punct_token(nxt):
        window.append(nxt)

    tagged = safe_pos_tag_universal(window)
    for tok, tag in tagged:
        if tok == cand:
            return tag
    return tagged[0][1] if tagged else "X"


def is_punct_token(tok: str) -> bool:
    return tok != "" and all(ch in PUNCT_CHARS for ch in tok)


def context_score(lm: BigramLM, prev: str, w: str, nxt: str) -> float:
    return lm.logprob(prev, w) + lm.logprob(w, nxt)


def is_oov_typo(word: str, vocab: set[str], threshold: int = 2) -> bool:
    if not word or not word.isalpha():
        return False

    w = word.lower()
    if w in vocab:
        return False

    for v in vocab:
        if edit_distance(w, v.lower()) <= threshold:
            return True

    return False


def out_of_context_by_bigram_pos(
    lm: BigramLM,
    prev: str,
    word: str,
    nxt: str,
    candidates: set[str] | list[str],
    tau_default: float = 2.0,
    tau_by_pos: dict[str, float] | None = None,
) -> bool:
    if tau_by_pos is None:
        tau_by_pos = {}

    if not word or is_punct_token(word):
        return False

    prev = (prev or "<s>").lower()
    word_l = word.lower()
    nxt = (nxt or "</s>").lower()

    cur_pos = candidate_pos_tag(prev, word_l, nxt)
    tau = tau_by_pos.get(cur_pos, tau_default)

    orig_score = context_score(lm, prev, word_l, nxt)
    best_score = orig_score
    best_word = word_l

    for c in candidates:
        if not c:
            continue
        c_l = c.lower()
        if is_punct_token(c_l):
            continue
        if candidate_pos_tag(prev, c_l, nxt) != cur_pos:
            continue

        sc = context_score(lm, prev, c_l, nxt)
        if sc > best_score:
            best_score = sc
            best_word = c_l
    
    print(word_l, orig_score, best_score)
    print(candidates)

    return (best_word != word_l) and ((best_score - orig_score) > tau)

# ----------------------------
# Non-word labeling + masking
# ----------------------------
def label_paragraph_and_mask_for_bert_nonword(
    paragraph: str,
    vocab: set[str],
    ed_thresh: int = 2,
    max_suggestions: int = 20,
    mask_if_wrong: bool = True,
    mask_token: str = "[MASK]",
) -> dict:
    """
    Detect NON-WORD (typo) errors only, attach suggestions, and also return a masked paragraph
    under SPECIAL_MASKED_KEY.

    Output dict:
      (token, idx) -> None
      (token, idx) -> {'e': 'n', 'suggestions': [..]}
      SPECIAL_MASKED_KEY -> <detokenized masked paragraph>

    idx is the index in the inner token stream (excluding <s> and </s>):
      i in [1 .. len(toks)-2], idx = i-1
    """
    toks = tokenize_paragraph(paragraph)
    detok = TreebankWordDetokenizer()

    vocab_lc = {v.lower() for v in vocab}

    out: dict = {}
    masked_tokens: list[str] = []

    for i in range(1, len(toks) - 1):
        word = toks[i]
        idx = i - 1  # index in inner tokens

        info = None
        is_alpha = _ALPHA_RE.match(word) is not None
        wrong = bool(is_alpha and (word.lower() not in vocab_lc))

        if wrong:
            sugg = words_within_edit_distance(word, vocab, ed_thresh)
            sugg = sorted(
                sugg,
                key=lambda w: (edit_distance(word.lower(), w.lower()), w.lower()),
            )
            sugg = sugg[:max_suggestions]
            info = {"e": "n", "suggestions": sugg}

        out[(word, idx)] = info

        do_mask = (wrong and mask_if_wrong) or (
            (not wrong) and (not mask_if_wrong)
        )
        if do_mask and is_alpha and not is_punct_token(word):
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append(word)

    out[SPECIAL_MASKED_KEY] = detok.detokenize(masked_tokens)
    return out


# ----------------------------
# Context labeling + masking
# ----------------------------
def label_paragraph_and_mask_for_bert_context(
    label_output: dict,
    lm: BigramLM,
    vocab: set[str] | None = None,
    cand_x: int = 3,
    tau_default: float = 2.0,
    tau_by_pos: dict[str, float] | None = None,
    mask_if_wrong: bool = True,
    mask_token: str = "[MASK]",
) -> dict:
    """
    Take the output of `fill_all(label_output)` and re-label tokens for
    CONTEXT errors using out_of_context_by_bigram_pos.

    Context for prev/word/next is built using:
      - the *first suggestion* in label_output[(token, idx)]["suggestions"]
        if it exists (BERT's best guess),
      - otherwise the original token itself.

    If a token is deemed a context error, its entry becomes:
        {'e': 'c', 'suggestions': []}

    Then we rebuild a masked paragraph under SPECIAL_MASKED_KEY where only
    context errors are masked (when mask_if_wrong=True).
    """
    if vocab is None:
        vocab = lm.V

    # Build ordered token list with "representative" tokens
    token_items: list[dict] = []
    for k, info in label_output.items():
        if k == SPECIAL_MASKED_KEY:
            continue
        orig_tok, idx = k
        rep_tok = orig_tok

        if isinstance(info, dict):
            suggs = info.get("suggestions")
            if suggs:
                # Use first suggestion as the token in context
                rep_tok = suggs[0]

        token_items.append(
            {
                "key": k,
                "idx": idx,
                "orig_tok": orig_tok,
                "rep_tok": rep_tok,
            }
        )

    # Sort by inner index
    token_items.sort(key=lambda x: x["idx"])

    # This is the stream the LM sees for context (all using first suggestions if present)
    base_tokens_lc = [item["rep_tok"].lower() for item in token_items]
    toks = ["<s>"] + base_tokens_lc + ["</s>"]

    # Pass 1: decide which tokens are context errors
    for pos, item in enumerate(token_items):
        key = item["key"]
        rep_tok = item["rep_tok"]
        info = label_output.get(key)

        # Skip non-alpha / punctuation / boundaries (based on representative token)
        if (
            not rep_tok
            or rep_tok in {"<s>", "</s>"}
            or is_punct_token(rep_tok)
            or not rep_tok.isalpha()
        ):
            continue

        prev = toks[pos]        # context before
        word = toks[pos + 1]    # current token (already lowercased)
        nxt = toks[pos + 2]     # context after

        # somewhere global
        FIXED_CANDIDATES_BY_POS: dict[str, list[str]] = {
            # Articles & basic determiners
            "DET": [
                "a", "an", "the",
                "this", "that", "these", "those",
                "some", "any", "no",
                "each", "every",
                "another", "other",
                "all", "both"
            ],

            # Prepositions
            "ADP": [
                "in", "on", "at", "to", "for", "from", "of", "with", "by", "about",
                "over", "under", "into", "onto",
                "within", "without",
                "between", "among",
                "before", "after",
                "around", "through",
                "against", "during"
            ],

            # Coordinating / subordinating conjunctions (universal tagset uses CONJ)
            "CONJ": [
                "and", "or", "but", "so", "yet", "nor",
                "although", "though", "because",
                "if", "while", "whereas", "whether"
            ],

            # Particles (often small words paired with verbs)
            "PRT": [
                "to", "not",
                "up", "down",
                "off", "on",
                "out", "in",
                "over", "back",
                "away", "around", "through"
            ],

            # Personal + reflexive pronouns (lowercase, since you lowercase in context)
            "PRON": [
                "i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "us", "them",
                "myself", "yourself", "himself", "herself",
                "itself", "ourselves", "themselves"
            ],
        }



        cand_list: list[str] = []
        fixed = FIXED_CANDIDATES_BY_POS.get(pos)
        pos = candidate_pos_tag(prev, word, nxt)
        if fixed is not None:
            # use fixed candidate set for closed-class tags
            cand_list = fixed
        elif cand_x >= 0:
            # open-class: fall back to edit-distance search
            cand_list = words_within_edit_distance(rep_tok, vocab, cand_x)
            filtered: list[str] = []
            for cand in cand_list:
                cand_pos = candidate_pos_tag(prev, cand, nxt)
                if cand_pos == pos:
                    filtered.append(cand)
            cand_list = filtered
        # else: cand_list stays []


        # Remove duplicates and the same word as baseline
        cand_set = {
            c for c in cand_list
            if c and c.lower() != word.lower()
        }
        if not cand_set:
            # No alternative to compare against, can't flag context error
            continue

        is_ctx_error = out_of_context_by_bigram_pos(
            lm=lm,
            prev=prev,
            word=word,   # representative word (first suggestion, lowercased)
            nxt=nxt,
            candidates=cand_set,
            tau_default=tau_default,
            tau_by_pos=tau_by_pos,
        )

        print(word, is_ctx_error)

        if is_ctx_error:
            # Mark as context error, and clear suggestions as requested
            label_output[key] = {"e": "c", "suggestions": []}

    # Pass 2: rebuild masked paragraph (using representative tokens)
    detok = TreebankWordDetokenizer()
    masked_tokens: list[str] = []

    for item in token_items:
        key = item["key"]
        rep_tok = item["rep_tok"]
        info = label_output.get(key)

        is_ctx_error = isinstance(info, dict) and info.get("e") == "c"

        do_mask = (is_ctx_error and mask_if_wrong) or (
            (not is_ctx_error) and (not mask_if_wrong)
        )

        if (
            do_mask
            and rep_tok
            and rep_tok.isalpha()
            and not is_punct_token(rep_tok)
        ):
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append(rep_tok)

    label_output[SPECIAL_MASKED_KEY] = detok.detokenize(masked_tokens)
    return label_output

from typing import Iterable, Optional, Dict, Any, List, Tuple, Callable

import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

# SPECIAL_MASKED_KEY is already defined earlier in your file; if not, uncomment:
# SPECIAL_MASKED_KEY = ("__MASKED__", -1)


# def make_unified_bert_suggester(
#     model_ref: str,
#     *,
#     candidate_vocab: Optional[Iterable[str]] = None,
#     tau: float = 0.0,
#     fallback_top_k: int = 30,
# ) -> Callable[[Dict[Any, Any]], Dict[Any, Any]]:
#     """
#     Unified BERT suggester.

#     label_output format:
#       - SPECIAL_MASKED_KEY -> masked sentence string with [MASK] tokens
#       - (token, idx) -> info dict, where:
#             info["e"] == "c"   => context error
#             info["e"] != "c"   => typo / nonword / other
#             info["suggestions"] (for typos) is an optional list of candidates.

#     Behaviour per [MASK] position:

#       1. Use BERT to score candidates.
#       2. Filter to top-k highest-probability candidates that are in candidate_vocab
#          (if provided; otherwise fall back to raw BERT tokens).
#       3. Sort those top-k by edit distance to the original word, ascending
#          (ties broken by probability descending) and store that order in
#          info["suggestions"].
#       4. Use tau margin (if > 0) to decide whether to actually replace the
#          original word in the context. The committed token is used to update
#          SPECIAL_MASKED_KEY and later masks.
#     """

#     device = torch.device(
#         "mps" if torch.backends.mps.is_available()
#         else "cuda" if torch.cuda.is_available()
#         else "cpu"
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_ref)
#     model = AutoModelForMaskedLM.from_pretrained(model_ref).to(device)
#     model.eval()

#     MASK = tokenizer.mask_token
#     MASK_ID = tokenizer.mask_token_id
#     detok = TreebankWordDetokenizer()

#     # ----------------- helpers -----------------
#     def _encode_one_piece(s: str) -> Optional[int]:
#         """Return token id if s is a single-piece token; otherwise None."""
#         ids = tokenizer.encode(s, add_special_tokens=False)
#         return ids[0] if len(ids) == 1 else None

#     def _mask_positions(input_ids_1d: torch.Tensor) -> List[int]:
#         ids = input_ids_1d.tolist()
#         return [i for i, x in enumerate(ids) if x == MASK_ID]

#     # Pre-encode candidate vocab once (for context errors / fallback)
#     vocab_token_ids: Dict[str, int] = {}
#     if candidate_vocab is not None:
#         for w in candidate_vocab:
#             tid = _encode_one_piece(w)
#             if tid is not None:
#                 vocab_token_ids[w] = tid

#     @torch.no_grad()
#     def update_with_bert(label_output: Dict[Any, Any]) -> Dict[Any, Any]:
#         if SPECIAL_MASKED_KEY not in label_output:
#             raise ValueError(
#                 f"label_output missing SPECIAL_MASKED_KEY={SPECIAL_MASKED_KEY}"
#             )

#         masked_text = label_output[SPECIAL_MASKED_KEY]

#         # No [MASK], nothing to do
#         if MASK not in masked_text:
#             return label_output

#         # Encode masked text
#         enc = tokenizer(masked_text, return_tensors="pt")
#         enc = {k: v.to(device) for k, v in enc.items()}
#         input_ids = enc["input_ids"][0]
#         attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

#         # Positions of [MASK] tokens in the HF tokenization
#         mask_pos_list = _mask_positions(input_ids)

#         # ---- Group error entries by error type ("n", "c", etc.) ----
#         error_groups: Dict[str, List[Tuple[int, Tuple[str, int]]]] = {}
#         for k, info in label_output.items():
#             if k == SPECIAL_MASKED_KEY:
#                 continue
#             if not isinstance(k, tuple) or len(k) != 2:
#                 continue
#             if not isinstance(info, dict):
#                 continue
#             if "e" not in info:
#                 continue

#             tok, idx = k
#             err_type = info.get("e")
#             if err_type is None:
#                 continue
#             error_groups.setdefault(err_type, []).append((idx, k))

#         # Sort each group by idx
#         for group in error_groups.values():
#             group.sort(key=lambda x: x[0])

#         total_errors = sum(len(g) for g in error_groups.values())
#         num_masks = len(mask_pos_list)

#         # Decide which error keys align with MASK positions
#         if num_masks == total_errors:
#             # All errors are masked
#             merged: List[Tuple[int, Tuple[str, int]]] = []
#             for g in error_groups.values():
#                 merged.extend(g)
#             merged.sort(key=lambda x: x[0])
#             label_keys: List[Tuple[str, int]] = [k for _, k in merged]
#         else:
#             # Mixed error types; choose the type whose count matches num_masks
#             chosen_type: Optional[str] = None
#             for t, g in error_groups.items():
#                 if len(g) == num_masks:
#                     chosen_type = t
#                     break

#             if chosen_type is None:
#                 raise ValueError(
#                     "Mismatch between [MASK] tokens and error entries.\n"
#                     f"found {num_masks} [MASK] in masked text but "
#                     f"error counts by type are: "
#                     f"{ {t: len(g) for t, g in error_groups.items()} }\n"
#                     f"masked_text: {masked_text}"
#                 )

#             label_keys = [k for _, k in error_groups[chosen_type]]

#         # -----------------------------------------------------------------
#         # Fill left-to-right, updating context as we go
#         # -----------------------------------------------------------------
#         for pos, key in zip(mask_pos_list, label_keys):
#             info = label_output.get(key)
#             if info is None or not isinstance(info, dict):
#                 continue

#             err_type = info.get("e")
#             orig_tok = key[0]

#             # Run model for current context
#             out = model(
#                 input_ids=input_ids.unsqueeze(0),
#                 attention_mask=attention_mask.unsqueeze(0),
#             )
#             logprobs = torch.log_softmax(out.logits[0, pos], dim=-1)

#             scored: List[Tuple[str, float, int]] = []  # (cand, logprob, token_id)

#             # ---- Build candidate list ----
#             if err_type == "c":
#                 # CONTEXT ERROR:
#                 #   use candidate_vocab ONLY (if provided),
#                 #   otherwise use BERT top-k (unconstrained).
#                 if vocab_token_ids:
#                     for cand, cand_id in vocab_token_ids.items():
#                         lp = float(logprobs[int(cand_id)].item())
#                         scored.append((cand, lp, int(cand_id)))
#                 else:
#                     top = torch.topk(logprobs, k=fallback_top_k)
#                     for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
#                         tok = tokenizer.decode([tid]).strip()
#                         scored.append((tok, float(lp), int(tid)))
#             else:
#                 # TYPO / NONWORD:
#                 #   start from existing suggestions (likely already vocab-based),
#                 #   but still enforce candidate_vocab if provided.
#                 raw_suggestions = list(info.get("suggestions", []))
#                 if vocab_token_ids:
#                     for cand in raw_suggestions:
#                         cand_id = vocab_token_ids.get(cand)
#                         if cand_id is None:
#                             continue
#                         lp = float(logprobs[int(cand_id)].item())
#                         scored.append((cand, lp, int(cand_id)))
#                 else:
#                     for cand in raw_suggestions:
#                         cand_id = _encode_one_piece(cand)
#                         if cand_id is None:
#                             continue
#                         lp = float(logprobs[int(cand_id)].item())
#                         scored.append((cand, lp, int(cand_id)))

#                 # If no usable suggestion, fallback to vocab or BERT top-k
#                 if not scored:
#                     if vocab_token_ids:
#                         for cand, cand_id in vocab_token_ids.items():
#                             lp = float(logprobs[int(cand_id)].item())
#                             scored.append((cand, lp, int(cand_id)))
#                     else:
#                         top = torch.topk(logprobs, k=fallback_top_k)
#                         for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
#                             tok = tokenizer.decode([tid]).strip()
#                             scored.append((tok, float(lp), int(tid)))

#             if not scored:
#                 # nothing usable; skip this position
#                 continue

#             # ---------------------------------------------------------
#             # 1) Take top-k by probability (still within vocab if given)
#             # ---------------------------------------------------------
#             scored.sort(key=lambda x: x[1], reverse=True)
#             scored = scored[:fallback_top_k]

#             # ---------------------------------------------------------
#             # 2) Sort these top-k by edit distance, then prob
#             # ---------------------------------------------------------
#             scored_with_dist: List[Tuple[str, int, float, int]] = []
#             for cand, lp, tid in scored:
#                 dist = edit_distance(orig_tok, cand)
#                 scored_with_dist.append((cand, dist, lp, tid))

#             # sort: smaller edit distance first; tie-breaker: higher log-prob
#             scored_with_dist.sort(key=lambda x: (x[1], -x[2]))

#             # suggestions list is now ordered by (edit distance, -prob)
#             suggestions = [cand for (cand, dist, lp, tid) in scored_with_dist]

#             # ---------------------------------------------------------
#             # 3) Decide which token to COMMIT into the context
#             #    (by default: closest edit-distance candidate)
#             # ---------------------------------------------------------
#             first_cand, first_dist, first_lp, first_tid = scored_with_dist[0]
#             commit_token = first_cand
#             commit_tid: Optional[int] = first_tid
#             commit_lp = first_lp

#             if tau > 0.0:
#                 orig_id = _encode_one_piece(orig_tok)
#                 if orig_id is not None:
#                     orig_lp = float(logprobs[int(orig_id)].item())
#                     # If the chosen candidate doesn't beat original by > tau, keep original
#                     if (commit_lp - orig_lp) <= tau:
#                         commit_token = orig_tok
#                         commit_tid = orig_id
#                         # ensure original is first in suggestions
#                         if orig_tok in suggestions:
#                             suggestions = [orig_tok] + [s for s in suggestions if s != orig_tok]
#                         else:
#                             suggestions = [orig_tok] + suggestions

#             # Save suggestions sorted by edit distance (with possible original at front)
#             info["suggestions"] = suggestions

#             # Update input_ids with committed token for later masks
#             if commit_tid is None:
#                 best_id = _encode_one_piece(commit_token)
#             else:
#                 best_id = commit_tid
#             if best_id is not None:
#                 input_ids[pos] = best_id

#         # -------------------------------------------------------------
#         # Rebuild SPECIAL_MASKED_KEY from best suggestions
#         # -------------------------------------------------------------
#         word_entries: List[Tuple[int, str, Any]] = []
#         for k, info in label_output.items():
#             if k == SPECIAL_MASKED_KEY:
#                 continue
#             if not isinstance(k, tuple) or len(k) != 2:
#                 continue
#             tok, idx = k
#             word_entries.append((idx, tok, info))

#         word_entries.sort(key=lambda t: t[0])

#         final_tokens: List[str] = []
#         for _, tok, info in word_entries:
#             if isinstance(info, dict):
#                 suggs = info.get("suggestions")
#                 if suggs:
#                     final_tokens.append(suggs[0])
#                     continue
#             final_tokens.append(tok)

#         label_output[SPECIAL_MASKED_KEY] = detok.detokenize(final_tokens)
#         return label_output

#     return update_with_bert
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

# assumes you already have:
# - SPECIAL_MASKED_KEY
# - edit_distance(orig, cand)

def make_unified_bert_suggester(
    model_ref: str,
    *,
    candidate_vocab: Optional[Iterable[str]] = None,
    tau: float = 0.0,
    fallback_top_k: int = 30,
) -> Callable[[Dict[Any, Any]], Dict[Any, Any]]:
    """
    Unified BERT/DistilBERT MLM suggester.

    Works with:
      - BERT (has token_type_ids)
      - DistilBERT (no token_type_ids)
      - other MLMs where AutoModelForMaskedLM applies

    See your original docstring for behavior details.
    """

    # Streamlit Cloud (Linux/CPU) often can't use MPS; guard it by platform.
    use_mps = (platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    use_cuda = torch.cuda.is_available()

    device = torch.device("mps" if use_mps else "cuda" if use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    # Be defensive: some HF loading paths can leave meta tensors if a load partially fails.
    # Always load weights onto CPU first, then move, and fall back to CPU if device move isn't supported.
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_ref,
            low_cpu_mem_usage=False,   # avoids meta-tensor init on some setups
            device_map=None,
        )
    except TypeError:
        # older transformers may not accept these kwargs
        model = AutoModelForMaskedLM.from_pretrained(model_ref)

    model.eval()
    try:
        model = model.to(device)
    except NotImplementedError:
        # If a backend isn't supported in this environment, just run on CPU.
        device = torch.device("cpu")
        model = model.to(device)
    MASK = tokenizer.mask_token
    MASK_ID = tokenizer.mask_token_id
    if MASK is None or MASK_ID is None:
        raise ValueError(
            f"Tokenizer for {model_ref!r} has no mask token; this model can't be used for MLM suggestions."
        )

    detok = TreebankWordDetokenizer()

    # ----------------- helpers -----------------
    def _encode_one_piece(s: str) -> Optional[int]:
        """Return token id if s is a single-piece token; otherwise None."""
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _mask_positions(input_ids_1d: torch.Tensor) -> List[int]:
        ids = input_ids_1d.tolist()
        return [i for i, x in enumerate(ids) if x == MASK_ID]

    # Pre-encode candidate vocab once
    vocab_token_ids: Dict[str, int] = {}
    if candidate_vocab is not None:
        for w in candidate_vocab:
            tid = _encode_one_piece(w)
            if tid is not None:
                vocab_token_ids[w] = tid

    @torch.inference_mode()
    def update_with_bert(label_output: Dict[Any, Any]) -> Dict[Any, Any]:
        if SPECIAL_MASKED_KEY not in label_output:
            raise ValueError(f"label_output missing SPECIAL_MASKED_KEY={SPECIAL_MASKED_KEY}")

        masked_text = label_output[SPECIAL_MASKED_KEY]

        # No [MASK], nothing to do
        if MASK not in masked_text:
            return label_output

        # Encode masked text (keep tensors 2D: [1, seq])
        enc = tokenizer(masked_text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        input_ids = enc["input_ids"]  # shape: (1, seq)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))  # (1, seq)
        token_type_ids = enc.get("token_type_ids", None)  # DistilBERT: usually None

        # Positions of [MASK] tokens in the HF tokenization
        mask_pos_list = _mask_positions(input_ids[0])  # pass 1D row

        # ---- Group error entries by error type ("n", "c", etc.) ----
        error_groups: Dict[str, List[Tuple[int, Tuple[str, int]]]] = {}
        for k, info in label_output.items():
            if k == SPECIAL_MASKED_KEY:
                continue
            if not isinstance(k, tuple) or len(k) != 2:
                continue
            if not isinstance(info, dict):
                continue
            err_type = info.get("e")
            if err_type is None:
                continue
            tok, idx = k
            error_groups.setdefault(err_type, []).append((idx, k))

        for group in error_groups.values():
            group.sort(key=lambda x: x[0])

        total_errors = sum(len(g) for g in error_groups.values())
        num_masks = len(mask_pos_list)

        if num_masks == total_errors:
            merged: List[Tuple[int, Tuple[str, int]]] = []
            for g in error_groups.values():
                merged.extend(g)
            merged.sort(key=lambda x: x[0])
            label_keys: List[Tuple[str, int]] = [k for _, k in merged]
        else:
            chosen_type: Optional[str] = None
            for t, g in error_groups.items():
                if len(g) == num_masks:
                    chosen_type = t
                    break
            if chosen_type is None:
                raise ValueError(
                    "Mismatch between [MASK] tokens and error entries.\n"
                    f"found {num_masks} [MASK] in masked text but error counts by type are: "
                    f"{ {t: len(g) for t, g in error_groups.items()} }\n"
                    f"masked_text: {masked_text}"
                )
            label_keys = [k for _, k in error_groups[chosen_type]]

        # -----------------------------------------------------------------
        # Fill left-to-right, updating context as we go
        # -----------------------------------------------------------------
        for pos, key in zip(mask_pos_list, label_keys):
            info = label_output.get(key)
            if info is None or not isinstance(info, dict):
                continue

            err_type = info.get("e")
            orig_tok = key[0]

            # Build model inputs (KEEP 2D; no unsqueeze)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids

            out = model(**model_inputs)
            logprobs = torch.log_softmax(out.logits[0, pos], dim=-1)

            scored: List[Tuple[str, float, int]] = []

            # ---- Build candidate list ----
            if err_type == "c":
                if vocab_token_ids:
                    for cand, cand_id in vocab_token_ids.items():
                        lp = float(logprobs[int(cand_id)].item())
                        scored.append((cand, lp, int(cand_id)))
                else:
                    top = torch.topk(logprobs, k=fallback_top_k)
                    for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
                        tok = tokenizer.decode([tid]).strip()
                        if tok:
                            scored.append((tok, float(lp), int(tid)))
            else:
                raw_suggestions = list(info.get("suggestions", []))

                if vocab_token_ids:
                    for cand in raw_suggestions:
                        cand_id = vocab_token_ids.get(cand)
                        if cand_id is None:
                            continue
                        lp = float(logprobs[int(cand_id)].item())
                        scored.append((cand, lp, int(cand_id)))
                else:
                    for cand in raw_suggestions:
                        cand_id = _encode_one_piece(cand)
                        if cand_id is None:
                            continue
                        lp = float(logprobs[int(cand_id)].item())
                        scored.append((cand, lp, int(cand_id)))

                if not scored:
                    if vocab_token_ids:
                        for cand, cand_id in vocab_token_ids.items():
                            lp = float(logprobs[int(cand_id)].item())
                            scored.append((cand, lp, int(cand_id)))
                    else:
                        top = torch.topk(logprobs, k=fallback_top_k)
                        for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
                            tok = tokenizer.decode([tid]).strip()
                            if tok:
                                scored.append((tok, float(lp), int(tid)))

            if not scored:
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:fallback_top_k]

            scored_with_dist: List[Tuple[str, int, float, int]] = []
            for cand, lp, tid in scored:
                dist = edit_distance(orig_tok, cand)
                scored_with_dist.append((cand, dist, lp, tid))
            scored_with_dist.sort(key=lambda x: (x[1], -x[2]))

            suggestions = [cand for (cand, _, _, _) in scored_with_dist]

            first_cand, _, first_lp, first_tid = scored_with_dist[0]
            commit_token = first_cand
            commit_tid: Optional[int] = first_tid
            commit_lp = first_lp

            if tau > 0.0:
                orig_id = _encode_one_piece(orig_tok)
                if orig_id is not None:
                    orig_lp = float(logprobs[int(orig_id)].item())
                    if (commit_lp - orig_lp) <= tau:
                        commit_token = orig_tok
                        commit_tid = orig_id
                        if orig_tok in suggestions:
                            suggestions = [orig_tok] + [s for s in suggestions if s != orig_tok]
                        else:
                            suggestions = [orig_tok] + suggestions

            info["suggestions"] = suggestions

            # Update context for later masks (mutate 2D tensor row)
            best_id = commit_tid if commit_tid is not None else _encode_one_piece(commit_token)
            if best_id is not None:
                input_ids[0, pos] = best_id

        # Rebuild SPECIAL_MASKED_KEY from best suggestions (unchanged)
        word_entries: List[Tuple[int, str, Any]] = []
        for k, info in label_output.items():
            if k == SPECIAL_MASKED_KEY:
                continue
            if not isinstance(k, tuple) or len(k) != 2:
                continue
            tok, idx = k
            word_entries.append((idx, tok, info))

        word_entries.sort(key=lambda t: t[0])

        final_tokens: List[str] = []
        for _, tok, info in word_entries:
            if isinstance(info, dict):
                suggs = info.get("suggestions")
                if suggs:
                    final_tokens.append(suggs[0])
                    continue
            final_tokens.append(tok)

        label_output[SPECIAL_MASKED_KEY] = detok.detokenize(final_tokens)
        return label_output

    return update_with_bert



# ----------------------------
# Export formatting
# ----------------------------
def format_label_output_for_export(label_output: dict) -> dict:
    """
    Take the final label_output (after non-word + context + BERT suggestions)
    and convert it into the format:

        {
            (token, idx): [suggestions...] or [],
            ('spelling', -1): [
                {('token', spelling_idx_from_zero): [suggestions...]},
                ...
            ],
            ('context', -2): [
                {('token', context_idx_from_zero): [suggestions...]},
                ...
            ],
            ('suggested', -3): "<final suggested sentence>",
        }

    where:
      - spelling_idx_from_zero is the order of spelling errors (e == 'n')
      - context_idx_from_zero is the order of context errors (e == 'c')
      - 'suggestions' is taken from label_output[(token, idx)]['suggestions']
        if present, else [].
    """

    out: dict = {}

    # Collect normal token keys (exclude the special masked key)
    token_keys = [
        k
        for k in label_output.keys()
        if k != SPECIAL_MASKED_KEY
        and isinstance(k, tuple)
        and len(k) == 2
    ]
    # Sort by original inner index
    token_keys.sort(key=lambda k: k[1])

    spelling_errors: list[dict] = []
    context_errors: list[dict] = []
    spelling_idx = 0
    context_idx = 0

    # First pass: per-token suggestions and build spelling/context lists
    for tok, idx in token_keys:
        info = label_output.get((tok, idx))

        # Extract suggestions or default to []
        suggestions: list[str] = []
        if isinstance(info, dict):
            s = info.get("suggestions")
            if isinstance(s, list):
                suggestions = s

        # Top-level entry: (token, original_idx) -> suggestions list (or [])
        out[(tok, idx)] = suggestions

        # Error-type specific indexing
        if isinstance(info, dict) and "e" in info:
            err_type = info["e"]

            if err_type == "n":
                # Spelling error: index from 0 in order of appearance
                spelling_errors.append({(tok, spelling_idx): suggestions})
                spelling_idx += 1

            elif err_type == "c":
                # Context error: index from 0 in order of appearance
                context_errors.append({(tok, context_idx): suggestions})
                context_idx += 1

    # Attach aggregated spelling and context error lists
    out[SPELLING_KEY] = spelling_errors
    out[CONTEXT_KEY] = context_errors

    # Suggested sentence: taken directly from SPECIAL_MASKED_KEY
    suggested_sentence = label_output.get(SPECIAL_MASKED_KEY, "")
    out[SUGGESTED_KEY] = suggested_sentence

    return out


# ----------------------------
# Pipelines
# ----------------------------
def setup():
    lm = BigramLM.from_json(
        unigram_path="unigrams.json",
        bigram_path="bigrams.json",
        bigram_format="packed",
        k=0.1,
        lowercase_vocab=True,
    )

    vocab = {w.lower() for w in lm.V}
    print(len(vocab))

    suggester = make_unified_bert_suggester(
        "JonathanChang/bert_finance_continued",
        candidate_vocab=vocab,
        tau=0,          # optional
        fallback_top_k=20 # optional
    )
    
    return lm, vocab, suggester


def spelling_errors(lm, vocab, suggester, paragraph: str) -> dict:
    labeled = label_paragraph_and_mask_for_bert_nonword(paragraph, vocab)
    labeled = suggester(labeled)

    print("\nUpdated labeled dict:")
    print(labeled)
    print(labeled[SPECIAL_MASKED_KEY])

    final_struct = format_label_output_for_export(labeled)
    print("Final:")
    print(final_struct)
    return final_struct


def context_errors(lm, vocab, suggester, paragraph: str) -> dict:
    # First: non-word (typo) detection + BERT suggestions
    labeled = label_paragraph_and_mask_for_bert_nonword(paragraph, vocab)
    labeled = suggester(labeled)

    print("\nUpdated labeled dict (after nonword stage):")
    print(labeled)
    print(labeled[SPECIAL_MASKED_KEY])

    with open("tau_by_pos.json", "r", encoding="utf-8") as f:
        tau_by_pos = json.load(f)

    # Then: context error detection using the bigram LM
    labeled = label_paragraph_and_mask_for_bert_context(
        labeled,
        lm,
        vocab=vocab,
        cand_x=2,
        tau_default=2.0,
        tau_by_pos=tau_by_pos,
    )
    
    # labeled = label_paragraph_and_mask_for_bert_context(
    #     labeled,
    #     lm,
    #     vocab=vocab,
    #     # cand_x=1,
    #     tau_default=-15.0,
    #     tau_by_pos=None,
    # )
    print(labeled[SPECIAL_MASKED_KEY])

    # And run BERT suggester again for the new [MASK]s from the context stage
    labeled = suggester(labeled)

    print("\nUpdated labeled dict (after context stage):")
    print(labeled)
    print(labeled[SPECIAL_MASKED_KEY])

    final_struct = format_label_output_for_export(labeled)
    return final_struct


def model(lm, vocab, suggester, paragraph: str, mode: str = "c") -> dict:
    if mode == "n":
        return spelling_errors(lm, vocab, suggester, paragraph)
    else:
        return context_errors(lm, vocab, suggester, paragraph)


# Quick test call
if __name__ == "__main__":
    paragraph = "I have an idet. I want to make many monei with it."
    # paragraph = "He is a god guy who happens to has done bad stuff."
    # paragraph = "i am rech and my friens is very pour."
    lm, vocab, suggester = setup()
    print(model(lm, vocab, suggester, paragraph=paragraph, mode="c"))
