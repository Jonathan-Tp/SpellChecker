from __future__ import annotations
#Refactored version
"""
Refactored spell + context error pipeline.

Goals:
- Keep the *final* `model(...)` output format identical to the original.
- Remove duplicated imports / debug prints.
- Fix correctness issues in context candidate selection.
- Improve efficiency: reuse precomputed vocab forms, cache POS tagging, avoid rebuilding constants in loops.
- Keep dependencies optional: NLTK / torch / transformers are imported lazily.
"""

from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import platform
import re
import string

# =============================================================================
# Constants / types
# =============================================================================

PUNCT_CHARS = set(string.punctuation)
_SPECIAL_MASK = "[MASK]"

TokenKey = Tuple[str, int]  # (token, inner_idx)
LabelInfo = Optional[Dict[str, Any]]
LabelOutput = Dict[Any, Any]  # includes TokenKey and SPECIAL_MASKED_KEY

SPECIAL_MASKED_KEY: Tuple[str, int] = ("__MASKED__", -1)
SPELLING_KEY: Tuple[str, int] = ("spelling", -1)
CONTEXT_KEY: Tuple[str, int] = ("context", -2)
SUGGESTED_KEY: Tuple[str, int] = ("suggested", -3)

_ALPHA_RE = re.compile(r"^[A-Za-z]+$")


# =============================================================================
# Robust NLTK utilities (tokenization + POS)
# =============================================================================

_NLTK_READY = False


def _try_import_nltk():
    try:
        import nltk  # type: ignore
        from nltk import pos_tag  # type: ignore
        from nltk.tokenize import sent_tokenize, word_tokenize  # type: ignore
        from nltk.tokenize.treebank import TreebankWordDetokenizer  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return nltk, pos_tag, sent_tokenize, word_tokenize, TreebankWordDetokenizer


def ensure_nltk(*, force: bool = False) -> None:
    """Best-effort download of NLTK data into a writable directory."""
    global _NLTK_READY
    if _NLTK_READY and not force:
        return

    imported = _try_import_nltk()
    if imported is None:  # NLTK not installed
        _NLTK_READY = True
        return

    nltk, _, _, _, _ = imported
    try:
        base_dir = Path.home() / "nltk_data"
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover
        base_dir = Path("/tmp/nltk_data")
        base_dir.mkdir(parents=True, exist_ok=True)

    if str(base_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(base_dir))

    packages = [
        "punkt",
        "punkt_tab",  # required by newer PunktTokenizer
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",  # newer tagger name (safe to attempt)
        "universal_tagset",
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, download_dir=str(base_dir), quiet=True)
        except Exception:
            # No hard fail: we have fallbacks.
            pass

    _NLTK_READY = True


_FALLBACK_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


def _fallback_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _fallback_words(sent: str) -> List[str]:
    return _FALLBACK_WORD_RE.findall(sent)


def tokenize_paragraph(paragraph: str) -> List[str]:
    """Tokenize paragraph into ['<s>', ...tokens..., '</s>'] with graceful fallbacks."""
    paragraph = paragraph or ""
    paragraph = paragraph.strip()
    if not paragraph:
        return ["<s>", "</s>"]

    ensure_nltk()
    imported = _try_import_nltk()
    if imported is not None:
        _, _, sent_tokenize, word_tokenize, _ = imported
        try:
            tokens: List[str] = []
            for sent in sent_tokenize(paragraph):
                tokens.extend(word_tokenize(sent))
            return ["<s>"] + tokens + ["</s>"]
        except LookupError:
            # fall through to regex fallback
            pass
        except Exception:
            pass

    tokens2: List[str] = []
    for sent in _fallback_sentences(paragraph):
        tokens2.extend(_fallback_words(sent))
    return ["<s>"] + tokens2 + ["</s>"]


# Penn Treebank -> Universal mapping (best-effort)
_PTB_TO_UNIVERSAL_EXACT: Dict[str, str] = {
    # Nouns
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "NOUN",
    "NNPS": "NOUN",
    # Verbs
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    # Adjectives
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    # Adverbs
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    # Determiners
    "DT": "DET",
    "PDT": "DET",
    "WDT": "DET",
    # Conjunctions
    "CC": "CONJ",
    # Prepositions / particles
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
    if not tag:
        return "X"
    if tag in _PTB_TO_UNIVERSAL_EXACT:
        return _PTB_TO_UNIVERSAL_EXACT[tag]
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"
    return "X"


def is_punct_token(tok: str) -> bool:
    return bool(tok) and all(ch in PUNCT_CHARS for ch in tok)


def _simple_universal(tok: str) -> str:
    """Tiny heuristic POS fallback."""
    t = (tok or "").strip()
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
    if low == "to":
        return "PRT"
    if low in {
        "in", "on", "at", "by", "for", "with", "from", "of", "into", "over", "under",
        "between", "after", "before", "about",
    }:
        return "ADP"
    if low in {
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
        "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    }:
        return "PRON"
    if low.endswith("ly"):
        return "ADV"
    return "NOUN"


@lru_cache(maxsize=50_000)
def _pos_tag_universal_cached(tokens_tuple: Tuple[str, ...]) -> Tuple[Tuple[str, str], ...]:
    tokens = list(tokens_tuple)

    ensure_nltk()
    imported = _try_import_nltk()
    if imported is not None:
        _, pos_tag, _, _, _ = imported
        try:
            return tuple(pos_tag(tokens, tagset="universal"))
        except LookupError:
            pass
        except Exception:
            pass
        try:
            ptb = pos_tag(tokens)
            return tuple((tok, _ptb_to_universal(tag)) for tok, tag in ptb)
        except LookupError:
            pass
        except Exception:
            pass

    return tuple((tok, _simple_universal(tok)) for tok in tokens)


def safe_pos_tag_universal(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    return list(_pos_tag_universal_cached(tuple(tokens)))


@lru_cache(maxsize=200_000)
def candidate_pos_tag(prev: str, cand: str, nxt: str) -> str:
    """POS tag for cand, using a tiny context window when available."""
    window: List[str] = []
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


# =============================================================================
# Bigram LM
# =============================================================================

def _read_json_dict(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object/dict.")
    return data


def _parse_bigram_key(key: str, delim: str) -> Tuple[str, str]:
    parts = key.split(delim)
    if len(parts) != 2:
        raise ValueError(f"Bigram key {key!r} did not split into 2 parts with delim={delim!r}")
    return parts[0], parts[1]


class BigramLM:
    def __init__(self, k: float = 0.1) -> None:
        self.k = float(k)
        self.unigram: Counter[str] = Counter()
        self.bigram: Counter[Tuple[str, str]] = Counter()
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

        for w, c in unigram_data.items():
            if not isinstance(w, str):
                continue
            w2 = w.lower() if lowercase_vocab else w
            lm.unigram[w2] += int(c)

        lm.V = {w for w in lm.unigram.keys() if w != "<s>"}

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

        if is_packed:
            vocab = bigram_data.get("vocab")
            triples = bigram_data.get("bigrams")
            if not isinstance(vocab, list) or not isinstance(triples, list):
                raise ValueError(
                    'Packed bigram JSON must be like: {"vocab":[...], "bigrams":[[prev_id,next_id,count], ...]}'
                )

            if lowercase_vocab:
                vocab2: List[str] = []
                for tok in vocab:
                    if not isinstance(tok, str):
                        raise ValueError("Packed vocab must be a list[str]")
                    vocab2.append(tok.lower())
                vocab = vocab2
            else:
                for tok in vocab:
                    if not isinstance(tok, str):
                        raise ValueError("Packed vocab must be a list[str]")

            for t in triples:
                if not (isinstance(t, list) and len(t) == 3):
                    raise ValueError("Packed bigrams must be a list of [i,j,c] triples")
                i, j, c = t
                if not (isinstance(i, int) and isinstance(j, int)):
                    raise ValueError("Packed bigram ids must be ints")
                if not isinstance(c, (int, float)):
                    raise ValueError("Packed bigram counts must be numeric")
                if i < 0 or j < 0 or i >= len(vocab) or j >= len(vocab):
                    raise ValueError(f"Bigram id out of range: {(i, j)} with vocab_size={len(vocab)}")
                prev = vocab[i]
                w = vocab[j]
                lm.bigram[(prev, w)] += int(c)
                if w != "<s>":
                    lm.V.add(w)
        else:
            for key, c in bigram_data.items():
                if not isinstance(key, str):
                    continue
                prev, w = _parse_bigram_key(key, bigram_delim)
                if lowercase_vocab:
                    prev = prev.lower()
                    w = w.lower()
                lm.bigram[(prev, w)] += int(c)
                if w != "<s>":
                    lm.V.add(w)

        lm.V_size = len(lm.V)
        return lm

    def prob(self, prev: str, w: str) -> float:
        num = self.bigram[(prev, w)] + self.k
        den = self.unigram[prev] + self.k * self.V_size
        return num / den if den > 0 else 1.0 / max(self.V_size, 1)

    def logprob(self, prev: str, w: str) -> float:
        return math.log(self.prob(prev, w))


# =============================================================================
# Edit distance (with optional cutoff) + vocab index
# =============================================================================

def edit_distance(a: str, b: str, *, max_dist: Optional[int] = None) -> int:
    """Levenshtein distance; if max_dist is set, may early-exit returning >max_dist."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    # If length difference already exceeds max_dist, early exit
    if max_dist is not None and (len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        row_min = curr[0]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            v = ins if ins < delete else delete
            if sub < v:
                v = sub
            curr[j] = v
            if v < row_min:
                row_min = v
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class VocabIndex:
    """Small helper to speed up edit-distance candidate lookup."""
    vocab: Tuple[str, ...]
    vocab_lc: frozenset[str]
    buckets_by_len: Dict[int, Tuple[str, ...]]

    @classmethod
    def build(cls, vocab: Iterable[str]) -> "VocabIndex":
        items = [v for v in vocab if isinstance(v, str) and v]
        vocab_lc = frozenset(v.lower() for v in items)
        buckets: Dict[int, List[str]] = {}
        for w in items:
            buckets.setdefault(len(w), []).append(w)
        buckets_final: Dict[int, Tuple[str, ...]] = {k: tuple(v) for k, v in buckets.items()}
        return cls(vocab=tuple(items), vocab_lc=vocab_lc, buckets_by_len=buckets_final)

    def is_oov_alpha(self, word: str) -> bool:
        return bool(word) and word.isalpha() and (word.lower() not in self.vocab_lc)

    def within_edit_distance(self, word: str, max_dist: int, *, max_candidates: Optional[int] = None) -> List[str]:
        if max_dist < 0:
            return []
        w = word.lower()
        L = len(word)
        out: List[str] = []
        # Only scan lengths within the band
        for l in range(max(1, L - max_dist), L + max_dist + 1):
            bucket = self.buckets_by_len.get(l)
            if not bucket:
                continue
            for cand in bucket:
                if cand.lower() == w:
                    continue
                if edit_distance(w, cand.lower(), max_dist=max_dist) <= max_dist:
                    out.append(cand)
                    if max_candidates is not None and len(out) >= max_candidates:
                        return out
        return out


# =============================================================================
# Non-word detection (typos)
# =============================================================================

def label_nonword_and_mask(
    paragraph: str,
    vocab_index: VocabIndex,
    *,
    ed_thresh: int = 2,
    max_suggestions: int = 20,
    mask_if_wrong: bool = True,
    mask_token: str = _SPECIAL_MASK,
) -> LabelOutput:
    """
    Detect non-word (OOV alphabetical tokens) and attach edit-distance suggestions.
    Produces a masked paragraph under SPECIAL_MASKED_KEY.

    Output dict:
      (token, idx) -> None
      (token, idx) -> {'e': 'n', 'suggestions': [..]}
      SPECIAL_MASKED_KEY -> masked paragraph
    """
    toks = tokenize_paragraph(paragraph)
    detok = _get_detokenizer()

    out: LabelOutput = {}
    masked_tokens: List[str] = []

    inner_len = max(0, len(toks) - 2)
    for i in range(1, 1 + inner_len):
        word = toks[i]
        idx = i - 1

        info: LabelInfo = None
        is_alpha = _ALPHA_RE.match(word) is not None
        wrong = bool(is_alpha and vocab_index.is_oov_alpha(word))

        if wrong:
            cand = vocab_index.within_edit_distance(word, ed_thresh, max_candidates=max_suggestions * 4)
            # Sort by (distance, lexicographic) then truncate
            cand.sort(key=lambda w: (edit_distance(word.lower(), w.lower()), w.lower()))
            cand = cand[:max_suggestions]
            info = {"e": "n", "suggestions": cand}

        out[(word, idx)] = info

        do_mask = (wrong and mask_if_wrong) or ((not wrong) and (not mask_if_wrong))
        if do_mask and is_alpha and not is_punct_token(word):
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append(word)

    out[SPECIAL_MASKED_KEY] = detok.detokenize(masked_tokens)
    return out


# =============================================================================
# Context labeling + masking
# =============================================================================

# Closed-class candidate pools (fast + deterministic)
CLOSED_CLASS_CANDIDATES_BY_POS: Dict[str, Tuple[str, ...]] = {
    "DET": (
        "a", "an", "the", "this", "that", "these", "those", "some", "any", "no",
        "each", "every", "another", "other", "all", "both",
    ),
    "ADP": (
        "in", "on", "at", "to", "for", "from", "of", "with", "by", "about",
        "over", "under", "into", "onto", "between", "through", "during", "before",
        "after", "against", "within", "without",
    ),
    "CONJ": ("and", "or", "but", "nor", "yet", "so"),
    "PRT": ("to", "up", "off", "out", "in", "on", "over", "down", "away", "back"),
    "PRON": (
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    ),
}


def context_score(lm: BigramLM, prev: str, w: str, nxt: str) -> float:
    # two-sided bigram score
    return lm.logprob(prev, w) + lm.logprob(w, nxt)


def out_of_context_by_bigram_pos(
    *,
    lm: BigramLM,
    prev: str,
    word: str,
    nxt: str,
    candidates: Iterable[str],
    tau_default: float = 2.0,
    tau_by_pos: Optional[Dict[str, float]] = None,
) -> bool:
    """Return True if word is out-of-context compared to best candidate by >= tau."""
    if not word or is_punct_token(word):
        return False

    tau_by_pos = tau_by_pos or {}

    prev_l = (prev or "<s>").lower()
    word_l = word.lower()
    nxt_l = (nxt or "</s>").lower()

    cur_pos = candidate_pos_tag(prev_l, word_l, nxt_l)
    tau = float(tau_by_pos.get(cur_pos, tau_default))

    orig_score = context_score(lm, prev_l, word_l, nxt_l)
    best_score = orig_score

    for c in candidates:
        if not c:
            continue
        c_l = c.lower()
        if c_l == word_l:
            continue
        score = context_score(lm, prev_l, c_l, nxt_l)
        if score > best_score:
            best_score = score

    return (best_score - orig_score) >= tau


def label_context_and_mask(
    label_output: LabelOutput,
    lm: BigramLM,
    vocab_index: VocabIndex,
    *,
    cand_x: int = 3,
    tau_default: float = 2.0,
    tau_by_pos: Optional[Dict[str, float]] = None,
    mask_if_wrong: bool = True,
    mask_token: str = _SPECIAL_MASK,
    filter_candidates_by_pos: bool = False,
) -> LabelOutput:
    """
    Re-label tokens for CONTEXT errors using the bigram LM.

    The LM context stream uses "representative tokens":
      - first suggestion if present, else original token.
    """
    # Extract ordered token items
    token_items: List[Dict[str, Any]] = []
    for k, info in label_output.items():
        if k == SPECIAL_MASKED_KEY:
            continue
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        orig_tok, idx = k
        rep_tok = orig_tok
        if isinstance(info, dict):
            suggs = info.get("suggestions")
            if isinstance(suggs, list) and suggs:
                rep_tok = suggs[0]
        token_items.append({"key": k, "idx": idx, "orig_tok": orig_tok, "rep_tok": rep_tok})

    token_items.sort(key=lambda x: x["idx"])

    base_tokens_lc = [(item["rep_tok"] or "").lower() for item in token_items]
    toks = ["<s>"] + base_tokens_lc + ["</s>"]

    # Decide context errors
    for pos, item in enumerate(token_items):
        key = item["key"]
        rep_tok = (item["rep_tok"] or "")
        info = label_output.get(key)

        if not rep_tok or rep_tok in {"<s>", "</s>"} or is_punct_token(rep_tok) or not rep_tok.isalpha():
            continue

        prev = toks[pos]
        word = toks[pos + 1]
        nxt = toks[pos + 2]

        pos_tag = candidate_pos_tag(prev, word, nxt)

        fixed = CLOSED_CLASS_CANDIDATES_BY_POS.get(pos_tag)
        cand_list: List[str] = []
        if fixed is not None:
            cand_list = list(fixed)
        elif cand_x >= 0:
            # open-class: edit-distance candidates from vocab
            cand_list = vocab_index.within_edit_distance(rep_tok, cand_x, max_candidates=200)
            if filter_candidates_by_pos:
                filtered: List[str] = []
                for cand in cand_list:
                    if candidate_pos_tag(prev, cand.lower(), nxt) == pos_tag:
                        filtered.append(cand)
                cand_list = filtered

        cand_set = {c for c in cand_list if c and c.lower() != word.lower()}
        if not cand_set:
            continue

        is_ctx_error = out_of_context_by_bigram_pos(
            lm=lm,
            prev=prev,
            word=word,
            nxt=nxt,
            candidates=cand_set,
            tau_default=tau_default,
            tau_by_pos=tau_by_pos,
        )

        if is_ctx_error:
            label_output[key] = {"e": "c", "suggestions": []}

    # Rebuild masked paragraph using representative tokens
    detok = _get_detokenizer()
    masked_tokens: List[str] = []
    for item in token_items:
        key = item["key"]
        rep_tok = item["rep_tok"]
        info = label_output.get(key)
        is_ctx_error = isinstance(info, dict) and info.get("e") == "c"
        do_mask = (is_ctx_error and mask_if_wrong) or ((not is_ctx_error) and (not mask_if_wrong))

        if do_mask and rep_tok and rep_tok.isalpha() and not is_punct_token(rep_tok):
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append(rep_tok)

    label_output[SPECIAL_MASKED_KEY] = detok.detokenize(masked_tokens)
    return label_output


# =============================================================================
# BERT suggester
# =============================================================================

def make_unified_bert_suggester(
    model_ref: str,
    *,
    candidate_vocab: Optional[Iterable[str]] = None,
    tau: float = 0.0,
    fallback_top_k: int = 30,
) -> Callable[[LabelOutput], LabelOutput]:
    """
    Unified masked-LM suggester.

    It expects SPECIAL_MASKED_KEY to contain a masked sentence with [MASK] tokens.
    For each masked token (in left-to-right order), it:
      1) scores candidates using the MLM
      2) restricts to candidate_vocab (single-wordpiece only) when provided
      3) stores suggestions in label_output[(token, idx)]["suggestions"]
      4) commits one token to the running context (with tau margin vs original token)
    """
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("make_unified_bert_suggester requires torch + transformers") from e

    # Device selection: prefer CUDA; use MPS only on macOS.
    use_mps = (
        platform.system() == "Darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    # Load weights on CPU first; then move.
    model = AutoModelForMaskedLM.from_pretrained(model_ref, low_cpu_mem_usage=False, device_map=None)
    try:
        model = model.to(device)
    except Exception:
        device = torch.device("cpu")
        model = model.to(device)
    model.eval()

    MASK = tokenizer.mask_token or _SPECIAL_MASK
    MASK_ID = tokenizer.mask_token_id
    if MASK_ID is None:
        raise ValueError("Tokenizer has no mask_token_id; cannot run MLM suggester.")

    def _encode_one_piece(s: str) -> Optional[int]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _mask_positions(input_ids_1d) -> List[int]:
        return [i for i, x in enumerate(input_ids_1d.tolist()) if x == MASK_ID]

    # Pre-encode candidate vocab (single-piece) once
    vocab_token_ids: Dict[str, int] = {}
    if candidate_vocab is not None:
        for w in candidate_vocab:
            tid = _encode_one_piece(w)
            if tid is not None:
                vocab_token_ids[str(w)] = tid

    @torch.inference_mode()
    def update_with_bert(label_output: LabelOutput) -> LabelOutput:
        if SPECIAL_MASKED_KEY not in label_output:
            raise ValueError(f"label_output missing SPECIAL_MASKED_KEY={SPECIAL_MASKED_KEY}")

        masked_text = label_output[SPECIAL_MASKED_KEY]
        if not isinstance(masked_text, str) or MASK not in masked_text:
            return label_output

        enc = tokenizer(masked_text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]  # (1, seq)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        token_type_ids = enc.get("token_type_ids", None)

        mask_pos_list = _mask_positions(input_ids[0])

        # Collect keys that correspond to errors (have 'e' field), sorted by idx.
        error_entries: List[Tuple[int, TokenKey]] = []
        for k, info in label_output.items():
            if k == SPECIAL_MASKED_KEY:
                continue
            if not (isinstance(k, tuple) and len(k) == 2):
                continue
            if isinstance(info, dict) and info.get("e") in {"n", "c"}:
                error_entries.append((int(k[1]), k))
        error_entries.sort(key=lambda x: x[0])
        label_keys = [k for _, k in error_entries]

        # If mismatch, fall back to "type with matching count" heuristic (kept from original behavior).
        if len(label_keys) != len(mask_pos_list):
            by_type: Dict[str, List[Tuple[int, TokenKey]]] = {}
            for idx, k in error_entries:
                err_type = str(label_output[k]["e"])
                by_type.setdefault(err_type, []).append((idx, k))
            chosen: Optional[List[Tuple[int, TokenKey]]] = None
            for g in by_type.values():
                if len(g) == len(mask_pos_list):
                    chosen = g
                    break
            if chosen is None:
                raise ValueError(
                    "Mismatch between [MASK] tokens and error entries.\n"
                    f"found {len(mask_pos_list)} [MASK] in masked text but error entries are {len(error_entries)}.\n"
                    f"masked_text: {masked_text}"
                )
            chosen.sort(key=lambda x: x[0])
            label_keys = [k for _, k in chosen]

        for pos, key in zip(mask_pos_list, label_keys):
            info = label_output.get(key)
            if not isinstance(info, dict):
                continue

            orig_tok = key[0]

            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids

            out = model(**model_inputs)
            logprobs = torch.log_softmax(out.logits[0, pos], dim=-1)

            # Candidate set
            if vocab_token_ids:
                # Restrict to candidate vocab ids
                scored: List[Tuple[float, str, int]] = []
                for w, tid in vocab_token_ids.items():
                    scored.append((float(logprobs[int(tid)].item()), w, tid))
                scored.sort(key=lambda t: t[0], reverse=True)
                top = scored[:max(1, fallback_top_k)]
                candidates = [(lp, w, tid) for lp, w, tid in top]
            else:
                # Use raw MLM top-k
                top_ids = torch.topk(logprobs, k=max(1, fallback_top_k)).indices.tolist()
                candidates = []
                for tid in top_ids:
                    w = tokenizer.decode([tid]).strip()
                    candidates.append((float(logprobs[int(tid)].item()), w, int(tid)))

            # sort by edit distance, then prob
            suggestions = [w for _, w, _ in sorted(
                candidates,
                key=lambda t: (edit_distance(orig_tok.lower(), t[1].lower()), -t[0], t[1].lower()),
            )]

            # Commit token with tau margin vs original token.
            commit_token = suggestions[0] if suggestions else orig_tok
            commit_tid = _encode_one_piece(commit_token) or (candidates[0][2] if candidates else None)

            if tau > 0:
                orig_id = _encode_one_piece(orig_tok)
                if orig_id is not None and candidates:
                    commit_lp = float(logprobs[int(commit_tid)].item()) if commit_tid is not None else -1e9
                    orig_lp = float(logprobs[int(orig_id)].item())
                    if (commit_lp - orig_lp) <= tau:
                        commit_token = orig_tok
                        commit_tid = orig_id
                        if orig_tok in suggestions:
                            suggestions = [orig_tok] + [s for s in suggestions if s != orig_tok]
                        else:
                            suggestions = [orig_tok] + suggestions

            info["suggestions"] = suggestions

            if commit_tid is not None:
                input_ids[0, pos] = int(commit_tid)

        # Rebuild SPECIAL_MASKED_KEY from committed tokens (matches original approach)
        word_entries: List[Tuple[int, str, Any]] = []
        for k, info in label_output.items():
            if k == SPECIAL_MASKED_KEY:
                continue
            if not (isinstance(k, tuple) and len(k) == 2):
                continue
            tok, idx = k
            word_entries.append((int(idx), tok, info))
        word_entries.sort(key=lambda t: t[0])

        final_tokens: List[str] = []
        for idx, tok, info in word_entries:
            if isinstance(info, dict) and info.get("e") in {"n", "c"}:
                suggs = info.get("suggestions")
                if isinstance(suggs, list) and suggs:
                    final_tokens.append(str(suggs[0]))
                else:
                    final_tokens.append(tok)
            else:
                final_tokens.append(tok)

        detok = _get_detokenizer()
        label_output[SPECIAL_MASKED_KEY] = detok.detokenize(final_tokens)
        return label_output

    return update_with_bert


# =============================================================================
# Final export formatting + public pipeline
# =============================================================================

def format_label_output_for_export(label_output: LabelOutput) -> Dict[Any, Any]:
    """
    Convert internal label_output into the required export format:

        {
          (token, idx): [suggestions...] or [],
          ('spelling', -1): [{('token', spelling_idx): [suggestions...]} ...],
          ('context',  -2): [{('token', context_idx):  [suggestions...]} ...],
          ('suggested', -3): "<final suggested sentence>",
        }
    """
    out: Dict[Any, Any] = {}

    token_keys = [
        k for k in label_output.keys()
        if k != SPECIAL_MASKED_KEY and isinstance(k, tuple) and len(k) == 2
    ]
    token_keys.sort(key=lambda k: k[1])

    spelling_errors: List[Dict[Any, Any]] = []
    context_errors: List[Dict[Any, Any]] = []
    spelling_idx = 0
    context_idx = 0

    for tok, idx in token_keys:
        info = label_output.get((tok, idx))
        suggestions: List[str] = []
        if isinstance(info, dict):
            s = info.get("suggestions")
            if isinstance(s, list):
                suggestions = s

        out[(tok, idx)] = suggestions

        if isinstance(info, dict) and "e" in info:
            err_type = info["e"]
            if err_type == "n":
                spelling_errors.append({(tok, spelling_idx): suggestions})
                spelling_idx += 1
            elif err_type == "c":
                context_errors.append({(tok, context_idx): suggestions})
                context_idx += 1

    out[SPELLING_KEY] = spelling_errors
    out[CONTEXT_KEY] = context_errors
    out[SUGGESTED_KEY] = label_output.get(SPECIAL_MASKED_KEY, "")
    return out


@dataclass
class SpellChecker:
    lm: BigramLM
    vocab_index: VocabIndex
    suggester: Callable[[LabelOutput], LabelOutput]
    tau_by_pos: Optional[Dict[str, float]] = None

    def spelling_only(self, paragraph: str) -> Dict[Any, Any]:
        labeled = label_nonword_and_mask(paragraph, self.vocab_index)
        labeled = self.suggester(labeled)
        return format_label_output_for_export(labeled)

    def with_context(self, paragraph: str) -> Dict[Any, Any]:
        labeled = label_nonword_and_mask(paragraph, self.vocab_index)
        labeled = self.suggester(labeled)
        labeled = label_context_and_mask(
            labeled,
            self.lm,
            self.vocab_index,
            cand_x=2,
            tau_default=2.0,
            tau_by_pos=self.tau_by_pos,
            filter_candidates_by_pos=False,
        )
        labeled = self.suggester(labeled)
        return format_label_output_for_export(labeled)

    def model(self, paragraph: str, mode: str = "c") -> Dict[Any, Any]:
        return self.spelling_only(paragraph) if mode == "n" else self.with_context(paragraph)


# Backwards-compatible wrappers ------------------------------------------------

_DETOK = None


def _get_detokenizer():
    global _DETOK
    if _DETOK is not None:
        return _DETOK
    imported = _try_import_nltk()
    if imported is not None:
        _, _, _, _, TreebankWordDetokenizer = imported
        _DETOK = TreebankWordDetokenizer()
        return _DETOK

    class _FallbackDetok:
        def detokenize(self, tokens: Sequence[str]) -> str:
            # Simple join + a little punctuation spacing fix.
            s = " ".join(tokens)
            s = re.sub(r"\s+([,.!?;:])", r"\1", s)
            s = re.sub(r"([(\[{])\s+", r"\1", s)
            s = re.sub(r"\s+([)\]}])", r"\1", s)
            return s

    _DETOK = _FallbackDetok()
    return _DETOK


def setup(
    *,
    unigram_path: str | Path = "unigrams.json",
    bigram_path: str | Path = "bigrams.json",
    bigram_format: str = "packed",
    lm_k: float = 0.1,
    lowercase_vocab: bool = True,
    bert_model_ref: str = "JonathanChang/bert_finance_continued",
    tau_by_pos: Optional[Dict[str, float]] = None,
    tau_by_pos_path: Optional[str | Path] = None,
    bert_tau: float = 0.0,
    bert_top_k: int = 20,
) -> SpellChecker:
    """
    Build a ready-to-use SpellChecker pipeline.

    You can provide tau_by_pos as a dict or as a JSON file path via tau_by_pos_path.
    """
    lm = BigramLM.from_json(
        unigram_path=unigram_path,
        bigram_path=bigram_path,
        bigram_format=bigram_format,
        k=lm_k,
        lowercase_vocab=lowercase_vocab,
    )
    vocab_index = VocabIndex.build(lm.V)

    if tau_by_pos is None and tau_by_pos_path is not None:
        with Path(tau_by_pos_path).open("r", encoding="utf-8") as f:
            tau_by_pos = json.load(f)

    suggester = make_unified_bert_suggester(
        bert_model_ref,
        candidate_vocab=vocab_index.vocab,
        tau=bert_tau,
        fallback_top_k=bert_top_k,
    )
    return SpellChecker(lm=lm, vocab_index=vocab_index, suggester=suggester, tau_by_pos=tau_by_pos)


def spelling_errors(lm, vocab, suggester, paragraph: str) -> dict:
    # Compatibility: expect vocab as set[str]; build index once.
    vocab_index = VocabIndex.build(vocab)
    labeled = label_nonword_and_mask(paragraph, vocab_index)
    labeled = suggester(labeled)
    return format_label_output_for_export(labeled)


def context_errors(lm, vocab, suggester, paragraph: str) -> dict:
    """Compatibility wrapper for the original `context_errors` function.

    The original implementation loaded `tau_by_pos.json` from the current working
    directory. We preserve that behavior when the file exists, otherwise fall
    back to default tau only.
    """
    vocab_index = VocabIndex.build(vocab)

    labeled = label_nonword_and_mask(paragraph, vocab_index)
    labeled = suggester(labeled)

    tau_by_pos = None
    try:
        p = Path("tau_by_pos.json")
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                tau_by_pos = json.load(f)
    except Exception:
        tau_by_pos = None

    labeled = label_context_and_mask(
        labeled,
        lm,
        vocab_index,
        cand_x=2,
        tau_default=2.0,
        tau_by_pos=tau_by_pos,
        filter_candidates_by_pos=False,
    )
    labeled = suggester(labeled)
    return format_label_output_for_export(labeled)


def model(lm, vocab, suggester, paragraph: str, mode: str = "c") -> dict:
    return spelling_errors(lm, vocab, suggester, paragraph) if mode == "n" else context_errors(lm, vocab, suggester, paragraph)
