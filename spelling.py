import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Tuple, Callable

from huggingface_hub import hf_hub_download

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
def ensure_nltk() -> None:
    """Ensure required NLTK resources are available."""
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)


# ----------------------------
# Tokenization
# ----------------------------
def tokenize_paragraph(paragraph: str) -> list[str]:
    """Tokenize a paragraph with <s> ... </s> boundaries."""
    ensure_nltk()
    inner: list[str] = []
    for sent in sent_tokenize(paragraph):
        inner.extend(word_tokenize(sent))
    return ["<s>"] + inner + ["</s>"]


# ----------------------------
# Bigram LM
# ----------------------------
from pathlib import Path
from typing import Optional
from collections import Counter
import math

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
        # --- NEW: Hugging Face options ---
        hf_repo_id: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_repo_type: str = "model",  # usually "model"
    ) -> "BigramLM":
        """
        Load unigram/bigram counts from JSON.

        - If hf_repo_id is None: unigram_path/bigram_path are treated as local paths.
        - If hf_repo_id is set: unigram_path/bigram_path are treated as *filenames in the HF repo*
          and will be downloaded to the local HF cache first.
        """
        lm = cls(k=k)

        # If HF repo is provided, download the files first.
        if hf_repo_id:
            # treat arguments as filenames inside the HF repo
            uni_file = str(unigram_path)
            bi_file = str(bigram_path)

            unigram_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=uni_file,
                revision=hf_revision,
                token=hf_token,
                repo_type=hf_repo_type,
            )
            bigram_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=bi_file,
                revision=hf_revision,
                token=hf_token,
                repo_type=hf_repo_type,
            )

        unigram_data = _read_json_dict(unigram_path)
        bigram_data = _read_json_dict(bigram_path)

        for w, c in unigram_data.items():
            w2 = w.lower() if lowercase_vocab else w
            lm.unigram[w2] += int(c)

        lm.V = {w for w in lm.unigram.keys() if w != "<s>"}

        for key, c in bigram_data.items():
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
def _read_json_dict(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{p} must contain a JSON object/dict, got {type(data).__name__}"
        )
    return data


def _parse_bigram_key(key: str, delim: str = " ") -> tuple[str, str]:
    if not isinstance(key, str):
        raise ValueError(f"Bigram key must be a string. Got: {type(key).__name__}")

    parts = key.rsplit(delim, 1) if delim == " " else key.split(delim, 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Bigram key {key!r} did not split into 2 parts with delim={delim!r}"
        )
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
def candidate_pos_tag(prev: str, cand: str, nxt: str) -> str:
    window: list[str] = []
    if prev not in {"<s>", "</s>"} and not is_punct_token(prev):
        window.append(prev)
    window.append(cand)
    if nxt not in {"<s>", "</s>"} and not is_punct_token(nxt):
        window.append(nxt)

    tagged = pos_tag(window, tagset="universal")
    for tok, tag in tagged:
        if tok == cand:
            return tag
    return tagged[0][1]


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

# from typing import Iterable, Optional, Dict, Any, List, Tuple, Callable

# import torch
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# # SPECIAL_MASKED_KEY is already defined earlier in your file; if not, uncomment:
# # SPECIAL_MASKED_KEY = ("__MASKED__", -1)


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
#     # ---- robust load: avoid meta tensors / accelerate dispatch ----
#     load_kwargs = {
#         "low_cpu_mem_usage": False,  # avoid meta-init pathways
#         "device_map": None,          # avoid accelerate device dispatch/offload
#     }

#     # MPS is happiest with float32 for many transformer ops
#     if device.type == "mps":
#         load_kwargs["torch_dtype"] = torch.float32

#     model = AutoModelForMaskedLM.from_pretrained(model_ref, **load_kwargs)

#     # Safety check: if anything is still meta, fail early with a clearer hint
#     if any(getattr(p, "is_meta", False) for p in model.parameters()):
#         raise RuntimeError(
#             "Model loaded with meta parameters. This usually means accelerate/device_map "
#             "or low_cpu_mem_usage meta-init got triggered. Ensure device_map=None and "
#             "low_cpu_mem_usage=False when calling from_pretrained."
#         )

#     model = model.to(device)
#     model.eval()
    # model = AutoModelForMaskedLM.from_pretrained(model_ref).to(device)
    # model.eval()

from typing import Iterable, Optional, Dict, Any, Callable
import os
from functools import lru_cache

import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

# SPECIAL_MASKED_KEY should exist somewhere in your project:
# SPECIAL_MASKED_KEY = ("__MASKED__", -1)


def make_unified_bert_suggester(
    model_ref: str,
    *,
    candidate_vocab: Optional[Iterable[str]] = None,
    tau: float = 0.0,
    fallback_top_k: int = 30,
) -> Callable[[Dict[Any, Any]], Dict[Any, Any]]:
    """
    Unified BERT suggester (HF Hub only).

    model_ref MUST be a Hugging Face repo id like: "your-username/bert-finance-continued".
    (No local paths.)

    If the HF repo is private, set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in env/secrets.
    """

    # ---- Enforce "HF Hub only" ----
    if os.path.exists(model_ref):
        raise ValueError(
            f"model_ref='{model_ref}' looks like a local path. "
            "Pass a Hugging Face repo id like 'user/model'."
        )
    if "/" not in model_ref:
        raise ValueError(
            f"model_ref='{model_ref}' doesn't look like a Hugging Face repo id. "
            "Expected format 'user/model'."
        )

    # ---- Device selection ----
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # ---- Read token (only needed for private repos) ----
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # ---- from_pretrained wrapper: supports both `token=` and `use_auth_token=` ----
    def _from_pretrained(cls, repo_id: str, token: Optional[str], **kwargs):
        if token:
            try:
                return cls.from_pretrained(repo_id, token=token, **kwargs)
            except TypeError:
                return cls.from_pretrained(repo_id, use_auth_token=token, **kwargs)
        return cls.from_pretrained(repo_id, **kwargs)

    # ---- Cached loader (so multiple calls won't re-download in the same process) ----
    @lru_cache(maxsize=2)
    def _load(repo_id: str):
        tok = _from_pretrained(AutoTokenizer, repo_id, hf_token)

        load_kwargs = {
            "low_cpu_mem_usage": False,  # avoid meta-init pathways
            "device_map": None,          # avoid accelerate device dispatch/offload
        }
        if device.type == "mps":
            load_kwargs["torch_dtype"] = torch.float32

        mdl = _from_pretrained(AutoModelForMaskedLM, repo_id, hf_token, **load_kwargs)

        if any(getattr(p, "is_meta", False) for p in mdl.parameters()):
            raise RuntimeError(
                "Model loaded with meta parameters. Ensure device_map=None and low_cpu_mem_usage=False."
            )

        mdl = mdl.to(device)
        mdl.eval()
        return tok, mdl

    tokenizer, model = _load(model_ref)
    detok = TreebankWordDetokenizer()

    MASK = tokenizer.mask_token
    MASK_ID = tokenizer.mask_token_id
    detok = TreebankWordDetokenizer()

    # ----------------- helpers -----------------
    def _encode_one_piece(s: str) -> Optional[int]:
        """Return token id if s is a single-piece token; otherwise None."""
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _mask_positions(input_ids_1d: torch.Tensor) -> List[int]:
        ids = input_ids_1d.tolist()
        return [i for i, x in enumerate(ids) if x == MASK_ID]

    # Pre-encode candidate vocab once (for context errors / fallback)
    vocab_token_ids: Dict[str, int] = {}
    if candidate_vocab is not None:
        for w in candidate_vocab:
            tid = _encode_one_piece(w)
            if tid is not None:
                vocab_token_ids[w] = tid

    @torch.no_grad()
    def update_with_bert(label_output: Dict[Any, Any]) -> Dict[Any, Any]:
        if SPECIAL_MASKED_KEY not in label_output:
            raise ValueError(
                f"label_output missing SPECIAL_MASKED_KEY={SPECIAL_MASKED_KEY}"
            )

        masked_text = label_output[SPECIAL_MASKED_KEY]

        # No [MASK], nothing to do
        if MASK not in masked_text:
            return label_output

        # Encode masked text
        enc = tokenizer(masked_text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"][0]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        # Positions of [MASK] tokens in the HF tokenization
        mask_pos_list = _mask_positions(input_ids)

        # ---- Group error entries by error type ("n", "c", etc.) ----
        error_groups: Dict[str, List[Tuple[int, Tuple[str, int]]]] = {}
        for k, info in label_output.items():
            if k == SPECIAL_MASKED_KEY:
                continue
            if not isinstance(k, tuple) or len(k) != 2:
                continue
            if not isinstance(info, dict):
                continue
            if "e" not in info:
                continue

            tok, idx = k
            err_type = info.get("e")
            if err_type is None:
                continue
            error_groups.setdefault(err_type, []).append((idx, k))

        # Sort each group by idx
        for group in error_groups.values():
            group.sort(key=lambda x: x[0])

        total_errors = sum(len(g) for g in error_groups.values())
        num_masks = len(mask_pos_list)

        # Decide which error keys align with MASK positions
        if num_masks == total_errors:
            # All errors are masked
            merged: List[Tuple[int, Tuple[str, int]]] = []
            for g in error_groups.values():
                merged.extend(g)
            merged.sort(key=lambda x: x[0])
            label_keys: List[Tuple[str, int]] = [k for _, k in merged]
        else:
            # Mixed error types; choose the type whose count matches num_masks
            chosen_type: Optional[str] = None
            for t, g in error_groups.items():
                if len(g) == num_masks:
                    chosen_type = t
                    break

            if chosen_type is None:
                raise ValueError(
                    "Mismatch between [MASK] tokens and error entries.\n"
                    f"found {num_masks} [MASK] in masked text but "
                    f"error counts by type are: "
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

            # Run model for current context
            out = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )
            logprobs = torch.log_softmax(out.logits[0, pos], dim=-1)

            scored: List[Tuple[str, float, int]] = []  # (cand, logprob, token_id)

            # ---- Build candidate list ----
            if err_type == "c":
                # CONTEXT ERROR:
                #   use candidate_vocab ONLY (if provided),
                #   otherwise use BERT top-k (unconstrained).
                if vocab_token_ids:
                    for cand, cand_id in vocab_token_ids.items():
                        lp = float(logprobs[int(cand_id)].item())
                        scored.append((cand, lp, int(cand_id)))
                else:
                    top = torch.topk(logprobs, k=fallback_top_k)
                    for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
                        tok = tokenizer.decode([tid]).strip()
                        scored.append((tok, float(lp), int(tid)))
            else:
                # TYPO / NONWORD:
                #   start from existing suggestions (likely already vocab-based),
                #   but still enforce candidate_vocab if provided.
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

                # If no usable suggestion, fallback to vocab or BERT top-k
                if not scored:
                    if vocab_token_ids:
                        for cand, cand_id in vocab_token_ids.items():
                            lp = float(logprobs[int(cand_id)].item())
                            scored.append((cand, lp, int(cand_id)))
                    else:
                        top = torch.topk(logprobs, k=fallback_top_k)
                        for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
                            tok = tokenizer.decode([tid]).strip()
                            scored.append((tok, float(lp), int(tid)))

            if not scored:
                # nothing usable; skip this position
                continue

            # ---------------------------------------------------------
            # 1) Take top-k by probability (still within vocab if given)
            # ---------------------------------------------------------
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:fallback_top_k]

            # ---------------------------------------------------------
            # 2) Sort these top-k by edit distance, then prob
            # ---------------------------------------------------------
            scored_with_dist: List[Tuple[str, int, float, int]] = []
            for cand, lp, tid in scored:
                dist = edit_distance(orig_tok, cand)
                scored_with_dist.append((cand, dist, lp, tid))

            # sort: smaller edit distance first; tie-breaker: higher log-prob
            scored_with_dist.sort(key=lambda x: (x[1], -x[2]))

            # suggestions list is now ordered by (edit distance, -prob)
            suggestions = [cand for (cand, dist, lp, tid) in scored_with_dist]

            # ---------------------------------------------------------
            # 3) Decide which token to COMMIT into the context
            #    (by default: closest edit-distance candidate)
            # ---------------------------------------------------------
            first_cand, first_dist, first_lp, first_tid = scored_with_dist[0]
            commit_token = first_cand
            commit_tid: Optional[int] = first_tid
            commit_lp = first_lp

            if tau > 0.0:
                orig_id = _encode_one_piece(orig_tok)
                if orig_id is not None:
                    orig_lp = float(logprobs[int(orig_id)].item())
                    # If the chosen candidate doesn't beat original by > tau, keep original
                    if (commit_lp - orig_lp) <= tau:
                        commit_token = orig_tok
                        commit_tid = orig_id
                        # ensure original is first in suggestions
                        if orig_tok in suggestions:
                            suggestions = [orig_tok] + [s for s in suggestions if s != orig_tok]
                        else:
                            suggestions = [orig_tok] + suggestions

            # Save suggestions sorted by edit distance (with possible original at front)
            info["suggestions"] = suggestions

            # Update input_ids with committed token for later masks
            if commit_tid is None:
                best_id = _encode_one_piece(commit_token)
            else:
                best_id = commit_tid
            if best_id is not None:
                input_ids[pos] = best_id

        # -------------------------------------------------------------
        # Rebuild SPECIAL_MASKED_KEY from best suggestions
        # -------------------------------------------------------------
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
# def setup():
#     lm = BigramLM.from_json(
#         "unigram.json",
#         "bigram.json",
#         k=0.1,
#         hf_repo_id="JonathanChang/bert-finance-continued",
#     )

#     vocab = {w.lower() for w in lm.V}
#     print(len(vocab))

#     suggester = make_unified_bert_suggester(
#         "JonathanChang/bert-finance-continued",
#         candidate_vocab=vocab,
#         tau=0,          # optional
#         fallback_top_k=20 # optional
#     )
    
#     return lm, vocab, suggester
def setup1():
    lm = BigramLM.from_json(
        "unigram.json",
        "bigram.json",
        k=0.1,
        hf_repo_id="JonathanChang/bert-finance-continued",
    )

    vocab = {w.lower() for w in lm.V}
    return lm, vocab

def setup2():
    suggester = make_unified_bert_suggester(
        "JonathanChang/bert-finance-continued",
        candidate_vocab=vocab,
        tau=0,          # optional
        fallback_top_k=20 # optional
    )
    return suggester


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
    paragraph = "I has an idet. I want to make many monei with it."
    # paragraph = "He is a god guy who happens to has done bad stuff."
    # paragraph = "i am rech and my friens is very pour."
    lm, vocab, suggester = setup()
    print(model(lm, vocab, suggester, paragraph=paragraph, mode="c"))
