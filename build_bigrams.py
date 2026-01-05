#!/usr/bin/env python3
"""
build_bigrams.py

Process a .txt file (1 article per line) into a JSON dict:
  key   = "word1 word2"
  value = count (int)

To keep the output file small *without hurting performance too much*, this script:
  1) Builds a unigram vocabulary of the top-N most frequent tokens (vocab filtering).
     - This removes rare words that create a combinatorial explosion of rare bigrams.
  2) Counts bigrams only when BOTH words are in that vocabulary.
  3) Truncates bigrams using:
     - per-prefix top-M (keep the most common next-words for each previous word)
     - optional global top-K cap (final hard limit)

Output is compact JSON. If output ends with ".gz", it writes gzipped JSON.

Example:
  python build_bigrams.py --input articles.txt --output bigrams.json.gz \
      --vocab-size 50000 --prefix-top 40 --top-k 250000 --min-count 2

Notes:
  - For best LM-like usefulness at small size, per-prefix truncation is often better
    than only global top-K (it preserves common continuations for many contexts).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
from collections import Counter, defaultdict
from heapq import heappush, heappushpop
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# Fast, simple tokenizer:
# - keeps alphabetic tokens + basic apostrophe contractions (don't, it's, we're)
# - lowercasing is recommended for smaller vocab + better counts
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ---------------------------
# IO / tokenization
# ---------------------------

def iter_lines(path: Path, encoding: str = "utf-8") -> Iterator[str]:
    with path.open("r", encoding=encoding, errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n")


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    toks = _TOKEN_RE.findall(text)
    if lowercase:
        toks = [t.lower() for t in toks]
    return toks


# ---------------------------
# Truncation helpers
# ---------------------------

def keep_top_per_prefix(
    bigram_counts: Dict[Tuple[str, str], int],
    prefix_top: int,
) -> Dict[Tuple[str, str], int]:
    """
    For each prefix word w1, keep only the top `prefix_top` bigrams (w1, w2) by count.
    Uses tiny heaps per prefix so memory doesn't blow up during pruning.
    """
    if prefix_top <= 0:
        return dict(bigram_counts)

    heaps: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    # heap entries: (count, w2)  -> min-heap; keep only largest counts
    for (w1, w2), c in bigram_counts.items():
        h = heaps[w1]
        if len(h) < prefix_top:
            heappush(h, (c, w2))
        else:
            # Keep if (c, w2) is larger than smallest
            if c > h[0][0] or (c == h[0][0] and w2 > h[0][1]):
                heappushpop(h, (c, w2))

    pruned: Dict[Tuple[str, str], int] = {}
    for w1, h in heaps.items():
        for c, w2 in h:
            pruned[(w1, w2)] = c
    return pruned


def keep_global_top_k(
    bigram_counts: Dict[Tuple[str, str], int],
    top_k: int,
) -> Dict[Tuple[str, str], int]:
    """
    Keep only the global top-K bigrams by count, using a single min-heap.
    """
    if top_k <= 0 or len(bigram_counts) <= top_k:
        return dict(bigram_counts)

    heap: List[Tuple[int, Tuple[str, str]]] = []
    for bg, c in bigram_counts.items():
        if len(heap) < top_k:
            heappush(heap, (c, bg))
        else:
            if c > heap[0][0]:
                heappushpop(heap, (c, bg))

    return {bg: c for c, bg in heap}


# ---------------------------
# Build steps
# ---------------------------

def build_vocab(
    input_path: Path,
    vocab_size: int,
    *,
    lowercase: bool,
    encoding: str,
    min_count: int,
    max_tokens_per_line: int,
    progress_every: int,
) -> Set[str]:
    """
    Count unigrams, then keep top `vocab_size` tokens (optionally excluding those < min_count).
    """
    counts: Counter[str] = Counter()
    for i, line in enumerate(iter_lines(input_path, encoding=encoding), start=1):
        toks = tokenize(line, lowercase=lowercase)
        if max_tokens_per_line > 0:
            toks = toks[:max_tokens_per_line]
        counts.update(toks)

        if progress_every > 0 and i % progress_every == 0:
            print(f"[pass1] lines={i:,} | unique_tokens={len(counts):,}", file=sys.stderr)

    # apply unigram min_count filter first (helps shrink vocab)
    if min_count > 1:
        counts = Counter({w: c for w, c in counts.items() if c >= min_count})

    if vocab_size > 0 and len(counts) > vocab_size:
        vocab = {w for w, _c in counts.most_common(vocab_size)}
    else:
        vocab = set(counts.keys())

    print(f"[pass1] vocab_size={len(vocab):,}", file=sys.stderr)
    return vocab


def count_bigrams(
    input_path: Path,
    vocab: Set[str],
    *,
    lowercase: bool,
    encoding: str,
    max_tokens_per_line: int,
    progress_every: int,
) -> Dict[Tuple[str, str], int]:
    """
    Count bigrams (w1,w2) only if both in vocab.
    """
    counts: Counter[Tuple[str, str]] = Counter()
    for i, line in enumerate(iter_lines(input_path, encoding=encoding), start=1):
        toks = tokenize(line, lowercase=lowercase)
        if max_tokens_per_line > 0:
            toks = toks[:max_tokens_per_line]

        # vocab filter on the fly
        toks = [t for t in toks if t in vocab]
        if len(toks) >= 2:
            for a, b in zip(toks, toks[1:]):
                counts[(a, b)] += 1

        if progress_every > 0 and i % progress_every == 0:
            print(f"[pass2] lines={i:,} | unique_bigrams={len(counts):,}", file=sys.stderr)

    print(f"[pass2] unique_bigrams={len(counts):,}", file=sys.stderr)
    return dict(counts)


def prune_bigrams(
    bigram_counts: Dict[Tuple[str, str], int],
    *,
    min_count: int,
    prefix_top: int,
    top_k: int,
) -> Dict[Tuple[str, str], int]:
    """
    Prune in this order:
      1) drop counts < min_count
      2) keep top-M per prefix (w1 -> top continuations)
      3) keep global top-K (hard cap)
    """
    if min_count > 1:
        bigram_counts = {bg: c for bg, c in bigram_counts.items() if c >= min_count}

    if prefix_top > 0:
        bigram_counts = keep_top_per_prefix(bigram_counts, prefix_top)

    if top_k > 0:
        bigram_counts = keep_global_top_k(bigram_counts, top_k)

    return bigram_counts


def write_json_bigram_dict(
    bigram_counts: Dict[Tuple[str, str], int],
    output_path: Path,
) -> None:
    """
    Write compact JSON dict: "w1 w2" -> count.
    If output_path endswith .gz, write gzipped.
    """
    # Convert keys to "w1 w2"
    out_dict = {f"{w1} {w2}": int(c) for (w1, w2), c in bigram_counts.items()}

    # Compact + deterministic output (sorted keys)
    # (Sorting costs some time but makes builds reproducible.)
    items = sorted(out_dict.items(), key=lambda kv: kv[0])

    # Stream JSON writing to avoid a second giant in-memory string
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            f.write("{")
            for idx, (k, v) in enumerate(items):
                if idx:
                    f.write(",")
                f.write(json.dumps(k, ensure_ascii=False))
                f.write(":")
                f.write(str(v))
            f.write("}")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            f.write("{")
            for idx, (k, v) in enumerate(items):
                if idx:
                    f.write(",")
                f.write(json.dumps(k, ensure_ascii=False))
                f.write(":")
                f.write(str(v))
            f.write("}")

    print(f"[write] wrote {len(items):,} bigrams -> {output_path}", file=sys.stderr)


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build truncated bigram JSON from 1-article-per-line .txt")

    p.add_argument("--input", required=True, type=Path, help="Input .txt (1 article per line)")
    p.add_argument("--output", required=True, type=Path, help="Output .json or .json.gz")

    # Controls
    p.add_argument("--vocab-size", type=int, default=50_000,
                   help="Keep top-N unigrams as vocabulary (0 = keep all)")
    p.add_argument("--unigram-min-count", type=int, default=2,
                   help="Drop unigrams below this count before taking top vocab-size")

    p.add_argument("--min-count", type=int, default=2,
                   help="Drop bigrams below this count before truncation")
    p.add_argument("--prefix-top", type=int, default=40,
                   help="Keep top-M bigrams per previous word (0 = disabled)")
    p.add_argument("--top-k", type=int, default=250_000,
                   help="Hard cap: keep only global top-K after prefix pruning (0 = disabled)")

    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing")
    p.add_argument("--encoding", type=str, default="utf-8", help="Input file encoding")
    p.add_argument("--max-tokens-per-line", type=int, default=0,
                   help="If >0, only process first N tokens of each line (speed/size control)")

    p.add_argument("--progress-every", type=int, default=50_000,
                   help="Print progress every N lines (0 = silent)")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    lowercase = not args.no_lowercase

    # Pass 1: vocab
    vocab = build_vocab(
        args.input,
        args.vocab_size,
        lowercase=lowercase,
        encoding=args.encoding,
        min_count=args.unigram_min_count,
        max_tokens_per_line=args.max_tokens_per_line,
        progress_every=args.progress_every,
    )

    # Pass 2: bigrams within vocab
    bigrams = count_bigrams(
        args.input,
        vocab,
        lowercase=lowercase,
        encoding=args.encoding,
        max_tokens_per_line=args.max_tokens_per_line,
        progress_every=args.progress_every,
    )

    # Prune / truncate
    pruned = prune_bigrams(
        bigrams,
        min_count=args.min_count,
        prefix_top=args.prefix_top,
        top_k=args.top_k,
    )

    print(
        f"[prune] kept={len(pruned):,} "
        f"(min_count>={args.min_count}, prefix_top={args.prefix_top}, top_k={args.top_k})",
        file=sys.stderr,
    )

    write_json_bigram_dict(pruned, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
