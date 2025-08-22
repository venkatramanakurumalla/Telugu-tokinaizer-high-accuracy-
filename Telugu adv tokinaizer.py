# -*- coding: utf-8 -*-
"""
🔥 Production-Ready Telugu Morphological Tokenizer Toolkit
==========================================================
Features:
- Unicode normalization + zero-width cleanup
- Morphological suffix splitting (Trie + vowel harmony)
- Pluggable exception dictionary (JSON)
- Cached ops for speed, syllable fallback (toggleable)
- Streaming file tokenizer for large corpora
- Benchmark CLI for single file or corpus (folder)

CLI Examples:
-------------
# Tokenize a file -> stdout
python tokenizer.py tokenize --input data.txt

# Tokenize and write to file
python tokenizer.py tokenize --input data.txt --out tokens.txt

# Join (space-separated) for BPE/LM training
python tokenizer.py join --input data.txt --out joined.txt

# Benchmark a single file
python tokenizer.py bench-file --input data.txt --repeat 5 --warmup 2

# Benchmark a whole folder (recurses *.txt by default)
python tokenizer.py bench-corpus --dir corpus --ext .txt --repeat 5 --warmup 2

# Use JSON exceptions and disable syllable fallback
python tokenizer.py tokenize --input data.txt --exceptions ex.json --no-fallback

# Also emit token frequency stats
python tokenizer.py tokenize --input data.txt --stats stats.tsv

Author: GPT-5
"""

import sys
import os
import time
import json
import unicodedata
import argparse
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Iterable, Tuple, Optional

try:
    import regex as re  # robust Unicode
except Exception as e:
    raise SystemExit(
        "This toolkit requires the 'regex' package.\n"
        "Install it with: pip install regex"
    ) from e

# ========================
# CORE CONFIGURATION
# ========================

_TELUGU_RANGE = r"\p{IsTelugu}"
_IS_TELUGU = re.compile(_TELUGU_RANGE).search

# Suffix Trie for morphological splitting
_SUFFIX_TRIE = {
    4: frozenset({
        "స్తున్న", "స్తారు", "తున్న", "తారు", "దున్న", "దాము",
        "ద్దాం", "ందరు", "వరకు", "కొరకు", "దగ్గర"
    }),
    3: frozenset({
        "వల్ల", "యొక్క", "ంటే", "య్యే", "య్యి", "స్తు",
        "స్తూ", "మ్మ", "వ్వ", "బడి", "లాగా", "వంటి"
    }),
    2: frozenset({
        "కు", "కి", "ను", "లో", "గా", "తో", "లు", "రు",
        "ళు", "ంత", "క్కి", "ంట", "పై", "చే", "కింద"
    }),
    1: frozenset({"ల", "క", "గ", "చ", "త", "ద", "న", "ప", "మ", "య", "ర"})
}

# Sandhi / vowel harmony patterns
_VOWEL_PATTERNS = [
    (re.compile(r"ా$"), "కి", "ం", "కు"),
    (re.compile(r"ు$"), "కి", "", "కు"),
    (re.compile(r"ి$"), "కి", "య", "కి"),
    (re.compile(r"ీ$"), "కి", "య", "కి"),
    (re.compile(r"ూ$"), "కి", "వ", "కి"),
]

# Token regex: Telugu runs | numbers | punctuation | other single
_TOKEN_RE = re.compile(rf"([{_TELUGU_RANGE}]+|\d+|\p{{P}}|\S)", re.UNICODE)

# Syllable fallback: consonant/vowel clusters
_SYLLABLE_RE = re.compile(rf"(?:[{_TELుగU_RANGE}][\u0C3E-\u0C56]?)+", re.UNICODE)

# =====================
# TOKENIZER
# =====================

class TeluguTokenizer:
    def __init__(
        self,
        exception_dict: Optional[Dict[str, List[str]]] = None,
        enable_fallback: bool = True,
    ):
        self.exceptions = {
            "పుస్తకానికి": ["పుస్తకం", "కు"],
            "పుస్తకంలో": ["పుస్తకం", "లో"],
            "పిల్లలకు": ["పిల్లలు", "కు"],
            "వాళ్ళతో": ["వాళ్ళు", "తో"],
            "గ్రంథాలయానికి": ["గ్రంథాలయం", "కు"],
            "రాష్ట్రంలో": ["రాష్ట్రం", "లో"],
        }
        if exception_dict:
            self.exceptions.update(exception_dict)
        self.enable_fallback = enable_fallback

    # -------------------------
    # Normalization
    # -------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        if text.isascii():
            return " ".join(text.split())
        text = unicodedata.normalize("NFKC", text)
        # Remove zero-width chars
        text = text.translate(dict.fromkeys(map(ord, "\u200B\u200C\u200D\uFEFF")))
        return " ".join(text.split())

    # -------------------------
    # Morphological Split
    # -------------------------
    @lru_cache(maxsize=65536)
    def _split_morph(self, word: str) -> List[str]:
        # Exceptions first
        if word in self.exceptions:
            return self.exceptions[word]

        # Trie suffix check
        for length in (4, 3, 2, 1):
            if len(word) <= length:
                continue
            suffix = word[-length:]
            if suffix in _SUFFIX_TRIE[length]:
                stem = word[:-length]
                # Vowel harmony adjustment
                for pattern, suf_start, stem_rep, suf_rep in _VOWEL_PATTERNS:
                    if pattern.search(stem) and suffix.startswith(suf_start):
                        return [pattern.sub(stem_rep, stem),
                                suf_rep + suffix[len(suf_start):]]
                return [stem, suffix]

        # Fallback: syllable-level split for unknown long words
        if self.enable_fallback and len(word) > 6:
            pieces = _SYLLABLE_RE.findall(word)
            if pieces:
                return pieces

        return [word]

    # -------------------------
    # Public API
    # -------------------------
    @lru_cache(maxsize=131072)
    def tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        if not text:
            return []
        out: List[str] = []
        for m in _TOKEN_RE.finditer(text):
            unit = m.group()
            if not _IS_TELUGU(unit):
                out.append(unit)
                continue
            out.extend(self._split_morph(unit))
        return out

    def tokenize_and_join(self, text: str) -> str:
        return " ".join(self.tokenize(text))

    # -------------------------
    # Benchmark
    # -------------------------
    def benchmark_string(self, text: str, repeat: int = 5, warmup: int = 2) -> Tuple[int, float, float]:
        """Return (tokens, avg_sec, tokens_per_sec)."""
        # Warmup
        for _ in range(max(0, warmup)):
            _ = self.tokenize(text)

        timings = []
        tok_count = 0
        for _ in range(max(1, repeat)):
            t0 = time.time()
            toks = self.tokenize(text)
            dt = time.time() - t0
            timings.append(dt)
            tok_count = len(toks)
        avg = sum(timings) / len(timings)
        tps = tok_count / avg if avg > 0 else float("inf")
        return tok_count, avg, tps


# =====================
# STREAMING HELPERS
# =====================

def iter_files(root: str, ext: str = ".txt") -> Iterable[str]:
    ext = ext.lower()
    if os.path.isfile(root):
        yield root
        return
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                yield os.path.join(base, f)

def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="ignore") as fh:
        return fh.read()

def stream_lines(path: str, encoding: str = "utf-8") -> Iterable[str]:
    with open(path, "r", encoding=encoding, errors="ignore") as fh:
        for line in fh:
            yield line.rstrip("\n")


# =====================
# CLI IMPLEMENTATION
# =====================

def load_exceptions(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Expecting { "word": ["morph1","morph2", ...], ... }
    # Validate lightly:
    cleaned = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
            cleaned[k] = v
    return cleaned

def cmd_tokenize(args: argparse.Namespace) -> None:
    ex = load_exceptions(args.exceptions)
    tok = TeluguTokenizer(exception_dict=ex, enable_fallback=not args.no_fallback)
    stats: Optional[Counter] = Counter() if args.stats else None

    # Streaming process for large files
    if not args.input:
        text = sys.stdin.read()
        tokens = tok.tokenize(text)
        if stats is not None:
            stats.update(tokens)
        out = " ".join(tokens) if args.mode == "join" else "\n".join(tokens)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(out)
        else:
            print(out)
    else:
        # file mode
        if args.out:
            outfh = open(args.out, "w", encoding="utf-8")
        else:
            outfh = None
        try:
            if args.mode == "join":
                # join per line for stability
                for line in stream_lines(args.input, args.encoding):
                    joined = tok.tokenize_and_join(line)
                    if stats is not None:
                        stats.update(joined.split())
                    (outfh.write(joined + "\n") if outfh else print(joined))
            else:
                # emit one token per line
                for line in stream_lines(args.input, args.encoding):
                    tks = tok.tokenize(line)
                    if stats is not None:
                        stats.update(tks)
                    if outfh:
                        for t in tks:
                            outfh.write(t + "\n")
                    else:
                        for t in tks:
                            print(t)
        finally:
            if outfh:
                outfh.close()

    # Save stats if requested
    if stats is not None:
        with open(args.stats, "w", encoding="utf-8") as sfh:
            for token, freq in stats.most_common():
                sfh.write(f"{token}\t{freq}\n")

def cmd_bench_file(args: argparse.Namespace) -> None:
    ex = load_exceptions(args.exceptions)
    tok = TeluguTokenizer(exception_dict=ex, enable_fallback=not args.no_fallback)
    text = read_text(args.input, args.encoding)
    n, avg, tps = tok.benchmark_string(text, repeat=args.repeat, warmup=args.warmup)
    print(f"File: {args.input}")
    print(f"Tokens processed: {n}")
    print(f"Average time per run: {avg:.4f} sec")
    print(f"Throughput: {tps:,.0f} tokens/sec")

def cmd_bench_corpus(args: argparse.Namespace) -> None:
    ex = load_exceptions(args.exceptions)
    tok = TeluguTokenizer(exception_dict=ex, enable_fallback=not args.no_fallback)

    totals = []
    print(f"Scanning: {args.dir} (ext={args.ext})")
    for path in iter_files(args.dir, args.ext):
        text = read_text(path, args.encoding)
        n, avg, tps = tok.benchmark_string(text, repeat=args.repeat, warmup=args.warmup)
        totals.append((n, avg))
        print(f"[{path}] tokens={n} avg={avg:.4f}s tput={tps:,.0f}/s")

    if totals:
        sum_tokens = sum(n for n, _ in totals)
        # Weighted average by tokens
        total_time = sum(avg for _, avg in totals)
        overall_avg = total_time / len(totals)
        overall_tps = (sum_tokens / overall_avg) if overall_avg > 0 else float("inf")
        print("\n== Aggregate ==")
        print(f"Files: {len(totals)}")
        print(f"Total tokens: {sum_tokens:,}")
        print(f"Average time/run (mean of files): {overall_avg:.4f} sec")
        print(f"Throughput (approx): {overall_tps:,.0f} tokens/sec")
    else:
        print("No files found.")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Telugu Morphological Tokenizer Toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # tokenize/join share options
    for mode in ("tokenize", "join"):
        pp = sub.add_parser(mode, help=f"{mode} a file or stdin")
        pp.add_argument("--input", "-i", help="Input file (default: stdin)")
        pp.add_argument("--out", "-o", help="Output file (default: stdout)")
        pp.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8)")
        pp.add_argument("--exceptions", help="Path to JSON exception dictionary")
        pp.add_argument("--no-fallback", action="store_true", help="Disable syllable fallback")
        pp.add_argument("--stats", help="Write token frequency TSV to this path")
        pp.set_defaults(func=cmd_tokenize, mode=mode)

    # bench-file
    b1 = sub.add_parser("bench-file", help="Benchmark a single file")
    b1.add_argument("--input", "-i", required=True, help="Input file")
    b1.add_argument("--encoding", default="utf-8")
    b1.add_argument("--exceptions")
    b1.add_argument("--no-fallback", action="store_true")
    b1.add_argument("--repeat", type=int, default=5)
    b1.add_argument("--warmup", type=int, default=2)
    b1.set_defaults(func=cmd_bench_file)

    # bench-corpus
    b2 = sub.add_parser("bench-corpus", help="Benchmark all files in a folder")
    b2.add_argument("--dir", "-d", required=True, help="Folder to scan recursively")
    b2.add_argument("--ext", default=".txt", help="File extension filter (default .txt)")
    b2.add_argument("--encoding", default="utf-8")
    b2.add_argument("--exceptions")
    b2.add_argument("--no-fallback", action="store_true")
    b2.add_argument("--repeat", type=int, default=5)
    b2.add_argument("--warmup", type=int, default=2)
    b2.set_defaults(func=cmd_bench_corpus)

    return p

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
