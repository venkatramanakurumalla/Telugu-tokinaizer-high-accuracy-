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
- Knowledge Graph ready output formats

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

# Export for Knowledge Graph (JSON format)
python tokenizer.py tokenize --input data.txt --format json --out tokens.json
"""

import sys
import os
import time
import json
import unicodedata
import argparse
import logging
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Iterable, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field

try:
    import regex as re
except ImportError as e:
    raise SystemExit(
        "This toolkit requires the 'regex' package.\n"
        "Install it with: pip install regex"
    ) from e

# ========================
# LOGGING SETUP
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================
# CORE CONFIGURATION
# ========================

# Telugu Unicode range
_TELUGU_RANGE = r"\p{Telugu}"
_IS_TELUGU = re.compile(_TELUGU_RANGE).search

# Configuration constants
CACHE_SIZE_MORPH = 65536
CACHE_SIZE_TOKEN = 131072
FALLBACK_LENGTH_THRESHOLD = 6
MAX_FILE_SIZE_MB = 500

# Suffix Trie for morphological splitting (using regular sets for O(1) lookup)
_SUFFIX_TRIE = {
    4: {
        "స్తున్న", "స్తారు", "తున్న", "తారు", "దున్న", "దాము",
        "ద్దాం", "ందరు", "వరకు", "కొరకు", "దగ్గర"
    },
    3: {
        "వల్ల", "యొక్క", "ంటే", "య్యే", "య్యి", "స్తు",
        "స్తూ", "మ్మ", "వ్వ", "బడి", "లాగా", "వంటి"
    },
    2: {
        "కు", "కి", "ను", "లో", "గా", "తో", "లు", "రు",
        "ళు", "ంత", "క్కి", "ంట", "పై", "చే", "కింద"
    },
    1: {
        "ల", "క", "గ", "చ", "త", "ద", "న", "ప", "మ", "య", "ర"
    }
}

# Vowel harmony patterns with named tuple for clarity
class VowelPattern(NamedTuple):
    """Represents a vowel harmony transformation rule."""
    stem_pattern: re.Pattern
    suffix_start: str
    stem_replacement: str
    suffix_replacement: str

_VOWEL_PATTERNS = [
    VowelPattern(re.compile(r"ా$"), "కి", "ం", "కు"),
    VowelPattern(re.compile(r"ు$"), "కి", "", "కు"),
    VowelPattern(re.compile(r"ి$"), "కి", "య", "కి"),
    VowelPattern(re.compile(r"ీ$"), "కి", "య", "కి"),
    VowelPattern(re.compile(r"ూ$"), "కి", "వ", "కి"),
]

# Token regex: Telugu runs | numbers | punctuation | other single
_TOKEN_RE = re.compile(rf"([{_TELUGU_RANGE}]+|\d+|\p{{P}}|\S)", re.UNICODE)

# Syllable fallback: consonant/vowel clusters (FIXED: removed typo)
_SYLLABLE_RE = re.compile(rf"(?:[{_TELUGU_RANGE}][\u0C3E-\u0C56]?)+", re.UNICODE)

# ========================
# DATA CLASSES
# ========================

@dataclass
class TokenizationResult:
    """Result of tokenization with metadata."""
    tokens: List[str]
    original_text: str
    num_tokens: int = field(init=False)
    num_telugu_tokens: int = field(init=False)
    fallback_used: bool = False
    
    def __post_init__(self):
        self.num_tokens = len(self.tokens)
        self.num_telugu_tokens = sum(1 for t in self.tokens if _IS_TELUGU(t))
    
    def to_dict(self) -> Dict:
        """Export as dictionary for JSON serialization."""
        return {
            "tokens": self.tokens,
            "original_text": self.original_text,
            "num_tokens": self.num_tokens,
            "num_telugu_tokens": self.num_telugu_tokens,
            "fallback_used": self.fallback_used
        }

# ========================
# MODULE-LEVEL CACHE
# ========================

@lru_cache(maxsize=CACHE_SIZE_MORPH)
def _cached_split_morph(
    word: str,
    exceptions_tuple: Tuple[Tuple[str, Tuple[str, ...]], ...],
    enable_fallback: bool
) -> Tuple[str, ...]:
    """
    Cached morphological split function.
    
    Uses tuple-based parameters for hashability.
    Separated from instance to avoid self in cache key.
    """
    # Convert tuple back to dict for lookup
    exceptions = dict((k, list(v)) for k, v in exceptions_tuple) if exceptions_tuple else {}
    
    # Exceptions first
    if word in exceptions:
        return tuple(exceptions[word])
    
    # Trie suffix check
    for length in (4, 3, 2, 1):
        if len(word) <= length:
            continue
        suffix = word[-length:]
        if suffix in _SUFFIX_TRIE[length]:
            stem = word[:-length]
            
            # Vowel harmony adjustment
            for pattern in _VOWEL_PATTERNS:
                if pattern.stem_pattern.search(stem) and suffix.startswith(pattern.suffix_start):
                    new_stem = pattern.stem_pattern.sub(pattern.stem_replacement, stem)
                    new_suffix = pattern.suffix_replacement + suffix[len(pattern.suffix_start):]
                    return (new_stem, new_suffix)
            
            return (stem, suffix)
    
    # Fallback: syllable-level split for unknown long words
    if enable_fallback and len(word) > FALLBACK_LENGTH_THRESHOLD:
        pieces = _SYLLABLE_RE.findall(word)
        if pieces:
            return tuple(pieces)
    
    return (word,)

# =====================
# TOKENIZER CLASS
# =====================

class TeluguTokenizer:
    """
    Production-ready Telugu morphological tokenizer.
    
    Performs Unicode normalization, morphological analysis with suffix
    splitting, vowel harmony, and optional syllable-level fallback.
    
    Args:
        exception_dict: Optional dictionary mapping words to their morphological splits
        enable_fallback: Whether to use syllable-based splitting for unknown long words
        debug: Enable debug logging
    """
    
    def __init__(
        self,
        exception_dict: Optional[Dict[str, List[str]]] = None,
        enable_fallback: bool = True,
        debug: bool = False
    ):
        # Default exceptions
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
        self.debug = debug
        
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Convert exceptions to tuple for caching
        self._exceptions_tuple = tuple(
            (k, tuple(v)) for k, v in sorted(self.exceptions.items())
        )
        
        logger.info(f"Initialized tokenizer with {len(self.exceptions)} exceptions")
    
    # -------------------------
    # Normalization
    # -------------------------
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize Unicode text.
        
        - Applies NFKC normalization
        - Removes zero-width characters
        - Normalizes whitespace
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Fast path for ASCII
        if text.isascii():
            return " ".join(text.split())
        
        # NFKC normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Remove zero-width characters
        zero_width_chars = {
            ord(c): None for c in "\u200B\u200C\u200D\uFEFF"
        }
        text = text.translate(zero_width_chars)
        
        # Normalize whitespace
        return " ".join(text.split())
    
    # -------------------------
    # Morphological Split
    # -------------------------
    def split_morph(self, word: str) -> List[str]:
        """
        Split a Telugu word into morphemes.
        
        Args:
            word: Telugu word to split
            
        Returns:
            List of morphemes
        """
        result = _cached_split_morph(
            word,
            self._exceptions_tuple,
            self.enable_fallback
        )
        
        if self.debug and len(result) > 1:
            logger.debug(f"Split '{word}' -> {list(result)}")
        
        return list(result)
    
    # -------------------------
    # Public API
    # -------------------------
    def tokenize(self, text: str, return_metadata: bool = False):
        """
        Tokenize Telugu text into morphological units.
        
        Args:
            text: Input text to tokenize
            return_metadata: If True, return TokenizationResult with metadata
            
        Returns:
            List of tokens or TokenizationResult object
        """
        original_text = text
        text = self.normalize(text)
        
        if not text:
            if return_metadata:
                return TokenizationResult([], original_text)
            return []
        
        out: List[str] = []
        fallback_used = False
        
        for m in _TOKEN_RE.finditer(text):
            unit = m.group()
            
            # Non-Telugu tokens pass through
            if not _IS_TELUGU(unit):
                out.append(unit)
                continue
            
            # Morphological split for Telugu
            morphs = self.split_morph(unit)
            out.extend(morphs)
            
            if len(morphs) > 1 and len(unit) > FALLBACK_LENGTH_THRESHOLD:
                fallback_used = True
        
        if return_metadata:
            result = TokenizationResult(out, original_text)
            result.fallback_used = fallback_used
            return result
        
        return out
    
    def tokenize_and_join(self, text: str) -> str:
        """
        Tokenize and join with spaces (for training pipelines).
        
        Args:
            text: Input text
            
        Returns:
            Space-separated tokens
        """
        return " ".join(self.tokenize(text))
    
    # -------------------------
    # Benchmark
    # -------------------------
    def benchmark_string(
        self,
        text: str,
        repeat: int = 5,
        warmup: int = 2
    ) -> Tuple[int, float, float, float]:
        """
        Benchmark tokenization performance.
        
        Args:
            text: Text to tokenize
            repeat: Number of benchmark iterations
            warmup: Number of warmup iterations
            
        Returns:
            Tuple of (token_count, avg_time, tokens_per_sec, std_dev)
        """
        # Warmup
        for _ in range(max(0, warmup)):
            _ = self.tokenize(text)
        
        timings = []
        tok_count = 0
        
        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            toks = self.tokenize(text)
            dt = time.perf_counter() - t0
            timings.append(dt)
            tok_count = len(toks)
        
        avg = sum(timings) / len(timings)
        
        # Calculate standard deviation
        variance = sum((t - avg) ** 2 for t in timings) / len(timings)
        std_dev = variance ** 0.5
        
        tps = tok_count / avg if avg > 0 else float("inf")
        
        return tok_count, avg, tps, std_dev


# =====================
# STREAMING HELPERS
# =====================

def iter_files(root: str, ext: str = ".txt") -> Iterable[str]:
    """
    Iterate over files with given extension.
    
    Args:
        root: Root directory or single file path
        ext: File extension filter
        
    Yields:
        File paths
    """
    ext = ext.lower()
    
    if os.path.isfile(root):
        yield root
        return
    
    if not os.path.isdir(root):
        logger.warning(f"Path does not exist: {root}")
        return
    
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                yield os.path.join(base, f)


def read_text(path: str, encoding: str = "utf-8") -> str:
    """
    Read text file with error handling.
    
    Args:
        path: File path
        encoding: Character encoding
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check file size
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"Large file detected: {size_mb:.1f} MB")
    
    try:
        with open(path, "r", encoding=encoding, errors="replace") as fh:
            return fh.read()
    except Exception as e:
        raise IOError(f"Cannot read file {path}: {e}") from e


def stream_lines(path: str, encoding: str = "utf-8") -> Iterable[str]:
    """
    Stream lines from a file.
    
    Args:
        path: File path
        encoding: Character encoding
        
    Yields:
        Lines without trailing newline
    """
    try:
        with open(path, "r", encoding=encoding, errors="replace") as fh:
            for line in fh:
                yield line.rstrip("\n")
    except Exception as e:
        logger.error(f"Error streaming file {path}: {e}")
        raise


# =====================
# CLI IMPLEMENTATION
# =====================

def load_exceptions(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """
    Load exception dictionary from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary mapping words to morpheme lists
    """
    if not path:
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logger.error(f"Failed to load exceptions from {path}: {e}")
        return None
    
    # Validate structure
    cleaned = {}
    skipped = 0
    
    for k, v in data.items():
        if not isinstance(k, str):
            skipped += 1
            continue
        if not isinstance(v, list):
            skipped += 1
            continue
        if not all(isinstance(x, str) and x for x in v):
            skipped += 1
            continue
        cleaned[k] = v
    
    if skipped:
        logger.warning(f"Skipped {skipped} invalid exception entries")
    
    logger.info(f"Loaded {len(cleaned)} exceptions from {path}")
    return cleaned


def cmd_tokenize(args: argparse.Namespace) -> None:
    """Execute tokenize/join command."""
    ex = load_exceptions(args.exceptions)
    tok = TeluguTokenizer(
        exception_dict=ex,
        enable_fallback=not args.no_fallback,
        debug=args.debug
    )
    
    stats: Optional[Counter] = Counter() if args.stats else None
    results = []  # For JSON export
    
    # Process input
    if not args.input:
        text = sys.stdin.read()
        tokens = tok.tokenize(text)
        if stats is not None:
            stats.update(tokens)
        
        output = format_output(tokens, args.format, text)
        write_output(output, args.out)
    else:
        # File mode with streaming
        try:
            for line_no, line in enumerate(stream_lines(args.input, args.encoding), 1):
                if not line.strip():
                    continue
                
                if args.format == "json":
                    result = tok.tokenize(line, return_metadata=True)
                    results.append(result.to_dict())
                    if stats is not None:
                        stats.update(result.tokens)
                else:
                    tks = tok.tokenize(line)
                    if stats is not None:
                        stats.update(tks)
                    
                    if args.mode == "join":
                        output = " ".join(tks)
                    else:
                        output = "\n".join(tks)
                    
                    write_output(output, args.out, append=(line_no > 1))
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            sys.exit(1)
    
    # JSON export
    if args.format == "json" and results:
        output = json.dumps(results, ensure_ascii=False, indent=2)
        write_output(output, args.out)
    
    # Save stats if requested
    if stats and args.stats:
        try:
            with open(args.stats, "w", encoding="utf-8") as sfh:
                sfh.write("token\tfrequency\n")
                for token, freq in stats.most_common():
                    sfh.write(f"{token}\t{freq}\n")
            logger.info(f"Saved token statistics to {args.stats}")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")


def format_output(tokens: List[str], fmt: str, original: str = "") -> str:
    """Format tokens based on output format."""
    if fmt == "json":
        return json.dumps({
            "tokens": tokens,
            "original": original,
            "count": len(tokens)
        }, ensure_ascii=False, indent=2)
    elif fmt == "lines":
        return "\n".join(tokens)
    else:  # space-separated
        return " ".join(tokens)


def write_output(content: str, path: Optional[str], append: bool = False):
    """Write output to file or stdout."""
    if path:
        mode = "a" if append else "w"
        try:
            with open(path, mode, encoding="utf-8") as fh:
                fh.write(content)
                if not content.endswith("\n"):
                    fh.write("\n")
        except Exception as e:
            logger.error(f"Failed to write to {path}: {e}")
            sys.exit(1)
    else:
        print(content)


def cmd_bench_file(args: argparse.Namespace) -> None:
    """Execute single-file benchmark."""
    ex = load_exceptions(args.exceptions)
    tok = TeluguTokenizer(exception_dict=ex, enable_fallback=not args.no_fallback)
    
    try:
        text = read_text(args.input, args.encoding)
        n, avg, tps, std = tok.benchmark_string(text, repeat=args.repeat, warmup=args.warmup)
        
        print(f"\n{'='*60}")
        print(f"File: {args.input}")
        print(f"{'='*60}")
        print(f"Tokens processed: {n:,}")
        print(f"Average time per run: {avg:.4f} sec")
        print(f"Standard deviation: {std:.4f} sec")
        print(f"Throughput: {tps:,.0f} tokens/sec")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


def cmd_bench_corpus(args: argparse.Namespace) -> None:
    """Execute corpus-wide benchmark."""
    ex = load_exceptions(args.exceptions)
    tok = TeluguTokenizer(exception_dict=ex, enable_fallback=not args.no_fallback)
    
    totals = []
    print(f"\nScanning: {args.dir} (ext={args.ext})")
    print(f"{'='*60}\n")
    
    for path in iter_files(args.dir, args.ext):
        try:
            text = read_text(path, args.encoding)
            n, avg, tps, std = tok.benchmark_string(text, repeat=args.repeat, warmup=args.warmup)
            totals.append((n, avg, std))
            print(f"[{os.path.basename(path)}]")
            print(f"  Tokens: {n:,}")
            print(f"  Avg: {avg:.4f}s ± {std:.4f}s")
            print(f"  Throughput: {tps:,.0f} tok/s\n")
        except Exception as e:
            logger.warning(f"Skipped {path}: {e}")
    
    if totals:
        sum_tokens = sum(n for n, _, _ in totals)
        total_time = sum(avg for _, avg, _ in totals)
        overall_avg = total_time / len(totals)
        overall_tps = (sum_tokens / overall_avg) if overall_avg > 0 else float("inf")
        
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"Files processed: {len(totals)}")
        print(f"Total tokens: {sum_tokens:,}")
        print(f"Average time/run: {overall_avg:.4f} sec")
        print(f"Overall throughput: {overall_tps:,.0f} tokens/sec")
        print(f"{'='*60}\n")
    else:
        print("No files found.")


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Telugu Morphological Tokenizer Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # tokenize/join commands
    for mode in ("tokenize", "join"):
        pp = sub.add_parser(mode, help=f"{mode.capitalize()} text")
        pp.add_argument("--input", "-i", help="Input file (default: stdin)")
        pp.add_argument("--out", "-o", help="Output file (default: stdout)")
        pp.add_argument("--encoding", default="utf-8", help="File encoding")
        pp.add_argument("--exceptions", help="Path to JSON exception dictionary")
        pp.add_argument("--no-fallback", action="store_true", help="Disable syllable fallback")
        pp.add_argument("--stats", help="Write token frequency TSV")
        pp.add_argument("--format", choices=["space", "lines", "json"], default="space",
                       help="Output format (for KG: use json)")
        pp.add_argument("--debug", action="store_true", help="Enable debug mode")
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
    b2 = sub.add_parser("bench-corpus", help="Benchmark corpus folder")
    b2.add_argument("--dir", "-d", required=True, help="Folder to scan")
    b2.add_argument("--ext", default=".txt", help="File extension filter")
    b2.add_argument("--encoding", default="utf-8")
    b2.add_argument("--exceptions")
    b2.add_argument("--no-fallback", action="store_true")
    b2.add_argument("--repeat", type=int, default=5)
    b2.add_argument("--warmup", type=int, default=2)
    b2.set_defaults(func=cmd_bench_corpus)
    
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
