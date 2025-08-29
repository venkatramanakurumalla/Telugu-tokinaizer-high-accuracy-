# Telugu Morphological Tokenizer Toolkit 🔥

A **production-ready**, **morphology-aware** tokenizer for **Telugu** (తెలుగు), designed for NLP pipelines, BPE pre-training, and linguistic analysis.

✨ Features:
- Morphological suffix splitting using a **Trie + vowel harmony**
- Unicode normalization & zero-width cleanup
- Pluggable exception dictionary (JSON)
- Syllable-level fallback (toggleable)
- Streaming file processing for large corpora
- Benchmarking CLI for performance testing
- Designed for integration with Hugging Face, spaCy, BPE, etc.

## 🚀 Usage

```bash
# Tokenize a file
python tokenizer.py tokenize --input data.txt --out tokens.txt

# Join tokens (e.g., for BPE training)
python tokenizer.py join --input data.txt --out joined.txt

# Benchmark performance
python tokenizer.py bench-file --input sample.txt --repeat 5
