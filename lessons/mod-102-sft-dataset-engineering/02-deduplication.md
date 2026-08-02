# Deduplication: Exact, Near, and Semantic

**Estimated read:** 35 minutes.
**You should be able to, by the end:** explain why duplicate SFT examples are worse than "wasted rows," implement or spec three flavors of deduplication (exact hash, MinHash + LSH near-dup, embedding-based semantic dedup) at the right granularity, and avoid the classic "each source deduped internally but not against the others" trap.

## Why duplicates hurt more in SFT than in pretraining

Pretraining research has known for years that duplicated documents inflate memorization and can slightly hurt held-out perplexity — **Lee et al. (2022)** ("Deduplicating Training Data Makes Language Models Better") is the canonical paper. The mitigation is usually "dedup once at ingestion and move on."

SFT is a much sharper case because:

1. **Dataset sizes are small.** Duplicates that look like a rounding error in a trillion-token corpus can be 5–20% of a 50k-example SFT set. That directly biases the model toward the duplicated behavior.
2. **Duplicates cluster in the "easy" region.** Popular prompts ("write a haiku about the ocean") get answered in every scraped ShareGPT dump. If those clusters go unfiltered, the model over-learns them and under-learns rarer but equally important behaviors.
3. **Duplicates leak eval prompts.** A near-duplicate of an MMLU question in your training set is not just a duplicate — it is contamination (chapter 3). Dedup is the first line of defense for decontamination.
4. **Duplicates poison downstream preference optimization.** DPO (mod-105) is a per-prompt comparison; if the same prompt appears many times in the SFT set, you have a very confident model on that prompt that DPO will struggle to update.

The goal is not "zero duplicates." It is a **documented, reproducible, tunable** dedup pass whose output you can defend.

## The three flavors of dedup

### 1. Exact deduplication

The straightforward one: normalize each row to a canonical form, hash it, keep one copy per hash.

```python
import hashlib

def normalize(text: str) -> str:
    # Choose a normalization once and document it.
    return " ".join(text.strip().lower().split())

def row_hash(row) -> str:
    canonical = normalize(row["prompt"]) + "\n<sep>\n" + normalize(row["response"])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

seen = {}
for r in rows:
    h = row_hash(r)
    if h not in seen:
        seen[h] = r
```

Choices to make explicitly:

- **Normalization.** Lowercase? Strip trailing whitespace? Collapse repeated whitespace? Normalize Unicode (`NFC`)? Every "yes" merges more rows but risks conflating semantically-different examples (e.g., case-sensitive code).
- **Granularity.** Hash on `(prompt, response)` together, or on `prompt` alone? Multi-turn: hash on the whole conversation, or on the last user message + assistant response?
- **Tie-breaking.** When two rows collide, which one wins? Highest quality score? Earliest ID? Random? Recording your rule matters for reproducibility.

Exact hashing catches near-perfect duplicates. It does not catch "same content with minor formatting differences," which is the majority of what actually appears in scraped SFT data.

### 2. Near-duplicate detection with MinHash + LSH

Near-dup techniques answer: "which pairs of rows have high Jaccard similarity of their shingled text?" without doing an O(n²) comparison.

**Shingling.** Break each document into overlapping n-grams (typically character 5-grams or word 3-grams). The Jaccard similarity of two documents is `|A ∩ B| / |A ∪ B|` over their shingle sets.

**MinHash.** For each document, hash every shingle with `k` independent hash functions and keep the minimum hash per function. Two documents' MinHash signatures agree on function `i` with probability equal to their Jaccard similarity. A `k`-length signature approximates Jaccard with error `~1/√k`.

**LSH (locality-sensitive hashing).** Bucket signatures so candidate pairs — those in at least one shared bucket — are almost surely similar. Non-candidate pairs are almost surely dissimilar. Tune with bands `b` and rows-per-band `r` (`k = b * r`) to hit a target similarity threshold; the classic curve is `1 - (1 - s^r)^b`.

Reference implementations you will actually use:

- **`datasketch`** — pure-Python MinHash + LSH; fine up to a few million rows.
- **`text-dedup`** (bigcode-project) — production-grade MinHash pipeline including Spark support; used to dedup The Stack and similar corpora. Contains the Jaccard implementation you should read before writing your own.

Recommended defaults from published pipelines (SlimPajama, The Stack, RedPajama):

- Word-3-gram or char-5-gram shingles.
- 128 or 256 permutations.
- Jaccard threshold of 0.7–0.8 as the "duplicate" cutoff.
- Dedup on `(prompt + response)` for SFT — you do not want to drop rows just because the prompt is common if the responses are meaningfully different.

Watch out for:

- **Very short responses.** "Yes." and "No." both have shingle sets of size 1; every pair looks like a duplicate. Impose a minimum length before applying MinHash.
- **Code.** Character-level shingling on code catches license headers and boilerplate as duplicates. Consider a separate dedup pass on code with a language-aware tokenizer.

### 3. Semantic (embedding-based) deduplication

Two examples can be Jaccard-dissimilar and still teach the same thing:

> "Write a Python function to check if a number is prime."
> "Give me a Python one-liner that tests primality of an integer."

MinHash will not catch this pair; embedding-based dedup will.

**The recipe.** Embed every prompt (or prompt + response) with a small sentence embedder (`sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, or comparable). Index the vectors with a nearest-neighbor library (FAISS, ScaNN, Milvus). For each row, query the index for its `k` nearest neighbors above a cosine-similarity threshold; keep one representative per cluster.

Choices:

- **Threshold.** 0.90 cosine on a modern embedder is aggressive; 0.95 is safer. Tune on a hand-labeled sample.
- **What to embed.** Prompt only will over-collapse "same question, wildly different answer" pairs. Prompt + response is usually what you want for SFT.
- **What to keep.** Highest quality score, longest response, most-recent generation? Record the tie-breaker.

Semantic dedup is *not* a strict superset of MinHash. It can miss text-level near-duplicates (same content, small wording changes) if the embedder happens to differentiate them. In practice, run exact → MinHash → semantic as three consecutive passes.

## The cross-source dedup trap

The single most common dedup mistake in SFT engineering is deduping each source in isolation. If you pull OpenAssistant, ShareGPT, and Alpaca separately, each may already have been deduped internally by its authors. That does *nothing* to catch a ShareGPT thread that reproduces an OpenAssistant conversation verbatim, or an Alpaca example whose prompt was copied into a Self-Instruct expansion.

The correct order is:

1. Load and lightly clean every source (chapter 1 filter stack up through structure/language/PII).
2. **Union** all rows into one pool.
3. Dedup the pool: exact → MinHash → semantic.
4. Then apply per-source labels for the mix stage (chapter 6).

When you record which row won a dedup conflict, record the *losing* source too. When mod-108 later asks "why is this eval regressing?", cross-source dedup collisions are one of the first things you will want to look at.

## Special-case dedup: eval-set overlap

Deduplication and decontamination are the same technique run against different reference sets. Chapter 3 handles this explicitly, but note that a MinHash LSH index built for internal dedup can be reused: add the eval prompts to the index and any candidate collisions with them are decontamination hits. Do not skip chapter 3 — it uses different thresholds and different n-gram sizes than internal dedup — but do notice that the machinery overlaps.

## An SFT dedup checklist

Before you ship a dedup pass, write down:

- The normalization function, exactly.
- The granularity (`prompt` only vs `prompt + response` vs full conversation).
- The three thresholds you used (exact = 1.0; MinHash Jaccard; semantic cosine).
- The tie-breaking rule.
- The count of rows dropped by each pass, per source and cross-source.
- 20 sampled dropped-row pairs per pass, so a reviewer can eyeball for false positives.

This lives alongside the curation report from chapter 1.

## Exercise pointer

The dedup techniques from this chapter reappear in the decontamination exercise (`exercises/exercise-02-eval-set-decontamination.md`), which reuses MinHash + LSH machinery against MMLU / HumanEval / GSM8K / MT-Bench prompts.

## Summary

- SFT duplicates hurt disproportionately because sets are small, duplicates cluster in easy regions, and near-dups of eval prompts leak into training.
- Three flavors, run in order: exact hash on a canonical form, MinHash + LSH for Jaccard near-dups, embedding-based semantic dedup for wording-invariant collapses.
- Recommended MinHash defaults (from SlimPajama/The Stack pipelines): word-3-gram or char-5-gram shingles, 128–256 permutations, 0.7–0.8 Jaccard threshold.
- Watch out for over-collapse on very short responses and code.
- Always union across sources before deduping — the cross-source trap is the number-one dedup mistake.
- Log normalization, granularity, thresholds, tie-breaking, and sampled drops. A dedup pass you cannot audit is not one you can ship.
