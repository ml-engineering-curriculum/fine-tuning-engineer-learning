# Converting Datasets Between Hugging Face and Managed JSONL

**Estimated read:** 35 minutes.
**You should be able to, by the end:** convert a Hugging Face `messages`-format `Dataset` into the JSONL that each of OpenAI, Anthropic, Vertex, Mistral, and Cohere expects; do the round-trip conversion back into `messages` format for local eval; correctly represent system prompts, multi-turn conventions, tool schemas, and image / multimodal rows for each target; and run the six-check validator that catches every common conversion bug before an upload consumes a training-budget dollar.

The dataset from mod-102 chapter 3 is in the **Hugging Face `messages` format**. Every managed vendor wants a JSONL in *their* format. The conversion is the "translation layer" between the two — small, subtle, silently-fatal-when-wrong. This chapter is that translation layer.

## The canonical intermediate form: messages

Mod-102 chapter 3 established `messages` as the canonical intermediate:

```python
# One example, in the canonical intermediate messages format:
{
    "messages": [
        {"role": "system",    "content": "You are a support triage agent."},
        {"role": "user",      "content": "My charge failed."},
        {"role": "assistant", "content": "I can help. Which card were you using?"},
        {"role": "user",      "content": "Visa ending 4402."},
        {"role": "assistant", "content": "Thanks. Checking that card now."},
    ],
    # Optional side-channel metadata that we do NOT ship to the vendor:
    "meta": {
        "source": "support/2026-07-transcripts.jsonl",
        "row_id": "s2026-07-t7-r138",
        "quality_score": 0.94,
    },
}
```

Two rules of the intermediate:

1. **`messages` is the payload; `meta` is out-of-band.** Never ship `meta` in the vendor JSONL — vendors either reject unknown fields (Anthropic, sometimes) or accept them silently and bill you for the tokens (OpenAI).
2. **Roles are `system`, `user`, `assistant`, and `tool`.** These are the intermediate-form values; the per-vendor conversion is responsible for mapping to `Chatbot`, `model`, etc.

Everything in this chapter is a function of shape `Dataset → JSONL`, with the canonical `messages` as the input.

## The rule of least surprise for the converter

The converter should:

1. **Never lose information.** If a row has content the target format cannot represent (an image, a tool call, an audio blob), the converter either transforms it losslessly or fails loud. Silent drop is the enemy.
2. **Round-trip.** The output JSONL, parsed back into the intermediate messages form, should equal the input modulo the metadata block. This is the primary correctness test (see below).
3. **Validate on the way out.** Every row that leaves the converter is checked against the target vendor's published schema. A row that would fail the vendor's ingest is caught locally.
4. **Be a pure function.** No I/O, no network, no globals. `convert_openai(row) → dict`. This makes the tests trivial.

## The per-vendor converters

Each is small. All five together fit in one file.

### OpenAI

```python
def to_openai(row):
    """messages format → OpenAI chat JSONL row."""
    out = {"messages": []}
    for m in row["messages"]:
        msg = {"role": m["role"]}
        # Tool-call turns encode content and tool structure separately.
        if "tool_calls" in m:
            msg["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m:
            msg["tool_call_id"] = m["tool_call_id"]
        if "content" in m and m["content"] is not None:
            msg["content"] = m["content"]
        if m.get("weight") is not None:
            msg["weight"] = int(m["weight"])
        out["messages"].append(msg)
    return out
```

- Roles pass through unchanged (`system`, `user`, `assistant`, `tool`).
- `weight` is preserved if present (mod-103 masking discipline; chapter 2 refresher).
- No top-level fields other than `messages`.

### Anthropic

```python
def to_anthropic(row):
    """messages format → Anthropic SFT JSONL row (top-level system + messages)."""
    out = {"messages": []}
    for m in row["messages"]:
        if m["role"] == "system":
            # Anthropic wants the system prompt at the top level, not in messages.
            existing = out.get("system", "")
            out["system"] = (existing + "\n\n" + m["content"]).strip() if existing else m["content"]
            continue
        out["messages"].append({"role": m["role"], "content": m["content"]})
    return out
```

- The system message moves to top-level.
- Multiple system messages (unusual but possible) are concatenated with `\n\n` — flag in a validator warning; the intent is usually a single system.
- Tool-calling format uses Anthropic's content-block shape (`{"type": "tool_use", ...}`); the converter must handle that if your dataset has tool rows.

### Google Vertex (Gemini)

```python
def to_vertex(row):
    """messages format → Vertex Gemini SFT JSONL row (systemInstruction + contents with parts)."""
    out = {"contents": []}
    for m in row["messages"]:
        if m["role"] == "system":
            out["systemInstruction"] = {"parts": [{"text": m["content"]}]}
            continue
        role = {"user": "user", "assistant": "model", "tool": "function"}.get(m["role"], m["role"])
        out["contents"].append({"role": role, "parts": [{"text": m["content"]}]})
    return out
```

- `assistant` → `model`. Silent bug source if you forget.
- Content becomes a list of `parts` (each a `{"text": ...}` or a `{"inlineData": ...}` / `{"fileData": ...}` for multimodal).
- `systemInstruction` at top level; not inside `contents`.

### Mistral

```python
def to_mistral(row):
    """messages format → Mistral La Plateforme SFT JSONL row (nearly identical to OpenAI)."""
    out = {"messages": []}
    for m in row["messages"]:
        msg = {"role": m["role"], "content": m["content"]}
        if "tool_calls" in m:
            msg["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m:
            msg["tool_call_id"] = m["tool_call_id"]
        out["messages"].append(msg)
    return out
```

- Nearly a no-op vs. OpenAI. If your dataset already produces OpenAI-shape rows, Mistral usually accepts them.
- No `weight` masking equivalent — if you're masking on OpenAI, you'll either need to accept the extra loss on Mistral or preprocess to remove the masked turns.

### Cohere

```python
def to_cohere(row):
    """messages format → Cohere Command SFT JSONL row (capitalized roles)."""
    ROLE_MAP = {"system": "System", "user": "User", "assistant": "Chatbot"}
    out = {"messages": []}
    for m in row["messages"]:
        if m["role"] not in ROLE_MAP:
            raise ValueError(f"Cohere does not accept role {m['role']!r} in generative SFT")
        out["messages"].append({"role": ROLE_MAP[m["role"]], "content": m["content"]})
    return out
```

- Role names capitalized and `assistant` → `Chatbot`. The most common Cohere bug is lowercase roles.
- No `tool` role in the generative SFT schema (as of Aug 2026); the Rerank / Embed schemas are qualitatively different formats — see chapter 3.

## The one-line integration into a Hugging Face pipeline

Given the converters, wire them through `datasets.Dataset.map` and write JSONL:

```python
from datasets import Dataset
import json

CONVERTERS = {
    "openai":    to_openai,
    "anthropic": to_anthropic,
    "vertex":    to_vertex,
    "mistral":   to_mistral,
    "cohere":    to_cohere,
}

def write_jsonl(ds: Dataset, vendor: str, out_path: str):
    convert = CONVERTERS[vendor]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            converted = convert(row)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")

# Usage
ds = Dataset.load_from_disk("data/support-triage-v3")
write_jsonl(ds, "openai",    "artifacts/openai/train.jsonl")
write_jsonl(ds, "anthropic", "artifacts/anthropic/train.jsonl")
```

Two conventions worth adopting from day one:

- **Separate output paths per vendor.** Never overwrite. If you find yourself diffing OpenAI and Anthropic JSONLs to debug, having both on disk is the difference between five minutes and an hour.
- **Emit `ensure_ascii=False`.** Non-English content silently becomes escape sequences with the default; harmless but painful when debugging.

## The round-trip test — your primary correctness signal

The single most valuable test in this pipeline is the round-trip:

```python
def from_openai(row):
    """OpenAI JSONL row → messages format."""
    out = {"messages": []}
    for m in row["messages"]:
        msg = {"role": m["role"]}
        for k in ("content", "tool_calls", "tool_call_id", "weight"):
            if k in m:
                msg[k] = m[k]
        out["messages"].append(msg)
    return out

def test_openai_roundtrip(ds):
    for row in ds:
        vendor_row = to_openai(row)
        recovered  = from_openai(vendor_row)
        # Strip `meta` before compare; it's out-of-band.
        original = {"messages": row["messages"]}
        assert recovered == original, f"round-trip mismatch: {row.get('meta', {}).get('row_id')}"
```

Do this for every vendor. A converter that passes round-trip on 100 % of the dataset is one that cannot be silently losing content; a converter that passes on 99.9 % is losing content on 0.1 % of your training data. Investigate every failure.

For vendors where round-trip cannot be exact (Anthropic's system-message concatenation, Vertex's role rename), define the "canonical recovery" explicitly and test against that.

## The six-check validator

Run every generated JSONL through the same six checks before uploading. This is the "validate what the vendor will validate, locally" discipline from chapter 2, generalized:

```python
def validate_jsonl(path, vendor):
    errors = []
    warnings = []
    n = 0
    total_tokens = 0

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            n += 1

            # Check 1: valid JSON
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((i, f"invalid JSON: {e}"))
                continue

            # Check 2: expected vendor schema shape
            if vendor in {"openai", "mistral", "cohere"}:
                if "messages" not in row or not isinstance(row["messages"], list):
                    errors.append((i, "missing or non-list 'messages'"))
                    continue
            elif vendor == "anthropic":
                if "messages" not in row:
                    errors.append((i, "missing 'messages'"))
                    continue
                if "system" in row and not isinstance(row["system"], str):
                    errors.append((i, "'system' must be a string"))
            elif vendor == "vertex":
                if "contents" not in row:
                    errors.append((i, "missing 'contents'"))
                    continue

            # Check 3: at least one assistant / model / Chatbot turn
            ASSISTANT_ROLE = {
                "openai": "assistant", "mistral": "assistant",
                "anthropic": "assistant", "cohere": "Chatbot", "vertex": "model",
            }[vendor]
            turns = row.get("messages", row.get("contents", []))
            if not any(m.get("role") == ASSISTANT_ROLE for m in turns):
                errors.append((i, f"no {ASSISTANT_ROLE} turn — trains nothing"))

            # Check 4: no forbidden top-level fields
            allowed = {
                "openai":    {"messages"},
                "anthropic": {"messages", "system"},
                "vertex":    {"contents", "systemInstruction"},
                "mistral":   {"messages"},
                "cohere":    {"messages"},
            }[vendor]
            extras = set(row.keys()) - allowed
            if extras:
                warnings.append((i, f"unknown top-level fields: {sorted(extras)}"))

            # Check 5: rough token budget (per-model max context)
            # ... use tiktoken / vendor tokenizer to count and check per vendor's docs ...

            # Check 6: content is non-empty on every message
            for m in turns:
                c = m.get("content", m.get("parts"))
                if not c:
                    warnings.append((i, "empty content on a turn"))

    print(f"{path}: rows={n}, errors={len(errors)}, warnings={len(warnings)}")
    return errors, warnings
```

Adopt this as `validate.py` in your dataset repo. Run it in CI before every managed-FT run.

## Handling the harder cases

### System prompts and where they live

The intermediate form treats the system prompt as the first `messages` element with `role == "system"`. Vendors put it in three different places:

| Vendor | Where the system prompt lives |
|---|---|
| OpenAI | First message in `messages` with `role: "system"`. |
| Mistral | First message in `messages` with `role: "system"`. |
| Cohere | First message in `messages` with `role: "System"` (capitalized). |
| Anthropic | Top-level `system` field, sibling to `messages`. |
| Vertex | Top-level `systemInstruction` with `parts`. |

The converters above encode each of these. If your dataset has *no* system message and production always sends one, add the production system prompt at conversion time (or, better, fix the dataset — a fine-tune trained without the system prompt behaves differently when the prompt is present at inference).

### Multi-turn conventions

The intermediate form allows arbitrary alternation of `user` and `assistant`. Some vendor validators reject `user → user` without an intervening `assistant`, or reject a trailing `user` message (i.e., they require the example to end on an `assistant` turn so there is something to learn from).

Rules to bake into the converter:

- Strip empty-content turns. An empty user turn followed by an empty assistant turn (a data pipeline bug from mod-102) trains noise; the validator warns, the converter drops if configured to.
- Verify `assistant` → `user` → `assistant` alternation (with `system` optionally at the start). Non-alternating multi-turn examples are usually a mod-102 bug (an assistant-only self-play row that leaked into the dataset).
- If a vendor requires a terminal assistant turn (some do for SFT), verify and warn.

### Tool-calling training rows

If your dataset trains the model to emit tool calls, each vendor represents them differently. The intermediate form for a tool-training turn:

```python
# Example turn in intermediate form
{
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "lookup_order",
                "arguments": '{"order_id": "40219"}',
            },
        }
    ],
}
# followed by the tool response turn
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": '{"status": "shipped", "carrier": "UPS", ...}',
}
```

- **OpenAI**: the intermediate shape above passes through with minimal changes.
- **Anthropic**: `assistant` turn's content becomes a list of blocks — `{"type": "text", "text": "..."}` mixed with `{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`; the tool response becomes a user turn with `content: [{"type": "tool_result", "tool_use_id": ..., "content": ...}]`. The structure is qualitatively different — plan your converter accordingly.
- **Vertex**: uses `functionCall` and `functionResponse` parts inside `contents`.
- **Mistral**: similar to OpenAI; verify the current tool JSONL format in the current docs.
- **Cohere**: generative SFT does not currently support tool-call training rows the same way; check the current docs before promising this capability on Cohere.

Testing tool-call conversion needs a dedicated round-trip suite — the shape divergence is large enough that a text-only round-trip test will not catch tool-shape bugs.

### Image / multimodal rows

If your dataset includes image inputs (chart interpretation, screenshot QA, receipt parsing), each vendor's multimodal JSONL shape differs. Intermediate form:

```python
{
    "role": "user",
    "content": [
        {"type": "text",  "text": "What is the total on this receipt?"},
        {"type": "image", "url":  "https://.../receipt-1042.png"},
    ],
}
```

- **OpenAI**: `content` is a list of `{"type": "text", ...}` and `{"type": "image_url", "image_url": {"url": ...}}` blocks. Image can be `data:image/png;base64,...` for inline.
- **Anthropic**: `content` is a list of `{"type": "text", ...}` and `{"type": "image", "source": {"type": "base64" | "url", ...}}` blocks.
- **Vertex**: `parts` is a list of `{"text": ...}` and `{"inlineData": {"mimeType": ..., "data": ...}}` or `{"fileData": {"mimeType": ..., "fileUri": ...}}` blocks.
- **Cohere / Mistral**: verify current multimodal FT support per model in current docs.

For multimodal training, image bytes vs. URL vs. GCS `gs://` vs. S3 `s3://` all matter — the vendor's ingest may reject URLs it cannot reach, and inline base64 rapidly blows up file size. Store the image bytes in the vendor's expected location before the JSONL points at them; do not link to bytes that the vendor cannot fetch.

## Tokenization discipline

You need accurate per-row token counts for two reasons: (1) cost estimation before the job runs, (2) rejection of rows that exceed the base model's context window. Use the tokenizer the base model actually uses.

| Base model family | Tokenizer to count with |
|---|---|
| GPT-4.1 / GPT-4o family | `tiktoken.encoding_for_model("gpt-4o")` (or the specific model) |
| Claude | Anthropic's token counting endpoint (`client.messages.count_tokens(...)`) |
| Gemini | Vertex SDK's `count_tokens` / `token_count` methods |
| Mistral | Mistral's tokenizer (via `mistral-common`) |
| Cohere | Cohere's tokenizer (via `co.tokenize`) |

Do **not** use `len(text.split())` or `len(text) / 4` as a proxy. Long code, non-English text, and emoji all tokenize non-linearly relative to character count.

Include the per-row token count in the manifest that ships alongside the JSONL:

```yaml
train_file:
  path: artifacts/openai/train.jsonl
  sha256: 3c...
  rows: 50247
  tokens_per_row_p50: 412
  tokens_per_row_p99: 3821
  total_tokens: 25123456
  base_model_context: 128000
  base_model_tokenizer: cl100k_base  # or o200k_base, gemini, etc.
```

## The ten most common conversion bugs

Compiled from real failure logs. Prevent, don't diagnose.

1. **Wrong role name.** `"role": "assistant"` sent to Cohere (which wants `"Chatbot"`) or to Vertex (which wants `"model"`). Rejected at ingest for some vendors, silently discarded for others. Validator check #3 catches.
2. **`meta` field leaks into the JSONL.** Extra top-level fields; some vendors bill you for the extra tokens or reject the row. Validator check #4 catches.
3. **System prompt in `messages` instead of top-level.** Anthropic and Vertex silently ignore a `system` role inside `messages` (or reject it). Converter must move it.
4. **Empty assistant turn.** `{"role": "assistant", "content": ""}` — trains the model to emit empty strings on that context. Round-trip test misses this; validator check #6 catches.
5. **`weight: 0` on OpenAI applied to too many turns.** If every assistant turn is `weight: 0`, the row trains nothing but you're billed for the tokens. Validator should warn when the assistant-token count with weight > 0 is zero.
6. **Vendor-specific token-count blow-up.** Content that tokenizes to 5000 tokens on `cl100k_base` may tokenize to 6000 on a different tokenizer. Use the right tokenizer.
7. **JSONL row splits mid-object.** Usually an editor bug — a JSON object with a literal newline in a string, un-escaped. Validator check #1 catches (JSON decode fails).
8. **BOM at the start of the file.** UTF-8 BOM (`\xef\xbb\xbf`) makes some vendor ingests reject the first row. Write files with `open(path, "w", encoding="utf-8")`, not `encoding="utf-8-sig"`.
9. **Escaped Unicode where you don't expect it.** `ensure_ascii=True` (the `json.dumps` default) turns `"café"` into `"café"`. Legal JSON, technically correct, but confuses eyeball audits and can differ from production text. Prefer `ensure_ascii=False`.
10. **Tool-call format mismatch.** Assistant turn's `tool_calls` shape differs across vendors; if you convert as if it were plain text, the row trains nothing or (worse) trains the model to emit the JSON literal as content.

The six-check validator plus a dedicated tool-call round-trip test catches all ten. Set them as a pre-upload CI gate.

## The mandatory manifest artifact

Every generated JSONL is accompanied by a `manifest.yaml` that records the conversion. Same discipline as chapter 2's job manifest, at the dataset layer:

```yaml
source_dataset:
  path: data/support-triage-v3
  git_commit: 8a3c...
  row_count: 50759
  tokenizer_for_source: none  # source is text; token counts computed at conversion
  license: internal-only

conversion:
  vendor: openai
  converter_version: 1.2.0
  converter_code_git_commit: 8a3c...
  timestamp: 2026-08-01T14:12:07Z

output:
  path: artifacts/openai/train.jsonl
  sha256: 3c...
  rows: 50247               # 512 dropped by filter
  dropped_reason:
    empty_assistant: 341
    over_context_window: 137
    tool_call_convert_failure: 34
  total_tokens: 25123456
  base_model_tokenizer: cl100k_base

validation:
  errors: 0
  warnings: 3
  warning_details:
    - line 12034: "empty content on a turn"

roundtrip_test:
  pass_rate: 1.0
  failing_row_ids: []
```

This is what the mod-108 release-gate report references and what the mod-111 model registry indexes. It is also what tells you, six months later, why a rerun on the same source dataset produces a slightly-different JSONL — versions moved.

## Where this fits with mod-102 and mod-108

- **mod-102 chapter 3** is the source of truth for the intermediate `messages` dataset. Every dataset-quality property (dedup, decontam, licensing, dataset card) is a precondition for this chapter. If the dataset is broken, the conversion is a waste.
- **mod-108 chapter 5** owns the decontamination discipline. If a row of your training data appears in your MMLU or task-specific held-out eval, the JSONL you generate here still contains it. Decontaminate first; convert second.
- **mod-108 chapter 6** owns the release-gate manifest. The conversion manifest (above) is one of the inputs to the release-gate manifest; the vendor JSONL SHA is what appears in the "training data" section of the release report.

## Summary

- The Hugging Face `messages`-format `Dataset` is the canonical intermediate. Every vendor JSONL is a pure-function conversion from it.
- Five converter functions, one per vendor, cover the JSONL generation. Total code is ~150 lines. Round-trip test proves correctness.
- The system prompt lives in five different places across vendors (in `messages` for OpenAI / Mistral / Cohere; top-level `system` for Anthropic; top-level `systemInstruction` for Vertex). The converter is responsible for moving it correctly.
- Role names differ: OpenAI/Mistral/Anthropic use `assistant`; Cohere wants `Chatbot`; Vertex wants `model`. The converter must map explicitly — silent typos here train nothing.
- Tool-call and multimodal rows have qualitatively different shapes per vendor. A text-only round-trip test does not catch tool-shape bugs — build a dedicated tool-call test suite.
- The six-check validator (JSON valid, schema, has-assistant, no-unknown-fields, token budget, non-empty content) catches all ten of the most-common conversion bugs before an upload consumes cost.
- Ship a `manifest.yaml` alongside every generated JSONL: source-dataset provenance, converter version, output hash, rejected-row counts, validation results, round-trip results. This is the artifact that lets a rerun in six months produce a comparable JSONL.
