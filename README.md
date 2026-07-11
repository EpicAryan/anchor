# anchor — Personal Knowledge OS (Phase 1: Screenshot Intelligence)

Local-first search over your screenshots. Watches a folder, OCRs every image,
embeds the text locally, answers questions from the command line.

## Setup

```bash
sudo apt-get install -y tesseract-ocr
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
anchor index ~/Screenshots          # backfill existing screenshots
anchor watch                        # keep indexing new ones (Ctrl-C to stop)
anchor ask "screenshot of the module object is not callable error"
```

Config lives in `~/.anchor/config.json`:

```json
{
  "watch_dir": "/mnt/c/Users/YOU/Pictures/Screenshots",
  "allow_cloud": true,
  "cloud_provider": "gemini"
}
```

## Cloud LLMs (optional, off by default)

Without a cloud key and without Ollama, `anchor ask` still works — it returns
the raw matching snippets with their file paths (extractive mode).

To use a free-tier cloud LLM for synthesized answers:

1. Get a key: [Gemini](https://aistudio.google.com/apikey),
   [Groq](https://console.groq.com/keys), or
   [OpenRouter](https://openrouter.ai/keys).
2. `install -m 600 /dev/null ~/.anchor/env` then add `GEMINI_API_KEY=...`,
   `GROQ_API_KEY=...`, or `OPENROUTER_API_KEY=...` to it.
3. Per-question consent: `anchor ask "..." --cloud`
   Standing consent: set `"allow_cloud": true` in config. `--local` overrides.

## Security model

- **Your corpus never leaves the machine.** Embedding is always local.
  Only the top-k retrieved snippets (a few KB) are ever sent to a cloud
  provider, and only after opt-in.
- **Redaction before egress**: credential-shaped strings (AWS keys, GitHub
  tokens, JWTs, private keys, `password=` assignments…) are replaced with
  `[REDACTED:…]` markers before any cloud call. Best-effort — screenshots of
  secrets are still safest with `--local`.
- **Keys**: env vars only; `~/.anchor/env` is refused unless `chmod 600`.
  Keys are sent in headers (never URLs) and never logged.
- **Prompt injection**: OCR'd text is untrusted; the prompt instructs the
  model to treat snippets as data. Treat answers about "what to run next"
  with normal skepticism — this mitigates, not eliminates, injection.
- **Filesystem**: watcher ignores symlinks, paths outside the watch dir, and
  files > 20 MB. Data dir is `0700`, SQLite file `0600`.

## Known free-tier limits (July 2026 — recheck before relying on them)

- Gemini free tier: per-minute and per-day request caps; 429s are retried
  twice with backoff, then anchor falls back to extractive results.
- Groq free tier: token-per-minute caps; same fallback applies.
- OpenRouter: defaults to a `:free` model
  (`meta-llama/llama-3.3-70b-instruct:free`) so no credit is consumed;
  free models have daily request caps. Override the model with
  `ANCHOR_OPENROUTER_MODEL=<model-id>` in `~/.anchor/env`.
- **Free tiers may use your inputs for training.** That's the price of free —
  redaction plus top-k-only egress limits the exposure; `--local` avoids it.
