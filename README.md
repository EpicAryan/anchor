# anchor — Personal Knowledge OS (Phase 1: Screenshot Intelligence)

Local-first search over your screenshots. Watches a folder, OCRs every image,
embeds the text locally, answers questions from the command line.

## Setup

```bash
sudo apt-get install -y tesseract-ocr poppler-utils
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
anchor index                        # sync ALL watch folders (adds new, removes deleted)
anchor index ~/any/folder           # index any folder on demand (recursive)
anchor watch                        # live: indexes new files, forgets deleted ones
anchor ask "screenshot of the module object is not callable error"
anchor ask "what does my note say about the deploy checklist"
anchor find "dashboard" --type pdf  # list matching PDFs (no LLM, offline)
anchor prune                        # drop index entries for deleted files
```

Indexes screenshots (OCR), PDFs (digital + scanned via OCR), notes
(`.md .txt .rst`), and code/config files. Folders are walked recursively;
`.git`, `node_modules`, virtualenvs, hidden and build directories are
skipped, and secret-shaped files (`.env*`, `*.pem`, `*.key`, `id_rsa*`,
`credentials*`, …) are never indexed.

Config lives in `~/.anchor/config.json`:

```json
{
  "watch_dirs": [
    "/mnt/c/Users/YOU/Pictures/Screenshots",
    "/mnt/c/Users/YOU/Documents/pdfs",
    "/mnt/c/Users/YOU/Documents/notes"
  ],
  "allow_cloud": true,
  "cloud_provider": "gemini"
}
```

(The old single `"watch_dir"` key still works.)

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
- **Filesystem**: watcher and indexer skip symlinks, paths outside watched
  roots, oversized files (20 MB; 50 MB for PDFs), and excluded directories —
  and secret-shaped files are blocked from the index entirely (redaction
  only guards cloud egress). Data dir is `0700`, SQLite file `0600`.

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
