# Security

anchor is **local-first**: your documents are indexed and searched on your
own machine. This file describes what that guarantees, what it does *not*,
and how to report a vulnerability.

## Threat model at a glance

| Data | Where it lives | Leaves your machine? |
|------|----------------|----------------------|
| Your files' full text | `~/.anchor/` (SQLite + Chroma) | **Never** |
| Embeddings (vectors) | `~/.anchor/chroma/` | **Never** — embedding is local-only, there is no cloud embedding path |
| Top-k matching snippets | in memory during a query | **Only** with `ask --cloud` (or `allow_cloud: true`), and only after redaction |
| API keys | `~/.anchor/env` (you provide) | Sent as request **headers** to the provider you chose, never logged, never in URLs |

The whole corpus never leaves the machine. At most a handful of the
snippets most relevant to a single question are sent to a cloud LLM — and
only when you opt in.

## What protects your data

- **Local-only embeddings.** `Embedder` has no cloud path by construction;
  the full corpus is never transmitted anywhere.
- **Opt-in cloud egress.** Answering with a cloud provider requires either
  `ask --cloud` (one-shot) or `allow_cloud: true` in config. Without it,
  queries run against a local LLM (Ollama) or degrade to local extractive
  matches. Nothing is sent.
- **Secret files are blocked at ingestion.** Files matching `.env*`,
  `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `*.p12`, `*.pfx`,
  `credentials*`, `secrets.*` are never indexed at all.
- **Redaction before egress.** Every snippet sent to a cloud provider is
  first run through a redactor that masks private keys, AWS keys, GitHub /
  Slack tokens, JWTs, bearer tokens, and `key=`/`secret=`/`password=`-style
  assignments.
- **Filesystem permissions.** `~/.anchor/` is created `0700`, the metadata
  DB `0600`, and the `env` file is *refused* if it is group/other-readable.
- **No content in logs.** Errors and progress log file paths, counts, and
  exception class names only — never file contents.
- **Injection-resistant.** SQL is fully parameterized. OCR/PDF tools are
  invoked with argument lists, never a shell. Retrieved snippets are
  labelled untrusted in the LLM prompt so embedded instructions are ignored.

## What is *not* guaranteed — read this before enabling cloud

- **Redaction is best-effort, not a guarantee.** It is pattern-based and
  will not catch every possible secret format. The real protection is that
  cloud egress is opt-in.
- **Indexable files that contain secrets are still indexed.** A
  `config.json`, `settings.json`, `.py`, etc. that contains an API token is
  a *supported* file type and will be indexed (its content stays local).
  If you then run `ask --cloud`, that content is subject only to
  best-effort redaction before being sent. **For anything sensitive, prefer
  `ask --local`, or don't point anchor at folders holding raw credentials.**
- **anchor trusts your local machine.** It does not defend against other
  processes or users already running as you on the same account.

## Reporting a vulnerability

Please report security issues privately via GitHub's "Report a
vulnerability" (Security Advisories) on this repository rather than opening
a public issue. A maintainer will respond as soon as possible.
