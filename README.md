# codeprint

## What's New in v1.1.0

### `codeprint diff <path-a> <path-b> [--ai]` — Side-by-side codebase comparison

Compare the quality metrics of two directories — before/after a refactor, or between two different projects. Green Δ means improvement, red means regression. Add `--ai` for a streamed narrative analysis.

```bash
codeprint diff ./v1 ./v2             # compare two versions
codeprint diff ./legacy ./rewrite --ai   # with AI commentary
```

```
╭─ Diff: legacy vs rewrite ─────────────────────────────────────╮
│  Metric          legacy   rewrite        Δ                     │
│  Files               42        38       -4                     │
│  Total LOC        6,841     4,203   -2638 ✓                    │
│  Functions          312       187    -125 ✓                    │
│  Open TODOs          14         3     -11 ✓                    │
│  Avg complexity     12.4       7.1     -5.3 ✓                  │
│  Max fn LOC         182        64    -118 ✓                    │
╰───────────────────────────────────────────────────────────────╯
  Green Δ = improvement (B better than A) · Red Δ = regression
```

### `--format json|csv|markdown` — Structured output for all commands

Export scan results, duplicate pairs, and search rankings in machine-readable formats for CI pipelines, dashboards, or further processing.

```bash
codeprint scan . --format json > report.json
codeprint scan . --format csv | csvcut -c file,loc,complexity
codeprint dupes . --format markdown > dupes.md
codeprint search . "auth middleware" --format json
```

### Config file (`~/.codeprint.toml`)

Persist default settings so you don't repeat flags every run.

```toml
# ~/.codeprint.toml
[defaults]
top = 20
threshold = 0.80
format = "table"
```

---

**Async codebase quality scanner with semantic duplicate detection and AI architectural insights.**

`codeprint` scans a Python or JavaScript project directory, extracts quality metrics from every source file concurrently (using `asyncio.TaskGroup`), finds near-duplicate code with TF-IDF cosine similarity, and can stream an AI architectural summary straight to your terminal.

## Breakthrough techniques

| Technique | Where |
|---|---|
| **Full async architecture** | `scanner.py` — `asyncio.TaskGroup` processes hundreds of files concurrently; `asyncio.to_thread` prevents blocking the event loop during file I/O |
| **Semantic vector search** | `vectors.py` — hand-rolled TF-IDF matrix (numpy) + cosine similarity to find near-duplicate files and power natural language search without heavy ML deps |
| **LLM integration** | `summarizer.py` — Claude Haiku streaming (`client.messages.stream`) delivers an AI architectural assessment in real-time |
| **Live Rich UI** | `cli.py` / `dashboard.py` — Rich `Progress` + `Live` shows scan progress as files are processed; structured result tables with colour-coded complexity scores |

---

## Install

```bash
pip install codeprint
# or from source:
git clone https://github.com/iamgeetarted/codeprint
cd codeprint && pip install -e .
```

Requires Python ≥ 3.11. Set `ANTHROPIC_API_KEY` for AI features.

---

## Usage

### `codeprint scan [path]` — quality report

Scans all `.py`, `.js`, `.ts` (and variants) files in a directory, skipping `node_modules`, `__pycache__`, `.venv`, etc.

```bash
codeprint scan .                   # scan current directory
codeprint scan ~/projects/myapp    # scan a specific path
codeprint scan . --top 20          # show top 20 files per table
codeprint scan . --summary-only    # just the overview panel
codeprint scan . --ai              # add AI architectural summary
```

**Sample output:**

```
╭─ codeprint  /home/user/myapp ──────────────────────────────╮
│  Files             42                                       │
│    Python          38                                       │
│    JavaScript       4                                       │
│  Total LOC      6,841                                       │
│  Functions        312                                       │
│  Classes           28                                       │
│  Imports          178                                       │
│  TODOs / FIXMEs    14                                       │
╰─────────────────────────────────────────────────────────────╯

╭─ Largest Files (top 15) ───────────────────────────────────╮
│  File                   Lang   LOC  Fns  TODOs             │
│  app/api/endpoints.py   Py     421   38      3  ████████   │
│  app/models/user.py     Py     318   22      —  ██████     │
│  tests/test_api.py      Py     287   41      —  █████      │
╰─────────────────────────────────────────────────────────────╯

╭─ Complexity Hotspots (top 15) ─────────────────────────────╮
│  File                   Score  Max fn LOC  Avg fn LOC  TODOs│
│  app/api/endpoints.py    48.5         182        11.1      3│
│  app/utils/parser.py     31.2          94         8.4      2│
╰─────────────────────────────────────────────────────────────╯
```

### `codeprint dupes [path]` — near-duplicate detection

Finds files with suspiciously similar token distributions — a signal of copy-paste code that should be extracted into shared utilities.

```bash
codeprint dupes .
codeprint dupes . --threshold 0.85   # stricter (default: 0.70)
codeprint dupes . --top 10
```

```
╭─ Near-Duplicate Files — 3 pairs ───────────────────────────────────────────╮
│  File A                     File B                   Sim%  Shared tokens   │
│  app/api/v1/users.py        app/api/v2/users.py      94%   serialize, user │
│  utils/validators.py        core/validators.py       78%   validate, field │
╰────────────────────────────────────────────────────────────────────────────╯
  Similarity ≥85% likely duplication · 70–84% significant overlap
```

### `codeprint search [path] [query]` — semantic file search

Find the files most likely to contain logic related to a plain-English query.

```bash
codeprint search . "database connection pool"
codeprint search . "authentication middleware jwt"
codeprint search . "retry logic exponential backoff" --top 5
```

```
╭─ Search: "database connection pool" ───────────────────────╮
│  Score  File                                               │
│   78%   app/db/connection.py                               │
│   61%   app/db/pool.py                                     │
│   34%   app/config.py                                      │
╰────────────────────────────────────────────────────────────╯
```

### `codeprint insights [path]` — AI architectural summary

Sends codebase metrics to Claude Haiku and streams a concise architectural assessment.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
codeprint insights .
```

```
╭─ AI Architectural Insights ─────────────────────────────────╮
╰─────────────────────────────────────────────────────────────╯

This appears to be a Flask/FastAPI web application with 42 Python
files totalling ~6,800 LOC. The main concern is **app/api/endpoints.py**
(421 LOC, score 48.5) which concentrates too much routing logic — typical
of a monolithic view layer that would benefit from splitting into domain-
specific blueprints. The 14 open TODOs cluster around authentication and
parser utilities, suggesting these areas have known debt. Recommend
extracting `app/utils/parser.py` into a standalone library and writing
integration tests before further expansion.
```

---

## Configuration

No config file required. All options are CLI flags. Supported file types: `.py`, `.js`, `.mjs`, `.cjs`, `.ts`, `.tsx`.

Skipped directories: `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`, `.next`, `coverage`, `.pytest_cache`.

---

## Architecture

```
codeprint/
├── cli.py        # argparse entrypoint, async orchestration
├── scanner.py    # asyncio.TaskGroup concurrent file scanner
├── metrics.py    # per-file static metrics extraction (Python + JS)
├── vectors.py    # TF-IDF matrix, cosine similarity, semantic search
├── dashboard.py  # Rich tables, panels, progress display
└── summarizer.py # Claude streaming AI insights
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
