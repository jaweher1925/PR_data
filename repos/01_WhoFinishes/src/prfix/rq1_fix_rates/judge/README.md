# LLM-as-judge — RQ1 "does Q fix P?"

This package decides, for a pair of pull requests **(P, Q)** in the same repo,
whether the later PR **Q** is a follow-up **fix** of the earlier AI-agent PR
**P**. It is the *precision layer* over the file-overlap recall net used by the
RQ1 follow-up signal: file overlap finds *candidate* pairs cheaply but
over-links (two PRs can touch the same file for unrelated reasons); the judge
reads the actual context and decides the relationship.

It is calibrated and scored against the **human consensus labels** produced in
[`../../../human-annotation/`](../../../human-annotation) (50 hand-labeled pairs,
two annotators), then run at scale over the net-diff candidate population.

## Two things this package does

1. **Learn** from the human labels: distil **all annotator-agreed pairs** (the 37
   pairs both reviewers labelled the same) into a compact **LEARNED INSIGHTS**
   block (`prompt.py`'s `insights build` CLI) that, together with the rubric and a
   few worked-example anchors, forms the judge's system prompt. This replaces the
   older fixed 6-shot prompt — every agreed pair now informs the judge.
2. **Run** the judge at scale over the **net-diff population** — ~2.5k candidate
   (P, Q) pairs that share files (agent-authored P; agent *or* human fixer Q) —
   with the rater(s) configured in `judge.yaml` (the published run: Opus 4.8 at
   low effort), resumably (`run_judge.py`). This produces the actual
   follow-up-fix labels.

**Validation is separate and independent:** draw a **blind human sample** from a
population CSV (`human-annotation/sample_pairs.py judge`) and score a human against the judge with
`human-annotation/score.py judge`. (There is no longer an in-prompt held-out gold
score — the agreed pairs are all training signal now, so scoring on them would
leak.)

## Label space

Matches the human task (`human-annotation/README.md`):

| label | meaning |
|---|---|
| **Direct fix** | Q repairs a defect/regression P introduced or left behind |
| **Related touch** | Q does incremental/additive work in the same area (not a repair) |
| **Unrelated/mislabeled** | Q touches the shared file for an independent reason; overlap is coincidental |
| **Indeterminate** | not enough context to decide |

For the downstream signal these collapse to binary: **strict** = `Direct fix`
vs rest; **lenient** = `Direct fix | Related touch` vs `Unrelated`.

## What the judge sees (inputs) — and what it does not

The judge is given **only** the per-pair "case file", in two layers.

**Layer 1 — system prompt (identical for every pair):** the rubric, then a
**LEARNED INSIGHTS** block distilled from **all 37 annotator-agreed pairs** and
their reason notes (`prompt.py`'s `insights build`, frozen to `insights.md`), then a few **worked-
example anchors** (`pick_exemplars`, 3 per class = 9, deliberately **compact** —
≤2 files each). The insights let every agreed pair inform the judge while the
prompt stays small and cacheable. It is
sent as a cached content block (Anthropic `cache_control: ephemeral`, forwarded by
LiteLLM), so after the first call subsequent pairs read it at ~0.1× input cost.

**Layer 2 — the case (the actual question):**
1. P description (title + body)
2. Q description (title + body)
3. P commit messages
4. Q commit messages
5. **Shared-file changes** — net-diff `P ∩ Q`, with *both* P's and Q's patch per
   file, each prefixed by a machine-computed **`[P-overlap]` attribution note**
   (see below)
6. light metadata — day gap, Q's conventional-commit type, and how many *other*
   PRs merged between P and Q that also touched the shared file(s)
7. *(gold `--full-diff` only)* each PR's **non-shared** changed files too, so a fix
   that lives outside the overlap is still visible

### The `[P-overlap]` attribution note (why it exists)

P's patch is the diff vs the state *before P*; Q's patch is the diff vs the state
*before Q* — **different baselines**. So an unprefixed *context* line in Q's patch
is not necessarily code P wrote: a PR merged *between* P and Q may have authored
it. Left unguarded, the judge over-credits Q with "fixing P" whenever the two
merely share a file (the pair-#69 failure mode). Each shared file therefore
carries a one-line attribution note answering *"does Q's own change actually land
on a line P introduced?"*, computed two ways:

* **Method B — `git blame` (ground truth), [`attribution.py`](attribution.py) (Method B section).**
  Blames the shared file at **Q's merge-base** on the exact lines Q *removes* and
  checks whether the authoring commit belongs to P (P's commit shas + squash
  `merge_commit_sha`). "blame confirms N line(s) Q removes were authored by P" is
  the strongest Direct-fix evidence; "none authored by P" reliably rules a direct
  fix out. History-aware, so it sees through reformatting and intermediate
  rewrites. All fetches (PR meta, merge-base, blame) are cached to JSON snapshots
  in `data/interim/`, so a rebuild is reproducible and free. (The published
  run's snapshots are already baked into the committed case files.)
* **Method A — content overlap (fallback), [`attribution.py`](attribution.py) (Method A section).** When
  blame is unavailable (pure-addition file, rename, doc file, rebase-merge that
  rewrote P's commits, or a `--no-blame` build), it falls back to matching line
  *content*: does a line Q removes / a context line in Q's hunk match a line P
  added? Also flags a **cosmetic-only** change (Q's net diff here is
  comment/whitespace) as "NOT a code fix".

When Method A finds Q's *only* change to the shared code is cosmetic, a
**provenance banner** is prepended to the whole case (the pair-#69 guard, 0/14
false positives on gold Direct-fixes). RUBRIC #6–#7 tell the judge to weight this
attribution — especially a blame confirmation — over mere file overlap or a
commit message that narrates a fix the net diff doesn't contain.

**How many files render in a case.** A case carries everything it stored — nothing
is dropped at render time.
* **Gold cases** show **all** shared files (no cap). With `--full-diff` they also
  store up to **12 non-shared files per PR** (`build_cases.MAX_EXTRA_FILES`): each
  PR's non-shared files are sorted substantive-first, then by patch size, and the
  top 12 are kept.
* **Population cases** are always **shared-only** (no non-shared files); each file
  patch is truncated to `patch_cap` chars (default 12,000).
* **Layer-1 exemplars** are squeezed to ≤2 files via an explicit small cap.

It does **not** get the rest of the codebase, PR comments, reviews, or repo
history. It judges the *relationship*, not whether Q is a good PR.

## How it works (flow)

```
build_cases.py  (one builder, two modes)
  ├─ build_gold()        human labels + reviewed net-diff (enriched set)
  │                       → data/judge/rq1_judge_cases.json          (50 pairs; 37 agreed)
  └─ build_population()   DuckDB candidate (P, Q) net-diff pool + blame attribution
                          → data/judge/rq1_judge_cases_population_12k.json  (~2.5k pairs)
        │
        ▼
prompt.py  `insights build`   distil ALL 37 agreed pairs (+ reason notes) → insights.md   (one LLM call, frozen)
        │
        ▼
prompt.py   build_system_prompt() = RUBRIC + LEARNED INSIGHTS + pick_exemplars() anchors (3/class = 9)
        │
        ▼
run_judge.py (+ judge.yaml)   judges ALL ~2.5k judgeable pairs with every configured rater,
                              resumable checkpoints → data/judge/rq1_llm_judge_pop_<rater>.csv
                              → label dist per rater + model↔model agreement κ + disagreement list
        │
        ▼
human-annotation/sample_pairs.py judge   draw a blind human-vs-judge sample from a population CSV
             (labelled via gen_labeler.py --mode judge, scored by score.py judge — the independent check)
```

The raters in `engine.py` (`AnthropicRater`, `OpenAIRater`, `make_rater`) wrap each
provider behind one `judge_case(case, system_prompt)` interface so the prompt never
changes between models. `_kappa`/`_binary` back the population agreement report.

---

## A0. Build the case files (`build_cases.py`)

Usually you **don't** run this by hand — **the published run's case files
are committed** (`data/judge/rq1_judge_cases.json`, `..._population_12k.json` and
`..._population_12k_humanP.json` ship with the package), so on a fresh clone
they are simply reused — the exact inputs the judge saw, no DB or `gh` needed.
The shipped `judge.yaml` pins `auto_build: false` so a *missing* cases file
errors instead of silently regenerating one off the pins.
Run the builder directly only to (re)generate a case file with non-default knobs,
to refresh the diff snapshot, or to rebuild the gold set — note a manual
population build **always overwrites** the committed file (no presence check).

```sh
# gold: the fixed 50-pair validation set → data/judge/rq1_judge_cases.json
uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode gold

# gold + each PR's NON-shared files too (a fix that lives outside the overlap)
uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode gold --full-diff

# population: the net-diff pool run_judge judges → data/judge/rq1_judge_cases_population_12k.json
uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode population --judgeable-only

# population smoke test: first 20 pairs only
uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode population --judgeable-only --limit 20

# just warm the gh diff snapshot (no assembly), e.g. to pre-fetch before an offline build
uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode population --fetch-only
```

| flag | mode | default | what it does |
|---|---|---|---|
| `--mode {gold,population}` | both | `gold` | which set to build |
| `--out PATH` | both | derived | explicit output JSON path (else derived from mode/patch-cap) |
| `--patch-cap N` | both | gold `6000`, pop `12000` | chars per file patch; **also names the population file** (`12000`→`..._12k.json`) |
| `--full-diff` | gold | off | also include each PR's non-shared changed files (≤12/PR, `MAX_EXTRA_FILES`) |
| `--judgeable-only` | population | off | keep only pairs with a real net-diff overlap (`n_shared>0`) — what production uses. Without it the derived filename gains an `_all` suffix (`..._population_12k_all.json`) |
| `--p-side {agent,human}` | population | `agent` | whose merges form the P side: the agent cohort (the paper's run) or the contemporaneous human cohort (the incidence baseline). `human` adds a `_humanP` suffix to the derived filename and marks every case `p_is_human` |
| `--limit N` | population | all | first N pairs (smoke test) |
| `--concurrency N` | population | `8` | parallel `gh` fetches |
| `--fetch-only` | population | off | only populate the diff snapshot; don't assemble the case file |
| `--no-blame` | population | off | skip Method-B `git blame` attribution (fewer `gh` calls); shared files keep only the content-overlap signal |
| `--keep-agreed` | population | off | **do not** drop the 37 annotator-agreed pairs (see below) |

### Agreed-pair exclusion (train/test disjointness)

By **default the population build drops the annotator-agreed gold pairs** — the 37
pairs both reviewers labelled the same. Those pairs *train* the prompt (they are
distilled into `insights.md`), so judging them would leak train into test. The
exclusion happens **after** positional `pair_id` numbering, so remaining ids stay
stable (gaps are fine), and it matches by `(p_url, q_url)` — gold and population use
separate `pair_id` spaces. `run_judge`'s auto-build inherits this default (there is
no `judge.yaml` key to keep them); pass **`--keep-agreed`** on a manual build only
if you deliberately want the full, non-disjoint pool.

---

## A. Build the LEARNED INSIGHTS

Distil all 37 annotator-agreed pairs into the frozen `insights.md` block the prompt
uses. One LLM call (needs `ANTHROPIC_API_KEY`); `--dry-run` and `show` need no key.

```sh
# inspect what will be sent (offline): counts + the full distillation prompt
uv run python -m prfix.rq1_fix_rates.judge.prompt insights build --dry-run

# distil + freeze insights.md  (default distiller: claude-opus-4-8, effort low)
uv run python -m prfix.rq1_fix_rates.judge.prompt insights build

# distil with a different model / effort (both optional; either can be given alone)
uv run python -m prfix.rq1_fix_rates.judge.prompt insights build --model claude-sonnet-4-6 --effort medium
# --effort "" (empty) sends no effort param — the model default applies

# print the current artifact
uv run python -m prfix.rq1_fix_rates.judge.prompt insights show
```

| `insights` command | what it does |
|---|---|
| `insights build` | distil the agreed pairs → overwrite `insights.md` (one live LLM call) |
| `insights build --dry-run` | print the exact distillation prompt + pair counts; **no** API call, no write |
| `insights build --model <id>` | override the distiller model (default `claude-opus-4-8`) |
| `insights build --effort <lvl>` | override the effort (default `low`; `""` omits the param) |
| `insights show` | print the current `insights.md` (no key needed) |

The written `insights.md` carries a provenance header recording the model, effort,
and agreed-pair count it was distilled from, so the frozen artifact is
self-documenting.

> **Note — the committed `insights.md` was distilled at `--effort xhigh`**, not the
> CLI default `low` (see its provenance header). These are two different knobs: the
> *distiller* effort is a one-off prompt-building call, while the *judge rater*
> effort (`anthropic:claude-opus-4-8:low`) applies per-pair at runtime. Re-running
> `insights build` with defaults would therefore produce a different artifact than
> the committed one — pass `--effort xhigh` to reproduce its settings.

`insights.md` is committed so the prompt is reproducible without re-calling the
model — regenerate it deliberately (e.g. after the agreed labels change). If it is
absent, the prompt falls back to RUBRIC + anchors only and `run_judge` warns.

---

## B. Run the judge over the net-diff population

The production path: one config (`judge.yaml`) defines **both** the net-diff set
to build **and** how to judge it, so the builder and the runner can never drift —
the cases filename is *derived* from `patch_cap` / `judgeable_only` / `p_side`.

```sh
# offline: build cases + print prompt sizes + a sample case (no API key)
uv run python -m prfix.rq1_fix_rates.judge.run_judge --config src/prfix/rq1_fix_rates/judge/judge.yaml --dry-run

# smoke test: a handful of real pairs
uv run python -m prfix.rq1_fix_rates.judge.run_judge --config src/prfix/rq1_fix_rates/judge/judge.yaml --limit 5

# full population run (reuses the committed cases file; auto_build: false)
uv run python -m prfix.rq1_fix_rates.judge.run_judge --config src/prfix/rq1_fix_rates/judge/judge.yaml
```

(Without `--config` the runner uses its built-in defaults for everything except
`raters`, which must come from the config or `--raters`. Explicit CLI flags
always override the config file.)

### Resumability

Each rater streams results to an append-only checkpoint
`data/judge/judge_pop[_humanP]_<provider>_<modelslug>[_<effort>].jsonl` — one JSON
line per completed pair, flushed as it lands. The `_humanP` segment appears when
the cases are human-P ones, keeping the two cohorts' runs in separate files
(`pair_id`s overlap between them). On
restart the runner **skips `pair_id`s already in that file**, so stopping mid-run
(Ctrl-C, crash, rate-limit) and rerunning the same command resumes with zero
duplicate API calls.

**The published runs' checkpoints are committed** under `data/judge/`
(`judge_pop_anthropic_claude-opus-4-8_low.jsonl` for the agent cohort and
`judge_pop_humanP_anthropic_claude-opus-4-8_low.jsonl` for the human-P
baseline): since LLM calls are not re-runnable deterministically, these files
*are* the published result — a fresh clone that runs the documented command
makes zero API calls and reproduces the CSV byte-exact. Only `--restart`
discards them.

```sh
# rebuild the CSVs + agreement report from existing checkpoints — no API calls
uv run python -m prfix.rq1_fix_rates.judge.run_judge --config src/prfix/rq1_fix_rates/judge/judge.yaml --consolidate-only

# ignore + delete checkpoints and judge from scratch
uv run python -m prfix.rq1_fix_rates.judge.run_judge --config src/prfix/rq1_fix_rates/judge/judge.yaml --restart

# re-judge ONLY the pairs whose checkpoint row is ERROR (e.g. earlier timeouts); keep the rest
uv run python -m prfix.rq1_fix_rates.judge.run_judge --config src/prfix/rq1_fix_rates/judge/judge.yaml --retry-errors
```

### Outputs

* `data/judge/rq1_llm_judge_pop[_humanP]_<provider>_<modelslug>[_<effort>].csv` — one
  per rater.
* Console: a **label distribution** per rater, the **model↔model agreement** κ
  (3-way / strict / lenient) on the pairs both raters judged, and a list of their
  **disagreements** — the seeds for a human audit.

### `judge.yaml` reference

Any key may be omitted (the built-in default is used). Equivalent `.json` / `.toml`
files also work, and keys may be written with dashes or underscores
(`patch-cap` ≡ `patch_cap`). Unknown keys are rejected, so a typo fails loudly.

| key | default | what it does |
|---|---|---|
| `patch_cap` | `12000` | chars per file patch (truncated beyond). **Also names the derived cases file** (12000 → `..._population_12k.json`). |
| `judgeable_only` | `true` | keep only pairs with a real net-diff overlap (`n_shared > 0`). |
| `auto_build` | `true` | build the derived cases file if missing before judging. (The shipped `judge.yaml` sets `false` so a reproduction run fails loudly rather than regenerating the pinned cases; `true` is only the built-in fallback when the key is omitted.) |
| `rebuild` | `false` | force-rebuild the population cases file first (picks up new build-time signals, e.g. blame/intervening-PR counts). Refreshes the cases; to actually re-judge you also need `restart`. Population mode only. |
| `raters` | — (**required**) | list of `provider:model[:effort]`. Each writes its own checkpoint + CSV. Set it here or via `--raters`; the model id is passed through to your deployment as-is (`openai:*` raters need an OpenAI key/gateway). |
| `concurrency` | `8` | parallel judge requests per rater. (The shipped `judge.yaml` sets `10` — the value the published run used; `8` is only the built-in fallback when the key is omitted.) |
| `limit` | `null` | judge only the first N cases (smoke test). |
| `restart` | `false` | ignore + delete existing checkpoints. |
| `retry_errors` | `false` | re-judge only the pairs whose checkpoint row is `ERROR` (e.g. earlier timeouts); keeps every other result and physically drops the stale ERROR lines. |
| `consolidate_only` | `false` | rebuild CSVs + report from checkpoints; no API. |
| `dry_run` | `false` | print prompt sizes + a sample case and exit; no API. |
| `cases` | `null` | explicit cases JSON path; overrides the derived path. |
| `p_side` | `agent` | P cohort: `agent` (the published run) or `human` (the incidence baseline). Names the derived cases file (`..._humanP.json`). The checkpoint/CSV identity is taken from the cases, not from this key — see [Human-P incidence baseline](#human-p-incidence-baseline). |

Every config key has a matching CLI flag (`--patch-cap`, `--judgeable-only`,
`--no-auto-build`, `--rebuild`, `--raters`, `--concurrency`, `--limit`,
`--restart`, `--retry-errors`, `--consolidate-only`, `--dry-run`, `--cases`,
`--p-side`); the flag wins over the config.

### Raters

A rater spec is `provider:model[:effort]` — the model is required, and the id is
passed through to your deployment as-is (a real Anthropic/OpenAI model id against
the public APIs, or whatever alias your gateway exposes):

```yaml
raters:
  - anthropic:claude-opus-4-8:low     # the published production rater
  - anthropic:claude-sonnet-4-6       # second judge → its own CSV, gives a model↔model κ
  # - openai:gpt-5.4:low              # OpenAI rater (needs OPENAI_API_KEY + an OpenAI-compatible gateway)
```

* **`anthropic:<model>[:effort]`** — Claude via forced tool-use. Set
  `ANTHROPIC_BASE_URL` (env or `.env`) to route through a custom
  proxy/gateway; leave it unset for the public Anthropic API. `effort` =
  `output_config.effort` (see [Effort](#effort)); omit it to send no effort param.
* **`openai:<model>[:effort]`** — GPT via Chat Completions function-calling.
  `effort` = `reasoning_effort` (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`);
  with no effort suffix the param is omitted (the model default applies — an
  explicit `""` or `default` suffix likewise omits it). Needs
  `OPENAI_API_KEY`; set `OPENAI_BASE_URL` to route through an OpenAI-compatible
  gateway.

The effort suffix is **part of the checkpoint/CSV filename**, so listing the same
model at several efforts A/B-tests them on the same pairs without clobbering.

---

## C. Sample the population for a human-vs-judge check

Once a population CSV exists, draw a **blind** human sample from it to spot-check the
judge at scale (beyond the 50 gold pairs). Two steps:

The sampler lives with the rest of the human-annotation tooling
(`human-annotation/sample_pairs.py judge`), so drawing → labelling → scoring all
happen in that folder:

```sh
# 1. draw the sample (writes to human-annotation/labels/judge/)
uv run python human-annotation/sample_pairs.py judge --mode random --n 100 \
    --pop-csv data/judge/rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv \
    --annotator "YOUR NAME"

# 2. build the offline HTML labeler for that sample (--n / --sample-mode must match)
uv run python human-annotation/gen_labeler.py --mode judge --n 100 --sample-mode random
```

`sample_pairs.py judge` writes three artifacts per sample: a **blind sheet** (human
labels here, blind to both the judge's call and the sampling stratum), a
**model-key CSV** (the judge's decision, held separately), and a **meta JSON**
(per-stratum population vs sampled counts + reweighting factors). Three modes:

* `random` — uniform over the judged (non-ERROR) pairs.
* `stratified` — proportional 3-way allocation across the judge's labels.
* `enriched` — oversample the judge's `Direct fix` calls (`--pos-frac`), camouflaged
  with proportionally-drawn decoys, to tighten the Direct-fix precision estimate
  without anchoring the annotator; the meta file carries the weights to reweight
  back to population rates.

`sample_pairs.py judge` flags:

| flag | default | what it does |
|---|---|---|
| `--mode {random,stratified,enriched}` | `random` | draw strategy (above) |
| `--pop-csv PATH` | the Opus-4.8-low population CSV | which judged population to sample from |
| `--n N` | `50` | sample size |
| `--seed N` | `42` | RNG seed — same seed + CSV + mode reproduces the exact draw |
| `--pos-frac F` | `0.5` | [enriched] target fraction of `Direct fix` positives |
| `--exclude PATH…` | none | one or more prior sample CSVs whose `pair_id`s are held out (draw a fresh, non-overlapping sample) |
| `--annotator NAME` | `""` | prefill the annotator column and output filename |
| `--tag NAME` | `""` | suffix the sample stem so a second draw over a different judged population never collides with an earlier one (e.g. `humanP`) |

`gen_labeler.py --mode judge` flags — **`--n` and `--sample-mode` must match the draw**
(they name the model-key CSV it reads): `--n` (default 50), `--sample-mode`
(`random`/`stratified`/`enriched`), `--tag` (must match the draw's `--tag`), and
`--cases` (the population cases JSON with the net-diff patches the judge saw;
defaults to the agent-cohort population file, so a `--tag`ged non-agent sample
must name its own).

`gen_labeler.py --mode judge` renders the sample's net-diff shared files (the exact
context the judge saw, plus the `[P-overlap]` note and provenance warning) into a
self-contained HTML page: label each pair, then "Reveal judge" to compare, and
export an annotator sheet.

---

## Effort

`--effort` (CLI) / the effort suffix (raters) sets the Anthropic
**`output_config.effort`** parameter, which trades response thoroughness for token
spend on a single model. It is GA — no beta header — and overrides the API default
of `high`. Levels, cheapest → most thorough: `low`, `medium`, `high`, `xhigh`, `max`.

**Model support** (the API rejects unsupported levels with a 400; the CLI prints a
warning first):

| level | supported on |
|---|---|
| `low` / `medium` / `high` | Opus 4.5/4.6/4.7/4.8, Sonnet 4.6, Fable 5 |
| `max` | Opus 4.6/4.7/4.8, Sonnet 4.6, Fable 5 (not Opus 4.5) |
| `xhigh` | Opus 4.7/4.8, Fable 5 only |

Effort is **not** supported on Sonnet 4.5 or Haiku 4.5 at all. For a judging task
like this, `low`/`medium` is usually plenty (the question is narrow and the rubric
does the heavy lifting); reach for `xhigh`/`max` only if you see the judge stumbling
on the fuzzy `Related ↔ Unrelated` boundary.

> **Temperature note.** On models that still accept sampling params (Sonnet 4.6,
> Haiku) the judge sends `temperature=0` for reproducible labels. Opus 4.7+/Fable
> removed sampling params and **400 on `temperature`** — so on those the call omits
> it automatically (effort, not temperature, is the depth control there). OpenAI
> reasoning models (GPT-5.4) likewise reject `temperature` and are steered with
> `reasoning_effort`, so they are only *near*-deterministic — that asymmetry is
> intentional. `--dry-run` prints exactly what each model will be sent.

## Keys & prerequisites

* **`ANTHROPIC_API_KEY`** — required for any Anthropic rater. Put it in `.env`
  (git-ignored) as `ANTHROPIC_API_KEY=sk-ant-...`, or export it.
* **`OPENAI_API_KEY`** — required only if a rater is `openai:*`. Same `.env`.
* **`gh`** — must be authenticated; used to fetch net-diff patches when building
  case files (cached into the frozen snapshot afterward).
* `--dry-run` and case-building alone need no API key.
* **Parallelism:** judge requests run in a thread pool (built-in default 8; the
  shipped `judge.yaml` sets 10); the client is thread-safe. Raise with `--concurrency`,
  mind your rate limits. The population runner fires one warm-up call before fanning
  out so the cached system prompt is written first (avoids a cache stampede).

## Files

| file | role |
|---|---|
| `build_cases.py` | one builder, two modes: `build_gold()` (fixed 50-pair set) and `build_population()` (config-sized net-diff pool + blame attribution); `gh` diffs (frozen to a snapshot) + DuckDB descriptions/commits. `population_cases_path()` derives the cases filename from `patch_cap` (and `p_side`). |
| `prompt.py` | the whole system-prompt layer. **Read side:** `RUBRIC`, `JUDGE_TOOL` schema, `coerce_label`, `truncate`, `render_case` (with the `[P-overlap]` attribution note + provenance banner), `pick_exemplars` (worked-example anchors), `load_insights`, `build_system_prompt` (RUBRIC + insights + anchors). **Write side:** `distil`/`build_insights` distil ALL annotator-agreed pairs (+ reason notes) into the frozen `insights.md` LEARNED INSIGHTS block via one LLM call — CLI `insights build`/`show` (`--dry-run` offline; reuses `engine.complete_text`) |
| `engine.py` | the model-calling engine: `complete_judgment`/`judge_case` (forced tool-use), `complete_text` (plain completion, used by `prompt`'s insights distiller), `load_cases`, `_kappa`/`_binary` (population agreement), **plus** the provider-agnostic raters (`AnthropicRater`, `OpenAIRater`, `make_rater`) |
| `run_judge.py` | resumable multi-rater run over the net-diff population; checkpoints → CSV + model↔model agreement κ; reads `judge.yaml`; imports the primitives from `engine.py` |
| `attribution.py` | the `[P-overlap]` note, both methods: **Method A** content-overlap + cosmetic-only signals from the two patches (no network; `is_code`, `file_provenance`, `pair_provenance`) and **Method B** `git blame` at Q's merge-base (ground truth) via `gh`, cached to snapshots. Pure diff/blame logic is unit-tested (`tests/test_blame_attribution.py`) |
| *(judge sampler)* | the blind human-vs-judge draw now lives in `human-annotation/sample_pairs.py judge` (random / stratified / enriched → blind sheet + held-back model key + reweighting meta), alongside the labeler and scorer — not in this package |
| `judge.yaml` | one config for the whole population pipeline: net-diff build params (`patch_cap`, `judgeable_only`) **and** rater/run params. The cases file is derived from `patch_cap` so build + judge never drift |
| `README.md` | this file |

The HTML labeler for a drawn sample lives alongside the human-annotation inputs:
`human-annotation/gen_labeler.py --mode judge` renders the sample's diffs into an
offline blind-first labeler (judge call hidden behind a "Reveal judge" button),
written to `human-annotation/data/judge/`.

**The population pair set** (`build_cases.PAIR_SQL`): P = a merged AI-agent PR in a
repo with ≥500 stars; Q = a later merged `fix`-typed (or revert-titled) PR in the
same repo, merged within 30 days of P and sharing at least one changed file. Q is
drawn from **both** the agent table (`pull_request`) **and** the contemporaneous
human table (`human_pull_request`) — a regression an agent merged is often fixed
by a *human*, so the pool includes human fixers (mirroring the RQ1 behavioural
linker). Agent-Q pairs are pre-gated on the published `pr_commit_details` overlap
(so their `pair_id`s stay stable); human-Q pairs have no such table and are gated
only by the live `gh` net-diff. `build_population` materializes the net-diff
(`P ∩ Q`) for each pair; `judgeable_only` keeps the ones with a non-empty overlap
(~2.5k: ~1.8k agent-Q + ~0.7k human-Q). With `--p-side human` the P side comes
from `build_cases.PAIR_SQL_HUMAN_P` instead (P = merged human-cohort PRs in the
same repos, no commit-details pre-gate on either Q branch).

Inputs (read-only) live in `human-annotation/` — the consensus labels (`labels/humans/`),
the master metadata, and `data/net_diff/pairs_enriched.json` (the reviewed net-diff
the annotators labeled against). The `data/rq1_pairs_diff_snapshot.json` P/Q diff cache is a
**local, regenerable** artifact (refetched from GitHub on demand), so it is
git-ignored rather than versioned. Reproducibility for external reviewers is served
by the **published validation sample** (the reviewed enriched set), the
**published judged-population outputs** (2k+ pairs), and the **committed
exact-reproduction pins** — the case files and the production checkpoints in
`data/judge/` (see §A0 and Resumability) — not by the raw diff snapshot.
Results land in `data/judge/`.

## Validating the judge

> **⚠ Methodology changed.** The judge's prompt is now built from **all 37
> annotator-agreed pairs** (distilled insights + anchors), so there is no in-prompt
> held-out gold set — scoring the judge on those pairs would leak. The old
> gold-scoring path (accuracy / κ / third-annotator on held-out consensus
> pairs) has been **removed**. Any previously-reported gold
> numbers predate this change *and* the blame-attribution prompt; do not quote them.

Validate the judge **independently**, on pairs it was not trained on, via a blind
human-vs-judge sample of the population:

```sh
# 1. draw a blind sample from a population CSV (writes to human-annotation/labels/judge/)
uv run python human-annotation/sample_pairs.py judge --mode enriched --n 50 \
    --pop-csv data/judge/rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv --annotator "YOU"
# 2. render the blind labeler, label it, then score human vs judge
uv run python human-annotation/gen_labeler.py --mode judge --n 50 --sample-mode enriched
uv run python human-annotation/score.py judge
```

Guidance from prior runs still holds qualitatively: use the judge as a *first-pass
is-fix filter*, not a final 3-way labeler; disagreements cluster on the fuzzy
`Related ↔ Unrelated` boundary. The population run's model↔model disagreement list
surfaces exactly those for audit (add a second model to `raters` — each writes its
own CSV).

## Scaling further

The population run already pushes the validated judge over the full net-diff pool
(~2.5k judgeable pairs at the default cap) with prompt caching on the constant
system prompt and resumable checkpoints. To go substantially bigger (e.g. the full
file-overlap candidate pool, `linker.SQL_FOLLOWUP_PAIRS`, ~4–5k pairs)
without the per-request thread-pool overhead, dispatch via the Anthropic **Message
Batches API** (still reusing the cached system prompt).

## Caveat — not "no paid API calls"

Unlike the rest of the pipeline, the live judge **calls the Anthropic / OpenAI APIs**
(via LiteLLM; costs money). `--dry-run`, `--consolidate-only`, and case-building do not.

## Human-P incidence baseline

Runs the same linker + judge over the CONTEMPORANEOUS HUMAN cohort as the P
side, answering "do human merges get follow-up fixes at the same rate?".
Three long-running steps (run them yourself; all resumable):

```bash
# 1. net-diff filenames for the ~3.1k uncached human-P pair PRs (~30-60 min, gh API)
uv run python -m prfix.common.loaders.fetch_netdiff --p-side human
uv run python -m prfix.common.loaders.build_net_files          # rebuild pr_net_files

# 2. linker rates (fast, after step 1)
uv run python -c "from prfix.rq1_fix_rates import linker as rb; \
  d = rb.compute(p_side='human'); u = d[d.censored_30d==0]; \
  print(len(u), u.has_followup_30d.mean())"

# 3. judge cases + judge run (gh diff fetch, then API spend; own checkpoint
#    judge_pop_humanP_*.jsonl -- can never resume/clobber the agent run)
uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode population \
  --judgeable-only --patch-cap 12000 --p-side human
uv run python -m prfix.rq1_fix_rates.judge.run_judge \
  --config src/prfix/rq1_fix_rates/judge/judge.yaml --p-side human
```

NOTE `p_side` may also be set in the config (`p_side: human`); the `--p-side`
flag above wins over it. The checkpoint/CSV identity is taken from the CASES
THEMSELVES (each case carries `p_is_human`; older case files fall back to the
`_humanP` file-name convention), not from `p_side`, so an explicit
`--cases <..._humanP.json>` lands on `judge_pop_humanP_*` too and can never
touch the agent-P pins. Cases that disagree with `p_side` abort the run rather
than judging one cohort into the other's checkpoint.

Run record (completed): 2,206 judged cases (agent-P run: 2,510) at ≈$223 with
Opus 4.8 @ low; label mix 65.2% Unrelated / 27.5% Related / 7.1% Direct, 4
indeterminate. The pre-gate pair volume is larger than the agent side because
human P's have no commit-details pre-gate, but the net-diff gate removes more of
it, so the bill came in slightly *below* the agent run rather than above it.
The judge's precision was validated on agent-P pairs and confirmed on the human
side by a blind spot-check of its 30 Direct calls (27/30 = 90.0%, identical to
the agent-side value): see
`human-annotation/labels/judge/rq1_judge_sample50_enriched_humanP_score.txt`.
