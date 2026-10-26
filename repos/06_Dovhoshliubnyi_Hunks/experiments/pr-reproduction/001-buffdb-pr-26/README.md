# PR reproduction pilot 001: buffdb/buffdb#26

## Status

This is a **preliminary, quasi-blind pilot**, not a strictly blinded experimental
observation. The coding agent was denied the historical patch and repository
history, but the agent was spawned with `fork_turns: "3"`. Consequently,
inheritance of recent conversational context cannot be excluded. The generated
solution is materially different from the historical patch, but that fact does
not prove absence of leakage.

Do not include this run in a confirmatory model-comparison dataset without
either marking it separately or repeating it under the strict protocol in
[`PROTOCOL.md`](PROTOCOL.md).

## Case

- Repository: `buffdb/buffdb`
- Pull request: <https://github.com/buffdb/buffdb/pull/26>
- Historical agent: Devin
- PR title: `Performance optimization: Reduce string allocations in indexing system`
- Pre-PR base: `9edcaacdd19daa6f7956633d38c0dcc1ac52889e`
- Historical head: `491d7cb487e2ecf87f2d558b8f22b05f2b254d8b`
- Historical merge commit: `873a9c2a146e0aa1ea2b34f99c0f37b3454ea0c8`
- New model: `gpt-5.6-sol`
- Reasoning effort: `xhigh`
- Run date: 2026-07-22
- Subagent task: `/root/buffdb_pr26_reproduction`

The initial uniform draw from 188 merged AI-agent rows selected
`epicweb-dev/restore-scroll#13`. It was rejected because it was a dependency
upgrade without a reproducible performance workload. A second draw was made
from 25 merged AI-agent PRs whose descriptions contained performance or
benchmark terms; it selected this PR. No deterministic random seed was stored,
which is another limitation of this pilot.

## Isolation actually used

The agent received:

1. An extracted source tree at the exact pre-PR base revision.
2. The task in [`agent-prompt.txt`](agent-prompt.txt).
3. A local Git repository containing only one synthetic baseline commit.

The target checkout had no remote or historical commits. The prompt explicitly
prohibited GitHub, web, PR, branch, commit-history, and parent-dataset access.
The accepted patch was extracted into a different directory only after the
agent had been launched.

The agent did **not** receive the accepted diff through its explicit task. It
did receive a task statement identifying secondary-index string allocations;
this information came from the ticket/PR description. Because the spawn used
three inherited turns, strict context isolation cannot be demonstrated.

## Patch comparison

The historical Devin patch changes `SecondaryIndex::insert` from accepting an
owned `String` to accepting `&str`, then creates a `String` at the storage site.
In the measured `IndexManager::update_indexes` path this moves the allocation
but does not remove it. The historical unique-index check additionally creates
a temporary `String` for `contains`.

The Sol patch instead:

- preserves the public `SecondaryIndex::insert(String, IndexValue)` API;
- adds private owned/shared key and value representations;
- shares keys and values with `Arc` when more than one index is updated;
- keeps an owned fast path for one index;
- moves removal lookup allocation outside the index loop;
- avoids cloning B-tree range bounds; and
- adds a test covering update and removal across hash and B-tree indexes.

Generated patch size: 217 insertions and 22 deletions in `src/index.rs`.

See [`patches/historical.patch`](patches/historical.patch) and
[`patches/sol-xhigh.patch`](patches/sol-xhigh.patch).

## Benchmark

The same public-API benchmark was applied to the pre-PR baseline, historical
head, and generated patch. Each workload:

- creates 1, 2, or 10 non-unique indexes, alternating hash and B-tree indexes;
- uses a 96-byte key and two 512-byte values;
- warms up with 10,000 alternating updates;
- measures 100,000 alternating remove-and-insert updates; and
- uses an isolated Cargo target directory for each variant.

Environment:

- official `rust:1.88-slim-bookworm` container;
- `protobuf-compiler` and `libsqlite3-dev` installed inside the disposable
  container;
- `--release --no-default-features --features sqlite`;
- five timing repetitions per variant and index count.

### Timing results

Mean nanoseconds per update; change is relative to the pre-PR baseline.

| Indexes | Baseline | Historical Devin | Devin change | Sol xhigh | Sol change |
|---:|---:|---:|---:|---:|---:|
| 1 | 587.26 | 597.50 | 1.74% slower | 592.64 | 0.92% slower |
| 2 | 852.41 | 850.81 | 0.19% faster | 819.46 | **3.86% faster** |
| 10 | 4,129.94 | 4,125.66 | 0.10% faster | 3,832.71 | **7.20% faster** |

Approximate 95% t-intervals for Sol do not overlap the baseline interval at 2
or 10 indexes. With only five repetitions and one synthetic workload, these
intervals are descriptive rather than a general performance claim.

### Allocation results

Allocation calls per update, measured with a counting global allocator:

| Indexes | Baseline | Historical Devin | Sol xhigh | Sol reduction |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 4 | 0% |
| 2 | 8 | 8 | 6 | 25% |
| 10 | 40 | 40 | 14 | 65% |

The allocation-instrumented timings are not used for the timing comparison.
Raw observations are in [`raw/timings.csv`](raw/timings.csv) and
[`raw/allocations.csv`](raw/allocations.csv).

## Validation

- The Sol patch compiled successfully with Rust 1.88.
- Four focused index tests passed: three existing tests and one generated test.
- `git diff --check` passed.
- File-specific `rustfmt --check` reported two formatting differences in the
  generated `src/index.rs`.
- The full library suite was not usable as a correctness gate: two FTS tests
  failed, an MVCC test failed, and the run later hung. These failures were not
  baseline-controlled in this pilot and therefore are not attributed either to
  the patch or to the baseline.
- The repository integration suite also contains unrelated feature/protobuf
  inconsistencies. Only the focused index result is treated as verified.

## Interpretation

For this workload, the historical patch neither reduced allocation count nor
measurably improved update time. The generated patch reduced allocations and
improved multi-index update time, but it is much larger and more invasive. It
did not improve the single-index case. This is encouraging evidence for the
research design, not evidence that newer models are generally better.

## Files

- [`agent-prompt.txt`](agent-prompt.txt): exact subagent instruction
- [`PROTOCOL.md`](PROTOCOL.md): strict procedure for subsequent PRs
- [`patches/historical.patch`](patches/historical.patch): accepted PR delta
- [`patches/sol-xhigh.patch`](patches/sol-xhigh.patch): generated source delta
- [`benchmark/index_time.rs`](benchmark/index_time.rs): uninstrumented timing
- [`benchmark/index_allocations.rs`](benchmark/index_allocations.rs): allocation counter
- [`raw/timings.csv`](raw/timings.csv): all timing repetitions
- [`raw/allocations.csv`](raw/allocations.csv): allocation measurements
- [`validation.md`](validation.md): commands, outcomes, and limitations

