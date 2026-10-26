# Validation record

## Focused generated-patch test

Toolchain container: `rust:1.88-slim-bookworm`

```sh
cargo test --lib --no-default-features --features sqlite index::tests
```

Result:

```text
running 4 tests
test index::tests::test_hash_index_basic ... ok
test index::tests::test_btree_index_range_queries ... ok
test index::tests::test_unique_constraint ... ok
test index::tests::test_manager_updates_and_removes_multiple_indexes ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out
```

The release benchmark builds also prove that all three variants compiled under
the benchmark feature configuration.

## Static checks

`git diff --check` passed for the generated tracked patch.

`rustfmt --check --edition 2021 src/index.rs` reported two generated formatting
differences: one expression wrapping change near the shared value construction
and one test-call wrapping change. Repository-wide format checking also reports
pre-existing formatting differences in unrelated files.

## Non-gating failures

An attempted complete library run reported failures in two FTS tests and an
MVCC test, then hung and was terminated. This pilot did not repeat that complete
run on the baseline, so no causal attribution is made.

An attempted integration build exposed missing protobuf fields, unavailable
feature-gated backend types, and malformed integration-module documentation.
Those files were not changed by the candidate. The focused index tests are the
only correctness gate used for this pilot.

## Benchmark commands

All variants used equivalent commands with distinct source and target mounts:

```sh
cargo build --quiet --release --example index_time \
  --no-default-features --features sqlite
/target/release/examples/index_time  # repeated five times

cargo run --quiet --release --example index_perf \
  --no-default-features --features sqlite
```

The first attempted comparison reused a common Cargo target path while every
variant was mounted at `/work`. Cargo consequently reused the baseline package
artifact. Those measurements were discarded. Final measurements use separate
target directories for baseline, historical, and generated variants.

