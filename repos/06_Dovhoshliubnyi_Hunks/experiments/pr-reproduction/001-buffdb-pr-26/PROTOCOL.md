# Strict protocol for subsequent PR reproductions

This protocol supersedes the weaker isolation used in pilot 001.

## 1. Pre-register eligibility and selection

Before drawing a PR, define executable eligibility criteria. At minimum:

- merged AI-agent performance PR;
- exact pre-PR revision is retrievable;
- dependencies can be restored in a container;
- a runnable existing benchmark or independently constructible workload exists;
- correctness can be checked with targeted tests; and
- required external services or proprietary data are absent.

Create the eligible list before random selection. Use and store a deterministic
random seed. Never silently redraw; record every rejected draw and reason.

## 2. Construct a task packet

Use only information available to the historical agent before it produced the
patch:

- linked issue/ticket at its historical state;
- repository files at the pre-PR base revision;
- documented build and test instructions; and
- original user instruction when it is available.

Do not use the historical PR body if it was authored after the solution and
describes the implementation. Do not include the PR diff, commits, reviews,
benchmark results produced by the patch, generated reports, or later repository
state. Record every included input verbatim.

## 3. Prepare an isolated checkout

- Extract the exact base revision into a new directory.
- Remove `.git`, remotes, branches, tags, and historical objects.
- Initialise a synthetic repository containing only the base snapshot.
- Keep the historical head and patch in a separate evaluator-only directory.
- Do not place oracle identifiers or outputs in the agent directory.

## 4. Spawn without inherited context

Use:

```text
fork_turns: "none"
model: gpt-5.6-sol
reasoning_effort: xhigh
```

The complete task packet must be supplied in the spawn message. Explicitly
prohibit network, parent-repository, benchmark-oracle, and historical-diff
access. Record the exact spawn message and returned report.

## 5. Freeze the candidate

When the agent finishes:

- record elapsed agent time if available;
- save `git diff --binary` before evaluator edits;
- save `git status --short` and diff statistics;
- do not format or repair the candidate before its first validation; and
- separately record evaluator repairs if any are needed.

## 6. Validate correctness

Run identical build and targeted test commands on baseline, historical patch,
and generated patch. Distinguish:

- candidate-caused failures;
- failures reproduced on the baseline; and
- environment/infrastructure failures.

Treat a non-compiling or behaviourally incorrect patch as an unsuccessful
attempt. Do not benchmark it as a valid solution.

## 7. Benchmark fairly

- Use one benchmark source across all variants.
- Build each variant in a distinct target/output directory.
- Confirm each resulting binary corresponds to the intended source hash.
- Use release builds and the same toolchain/container/feature flags.
- Warm up before measurement.
- Randomise or counterbalance variant execution order.
- Use multiple process-level repetitions.
- Save every raw observation, not only summaries.
- Measure task-relevant resources such as time and allocations separately.
- Report uncertainty and paired effect sizes.

Never infer runtime improvement solely from static allocation counts.

## 8. Report both effectiveness and cost

For every model attempt record:

- build/test success;
- performance relative to baseline and historical patch;
- allocation or memory change where relevant;
- patch size and files touched;
- public API changes;
- model, reasoning effort, retries, and tool access;
- agent runtime and inference cost when available; and
- leakage, benchmark, and external-validity threats.

