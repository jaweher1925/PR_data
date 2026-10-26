# Performance-PR reproduction experiments

Each numbered directory is a self-contained reproduction record containing the
task packet, model configuration, leakage assessment, historical and generated
patches, benchmark source, raw observations, and validation evidence.

| ID | PR | Historical agent | Candidate | Blinding | Outcome |
|---|---|---|---|---|---|
| [`001-buffdb-pr-26`](001-buffdb-pr-26/) | `buffdb/buffdb#26` | Devin | `gpt-5.6-sol` xhigh | Preliminary/quasi-blind | Sol reduced multi-index update time by 3.9–7.2%; historical patch was neutral |

Pilot 001 must not be presented as strictly blinded because its subagent was
spawned with inherited conversational turns. All subsequent experiments should
follow [`001-buffdb-pr-26/PROTOCOL.md`](001-buffdb-pr-26/PROTOCOL.md), especially
`fork_turns: "none"`, deterministic selection, agent-visible input capture, and
isolated build directories.

