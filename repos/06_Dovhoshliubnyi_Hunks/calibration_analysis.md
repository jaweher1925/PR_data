# Calibration Analysis — Round 1

**Date:** 2026-03-27
**Sample:** 40 random patches (excl. Mochi), classified by `claude-haiku-4-5-20251001`
**Results file:** `calibration_results.csv`

## Accuracy Summary

| Verdict | Count | % |
|---------|-------|---|
| Correct | 10 | 25% |
| Partially correct | 25 | 62.5% |
| Incorrect | 5 | 12.5% |

"Partially correct" means at least one category was right but others were wrong or missing.

## Issue 1: Category 8 (inlining) systematically misapplied

**Frequency:** ~10/40 patches had incorrect `inlining` labels.

**Problem:** The classifier applies "inlining" to any code restructuring that moves logic between functions and call sites, regardless of direction. In most cases the actual transformation was the **opposite** — method extraction, variable extraction, or delegation to a library.

**Examples:**
- Patch 12: Extracting repeated logic into `addCountParam()` function → labeled inlining (should be extraction)
- Patch 13: Extracting repeated matrix operations into intermediate variables → labeled inlining
- Patch 20: Extracting duplicated tray-update code into `updateTrayRunningTime()` → labeled inlining
- Patch 24: Replacing inline validation with `ValidateUniqueExtension()` helper → labeled inlining
- Patch 37: Replacing manual notification logic with DiffUtil library → labeled inlining

**Fix:** Add explicit guidance in the prompt: "Inlining means replacing a function/method call with the function's body (expanding it at the call site). The reverse — extracting inline code into a new function — is NOT inlining."

## Issue 2: Missing category 7 (type_change) when types clearly change

**Frequency:** ~5 patches where type_change should have been assigned but wasn't.

**Problem:** When a type swap also changes the name (e.g. `AHashMap` → `FxHashMap`), the classifier catches `name_modification` but misses that the underlying data type changed.

**Examples:**
- Patch 25: Removing `.to_string()` changes argument type from String to integer — labeled as arithmetic_manipulation instead of type_change
- Patch 31: `String` accumulation → `StringBuilder` + `FixedArray` — type_change missing
- Patch 35: `List<T>` → `List<T>?` (nullable) — type_change missing
- Patch 40: `std::vector<char>` → `PooledBuffer` — type_change missing

**Fix:** Add prompt guidance: "When the data type of a variable, parameter, return value, or collection changes (even if the name also changes), always include type_change."

## Issue 3: Category 16 (arithmetic_manipulation) over-applied

**Frequency:** ~4 patches with questionable arithmetic_manipulation labels.

**Problem:** The classifier treats any expression-level change as "arithmetic manipulation," including null-coalescing operators, Yoda conditions, and method call argument changes that have nothing to do with arithmetic or boolean logic.

**Examples:**
- Patch 5: `fileName.equals("empty.tmp")` → `"empty.tmp".equals(fileName)` — Yoda condition, not arithmetic
- Patch 35: `??=` null-coalescing assignment — not arithmetic
- Patch 7: Database membership check → set membership check — not arithmetic

**Fix:** Narrow the prompt definition: "Arithmetic manipulation means changing mathematical operators (+, -, *, /), comparison operators (<, >, ==), boolean operators (&&, ||, !), or numeric literal values. Reordering method call arguments or adding null checks is NOT arithmetic manipulation."

## Issue 4: Whitespace/formatting changes misclassified

**Frequency:** 2 patches.

**Problem:** Trailing whitespace removal or formatting-only changes are classified as `comment_modification` instead of `no_change`.

**Examples:**
- Patch 8: Trailing spaces removed from blank lines → labeled comment_modification
- Patch 18: BOM marker change → labeled comment_modification

**Fix:** Add: "Whitespace-only changes (trailing spaces, blank line additions/removals, indentation changes, BOM markers) are no_change, not comment_modification."

## Issue 5: Build/config files don't fit categories

**Frequency:** 3 patches.

**Problem:** Changes to `Cargo.toml`, `.xlf` localization files, and similar non-code files get forced into categories like `added_code_from_github` or `object_creation` that don't apply.

**Examples:**
- Patch 33: Adding a dependency line to Cargo.toml → labeled added_code_from_github
- Patch 38: Changing dependency specification format → labeled object_creation

**Fix:** Add: "For build configuration files (Cargo.toml, package.json, .csproj, etc.) that only add/remove dependencies or change versions, use no_change. These are not source code mutations."

## Round 1 → Round 2 Prompt Fixes Applied

1. Category 8 (inlining): clarified direction — "replacing a function CALL with the function's body, NOT the reverse"
2. Category 16 (arithmetic_manipulation): narrowed — "NOT: reordering method arguments, null-coalescing, or method call substitution"
3. Category 7 (type_change): added rule — "ALWAYS include type_change when data type changes even if name_modification also applies"
4. Whitespace-only → no_change rule added
5. Build config files → no_change rule added

---

# Calibration Analysis — Round 2

**Date:** 2026-03-27
**Sample:** Same 40 patches (seed=42, excl. Mochi), classified by `claude-haiku-4-5-20251001`
**Results file:** `calibration_results_round2.csv`

## Accuracy Summary

| Verdict | Count | % | Change from R1 |
|---------|-------|---|-----------------|
| Correct | 15 | 37.5% | +12.5% |
| Partially correct | 20 | 50% | -12.5% |
| Incorrect | 5 | 12.5% | same |

Improvement was meaningful but did not hit the >60% correct target.

## Issue 1 (persists): Category 16 (arithmetic_manipulation) still over-applied

**Frequency:** ~8–10/40 patches still have incorrect or questionable `arithmetic_manipulation` labels. This is the single biggest accuracy issue.

**Problem:** Despite the narrower definition, the classifier still applies arithmetic_manipulation to any expression-level change, including things explicitly excluded by the prompt.

**Examples:**
- Patch 5: `fileName.equals("empty.tmp")` → `"empty.tmp".equals(fileName)` — argument reordering, explicitly excluded
- Patch 7: `DB.sismember(...)` → `other in pair_cache[...]` — method call substitution, explicitly excluded
- Patch 12: `UNION` → `UNION ALL` — SQL keyword change, not an operator change
- Patch 21: Single → double quote style changes — not arithmetic
- Patch 25: Removing `.to_string()` calls — method call removal, not arithmetic
- Patch 35: `??=` null-coalescing assignment — explicitly excluded by prompt definition

**Round 3 fix:** Even stricter definition: "An arithmetic, comparison, or boolean OPERATOR must be added, removed, or changed, OR a numeric/boolean LITERAL must be changed. NOT: adding/removing function arguments, null-coalescing (??/??=), method call substitution, quote style changes, or reordering method arguments."

## Issue 2 (persists, improved): Category 8 (inlining) still confused with extraction

**Frequency:** 3/40 patches (down from ~10 in Round 1).

**Problem:** The classifier now acknowledges the distinction in its reasoning (Patch 24 says "this is the reverse of inlining") but still assigns the label anyway.

**Examples:**
- Patch 2: Replacing inline code with calls to new helper functions → labeled inlining (should be deleted_blocks + added_code)
- Patch 20: Creating `updateTrayRunningTime()` and replacing inline code with calls to it → labeled inlining (this is extraction)
- Patch 24: Classifier's own reasoning says "this is the reverse of inlining" yet still assigns inlining(0.9)

**Round 3 fix:** Add explicit anti-rule: "Do NOT confuse extraction with inlining: if code is moved FROM inline INTO a new named function, that is NOT category 8."

## Issue 3 (persists, improved): Category 7 (type_change) over-applied to library swaps

**Frequency:** ~4/40 patches.

**Problem:** The classifier now correctly catches type_change for real type changes (HashMap→FxHashMap, Array→Vector, List→List?). However, it also applies type_change to library/dependency swaps where the API is identical and no declared types change (chalk→picocolors).

**Examples:**
- Patches 1, 11, 28: `chalk` → `picocolors` labeled as type_change — same API, no declared type changes, this is name_modification only
- Patch 18: `.ToArray()` → `[..]` collection expression syntax — result type stays `T[]`, not a type change

**Round 3 fix:** Clarify: "Swapping one library for another with the same API (e.g. chalk→picocolors) is name_modification, NOT type_change, unless the actual type signature changes."

## Issue 4 (new): Category 5 (return_statement_mods) hallucinated

**Frequency:** 2/40 patches.

**Problem:** The classifier assigns return_statement_mods to patches where no return statements are added, removed, or modified.

**Examples:**
- Patch 16 (ggml_interface.cpp): OpenMP→thread pool refactoring — no return statements changed, yet labeled return_statement_mods(0.8)
- Patch 20 (PiloramaTimer.qml): Method extraction — no return statements changed, yet labeled return_statement_mods(0.8)

**Round 3 fix:** Add rule: "Do NOT use category 5 unless a return/yield keyword is explicitly added, removed, or its returned expression is directly modified." Tighten category definition.

## Issue 5 (new): Category 14 (object_creation) applied to variable caching

**Frequency:** ~2/40 patches.

**Problem:** Assigning intermediate variables to cache computed values (e.g. `obj_matrix_world = obj.matrix_world`) is labeled as object_creation when it's really statement_splitting.

**Examples:**
- Patch 13: Caching `obj.matrix_world` into a local variable → labeled object_creation (should be statement_splitting)

**Round 3 fix:** Clarify category 14: "Caching a value into a local variable is statement_splitting, not object_creation."

## Issue 6 (new): Missing comment_modification labels

**Frequency:** ~3/40 patches.

**Problem:** When comments are added/removed alongside code changes, the classifier sometimes omits comment_modification.

**Examples:**
- Patch 34: Comment "Directly echo the value with fallback" was deleted — not flagged
- Patch 37: Two comments deleted, one added — not flagged

**Round 3 fix:** No prompt change needed — this is a minor omission; the classifier catches it in most cases. Sonnet 4.6 should handle this better due to higher overall accuracy.

## Per-patch verdicts (Round 2)

| # | Filename | Classifier categories | Verdict | Issue |
|---|----------|-----------------------|---------|-------|
| 1 | connect.ts | 12,7 | PARTIAL | type_change wrong (library swap) |
| 2 | op_resize.py | 3,8,12,2 | PARTIAL | inlining wrong (extraction) |
| 3 | TestHostOrchestratorManager.cs | 3,13,14,16 | PARTIAL | arithmetic_manipulation wrong |
| 4 | EFFICIENCY_REPORT.md | 1 | PARTIAL | should be added_code_from_github |
| 5 | TempFileCleanupServiceTest.java | 16 | INCORRECT | argument reordering ≠ arithmetic |
| 6 | alpha_detection.py | 16 | PARTIAL | adding function arg ≠ arithmetic |
| 7 | collectors.py | 2,14,16 | PARTIAL | method call substitution ≠ arithmetic |
| 8 | CommandLineOptionsValidator.cs | 1 | CORRECT | |
| 9 | get.handler.ts | 0,15 | PARTIAL | statement_splitting wrong |
| 10 | extendeddatacrossword.go | 8 | INCORRECT | method call swap ≠ inlining |
| 11 | cmd.ts | 7,12 | PARTIAL | type_change wrong (library swap) |
| 12 | get.handler.ts | 14,12,16 | PARTIAL | UNION→UNION ALL ≠ arithmetic |
| 13 | handler.py | 14,15,16 | PARTIAL | variable caching ≠ object_creation |
| 14 | FocusZoneStories.module.css | 3 | CORRECT | |
| 15 | incremental.rs | 7 | CORRECT | |
| 16 | ggml_interface.cpp | 2,3,5,13,14,16 | PARTIAL | return_statement_mods hallucinated |
| 17 | MonoSignal.as | 7,13,16,15,2 | PARTIAL | missing object_creation |
| 18 | ExtensionValidationHelper.cs | 7,2 | PARTIAL | type_change wrong (syntax sugar) |
| 19 | ActionList.stories.tsx | 3,16 | PARTIAL | expression removal ≠ arithmetic |
| 20 | PiloramaTimer.qml | 8,5 | INCORRECT | extraction ≠ inlining; no return mods |
| 21 | piclumen.py | 15,16 | PARTIAL | quote changes ≠ arithmetic |
| 22 | isJsonRpcRequestPayload.spec.ts | 16,7 | PARTIAL | type assertion ≠ arithmetic |
| 23 | perfetto.rs | 7,12 | CORRECT | |
| 24 | TestHostControllersManager.cs | 3,8 | PARTIAL | extraction ≠ inlining |
| 25 | full_features_demo.rs | 16,12 | INCORRECT | removing .to_string() ≠ arithmetic |
| 26 | rf05.rs | 7,12,14 | CORRECT | |
| 27 | replace-title-with-inline.js | 0 | CORRECT | |
| 28 | request-promote.ts | 12,7 | CORRECT | (borderline on type_change) |
| 29 | env-target.ts | 12 | CORRECT | |
| 30 | MEMORY_ANALYSIS_FIX.md | 1 | INCORRECT | 112-line new file ≠ no_change |
| 31 | bigint_nonjs.mbt | 14,7,13,3,16 | CORRECT | |
| 32 | getUserAvailability.ts | 0,2,13,14 | CORRECT | |
| 33 | Cargo.toml | 1 | CORRECT | |
| 34 | env.sh | 13,16 | PARTIAL | arithmetic stretch; missing comment_mod |
| 35 | ValidationEndpointFilterFactory.cs | 7,14,16 | PARTIAL | null-coalescing ≠ arithmetic |
| 36 | util.ts | 3,13,0 | CORRECT | |
| 37 | GenericAdapter.kt | 3,13,14 | PARTIAL | missing comment_modification |
| 38 | Cargo.toml | 1 | CORRECT | |
| 39 | rollback/status.ts | 12 | CORRECT | |
| 40 | ggml_interface.cpp | 0,7,11,14 | CORRECT | |

## Round 2 → Round 3 Fixes Applied

1. Model switched from `claude-haiku-4-5-20251001` to `claude-sonnet-4-6`
2. Category 16 (arithmetic_manipulation): even stricter — "NOT: adding/removing function arguments, null-coalescing (??/??=), method call substitution, quote style changes"
3. Category 8 (inlining): added explicit anti-rule — "If code moves FROM a call site INTO a new function, that is extraction, not inlining"
4. Category 7 (type_change): clarified — "library swap with same API is name_modification, NOT type_change, unless actual type signature changes"
5. Category 5 (return_statement_mods): tightened — "Do NOT use unless a return/yield keyword is explicitly added/removed/changed"
6. Category 14 (object_creation): clarified — "Caching a value into a local variable is statement_splitting, not object_creation"
7. Documentation-only files (.md) → no_change rule added

---

# Calibration Analysis — Round 3

**Date:** 2026-03-27
**Sample:** Same 40 patches (seed=42, excl. Mochi), classified by `claude-sonnet-4-6`
**Results file:** `calibration_results_round3.csv`

## Accuracy Summary

| Verdict | Count | % | Change from R2 |
|---------|-------|---|-----------------|
| Correct | 16 | 42% | +4.5% |
| Partially correct | 17 | 45% | -5% |
| Incorrect | 5 | 13% | same |

(1 patch dropped due to API error, 39 reviewed)

## Cross-round Comparison

| Verdict | R1 (Haiku) | R2 (Haiku+fixes) | R3 (Sonnet+fixes) |
|---------|------------|-------------------|---------------------|
| Correct | 10 (25%) | 15 (37.5%) | 16 (42%) |
| Partial | 25 (62.5%) | 20 (50%) | 17 (45%) |
| Incorrect | 5 (12.5%) | 5 (12.5%) | 5 (13%) |

## What improved in Round 3

- **Library swap type_change fixed:** chalk→picocolors now correctly classified as name_modification only (patches 1, 11, 26, 27)
- **Category 16 significantly reduced:** No longer applied to null-coalescing, quote style changes, or method call substitutions
- **Category 5 hallucination mostly fixed:** No false return_statement_mods in R3
- **Inlining awareness improved:** Patch 22 gives inlining only 0.1 confidence (was 0.9 in R2)

## Remaining issues

### Issue 1: Category 6 (method_name_change) over-applied (NEW in R3)

**Frequency:** 5/39 patches with incorrect method_name_change labels.

**Problem:** Sonnet applies method_name_change when methods are deleted and replaced by entirely different APIs, or when commands are swapped between conditions. These are not renames.

**Examples:**
- Patch 2: Old functions deleted, new functions added → labeled method_name_change (should be deleted_blocks)
- Patch 7: `DB.sismember(...)` replaced with `other in pair_cache[...]` → labeled method_name_change (call substitution)
- Patch 22: Inline code extracted to helper method → labeled method_name_change (extraction)
- Patch 32: tput/stty commands swapped between if-branches → labeled method_name_change
- Patch 35: `notifyDataSetChanged()` → `dispatchUpdatesTo()` → labeled method_name_change (different API)

### Issue 2: Category 12 (name_modification) over-applied

**Frequency:** ~5/39 patches with incorrect name_modification labels.

**Problem:** Applied when no actual name change occurs.

**Examples:**
- Patch 14: `@layer` CSS wrapper removed → labeled name_modification (should be deleted_blocks)
- Patch 18: `clsx(...)` wrapper removed → labeled name_modification (should be deleted_blocks)
- Patch 20: Type cast added `null as unknown as undefined` → labeled name_modification (should be type_change only)
- Patch 23: `.to_string()` removed → labeled name_modification (should be type_change)

### Issue 3: Category 16 (arithmetic_manipulation) still leaks

**Frequency:** ~3/39 patches.

**Examples:**
- Patch 5: `.equals()` Yoda condition still labeled arithmetic_manipulation(0.8)
- Patch 12: `UNION` → `UNION ALL` labeled arithmetic_manipulation(0.8)
- Patch 32: Shell parameter expansion change labeled arithmetic_manipulation(0.7)

### Issue 4: Build config no_change too aggressive

**Frequency:** 2/39 patches.

**Problem:** Cargo.toml dependency swaps (ahash→rustc-hash) labeled no_change when real dependencies change.

## Confidence threshold analysis

Most wrong labels cluster at lower confidence while correct labels are high-confidence:

| Threshold | Labels kept | Wrong labels | Error rate |
|-----------|------------|--------------|------------|
| None | 97 | ~21 | ~22% |
| >= 0.8 | 83 (86%) | ~14 | ~17% |
| >= 0.9 | 50 (52%) | ~2 | ~4% |

**Decision:** Apply MIN_CONFIDENCE = 0.8 threshold to filter low-confidence noise. This removes 7 wrong labels while keeping 86% of all labels.

## Round 3 → Round 4 Fixes Applied

1. Added `MIN_CONFIDENCE = 0.8` threshold in `_parse_response()` to filter low-confidence labels
2. Changed calibration seed to 123 (fresh 40 patches to avoid overfitting to the same sample)

---

# Calibration Analysis — Round 4

**Date:** 2026-03-27
**Sample:** 40 fresh patches (seed=123, excl. Mochi), classified by `claude-sonnet-4-6` with MIN_CONFIDENCE=0.8
**Results file:** `calibration_results.csv`

## Accuracy Summary

| Verdict | Count | % | Change from R3 |
|---------|-------|---|-----------------|
| Correct | 26 | 67% | +25% |
| Partially correct | 8 | 20% | -25% |
| Incorrect | 5 | 13% | same |

**Target met:** >60% correct, <25% partial.

## Cross-round Comparison

| Verdict | R1 (Haiku) | R2 (Haiku+fixes) | R3 (Sonnet) | R4 (Sonnet+threshold) |
|---------|------------|-------------------|-------------|------------------------|
| Correct | 10 (25%) | 15 (37.5%) | 16 (42%) | 26 (67%) |
| Partial | 25 (62.5%) | 20 (50%) | 17 (45%) | 8 (20%) |
| Incorrect | 5 (12.5%) | 5 (12.5%) | 5 (13%) | 5 (13%) |

The confidence threshold was the single biggest improvement — eliminated noisy low-confidence wrong labels.

## Remaining edge cases (5 incorrect)

1. **Patch 6:** Primitive field declarations (`int batchSize = 0`) labeled as object_creation — should be no_change or no category
2. **Patch 9:** New import statement labeled as name_modification — should be no_change
3. **Patch 26:** Aliased library swap (`FxHashSet as HashSet`) labeled as type_change — should be name_modification per rules
4. **Patch 31:** Optional chaining `?.` and `|| null` labeled as arithmetic_manipulation — should be control_flow (null-coalescing excluded)
5. **Patch 38:** Method extraction (duplicate code elimination) labeled as deleted_blocks — should be duplicate_code

## Per-patch verdicts (Round 4)

| # | Filename | Classifier categories | Verdict | Issue |
|---|----------|-----------------------|---------|-------|
| 1 | tailwind.config.ts | 3 | CORRECT | |
| 2 | settings.yml.template | 1 | CORRECT | |
| 3 | FocusZoneStories.module.css | 3 | CORRECT | |
| 4 | cache.ts | 2,14 | CORRECT | |
| 5 | apps.server.ts | 2,13,14,15,16 | PARTIAL | missing return_statement_mods |
| 6 | ApplicationProperties.java | 14 | INCORRECT | primitive fields ≠ object_creation |
| 7 | notify.js | 2,15,14 | CORRECT | |
| 8 | middleware.ts | 1 | CORRECT | |
| 9 | slots.controller.ts | 12 | INCORRECT | new import ≠ name_modification |
| 10 | interop.rs | 1 | CORRECT | |
| 11 | ScannerController.php | 3,13,15 | CORRECT | |
| 12 | Unsafe.java | 6,12 | PARTIAL | method_name_change debatable |
| 13 | postcss.config.js | 3 | CORRECT | |
| 14 | Cargo.toml | 1 | CORRECT | |
| 15 | BuildEventArgs_Tests.cs | 2,16 | PARTIAL | string literal "0" ≠ numeric literal |
| 16 | Result.cs | 13,14,7 | PARTIAL | type_change wrong (runtime check); missing return_mods |
| 17 | index.rs | 1 | CORRECT | |
| 18 | app.js | 14,15 | CORRECT | |
| 19 | format-environments.ts | 12 | CORRECT | |
| 20 | agent_experience_entrypoint.py | 9 | CORRECT | |
| 21 | noncode.rs | 7,12 | PARTIAL | type_change wrong (library swap) |
| 22 | array.jl | 6,13,5,15 | CORRECT | |
| 23 | stubs.ts | 0 | CORRECT | |
| 24 | members/page.tsx | 12,14,15 | CORRECT | |
| 25 | ActionList.stories.tsx | 12 | CORRECT | |
| 26 | lib.rs | 7 | INCORRECT | aliased library swap ≠ type_change |
| 27 | SipMetrics.java | 0 | CORRECT | |
| 28 | fonts.ts | 3,2,14 | PARTIAL | object_creation wrong |
| 29 | ValidationEndpointFilterFactory.cs | 14,13,7 | PARTIAL | object_creation debatable |
| 30 | generate.go | 2 | CORRECT | |
| 31 | TeamsListing.tsx | 16 | INCORRECT | null-coalescing ≠ arithmetic |
| 32 | getBusyTimesFromLimits.ts | 6,12 | CORRECT | |
| 33 | datasquare_test.go | 6 | CORRECT | |
| 34 | fragmentcommands.ts | 12 | CORRECT | |
| 35 | useAnchoredPosition.stories.tsx | 7 | CORRECT | |
| 36 | errors-ts.ts | 12 | CORRECT | |
| 37 | CommandLineOptionsValidator.cs | 16 | PARTIAL | missed dead_code (unused import) |
| 38 | GivenAGenerate...Tests.cs | 3 | INCORRECT | extraction ≠ deleted_blocks |
| 39 | run/index.ts | 12,7,6,13,9,2,14 | CORRECT | |

## Conclusion

The combination of Sonnet 4.6 + improved prompt + MIN_CONFIDENCE=0.8 threshold achieves 67% fully correct classification with only 13% incorrect. The remaining errors are edge cases that are unlikely to significantly bias aggregate statistics across ~1,666 patches.

**Next step:** Compare with Haiku + same prompt + threshold to determine if the more expensive Sonnet model is necessary, or if Haiku with the improved prompt and threshold achieves comparable results.

---

# Calibration Analysis — Round 5 (Haiku vs Sonnet Comparison)

**Date:** 2026-03-27
**Sample:** 37 patches (seed=123, excl. Mochi), classified by `claude-haiku-4-5-20251001` with MIN_CONFIDENCE=0.8
**Results file:** `calibration_results.csv`
**Purpose:** Determine if Haiku can match Sonnet accuracy to save cost on the ~1,666-patch full run

## Notes on Sample Alignment

Both runs used seed=123, but 36 patches overlap (1 Haiku-only: `rpkiv.ch`; 3 Sonnet-only: `apps.server.ts`, `datasquare_test.go`, `run/index.ts`). Comparison is made on the 36 common patches using the R4 manual review verdicts as ground truth.

## Accuracy Summary (36 shared patches)

| Verdict | Sonnet R4 | Haiku R5 | Delta |
|---------|-----------|----------|-------|
| **Correct** | **24 (67%)** | **19 (53%)** | **-14%** |
| Partially correct | 7 (19%) | 10 (28%) | +9% |
| Incorrect | 5 (14%) | 7 (19%) | +5% |

## Where Haiku is worse (7 patches degraded vs Sonnet)

| Patch | Haiku output | Sonnet output | Issue |
|-------|-------------|---------------|-------|
| notify.js | 15,2,**16**,14 | 2,15,14 | Over-applies arithmetic_manipulation |
| Cargo.toml | **7,12** | 1:no_change | Ignores build config → no_change rule |
| index.rs | **15:statement_splitting** | 1:no_change | Reformatting misclassified as statement_splitting |
| agent_experience.py | **16:arith_manip** | 9:exception_handling | Completely wrong — exception type narrowing ≠ arithmetic |
| array.jl | 13,15,**8:inlining** | 6,13,5,15 | Wrong inlining; misses method_name_change and return_mods |
| members/page.tsx | 12,**7,8**,15 | 12,14,15 | Hallucinated type_change and inlining |
| getBusyTimes.ts | 6,12,**14:obj_creation** | 6,12 | Spurious object_creation label |

## Where Haiku is better (2 patches improved vs Sonnet)

| Patch | Haiku output | Sonnet output | Why better |
|-------|-------------|---------------|------------|
| slots.controller.ts | 0:added_code | 12:name_mod | New import = added code (more accurate) |
| fonts.ts | 3,2 | 3,2,~~14~~ | Avoided Sonnet's spurious object_creation |

## Root causes of Haiku's lower accuracy

1. **arithmetic_manipulation over-application** — still the persistent Haiku weakness. `except ImportError` classified as arithmetic (completely wrong); `?.` and `is null` checks still caught as arithmetic despite prompt exclusions.
2. **Inlining misapplied in 3 patches** — Haiku still confuses extraction with inlining (array.jl, members/page.tsx) and applies inlining where Sonnet correctly picks method_name_change.
3. **Build config and formatting rules not reliably followed** — Cargo.toml treated as type_change; reformatting-only (index.rs) treated as statement_splitting.

## Cross-round comparison (all rounds)

| Verdict | R1 (Haiku) | R2 (Haiku+fixes) | R3 (Sonnet) | R4 (Sonnet+threshold) | R5 (Haiku+threshold) |
|---------|------------|-------------------|-------------|------------------------|----------------------|
| Correct | 10 (25%) | 15 (37.5%) | 16 (42%) | 24/36 (67%) | 19/36 (53%) |
| Partial | 25 (62.5%) | 20 (50%) | 17 (45%) | 7/36 (19%) | 10/36 (28%) |
| Incorrect | 5 (12.5%) | 5 (12.5%) | 5 (13%) | 5/36 (14%) | 7/36 (19%) |

## Decision: Use Sonnet for full run

Haiku is 14 percentage points below Sonnet on correct rate (53% vs 67%) and has a higher incorrect rate (19% vs 14%). The same prompt improvements and threshold do not bridge the gap — Haiku's weaker instruction-following causes systematic misfires on arithmetic_manipulation and inlining that Sonnet reliably avoids.

**Next step:** Re-run calibration with Sonnet and a new seed (≠42, ≠123) to confirm R4 accuracy generalises to a third sample before the full run.

---

# Calibration Analysis — Round 6 (Sonnet, seed=777, confirmation run)

**Date:** 2026-03-27
**Sample:** 40 fresh patches (seed=777, excl. Mochi), classified by `claude-sonnet-4-6` with MIN_CONFIDENCE=0.8
**Results file:** `calibration_results.csv`
**Review file:** `calibration_review_r6.csv`
**Purpose:** Confirm R4 accuracy (67%) generalises to a third independent sample

## Accuracy Summary

| Verdict | Count | % | vs R4 |
|---------|-------|---|-------|
| **Correct** | **27** | **67.5%** | **+0.5%** |
| Partially correct | 9 | 22.5% | +2.5% |
| Incorrect | 4 | 10% | -3% |

**Target confirmed:** 67.5% correct, 10% incorrect — consistent with R4.

## Per-patch verdicts (Round 6)

| # | Filename | Classifier categories | Verdict | Issue |
|---|----------|-----------------------|---------|-------|
| 1 | Result.cs | 5 | CORRECT | |
| 2 | build.gradle (d5520d00) | 1 | CORRECT | |
| 3 | build.gradle (febf5019) | 1 | CORRECT | |
| 4 | jsonCompare.ts | 6,12 | CORRECT | |
| 5 | PERFORMANCE_ANALYSIS_REPORT.md | 1 | CORRECT | |
| 6 | ConvertImgPDFController.java | 1 | CORRECT | |
| 7 | engine.go | 3,2,16 | PARTIAL | Missing 7:type_change for new bool param on RunSessionWithHook |
| 8 | slots-worker.service.ts | 2,7,12,14,0 | PARTIAL | 14:object_creation unsupported — new Worker() existed in original |
| 9 | ThreadMessages.tsx | 1 | CORRECT | |
| 10 | chatparticipant.ts | 12 | CORRECT | |
| 11 | fetch-types.ts | 0 | CORRECT | |
| 12 | EFFICIENCY_REPORT.md | 1 | INCORRECT | New .md file is 0:added_code_from_github; doc-only rule covers existing-file modifications |
| 13 | extension.py | 1 | CORRECT | |
| 14 | data_collectors/__init__.py | 0 | CORRECT | |
| 15 | rspack_tracing/Cargo.toml | 1 | CORRECT | |
| 16 | sliding_logits.py | 3,0,6,12,14,13 | CORRECT | |
| 17 | GenerateDepsFile.cs | 14,15,2,17 | PARTIAL | 17:dead_code wrong — new XxHash64() instance is immediately used |
| 18 | op_resize.py | 1 | CORRECT | |
| 19 | inspect/index.ts | 12 | CORRECT | |
| 20 | workflow_orchestrator_test.go | 0 | CORRECT | |
| 21 | CHANGELOG.md | 1 | CORRECT | |
| 22 | tests/tests.rs (0f04c8bd) | 3 | CORRECT | |
| 23 | get.handler.ts | 12,4 | PARTIAL | 4:duplicate_code is a stretch; missing 0 or 14 for new extended-params block |
| 24 | test_benchmark.py | 2 | INCORRECT | New pytest.param entries are 0:added_code_from_github; inline comments are incidental |
| 25 | integration/add.ts | 12 | CORRECT | |
| 26 | go.mod | 1 | CORRECT | |
| 27 | ResultTValueTests.cs | 5,7,15 | PARTIAL | Missing 0:added_code_from_github for new test method in first hunk |
| 28 | BehaviorCollection.cs | 14,16,2 | CORRECT | |
| 29 | codegen_result.rs | 7,12 | PARTIAL | 12 borderline — no call-site renames, only import path; rules cover it but 7 is primary |
| 30 | TempFileCleanupServiceTest.java | 6,7,2,12 | PARTIAL | All four correct; missing 0:added_code_from_github for new directoryStreamOf helper |
| 31 | tests/tests.rs (e2759c20) | 2 | INCORRECT | New test function with assertions is 0:added_code_from_github; comments are incidental |
| 32 | deploy/index.ts | 12 | CORRECT | |
| 33 | diff-env-files.ts | 12 | CORRECT | |
| 34 | infer_tests.rs | 7,12,14 | PARTIAL | 14:object_creation wrong — constructor change is part of type/name substitution |
| 35 | gcloud/util_test.go | 2 | PARTIAL | //go:build directive controls compilation — 13:control_flow more accurate than 2:comment_modification |
| 36 | showUI.spec.ts | 12 | CORRECT | |
| 37 | GivenAGenerateDepsFile.cs | 15 | INCORRECT | Code de-duplication via extraction is 4:duplicate_code not 15:statement_splitting |
| 38 | AppManager.tsx | 12,15,14 | CORRECT | |
| 39 | raw_html.rs | 7 | CORRECT | |
| 40 | TYGENT_INTEGRATION.md | 1 | CORRECT | |

## Remaining error patterns (4 incorrect)

1. **Patch 12 (EFFICIENCY_REPORT.md):** New .md file classified as no_change — the doc-only → no_change rule is over-applied to brand new files (should be added_code_from_github).
2. **Patch 24 (test_benchmark.py):** New pytest.param entries classified as comment_modification — comments inside added code are mistaken for the primary change.
3. **Patch 31 (tests/tests.rs):** New test function classified as comment_modification — same pattern as patch 24.
4. **Patch 37 (GivenAGenerateDepsFile.cs):** Code extraction/de-duplication classified as statement_splitting — persistent confusion between extraction and statement reorganisation.

## Cross-round comparison (all rounds)

| Verdict | R1 (Haiku) | R2 (Haiku+fixes) | R3 (Sonnet) | R4 (Sonnet+threshold) | R5 (Haiku) | R6 (Sonnet, seed=777) |
|---------|------------|-------------------|-------------|------------------------|------------|------------------------|
| Correct | 10 (25%) | 15 (37.5%) | 16 (42%) | 26/40 (67%) | 19/36 (53%) | 27/40 (67.5%) |
| Partial | 25 (62.5%) | 20 (50%) | 17 (45%) | 8/40 (20%) | 10/36 (28%) | 9/40 (22.5%) |
| Incorrect | 5 (12.5%) | 5 (12.5%) | 5 (13%) | 5/40 (13%) | 7/36 (19%) | 4/40 (10%) |

## Conclusion

R6 replicates R4 accuracy on a completely independent sample (seed=777 vs seed=123):
- 67.5% vs 67% correct — within noise
- 10% vs 13% incorrect — slightly better

**The classifier is stable.** Proceed with the full run: `python pipeline.py --skip-load --classify-patches`
