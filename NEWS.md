# bacchuss 0.2.4

This changelog was introduced in October 2026. Earlier changes and releases are not documented here.

## Added

- Added optional parallel annotation to `briseus()`, `isodaetes()`, and `liknites()` through `parallel = TRUE`, using the existing `bacchuss_lyaeus()` function. Sequential execution remains the default. In `isodaetes()`, both initial annotations and jury calls can run in parallel; in `liknites()`, repeated annotations of each held-out example can run concurrently.
- Exposed `parallel_calls`, `future_strategy`, and `show_progress` in these functions to configure worker counts, execution strategy, and progress reporting. `show_progress` also controls the progress bar during sequential `liknites()` runs.
- Added per-observation vector-seed support to `bacchuss()`, `bacchuss_lyaeus()`, `briseus()`, and `isodaetes()`, ensuring consistent seed assignment in sequential and parallel execution. Added per-example vector-seed support to `liknites()`. Existing scalar-seed behavior is preserved, and seed validation is now shared across the annotation functions.

## Changed

- Standardized the default `max_tokens` value to `2048` across `bacchuss_satyr()`, `bacchuss()`, `bacchuss_lyaeus()`, `briseus()`, `isodaetes()`, and `liknites()`. Previously, all except `bacchuss_lyaeus()` defaulted to `NULL`. Calls that omit this argument now request an explicit response-token cap, which may affect annotations that produce long outputs. Set `max_tokens = NULL` explicitly to omit the cap from the request (subject to backend defaults).
- `isodaetes()` now sends observations to the jury when agreement is **at or below** `agreement_threshold`, rather than only below it. At the default threshold of `0.5`, observations with exactly 50% agreement now receive jury adjudication. This can change their final labels compared with earlier versions.
- Improved parallel execution in `bacchuss_lyaeus()`: worker counts are capped at the number of input rows, and warm-up calls retain the corresponding observation's seed.

## Fixed

- Protected sequential annotation and warm-up inputs from being overwritten by generated output columns with the same names.
- Preserved data-frame structure when warm-up input contains only one column.
- Made `isodaetes()` jury prompts use the original input text and copy original context estimates into a dedicated jury-context column.
- Made `isodaetes()` select its initial `final_explanation` from a run that assigned the majority label, rather than always using the first run's explanation.
- Corrected handling of `explanations = TRUE` in `liknites()` leave-one-out runs.
- Strengthened `n_runs` validation in `briseus()`, `isodaetes()`, and `liknites()` to reject non-integer, non-finite, and out-of-range values. Added validation that `parallel` is a single non-missing logical value and strengthened `parallel_calls` validation in `bacchuss_lyaeus()`.
- Corrected documentation for plaintext `Label:` output, `liknites()` return-column names, the `isodaetes()` jury threshold, and function descriptions.
