# Recovered regression harnesses

44 test harnesses rescued on 2026-09-04 from the per-session scratch directories under
`C:\Users\kerr\AppData\Local\Temp\claude\...`, where they had been written and then left.
The development notes in `handoff/` cite many of them by name as the evidence for claims
the audit will check — "validated headless, `verify_x.py`" — with `verify_x.py` existing
nowhere durable. They were one temp cleanup away from gone.

**They are committed here exactly as found: unmodified, unadapted, and not yet run.**
That is deliberate — this commit is the provenance record. Expect a good fraction to fail
until they are fixed up, and expect that to be informative rather than alarming.

Fixing them up is Phase 0 of `handoff/12-work-package-preaudit.md`. The provenance
commit is the one that added this directory; the adaptation (P-3b, mechanical edits only)
is recorded in the status table at the end of this file.

## Known reasons they will need work

- **Hardcoded absolute paths**, including scratch directories that no longer exist.
- **`Data/tim3_hmmm_ext_all_bestframe.pdb` was removed from the repo** on 2026-09-04
  (unpublished structure). Anything loading a PDB needs repointing at `Data/1ubq.pdb`,
  and note that fitted values in those scripts were obtained with the old structure, so
  numeric expectations tied to a specific fit result are no longer meaningful — the
  structural and mechanical assertions still are.
- **The `.par` fixtures moved** to `tests/fixtures/legacy_xlayerspdb_2024.par`.
- **Spawn `multiprocessing.Pool`** is reachable in several; each needs
  `if __name__ == '__main__':` or it re-imports itself and fork-bombs. Some already have
  it, some do not.
- They need the `cmc-py39` interpreter, `QT_QPA_PLATFORM=offscreen`, and a stubbed
  `xraydb`. See `handoff/09-validation-and-env.md`.
- Most define their own local `check()` that only prints and sets a flag — the same
  non-asserting pattern being fixed under work-package item P-1. They should move to the
  shared helper.

## What came from where

Full provenance, including the ~60 one-off diagnostics, benchmarks and probes that were
**not** brought across, is in the sibling repository
`../xmodjointfit-session-archive`, one directory per session id with an index.

| origin session | topic | harnesses here |
|---|---|---|
| `d63b5d0d` | env, GUI construction, PDB persistence | `test_gui_construct`, `test_gui_switch`, `test_pdb_persist`, `test_phase1` |
| `6f65f848` | PDB MCMC seeding, parallel sampling, pickling | `validate_parallel_seed`, `validate_parallel_sampling`, `compare_models`, `test_pickle_fit`, `test_logprob`, `test_joint_emcee`, `test_joint_parallel`, `test_joint_abort`, `test_autocorr_window` |
| `22bf29f6` | joint MCMC, orientation seeding across sharing configs | `verify_joint_orient`, `verify_joint_neither`, `verify_joint_mixed`, `verify_joint_seed_worker`, `verify_seed_helpers`, `verify_seedmode`, `verify_seed_gui`, `verify_reuse_wrap`, `verify_fingerprint`, `verify_jointmcmc`, `verify_jointwindow`, `repro_full`, `repro_switch`, `repro_backend` |
| `c0878594` | plotting, overlay, legend, detail switching | `verify_legend`, `verify_plotting_B`, `verify_banner_lifecycle`, `verify_detail_switch`, `verify_multifit_selection`, `verify_joint_detail_and_legend`, `verify_joint_parallel_sample`, `verify_joint_parallel_edge`, `verify_joint_4ds_parallel` |
| `7068116c` | GID2D, per-column fit limits | `smoke_fit`, `repro_raw`, `repro_tab` |
| `85efc5b2` | XPR Julia backend | `test_xpr_stage1`, `test_xpr_fitting`, `test_gui_listing`, `check_aproj` |
| `1fc38b3e` | upstream merge smoke test | `smoke_merge` |

## Deliberately excluded

- **`verify_joint_parallel_scan.py`** — asserts `sampler.pool is None`, i.e. that joint
  sampling is serial. That behaviour was deliberately flipped when parallel joint sampling
  landed, so the script now asserts the opposite of the intended design.
  `verify_joint_4ds_parallel.py` is its corrected replacement. Kept in the archive for the
  record.
- The one-off diagnostics: `chk_*`, `diag_*`, `probe*`, `shot*`, `t_axis*`, `t_grid`,
  `t_img`, `bench_*`, `explore_cov*`, `profile_seeding`, `timing_mcmc`, `worker_cost`,
  `measure`, `seed_split`, `try_loadpath`, `bootstrap_julia`, `tb_clean`. They record how
  a question was answered rather than guarding behaviour. Several are the source of
  measured numbers quoted in the handoff docs, so the archive matters for the audit's
  P3 items.
- The seeding study's own scripts, which went to `../xpr-seeding-study`.

## Status (P-3b, 2026-11-04)

Adapted by mechanical edits only (paths to `ROOT`-relative, `Data/1ubq.pdb`, `Data/i2_tim3_1000_test.txt`
in place of the removed tim3 inputs, `__main__` guards, `tests.harness`). Run the lot with
`python tests/run_all.py --dir tests/recovered` (add `--quick` to skip slow/external ones); the
registry is `REGISTRY` in `tests/run_all.py`.

Kinds: **assert** = fails through a bare `assert`/uncaught exception, form kept; **converted** =
computed a verdict but only printed it or kept it in a local flag, now on `tests.harness`
`check()`/`finish()` so the verdict reaches the exit code; **smoke** = asserts nothing, passes if it
exits 0 (registry `smoke: true`, no invented checks); **known** = expected to fail (registry
`known:` carries the reason).

| harness | batch | status | kind | note |
|---|---|---|---|---|
| `verify_joint_orient` | a | PASS | assert | |
| `verify_joint_neither` | a | PASS | assert | |
| `verify_joint_mixed` | a | PASS | assert | |
| `verify_joint_seed_worker` | a | PASS | assert | |
| `verify_seed_helpers` | a | PASS | assert | |
| `verify_seedmode` | a | PASS | assert | |
| `verify_seed_gui` | a | PASS | assert | |
| `verify_reuse_wrap` | a | PASS | assert | |
| `verify_fingerprint` | a | PASS | assert | |
| `verify_jointmcmc` | a | KNOWN-FAIL | assert | binds a fixed method list onto a `Fake`; `_draw_corner` now calls `_corner_render_data` (b953c3f), not in the list |
| `verify_jointwindow` | a | PASS | assert | |
| `repro_full` | a | PASS | assert | |
| `repro_switch` | a | PASS | assert | |
| `repro_backend` | a | PASS | smoke | prints backend state, no assertion |
| `validate_parallel_seed` | b | PASS | converted | parallel == serial, costs finite; tim3 minima numbers (chi2 148.9 / 157.2) no longer asserted |
| `validate_parallel_sampling` | b | PASS | converted | objective identity, finite chains, progress hook, abort; ~121 s |
| `compare_models` | b | PASS | smoke | no assertions (registered `smoke`); exercises `XLayersPDB_dev_smooth`, retired under D1; slow, timeout 900 |
| `test_pickle_fit` | b | PASS | smoke | prints that `Fit` does not pickle (the reason `PdbLogProb` exists) |
| `test_logprob` | b | PASS | converted | tim3 chi2 targets replaced by "lnprob finite and <= 0", out-of-bounds -> -inf, pickles |
| `test_joint_emcee` | b | PASS | converted | chain finite; ~102 s |
| `test_joint_parallel` | b | PASS | converted | `JointLogProb` == -0.5 sum(residual^2) (rel. 1e-6), chain finite |
| `test_joint_abort` | b | PASS | converted | each abort scenario must not raise; 343 s, slow, timeout 900 |
| `test_autocorr_window` | b | PASS | smoke | timing table only |
| `verify_legend` | c | PASS | converted | |
| `verify_plotting_B` | c | PASS | converted | |
| `verify_banner_lifecycle` | c | PASS | converted | |
| `verify_detail_switch` | c | PASS | converted | |
| `verify_multifit_selection` | c | PASS | converted | |
| `verify_joint_detail_and_legend` | c | PASS | converted | |
| `verify_joint_parallel_sample` | c | PASS | assert | |
| `verify_joint_parallel_edge` | c | PASS | assert | |
| `verify_joint_4ds_parallel` | c | PASS | assert | |
| `test_gui_construct` | d | PASS | smoke | needed the conditional `xraydb` stub; prints CONSTRUCT_OK |
| `test_gui_switch` | d | PASS | assert | |
| `test_pdb_persist` | d | PASS | converted | |
| `test_phase1` | d | PASS | converted | |
| `smoke_fit` | e | KNOWN-FAIL | smoke-shaped | reads `chi2_per_N_reported`; GID2D now emits `chi2_per_N_reported_LINEAR` (stale key names) |
| `repro_raw` | e | PASS | smoke | rebuilds the raw map in `tests/_logs`, no longer writes into `Data/` |
| `repro_tab` | e | PASS | smoke | |
| `test_xpr_stage1` | f | PASS | converted | `cmcars`, external `XPR-main/test_data`; ~44 s |
| `test_xpr_fitting` | f | PASS | converted | `cmcars`, external, slow, 232 s, timeout 900 |
| `test_gui_listing` | f | PASS | assert | `cmcars` |
| `check_aproj` | f | PASS | smoke | `cmcars`, external `XPR-main/test_data`; ratio table only |
| `smoke_merge` | g | PASS | converted | local `check(label, fn)` wrapper now feeds `tests.harness` |

The two KNOWN-FAIL entries (`verify_jointmcmc`, `smoke_fit`) are stale harnesses, not engine
defects; both are tracked as `handoff/10-fix-list.md` A1.17.
