# Recovered regression harnesses

44 test harnesses rescued on 2026-09-04 from the per-session scratch directories under
`C:\Users\kerr\AppData\Local\Temp\claude\...`, where they had been written and then left.
The development notes in `handoff/` cite many of them by name as the evidence for claims
the audit will check — "validated headless, `verify_x.py`" — with `verify_x.py` existing
nowhere durable. They were one temp cleanup away from gone.

**They are committed here exactly as found: unmodified, unadapted, and not yet run.**
That is deliberate — this commit is the provenance record. Expect a good fraction to fail
until they are fixed up, and expect that to be informative rather than alarming.

Fixing them up is Phase 0 of `handoff/12-work-package-preaudit.md`.

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
