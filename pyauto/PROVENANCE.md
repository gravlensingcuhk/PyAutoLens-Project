# Vendored PyAuto packages

These are copies of Sam's PyAuto development branches (the pyauto/ codespace), vendored here so
the `uv` environment of this project can install them as path packages. Only the importable
package, `pyproject.toml`, `setup.py`, `README.md`, `LICENSE` and `MANIFEST.in` are copied
(no tests, docs or `.git`). The version each `setup.py` stamps is the branch's base release tag
plus a `.postN` for commits beyond it, so the intra-family `>=` floors stay satisfied.

| Package     | Branch                        | Commit                                     | Version            |
|-------------|-------------------------------|--------------------------------------------|--------------------|
| autoarray   | `production/hst`              | `4bc7b5a5` (2026-11-02; trim-to-mask, cherry-picked from `feature/trim-to-mask` `a7f9fbc7`) | 2026.10.23.1.post6 |
| autofit     | `production/hst`              | `production/hst` head, 2026-10-26 (lock patch committed upstream) | 2026.10.23.1.post1 |
| autogalaxy  | `feature/elliptical-multipole`| `b1f4cabfe4276bebcd2c64cf22dcaebf10ff8d7b` | 2026.10.16.1.post5 |

`autolens` and `autonerves` (the renamed `autoconf`) are on `main` in the codespace and are
pinned to their PyPI releases in `pyproject.toml` instead.

Exported 2026-10-25 with `git archive <branch> <package> pyproject.toml setup.py README.md LICENSE MANIFEST.in`.
The committed branch state was copied. autofit was re-exported on 2026-10-26 after its running-search
lock / read-only archive replay change (`c9264d112`) was committed.

Why these branches: `production/hst` autoarray carries the `pixel_signal_normalisation: weights`
option (continuous adaptive-regularization signals) and the RTU rectangular meshes; autogalaxy's
branch carries `PowerLawEllipticalMultipole`. See `slam_pipeline/source_pix.py`.

To refresh: re-run the export from the codespace and update this table.

Lock patch on autofit (2026-10-26, committed on `production/hst` and `feature/paths-running-lock-replay` in the codespace, pushed to the fork):
`autofit/non_linear/paths/abstract.py` running-search lock made scheduler-agnostic. The holder
records host, pid, pid start time and a generic job id (`PYAUTO_JOB_ID`, SLURM, PBS, LSF, SGE or
the HTCondor job ad); liveness is decided by own-process / own-job, a pid probe on the same host,
a scheduler queue query when that client exists, else a heartbeat that a daemon thread refreshes
every 60 s while the lock is held (stale after 30 min). Before, only SLURM was queried and the
heartbeat was only touched on search updates (never, with `iterations_per_quick_update: 1e99`).

autoarray re-exported 2026-10-26 at `82cc4396`: adds the opt-in sparse curvature
(`Settings(use_sparse_curvature=True)` / `[inversion] use_sparse_curvature`, exact so identifiers are
unchanged, JAX backend only) with the transposed PSF blur in column blocks.

Trim-to-mask on autoarray (2026-11-02, `feature/trim-to-mask` `a7f9fbc7` on `origin/main`, cherry-picked onto
`production/hst` as `4bc7b5a5`; write-up `z_projects/pyauto_prs/pyautoarray/5_*`): `Imaging.apply_mask(trim_to_mask=...)`
and `Imaging.trimmed_to_mask_from` cut the native arrays to the mask plus a PSF half-width
(`mask_2d_util.trimmed_shape_for_kernel`); `general.yaml [dataset] trim_to_mask`, packaged false, `true` in this
project's `config/general.yaml`. Three source files re-copied from `production/hst`; the tests are not vendored.
