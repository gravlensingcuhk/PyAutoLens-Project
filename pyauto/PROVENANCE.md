# Vendored PyAuto packages

These are copies of Sam's PyAuto development branches (the pyauto/ codespace), vendored here so
the `uv` environment of this project can install them as path packages. Only the importable
package, `pyproject.toml`, `setup.py`, `README.md`, `LICENSE` and `MANIFEST.in` are copied
(no tests, docs or `.git`). The version each `setup.py` stamps is the branch's base release tag
plus a `.postN` for commits beyond it, so the intra-family `>=` floors stay satisfied.

| Package     | Branch                        | Commit                                     | Version            |
|-------------|-------------------------------|--------------------------------------------|--------------------|
| autoarray   | `production/hst`              | `40ae61b47b36991bf021a4fd14cc9b804ec9ce2d` | 2026.10.23.1.post3 |
| autofit     | `production/hst`              | `d3a7721294652f81ec726b92931ab610ed27e3ac` | 2026.10.23.1       |
| autogalaxy  | `feature/elliptical-multipole`| `b1f4cabfe4276bebcd2c64cf22dcaebf10ff8d7b` | 2026.10.16.1.post5 |

`autolens` and `autonerves` (the renamed `autoconf`) are on `main` in the codespace and are
pinned to their PyPI releases in `pyproject.toml` instead.

Exported 2026-10-25 with `git archive <branch> <package> pyproject.toml setup.py README.md LICENSE MANIFEST.in`.
The committed branch state was copied; uncommitted working-tree edits in the codespace (PyAutoFit
had an in-progress running-lock change in `autofit/non_linear/paths/`) were not.

Why these branches: `production/hst` autoarray carries the `pixel_signal_normalisation: weights`
option (continuous adaptive-regularization signals) and the RTU rectangular meshes; autogalaxy's
branch carries `PowerLawEllipticalMultipole`. See `slam_pipeline/source_pix.py`.

To refresh: re-run the export from the codespace and update this table.
