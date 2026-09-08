"""
Combine Subhalo Tile Results
============================

Assemble the per-tile summary JSONs (written by ``main_subhalo.py``) into a
single delta-log-evidence map over the image plane.

    python combine_tiles.py COSJ100024+021749 F444W

Reads ``output/subhalo_tiles/<dataset>/<filter>/tile_###.json``, prints the
``n x n`` maps of delta-log-evidence and best-fit subhalo mass/centre, flags
any missing tiles, and saves a CSV.
"""

import json
import sys
import os
from os import path

import numpy as np


def main():
    dataset_name = str(sys.argv[1])
    filt = str(sys.argv[2]) if len(sys.argv) > 2 else "F444W"

    cosma_path = path.join(path.sep, "users", "wing-yan.chan", "PyAutoLens-Project")
    output_path = path.join(cosma_path, "output")
    summary_dir = path.join(output_path, "subhalo_tiles", dataset_name, filt)

    files = sorted(f for f in os.listdir(summary_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"No tile JSONs found in {summary_dir}")

    records = []
    for f in files:
        with open(path.join(summary_dir, f), "r") as fh:
            records.append(json.load(fh))

    # Determine grid size from the max tile index + iy/ix fields.
    n = int(np.sqrt(max(r["tile_index"] for r in records) + 1))
    expected = n * n

    delta = np.full((n, n), np.nan)
    mass_at_200 = np.full((n, n), np.nan)
    concentration = np.full((n, n), np.nan)
    kappa_s = np.full((n, n), np.nan)
    scale_radius = np.full((n, n), np.nan)
    best_y = np.full((n, n), np.nan)
    best_x = np.full((n, n), np.nan)
    loge_sub = np.full((n, n), np.nan)

    for r in records:
        iy, ix = r["tile"]["iy"], r["tile"]["ix"]
        delta[iy, ix] = r["delta_log_evidence"]
        mass_at_200[iy, ix] = r["best_fit_mass_at_200"]
        concentration[iy, ix] = r["best_fit_concentration"]
        kappa_s[iy, ix] = r["best_fit_kappa_s"]
        scale_radius[iy, ix] = r["best_fit_scale_radius"]
        best_y[iy, ix] = r["best_fit_centre_y"]
        best_x[iy, ix] = r["best_fit_centre_x"]
        loge_sub[iy, ix] = r["log_evidence_with_subhalo"]

    baseline = records[0]["log_evidence_no_subhalo"]

    print(f"\nDelta log-evidence map ({dataset_name}, {filt})  [rows=y, cols=x]")
    print(np.round(delta, 2))
    print("\nBest-fit subhalo M_200 (Msun) per tile:")
    print(np.array2string(mass_at_200, formatter={"float_kind": lambda v: f"{v:.2e}"}))
    print("\nBest-fit subhalo concentration per tile:")
    print(np.array2string(concentration, formatter={"float_kind": lambda v: f"{v:.2f}"}))
    print("\nBest-fit subhalo kappa_s per tile:")
    print(np.array2string(kappa_s, formatter={"float_kind": lambda v: f"{v:.2e}"}))
    print("\nBest-fit subhalo scale_radius (arcsec) per tile:")
    print(np.array2string(scale_radius, formatter={"float_kind": lambda v: f"{v:.3f}"}))
    print("\nBest-fit centre y per tile:")
    print(np.round(best_y, 3))
    print("\nBest-fit centre x per tile:")
    print(np.round(best_x, 3))

    n_done = int(np.sum(~np.isnan(delta)))
    if n_done < expected:
        print(
            f"\nWARNING: {expected - n_done} of {expected} tiles are MISSING. "
            f"Re-run those tile jobs before trusting the map."
        )

    iy, ix = np.unravel_index(np.nanargmax(delta), delta.shape)
    print(
        f"\nStrongest detection: tile ({iy},{ix}) [index {iy*n+ix}], "
        f"delta_logE = {delta[iy, ix]:.2f}, "
        f"M_200 = {mass_at_200[iy, ix]:.2e} Msun, c = {concentration[iy, ix]:.2f}, "
        f"kappa_s = {kappa_s[iy, ix]:.2e}, r_s = {scale_radius[iy, ix]:.3f} arcsec, "
        f"centre = ({best_y[iy, ix]:.3f}, {best_x[iy, ix]:.3f}) arcsec"
    )
    print(f"No-subhalo baseline log_evidence = {baseline:.4f}")

    out_csv = path.join(summary_dir, "delta_log_evidence.csv")
    np.savetxt(out_csv, delta, delimiter=",", fmt="%.4f")
    for label, arr in [
        ("best_fit_mass_at_200", mass_at_200),
        ("best_fit_concentration", concentration),
        ("best_fit_kappa_s", kappa_s),
        ("best_fit_scale_radius", scale_radius),
    ]:
        out = path.join(summary_dir, f"{label}.csv")
        np.savetxt(out, arr, delimiter=",", fmt="%.6e")
        print(f"Saved {label} map -> {out}")
    print(f"\nSaved delta map -> {out_csv}")


if __name__ == "__main__":
    main()
