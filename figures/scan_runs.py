"""
Run inventory + informative figure names (2026-11-14).

Scans the local zip mirrors, classifies every run from the zip's own `model.info`
(mesh, regularization, shear-prior width, point-source tag), then:

  1. Regenerates `docs/update_2026-11-12/run_inventory.txt`.
  2. Renames the remade fit pngs in `figures/fit_png_caustic_zoom/<output_dir>/` to
     `<lens>_<stage>_<noise>_<mesh+reg>_<shear>_<band>[_extra]_<id8>.png`.

`remake_fit_pngs.py` uses the same `figure_name()` so new remakes land with these
names directly. Re-run this after every rsync pull from phygpu:

    uv run python figures/scan_runs.py
"""

import os
import re
import zipfile

LENS_SHORT = {
    "COSJ100018+022138": "J18",
    "COSJ100024+021749": "J24",
    "COSJ095921+020638": "J0959",
}
LENS_BAND = {"J18": "f150w", "J24": "f444w", "J0959": "f150w"}

STAGE_ORDER = [
    ("model_setup_free_source", "source_lp[1]", "source-lp"),
    ("model_setup_free_source", "source_pix[1]", "source-pix1"),
    ("model_setup_free_source", "source_pix[2]", "source-pix2"),
    ("model_setup_free_source", "light[1]", "light"),
    ("mass_models", "mass_EPL", "mass-EPL"),
    ("mass_models", "mass_multipole", "mass-multipole"),
    ("mass_models", "mass_light_dark[1]", "mass-mld"),
]

MESH_SHORT = {
    "RectangularRTUAdaptDensity": "rtu-dens",
    "RectangularRTUAdaptImage": "rtu-img",
    "Delaunay": "delaunay",
}
REG_SHORT = {
    "AdaptPower": "adaptpow",
    "ConstantSplit": "constsplit",
    "AdaptSplit": "adaptsplit",
    "AdaptSplitPower": "aspower",
    "Constant": "const",
    "Adapt": "adapt",
}

# J0959 has two source_lp chains: the old failing bulge+disk MGE source (source_lp
# 81040185, finished Nov 9-12) and the fixed point+bulge source (9017b6f2, Nov 13+).
# Chain membership is not recoverable from the zips (priors are not chained), so split
# by archive date: everything archived before 2026-11-13 descends from the old source.
CHAIN_SPLITS = {
    ("output_sam_v5", "COSJ095921+020638"): ((2026, 11, 13), "oldsrc", "ptsrc"),
}
# Static per-(dir, lens) tags where a single chain needs no date split.
DIR_LENS_EXTRA = {
    ("output_sam_v4", "COSJ095921+020638"): ["ptsrc"],
}

# output_dir -> tags. noise = noise-map version (v4 plain, v5 psffloor, v6 psfcoh).
DIRS = {
    "output_sam_v4": {
        "noise": "v4",
        "desc": "production on data/cowls_sam, plain noise map",
    },
    "output_sam_v4_f277": {
        "noise": "v4",
        "band": "f277w",
        "desc": "J18 in F277W, plain noise map",
    },
    "output_sam_v5": {
        "noise": "v5",
        "desc": "psffloor noise map (independent-scatter PSF floor)",
    },
    "output_sam_v6": {
        "noise": "v6",
        "desc": "psfcoh noise map (coherent PSF-error floor) + wideshear everywhere",
    },
    "output_sam_mb_v6": {
        "noise": "v6",
        "band": "mb",
        "desc": "multi-band shared-mass chains, psfcoh noise",
    },
    "output_test_as": {
        "noise": "v4",
        "extra": ["asfix"],
        "desc": "J18 AdaptSplit test: weight_floor 0.1 + adapt_floor 0.02",
    },
    "output_test_as_sp1": {
        "noise": "v4",
        "extra": ["asfix", "sp1adapt"],
        "desc": "as output_test_as but source_pix[2] adapts to source_pix[1]",
    },
    "output_test_asp": {
        "noise": "v4",
        "extra": ["asfix"],
        "desc": "J18 AdaptSplitPower test (same fix settings)",
    },
    "output_sam_v3": {
        "noise": "v3",
        "desc": "SUPERSEDED 2026-11-04 chains (single-group light, 50x50 RTU)",
    },
}

INVENTORY_PATH = os.path.join("docs", "update_2026-11-12", "run_inventory.txt")
FIG_ROOT = os.path.join("figures", "fit_png_caustic_zoom")

SHEAR_RE = re.compile(r"gamma_1\s+.*Prior.*upper_limit = ([\d.]+)")
MESH_RE = re.compile(r"^\s*mesh\s+(\w+) \(N=", re.M)
REG_RE = re.compile(r"^\s*regularization\s+(\w+) \(N=", re.M)


def classify_zip(zip_path):
    """Mesh / regularization / shear-width tags from the zip's model.info."""
    info = {"mesh": None, "reg": None, "shear": None}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            text = zf.read("model.info").decode("utf-8", errors="replace")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    m = MESH_RE.search(text)
    if m:
        info["mesh"] = MESH_SHORT.get(m.group(1), m.group(1))
    r = REG_RE.search(text)
    if r:
        info["reg"] = REG_SHORT.get(r.group(1), r.group(1))
    s = SHEAR_RE.search(text)
    if s:
        info["shear"] = "wideshear" if float(s.group(1)) > 0.15 else "narrowshear"
    return info


def zip_date(zip_path):
    """Latest entry date inside the zip ~ when the search finished / was archived."""
    with zipfile.ZipFile(zip_path) as zf:
        return max(i.date_time for i in zf.infolist())


def extras_for(output_dir, lens, zip_path):
    """Chain/dataset tags beyond the dir-level ones (see CHAIN_SPLITS, DIR_LENS_EXTRA)."""
    extra = list(DIRS[output_dir].get("extra", ()))
    extra += DIR_LENS_EXTRA.get((output_dir, lens), ())
    split = CHAIN_SPLITS.get((output_dir, lens))
    if split:
        boundary, old_tag, new_tag = split
        extra.append(old_tag if zip_date(zip_path) < boundary else new_tag)
    return extra


def figure_name(lens, stage_short, noise, band, cls, id8, extra=(), log10=False):
    parts = [LENS_SHORT[lens], stage_short, noise]
    if cls.get("mesh"):
        parts.append(f"{cls['mesh']}+{cls.get('reg') or '?'}")
    if cls.get("shear"):
        parts.append(cls["shear"])
    parts.append(band)
    parts.extend(extra)
    parts.append(id8)
    return "_".join(parts) + ("_log10.png" if log10 else ".png")


def scan():
    """Every zip on disk: {output_dir: [(lens, stage, stage_short, id16, cls, extra, date), ...]}."""
    runs = {}
    for output_dir in DIRS:
        if not os.path.isdir(output_dir):
            continue
        entries = []
        for prefix, stage, stage_short in STAGE_ORDER:
            for lens in LENS_SHORT:
                stage_dir = os.path.join(output_dir, prefix, lens, stage)
                if not os.path.isdir(stage_dir):
                    continue
                for z in sorted(os.listdir(stage_dir)):
                    if not z.endswith(".zip"):
                        continue
                    zp = os.path.join(stage_dir, z)
                    cls = classify_zip(zp)
                    entries.append((lens, stage, stage_short, z[:-4], cls,
                                    extras_for(output_dir, lens, zp), zip_date(zp)))
        if entries:
            runs[output_dir] = entries
    return runs


def rename_figures(runs):
    renamed = skipped = 0
    for output_dir, entries in runs.items():
        fig_dir = os.path.join(FIG_ROOT, output_dir)
        if not os.path.isdir(fig_dir):
            continue
        cfg = DIRS[output_dir]
        by_id8 = {}
        for lens, stage, stage_short, id16, cls, extra, _ in entries:
            by_id8.setdefault(id16[:8], []).append((lens, stage_short, cls, extra))
        for f in sorted(os.listdir(fig_dir)):
            if not f.endswith(".png"):
                continue
            log10 = f.endswith("_log10.png")
            stem = f[:-10] if log10 else f[:-4]
            id8 = stem.split("_")[-1]
            head = stem[: -len(id8) - 1] if stem.endswith("_" + id8) else stem
            matches = by_id8.get(id8, [])
            lens_from_file = next((k for k in list(LENS_SHORT) + list(LENS_SHORT.values())
                                   if head.startswith(k)), None)
            if lens_from_file:
                short = LENS_SHORT.get(lens_from_file, lens_from_file)
                matches = [m for m in matches if LENS_SHORT[m[0]] == short]
            if len(matches) != 1:
                print(f"  SKIP {output_dir}/{f} ({len(matches)} zip matches for {id8})")
                skipped += 1
                continue
            lens, stage_short, cls, extra = matches[0]
            band = cfg.get("band") or LENS_BAND[LENS_SHORT[lens]]
            new = figure_name(lens, stage_short, cfg["noise"], band, cls, id8,
                              extra=extra, log10=log10)
            if new != f:
                os.rename(os.path.join(fig_dir, f), os.path.join(fig_dir, new))
                renamed += 1
    print(f"renamed {renamed} figures, skipped {skipped}")


def write_inventory(runs):
    lines = [
        "RUN INVENTORY — regenerated by figures/scan_runs.py (2026-11-14)",
        "=" * 80,
        "Path layout: <output_dir>/<model_setup_free_source|mass_models>/<lens>/<stage>/<identifier>.zip",
        "Lens names: J18 = COSJ100018+022138, J24 = COSJ100024+021749, J0959 = COSJ095921+020638",
        "",
        "Figure naming (figures/fit_png_caustic_zoom/<output_dir>/):",
        "  <lens>_<stage>_<noise>_<mesh+reg>_<shear>_<band>[_extra]_<id8>.png",
        "  noise:  v4 = plain noise_map.fits | v5 = psffloor | v6 = psfcoh (coherent floor)",
        "  mesh+reg: rtu-dens/rtu-img (50x50 RectangularRTU) or delaunay, plus",
        "          adaptpow | constsplit | adaptsplit | aspower",
        "  shear:  narrowshear = +/-0.1 prior | wideshear = +/-0.2 (only where shear is free)",
        "  extra:  oldsrc/ptsrc = J0959 chain from the old bulge+disk MGE source_lp",
        "          (81040185) / the fixed point+bulge source_lp (9017b6f2) | asfix =",
        "          weight_floor 0.1 + adapt_floor 0.02 | sp1adapt = source_pix[2] adapts",
        "          to source_pix[1]",
        "",
    ]
    for output_dir, entries in runs.items():
        cfg = DIRS[output_dir]
        lines.append(f"### {output_dir}  [{cfg['noise']}] {cfg['desc']}")
        for lens, stage, stage_short, id16, cls, extra, date in entries:
            mesh_reg = f"{cls['mesh']}+{cls['reg']}" if cls.get("mesh") else "-"
            shear = cls.get("shear") or "-"
            flags = list(extra)
            if cls.get("error"):
                flags.append(f"UNREADABLE ({cls['error']})")
            flag = f"  [{', '.join(flags)}]" if flags else ""
            day = f"{date[0]}-{date[1]:02d}-{date[2]:02d}"
            lines.append(
                f"  {LENS_SHORT[lens]:<6} {stage:<20} {mesh_reg:<22} {shear:<12} {id16[:16]}*  {day}{flag}"
            )
        lines.append("")
    with open(INVENTORY_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {INVENTORY_PATH} ({sum(len(v) for v in runs.values())} zips)")


if __name__ == "__main__":
    found = scan()
    rename_figures(found)
    write_inventory(found)
