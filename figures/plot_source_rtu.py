"""
Plot an RTU rectangular source reconstruction with bilinear interpolation.

The `source_plane_reconstruction_0.csv` a search writes holds the (y, x) centre and value of
every mesh pixel of the warped RECT_SHAPE lattice in row-major order. The library's plot fills
each warped cell with a flat colour, which makes a 40 x 40 source look blocky next to the old
Delaunay plots, whose triangles were colour-interpolated. This draws the same lattice with
Gouraud (bilinear) shading between the cell centres, the same cosmetic treatment.

Usage:
    uv run python figures/plot_source_rtu.py <result_dir_or_zip> <out.png> [shape] [zoom_frac]

`result_dir_or_zip` is a search's unzipped output folder (or its .zip). `shape` is the mesh
side (default 40). `zoom_frac` sets the zoom panel: the box holding every pixel above that
fraction of the peak, padded (default 0.05).
"""
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(result, shape):
    result = Path(result)
    if result.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(result) as z:
            z.extract("image/source_plane_reconstruction_0.csv", tmp)
        result = tmp
    d = np.loadtxt(result / "image/source_plane_reconstruction_0.csv", delimiter=",", skiprows=1)
    y, x, v = (a.reshape(shape, shape) for a in d[:, :3].T)
    return y, x, v


def main(result, out, shape=40, zoom_frac=0.05):
    y, x, v = load(result, shape)
    vmax = v.max()
    sel = v > zoom_frac * vmax
    pad = 0.08
    ylo, yhi = y[sel].min() - pad, y[sel].max() + pad
    xlo, xhi = x[sel].min() - pad, x[sel].max() + pad

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cmap = "jet"
    for ax, shading, title in (
        (axes[0], "nearest", "RTU cells (as the library plots)"),
        (axes[1], "gouraud", "RTU cells, bilinear"),
        (axes[2], "gouraud", "RTU cells, bilinear (full plane)"),
    ):
        m = ax.pcolormesh(x, y, v, shading=shading, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel('x / "')
        ax.set_ylabel('y / "')
        if ax is not axes[2]:
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
    fig.colorbar(m, ax=axes, label=r"e$^-$ s$^{-1}$", fraction=0.025)
    fig.suptitle(Path(result).name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 40, float(args[3]) if len(args) > 3 else 0.05)
