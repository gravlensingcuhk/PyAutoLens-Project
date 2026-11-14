"""
Caustic-square source-plane max zoom (2026-11-14).

__Why__

The stock "Source Plane (Max Zoom)" panel of `fit.png` zooms to the brightest
region of the plane image (`visualize/general/zoom/plane_percent`). For a
source whose MGE spreads a faint pedestal over the whole plane (J0959's v5
source_lp), the "bright region" is the whole mask and the max zoom is no zoom
at all — the actual source is a speck in the middle.

__What__

`patch_source_plane_zoom()` wraps
`autolens.imaging.plot.fit_imaging_plots._plot_source_plane` so that the
max-zoom panel (the parametric path with `zoom_to_brightest=True` and
`zoom_extent_scale=1.0`) evaluates the plane image on a square uniform grid
covering the caustic bounding box plus `CAUSTIC_ZOOM_MARGIN`, centred on the
caustic. The caustics already reach the panel as `lines`, so the box costs
nothing to compute. The mid-zoom (scale 2.0), no-zoom and inversion panels
keep the stock behaviour. Any failure falls back to the stock panel.

Applied in `slam_pipeline/__init__.py`, so every runner gets it at import.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Padding (arcsec) around the caustic bounding box, and the floor on the box
# half-width so a tiny caustic still gets a useful panel.
CAUSTIC_ZOOM_MARGIN = 0.1
CAUSTIC_ZOOM_MIN_HALF = 0.15
# Pixels across the zoomed evaluation grid.
CAUSTIC_ZOOM_NPIX = 240


def patch_source_plane_zoom():
    from autolens.imaging.plot import fit_imaging_plots as fip

    if getattr(fip, "_caustic_zoom_patched", False):
        return

    import autoarray as aa
    from autolens.lens.plot.tracer_plots import plane_image_from
    from autogalaxy.util.plot_utils import plot_array
    from autoarray.plot.utils import plot_regions

    stock_plot_source_plane = fip._plot_source_plane

    def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                           colormap=None, use_log10=False, title=None,
                           lines=None, line_colors=None, vmax=None,
                           zoom_extent_scale=1.0, regions=None,
                           region_alpha=0.25, region_labels=None):
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        is_parametric = not tracer.planes[plane_index].has(cls=aa.Pixelization)
        if not (zoom_to_brightest and zoom_extent_scale == 1.0 and is_parametric and lines):
            return stock_plot_source_plane(
                fit, ax, plane_index, zoom_to_brightest=zoom_to_brightest,
                colormap=colormap, use_log10=use_log10, title=title, lines=lines,
                line_colors=line_colors, vmax=vmax, zoom_extent_scale=zoom_extent_scale,
                regions=regions, region_alpha=region_alpha, region_labels=region_labels,
            )
        try:
            pts = np.vstack([np.asarray(line) for line in lines if np.asarray(line).size])
            y0, y1 = pts[:, 0].min(), pts[:, 0].max()
            x0, x1 = pts[:, 1].min(), pts[:, 1].max()
            centre_y, centre_x = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
            half = max(
                0.5 * max(y1 - y0, x1 - x0) + CAUSTIC_ZOOM_MARGIN,
                CAUSTIC_ZOOM_MIN_HALF,
            )
            grid = aa.Grid2D.uniform(
                shape_native=(CAUSTIC_ZOOM_NPIX, CAUSTIC_ZOOM_NPIX),
                pixel_scales=2.0 * half / CAUSTIC_ZOOM_NPIX,
                origin=(centre_y, centre_x),
            )
            image = plane_image_from(
                galaxies=tracer.planes[plane_index], grid=grid, zoom_to_brightest=False,
            )
            plot_array(
                array=image, ax=ax,
                title=title if title is not None else f"Source Plane {plane_index}",
                colormap=colormap, use_log10=use_log10, vmax=vmax, lines=lines,
                line_colors=line_colors,
            )
            plot_regions(ax=ax, regions=regions, region_alpha=region_alpha,
                         region_labels=region_labels)
        except Exception as exc:
            logger.warning(f"caustic-zoom panel failed ({exc}); using the stock panel")
            return stock_plot_source_plane(
                fit, ax, plane_index, zoom_to_brightest=zoom_to_brightest,
                colormap=colormap, use_log10=use_log10, title=title, lines=lines,
                line_colors=line_colors, vmax=vmax, zoom_extent_scale=zoom_extent_scale,
                regions=regions, region_alpha=region_alpha, region_labels=region_labels,
            )

    fip._plot_source_plane = _plot_source_plane
    fip._caustic_zoom_patched = True
