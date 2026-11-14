"""
Caustic-square source-plane max zoom (2026-11-14).

__Why__

The stock "Source Plane (Max Zoom)" panel of `fit.png` zooms to the brightest
region of the source (`visualize/general/zoom/plane_percent` for parametric
sources, `inversion_percent` of the reconstructed pixel fluxes for pixelized
ones). Both fail in practice:

- A parametric MGE source that spreads a faint pedestal over the plane (J0959's
  v5 source_lp) makes the "bright region" the whole mask — no zoom at all.
- A pixelized source whose flux is spread over many mesh pixels at similar
  brightness (the RTU meshes on J18/J0959) likewise zooms to the full mesh
  extent, hiding the source's actual morphology — the very thing the panel is
  for (the RTU zoom failure hid the source problems until 2026-11).

__What__

`patch_source_plane_zoom()` wraps
`autolens.imaging.plot.fit_imaging_plots._plot_source_plane` so that the
max-zoom panel (`zoom_to_brightest=True`, `zoom_extent_scale=1.0`) is centred
on a square around the CAUSTIC bounding box plus `CAUSTIC_ZOOM_MARGIN`, floored
at `CAUSTIC_ZOOM_MIN_HALF`:

- Parametric path: the plane image is evaluated on a uniform grid over the box.
- Inversion path: the mapper's `extent_from` is temporarily replaced with the
  box, so `plot_mapper` renders the reconstruction in the caustic frame.

The caustics already reach the panel as `lines`, so the box costs nothing to
compute. Mid-zoom (scale 2.0), no-zoom and line-less panels keep the stock
behaviour. Any failure falls back to the stock panel.

Applied in `slam_pipeline/__init__.py`, so every runner gets it at import.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Padding (arcsec) around the caustic bounding box, and the floor on the box
# half-width so a tiny caustic still gets a useful panel.
CAUSTIC_ZOOM_MARGIN = 0.2
CAUSTIC_ZOOM_MIN_HALF = 0.15
# Pixels across the zoomed evaluation grid (parametric path).
CAUSTIC_ZOOM_NPIX = 240


def _caustic_square(lines):
    """(centre_y, centre_x, half_width) of a square around the caustic lines."""
    pts = np.vstack([np.asarray(line) for line in lines if np.asarray(line).size])
    y0, y1 = pts[:, 0].min(), pts[:, 0].max()
    x0, x1 = pts[:, 1].min(), pts[:, 1].max()
    half = max(0.5 * max(y1 - y0, x1 - x0) + CAUSTIC_ZOOM_MARGIN, CAUSTIC_ZOOM_MIN_HALF)
    return 0.5 * (y0 + y1), 0.5 * (x0 + x1), half


def patch_source_plane_zoom():
    from autolens.imaging.plot import fit_imaging_plots as fip

    if getattr(fip, "_caustic_zoom_patched", False):
        return

    import autoarray as aa
    from autoarray.inversion.mappers.abstract import Mapper
    from autoarray.inversion.plot.mapper_plots import plot_mapper
    from autolens.lens.plot.tracer_plots import plane_image_from
    from autogalaxy.util.plot_utils import plot_array
    from autoarray.plot.utils import plot_regions

    stock_plot_source_plane = fip._plot_source_plane

    def _stock(fit, ax, plane_index, **kwargs):
        return stock_plot_source_plane(fit, ax, plane_index, **kwargs)

    def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                           colormap=None, use_log10=False, title=None,
                           lines=None, line_colors=None, vmax=None,
                           zoom_extent_scale=1.0, regions=None,
                           region_alpha=0.25, region_labels=None):
        kwargs = dict(zoom_to_brightest=zoom_to_brightest, colormap=colormap,
                      use_log10=use_log10, title=title, lines=lines,
                      line_colors=line_colors, vmax=vmax,
                      zoom_extent_scale=zoom_extent_scale, regions=regions,
                      region_alpha=region_alpha, region_labels=region_labels)
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        is_parametric = not tracer.planes[plane_index].has(cls=aa.Pixelization)
        if not (zoom_to_brightest and zoom_extent_scale == 1.0 and lines):
            return _stock(fit, ax, plane_index, **kwargs)
        try:
            centre_y, centre_x, half = _caustic_square(lines)

            if is_parametric:
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
                return

            # Inversion path: render the reconstruction in the caustic frame by
            # standing the mapper's extent_from on the caustic box.
            inversion = fit.inversion
            mapper_list = inversion.cls_list_from(cls=Mapper)
            mapper = mapper_list[plane_index - 1] if plane_index > 0 else mapper_list[0]
            pixel_values = inversion.reconstruction_dict[mapper]
            extent = (centre_x - half, centre_x + half, centre_y - half, centre_y + half)
            stock_extent_from = mapper.extent_from
            mapper.extent_from = lambda **_: extent
            try:
                plot_mapper(
                    mapper,
                    solution_vector=pixel_values,
                    ax=ax,
                    title=title if title is not None else f"Source Reconstruction (plane {plane_index})",
                    colormap=colormap,
                    use_log10=use_log10,
                    vmax=vmax,
                    zoom_to_brightest=True,
                    lines=lines,
                    line_colors=line_colors,
                    regions=regions,
                    region_alpha=region_alpha,
                    region_labels=region_labels,
                )
            finally:
                mapper.extent_from = stock_extent_from
        except Exception as exc:
            logger.warning(f"caustic-zoom panel failed ({exc}); using the stock panel")
            return _stock(fit, ax, plane_index, **kwargs)

    fip._plot_source_plane = _plot_source_plane
    fip._caustic_zoom_patched = True
