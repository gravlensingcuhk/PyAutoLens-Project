"""
The multiple-image position penalty for every stage after SOURCE LP.

Ported from the HST project (pyauto `z_projects/HST_lensing_mass_light/hpc/slam_pipeline/
positions_util.py`) on 2026-10-28.

__Why This Exists__

The SLaM templates chain the position penalty with
`previous_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.075)` and no
`positions=` argument, which re-derives the multiple images from the previous result: the
source centre is ray-traced back through the maximum-likelihood mass model. For a pixelized
source the "centre" is the *brightest source pixel*, which on a 40 x 40 RTU reconstruction can
be a noise spike far outside the caustic. The point solver then finds a single image and falls
back to walking the source centre inwards until something multiply images, returning positions
that are nowhere near the arc.

That is what undermagnified J18 (`updated_output/`, 2026-10-27): `mass_EPL` and
`mass_multipole` were handed one point on the lens centre at radius 0.11" and one in empty sky
at radius 0.67", with a 0.075" threshold. The only power laws that map those to the same source
position have a tiny Einstein radius and a steep slope, and the chain went from theta_E 0.40"
(`source_pix[2]`) to 0.20" with slope 2.8, the source reconstructing the ring itself. J24 got
the same kind of positions (radius 0.17" and 1.07") and kept theta_E 0.66" only because its
best model sits at the threshold (0.074" against 0.075"), so its mass model was biased too.

__Which Positions__

The hand-marked `positions.json` positions are passed through every stage: they are
observations, and SOURCE LP was fitted to them, so the chain keeps the same constraint instead
of swapping in whatever images the solver happens to find. A lens with no marked positions
falls back to the solved ones, as upstream.

__The Threshold__

Unchanged from SLaM: `factor` times the previous result's maximum source-plane separation of
these positions, floored at `minimum_threshold`. The result still sets the threshold; it no
longer chooses the positions.

A lens whose penalty cannot be built (fewer than two finite positions, `PositionsException`)
runs without the penalty, with a warning, rather than aborting the stage.
"""

import logging
from typing import List, Optional

import autofit as af
import autolens as al

logger = logging.getLogger(__name__)

# The runners' chaining threshold floor (arcsec), shared by source_pix and the mass stages.
MINIMUM_THRESHOLD = 0.075


def positions_likelihood_list_from(
    result: af.Result,
    positions: Optional[al.Grid2DIrregular] = None,
    factor: float = 3.0,
    minimum_threshold: float = MINIMUM_THRESHOLD,
) -> List[al.PositionsLH]:
    """
    The `positions_likelihood_list` for a stage chained from `result`: one penalty on the
    hand-marked `positions` (or, when there are none, on the images solved from `result`), with
    the threshold set from `result`; empty when there are fewer than two finite positions to
    compare. See the module docstring.
    """
    try:
        positions_lh = result.positions_likelihood_from(
            factor=factor,
            minimum_threshold=minimum_threshold,
            positions=positions,
        )
    except al.exc.PositionsException as e:
        logger.warning(f"Position penalty dropped: {e}")
        return []
    return [] if positions_lh is None else [positions_lh]
