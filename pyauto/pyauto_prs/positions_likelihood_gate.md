# Positions likelihood: the threshold gate discards usable information

2026-11-18, from the COWLS SLaM chains (branch `sam`, lenses J18 / J24 / J0959).

## The problem

Every stage after source_lp adds one `PositionsLH` per marked position set: zero penalty
while the max source-plane separation of the marks is below a threshold, and
`-1e8 * (separation - threshold)` above it. Three structural weaknesses showed up in
production:

1. **A gate, not a gradient.** Inside the threshold every model is positions-equal, so a
   mass model that fits the marks at 0.05" feels no pull toward one that fits them at 0.01".
   J18's mass stage escaped along the ellipticity-shear degeneracy to theta_E ~ 0.48 with
   the shear railed at its prior edge, passing the 0.075" gate at 0.026" separation, while
   the marked images sit at 0.30-0.44" (theta_E ~ 0.36-0.42 by geometry). The gate was
   satisfied; the constraint was not used.
2. **The threshold is floored by model inaccuracy, not marking precision.** Positions-only
   scans (SPL+shear sampled over wide ranges, marks ray-shot to the source plane, models
   kept when the marks reunite) show that for J0959's 4-mark quad *no* smooth SPL+shear
   model reunites the marks better than ~0.05" (5/60000 samples at 0.05", none at 0.03");
   the real fits saturate at 0.084". So the threshold cannot be set near the ~0.01" marking
   precision without nuking the true model too - and within the achievable floor the gate
   is flat.
3. **The chained threshold inflates.** factor-3 x the previous stage's achieved separation,
   capped: a sloppy upstream stage loosens the gate downstream (J24's reached 0.15"), and
   the -1e7 plateau over most of the prior then also stalls Nautilus's bound learning
   (J0959's Delaunay mass_EPL: 300k likelihood calls, f_live pinned at 1.0, ESS ~ 1).

## What we did project-side

Derived mass-stage theta_E priors from the marks alone: sample SPL+shear (centre fixed on
the mark origin, ellipticity/shear/slope free), keep models reuniting the marks within
tolerance, and read the accepted theta_E distribution as a TruncatedGaussian prior with
limits at its +/-3 sigma percentiles:

| Lens | marks | positions-only theta_E | mass-stage mode before |
| --- | --- | --- | --- |
| J0959 | 4 (quad) | 0.725 +/- 0.046 | 0.70 (agrees; sampler stalled instead) |
| J18 | 2 x 2 | 0.418 +/- 0.060 | 0.47-0.49, railed shear (84-97th pct edge) |
| J24 | 2 (double) | 0.73 +/- 0.23 | 0.69 (consistent; intrinsically weak) |

This injects the positions information as prior volume - no likelihood change, no
double-counting with the imaging data, no plateau.

## The follow-up: a point-source chi2 likelihood

For systems with point-like images (J0959 is an AGN quad), the standard treatment in the
literature (Anguita+09, SL2S for this very lens) is positions-as-data: a chi2 on the
image-plane marks with astrometric errors, the source position solved or marginalized per
model. That uses the marks at their actual precision (~0.005-0.01") instead of at the
model-family reunion floor, and gives an in-gate gradient that the penalty gate lacks.

Open questions for a PyAutoLens discussion/PR:

- **Double counting.** The imaging likelihood already contains the position information in
  principle; SLaM's penalty was designed to use positions as rejection, not data, to avoid
  counting it twice. For faint/pixelized sources the imaging constraint is much weaker
  than the marks, so the trade may be worth it for point images specifically. Where should
  this live: an alternative `PositionsLH` mode, or a separate likelihood object?
- **Errors.** What astrometric error to assign hand-marked positions (marking GUI pixel
  scale, centroiding of a point image vs an extended knot), and whether to solve or
  marginalize the source-plane position.
- **Interaction with the penalty threshold chaining.** If a chi2 version exists, the
  factor-3 threshold chaining (and its inflation/compounding) could be retired for lenses
  with marked positions; the threshold should at minimum be floored by the model-family
  reunion error (measurable with the scan above) rather than only by the previous stage's
  achieved separation.
- **Sampler interaction.** A hard penalty plateau over most of the prior defeats
  Nautilus's bound learning; a graded positions term (chi2 or a soft-band penalty) would
  remove the plateau and likely the stalls we saw.

Relevant code: `autolens/analysis/positions.py` (`PositionsLH`), SLaM usage in this repo's
`slam_pipeline/positions_util.py` and `slam_pipeline/source_pix.py` / `mass_total.py`.
