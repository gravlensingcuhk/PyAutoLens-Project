"""
Elliptical multipole perturbation of the elliptical power law
=============================================================

`PowerLawEllipticalMultipole` perturbs the *isodensity contours* of an elliptical power law
(`PowerLaw`) with an angular multipole of order `m`, in the same way that an isophote ellipse
is perturbed by `EllipseMultipoleShape` on the light side. The two are therefore comparable
like-for-like: the same `(k_m, s_m)` describe the same contour shape, whatever the axis ratio
`q` or the slope `gamma`.

__Contents__

- **The Definition:** The contour family, and how it linearises to a convergence perturbation.
- **The Shape Frame:** Why the sampled components encode the shape angle, not a sky angle.
- **The Deflections:** The harmonic expansion that gives the deflection and potential for any slope.
- **The n = 1 Harmonic:** The resonance at `gamma = 2` and the regularisation convention.
- **Relation To `PowerLawMultipole`:** The circular limit and the amplitude scaling.
- **Numerics:** Truncation, cost and the JAX path.

__The Definition__

In the frame rotated to the major axis (polar angle `theta'` from the major axis), a `PowerLaw`
contour of major-axis radius `a` is `r(theta') = a rho(theta')`, with

    rho(theta') = q / sqrt(q^2 cos^2 theta' + sin^2 theta'),

and its convergence is `kappa_PL = kappa_resc a^(1 - gamma)`, with `a = r / rho` the major-axis
elliptical radius and `kappa_resc = einstein_radius_rescaled`. The perturbed contour family is

    r(theta') = a [ rho(theta') + k_m cos(m (theta' + s_m)) ],

which is the family `Ellipse` + `EllipseMultipoleShape` draws for the light (the perturbation
is added along the polar radial direction, in units of the major axis `a`, as a function of
the polar angle). Assigning each point the major-axis radius `a` of the contour through it and
linearising in `k_m`,

    delta kappa(r, theta') = kappa_resc k_m (gamma - 1) r^(1 - gamma) rho(theta')^(gamma - 2)
                             cos(m (theta' + s_m)).

`k_m` is the fractional contour deformation in units of the major axis -- the isophote's
`k/a` -- and `s_m` is the shape angle. The linearisation error is second order in `k_m`
(0.7 per cent of the perturbation at `k_m = 0.05`, `q = 0.6`).

__The Shape Frame__

`shape_multipole_comps = (k_m sin(m s_m), k_m cos(m s_m))`, converted by
`convert.multipole_k_m_and_phi_m_from`: the angle it returns *is* the shape angle `s_m`, measured
from the major axis, with the sign convention of `get_shape_angle` everywhere in PyAutoGalaxy
(`s_m = phi_base - phi_m`; for `m = 4`, `s_4 = 0` is disky and `+-45` degrees boxy). The sky
angle of the multipole is `phi_m = phi_base - s_m` (`phi_m_from`). It is never a parameter: a
multipole is a deformation *of an ellipse*, so its orientation only means something relative to
that ellipse, and sampling a sky angle couples it to the ellipse angle (rotating the ellipse has
to rotate the multipole to keep the same shape). Sampling `s_m` removes that coupling and makes
the posterior marginal over the shape directly.

__The Deflections__

`delta kappa = r^(1 - gamma) H(theta')` is separable, so `H` is expanded in circular harmonics,

    H(theta') = sum_n [A_n cos(n theta') + B_n sin(n theta')],

numerically on `N_ANGLES` equally spaced angles up to order `N_HARMONICS`, and the lens
equation is solved per harmonic (Evans & Witt 2003; Chu, Hu & Kneib 2013): for `n != 1`,

    psi_n     = c_n r^(3 - gamma) [A_n cos(n theta') + B_n sin(n theta')],   c_n = 2 / ((3 - gamma)^2 - n^2),
    alpha_r   = d psi_n / dr,          alpha_theta = (1 / r) d psi_n / d theta'.

An elliptical multipole of order `m` contains every harmonic of the same parity, including, for
even `m`, an `n = 0` term (the perturbation changes the azimuthally averaged mass slightly:
`A_0 / A_4 ~ 0.3 per cent` at `q = 0.6`) and an `n = 2` term; `convergence_func` returns the
`n = 0` term, so enclosed-mass calculations are right.

__The n = 1 Harmonic__

`c_1` has a pole at `gamma = 2`, and every odd `m` on an ellipse has `n = 1` content, so the
raw solution is singular on the isothermal slope -- the hazard of `PowerLawMultipole` with
`m = 1`. The residue of the pole is the potential `r cos theta'`, a pure translation, so the
regular solution is the generic one minus that translation:

    psi_1 = 2 r theta_E^d g / (4 - gamma) [A_1 cos theta' + B_1 sin theta'],
    d = 2 - gamma,  L = ln(r / theta_E),  g = expm1(d L) / d  ->  L as d -> 0,

(`theta_E` the major-axis Einstein radius) which is `psi_1 = r ln(r / theta_E) [...]` at
`gamma = 2`: the counter-term of Paugnat & Gilman (2025, Appendix A 2 a) generalised to every
slope. It differs from the generic solution by the constant deflection
`2 theta_E^d / ((3 - gamma)^2 - 1) (A_1, B_1)`, which is degenerate with the source position, so
the lensing observables are unchanged away from `gamma = 2` and finite at it.

__Relation To Paugnat & Gilman, And Where The Linearisation Ends__

Paugnat & Gilman (2025) define their elliptical multipole on the *elliptical* angle of the
isodensity contour and give closed forms for the isothermal slope; this class uses the polar
angle, because that is the variable the isophote fit perturbs in, and solves every slope. The
two agree in spirit (a deformed ellipse, amplitude relative to the ellipse), not in number: an
`a_m` from their paper is not this `k_m` away from `q = 1`. Only the `n = 1` counter-term is
taken from them.

The perturbation is first order in `k_m`. The fractional density change is
`k_m (gamma - 1) / rho(theta')`, largest on the minor axis where `1 / rho = 1 / q`, so for a
flat lens and a strong multipole (`q = 0.3`, `k_m = 0.2`, `gamma = 2.5`) it reaches unity and
the convergence can go negative there. At the isothermal slope it stays below a quarter for
`k_m` below 0.1 and `q` above 0.4; a fit that wants more than that is outside what a linear
multipole means.

__Relation To `PowerLawMultipole`__

At `q = 1` the convergence reduces to `PowerLawMultipole`'s with `k_circ = (3 - gamma)(gamma - 1) k_m`
and `phi_m = -s_m` (equal strengths at `gamma = 2`), and the deflections agree exactly for
`m = 3, 4` (for `m = 1` up to the constant above). Away from `q = 1` the circular class keeps a
circular perturbation on an elliptical base, and its `k_m` is not the contour deformation.

__Numerics__

The harmonic coefficients of `rho^(gamma - 2)` decay as `((1 - q) / (1 + q))^n`; with
`N_HARMONICS = 64` the truncation is below `2e-10` at `q = 0.3` and `5e-7` at `q = 0.2`, `gamma = 3`.
The per-point cost is one two-term recurrence per harmonic (`cos`, `sin` of `n theta'` from
`(n - 1) theta'`), a Python loop on the NumPy path and `jax.lax.scan` on the JAX path, after the
pattern of the `PowerLaw` omega series. All shapes are static, so the profile traces under
`jax.jit`, `jax.grad` and `jax.vmap`; `m` is an `int` and stays static.

References
----------
Paugnat & Gilman (2025), elliptical multipoles for gravitational lenses.
Chu, Hu & Kneib (2013), ApJ, 765, 134.  arXiv:1302.5482
Evans & Witt (2003), MNRAS, 345, 1351.
"""

import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy import convert

from autogalaxy.profiles.mass.total.power_law import PowerLaw

# The angular grid the harmonic expansion is taken on and the order it is truncated at. Both
# are static: the cos / sin matrices below are built once and the JAX path traces fixed shapes.
N_ANGLES = 512
N_HARMONICS = 64

_ANGLES = 2.0 * np.pi * np.arange(N_ANGLES) / N_ANGLES
_ORDERS = np.arange(N_HARMONICS + 1)
_COS_N_THETA = np.cos(np.outer(_ORDERS, _ANGLES))
_SIN_N_THETA = np.sin(np.outer(_ORDERS, _ANGLES))
# A_0 is the mean, the others twice the mean; sin(0) = 0 makes B_0 = 0 by itself.
_COEFF_SCALE = np.where(_ORDERS == 0, 1.0, 2.0) / N_ANGLES


def harmonic_coefficients_from(
    axis_ratio, slope, m: int, shape_angle_radians, xp=np
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The cosine and sine coefficients `(A_n, B_n)`, `n = 0 .. N_HARMONICS`, of the angular part of
    the perturbation, `H(theta') = (gamma - 1) rho(theta')^(gamma - 2) cos(m (theta' + s_m))`,
    in the frame rotated to the major axis (see the module docstring).

    Parameters
    ----------
    axis_ratio
        The axis ratio `q` of the base power law.
    slope
        The density slope `gamma`.
    m
        The multipole order.
    shape_angle_radians
        The shape angle `s_m` in radians.
    """
    angles = xp.asarray(_ANGLES)
    rho = axis_ratio / xp.sqrt(
        axis_ratio**2 * xp.cos(angles) ** 2 + xp.sin(angles) ** 2
    )
    h = (
        (slope - 1.0)
        * rho ** (slope - 2.0)
        * xp.cos(m * (angles + shape_angle_radians))
    )
    scale = xp.asarray(_COEFF_SCALE)
    return scale * (xp.asarray(_COS_N_THETA) @ h), scale * (
        xp.asarray(_SIN_N_THETA) @ h
    )


def _harmonic_sums_numpy(cos_theta, sin_theta, coeff_a, coeff_b, weights):
    sum_1 = np.zeros_like(cos_theta)
    sum_2 = np.zeros_like(cos_theta)
    cos_n = np.ones_like(cos_theta)
    sin_n = np.zeros_like(cos_theta)
    for n in range(N_HARMONICS + 1):
        sum_1 += weights[n] * (coeff_a[n] * cos_n + coeff_b[n] * sin_n)
        sum_2 += weights[n] * n * (coeff_b[n] * cos_n - coeff_a[n] * sin_n)
        cos_n, sin_n = (
            cos_n * cos_theta - sin_n * sin_theta,
            sin_n * cos_theta + cos_n * sin_theta,
        )
    return sum_1, sum_2


def _harmonic_sums_jax(cos_theta, sin_theta, coeff_a, coeff_b, weights, xp):
    import jax

    def body(carry, inputs):
        cos_n, sin_n, sum_1, sum_2 = carry
        a_n, b_n, w_n, n = inputs
        sum_1 = sum_1 + w_n * (a_n * cos_n + b_n * sin_n)
        sum_2 = sum_2 + w_n * n * (b_n * cos_n - a_n * sin_n)
        cos_next = cos_n * cos_theta - sin_n * sin_theta
        sin_next = sin_n * cos_theta + cos_n * sin_theta
        return (cos_next, sin_next, sum_1, sum_2), None

    init = (
        xp.ones_like(cos_theta),
        xp.zeros_like(cos_theta),
        xp.zeros_like(cos_theta),
        xp.zeros_like(cos_theta),
    )
    (_, _, sum_1, sum_2), _ = jax.lax.scan(
        body,
        init,
        (coeff_a, coeff_b, weights, xp.asarray(_ORDERS, dtype=cos_theta.dtype)),
    )
    return sum_1, sum_2


def harmonic_sums_from(cos_theta, sin_theta, coeff_a, coeff_b, weights, xp=np):
    """
    The two weighted harmonic sums every point of the grid needs,

        S_1 = sum_n w_n [A_n cos(n theta') + B_n sin(n theta')],
        S_2 = sum_n w_n n [B_n cos(n theta') - A_n sin(n theta')],

    from `cos theta'`, `sin theta'` by the angle-addition recurrence, so no `cos(n theta')` is
    ever evaluated directly. `S_1` carries the potential and the radial deflection, `S_2` the
    tangential deflection.
    """
    if xp.__name__.startswith("jax"):
        return _harmonic_sums_jax(cos_theta, sin_theta, coeff_a, coeff_b, weights, xp)
    return _harmonic_sums_numpy(cos_theta, sin_theta, coeff_a, coeff_b, weights)


def polar_to_cartesian_yx(
    a_r, a_angle, polar_angle, xp=np
) -> Tuple[np.ndarray, np.ndarray]:
    """The `(y, x)` components of a vector given by its radial and tangential components."""
    return (
        a_r * xp.sin(polar_angle) + a_angle * xp.cos(polar_angle),
        a_r * xp.cos(polar_angle) - a_angle * xp.sin(polar_angle),
    )


class PowerLawEllipticalMultipole(PowerLaw):
    latent_parameters = {
        (
            "shape_multipole_comps_0",
            "shape_multipole_comps_1",
        ): convert.latent_shape_multipole_from
    }

    r"""Elliptical multipole perturbation of an elliptical power-law mass distribution.

    The perturbation deforms the power law's elliptical isodensity contours exactly as
    :class:`EllipseMultipoleShape` deforms an isophote ellipse, so the strength and shape
    angle are directly comparable between mass and light (see the module docstring for the
    definition, the deflection solution and the conventions). To first order in the strength,

    .. math::

        \delta\kappa(r, \theta') = \kappa_{\rm resc}\, k_m\, (\gamma - 1)\, r^{1 - \gamma}
            \rho(\theta')^{\gamma - 2} \cos\!\bigl(m(\theta' + s_m)\bigr),

    with :math:`\theta'` the polar angle from the major axis and
    :math:`\rho(\theta') = q / \sqrt{q^2 \cos^2\theta' + \sin^2\theta'}`.

    This profile provides only the perturbation; it must sit alongside a :class:`PowerLaw`
    sharing the same ``centre``, ``ell_comps``, ``einstein_radius`` and ``slope`` (prior links
    in a model). Each order then adds exactly two free parameters.

    Parameters
    ----------
    m : int
        Multipole order (1, 3 and 4 are the orders used for lens galaxies).
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre (the power law's).
    ell_comps : (float, float)
        Ellipticity components of the power law whose contours are perturbed.
    einstein_radius : float
        Einstein radius in arcseconds (the power law's).
    slope : float
        Logarithmic density slope :math:`\gamma` (the power law's).
    shape_multipole_comps : (float, float)
        :math:`(k_m \sin m s_m,\, k_m \cos m s_m)`: the strength :math:`k_m` (fractional
        contour deformation in units of the major axis) and the shape angle :math:`s_m`
        measured from the major axis.

    Examples
    --------

    mass = al.mp.PowerLaw(
        centre=(0.0, 0.0), ell_comps=(-0.1, 0.2), einstein_radius=1.0, slope=2.2
    )

    multipole = al.mp.PowerLawEllipticalMultipole(
        m=4,
        centre=(0.0, 0.0),
        ell_comps=(-0.1, 0.2),
        einstein_radius=1.0,
        slope=2.2,
        shape_multipole_comps=(0.0, 0.02),   # k_4 = 0.02, s_4 = 0: disky
    )

    galaxy = al.Galaxy(redshift=0.5, mass=mass, multipole=multipole)
    """

    def __init__(
        self,
        m: int = 4,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        shape_multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=slope,
        )
        self.m = int(m)
        self.shape_multipole_comps = shape_multipole_comps

    def k_m_and_shape_angle_from(self, xp=np) -> Tuple[float, float]:
        """The strength `k_m` and the shape angle `s_m` in degrees."""
        return convert.multipole_k_m_and_phi_m_from(
            multipole_comps=self.shape_multipole_comps, m=self.m, xp=xp
        )

    def get_shape_angle(self, xp=np) -> float:
        """
        The shape angle `s_m`, wrapped to `[-180/m, 180/m)` degrees. For `m = 4`, 0 is pure
        diskiness and +-45 pure boxiness. It is a parameter of this profile, not a derived
        quantity, so no base profile is needed.
        """
        period = 360.0 / self.m
        return (
            xp.mod(self.k_m_and_shape_angle_from(xp)[1] + period / 2.0, period)
            - period / 2.0
        )

    def phi_m_from(self, xp=np) -> float:
        """The multipole's sky orientation `phi_m = phi_base - s_m` in degrees, for reference."""
        return self.angle(xp) - self.k_m_and_shape_angle_from(xp)[1]

    def _amplitude_and_coefficients(self, xp=np):
        """`kappa_resc k_m` and the harmonic coefficients `(A_n, B_n)` of this profile."""
        k_m, shape_angle = self.k_m_and_shape_angle_from(xp)
        coeff_a, coeff_b = harmonic_coefficients_from(
            axis_ratio=self.axis_ratio(xp),
            slope=self.slope,
            m=self.m,
            shape_angle_radians=shape_angle * xp.asarray(np.pi / 180.0),
            xp=xp,
        )
        return self.einstein_radius_rescaled(xp) * k_m, coeff_a, coeff_b

    def _harmonic_weights(self, xp=np):
        """`c_n = 2 / ((3 - gamma)^2 - n^2)`, with `c_1 = 0` (the `n = 1` term is added separately)."""
        orders = xp.asarray(_ORDERS, dtype=float)
        is_one = orders == 1
        denominator = xp.where(is_one, 1.0, (3.0 - self.slope) ** 2 - orders**2)
        return xp.where(is_one, 0.0, 2.0 / denominator)

    def _n_1_radial_terms(self, r, xp=np):
        """
        The radial functions of the regularised `n = 1` harmonic (module docstring): the
        prefactors of `[A_1 cos theta' + B_1 sin theta']` in the potential, the radial
        deflection, and of `[B_1 cos theta' - A_1 sin theta']` in the tangential deflection.
        """
        einstein_radius = self.einstein_radius_major_from(xp)
        d = 2.0 - self.slope
        log_r = xp.log(r / einstein_radius)
        near_resonance = xp.abs(d) < 1.0e-10
        d_safe = xp.where(near_resonance, 1.0, d)
        g = xp.where(near_resonance, log_r, xp.expm1(d_safe * log_r) / d_safe)
        prefactor = 2.0 * einstein_radius**d / (4.0 - self.slope)
        potential = prefactor * r * g
        radial = prefactor * (g + xp.exp(d * log_r))
        tangential = prefactor * g
        return potential, radial, tangential

    def _polar_grid(self, grid, xp=np):
        y = grid.array[:, 0]
        x = grid.array[:, 1]
        r = xp.sqrt(y**2 + x**2 + 1.0e-16)
        return r, xp.arctan2(y, x)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        The deflection angles of the perturbation on a grid of (y,x) arc-second coordinates,
        from the harmonic expansion of the module docstring, computed in the frame rotated to
        the major axis and rotated back by the decorator.
        """
        amplitude, coeff_a, coeff_b = self._amplitude_and_coefficients(xp)
        r, theta = self._polar_grid(grid, xp)
        cos_theta, sin_theta = xp.cos(theta), xp.sin(theta)

        sum_1, sum_2 = harmonic_sums_from(
            cos_theta, sin_theta, coeff_a, coeff_b, self._harmonic_weights(xp), xp=xp
        )
        _, radial_1, tangential_1 = self._n_1_radial_terms(r, xp)

        r_power = r ** (2.0 - self.slope)
        a_r = (3.0 - self.slope) * r_power * sum_1 + radial_1 * (
            coeff_a[1] * cos_theta + coeff_b[1] * sin_theta
        )
        a_angle = r_power * sum_2 + tangential_1 * (
            coeff_b[1] * cos_theta - coeff_a[1] * sin_theta
        )

        return amplitude * xp.stack(
            polar_to_cartesian_yx(a_r=a_r, a_angle=a_angle, polar_angle=theta, xp=xp),
            axis=-1,
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        The convergence of the perturbation on a grid of (y,x) arc-second coordinates (the
        linearised contour deformation of the module docstring).
        """
        k_m, shape_angle = self.k_m_and_shape_angle_from(xp)
        r, theta = self._polar_grid(grid, xp)
        axis_ratio = self.axis_ratio(xp)
        rho = axis_ratio / xp.sqrt(
            axis_ratio**2 * xp.cos(theta) ** 2 + xp.sin(theta) ** 2
        )
        return (
            self.einstein_radius_rescaled(xp)
            * k_m
            * (self.slope - 1.0)
            * r ** (1.0 - self.slope)
            * rho ** (self.slope - 2.0)
            * xp.cos(self.m * (theta + shape_angle * xp.asarray(np.pi / 180.0)))
        )

    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """The lensing potential of the perturbation on a grid of (y,x) arc-second coordinates."""
        amplitude, coeff_a, coeff_b = self._amplitude_and_coefficients(xp)
        r, theta = self._polar_grid(grid, xp)
        cos_theta, sin_theta = xp.cos(theta), xp.sin(theta)
        sum_1, _ = harmonic_sums_from(
            cos_theta, sin_theta, coeff_a, coeff_b, self._harmonic_weights(xp), xp=xp
        )
        potential_1, _, _ = self._n_1_radial_terms(r, xp)
        return amplitude * (
            r ** (3.0 - self.slope) * sum_1
            + potential_1 * (coeff_a[1] * cos_theta + coeff_b[1] * sin_theta)
        )

    def convergence_func(self, grid_radius, xp=np):
        """
        The azimuthally averaged convergence of the perturbation at a circular radius: its
        `n = 0` harmonic, `kappa_resc k_m A_0 r^(1 - gamma)`, which is not zero for even `m`
        on an ellipse. Reached by the radial mass integrals (`mass_angular_within_circle_from`).
        """
        radii = grid_radius.array if hasattr(grid_radius, "array") else grid_radius
        amplitude, coeff_a, _ = self._amplitude_and_coefficients(xp)
        return amplitude * coeff_a[0] * radii ** (1.0 - self.slope)
