import numpy as np
from typing import Tuple
from autogalaxy.convert import multipole_k_m_and_phi_m_from
from autogalaxy import convert


from autogalaxy.ellipse.ellipse.ellipse import Ellipse


def points_perturbed_from(
    points, ellipse: Ellipse, k_m, phi_m, m: int, xp=np
) -> np.ndarray:
    """
    The (y,x) points of an ellipse perturbed by a multipole of order `m`, strength `k_m`
    (arc-seconds) and sky orientation `phi_m` (degrees), shared by every ellipse multipole class.

    The perturbation `k_m cos(m (theta - phi_m))` is added along the polar radial direction at
    the polar angle `theta` of each point. In the y-up frame of the image (the frame the mass
    profiles use; `Ellipse.y_from_major_axis_from` negates y and `DatasetInterp` undoes it when
    it samples the image), a point at polar angle `theta` on an ellipse of position angle `A`
    moves to radius `r_ellipse(theta) + k_m cos(m (theta - phi_m))`, with `phi_m` measured like
    `A`, so `A - phi_m` is the shape angle. This is the contour family the mass side's
    `PowerLawEllipticalMultipole` deforms the power law's isodensity contours into (its tests
    check the two coincide). The expression below is what the former `comps_adjusted`
    re-alignment algebra reduced to (checked to 1e-16 against it); the pinned tests hold.

    Parameters
    ----------
    points
        The (y,x) coordinates of the ellipse that are perturbed.
    ellipse
        The ellipse the points lie on; only its position angle is used.
    k_m
        The multipole strength in arc-seconds (an absolute radial displacement).
    phi_m
        The multipole's sky orientation in degrees.
    m
        The multipole order.
    """
    theta = xp.arctan2(points[:, 0], points[:, 1])
    # The points carry the y flip of `Ellipse.y_from_major_axis_from`, which is why the sky
    # orientation enters with a plus sign here; in the y-up frame this is k cos(m (theta - phi_m)).
    radial = k_m * xp.cos(m * (theta + phi_m * xp.asarray(np.pi / 180.0)))
    return xp.stack(
        arrays=(
            points[:, 0] + radial * xp.sin(theta),
            points[:, 1] + radial * xp.cos(theta),
        ),
        axis=-1,
    )


class EllipseMultipole:
    latent_parameters = {
        ("multipole_comps_0", "multipole_comps_1"): convert.latent_multipole_from
    }

    def __init__(
        self,
        m=4,
        multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        r"""
        class representing the multipole of an ellispe with, which is used to perform ellipse fitting to
        2D data (e.g. an image).

        The multipole is added to the (y,x) coordinates of an ellipse that are already computed via the `Ellipse` class.

        The addition of the multipole is performed as follows:

        :math: r_m = \sum_{i=1}^{m} \left( a_i \cos(i(\theta - \phi)) + b_i \sin(i(\theta - \phi)) \right)
        :math: y_m = r_m \sin(\theta)
        :math: x_m = r_m \cos(\theta)

        Where:

        m = The order of the multipole.
        r = The radial coordinate of the ellipse perturbed by the multipole.
        \phi = The angle of the ellipse.
        a = The amplitude of the cosine term of the multipole.
        b = The amplitude of the sine term of the multipole.
        y = The y-coordinate of the ellipse perturbed by the multipole.
        x = The x-coordinate of the ellipse perturbed by the multipole.
        """

        self.m = m
        self.multipole_comps = multipole_comps

    def get_shape_angle(
        self,
        ellipse: Ellipse,
        xp=np,
    ) -> float:
        """
        The shape angle is the offset between the angle of the ellipse and the angle of the multipole,
        this defines the shape that the multipole takes.

        In the case of the m=4 multipole, angles of 0 indicate pure diskiness, angles +- 45
        indicate pure boxiness.

        Parameters
        ----------
        ellipse
            The base ellipse profile that is perturbed by the multipole.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The angle between the ellipse and the multipole, in degrees between +- 180/m.
        The boundary case `angle == period/2.0` exactly maps to `-period/2.0` after wrap.
        """

        angle = (
            ellipse.angle(xp=xp)
            - multipole_k_m_and_phi_m_from(self.multipole_comps, self.m, xp=xp)[1]
        )
        period = 360.0 / self.m
        return xp.mod(angle + period / 2.0, period) - period / 2.0

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0, xp=np
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points, which are perturbed by the multipole of the ellipse.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        points
            The (y,x) coordinates of the ellipse that are perturbed by the multipole.
        ellipse
            The ellipse that is perturbed by the multipole, which is used to compute the angles of the ellipse.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The (y,x) coordinates of the input points, which are perturbed by the multipole.
        """
        k_m, phi_m = multipole_k_m_and_phi_m_from(self.multipole_comps, self.m, xp=xp)
        return points_perturbed_from(
            points=points, ellipse=ellipse, k_m=k_m, phi_m=phi_m, m=self.m, xp=xp
        )


class EllipseMultipoleScaled(EllipseMultipole):
    latent_parameters = {
        (
            "scaled_multipole_comps_0",
            "scaled_multipole_comps_1",
        ): convert.latent_scaled_multipole_from
    }

    def __init__(
        self,
        m=4,
        scaled_multipole_comps: Tuple[float, float] = (0.0, 0.0),
        major_axis: float = 1.0,
    ):
        r"""
        class representing the multipole of an ellipse, which is used to perform ellipse fitting to
        2D data (e.g. an image). This multipole is fit with its strength held relative to an ellipse with a
        major_axis of 1, allowing for a set of ellipse multipoles to be fit at different major axes but with
        the same scaled strength k/a.

        The scaled_multipole_comps (for all ellipses) are converted to a k value, which is then reset to
        its `true' value for a multipole at the given major axis value, which is then used to perturb an ellipse
        as per the normal `EllipseMultipole' class and below.

        The multipole is added to the (y,x) coordinates of an ellipse that are already computed via the `Ellipse` class.

        The addition of the multipole is performed as follows:

        :math: r_m = \sum_{i=1}^{m} \left( a_i \cos(i(\theta - \phi)) + b_i \sin(i(\theta - \phi)) \right)
        :math: y_m = r_m \sin(\theta)
        :math: x_m = r_m \cos(\theta)

        Where:

        m = The order of the multipole.
        r = The radial coordinate of the ellipse perturbed by the multipole.
        \phi = The angle of the ellipse.
        a = The amplitude of the cosine term of the multipole.
        b = The amplitude of the sine term of the multipole.
        y = The y-coordinate of the ellipse perturbed by the multipole.
        x = The x-coordinate of the ellipse perturbed by the multipole.

        Notes
        -----
        Unlike the previous implementation, the derivation of
        ``specific_multipole_comps`` from ``scaled_multipole_comps`` is now
        performed inside :meth:`points_perturbed_from` rather than at
        ``__init__`` time. This makes the class JAX-traceable: under
        ``jax.jit`` / ``jax.vmap``, ``scaled_multipole_comps`` arrive as
        JAX tracers, and a deferred derivation can thread ``xp`` through to
        the ``convert.py`` helpers. Pre-computing at ``__init__`` time would
        (a) hardcode ``np.*`` (breaks under JAX) and (b) cache stale values
        across ``vmap`` batch elements.
        """

        self.scaled_multipole_comps = scaled_multipole_comps
        self.major_axis = major_axis
        self.m = m

    def get_shape_angle(
        self,
        ellipse: Ellipse,
        xp=np,
    ) -> float:
        """
        The shape angle is the offset between the angle of the ellipse and the angle of the multipole,
        this defines the shape that the multipole takes.

        In the case of the m=4 multipole, angles of 0 indicate pure diskiness, angles +- 45
        indicate pure boxiness.

        This override derives (k_scaled, phi) directly from ``scaled_multipole_comps`` and
        ``major_axis`` rather than from ``self.multipole_comps`` (which is not set on this class).

        Parameters
        ----------
        ellipse
            The base ellipse profile that is perturbed by the multipole.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The angle between the ellipse and the multipole, in degrees between +- 180/m.
        The boundary case `angle == period/2.0` exactly maps to `-period/2.0` after wrap.
        """

        k_scaled, phi = multipole_k_m_and_phi_m_from(
            self.scaled_multipole_comps, self.m, xp=xp
        )
        angle = ellipse.angle(xp=xp) - phi
        period = 360.0 / self.m
        return xp.mod(angle + period / 2.0, period) - period / 2.0

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0, xp=np
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points, which are perturbed by the multipole of the ellipse.

        The derivation of ``specific_multipole_comps`` from ``scaled_multipole_comps`` is performed
        here (rather than at ``__init__`` time) so that ``xp`` can be threaded through the
        ``convert.py`` helpers. This makes the method JAX-traceable when ``xp=jnp`` is passed.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        points
            The (y,x) coordinates of the ellipse that are perturbed by the multipole.
        ellipse
            The ellipse that is perturbed by the multipole, which is used to compute the angles of the ellipse.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The (y,x) coordinates of the input points, which are perturbed by the multipole.
        """
        k_scaled, phi = multipole_k_m_and_phi_m_from(
            multipole_comps=self.scaled_multipole_comps, m=self.m, xp=xp
        )
        return points_perturbed_from(
            points=points,
            ellipse=ellipse,
            k_m=k_scaled * self.major_axis,
            phi_m=phi,
            m=self.m,
            xp=xp,
        )


class EllipseMultipoleShape(EllipseMultipoleScaled):
    latent_parameters = {
        (
            "shape_multipole_comps_0",
            "shape_multipole_comps_1",
        ): convert.latent_shape_multipole_from
    }

    def __init__(
        self,
        m=4,
        shape_multipole_comps: Tuple[float, float] = (0.0, 0.0),
        major_axis: float = 1.0,
    ):
        r"""
        The scaled multipole of an ellipse (`EllipseMultipoleScaled`, strength held as k/a),
        parameterised by its strength and **shape angle** rather than its sky orientation.

        `shape_multipole_comps = ((k/a) sin(m s_m), (k/a) cos(m s_m))`: the angle that
        `convert.multipole_k_m_and_phi_m_from` returns from them is the shape angle `s_m`,
        measured from the ellipse's major axis with the sign convention of `get_shape_angle`
        (`s_m = phi_ellipse - phi_m`; for m=4, 0 is disky and +-45 degrees boxy). The sky
        orientation is derived, `phi_m = phi_ellipse - s_m`, and the points are then perturbed
        exactly as `EllipseMultipoleScaled` does.

        A multipole is a deformation of an ellipse, so its orientation only means something
        relative to that ellipse: sampling a sky angle couples it to the ellipse angle (rotating
        the ellipse has to rotate the multipole to keep the same shape), whereas sampling `s_m`
        keeps the shape fixed as the ellipse turns and gives the posterior over the shape
        directly. This is the light-side counterpart of
        `al.mp.PowerLawEllipticalMultipole`, which uses the same parameter with the same
        meaning, so the two sides are compared in the same `(k/a, s_m)`.

        Parameters
        ----------
        m
            The multipole order.
        shape_multipole_comps
            The components encoding the scaled strength k/a and the shape angle s_m.
        major_axis
            The major axis of the ellipse this multipole perturbs, which scales k/a to k.
        """
        self.shape_multipole_comps = shape_multipole_comps
        self.major_axis = major_axis
        self.m = m

    def k_m_and_shape_angle_from(self, xp=np) -> Tuple[float, float]:
        """The scaled strength k/a and the shape angle s_m in degrees."""
        return multipole_k_m_and_phi_m_from(
            multipole_comps=self.shape_multipole_comps, m=self.m, xp=xp
        )

    def get_shape_angle(self, ellipse: Ellipse = None, xp=np) -> float:
        """
        The shape angle s_m, wrapped to +- 180/m degrees. It is a parameter of this class, so
        the ellipse is not needed (accepted for interface compatibility).
        """
        period = 360.0 / self.m
        return (
            xp.mod(self.k_m_and_shape_angle_from(xp)[1] + period / 2.0, period)
            - period / 2.0
        )

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0, xp=np
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points perturbed by the multipole, with the
        sky orientation `phi_m = phi_ellipse - s_m` taken from the ellipse being perturbed.
        """
        k_scaled, shape_angle = self.k_m_and_shape_angle_from(xp)
        return points_perturbed_from(
            points=points,
            ellipse=ellipse,
            k_m=k_scaled * self.major_axis,
            phi_m=ellipse.angle(xp=xp) - shape_angle,
            m=self.m,
            xp=xp,
        )
