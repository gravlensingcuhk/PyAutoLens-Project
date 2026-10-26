from dataclasses import dataclass
from functools import partial
import numpy as np
from typing import Optional, List, Tuple


def psf_weighted_data_from(
    weight_map_native: np.ndarray,
    kernel_native: np.ndarray,
    native_index_for_slim_index,
    xp=np,
) -> np.ndarray:
    """
    The sparse linear algebra uses a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When it is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `psf_weighted_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix psf_weighted_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    weight_map_native
        The two dimensional masked weight-map of values the PSF convolution is computed from, which is the data
        divided by the noise-map squared.
    kernel_native
        The two dimensional PSF kernel that `psf_weighted_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """
    Ky, Kx = kernel_native.shape
    ph, pw = Ky // 2, Kx // 2

    # 2) pad so neighbourhood gathers never go OOB
    padded = xp.pad(
        weight_map_native, ((ph, ph), (pw, pw)), mode="constant", constant_values=0.0
    )

    # 3) build broadcasted neighbourhood indices for all requested pixels
    # shift pixel coords into the padded frame
    ys = native_index_for_slim_index[:, 0] + ph  # (N,)
    xs = native_index_for_slim_index[:, 1] + pw  # (N,)

    # kernel-relative offsets
    dy = xp.arange(Ky) - ph  # (Ky,)
    dx = xp.arange(Kx) - pw  # (Kx,)

    # broadcast to (N, Ky, Kx)
    Y = ys[:, None, None] + dy[None, :, None]
    X = xs[:, None, None] + dx[None, None, :]

    # 4) gather patches and correlate (no kernel flip)
    patches = padded[Y, X]  # (N, Ky, Kx)
    return xp.sum(patches * kernel_native[None, :, :], axis=(1, 2))  # (N,)


def data_vector_via_psf_weighted_data_from(
    psf_weighted_data: np.ndarray,  # (M_pix,) float64
    rows: np.ndarray,  # (nnz,) int32  each triplet's data pixel (slim index)
    cols: np.ndarray,  # (nnz,) int32  source pixel index
    vals: np.ndarray,  # (nnz,) float64 mapping weights incl sub_fraction
    S: int,  # number of source pixels
    xp=np,
) -> np.ndarray:
    r"""
    Returns the data vector `D` from the `psf_weighted_data` matrix (see `psf_weighted_data_from`), which encodes the
    the 1D image `d` and 1D noise-map values `\sigma` (see Warren & Dye 2003).

    This uses the sparse matrix triplet representation of the mapping matrix to efficiently compute the data vector
    without having to compute the full mapping matrix.

    Computes:
        D[p] = sum_{triplets t with col_t=p} vals[t] * weighted_data_slim[slim_rows[t]]

    Parameters
    ----------
    psf_weighted_data
        The matrix representing the PSF convolution of the imaging data divided by the noise-map squared.
    rows
        The row indices of the sparse mapping matrix triplet representation, which map to data pixels.
    cols
        The column indices of the sparse mapping matrix triplet representation, which map to source pixels.
    vals
        The values of the sparse mapping matrix triplet representation, which map the image pixels to source pixels.
    S
        The number of source pixels.
    xp
        The array module: `np` scatter-adds with `np.bincount` (no JAX import, so this runs without
        JAX installed); `jax.numpy` uses `jax.ops.segment_sum`. The triplets must be masked (see
        `mapper_util.sparse_triplets_from(pix_sizes_for_sub=...)`) so every column is non-negative.
    """
    if xp is np:
        w = np.asarray(psf_weighted_data)[np.asarray(rows)]  # (nnz,)
        contrib = np.asarray(vals) * w  # (nnz,)
        return np.bincount(np.asarray(cols), weights=contrib, minlength=S)[:S]  # (S,)

    from jax.ops import segment_sum

    w = psf_weighted_data[rows]  # (nnz,)
    contrib = vals * w  # (nnz,)
    return segment_sum(contrib, cols, num_segments=S)  # (S,)


def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix: np.ndarray, image: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    r"""
    Returns the data vector `D` from a blurred mapping matrix `f` and the 1D image `d` and 1D noise-map $\sigma$`
    (see Warren & Dye 2003).

    Parameters
    ----------
    blurred_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    return (image / noise_map**2.0) @ blurred_mapping_matrix


def data_linear_func_matrix_from(
    curvature_weights_matrix: np.ndarray, kernel_native, mask
) -> np.ndarray:
    """
    Returns a matrix that for each data pixel, maps it to the sum of the values of a linear object function convolved
    with the PSF kernel at the data pixel.

    If a linear function in an inversion is fixed, its values can be evaluated and preloaded beforehand. For every
    data pixel, the PSF convolution with this preloaded linear function can also be preloaded, in a matrix of
    shape [data_pixels, 1].

    Given that multiple linear functions can be used and fixed in an inversion, this matrix is extended to have
    dimensions [data_pixels, total_fixed_linear_functions].

    When mapper objects and linear functions are used simultaneously in an inversion, this preloaded matrix
    significantly speed up the computation of their off-diagonal terms in the curvature matrix.

    This is similar to the preloading performed via the w-tilde formalism, except that there it is the PSF convolved
    values of each noise-map value pair that are preloaded.

    In **PyAutoGalaxy** and **PyAutoLens**, this preload is used when linear light profiles are fixed in the model.
    For example, when using a multi Gaussian expansion, the values defining how those Gaussians are evaluated
    (e.g. `centre`, `ell_comps` and `sigma`) are often fixed in a model, meaning this matrix can be preloaded and
    used for speed up.

    Parameters
    ----------
    curvature_weights_matrix
        The operated values of each linear function divided by the noise-map squared, in a matrix of shape
        [data_pixels, total_fixed_linear_functions].
    kernel_native
        The 2D PSF kernel.

    Returns
    -------
    ndarray
        A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum of
        the values of a linear object function convolved with the PSF kernel at the data pixel.
    """

    ny, nx = mask.shape_native
    n_unmasked, n_funcs = curvature_weights_matrix.shape

    # Expand masked -> native grid
    native = np.zeros((ny, nx, n_funcs))
    native[~mask] = curvature_weights_matrix  # put values into unmasked positions

    # Convolve each function with PSF kernel
    from scipy.signal import fftconvolve

    blurred_list = []
    for i in range(n_funcs):
        blurred = fftconvolve(native[..., i], kernel_native, mode="same")
        # Re-mask: only keep unmasked pixels
        blurred_list.append(blurred[~mask])

    return np.stack(blurred_list, axis=1)  # shape (n_unmasked, n_funcs)


# Rows of the sparse contraction of `sparse_mapping_transpose_dot_from` processed per
# `fori_loop` step on the JAX backend. The step gathers a `(chunk, S_tot)` fp64 block of the
# dense operand, so the chunk bounds the scratch memory per batch element (16384 rows x 1000
# columns x 8 bytes = 130 MB) independently of nnz. The product itself is memory-bound, so
# the chunk size changes the launch count, not the arithmetic.
SPARSE_TRANSPOSE_DOT_CHUNK = 16384

# Columns of the noise-weighted blurred mapping matrix sent through the transposed FFT blur per
# block in `curvature_matrix_via_sparse_transpose_from`. The blur holds a complex FFT cube of
# (fft_shape x columns) per batch element -- complex128 for the default fp64 transposed blur,
# where blurring all 970 columns of an HST-sized inversion at once cost ~1 GB per element and
# put a batch of 20 out of memory on a 48 GB card (2026-10-26) -- so the columns go through in
# blocks of this size, which bounds the cube whatever the mesh size. The arithmetic is unchanged.
SPARSE_TRANSPOSE_BLUR_COLUMNS = 128


def sparse_mapping_transpose_dot_from(
    rows,
    cols,
    vals,
    dense,
    n_cols: int,
    xp=np,
    chunk: int = SPARSE_TRANSPOSE_DOT_CHUNK,
):
    r"""
    Returns ``A^T @ dense`` for a sparse matrix ``A`` given as COO triplets, without forming ``A``.

    ``A`` has ``A[rows[t], cols[t]] += vals[t]`` for every triplet ``t`` (the form returned by
    `Mapper.sparse_triplets_data`, rows on the slim image grid, repeated row/column pairs allowed),
    so

        out[c, :] = sum_{t : cols[t] == c} vals[t] * dense[rows[t], :]

    a gather of the dense operand's rows followed by a scatter-add over the source pixels. The
    arithmetic is ``nnz * dense.shape[1]`` multiply-adds, against ``N_rows * S * dense.shape[1]``
    for the dense ``A.T @ dense``: for a bilinear pixelization stencil nnz is 4 sub-pixels per
    image pixel, so the saving is the number of source pixels over ``4 * sub_size**2``.

    Parameters
    ----------
    rows
        Row (slim image pixel) index of every triplet, shape ``(nnz,)``.
    cols
        Column (source pixel) index of every triplet, shape ``(nnz,)``; masked-out stencil slots
        must carry a zero value and a valid (non-negative) column, as `mapper_util.sparse_triplets_from`
        guarantees.
    vals
        Value of every triplet, shape ``(nnz,)``, including any sub-pixel normalisation.
    dense
        The dense right operand, shape ``(N_rows, K)``; only its rows indexed by `rows` are read.
    n_cols
        The number of columns of ``A`` (source pixels), the number of rows of the result.
    xp
        `np` scatter-adds with `np.add.at`; `jax.numpy` uses `jax.ops.segment_sum` over fixed-size
        chunks of the triplet axis inside a `lax.fori_loop`, so the `(chunk, K)` gather is bounded in
        memory whatever nnz is and the loop is `vmap`-safe (static shapes; the padding triplets
        contribute zero to column 0).
    chunk
        Triplets per loop step on the JAX backend (see `SPARSE_TRANSPOSE_DOT_CHUNK`).

    Returns
    -------
    ndarray of shape ``(n_cols, K)``.
    """
    if xp is np:
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        vals = np.asarray(vals, dtype=np.float64)
        dense = np.asarray(dense, dtype=np.float64)
        out = np.zeros((n_cols, dense.shape[1]), dtype=np.float64)
        np.add.at(out, cols, vals[:, None] * dense[rows])
        return out

    import jax
    from jax.ops import segment_sum

    nnz = rows.shape[0]
    k = dense.shape[1]

    if nnz <= chunk:
        return segment_sum(vals[:, None] * dense[rows], cols, num_segments=n_cols)

    n_chunks = -(-nnz // chunk)
    pad = n_chunks * chunk - nnz

    rows = xp.pad(rows, (0, pad))
    cols = xp.pad(cols, (0, pad))
    vals = xp.pad(vals, (0, pad))

    def body(i, out):
        start = i * chunk
        rows_c = jax.lax.dynamic_slice(rows, (start,), (chunk,))
        cols_c = jax.lax.dynamic_slice(cols, (start,), (chunk,))
        vals_c = jax.lax.dynamic_slice(vals, (start,), (chunk,))
        contrib = vals_c[:, None] * dense[rows_c]
        return out + segment_sum(contrib, cols_c, num_segments=n_cols)

    out = xp.zeros((n_cols, k), dtype=dense.dtype)

    return jax.lax.fori_loop(0, n_chunks, body, out)


def curvature_matrix_via_sparse_transpose_from(
    operated_mapping_matrix,
    noise_map,
    psf,
    mask,
    mapper_triplets_list: List[Tuple],
    mapper_param_range_list: List[List[int]],
    settings,
    add_to_curvature_diag: bool = False,
    no_regularization_index_list: Optional[List] = None,
    xp=np,
):
    r"""
    Returns the curvature matrix ``F`` of an imaging inversion through its mappers' sparse
    unblurred mapping, instead of the dense ``(B / \sigma)^T (B / \sigma)`` product of the blurred
    mapping matrix ``B`` (`inversion_util.curvature_matrix_via_mapping_matrix_from`).

    **The identity.** For a mapper block ``m`` of the linear parameters, ``B[:, m] = P A_m`` with ``P``
    the PSF convolution restricted to the mask and ``A_m`` the mapper's unblurred mapping matrix, so

        F[m, :] = B[:, m]^T \Sigma^{-1} B = A_m^T (P^T \Sigma^{-1} B)

    with ``\Sigma^{-1} = diag(1 / \sigma^2)``. The bracket is one convolution of the noise-weighted
    ``B`` with the *transposed* PSF (`Convolver.transposed`: the kernel flipped on both axes, which
    turns the convolution into the correlation ``P^T``, with one zero row or column appended on an
    even axis so the flipped centre stays on the transposed positions; embedding the slim matrix in
    the mask before the FFT and slimming after is the transpose of the forward blur's restriction to
    the mask, because ``\Sigma^{-1}`` is zero outside it), and ``A_m^T`` applied to it is the sparse
    contraction `sparse_mapping_transpose_dot_from` over the mapper's triplets
    (`Mapper.sparse_triplets_data`). Rows of linear-func objects are the transpose of the mapper
    rows just computed, and their own ``[f, f]`` block is the dense product of their few columns.
    With several mappers each mapper's rows cover every column, so the mapper-mapper cross blocks
    come out of the same contraction.

    **Memory.** The transposed blur goes through the columns in blocks of
    `SPARSE_TRANSPOSE_BLUR_COLUMNS`, so its FFT cube (complex128 by default, twice the forward blur's
    complex64) is bounded per batch element whatever the mesh size; all columns at once cost ~1 GB per
    element for an HST lens at a 30 x 30 mesh and put a batch of 20 out of memory on a 48 GB card.

    **Cost.** The dense product is ``N_pix * S_tot**2`` multiply-adds (for an HST lens at 24.5k image
    pixels and a 30 x 30 mesh plus 70 linear-light columns, 0.09 s of a 0.19 s likelihood call on an
    A6000, and it grows with the square of the mesh size). This route costs one extra FFT blur of
    ``S_tot`` columns (``~ S_tot``, the same as the forward blur) and ``nnz(A) * S_tot`` for the
    contraction, with nnz four entries per over-sampled sub-pixel for a bilinear stencil. It is
    exact in exact arithmetic, so it changes the speed of a likelihood call and nothing else.

    **Precision.** ``B`` arrives fp64 (the forward blur upcasts at its kernel multiply even under
    `Settings.use_mixed_precision`); the noise weighting, the transposed blur and the contraction are
    fp64 unless `settings.sparse_curvature_mixed_precision` asks for the fp32-cube / complex64-forward
    FFT variant of the transposed blur. ``F`` feeds the positive-only solve and a log-determinant,
    both of which amplify fp32 round-off into nats (an fp32 ``F`` shifted an HST-sized pixelized fit
    by several nats), so the default keeps the whole route in double.

    Parameters
    ----------
    operated_mapping_matrix
        The concatenated blurred mapping matrix ``B`` of every linear object, shape ``(N_pix, S_tot)``.
    noise_map
        The slim noise map ``\sigma``, shape ``(N_pix,)``.
    psf
        The dataset's `Convolver`; its `transposed` convolver performs the transposed blur.
    mask
        The dataset's `Mask2D`, the grid the blur embeds the slim matrices in.
    mapper_triplets_list
        One ``(rows, cols, vals)`` tuple per mapper, in the order of `mapper_param_range_list`.
    mapper_param_range_list
        The ``[start, stop)`` column range of each mapper in ``B``
        (`AbstractInversion.param_range_list_from(cls=Mapper)`).
    settings
        The inversion `Settings` (`sparse_curvature_mixed_precision`,
        `no_regularization_add_to_curvature_diag_value`).
    add_to_curvature_diag
        If `True`, `settings.no_regularization_add_to_curvature_diag_value` is added to the diagonal
        entries listed in `no_regularization_index_list`, as the dense route does.
    no_regularization_index_list
        The parameter indices with no regularization (linear-func objects).
    xp
        The array module; the NumPy branch is a reference implementation (the dense product is
        cheaper there), the JAX branch the production one.

    Returns
    -------
    ndarray of shape ``(S_tot, S_tot)``, the curvature matrix ``F``.
    """
    from autoarray.inversion.inversion import inversion_util

    B = xp.asarray(operated_mapping_matrix, dtype=xp.float64)
    noise_map = xp.asarray(noise_map, dtype=xp.float64)
    s_tot = B.shape[1]

    # P^T Sigma^-1 B : one transposed blur of every column, in column blocks so the FFT cube
    # stays bounded (see `SPARSE_TRANSPOSE_BLUR_COLUMNS`).
    Y = B / noise_map[:, None] ** 2
    transposed = psf.transposed
    Z = xp.hstack(
        [
            xp.asarray(
                transposed.convolved_mapping_matrix_from(
                    mapping_matrix=Y[:, start : start + SPARSE_TRANSPOSE_BLUR_COLUMNS],
                    mask=mask,
                    use_mixed_precision=settings.sparse_curvature_mixed_precision,
                    xp=xp,
                ),
                dtype=xp.float64,
            )
            for start in range(0, s_tot, SPARSE_TRANSPOSE_BLUR_COLUMNS)
        ]
    )

    if xp is np:
        F = np.zeros((s_tot, s_tot), dtype=np.float64)
    else:
        F = xp.zeros((s_tot, s_tot), dtype=xp.float64)

    mapper_indices = []

    for (rows, cols, vals), (start, stop) in zip(
        mapper_triplets_list, mapper_param_range_list
    ):
        block = sparse_mapping_transpose_dot_from(
            rows=rows, cols=cols, vals=vals, dense=Z, n_cols=stop - start, xp=xp
        )
        # The mapper's own block is symmetric in exact arithmetic; the dense product returns
        # it exactly symmetric, and the Cholesky-based log-determinant assumes it.
        diag = block[:, start:stop]
        diag = 0.5 * (diag + diag.T)
        if xp is np:
            block[:, start:stop] = diag
            F[start:stop, :] = block
        else:
            block = block.at[:, start:stop].set(diag)
            F = F.at[start:stop, :].set(block)
        mapper_indices += list(range(start, stop))

    func_indices = [i for i in range(s_tot) if i not in set(mapper_indices)]

    if func_indices:
        f_idx = np.asarray(func_indices)
        m_idx = np.asarray(mapper_indices)

        # Rows of the linear-func objects: the transpose of the mapper rows already computed ...
        cross = F[xp.ix_(m_idx, f_idx)].T if xp is np else F[m_idx][:, f_idx].T
        # ... and their own block, dense over their few columns.
        Bf = B[:, f_idx] / noise_map[:, None]
        own = Bf.T @ Bf

        if xp is np:
            F[np.ix_(f_idx, m_idx)] = cross
            F[np.ix_(f_idx, f_idx)] = own
        else:
            F = F.at[xp.ix_(f_idx, m_idx)].set(cross)
            F = F.at[xp.ix_(f_idx, f_idx)].set(own)

    if (
        add_to_curvature_diag
        and no_regularization_index_list
        and len(no_regularization_index_list) > 0
    ):
        F = inversion_util.curvature_matrix_with_added_to_diag_from(
            curvature_matrix=F,
            value=settings.no_regularization_add_to_curvature_diag_value,
            no_regularization_index_list=no_regularization_index_list,
            xp=xp,
        )

    return F


def curvature_matrix_mirrored_from(
    curvature_matrix,
    *,
    xp=np,
):
    """
    Mirror a curvature matrix so that any non-zero entry C[i,j]
    is copied to C[j,i].

    Supports:
      - NumPy (xp=np)
      - JAX  (xp=jax.numpy)

    Parameters
    ----------
    curvature_matrix : (N, N) array
        Possibly triangular / partially-filled curvature matrix.

    xp : module
        np or jax.numpy

    Returns
    -------
    curvature_matrix_mirrored : (N, N) array
        Symmetric curvature matrix.
    """

    # Ensure array type
    C = curvature_matrix

    # Boolean mask of where entries exist
    mask = C != 0

    # Copy entries symmetrically
    C_T = xp.swapaxes(C, 0, 1)

    # Prefer non-zero values from either side
    curvature_matrix_mirrored = xp.where(mask, C, C_T)

    return curvature_matrix_mirrored


def curvature_matrix_with_added_to_diag_from(
    curvature_matrix,
    value: float,
    no_regularization_index_list: Optional[List[int]] = None,
    *,
    xp=np,
):
    """
    Add a small stabilizing value to the diagonal entries of the curvature matrix.

    Supports:
      - NumPy (xp=np): in-place update
      - JAX  (xp=jax.numpy): functional `.at[].add()`

    Parameters
    ----------
    curvature_matrix : (N, N) array
        Curvature matrix to modify.

    value : float
        Value added to selected diagonal entries.

    no_regularization_index_list : list of int
        Indices where diagonal should be boosted.

    xp : module
        np or jax.numpy

    Returns
    -------
    curvature_matrix : array
        Updated matrix (new array in JAX, modified in NumPy).
    """

    if no_regularization_index_list is None:
        return curvature_matrix

    inds = xp.asarray(no_regularization_index_list, dtype=xp.int32)

    if xp is np:
        # -----------------------
        # NumPy: in-place update
        # -----------------------
        curvature_matrix[inds, inds] += value
        return curvature_matrix

    else:
        # -----------------------
        # JAX: functional update
        # -----------------------
        return curvature_matrix.at[inds, inds].add(value)


def mapped_reconstructed_operated_data_via_sparse_operator_from(
    reconstruction,  # (S,)
    rows,
    cols,
    vals,  # (nnz,)
    fft_index_for_masked_pixel,
    data_shape: int,  # y_shape * x_shape
    xp=np,
):
    """
    Returns the slim image `A s` of a reconstruction `s` mapped through a mapper's sparse mapping
    operator `A`, whose COO triplets `(rows, cols, vals)` index rows on the rectangular FFT grid
    (`Mapper.sparse_triplets_curvature`).

    The scatter-add is performed on the rectangular grid of `data_shape` and the masked pixels are
    then gathered via `fft_index_for_masked_pixel`. With `xp=np` it uses `np.bincount` and imports
    no JAX, so it runs without JAX installed; with `xp=jax.numpy` it uses `jax.ops.segment_sum`.
    """
    num_segments = data_shape[0] * data_shape[1]

    if xp is np:
        reconstruction = np.asarray(reconstruction, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)

        contrib = vals * reconstruction[cols]  # (nnz,)
        image_rect = np.bincount(rows, weights=contrib, minlength=num_segments)[
            :num_segments
        ]  # (M_rect,)

        return image_rect[np.asarray(fft_index_for_masked_pixel)]  # (M_pix,)

    import jax.numpy as jnp
    from jax.ops import segment_sum

    reconstruction = jnp.asarray(reconstruction, dtype=jnp.float64)
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    contrib = vals * reconstruction[cols]  # (nnz,)
    image_rect = segment_sum(contrib, rows, num_segments=num_segments)  # (M_rect,)

    image_slim = image_rect[fft_index_for_masked_pixel]  # (M_pix,)
    return image_slim


@dataclass(frozen=True)
class ImagingSparseOperator:

    data_native: np.ndarray
    noise_map_native: np.ndarray
    weight_map: np.ndarray
    inverse_variances_native: "jax.Array"  # (y, x) float64
    y_shape: int
    x_shape: int
    Ky: int
    Kx: int
    fft_shape: Tuple[int, int]
    batch_size: int
    col_offsets: "jax.Array"  # (batch_size,) int32
    Khat_r: "jax.Array"  # (Fy, Fx//2+1) complex
    Khat_flip_r: "jax.Array"  # (Fy, Fx//2+1) complex

    @classmethod
    def from_noise_map_and_psf(
        cls,
        data,
        noise_map,
        psf,
        *,
        batch_size: int = 128,
        dtype=None,
    ) -> "ImagingSparseOperator":
        """
        Construct an `ImagingSparseOperator` from imaging arrays and a PSF.

        This factory method builds all static FFT state required to apply

            W = Hᵀ N⁻¹ H

        repeatedly during curvature matrix construction:

        - `inverse_variances_native` is computed from `noise_map` as 1/noise^2 (masked-safe).
        - `weight_map` is computed from `data` as data/noise^2 (masked-safe).
        - `Khat_r` and `Khat_flip_r` are precomputed as rFFT transforms of the padded PSF
          and its flipped version on the required FFT shape.

        Parameters
        ----------
        data
            Imaging data object (e.g. `Array2D`) providing `.array` (slim) and `.native`.
        noise_map
            Imaging noise-map object (e.g. `Array2D`) providing `.array` (slim),
            `.native`, `.mask`, and `.shape_native`.
        psf
            PSF kernel in native 2D form. Can be NumPy or JAX convertible.
        batch_size
            Number of curvature columns processed per FFT batch. Larger values can improve
            GPU utilization but increase VRAM usage.
        dtype
            JAX dtype for PSF / FFT precompute. Defaults to `jax.numpy.float64`.

        Returns
        -------
        ImagingSparseOperator
            Fully initialized operator containing all precomputed FFT state.

        Notes
        -----
        - This method assumes the FFT shape is `(Hy+Ky-1, Hx+Kx-1)` which is sufficient
          for linear convolution followed by cropping to “same”.
        - Mask handling: the `Array2D(..., mask=...)` wrapper ensures masked pixels are
          correctly represented in native arrays; you may still explicitly zero masked
          pixels if your downstream code expects it.
        """
        import jax.numpy as jnp

        from autoarray.structures.arrays.uniform_2d import Array2D

        if dtype is None:
            dtype = jnp.float64

        # ----------------------------
        # Shapes
        # ----------------------------
        y_shape = int(noise_map.shape_native[0])
        x_shape = int(noise_map.shape_native[1])

        # ----------------------------
        # inverse_variances_native (native 2D)
        # Make safe (0 where invalid)
        # ----------------------------
        # Try to get a plain native ndarray from your Array2D-like object:
        inverse_variances_native = 1.0 / noise_map**2
        inverse_variances_native = Array2D(
            values=inverse_variances_native, mask=noise_map.mask
        )
        inverse_variances_native = inverse_variances_native.native

        weight_map = data.array / (noise_map.array**2)
        weight_map = Array2D(values=weight_map, mask=noise_map.mask)

        # If you *also* want to zero masked pixels explicitly:
        # mask_native = noise_map.mask  (depends on your API; might be bool native)
        # inverse_variances_native = inverse_variances_native.at[mask_native].set(0.0)

        # ----------------------------
        # PSF + FFT precompute
        # ----------------------------
        psf = jnp.asarray(psf, dtype=dtype)
        Ky, Kx = map(int, psf.shape)

        fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)
        Fy, Fx = fft_shape

        def precompute(psf2d):
            psf_pad = jnp.pad(psf2d, ((0, Fy - Ky), (0, Fx - Kx)))
            return jnp.fft.rfft2(psf_pad, s=(Fy, Fx))

        Khat_r = precompute(psf)
        Khat_flip_r = precompute(jnp.flip(psf, axis=(0, 1)))

        return cls(
            data_native=data.native,
            noise_map_native=noise_map.native,
            weight_map=weight_map.native,
            inverse_variances_native=inverse_variances_native,
            y_shape=y_shape,
            x_shape=x_shape,
            Ky=Ky,
            Kx=Kx,
            fft_shape=(int(Fy), int(Fx)),
            batch_size=int(batch_size),
            col_offsets=jnp.arange(batch_size, dtype=jnp.int32),
            Khat_r=Khat_r,
            Khat_flip_r=Khat_flip_r,
        )

    def apply_operator(self, Fbatch_flat):
        """
        Apply the imaging precision operator W = Hᵀ N⁻¹ H to a batch of vectors.

        This is the fundamental linear operator application used throughout curvature
        assembly. Given an input matrix `Fbatch_flat` with shape:

            (M_rect, B)

        where:
        - `M_rect = y_shape * x_shape` is the number of pixels on the rectangular FFT grid,
        - `B` is the number of right-hand-side vectors (typically `batch_size`),

        this method computes:

            G = W Fbatch_flat

        using two FFT-based convolutions:
        1) Forward blur:   H (correlation with PSF)
        2) Weighting:      multiply by inverse variances (N⁻¹)
        3) Backproject:    Hᵀ (correlation with flipped PSF)

        Parameters
        ----------
        Fbatch_flat
            Array of shape (M_rect, B) representing B vectors on the rectangular grid.

        Returns
        -------
        ndarray
            Array of shape (M_rect, B) equal to W applied to the batch.

        Notes
        -----
        - The PSF is applied via rFFT on a padded grid of shape `fft_shape`.
        - The output is cropped back to the native `(y_shape, x_shape)` “same” region.
        - This method expects rectangular-grid indexing (flat). If you have slim masked
          indexing you must scatter/gather appropriately outside this function.
        """
        import jax.numpy as jnp

        y_shape, x_shape = self.y_shape, self.x_shape
        Ky, Kx = self.Ky, self.Kx
        Fy, Fx = self.fft_shape
        M = y_shape * x_shape

        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))

        # forward blur
        Fpad = jnp.pad(Fimg, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
        Fhat = jnp.fft.rfft2(Fpad, s=(Fy, Fx))
        blurred_pad = jnp.fft.irfft2(Fhat * self.Khat_r[None, :, :], s=(Fy, Fx))

        cy, cx = Ky // 2, Kx // 2
        blurred = blurred_pad[:, cy : cy + y_shape, cx : cx + x_shape]

        weighted = blurred * self.inverse_variances_native[None, :, :]

        # backprojection
        Wpad = jnp.pad(weighted, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
        What = jnp.fft.rfft2(Wpad, s=(Fy, Fx))
        back_pad = jnp.fft.irfft2(What * self.Khat_flip_r[None, :, :], s=(Fy, Fx))
        back = back_pad[:, cy : cy + y_shape, cx : cx + x_shape]

        return back.reshape((B, M)).T  # (M, B)

    def curvature_matrix_diag_from(self, rows, cols, vals, *, S: int):
        """
        Compute the diagonal (mapper–mapper) curvature matrix block F = Aᵀ W A.

        This method computes the curvature matrix for a single mapper/operator `A`
        represented in COO triplets (rows, cols, vals), using the FFT-backed W operator.

        Conceptually, if A is the (M_rect × S) mapping matrix, then:

            F = Aᵀ W A

        where:
        - `S` is the number of source pixels / parameters for this mapper.
        - `M_rect = y_shape * x_shape` is the number of rectangular grid pixels.

        The computation proceeds in column blocks of width `batch_size`:
        1) Assemble Fbatch = A[:, start:start+B] on the rectangular grid via scatter-add.
        2) Apply W to the block: Gbatch = W(Fbatch).
        3) Project back with Aᵀ via a segment_sum over `cols`.

        Parameters
        ----------
        rows, cols, vals
            COO triplets encoding the sparse mapping operator A.
            Expected conventions:
            - `rows`: rectangular-grid pixel indices (flat), shape (nnz,)
            - `cols`: source pixel indices in [0, S), shape (nnz,)
            - `vals`: mapping weights (incl sub-fraction / interpolation), shape (nnz,)
        S
            Number of source pixels / parameters for this mapper.

        Returns
        -------
        ndarray
            Curvature matrix of shape (S, S), symmetric.

        Notes
        -----
        - The output is symmetrized as 0.5*(F + Fᵀ) to mitigate numerical asymmetry
          introduced by floating-point reductions.
        - Padding to `S_pad = ceil(S/B)*B` ensures `dynamic_update_slice` is always legal
          even when S is not a multiple of batch_size.
        """
        import jax.numpy as jnp
        from jax import lax
        from jax.ops import segment_sum

        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)

        y_shape, x_shape = self.y_shape, self.x_shape
        M = y_shape * x_shape
        B = self.batch_size

        n_blocks = (S + B - 1) // B
        S_pad = n_blocks * B

        C0 = jnp.zeros((S, S_pad), dtype=jnp.float64)

        def body(block_i, C):
            start = block_i * B

            in_block = (cols >= start) & (cols < (start + B))
            bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
            v = jnp.where(in_block, vals, 0.0)

            F = jnp.zeros((M, B), dtype=jnp.float64)
            F = F.at[rows, bc].add(v)

            G = self.apply_operator(F)  # (M, B)

            contrib = vals[:, None] * G[rows, :]
            Cblock = segment_sum(contrib, cols, num_segments=S)  # (S, B)

            width = jnp.minimum(B, jnp.maximum(0, S - start))
            Cblock = Cblock * (self.col_offsets < width)[None, :]

            return lax.dynamic_update_slice(C, Cblock, (0, start))

        C_pad = lax.fori_loop(0, n_blocks, body, C0)
        C = C_pad[:, :S]
        return 0.5 * (C + C.T)

    def curvature_matrix_off_diag_from(
        self, rows0, cols0, vals0, rows1, cols1, vals1, *, S0: int, S1: int
    ):
        """
        Compute the off-diagonal curvature block F01 = A0ᵀ W A1.

        Given two sparse mapping operators:
        - A0 : (M_rect × S0)
        - A1 : (M_rect × S1)

        this method computes:

            F01 = A0ᵀ W A1

        using the same FFT-backed W operator as in the diagonal computation.

        The computation proceeds in column blocks over A1:
        1) Assemble Fbatch = A1[:, start:start+B] on the rectangular grid.
        2) Apply W: Gbatch = W(Fbatch).
        3) Project with A0ᵀ via segment_sum over cols0.

        Parameters
        ----------
        rows0, cols0, vals0
            COO triplets for A0. `rows0` are rectangular-grid indices.
        rows1, cols1, vals1
            COO triplets for A1. `rows1` are rectangular-grid indices.
        S0
            Number of source parameters for mapper/operator 0.
        S1
            Number of source parameters for mapper/operator 1.

        Returns
        -------
        ndarray
            Off-diagonal curvature block of shape (S0, S1).

        Notes
        -----
        - The result is *not* symmetrized here because it is not square in general.
          If you need the symmetric counterpart, use F10 = F01ᵀ when A0/A1 share W.
        - Padding to `S1_pad = ceil(S1/B)*B` ensures `dynamic_update_slice` is always legal.
        """
        import jax.numpy as jnp
        from jax import lax
        from jax.ops import segment_sum

        rows0 = jnp.asarray(rows0, dtype=jnp.int32)
        cols0 = jnp.asarray(cols0, dtype=jnp.int32)
        vals0 = jnp.asarray(vals0, dtype=jnp.float64)

        rows1 = jnp.asarray(rows1, dtype=jnp.int32)
        cols1 = jnp.asarray(cols1, dtype=jnp.int32)
        vals1 = jnp.asarray(vals1, dtype=jnp.float64)

        y_shape, x_shape = self.y_shape, self.x_shape
        M = y_shape * x_shape
        B = self.batch_size

        n_blocks = (S1 + B - 1) // B
        S1_pad = n_blocks * B

        F01_0 = jnp.zeros((S0, S1_pad), dtype=jnp.float64)

        def body(block_i, F01):
            start = block_i * B

            in_block = (cols1 >= start) & (cols1 < (start + B))
            bc = jnp.where(in_block, cols1 - start, 0).astype(jnp.int32)
            v = jnp.where(in_block, vals1, 0.0)

            F = jnp.zeros((M, B), dtype=jnp.float64)
            F = F.at[rows1, bc].add(v)

            G = self.apply_operator(F)  # (M, B)

            contrib = vals0[:, None] * G[rows0, :]
            block = segment_sum(contrib, cols0, num_segments=S0)

            width = jnp.minimum(B, jnp.maximum(0, S1 - start))
            block = block * (self.col_offsets < width)[None, :]

            return lax.dynamic_update_slice(F01, block, (0, start))

        F01_pad = lax.fori_loop(0, n_blocks, body, F01_0)
        return F01_pad[:, :S1]

    def curvature_matrix_off_diag_func_list_from(
        self,
        curvature_weights,  # (M_pix, n_funcs)
        fft_index_for_masked_pixel,  # (M_pix,)  slim -> rect(flat)
        rows,
        cols,
        vals,  # triplets where rows are RECT indices
        *,
        S: int,
    ):
        """
        Compute the mapper–linear-function off-diagonal block: Aᵀ Hᵀ (curvature_weights).

        This method computes the off-diagonal block between a sparse mapper/operator A
        and a set of fixed linear functions (e.g. linear light profiles) whose operated
        images have already been PSF-convolved and noise-weighted.

        The input `curvature_weights` is assumed to already contain:

            curvature_weights = (H B) / noise^2

        evaluated on the *slim masked* grid, for each linear function column.

        The returned matrix is:

            off_diag = Aᵀ [ Hᵀ (curvature_weights_native) ]

        which has shape (S, n_funcs).

        Parameters
        ----------
        curvature_weights
            Array of shape (M_pix, n_funcs) on the *slim masked* grid.
            Each column corresponds to one linear function already convolved by H and
            weighted by inverse variance.
        fft_index_for_masked_pixel
            Array of shape (M_pix,) mapping slim masked pixel indices to rectangular
            FFT-grid flat indices. Used to scatter values onto the rectangular grid.
        rows, cols, vals
            COO triplets for the mapper A, where:
            - `rows` are rectangular-grid indices (flat), shape (nnz,)
            - `cols` are source pixel indices, shape (nnz,)
            - `vals` are mapping weights, shape (nnz,)
        S
            Number of source pixels / parameters in the mapper.

        Returns
        -------
        ndarray
            Off-diagonal block of shape (S, n_funcs).

        Notes
        -----
        - This method performs only one convolution per function column (batched),
          using the flipped PSF (Hᵀ), because `curvature_weights` is assumed to already
          include the forward blur H.
        - No batch_size parameter is required here because we are convolving `n_funcs`
          columns (often small) rather than sweeping over source pixels S.
        """
        import jax.numpy as jnp
        from jax.ops import segment_sum

        curvature_weights = jnp.asarray(curvature_weights, dtype=jnp.float64)
        fft_index_for_masked_pixel = jnp.asarray(
            fft_index_for_masked_pixel, dtype=jnp.int32
        )

        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)

        y_shape, x_shape = self.y_shape, self.x_shape
        Ky, Kx = self.Ky, self.Kx
        Fy, Fx = self.fft_shape
        M_rect = y_shape * x_shape

        M_pix, n_funcs = curvature_weights.shape

        # 1) scatter slim -> rect(flat)
        grid_flat = jnp.zeros((M_rect, n_funcs), dtype=jnp.float64)
        grid_flat = grid_flat.at[fft_index_for_masked_pixel, :].set(curvature_weights)

        # 2) apply H^T: conv with flipped PSF
        images = grid_flat.T.reshape((n_funcs, y_shape, x_shape))  # (B=n_funcs, Hy, Hx)

        # --- rfft conv (same as your helper) ---
        images_pad = jnp.pad(images, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
        Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))
        out_pad = jnp.fft.irfft2(Fhat * self.Khat_flip_r[None, :, :], s=(Fy, Fx))

        cy, cx = Ky // 2, Kx // 2
        back_native = out_pad[
            :, cy : cy + y_shape, cx : cx + x_shape
        ]  # (n_funcs, Hy, Hx)

        # 3) gather at mapper rows (rect coords)
        back_flat = back_native.reshape((n_funcs, M_rect)).T  # (M_rect, n_funcs)
        back_at_rows = back_flat[rows, :]  # (nnz, n_funcs)

        # 4) accumulate to source pixels
        contrib = vals[:, None] * back_at_rows
        return segment_sum(contrib, cols, num_segments=S)  # (S, n_funcs)
