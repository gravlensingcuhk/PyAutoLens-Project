"""
Result-loading helpers for the tiled subhalo analysis
=====================================================

The whole point of the tiled job is **not** to re-run the
source -> light -> mass SLaM chain (days each). Those results are already on
disk under ``output/`` from your submitted ``main_runner_*.py`` runs. This
module loads them back in two complementary ways:

1. ``load_search_output(...)`` returns the lightweight Aggregator
   ``SearchOutput`` for a completed search. This carries everything needed to
   *build a new model* (``.model``, ``.instance``, ``.samples.log_evidence``,
   pickled fit attributes, etc.) and is fast to load.

2. ``load_result(...)`` returns the full ``Result`` object (the same object a
   live run returns) by pointing a search at the completed output directory and
   calling ``search.fit``. PyAutoFit sees the ``.completed`` marker and returns
   immediately via ``result_via_completed_fit`` -- no sampler is re-run. This is
   what you need when downstream code calls Result methods such as
   ``positions_likelihood_from(...)`` or ``model_centred``.

Both rely only on the directory ``Aggregator.from_directory`` API, so they work
with zipped or unzipped outputs alike.

Note: the directory aggregator lives at ``autofit.aggregator.aggregator.Aggregator``.
``af.Aggregator`` is the *database* aggregator (it only has ``add_directory``, not
``from_directory``), so it must not be used here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import autofit as af
from autofit.aggregator.aggregator import Aggregator


def _aggregator_for(output_path: str | Path, path_prefix: str) -> Aggregator:
    """
    Aggregate all searches under ``<output_path>/<path_prefix>``.

    Zipped results are auto-extracted on first encounter (the same behaviour
    the database script relies on).
    """
    directory = Path(output_path) / path_prefix
    if not directory.exists():
        raise FileNotFoundError(
            f"Expected completed pipeline output at {directory} but it does not exist."
        )
    return Aggregator.from_directory(directory=str(directory), completed_only=False)


def load_search_output(
    output_path: str | Path,
    path_prefix: str,
    unique_tag: str,
    name: str,
):
    """
    Return the single Aggregator ``SearchOutput`` matching
    ``(unique_tag, name)`` under ``<output_path>/<path_prefix>``.

    Raises
    ------
    FileNotFoundError
        If no matching search is found (e.g. the upstream pipeline has not
        finished / was written to a different path_prefix).
    """
    agg = _aggregator_for(output_path=output_path, path_prefix=path_prefix)

    matched = [
        so
        for so in agg
        if getattr(so, "unique_tag", None) == unique_tag
        and getattr(so, "name", None) == name
    ]

    if not matched:
        # Helpful error: list what IS there so path_prefix/name typos are obvious.
        available = sorted(
            {
                (getattr(so, "unique_tag", None), getattr(so, "name", None))
                for so in agg
            }
        )
        raise FileNotFoundError(
            f"No search found with unique_tag={unique_tag!r}, name={name!r} "
            f"under {Path(output_path) / path_prefix}.\n"
            f"Available (unique_tag, name) pairs: {available}"
        )

    if len(matched) > 1:
        # Same (unique_tag, name) should be unique; pick the most recently completed.
        matched.sort(key=lambda so: so.directory.stat().st_mtime, reverse=True)

    return matched[0]


def load_result(
    output_path: str | Path,
    path_prefix: str,
    unique_tag: str,
    name: str,
    analysis: Optional[af.Analysis] = None,
):
    """
    Return the full ``Result`` of a completed search without re-running it.

    This reconstructs a search pointing at the existing output directory and
    calls ``.fit``. Because the directory contains a ``.completed`` marker,
    PyAutoFit returns instantly via ``result_via_completed_fit``.

    Parameters
    ----------
    analysis
        The original ``Analysis`` is only required if you intend to call Result
        methods that recompute a fit (e.g. ``max_log_likelihood_fit``) or
        positions from the mass model. For plain model/instance/evidence access
        it can be ``None``. For subhalo tiling we pass the real analysis so
        ``positions_likelihood_from`` / adapt-image lookups work.

    Notes
    -----
    The search type/settings do not need to match the original exactly to load
    a completed result; we use a bare ``af.Nautilus`` purely as a vehicle to
    reach the on-disk result. Any NonLinearSearch subclass with the same
    ``name`` / ``path_prefix`` / ``unique_tag`` resolves to the same directory.
    """
    # Validate the output exists and is complete before constructing the search.
    search_output = load_search_output(
        output_path=output_path,
        path_prefix=path_prefix,
        unique_tag=unique_tag,
        name=name,
    )
    if not search_output.is_complete:
        raise RuntimeError(
            f"Search {name!r} for {unique_tag!r} under {path_prefix!r} is not "
            f"marked complete (.completed missing). The upstream pipeline must "
            f"finish before the subhalo tiles run."
        )

    search = af.Nautilus(
        name=name,
        path_prefix=path_prefix,
        unique_tag=unique_tag,
        n_live=1,  # never used: fit is already complete
        n_batch=1,
    )

    # We do not have the original model object to hand, but for a completed fit
    # ``result_via_completed_fit`` loads the model from disk via paths. Passing
    # the persisted model (search_output.model) keeps the resume path honest if
    # it is needed; otherwise fit() reconstructs it.
    model = search_output.model

    if analysis is None:
        # Construct a no-op analysis sufficient to build the Result. PyAutoFit
        # only needs analysis.make_result; use the abstract base.
        raise ValueError(
            "load_result requires a concrete `analysis` to reconstruct the "
            "Result object (positions_likelihood_from etc.). For lightweight "
            "access use load_search_output(...) instead."
        )

    return search.fit(model=model, analysis=analysis)


def log_evidence_of(search_output) -> float:
    """Safely read log evidence from a SearchOutput (None -> NaN)."""
    try:
        return float(search_output.samples.log_evidence)
    except Exception:
        return float("nan")
