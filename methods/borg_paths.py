"""Resolve BORG MOEA result CSV paths (seed + optional MRF-filtered objective variant) from environment."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from methods.config import BORG_SEED_FULL, BORG_SEED_MRF, NFE, ISLANDS, OUTPUT_DIR


def _first_int_env(keys, default: int) -> int:
    """Return the first environment variable that parses as int; else ``default``."""
    for k in keys:
        s = os.environ.get(k, "").strip()
        if s:
            return int(s)
    return int(default)


def normalize_borg_variant(token: str) -> str:
    """
    Map a user-facing token to canonical Borg run variant: ``full`` | ``regression`` | ``perfect``.

    ``full`` — unfiltered objectives, filenames ``..._seed{N}.csv`` (no ``_mrffiltered_``).

    ``regression`` / ``perfect`` — MRF-filtered objectives, ``..._seed{N}_mrffiltered_regression.csv`` /
    ``..._seed{N}_mrffiltered_perfect.csv``.
    """
    v = token.strip().lower().replace("-", "_")
    aliases = {
        "full": "full",
        "nofilter": "full",
        "none": "full",
        "unfiltered": "full",
        "full_objectives": "full",
        "regression": "regression",
        "reg": "regression",
        "regression_disagg": "regression",
        "mrffiltered_regression": "regression",
        "perfect": "perfect",
        "perfect_foresight": "perfect",
        "perfect_information": "perfect",
        "pi": "perfect",
        "perf": "perfect",
        "mrffiltered_perfect": "perfect",
    }
    if v not in aliases:
        raise ValueError(
            "Unknown Borg variant {!r}; expected one of: {}".format(
                token, ", ".join(sorted(set(aliases.keys())))
            )
        )
    return aliases[v]


def borg_variant_resolve_kwargs(variant: str) -> Dict[str, Any]:
    """
    Keyword arguments for :func:`resolve_borg_moea_csv_path` / ``load_filtered_borg_solution_tables``.

    Seed resolution (first match wins):

    - **full** (unfiltered ``..._seed{N}.csv``) — :func:`resolve_full_borg_seed` uses
      ``pywrdrb.release_policies.config.BORG_SEED_FULL`` (optional ``CEE_BORG_SEED_FULL``), then
      ``BORG_SEED_MRF`` / ``CEE_BORG_SEED_FULL_TRY`` and probes disk (``CEE_BORG_FULL_PROBE_*``).
    - **regression** — ``CEE_BORG_SEED_REGRESSION``, ``CEE_BORG_SEED_MRF``, ``CEE_BORG_SEED``, …
    - **perfect** — ``CEE_BORG_SEED_PERFECT``, ``CEE_BORG_SEED_MRF``, ``CEE_BORG_SEED``, …
    """
    v = normalize_borg_variant(variant)
    if v == "full":
        return {
            "borg_seed": resolve_full_borg_seed(),
            "borg_mrf_filtered": False,
            "borg_mrf_filter_tag": None,
        }
    if v == "regression":
        return {
            "borg_seed": _first_int_env(
                ("CEE_BORG_SEED_REGRESSION", "CEE_BORG_SEED_MRF", "CEE_BORG_SEED", "CEE_SEED"),
                BORG_SEED_MRF,
            ),
            "borg_mrf_filtered": True,
            "borg_mrf_filter_tag": "regression_disagg",
        }
    if v == "perfect":
        return {
            "borg_seed": _first_int_env(
                ("CEE_BORG_SEED_PERFECT", "CEE_BORG_SEED_MRF", "CEE_BORG_SEED", "CEE_SEED"),
                BORG_SEED_MRF,
            ),
            "borg_mrf_filtered": True,
            "borg_mrf_filter_tag": "perfect",
        }
    raise ValueError("unsupported variant {!r}".format(variant))


def borg_moea_csv_dir() -> str:
    """
    Directory containing ``MMBorg_*M_*_*_nfe*_seed*.csv``.

    Default: project ``outputs/`` (``OUTPUT_DIR``). Override when CSVs live elsewhere, e.g.::

        export CEE_BORG_OUTPUT_DIR=/path/to/dir/with/MMBorg_csvs
    """
    v = os.environ.get("CEE_BORG_OUTPUT_DIR", "").strip()
    return os.path.abspath(v) if v else OUTPUT_DIR


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def borg_mrf_filtered_enabled() -> bool:
    """
    True when Borg outputs should use the MRF-filtered objective suffix.
    """
    return _env_truthy("CEE_BORG_MRF_FILTERED")


def _normalize_mrf_filter_tag_for_suffix(tag: str) -> str:
    """
    Map ``CEE_MRF_FILTER_TAG`` / ``CEE_MRF_FILTER_SOURCE`` values to the canonical
    filesystem segment used in ``*_mrffiltered_<segment>.csv`` names:

    - ``regression`` — pub-reconstruction MRF filter (``_mrffiltered_regression``)
    - ``perfect`` — perfect-information MRF filter

    User-facing tags such as ``regression_disagg`` / ``mrffiltered_regression`` map to ``regression``.
    """
    t = tag.strip().lower().replace("-", "_")
    if t in (
        "regression_disagg",
        "mrffiltered_regression",
        "borg_mrffiltered_regression",
        "regression",
    ):
        return "regression"
    if t in (
        "perfect",
        "perfect_foresight",
        "perfect_information",
        "pi",
        "mrffiltered_perfect",
        "borg_mrffiltered_perfect_foresight",
    ):
        return "perfect"
    return "regression"


def mrf_filtered_file_suffix() -> str:
    """
    Filename token for MRF-filtered Borg outputs, e.g. ``_mrffiltered_regression`` or ``_mrffiltered_perfect``.

    Set ``CEE_MRF_FILTER_TAG`` to a user-facing name (e.g. ``regression_disagg`` or ``perfect``)
    or derive from ``CEE_MRF_FILTER_SOURCE``.
    """
    tag = os.environ.get("CEE_MRF_FILTER_TAG", "").strip().lower()
    if not tag:
        src = os.environ.get("CEE_MRF_FILTER_SOURCE", "regression_disagg").strip().lower().replace("-", "_")
        if src in ("perfect", "perfect_information", "pi", "perfect_foresight"):
            tag = "perfect"
        elif src in ("regression_disagg", "regression"):
            tag = "regression_disagg"
        else:
            tag = "regression_disagg"
    canon = _normalize_mrf_filter_tag_for_suffix(tag)
    return f"_mrffiltered_{canon}"


def borg_moea_csv_path(
    policy_type: str,
    reservoir_name: str,
    *,
    seed: Optional[int] = None,
    mrf_filtered: Optional[bool] = None,
    mrf_filter_tag: Optional[str] = None,
) -> str:
    """
    Canonical path to ``MMBorg_*M_*_*_nfe*_seed{N}.csv`` or
    ``..._seed{N}_mrffiltered_regression.csv`` / ``..._mrffiltered_perfect.csv``.

    Overrides (for manifest-driven batches) apply when not ``None``. Otherwise behavior
    follows environment variables as in the legacy signature.

    Environment (when overrides omitted):

    - ``CEE_BORG_SEED`` / ``CEE_SEED``, ``CEE_BORG_MRF_FILTERED``, ``CEE_MRF_FILTER_TAG``, …
    """
    if seed is None:
        s = os.environ.get("CEE_BORG_SEED", os.environ.get("CEE_SEED", "")).strip()
        seed = int(s) if s else BORG_SEED_MRF
    base = (
        f"{borg_moea_csv_dir()}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{seed}"
    )
    use_mrf = mrf_filtered if mrf_filtered is not None else borg_mrf_filtered_enabled()
    if use_mrf:
        if mrf_filter_tag is not None and str(mrf_filter_tag).strip():
            canon = _normalize_mrf_filter_tag_for_suffix(str(mrf_filter_tag).strip())
            base += f"_mrffiltered_{canon}"
        else:
            base += mrf_filtered_file_suffix()
    return base + ".csv"


def resolve_full_borg_seed() -> int:
    """
    Seed for unfiltered Borg CSVs (``..._seed{N}.csv``, no ``_mrffiltered_``).

    Preferred: ``BORG_SEED_FULL`` from ``pywrdrb.release_policies.config`` (see
    :data:`methods.config.BORG_SEED_FULL`), with optional ``CEE_BORG_SEED_FULL`` override.
    Then try ``BORG_SEED_MRF`` and any integers in ``CEE_BORG_SEED_FULL_TRY`` until a probe file
    exists (``CEE_BORG_FULL_PROBE_POLICY`` / ``CEE_BORG_FULL_PROBE_RESERVOIR``, default
    ``STARFIT`` / ``blueMarsh``).
    """
    explicit = os.environ.get("CEE_BORG_SEED_FULL", "").strip()
    preferred = int(explicit) if explicit else BORG_SEED_FULL
    raw = os.environ.get("CEE_BORG_SEED_FULL_TRY", "").strip()
    extra: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            extra.append(int(t))
        except ValueError:
            continue
    candidates: list[int] = []
    for s in [preferred, BORG_SEED_MRF] + extra:
        if s not in candidates:
            candidates.append(s)
    probe_policy = os.environ.get("CEE_BORG_FULL_PROBE_POLICY", "STARFIT").strip() or "STARFIT"
    probe_res = os.environ.get("CEE_BORG_FULL_PROBE_RESERVOIR", "blueMarsh").strip() or "blueMarsh"
    for seed in candidates:
        p = Path(
            borg_moea_csv_path(
                probe_policy,
                probe_res,
                seed=seed,
                mrf_filtered=False,
            )
        )
        if p.is_file():
            return seed
    return preferred


def resolve_borg_moea_csv_path(
    policy_type: str,
    reservoir_name: str,
    *,
    seed: Optional[int] = None,
    mrf_filtered: Optional[bool] = None,
    mrf_filter_tag: Optional[str] = None,
) -> str:
    """
    Return a path to the canonical Borg CSV (``_mrffiltered_*`` when filtering is enabled).
    """
    canonical = Path(
        borg_moea_csv_path(
            policy_type,
            reservoir_name,
            seed=seed,
            mrf_filtered=mrf_filtered,
            mrf_filter_tag=mrf_filter_tag,
        )
    )
    return str(canonical)


def resolve_figure_root(fig_dir: str) -> str:
    """``figures/`` or ``figures/<CEE_FIG_SUBDIR>/`` when ``CEE_FIG_SUBDIR`` is set."""
    sub = os.environ.get("CEE_FIG_SUBDIR", "").strip()
    if sub:
        return os.path.join(fig_dir, sub)
    return fig_dir
