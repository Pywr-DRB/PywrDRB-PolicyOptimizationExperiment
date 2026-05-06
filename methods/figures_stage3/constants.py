"""Stage-3 full-Pareto figure constants: reservoir display order and policy colors."""

from methods.plotting.theme import policy_type_colors

# Internal Pywr / Borg names (order for multipanel columns, left → right)
RESERVOIR_KEYS = (
    "blueMarsh",
    "beltzvilleCombined",
    "fewalter",
    "prompton",
)

# Human-readable titles (figure panels)
RESERVOIR_DISPLAY_NAMES = {
    "blueMarsh": "Blue Marsh",
    "beltzvilleCombined": "Beltzville",
    "fewalter": "F.E. Walter",
    "prompton": "Prompton",
}

POLICY_ORDER = ("STARFIT", "PWL", "RBF")

# Align with ``methods.plotting.theme.policy_type_colors`` (project-wide)
STAGE3_POLICY_COLORS = {
    p: policy_type_colors.get(p, "#333333") for p in POLICY_ORDER
}

# Match Pywr-DRB release-policy operational constant ``mrf_baseline_delTrenton``.
DEFAULT_TRENTON_TARGET_MGD: float = 1938.950669

MONTH_LABELS = ("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")

# Distinct from policy ribbons and from each other (multipanels / diagnostics)
STAGE3_OBSERVED_COLOR = "#1f2937"
STAGE3_TRENTON_TARGET_COLOR = "#b91c1c"

# Output subfolders under the Stage 3 base dir (align with CEE_FIG_SUBDIR / README)
STAGE3_BORG_VARIANT_SUBDIRS = {
    "full": "borg_full_series",
    "regression": "borg_mrffiltered_regression",
    "perfect": "borg_mrffiltered_perfect_foresight",
}
