#!/usr/bin/env python3
"""
Compare Pywr-DRB diagnostics between master and release-policy branch outputs.

Generates:
1) Per-reservoir inflow/storage/release (obs + master + release)
2) Trenton flow comparison
3) Lower-basin MRF contribution comparison (stacked areas)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pywrdrb


RESULTS_SETS = [
    "major_flow",
    "res_storage",
    "reservoir_downstream_gage",
    "lower_basin_mrf_contributions",
]

# Storage capacities (MG) used to convert storage to percent-of-capacity.
RESERVOIR_CAPACITY_MG = {
    "beltzvilleCombined": 48317.0588,
    "blueMarsh": 42320.35,
    "fewalter": 35000.0,
    "prompton": 13000.0,
}


def _slice_df(df: Optional[pd.DataFrame], start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if start and end:
        return df.loc[start:end]
    if start:
        return df.loc[start:]
    if end:
        return df.loc[:end]
    return df


def _load_obs(obs_hdf5: Path) -> Dict[str, pd.DataFrame]:
    obs_data = pywrdrb.Data(
        print_status=True,
        results_sets=RESULTS_SETS,
        output_filenames=[str(obs_hdf5)],
    )
    obs_data.load_observations()
    return {
        "major_flow": obs_data.major_flow["obs"][0],
        "res_storage": obs_data.res_storage["obs"][0],
        "reservoir_downstream_gage": obs_data.reservoir_downstream_gage["obs"][0],
    }


def _load_sim(sim_hdf5: Path) -> Dict[str, pd.DataFrame]:
    sim_data = pywrdrb.Data(
        print_status=True,
        results_sets=RESULTS_SETS,
        output_filenames=[str(sim_hdf5)],
    )
    sim_data.load_output()
    key = sim_hdf5.stem
    out = {
        "major_flow": sim_data.major_flow[key][0],
        "res_storage": sim_data.res_storage[key][0],
        "reservoir_downstream_gage": sim_data.reservoir_downstream_gage[key][0],
        "lower_basin_mrf_contributions": sim_data.lower_basin_mrf_contributions[key][0],
    }

    # Inflow is optional in some outputs.
    try:
        inflow_data = pywrdrb.Data(
            print_status=False,
            results_sets=["inflow"],
            output_filenames=[str(sim_hdf5)],
        )
        inflow_data.load_output()
        out["inflow"] = inflow_data.inflow[key][0]
    except Exception:
        out["inflow"] = pd.DataFrame()

    return out


def _plot_reservoir_dynamics(
    reservoir: str,
    obs: Dict[str, pd.DataFrame],
    master: Dict[str, pd.DataFrame],
    release: Dict[str, pd.DataFrame],
    start: Optional[str],
    end: Optional[str],
    out_path: Path,
    perfect: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    m_in = _slice_df(master.get("inflow"), start, end)
    r_in = _slice_df(release.get("inflow"), start, end)
    if m_in is not None and reservoir in m_in.columns:
        axes[0].plot(m_in.index, m_in[reservoir].astype(float), color="tab:blue", label="Master")
    if r_in is not None and reservoir in r_in.columns:
        axes[0].plot(r_in.index, r_in[reservoir].astype(float), color="tab:orange", label="Release policy")
    p_in = _slice_df(perfect.get("inflow"), start, end) if perfect else None
    if p_in is not None and reservoir in p_in.columns:
        axes[0].plot(p_in.index, p_in[reservoir].astype(float), color="tab:green", label="Perfect information")
    axes[0].set_ylabel("Inflow (MGD)")
    axes[0].set_title(f"{reservoir} inflow")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    obs_s = _slice_df(obs.get("res_storage"), start, end)
    m_s = _slice_df(master.get("res_storage"), start, end)
    r_s = _slice_df(release.get("res_storage"), start, end)
    cap = RESERVOIR_CAPACITY_MG.get(reservoir)

    def _to_pct(series: pd.Series) -> pd.Series:
        if cap and cap > 0:
            return 100.0 * series.astype(float) / cap
        return series.astype(float)

    if obs_s is not None and reservoir in obs_s.columns:
        axes[1].plot(obs_s.index, _to_pct(obs_s[reservoir]), "k--", label="Observed")
    if m_s is not None and reservoir in m_s.columns:
        axes[1].plot(m_s.index, _to_pct(m_s[reservoir]), color="tab:blue", label="Master")
    if r_s is not None and reservoir in r_s.columns:
        axes[1].plot(r_s.index, _to_pct(r_s[reservoir]), color="tab:orange", label="Release policy")
    p_s = _slice_df(perfect.get("res_storage"), start, end) if perfect else None
    if p_s is not None and reservoir in p_s.columns:
        axes[1].plot(p_s.index, _to_pct(p_s[reservoir]), color="tab:green", label="Perfect information")
    axes[1].set_ylabel("Storage (% capacity)" if cap else "Storage (MG)")
    axes[1].set_title(f"{reservoir} storage")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    obs_r = _slice_df(obs.get("reservoir_downstream_gage"), start, end)
    m_r = _slice_df(master.get("reservoir_downstream_gage"), start, end)
    r_r = _slice_df(release.get("reservoir_downstream_gage"), start, end)
    if obs_r is not None and reservoir in obs_r.columns:
        axes[2].plot(obs_r.index, obs_r[reservoir].astype(float), "k--", label="Observed")
    if m_r is not None and reservoir in m_r.columns:
        axes[2].plot(m_r.index, m_r[reservoir].astype(float), color="tab:blue", label="Master")
    if r_r is not None and reservoir in r_r.columns:
        axes[2].plot(r_r.index, r_r[reservoir].astype(float), color="tab:orange", label="Release policy")
    p_r = _slice_df(perfect.get("reservoir_downstream_gage"), start, end) if perfect else None
    if p_r is not None and reservoir in p_r.columns:
        axes[2].plot(p_r.index, p_r[reservoir].astype(float), color="tab:green", label="Perfect information")
    axes[2].set_ylabel("Release / gage (MGD)")
    axes[2].set_title(f"{reservoir} release dynamics")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle(f"{reservoir}: observed vs master vs release-policy", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trenton(
    obs: Dict[str, pd.DataFrame],
    master: Dict[str, pd.DataFrame],
    release: Dict[str, pd.DataFrame],
    start: Optional[str],
    end: Optional[str],
    out_path: Path,
    perfect: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    node = "delTrenton"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    ax = axes[0]

    obs_q = _slice_df(obs.get("major_flow"), start, end)
    m_q = _slice_df(master.get("major_flow"), start, end)
    r_q = _slice_df(release.get("major_flow"), start, end)
    if obs_q is not None and node in obs_q.columns:
        ax.plot(obs_q.index, obs_q[node].astype(float), "k--", label="Observed")
    if m_q is not None and node in m_q.columns:
        ax.plot(m_q.index, m_q[node].astype(float), color="tab:blue", label="Master")
    if r_q is not None and node in r_q.columns:
        ax.plot(r_q.index, r_q[node].astype(float), color="tab:orange", label="Release policy")
    p_q = _slice_df(perfect.get("major_flow"), start, end) if perfect else None
    if p_q is not None and node in p_q.columns:
        ax.plot(p_q.index, p_q[node].astype(float), color="tab:green", label="Perfect information")

    ax.set_title("Trenton flow comparison")
    ax.set_ylabel("Flow (MGD)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    # FDC panel (flow duration curve)
    def _fdc_xy(series: pd.Series):
        vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).values
        if len(vals) == 0:
            return None, None
        vals = vals[vals >= 0.0]
        if len(vals) == 0:
            return None, None
        vals_desc = sorted(vals, reverse=True)
        n = len(vals_desc)
        exceed = [100.0 * (i + 1) / (n + 1) for i in range(n)]
        return exceed, vals_desc

    ax_fdc = axes[1]
    for label, df, color, linestyle in [
        ("Observed", obs_q, "k", "--"),
        ("Master", m_q, "tab:blue", "-"),
        ("Release policy", r_q, "tab:orange", "-"),
        ("Perfect information" if perfect else "", p_q if perfect else None, "tab:green", "-"),
    ]:
        if not label or df is None or node not in df.columns:
            continue
        x, y = _fdc_xy(df[node])
        if x is None:
            continue
        ax_fdc.plot(x, y, color=color, linestyle=linestyle, label=label)

    ax_fdc.set_title("Trenton flow FDC")
    ax_fdc.set_xlabel("Exceedance probability (%)")
    ax_fdc.set_ylabel("Flow (MGD)")
    ax_fdc.grid(alpha=0.3)
    ax_fdc.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mrf_stacks(
    master: Dict[str, pd.DataFrame],
    release: Dict[str, pd.DataFrame],
    start: Optional[str],
    end: Optional[str],
    out_path: Path,
    perfect: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    m = _slice_df(master.get("lower_basin_mrf_contributions"), start, end)
    r = _slice_df(release.get("lower_basin_mrf_contributions"), start, end)
    p = _slice_df(perfect.get("lower_basin_mrf_contributions"), start, end) if perfect else None
    cols = [
        "mrf_trenton_beltzvilleCombined",
        "mrf_trenton_blueMarsh",
        "mrf_trenton_nockamixon",
    ]

    nrows = 3 if p is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    elif nrows > 1:
        axes = list(axes)
    for ax, df, title in [
        (axes[0], m, "Master"),
        (axes[1], r, "Release policy"),
    ] + ([(axes[2], p, "Perfect information")] if p is not None else []):
        if df is None or any(c not in df.columns for c in cols):
            ax.text(0.5, 0.5, "Required MRF columns not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}: lower-basin MRF contributions")
            continue
        ax.stackplot(
            df.index,
            df[cols[0]].astype(float),
            df[cols[1]].astype(float),
            df[cols[2]].astype(float),
            labels=["Beltzville Combined", "Blue Marsh", "Nockamixon"],
            alpha=0.8,
        )
        ax.set_title(f"{title}: lower-basin MRF contributions at Trenton")
        ax.set_ylabel("Contribution (MGD)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostics for master vs release-policy outputs.")
    parser.add_argument(
        "--master-output",
        default="/home/fs02/pmr82_0001/ms3654/Research/PywrDRB_master/model_outputs/pywrdrb_output_pub_nhmv10_BC_withObsScaled.hdf5",
        help="Path to master-branch output HDF5.",
    )
    parser.add_argument(
        "--release-output",
        required=True,
        help="Path to release-policy-branch output HDF5.",
    )
    parser.add_argument(
        "--perfect-output",
        default="",
        help="Optional path to perfect-information output HDF5.",
    )
    parser.add_argument(
        "--reservoirs",
        nargs="+",
        default=["beltzvilleCombined", "blueMarsh", "fewalter", "prompton"],
        help="Reservoirs for per-reservoir dynamics plots.",
    )
    parser.add_argument("--start", default="2019-01-01", help="Plot start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2024-01-30", help="Plot end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default="/home/fs02/pmr82_0001/ms3654/Research/CEE6400Project/figures/branch_diagnostics",
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    master_path = Path(args.master_output).resolve()
    release_path = Path(args.release_output).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = _load_obs(master_path)
    master = _load_sim(master_path)
    release = _load_sim(release_path)
    perfect = _load_sim(Path(args.perfect_output).resolve()) if args.perfect_output else None

    suffix = f"{args.start.replace('-', '')}_{args.end.replace('-', '')}"

    for reservoir in args.reservoirs:
        out_path = out_dir / f"{reservoir}_master_vs_release_{suffix}.png"
        _plot_reservoir_dynamics(
            reservoir=reservoir,
            obs=obs,
            master=master,
            release=release,
            start=args.start,
            end=args.end,
            out_path=out_path,
            perfect=perfect,
        )
        print(f"[saved] {out_path}")

    trenton_path = out_dir / f"trenton_flow_master_vs_release_{suffix}.png"
    _plot_trenton(
        obs=obs,
        master=master,
        release=release,
        start=args.start,
        end=args.end,
        out_path=trenton_path,
        perfect=perfect,
    )
    print(f"[saved] {trenton_path}")

    mrf_path = out_dir / f"lower_basin_mrf_master_vs_release_{suffix}.png"
    _plot_mrf_stacks(
        master=master,
        release=release,
        start=args.start,
        end=args.end,
        out_path=mrf_path,
        perfect=perfect,
    )
    print(f"[saved] {mrf_path}")


if __name__ == "__main__":
    main()
