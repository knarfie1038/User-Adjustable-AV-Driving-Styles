# phase_1_gpu.py
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path
from math import isclose, sqrt
from datetime import datetime
from itertools import product
from typing import Any
from scipy.stats import chi2, norm, t


import numpy as np
import pandas as pd

from Classes import config as C
from Classes.gpu_simulation import _load_data, simulate_gpu
import pyfiglet

DATASETS = ["trainHH.zarr", "trainHA.zarr", "trainAH.zarr"]


def build_grid(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:   raise ValueError("Step must be positive")
    if lo  > hi:    raise ValueError("Minimum exceeds maximum")

    g = np.arange(lo, hi + step*0.5, step)
    if not isclose(g[-1], hi) and g[-1] > hi + 1e-9:
        g = g[:-1]
    return [round(float(x), 6) for x in g]

def classify_cases(
    metrics: np.ndarray,
    idx: np.ndarray,
    size_lead: np.ndarray,
    size_follow: np.ndarray,
    x_lead: np.ndarray,
    v_lead: np.ndarray,
    x_follow: np.ndarray,
) -> tuple[int, int, int, list[int], list[int]]:
    ttc_min   = metrics[:, 0]
    tte_count = metrics[:, 1]
    drac_max  = metrics[:, 2]
    dx_min    = metrics[:, 3]

    

    starts = idx[:, 0].astype(np.int32)
    initial_dx = (x_lead[starts]- x_follow[starts]- 0.5 * size_lead- 0.5 * size_follow)

    exclusions = []
    for i, (dx0, v0) in enumerate(zip(initial_dx, v_lead[starts])):
        if (dx0 < 0) or (dx0 < v0 * 0.15):
            exclusions.append(i)

    crash_cases: list[int] = []
    near_cases: list[int] = []

    for i in range(metrics.shape[0]):
        if i in exclusions:
            continue

        crash_cond = (dx_min[i] < C.DX_THRESHOLD)
        near_cond = (
            (not crash_cond)
            and (
                (ttc_min[i] < C.TTC_THRESHOLD and tte_count[i] > C.TET_THRESHOLD)
            )
        )
        if crash_cond:
            crash_cases.append(i)
        elif near_cond:
            near_cases.append(i)
    all_cases = crash_cases + near_cases
    all_ttcs  = ttc_min[all_cases].tolist()            
    return (len(crash_cases), len(near_cases), len(exclusions), crash_cases, near_cases, all_ttcs)


def _format_df(df: pd.DataFrame, param_cols: list[str], ds_path: str) -> pd.DataFrame:
    if not param_cols:
        wide = df.copy().reset_index(drop=True)
    else:
        wide = df.sort_values(param_cols).reset_index(drop=True)
        if len(param_cols) > 1:
            first = param_cols[0]
            parts: list[pd.DataFrame] = []
            for v in wide[first].unique():
                grp = wide[wide[first] == v].reset_index(drop=True)
                parts.append(grp)
                parts.append(pd.DataFrame({"": [""] * len(grp)}))
            parts.pop() 
            wide = pd.concat(parts, axis=1)

    cols = wide.columns.tolist()
    ncols = len(cols)
    meta_data: list[list[Any]] = []

    blank_row = [""] * ncols
    for _ in range(3):
        meta_data.append(blank_row.copy())

    ds_row = [""] * ncols
    ds_row[0] = "dataset_path"
    ds_row[1] = ds_path
    meta_data.append(ds_row)
    cfg = {k: v for k, v in vars(C).items() if k.isupper()}
    for name, val in cfg.items():
        row = [""] * ncols
        row[0] = name
        row[1] = val
        meta_data.append(row)

    meta_df = pd.DataFrame(meta_data, columns=cols)
    return pd.concat([wide, meta_df], ignore_index=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jerk",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="Sweep DESIRED_JERK (three floats: min max step)",
    )
    parser.add_argument(
        "--accel",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="Sweep DESIRED_ACCELERATION",
    )
    parser.add_argument(
        "--decel",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="Sweep DESIRED_DECELERATION",
    )
    parser.add_argument(
        "--time",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="Sweep SAFE_FOLLOW_TIME",
    )    
    parser.add_argument(
        "--model",
        choices=["idm", "hidm"],
        default="idm",
        help="Car-following model (default idm)",
    )
    parser.add_argument(
        "--path",
        choices=DATASETS,
        default="trainHH.zarr",
        help="Dataset to simulate (default trainHH.zarr)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV filename; "
        "if omitted, a timestamped name is generated.",
    )
    parser.add_argument(
        "--sweep_count",
        nargs=1,
        type=int,
        metavar=("How many times the simulation should be ran"),
        default=[1],
        help="Value from at least 1",
    )    
    return parser.parse_args()

def plot_mcnemar(counts: dict[str,int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5,3))
    ax.axis('off')
    tbl = ax.table(
        cellText=[[counts['a'], counts['b']], [counts['c'], counts['d']]],
        rowLabels=['Baseline (TTC < 2 s)', 'Baseline (TTC ≥ 2 s)'],
        colLabels=['New (TTC < 2 s)', 'New (TTC ≥ 2 s)'],
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(True)
    tbl.auto_set_column_width(col=[0,1])
    plt.title(title)
    b, c = counts['b'], counts['c']
    z = (b - c) / sqrt(b + c) if b+c>0 else float('nan')
    pz = 2*(1 - norm.cdf(abs(z))) if not np.isnan(z) else float('nan')
    chi2_stat = ((abs(b - c)-1)**2)/(b+c) if b+c>0 else float('nan')
    pchi = 1 - chi2.cdf(chi2_stat, df=1) if not np.isnan(chi2_stat) else float('nan')
    fig.text(0.02, 0.02, f"z={z:.6f} \np(z)={pz:.6f}\nchi2={chi2_stat:.6f} \np(chi2)={pchi:.6f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

    

def run_dataset(ds_path: str, sweep_spec: dict[str, Any], out_dir) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    idx,dt,size_lead,size_follow,x_lead,v_lead,a_lead,jerk_lead,x_follow,v_follow,a_follow,jerk_follow= _load_data(ds_path)

    provided = {k: v for k, v in sweep_spec.items() if v is not None}
    
    if not provided:
        metrics = simulate_gpu(ds_path, base = True)
        results = None
        ttc_min = None
        crashes, near_misses, exclusions, _, _, _= classify_cases(metrics, idx, size_lead, size_follow, x_lead, v_lead, x_follow)
        rows.append({
            "crashes": crashes,
            "near_misses": near_misses,
            "exclusions": exclusions,
        })
        param_cols: list[str] = []
        ttc_min_base = None

    elif len(provided) == 1:
        p_name, trio = next(iter(provided.items()))
        lo, hi, step = trio
        grid = build_grid(lo, hi, step)
        param_cols = [p_name]
        results: dict[float, dict[str,int]] = {}
        print(f"  sweeping {p_name}: {grid}")

        metrics_base = simulate_gpu(ds_path, base = True)
        _, _, _, crash_ids_base, near_ids_base, ttc_min_base = classify_cases(metrics_base, idx, size_lead, size_follow, x_lead, v_lead, x_follow)
        base_critical_cases = set(crash_ids_base) | set(near_ids_base)
        n_cases = metrics_base.shape[0]

        for val in grid:
            new_metrics = simulate_gpu(ds_path, **{p_name:val})
            crashes_new_no, near_misses_new_no, exclusions, crashes_new, near_misses_new, ttc_min = classify_cases(new_metrics, idx, size_lead, size_follow, x_lead, v_lead, x_follow)
            new_crit = set(crashes_new) | set(near_misses_new)
            a = len(base_critical_cases & new_crit)
            b = len(base_critical_cases - new_crit)
            c = len(new_crit - base_critical_cases)
            d = n_cases - (a+b+c)
            results[val] = {'a':a, 'b':b, 'c':c, 'd':d}            
            rows.append({
                p_name: val,
                "crashes": crashes_new_no,
                "near_misses": near_misses_new_no,
                "exclusions": exclusions,
            })
       

    else:
        results = None
        ttc_min = None
        grid_dict = {k: build_grid(*v) for k, v in provided.items()}
        param_cols = sorted(grid_dict)
        print("Multi Parameter Sweep over:")
        for k in param_cols:
            print(f"    {k}: {grid_dict[k]}")
        for combo in product(*(grid_dict[k] for k in param_cols)):
            kwargs = dict(zip(param_cols, combo))
            tag = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"    → {tag}")
            metrics = simulate_gpu(ds_path, **kwargs)

            crashes, near, excl,_, _, _  = classify_cases(metrics, idx, size_lead, size_follow, x_lead, v_lead, x_follow)
            row = {**kwargs,
                   "crashes": crashes,
                   "near_misses": near,
                   "exclusions": excl,}
            rows.append(row)
        ttc_min_base = None

    df = pd.DataFrame(rows)
    other_cols = [c for c in df.columns if c not in param_cols]
    raw_df = df[param_cols + other_cols]

    df = _format_df(df, param_cols, ds_path)

    return raw_df, df, results, [ttc_min_base, ttc_min]

def main() -> None:
    t0 = time.perf_counter()
    ascii_art = pyfiglet.figlet_format("UNICORN")
    print(ascii_art)
    args = parse_args()
    sweep_spec = {"desired_jerk":  args.jerk,"desired_accel": args.accel,"desired_decel": args.decel,"desired_sft":   args.time,}
    sweep_count = args.sweep_count[0]
    base = Path(args.out or f"simulation_{datetime.now().strftime('%m-%d_%H-%M')}").with_suffix("").name

    out_dir = Path(base)
    out_dir.mkdir(exist_ok=True, parents=True)

    combined_totals: dict[tuple, dict[str, Any]] = {}
    
    base_ttc_list = []
    new_ttc_list = []
    p_chi_list   = []
    for value in range(sweep_count):
        base_ttc_list.append([]);new_ttc_list.append([])
        combined_mcnemar: dict[float, dict[str,int]] = {}
        for ds in DATASETS:
            print("\n" + "=" * 72)
            print(f"▶ Dataset {ds}")
            print("=" * 72)

            raw_df, formatted_df, results, ttc_mins = run_dataset(ds, sweep_spec, out_dir)
            #base_ttc_list[value].extend(ttc_mins[0]); new_ttc_list[value].extend(ttc_mins[1])

            if sweep_count == 1:
                provided = {k: v for k, v in sweep_spec.items() if v is not None}
                if len(provided) == 1:
                    for val, cnts in results.items():
                        combined_mcnemar.setdefault(val, {'a':0,'b':0,'c':0,'d':0})
                        for k in cnts:
                            combined_mcnemar[val][k] += cnts[k]        
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                stem = Path(ds).stem
                out_path = out_dir / f"{base}_{stem}.csv"
                formatted_df.to_csv(out_path, index=False)
                print(f"  ↳ wrote {out_path.resolve()}")        


                param_columns = [c for c in raw_df.columns
                            if c not in {"crashes", "near_misses", "exclusions", ""}]
                for _, row in raw_df.iterrows():
                    key = tuple(row[c] for c in param_columns)
                    if key not in combined_totals:
                        combined_totals[key] = {c: row[c] for c in param_columns}
                        combined_totals[key].update({
                            "crashes":     0,
                            "near_misses": 0,
                            "exclusions":  0,
                        })
                    entry = combined_totals[key]
                    entry["crashes"]     += int(row["crashes"])
                    entry["near_misses"] += int(row["near_misses"])
                    entry["exclusions"]  += int(row["exclusions"])
        # b, c = combined_mcnemar[0.1]['b'], combined_mcnemar[0.1]['c']
        # chi2_stat = ((abs(b - c)-1)**2)/(b+c) if b+c>0 else float('nan')
        # pchi = 1 - chi2.cdf(chi2_stat, df=1) if not np.isnan(chi2_stat) else float('nan')
        # p_chi_list.append(pchi)
    print(p_chi_list)
    if len(base_ttc_list) and len(new_ttc_list) > 1:
        bins = np.arange(0, C.TTC_THRESHOLD + 0.2, 0.2)
        num_runs = len(base_ttc_list)
        num_bins = len(bins) - 1

        base_counts = np.zeros((num_runs, num_bins))
        new_counts  = np.zeros((num_runs, num_bins))

        for i, run in enumerate(base_ttc_list):
            arr = np.asarray(run)
            arr = arr[arr < C.TTC_THRESHOLD]
            base_counts[i], _ = np.histogram(arr, bins=bins)

        for i, run in enumerate(new_ttc_list):
            arr = np.asarray(run)
            arr = arr[arr < C.TTC_THRESHOLD]
            new_counts[i], _ = np.histogram(arr, bins=bins)  

        def mean_clm(counts_arr):
            N = counts_arr.shape[0]
            mean_vals = counts_arr.mean(axis=0)
            std_dev   = counts_arr.std(axis=0, ddof=1)
            sem       = std_dev / np.sqrt(N)
            t_crit    = t.ppf(0.975, df=N - 1)
            clm       = t_crit * sem
            return mean_vals, clm     

        base_mean, base_clm = mean_clm(base_counts)
        new_mean, new_clm   = mean_clm(new_counts)  

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axisbelow(True)
        left_edges = bins[:-1]
        full_bin_width = 0.18
        bar_width  = (full_bin_width) / 2


        baseline_dark  = "#4C72B0" 
        test_dark    = "#DD8452"

        parameter = "Safe Follow Time"

        ax.bar(0.01 + left_edges + bar_width/2, base_mean, width=bar_width, capsize=5, label='Baseline', color=baseline_dark, edgecolor='black', alpha = 0.7, zorder = 3)
        ax.bar(left_edges+0.01 + (bar_width*1.5), new_mean, width=bar_width, yerr=new_clm, capsize=5, label=f'Guassian-{parameter} Variation', color=test_dark, edgecolor='black', alpha = 0.7, zorder = 3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0.0, C.TTC_THRESHOLD)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_xlabel("Time-to-Collision (TTC) [s]")
        ax.set_ylabel("Number of Critical Events per Bin")
        ax.set_title(f"TTC Distribution (<2 s) Baseline vs. Gaussian-{parameter} Variation with 95% CLM ({sweep_count} Runs)")
        xticks = list(left_edges) + [C.TTC_THRESHOLD]
        ax.set_xticks(xticks)
        p_arr = np.array(p_chi_list)
        percentage_above = 100 * np.count_nonzero(p_arr > 0.05) / p_arr.size
        fig.text(0.02, 0.02, f"{percentage_above:.1f}% of p-values (χ²) > 0.05", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_dir / "ttc_mean_histogram_clm.png", dpi=600)
    if combined_mcnemar:
        for val, cnts in combined_mcnemar.items():
            title = f"McNemar Gaussian {next(iter(provided.items()))[0]}"
            path = out_dir / f"mcnemar_{val}.png"
            plot_mcnemar(cnts, title, path)
            print(f"Saved McNemar plot: {path.resolve()}")


    if combined_totals:
        combined_rows: list[dict[str, Any]] = []
        for entry in combined_totals.values():
            combined_rows.append(entry)

        raw_combined_df = pd.DataFrame(combined_rows)
        raw_combined_df["critical"] = raw_combined_df["crashes"] + raw_combined_df["near_misses"]

        combined_df = raw_combined_df.drop(columns="critical")

        param_cols = [c for c in combined_df.columns
                      if c not in {"crashes", "near_misses", "exclusions", ""}]
        others = [c for c in combined_df.columns if c not in param_cols]
        combined_df = combined_df[param_cols + others]
        combined_df = _format_df(combined_df, param_cols, "Combined")

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_comb = out_dir / f"{base}_COMBINED.csv"
        combined_df.to_csv(out_comb, index=False)
        print(f"\n✔ Combined summary written to {out_comb.resolve()}")

        if len(param_columns)==2:
            x_param, line_param = param_columns
            plt.figure()
            for lvl in sorted(raw_combined_df[line_param].unique()):
                sub = raw_combined_df[raw_combined_df[line_param]==lvl]
                plt.plot(sub[x_param], sub["critical"], "o-", label=f"{line_param}={lvl}")
            plt.xlabel(x_param)
            plt.ylabel("Total critical cases")
            plt.title("Combined sweep results")
            plt.legend()
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
            plt.tight_layout(rect=[0, 0, 0.85, 1.0])

            big_png = out_dir/f"{base}_COMBINED_plot.png"
            plt.savefig(big_png); plt.close()
            print(f"  ↳ wrote plot {big_png}")

            for lvl in sorted(raw_combined_df[line_param].unique()):
                sub = raw_combined_df[raw_combined_df[line_param]==lvl]
                plt.figure()
                plt.plot(sub[x_param], sub["critical"], "o-")
                plt.xlabel(x_param)
                plt.ylabel("Total critical cases")
                plt.title(f"{line_param} = {lvl}")
                small_png = out_dir/f"{base}_plot_{line_param}_{lvl}.png"
                plt.savefig(small_png); plt.close()
                print(f"  ↳ wrote plot {small_png}")

        if len(param_columns) == 3:
            const_param = next(p for p in param_columns if raw_combined_df[p].nunique() == 1)
            const_val = raw_combined_df[const_param].iloc[0]

            x_param, y_param = [p for p in param_columns if p != const_param]

            pivot = raw_combined_df.pivot_table(index=y_param,columns=x_param,values="critical")
            X_vals = pivot.columns.values; Y_vals = pivot.index.values
            Z = pivot.values

            i_max, j_max = np.unravel_index(np.argmax(Z), Z.shape)
            x_max, y_max = X_vals[j_max], Y_vals[i_max]
            x_mid = 0.5*(X_vals.min()+X_vals.max())
            y_mid = 0.5*(Y_vals.min()+Y_vals.max())    
            swapped = False
            if x_param != "desired_decel":     
                pivot   = pivot.T                 
                x_param, y_param = y_param, x_param 

            Z      = pivot.values
            X_vals = pivot.columns.values
            Y_vals = pivot.index.values

            x_order = np.argsort(X_vals)
            X_vals  = X_vals[x_order]
            Z       = Z[:, x_order]           
            y_order = np.argsort(Y_vals)
            Y_vals  = Y_vals[y_order]
            Z       = Z[y_order, :] 

            X, Y = np.meshgrid(X_vals, Y_vals)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=135)
            ax.invert_xaxis()
            ax.invert_yaxis() 
            surf = ax.plot_surface(
                X, Y, Z,
                cmap="viridis",
                edgecolor="none",
                antialiased=True
            )
            ax.set_xlabel("Deceleration (m/s²)")
            ax.set_ylabel("Safe Follow Time (s)")
            ax.set_zlabel("Total Critical Cases")
            title = f"Total Critical Cases at Acceleration = 1.4 m/s², Jerk = 2.0 m/s³"
            if swapped:
                title += " (axes swapped)"            
            ax.set_title(title)
            cbar = fig.colorbar(surf, pad=0.1, shrink=0.6)
            cbar.set_label("Critical Cases")
            plt.tight_layout()       
            topo_png = out_dir / f"{base}_topo_{const_param}_{const_val}.png"
            plt.savefig(topo_png)
            plt.close()
            print(f"  ↳ wrote topography plot {topo_png}")
           


        mm, ss = divmod(time.perf_counter() - t0, 60)
        print(f"Time elapsed on this simulation: {int(mm):02d}:{int(ss):02d}")


if __name__ == "__main__":
    main()
