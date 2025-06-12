# phase_2.py
# -------------------------------------------------
from __future__             import annotations
from matplotlib.ticker      import MaxNLocator, FormatStrFormatter
from pathlib                import Path
from math                   import sqrt, isclose
from datetime               import datetime
from typing                 import Any
from scipy.stats            import chi2, norm, t

import      argparse
import      sys
import      time
import      matplotlib.pyplot as plt
import      numpy as np
import      pandas as pd
import      pyfiglet

from Classes_2 import config as C
from Classes_2.gpu_simulation import _load_data, simulate_gpu

DATASETS = ["trainHH.zarr", "trainHA.zarr", "trainAH.zarr"]


#Helper function to build a grid of variables to sweep over
def build_grid(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:   raise ValueError("Step size must be positive")
    if lo  > hi:    raise ValueError("Low value exceeds High value")

    g = np.arange(lo, hi + step*0.5, step)
    if not isclose(g[-1], hi) and g[-1] > hi + 1e-9:
        g = g[:-1]
    return [round(float(x), 6) for x in g]

#Classify crashes and near misses
def classify_cases(
    metrics,
    idx: np.ndarray,
    size_lead: np.ndarray,
    size_follow: np.ndarray,
    x_lead: np.ndarray,
    v_lead: np.ndarray,
    x_follow: np.ndarray,
) -> tuple[int, int, int, list[int], list[int], list[tuple[float,int]], np.ndarray, np.ndarray]:

    #Index gpu_simulation variables
    ttc_min, tte_count, dx_min, avg_lead, avg_last, car_pos, case_id_of_car = metrics

    #Establish initial conditions and exclusions
    starts = idx[:, 0].astype(np.int32)
    initial_dx = (x_lead[starts]- x_follow[starts]- 0.5 * size_lead- 0.5 * size_follow)
    exclusions  = (initial_dx < 0) | (initial_dx < v_lead[starts] * 0.15)
    exclusions_car = exclusions[case_id_of_car]

    #Define crashes & Near misses
    crash = dx_min < C.DX_THRESHOLD
    near  = (~crash) & (ttc_min < C.TTC_THRESHOLD) & (tte_count > C.TET_THRESHOLD)
    crash &= ~exclusions_car
    near  &= ~exclusions_car    
    crash_ids = np.nonzero(crash)[0].tolist()
    near_ids  = np.nonzero(near )[0].tolist()

    #Export TTC under 2 list
    ttcs_under2_mask = crash | near
    ttcs_under_2 = [(float(ttc_min[i]), int(car_pos[i]))    for i in np.nonzero(ttcs_under2_mask)[0]]
        
    return (crash_ids, near_ids, ttcs_under_2, avg_lead, avg_last)
  
#Parse input arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--penetration",
        nargs="+",
        type=float,
        metavar=("Penetration Value"),
        default=[0.7],
        help="Penetration value for the chance a car is an IDM AV with slider. (0 - 1.0), or sweep between l_bound, up_bound, step",
    )   
    parser.add_argument(
        "--chance",
        nargs=3,
        type=float,
        metavar=("Conservative", "Balanced", "Aggressive"),
        default= (0.25, 0.5, 0.25),
        help="Chance of a certain driving style to be in the random styles (value between 0 and 1.0, capped at 1.0.)",
    )   
    parser.add_argument(
        "--sweep_count",
        nargs=1,
        type=int,
        metavar=("How many times the simulation should be ran"),
        default=[1],
        help="Value from at least 1",
    )    
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output folder filename; "
        "if omitted, a timestamped name is generated.",
    )
    return parser.parse_args()

#Helper function to run the simulation and output variables to generate plots
def run_dataset(variables: dict[str, Any]) -> pd.DataFrame:
    penetration_rate = variables["penetration_rate"][0]
    chance = variables["chance"]
    results: dict[float, dict[str,int]] = {}
    base_crash_counts = [];         base_total_counts = [];         new_crash_counts  = []; new_total_counts = []
    total_base_ttcs_under_2 = [];   total_new_ttcs_under_2 = []
    base_avg_last_list = [];        new_avg_last_list = []

    for ds in DATASETS:
        print(f"  sweeping penetration rate:{penetration_rate} with chances: {chance} over dataset '{ds}'")
        idx,_,size_lead,size_follow,x_lead,v_lead,_,_,x_follow,_,_,_= _load_data(ds)
        n_cases = idx.shape[0]
        #Run base metrics
        base_metrics = simulate_gpu(ds, penetration = penetration_rate, probability = chance,  base = True)
        crash_ids_base, near_ids_base, base_ttcs_under_2, _, base_avg_last = classify_cases(base_metrics, idx, size_lead, size_follow, x_lead, v_lead, x_follow)
        #Run test metrics
        new_metrics = simulate_gpu(ds, penetration = penetration_rate, probability = chance)
        crash_ids_new, near_ids_new, new_ttcs_under_2, _, new_avg_last  = classify_cases(new_metrics, idx, size_lead, size_follow, x_lead, v_lead, x_follow)

        #Fill output lists
        total_base_ttcs_under_2.append(base_ttcs_under_2); total_new_ttcs_under_2.append(new_ttcs_under_2)
        base_avg_last_list.extend(base_avg_last); new_avg_last_list.extend(new_avg_last)

        #Define critical cases
        case_ids = base_metrics[6]
        base_flag = np.zeros(n_cases, dtype=bool)
        new_flag  = np.zeros_like(base_flag)
        base_flag[case_ids[crash_ids_base + near_ids_base]] = True
        new_flag [case_ids[crash_ids_new  + near_ids_new ]] = True
        base_crash_counts.append(len(crash_ids_base)); new_crash_counts.append(len(crash_ids_new))
        base_total_counts.append(int(base_flag.sum())); new_total_counts.append(int(new_flag.sum()))

        #Export mcnemar results
        a = int(( base_flag &  new_flag).sum())
        b = int(( base_flag & ~new_flag).sum())
        c = int((~base_flag &  new_flag).sum())
        d = n_cases - (a + b + c)
        results[ds] = {'a':a, 'b':b, 'c':c, 'd':d}       


    return results, total_base_ttcs_under_2, total_new_ttcs_under_2, base_avg_last_list, new_avg_last_list, [base_crash_counts, base_total_counts, new_crash_counts, new_total_counts]

#Run main function
def main() -> None:  
    t0 = time.perf_counter()
    args = parse_args()
    ascii_art = pyfiglet.figlet_format("UNICORN")
    print(ascii_art)    
    
    #Define if penetration sweep or single penetration rate
    raw_penetration = args.penetration
    if len(raw_penetration) == 1:
        pen_grid = raw_penetration
    elif len(raw_penetration) == 3:
        pen_grid = build_grid(*raw_penetration)
    else:
        sys.exit("❌  --penetration must have 1 or 3 floats")
    
    #Define arguments
    variables = {"penetration_rate":  args.penetration,"chance": args.chance, "sweeps": args.sweep_count}
    sweep_count = variables["sweeps"][0]

    #Make path
    base = Path(args.out or f"simulation_{datetime.now().strftime('%m-%d_%H-%M')}").with_suffix("").name
    out_dir = Path(base)
    out_dir.mkdir(exist_ok=True, parents=True)
    plotcount = 0

    print("\n" + "=" * 72)
    print(f"▶ Running simulation over {DATASETS[0]}, {DATASETS[1]}, {DATASETS[2]}.")
    print("=" * 72)    

    #Define simulation lists
    new_penetration_rate_crashes = [];  base_penetration_rate_crashes = []
    new_penetration_rate_total = [];    base_penetration_rate_total = []
    avg_last_new = [];                  avg_last_base = []    
    p_values = []
    for value in range(sweep_count):
        print(pen_grid)
        if plotcount == 0:
            #Build lists logic, [ [0.1], [0.2], etc.] for each value in penetration sweep
            for sweep_id, pen in enumerate(pen_grid):
                new_penetration_rate_crashes.append([]); new_penetration_rate_total.append([])
                base_penetration_rate_crashes.append([]); base_penetration_rate_total.append([])
                avg_last_new.append([]); avg_last_base.append([])
                p_values.append([])
        plotcount+=1
        for sweep_id, pen in enumerate(pen_grid):
            print(f"\n=== Penetration {pen:.2f} ===")
            variables = {"penetration_rate": [pen],"chance":args.chance}
            results, base_ttcs, new_ttcs, base_speed, last_speed, CC_Counts = run_dataset(variables)

            #Build the mcnemar values from simulation run over all datasets
            a = 0; b = 0; c = 0; d =0
            for ds, vals in results.items():
                a += vals['a']
                b += vals['b']
                c += vals['c']
                d += vals['d']                

            #Calculate p_chi for each sweep run
            if b + c > 0:
                chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
                pchi = 1 - chi2.cdf(chi2_stat, df=1)
            else:
                pchi = float('nan')  
            print(f"{pen}: a: {a}, b: {b}, c: {c}, d: {d}")  

            #Fill lists        
            base_penetration_rate_crashes[sweep_id].append(sum(CC_Counts[0]))
            base_penetration_rate_total[sweep_id].append(sum(CC_Counts[1]))            
            new_penetration_rate_crashes[sweep_id].append(sum(CC_Counts[2]))
            new_penetration_rate_total[sweep_id].append(sum(CC_Counts[3]))
            p_values[sweep_id].append(pchi)
            last_average = np.nanmean(last_speed); base_average = np.nanmean(base_speed)
            avg_last_new[sweep_id].append(last_average); avg_last_base[sweep_id].append(base_average)
        print(f"Base crashes: {base_penetration_rate_crashes}\n"
              f"Base total crit: {base_penetration_rate_total}\n"
              f"New crashes: {new_penetration_rate_crashes}\n"
              f"New total crit: {new_penetration_rate_total}\n")                  

    #Define mean and CLM for the simulation - Critical cases plot
    new_histo_array     = np.asarray(new_penetration_rate_total)
    N_new               = new_histo_array.shape[1]
    std_dev_new         = np.std(new_histo_array, axis=1, ddof=1)
    crit_cases_new      = np.mean(new_histo_array, axis=1)  
    crashes_new         = np.mean(np.asarray(new_penetration_rate_crashes), axis = 1)
    crit_cases_new_err  = (t.ppf(0.975, df=N_new - 1))*(std_dev_new / np.sqrt(N_new))
    near_new            = crit_cases_new - crashes_new

    #Define mean and CLM for the baseline control - Critical cases plot
    base_histo_array     = np.asarray(base_penetration_rate_total)
    N_base               = base_histo_array.shape[1]
    std_dev_base         = np.std(base_histo_array, axis=1, ddof=1)
    crit_cases_base      = np.mean(base_histo_array, axis=1)  
    crashes_base         = np.mean(np.asarray(base_penetration_rate_crashes), axis = 1)    
    crit_cases_base_err  = (t.ppf(0.975, df=N_base - 1))*(std_dev_base / np.sqrt(N_base)) 
    near_base            = crit_cases_base - crashes_base    

    p_rows = [np.asarray(row) for row in p_values]
    p_array = np.array([
        100.0 * np.count_nonzero(row > 0.05) / row.size
        if row.size > 0 else np.nan
        for row in p_rows
    ], dtype=float)

    if len(pen_grid) > 1: width = (pen_grid[1] - pen_grid[0]) * 0.4
    else: width = 0.05
    x = np.asarray(pen_grid)
    offset = width / 2   

    #Plot colour hex codes 
    baseline_dark  = "#4C72B0" 
    baseline_light = "#A0CBE8"
    slider_dark    = "#DD8452"
    slider_light   = "#F2D0B1"
    dot_colour     = "#808080"
    line_alpha     = 0.5

    fig, ax = plt.subplots(figsize=(8, 4))

    #Plot penetration rate
    ax.bar(x-offset,crashes_base,width=width,edgecolor="black",alpha=0.7, capsize=5,zorder=3,label="Baseline Crashes", color = baseline_dark)
    ax.bar(x-offset,near_base,bottom = crashes_base, width=width,edgecolor="black",alpha=0.7, yerr = crit_cases_base_err, capsize=5,zorder=3,label="Baseline Near Misses", color = baseline_light)
    ax.bar(x+offset,crashes_new,width=width,edgecolor="black",alpha=0.7, capsize=5, zorder=3,label="Slider Crashes", color = slider_dark)
    ax.bar(x+offset,near_new,bottom = crashes_new,width=width,edgecolor="black",alpha=0.7,yerr = crit_cases_new_err, capsize=5, zorder=3,label="Slider Near Misses", color = slider_light)

    ax.set_xlabel("AV Penetration Rate (%)")
    ax.set_ylabel("Number of Critical Cases")
    ax.set_title(f"Total Critical Cases vs. Penetration Rate, ({sweep_count} Runs, {args.chance[0]}-{args.chance[1]}-{args.chance[2]} Slider Distribution)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}" for t in x])
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)    
    tops_base = crit_cases_base + crit_cases_base_err
    tops_new  = crit_cases_new + crit_cases_new_err
    max_bar_top = max(np.nanmax(tops_base), np.nanmax(tops_new))
    ax.set_ylim(0, max_bar_top * 1.1)  

    #Plot percentage of p_chi values over 0.05
    ax2 = ax.twinx()
    ax2.plot(x,p_array,linestyle="-",linewidth=1.25,color=dot_colour,alpha=line_alpha,zorder=2)    
    ax2.scatter(x,p_array,marker="o",s=60, linewidths = 1, facecolors = dot_colour, edgecolors = "black", zorder=3,label="% p_chi > 0.05")
    ax2.set_ylabel("Percentage of p-values (χ²) > 0.05")
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])    
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.5, markerscale=0.6, labelspacing = 0.25, handletextpad = 0.25) #bbox_to_anchor=(1.0, 0.8),
    fig.tight_layout()
    fig.savefig(out_dir / "critical_vs_penetration.png", dpi=600)
    plt.close(fig)     

    #Define mean for the baseline control - Critical cases plot
    base_histo_speed     = np.asarray(avg_last_base)
    last_speed_base      = np.mean(base_histo_speed, axis = 1)  

    #Define mean for the simulation - Critical cases plot
    new_histo_speed     = np.asarray(avg_last_new)
    last_speed_new      = np.mean(new_histo_speed, axis = 1)  
  
    #Plot average speed figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x-offset,last_speed_base,width=width,edgecolor="black",alpha=0.7, capsize = 5, label="Baseline Speed",color=baseline_dark)
    ax.bar(x+offset,last_speed_new,width=width,edgecolor="black",alpha=0.7, capsize = 5, label="Slider Speed",color=slider_dark)
    ax.set_xlabel("AV Penetration Rate (%)")
    ax.set_ylabel("Average Speed (m/s)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}" for t in x])    
    ax.set_ylim(0,10)
    ax.set_title(f"Average Speed Last Car, Baseline vs. Slider, ({sweep_count} Runs, {args.chance[0]}-{args.chance[1]}-{args.chance[2]} Slider Distribution)")
    ax.legend(loc="upper right", fontsize=6.5, markerscale=0.7, labelspacing = 0.3, handletextpad = 0.3)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "speed_critical_vs_penetration.png", dpi=600)
    plt.close(fig)     
           
    mm, ss = divmod(time.perf_counter() - t0, 60)
    print(f"Time elapsed on this simulation: {int(mm):02d}:{int(ss):02d}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)