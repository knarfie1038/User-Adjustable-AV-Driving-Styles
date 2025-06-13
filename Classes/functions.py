# carfollow/functions.py
# -------------------------------------------------
import zarr
import os
import numpy                as np
import matplotlib.pyplot    as plt
from numba                  import njit, cuda

from .                      import config as C

def path_parse(path: str = "/trainHH.zarr"):
    base_path = C.PATH_ZARR
    return zarr.open(os.path.join(base_path, path), mode="r")


#Sample random from gaussian distribution
def sample_gaussian(n, param_list):
    μ, σ = param_list
    return np.random.normal(loc=μ, scale=σ, size=n)    

@cuda.jit(device=True, inline=True)
def ttc_gpu(v_next: float,
               v_lead: float,
               x_next: float,
               x_lead: float,
               size_lead: float,
               size_follow: float) -> float:
    dx = (x_lead - x_next) - 0.5 * size_lead - 0.5 * size_follow
    rel_v = v_next - v_lead
    if (rel_v > 0.0) and (v_lead > 0.0):
        return dx / rel_v
    else:
        return float('inf')
    
@cuda.jit(device=True, inline=True)
def drac_gpu(v_f: float,
                v_lead: float,
                x_f: float,
                x_lead: float,
                size_lead: float,
                size_follow: float) -> float:
    dv = v_f - v_lead
    num = dv * dv
    denominator = (x_lead - x_f) - 0.5 * size_lead - 0.5 * size_follow
    
    if denominator <= 0.0:      return float('inf')
    return num / (2.0 * denominator)    

@njit(fastmath=True, cache=True)
def ttc(vnext, vl, xnext, xl, size_lead, size_follow):
    dx = (xl - xnext) - 0.5*size_lead - 0.5*size_follow
    relative_velocity = vnext - vl
    if relative_velocity > 0 and vl>0:
        ttc = dx / relative_velocity
    else:
        ttc = float('inf') 
    return ttc   

def size(size_lead, size_follow):
    return - 0.5*size_lead - 0.5*size_follow

@njit(fastmath=True, cache=True)
def drac(v_follower, v_lead, x_follower, x_lead, size_lead, size_follow):
    numerator = (v_follower - v_lead) ** 2
    denominator = 2 * ((x_lead - x_follower) - 0.5 * size_lead - 0.5 * size_follow)
    
    if denominator <= 0:
        drac =  float('inf')  
    else:
        drac = numerator / denominator
    return drac

def dataset(n, time, store, accel=False):
    timestamps = store.timestamp[time]
    dt = np.median(np.diff(timestamps))        
    
    try:
        size_lead = store.lead_size[n]
    except:
        size_lead = 4.85   
    try:
        size_follow = store.follow_size[n]
    except:
        size_follow = 4.85

    x_lead = store.lead_centroid[time]
    v_lead = store.lead_velocity[time]
    x_follow = store.follow_centroid[time]
    v_follow = store.follow_velocity[time]  

    if accel == False:
        return timestamps, dt, size_lead, size_follow, x_lead, v_lead, x_follow, v_follow
    if accel == True:
        a_lead = store.lead_acceleration[time]
        jerk_lead = np.gradient(a_lead, timestamps)
        a_follow = store.follow_acceleration[time]
        jerk_follow = np.gradient(a_follow, timestamps)        
        return timestamps, dt, size_lead, size_follow, x_lead, v_lead, a_lead, jerk_lead, x_follow, v_follow, a_follow, jerk_follow

def plot_function(axs, n, timestamps, lead, follow, idm,hidm,  label):
        axs[n].plot(timestamps, lead, label="Lead (data)", color="blue")
        axs[n].plot(timestamps, follow, label="Follower (data)", color="cyan")
        axs[n].plot(timestamps, idm, label="Follower (IDM)", linestyle=":", color="red")
        axs[n].plot(timestamps, hidm, label="Follower (HIDM)", linestyle=":", color="black")        
        axs[n].set_ylabel(label)
        axs[n].legend() 
        axs[n].legend(loc='center left', bbox_to_anchor=(1.01,0.5))


def fleetplot(no,x_list, v_list, a_list, typelist, timestamps, ttc):
    n = len(x_list)
    titles  = ["Position  [m]", "Speed  [m/s]", "Acceleration  [m/s²]", "Jerk  [m/s³]"]
    
    jerk_list = [np.gradient(a, timestamps) for a in a_list]
    variables  = (x_list, v_list, a_list, jerk_list)    
    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(right=0.78, hspace=0.7) 
    gs  = fig.add_gridspec(3, 4, height_ratios=[3, 1.5, 1.5],hspace=0.7)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_ttc = fig.add_subplot(gs[1, :], sharex=axes[0])

    if n == 1:
        axes = [axes]

    for ax, series_list, title in zip(axes, variables, titles):
        ax.set_title(title)
        for i, (y, modeltype) in enumerate(zip(series_list, typelist)):
            if i == 0:
                ax.plot(timestamps, y, label = "Lead car")
            else:
                ax.plot(timestamps, y, label=f"Car {i}: {modeltype}")
        ax.grid(alpha=.3)
    handles, labels = axes[0].get_legend_handles_labels()        

    ax_ttc._get_lines.get_next_color()
    
    ax_ttc.set_title("Time to Collision [s]")
    for i, y in enumerate(ttc, start=1):
        ax_ttc.plot(timestamps, np.clip(y, -1, C.TTC_CLIP), label=f"Car {i}")
    ax_ttc.set(xlabel="Time [s]")
    ax_ttc.set_ylim(-.5, 5.5)
    ttc_line = ax_ttc.axhline(y=C.TTC_THRESHOLD, color='blue', linestyle=':', linewidth=2, label="TTC = 2s")
    ax_ttc.grid(alpha=.3)

    handles.extend([ttc_line])
    labels.extend([ttc_line.get_label()])
    fig.legend(handles, labels,  
            loc='center left',       
            bbox_to_anchor=(0.80, 0.55), 
            ncol=1, fontsize='small', frameon=False)     
    for ax in axes:
        ax.set_xlabel("Time [s]")
    plt.suptitle(f"Kinematics and TTC of a Multi-car Simulation (Case ID: {no})", fontsize=16)
    filename = f"fleet_case_{no:03d}.pdf"
    fig.savefig(filename,dpi=300,bbox_inches="tight",pad_inches=0.1, transparent=False)
    print(f"Saved figure to {filename}")    

