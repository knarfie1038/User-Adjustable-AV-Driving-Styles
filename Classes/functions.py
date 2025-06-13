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



