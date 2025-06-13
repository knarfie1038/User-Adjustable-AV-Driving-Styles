# carfollow/gpu_simulation.py
# -------------------------------------------------
from __future__         import annotations

import numpy            as np
import zarr
from numba              import cuda
from pathlib            import Path

from .functions         import path_parse, ttc_gpu, drac_gpu
from .idm               import idm_gpu
from .                  import config as C

import numpy as np

def sample_gaussian(n, param_list):
    μ, σ = param_list
    return np.random.normal(loc=μ, scale=σ, size=n)

def _load_data(path: str):
    store = path_parse(path)
    idx = np.array(store["index_range"][:], dtype=np.int32)
    n_cases = idx.shape[0]
    if "lead_size" in store:
        size_lead = np.asarray(store["lead_size"][:], dtype=np.float64)
    else:
        size_lead = np.full(n_cases, float(store.attrs.get("lead_size", 4.85)), dtype=np.float64)

    if "follow_size" in store:
        size_follow = np.asarray(store["follow_size"][:], dtype=np.float64)
    else:
        size_follow = np.full(n_cases, float(store.attrs.get("follow_size", 4.85)), dtype=np.float64)

    timestamp = np.asarray(store["timestamp"][:], dtype=np.float64)
    dt = float(np.median(np.diff(timestamp)))
    x_lead      = store["lead_centroid"][:].astype(np.float64)
    v_lead      = store["lead_velocity"][:].astype(np.float64)
    a_lead      = store["lead_acceleration"][:].astype(np.float64)
    jerk_lead   = np.gradient(a_lead, timestamp).astype(np.float64)

    x_follow    = store["follow_centroid"][:].astype(np.float64)
    v_follow    = store["follow_velocity"][:].astype(np.float64)
    a_follow    = store["follow_acceleration"][:].astype(np.float64)
    jerk_follow = np.gradient(a_follow, timestamp).astype(np.float64)

    return (idx,dt,size_lead,size_follow,x_lead,v_lead,a_lead,jerk_lead,x_follow,v_follow,a_follow,jerk_follow)


@cuda.jit
def _gpu_kernel(
    idx,
    dt,
    size_lead_arr,
    size_follow_arr,
    x_lead,
    v_lead,
    a_lead,
    jerk_lead,
    x_follow,
    v_follow,
    a_follow,
    jerk_follow,
    jerk_array,    
    accel_array,
    decel_array,
    sft_array,
    ttc_threshold,
    max_accel,
    max_decel,
    max_jerk,
    min_dist,
    out_ttc,
    out_tet,
    out_drac,
    out_dx,
    out_jerk,
):
    i = cuda.grid(1)
    n_cases = idx.shape[0]
    if i >= n_cases:        return
    jerk_val = jerk_array[i]; accel_val = accel_array[i]; decel_val = decel_array[i]; sft_val = sft_array[i]

    size_lead = size_lead_arr[i]
    size_follow = size_follow_arr[i]

    start = int(idx[i, 0])
    end   = int(idx[i, 1])

    ttc_min     = 1e12
    dx_min      = 1e12
    tet_count   = 0
    drac_max    = 0.0
    jerk_max    = 0.0

    v_curr   = v_lead[start]
    x_curr   = x_follow[start] - v_curr * sft_val
    acc0 = a_lead[start]

    if acc0 < -max_decel: acc0 = -max_decel
    elif acc0 >  max_accel: acc0 =  max_accel
    prev_acc = acc0

    

    for t in range(start, end):
        v_next, x_next, a_next = idm_gpu(
            v_curr,
            x_curr,
            v_lead[t],
            x_lead[t],
            dt,
            size_lead,
            size_follow,
            prev_acc,
            C.MAX_DESIRED_VELOCITY,
            4,
            jerk_val,
            accel_val,
            decel_val,
            sft_val,
            ttc_threshold,
            max_accel,
            max_decel,
            max_jerk,
            min_dist,
        )

        ttc_val = ttc_gpu(v_next,v_lead[t],x_next,x_lead[t],size_lead,size_follow)
        if ttc_val < ttc_min:           ttc_min = ttc_val
        if ttc_val < ttc_threshold:     tet_count += 1

        drac_val = drac_gpu(v_next,v_lead[t],x_next,x_lead[t],size_lead,size_follow)
        if drac_val > drac_max:         drac_max = drac_val

        dx = (x_lead[t] - x_next) - 0.5 * size_lead - 0.5 * size_follow
        if dx < dx_min:                 dx_min = dx

        v_curr = v_next; x_curr = x_next; prev_acc = a_next

    out_ttc[i]  = ttc_min
    out_tet[i]  = tet_count
    out_drac[i] = drac_max
    out_dx[i]   = dx_min




def simulate_gpu(
    path: str,
    *,
    desired_jerk: float | None = None,
    desired_accel: float | None = None,
    desired_decel: float | None = None,
    desired_sft: float | None = None,
    base = False,
) -> np.ndarray:
    (
        idx,
        dt,
        size_lead_array,
        size_follow_array,
        x_lead,
        v_lead,
        a_lead,
        jerk_lead,
        x_follow,
        v_follow,
        a_follow,
        jerk_follow,
    ) = _load_data(path)

    n_cases = idx.shape[0]
    if base == True:
        random_jerk_array = np.full(n_cases, C.DESIRED_JERK, dtype=np.float64)
        random_accel_array = np.full(n_cases, C.DESIRED_ACCELERATION, dtype=np.float64)
        random_decel_array = np.full(n_cases, C.DESIRED_DECELERATION, dtype=np.float64)
        random_sft_array = np.full(n_cases, C.SAFE_FOLLOW_TIME, dtype=np.float64)     
    else:   
        # random_jerk_array = np.full(n_cases, desired_jerk, dtype=np.float64)  if desired_jerk  is not None else np.full(n_cases, C.DESIRED_JERK, dtype=np.float64)
        # random_accel_array = np.full(n_cases, desired_accel, dtype=np.float64) if desired_accel is not None else np.full(n_cases, C.DESIRED_ACCELERATION, dtype=np.float64)
        # random_decel_array = np.full(n_cases, desired_decel, dtype=np.float64) if desired_decel is not None else np.full(n_cases, C.DESIRED_DECELERATION, dtype=np.float64)
        # random_sft_array = np.full(n_cases, desired_sft, dtype=np.float64)   if desired_sft   is not None else np.full(n_cases, C.SAFE_FOLLOW_TIME, dtype=np.float64)
        gaussian_params = [[2, 0.25],[1.4, 0.3],[2.2, 0.625], [2, 0.25]]    
        random_jerk_array = sample_gaussian(n_cases, gaussian_params[0])  if desired_jerk  is not None else np.full(n_cases, C.DESIRED_JERK, dtype=np.float64)
        random_accel_array = sample_gaussian(n_cases, gaussian_params[1])  if desired_accel  is not None else np.full(n_cases, C.DESIRED_ACCELERATION, dtype=np.float64)
        random_decel_array = sample_gaussian(n_cases, gaussian_params[2])  if desired_decel  is not None else np.full(n_cases, C.DESIRED_DECELERATION, dtype=np.float64)
        random_sft_array = sample_gaussian(n_cases, gaussian_params[3])  if desired_sft  is not None else np.full(n_cases, C.SAFE_FOLLOW_TIME, dtype=np.float64)


    print(f"Jerk: {random_jerk_array}\n")
    print(f"Accel: {random_accel_array}\n")
    print(f"Decel: {random_decel_array}\n")
    print(f"SFT: {random_sft_array}")

    idx_d           = cuda.to_device(idx)
    size_lead_d     = cuda.to_device(size_lead_array)
    size_follow_d   = cuda.to_device(size_follow_array)
    x_lead_d        = cuda.to_device(x_lead)
    v_lead_d        = cuda.to_device(v_lead)
    a_lead_d        = cuda.to_device(a_lead)
    jerk_lead_d     = cuda.to_device(jerk_lead)
    x_follow_d      = cuda.to_device(x_follow)
    v_follow_d      = cuda.to_device(v_follow)
    a_follow_d      = cuda.to_device(a_follow)
    jerk_follow_d   = cuda.to_device(jerk_follow)
    jerk_array      = cuda.to_device(random_jerk_array.astype(np.float64))
    accel_array     = cuda.to_device(random_accel_array.astype(np.float64))
    decel_array     = cuda.to_device(random_decel_array.astype(np.float64))
    sft_array       = cuda.to_device(random_sft_array.astype(np.float64))

    out_ttc  = cuda.device_array(n_cases, dtype=np.float64)
    out_tet  = cuda.device_array(n_cases, dtype=np.int32)
    out_drac = cuda.device_array(n_cases, dtype=np.float64)
    out_dx   = cuda.device_array(n_cases, dtype=np.float64)
    out_jerk = cuda.device_array(n_cases, dtype=np.float64)



    threads_per_block = 128
    blocks = (n_cases + threads_per_block - 1) // threads_per_block

    _gpu_kernel[blocks, threads_per_block](
        idx_d,
        dt,
        size_lead_d,
        size_follow_d,
        x_lead_d,
        v_lead_d,
        a_lead_d,
        jerk_lead_d,
        x_follow_d,
        v_follow_d,
        a_follow_d,
        jerk_follow_d,
        jerk_array,
        accel_array, 
        decel_array,
        sft_array,
        C.TTC_THRESHOLD,
        C.MAX_ACCELERATION,
        C.MAX_DECELERATION,
        C.MAX_JERK,
        C.MIN_DISTANCE,
        out_ttc,
        out_tet,
        out_drac,
        out_dx,
        out_jerk,
    )
    res = np.empty((n_cases, 4), dtype=np.float64)
    res[:, 0] = out_ttc.copy_to_host()
    res[:, 1] = out_tet.copy_to_host()
    res[:, 2] = out_drac.copy_to_host()
    res[:, 3] = out_dx.copy_to_host()

    return res    