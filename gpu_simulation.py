# carfollow/gpu_simulation.py
# -------------------------------------------------
from __future__         import annotations
from numba              import cuda
import numpy            as np

from .functions         import path_parse, ttc_gpu
from .idm               import idm_gpu
from .                  import config as C

#Sample random from gaussian distribution
def sample_gaussian(n, param_list):
    μ, σ = param_list
    return np.random.normal(loc=μ, scale=σ, size=n)

#Load data from dataset
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

#GPU function for the simulation of a case
@cuda.jit(fastmath = True)
def _gpu_kernel(
    idx,
    dt,
    size_lead_arr,
    size_follow_arr,
    x_lead,
    x_follow,
    v_lead,
    a_lead,
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
    out_dx,
    out_avg_lead,
    out_avg_last,
    car_count,
    start_distance,
    case_id_d,
):
    case_id = cuda.grid(1)
    n_cases = idx.shape[0]
    if case_id >= n_cases:        return

    size_lead = size_lead_arr[case_id]
    size_follow = size_follow_arr[case_id]

    start = int(idx[case_id, 0])
    end   = int(idx[case_id, 1])
    span = end - start   

    v_lead_total = 0.0
    v_last_total = 0.0

    for t in range(start, end):
        v_lead_total += v_lead[t]
    out_avg_lead[case_id] = v_lead_total / span

    for j in range(car_count):
        param_car = case_id * car_count + j
        jerk_val  = jerk_array[param_car]
        accel_val = accel_array[param_car]
        decel_val = decel_array[param_car]
        sft_val   = sft_array[param_car]
        case_id_d[param_car] = case_id       

        if j == 0:
            v_curr   = v_lead[start]
            x_curr   = x_follow[start] - v_curr * sft_val           
        else:
            v_curr = v_lead[start]
            x_curr = x_lead[start] - start_distance

        prev_acc = a_lead[start]
        if prev_acc < -max_decel:
            prev_acc = -max_decel
        elif prev_acc > max_accel:
            prev_acc = max_accel

        ttc_min_local = 1e12
        tet_local     = 0
        dx_min_local  = 1e12

        for t in range(start, end):
            v_L = v_lead[t]; x_L = x_lead[t]

            v_next, x_next, a_next = idm_gpu(
                v_curr, x_curr, v_L, x_L,
                dt,
                size_lead, size_follow,
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
            ttc_val = ttc_gpu(v_next, v_L, x_next, x_L,
                              size_lead, size_follow)
            dx      = (x_L - x_next) - 0.5*size_lead - 0.5*size_follow

            if ttc_val < ttc_min_local:
                ttc_min_local = ttc_val
            if ttc_val < ttc_threshold:
                tet_local += 1
            if dx < dx_min_local:
                dx_min_local = dx

            if j == car_count-1:
                v_last_total += v_next                

            v_curr, x_curr, prev_acc = v_next, x_next, a_next     

        out_ttc[param_car] = ttc_min_local
        out_tet[param_car] = tet_local
        out_dx[param_car]  = dx_min_local 

    out_avg_last[case_id]   = v_last_total / span

#Simulate gpu function
def simulate_gpu(
    path: str,
    *,
    penetration: float | None = None,
    probability: list | None = None,
    base = False,
) -> np.ndarray:
    (idx,dt,size_lead_array,size_follow_array,x_lead,v_lead,a_lead,_,x_follow,_,_,_) = _load_data(path)
    n_cases = idx.shape[0] * C.TOTAL_CARS

    #Helper function for assigning random arrays
    def _set(mask, params):
        random_jerk_array[mask]       = params[0]
        random_accel_array[mask]      = params[1]
        random_decel_array[mask]      = params[2]
        random_safefollow_array[mask] = params[3]    

    #Basline reading
    if base == True:
        penetration_value = np.random.rand(n_cases)
        is_AV   = penetration_value < penetration
        is_human = ~is_AV

        random_jerk_array       = np.empty(n_cases, dtype=np.float64)
        random_accel_array      = np.empty(n_cases, dtype=np.float64)
        random_decel_array      = np.empty(n_cases, dtype=np.float64)
        random_safefollow_array = np.empty(n_cases, dtype=np.float64)

        _set(is_AV,   C.BALANCED_PARAMS)

        if is_human.any():
            hm, hd = C.HUMAN_MEANS, C.HUMAN_DEVIATIONS
            count  = int(is_human.sum())
            random_jerk_array[is_human]       = sample_gaussian(count, [hm[0], hd[0]])
            random_accel_array[is_human]      = sample_gaussian(count, [hm[1], hd[1]])
            random_decel_array[is_human]      = sample_gaussian(count, [hm[2], hd[2]])
            random_safefollow_array[is_human] = sample_gaussian(count, [hm[3], hd[3]])       

    #Simulation reading             
    else:
        penetration_value = np.random.rand(n_cases)
        is_AV   = penetration_value < penetration
        is_human = ~is_AV

        probability_value = np.random.rand(n_cases)
        Conservative = is_AV & (probability_value < probability[0])
        Balanced = is_AV & (probability_value >= probability[0]) & (probability_value < probability[0] + probability[1])
        Aggressive = is_AV & ~(Conservative | Balanced)

        random_jerk_array       = np.empty(n_cases, dtype=np.float64)
        random_accel_array      = np.empty(n_cases, dtype=np.float64)
        random_decel_array      = np.empty(n_cases, dtype=np.float64)
        random_safefollow_array = np.empty(n_cases, dtype=np.float64)

        _set(Conservative, C.CONSERVATIVE_PARAMS)
        _set(Balanced, C.BALANCED_PARAMS)
        _set(Aggressive, C.AGGRESIVE_PARAMS)

        if is_human.any():
            hm, hd = C.HUMAN_MEANS, C.HUMAN_DEVIATIONS
            count  = int(is_human.sum())
            random_jerk_array[is_human]       = sample_gaussian(count, [hm[0], hd[0]])
            random_accel_array[is_human]      = sample_gaussian(count, [hm[1], hd[1]])
            random_decel_array[is_human]      = sample_gaussian(count, [hm[2], hd[2]])
            random_safefollow_array[is_human] = sample_gaussian(count, [hm[3], hd[3]])

    print(random_jerk_array, random_accel_array, random_decel_array, random_safefollow_array)

    car_count = int(C.TOTAL_CARS)  
    start_distance = int(C.START_DISTANCE)     

    idx_d               = cuda.to_device(idx)
    size_lead_d         = cuda.to_device(size_lead_array)
    size_follow_d       = cuda.to_device(size_follow_array)
    x_lead_d            = cuda.to_device(x_lead)
    x_follow_d          = cuda.to_device(x_follow)
    v_lead_d            = cuda.to_device(v_lead)
    a_lead_d            = cuda.to_device(a_lead)
    jerk_array          = cuda.to_device(random_jerk_array.astype(np.float64))
    accel_array         = cuda.to_device(random_accel_array.astype(np.float64))
    decel_array         = cuda.to_device(random_decel_array.astype(np.float64))
    safefollow_array    = cuda.to_device(random_safefollow_array.astype(np.float64))
    case_id_d           = cuda.to_device(np.repeat(np.arange(idx.shape[0], dtype=np.int32),C.TOTAL_CARS))    

    out_ttc_min         = cuda.device_array(n_cases, dtype=np.float64)
    out_tet_all         = cuda.device_array(n_cases, dtype=np.int32)
    out_dx_min          = cuda.device_array(n_cases, dtype=np.float64)
    out_avg_lead        = cuda.device_array(idx.shape[0], dtype=np.float64)
    out_avg_last        = cuda.device_array(idx.shape[0], dtype=np.float64)

    n_threads = n_cases
    threads_per_block = 64
    blocks = (n_threads + threads_per_block - 1) // threads_per_block

    car_positions = np.tile(np.arange(1, C.TOTAL_CARS+1, dtype=np.int32),idx.shape[0])

    _gpu_kernel[blocks, threads_per_block](
        idx_d,
        dt,
        size_lead_d,
        size_follow_d,
        x_lead_d,
        x_follow_d,
        v_lead_d,
        a_lead_d,
        jerk_array,  
        accel_array,
        decel_array,
        safefollow_array,  
        C.TTC_THRESHOLD,
        C.MAX_ACCELERATION,
        C.MAX_DECELERATION,
        C.MAX_JERK,
        C.MIN_DISTANCE,
        out_ttc_min,
        out_tet_all,
        out_dx_min,
        out_avg_lead,
        out_avg_last,
        car_count,
        start_distance,
        case_id_d,
    )

    return (
    out_ttc_min.copy_to_host(),
    out_tet_all.copy_to_host(),
    out_dx_min.copy_to_host(),
    out_avg_lead.copy_to_host(),
    out_avg_last.copy_to_host(),
    car_positions,
    case_id_d.copy_to_host(),    
    )