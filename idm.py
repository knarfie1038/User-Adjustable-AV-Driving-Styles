# carfollow/idm.py
# -------------------------------------------------
from __future__         import annotations
import math
from numba              import cuda

@cuda.jit(device=True, inline=True)
def idm_gpu(
    v_f: float,
    x_f: float,
    v_lead: float,
    x_lead: float,
    dt: float,
    size_lead: float,
    size_follow: float,
    prev_accel: float,
    v_des: float,
    delta: int,
    desired_jerk: float,
    desired_accel: float,
    desired_decel: float,
    desired_sft: float,
    ttc_threshold: float,
    max_accel: float,
    max_decel: float,
    max_jerk: float,
    min_dist: float,
) -> tuple[float, float, float]:
    
    dx = x_lead - x_f - 0.5 * size_lead - 0.5 * size_follow
    rel_v = v_f - v_lead
    if (rel_v > 0.0) and (v_lead>0):    ttc_val = dx / rel_v
    else:                               ttc_val = float('inf')

    if ttc_val < ttc_threshold:
        a_max = max_accel
        b_limit = max_decel
        j_max = max_jerk
    else:
        a_max = desired_accel
        b_limit = desired_decel
        j_max = desired_jerk

    T = desired_sft
    dv = rel_v
    dx = dx if dx > 0.1 else 0.1

    s_star = min_dist + v_f * T + v_f * dv / (2.0 * math.sqrt(a_max * b_limit))
    accel_idm = a_max * (1.0 - (v_f / v_des) ** delta - (s_star / dx) ** 2)

    # Accel Limiter
    if accel_idm < -b_limit:                    accel_idm = -b_limit
    elif accel_idm > a_max:                     accel_idm = a_max

    # Jerk Limiter
    da_max = j_max * dt
    if accel_idm < (prev_accel - da_max):       accel_idm = prev_accel - da_max
    elif accel_idm > (prev_accel + da_max):     accel_idm = prev_accel + da_max

    # Integrate to obtain v and x from a
    v_next = v_f + accel_idm * dt
    if v_next < 0.0:
        v_next = 0.0
    x_next = x_f + v_f * dt + 0.5 * accel_idm * dt * dt

    return v_next, x_next, accel_idm