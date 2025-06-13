# carfollow/idm.py
# -------------------------------------------------
from __future__     import annotations
import math
import numpy            as np
from numba              import njit, cuda

from .                  import config as C
from .functions         import ttc

@njit(inline="always")  
def _clip_scalar(val, lo, hi):
    return lo if val < lo else hi if val > hi else val

@njit(fastmath=True, cache=True)
def idm(
    v_f             : float,
    x_f             : float,
    v_lead          : float,
    x_lead          : float,
    dt              : float,
    size_lead       : float,
    size_follow     : float,
    prev_accel      : float = 0.0,     
    v_des           : float = C.MAX_DESIRED_VELOCITY,
    delta           : int   = 4,
    desired_jerk    : float = C.DESIRED_JERK,
    desired_accel   : float = C.DESIRED_ACCELERATION,
    desired_decel   : float = C.DESIRED_DECELERATION,   
    desired_sft     : float = C.SAFE_FOLLOW_TIME,       
) -> tuple[float, float]:

    ttc_val = ttc(v_f, v_lead, x_f, x_lead, size_lead, size_follow)

    if ttc_val < C.TTC_THRESHOLD:
        a_max   =  C.MAX_ACCELERATION
        b_limit = C.MAX_DECELERATION
        j_max   =  C.MAX_JERK   
    else:
        a_max   =  desired_accel
        b_limit = desired_decel
        j_max   =  desired_jerk   

    s0   = C.MIN_DISTANCE
    T    = desired_sft
    b_comf = b_limit
    dv   = v_f - v_lead
    dx   = x_lead - x_f - 0.5*size_lead - 0.5*size_follow
    dx = max(dx, 0.1)
    
    s_star = s0 + v_f*T + v_f*dv / (2*np.sqrt(a_max * b_comf))

    accel_idm = a_max * (1 - (v_f/v_des)**delta - (s_star/dx)**2) 


    # Accel limiter
    accel_clip = _clip_scalar(accel_idm, -b_limit, a_max)


    # Jerk limiter
    max_d_a = j_max * dt

    accel = _clip_scalar(accel_clip, prev_accel-max_d_a, prev_accel+max_d_a)

    v_next = max(0, v_f + accel * dt)
    x_next = x_f + v_f * dt + 0.5 * accel * dt**2

    return v_next, x_next, accel


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
