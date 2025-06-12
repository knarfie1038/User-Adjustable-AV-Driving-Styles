# carfollow/functions.py
# -------------------------------------------------
import      zarr
import      os
from numba  import cuda

from .      import config as C

#Parse the path to dataset
def path_parse(path: str = "/trainHH.zarr"):
    base_path = C.PATH_ZARR
    return zarr.open(os.path.join(base_path, path), mode="r")

#TTC function
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