import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable, Sequence

ArrayLike = Union[np.ndarray, Sequence[float], None]

def build_cycles_fromEvents(
        ipsi_fs: ArrayLike,
        ipsi_fo: ArrayLike = None,
        contra_fs: ArrayLike = None,
        contra_fo: ArrayLike = None,
        check_order_for_complete: bool = True,
    ) -> List[Dict[str, object]]:
    """
    Returns a list of dicts with mandatory keys:
      - start
      - end
      - type : "cycle" | "partial_gaitCycle" | "complete_gaitCycle"

    Optional keys (added only if present in window):
      - footOff
      - controlateral_footStrike
      - controlateral_footOff

    If check_order_for_complete=True, complete cycles must satisfy:
      start < controlateral_footOff < controlateral_footStrike < footOff < end
    otherwise they are downgraded to "partial_gaitCycle" (kept, but not 'complete').
    """
    def _as_float_array(x: ArrayLike) -> np.ndarray:
        if x is None:
            return np.asarray([], dtype=float)
        return np.asarray(x, dtype=float).reshape(-1)


    def _first_in_window(events: np.ndarray, t0: float, t1: float) -> Optional[float]:
        if events.size == 0:
            return None
        cand = events[(events > t0) & (events < t1)]
        return float(cand[0]) if cand.size else None


    ipsi_fs_arr = _as_float_array(ipsi_fs)
    if ipsi_fs_arr.size < 2:
        return []

    ipsi_fo_arr   = _as_float_array(ipsi_fo)
    contra_fs_arr = _as_float_array(contra_fs)
    contra_fo_arr = _as_float_array(contra_fo)

    out: List[Dict[str, object]] = []

    for i in range(len(ipsi_fs_arr) - 1):
        start = float(ipsi_fs_arr[i])
        end   = float(ipsi_fs_arr[i + 1])

        footOff = _first_in_window(ipsi_fo_arr, start, end)
        cFS     = _first_in_window(contra_fs_arr, start, end)
        cFO     = _first_in_window(contra_fo_arr, start, end)

        d: Dict[str, object] = {"start": start, "end": end}

        # always start from minimal classification
        cycle_type = "cycle"

        if footOff is not None:
            d["footOff"] = footOff
            cycle_type = "partial_gaitCycle"

        # complete only if all three exist
        if (footOff is not None) and (cFS is not None) and (cFO is not None):
            d["controlateral_footStrike"] = cFS
            d["controlateral_footOff"] = cFO

            if check_order_for_complete:
                ok = (start < cFO < cFS < footOff < end)
                if ok:
                    cycle_type = "complete_gaitCycle"
                else:
                    # keep it, but don't call it complete
                    cycle_type = "partial_gaitCycle"
                    d["order_issue"] = True
            else:
                cycle_type = "complete_gaitCycle"
        else:
            # if one of controlat events exists but not both, you can still store it
            if cFS is not None:
                d["controlateral_footStrike"] = cFS
            if cFO is not None:
                d["controlateral_footOff"] = cFO

        d["type"] = cycle_type
        out.append(d)

    return out
