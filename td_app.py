# wellpath.py  —  ASCII only
# Build synthetic J / S / Horizontal surveys and convert to TVD/N/E via
# the Minimum Curvature Method (MCM).

from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def _rf(dogs_rad: float) -> float:
    """Ratio Factor (RF) for Minimum Curvature.
       RF = 2/dogs * tan(dogs/2); as dogs -> 0, RF -> 1."""
    if abs(dogs_rad) < 1e-12:
        return 1.0
    return 2.0 / dogs_rad * math.tan(0.5 * dogs_rad)

def _mcm_step(md1, inc1_deg, az1_deg, md2, inc2_deg, az2_deg,
              n1, e1, tvd1) -> Tuple[float, float, float, float]:
    """One Minimum Curvature step. Returns (N2, E2, TVD2, DLS_deg_per_100ft)."""
    ds = md2 - md1
    inc1 = inc1_deg * DEG2RAD; inc2 = inc2_deg * DEG2RAD
    az1  = az1_deg  * DEG2RAD; az2  = az2_deg  * DEG2RAD

    cos_dog = (math.cos(inc1)*math.cos(inc2) +
               math.sin(inc1)*math.sin(inc2)*math.cos(az2-az1))
    cos_dog = max(-1.0, min(1.0, cos_dog))
    dogs = math.acos(cos_dog)     # dogleg angle (rad)
    rf = _rf(dogs)

    dN = 0.5 * ds * (math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2)) * rf
    dE = 0.5 * ds * (math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2)) * rf
    dT = 0.5 * ds * (math.cos(inc1) + math.cos(inc2)) * rf

    n2 = n1 + dN; e2 = e1 + dE; tvd2 = tvd1 + dT
    dls = 0.0 if ds <= 0 else dogs * RAD2DEG / ds * 100.0  # deg/100 ft
    return n2, e2, tvd2, dls

def _synthesize(profile: str,
                kop_md_ft: float,
                azimuth_deg: float,
                ds_ft: float,
                build_rate: float,
                theta_hold_deg: Optional[float],
                target_md_ft: Optional[float],
                hold_length_ft: Optional[float],
                drop_rate: Optional[float],
                final_inc_after_drop_deg: Optional[float],
                lateral_length_ft: Optional[float]) -> pd.DataFrame:
    """Build synthetic survey table (MD_ft, Inc_deg, Az_deg) for J / S / Horizontal."""
    md = [0.0]; inc = [0.0]; az = [azimuth_deg]
    cur_md = 0.0

    # Section flags
    after_kop = False
    built_to_theta = False
    in_hold = False
    in_drop = False
    hold_left = hold_length_ft if hold_length_ft is not None else None
    lateral_left = lateral_length_ft if lateral_length_ft is not None else None

    def stop_on_md() -> bool:
        return (target_md_ft is not None) and (cur_md >= target_md_ft - 1e-6)

    while True:
        if stop_on_md():
            break

        ds = ds_ft
        next_md = cur_md + ds
        prev_inc = inc[-1]
        nxt_inc = prev_inc

        if (not after_kop) and (next_md >= kop_md_ft):
            after_kop = True

        if after_kop:
            if profile == "Build & Hold" or profile.startswith("Horizontal"):
                # Build to theta_hold_deg (J) or to 90 (Horizontal), then hold / lateral.
                target_theta = theta_hold_deg if profile == "Build & Hold" else 90.0
                if not built_to_theta:
                    nxt_inc = min(prev_inc + build_rate * (ds / 100.0), target_theta)
                    built_to_theta = abs(nxt_inc - target_theta) < 1e-9
                else:
                    nxt_inc = target_theta
                    if profile.startswith("Horizontal") and lateral_left is not None:
                        lateral_left -= ds
                        if lateral_left <= 0.0:
                            # truncate last step neatly
                            next_md = cur_md + (ds + lateral_left)
                            md.append(next_md); inc.append(nxt_inc); az.append(azimuth_deg)
                            break

            elif profile == "Build & Hold & Drop":
                # Build to theta_hold, optional hold, then drop to final inc.
                th = float(theta_hold_deg or 0.0)
                if not built_to_theta:
                    nxt_inc = min(prev_inc + build_rate * (ds / 100.0), th)
                    built_to_theta = abs(nxt_inc - th) < 1e-9
                    in_hold = (built_to_theta and (hold_left is not None) and (hold_left > 0.0))
                elif in_hold:
                    nxt_inc = th
                    if hold_left is not None:
                        hold_left -= ds
                        if hold_left <= 0.0:
                            in_hold = False; in_drop = True
                elif in_drop:
                    final_th = float(final_inc_after_drop_deg or 0.0)
                    nxt_inc = max(prev_inc - float(drop_rate or 0.0) * (ds / 100.0), final_th)
                    if abs(nxt_inc - final_th) < 1e-9:
                        in_drop = False
                else:
                    nxt_inc = float(final_inc_after_drop_deg or 0.0)

        md.append(next_md); inc.append(nxt_inc); az.append(azimuth_deg)
        cur_md = next_md

        if len(md) > 25000:  # safety
            break

    return pd.DataFrame({"MD_ft": md, "Inc_deg": inc, "Az_deg": az})

def build_survey(profile: str,
                 kop_md_ft: float,
                 azimuth_deg: float,
                 ds_ft: float,
                 build_rate: float,
                 theta_hold_deg: Optional[float] = None,
                 target_md_ft: Optional[float] = None,
                 hold_length_ft: Optional[float] = None,
                 drop_rate: Optional[float] = None,
                 final_inc_after_drop_deg: Optional[float] = None,
                 lateral_length_ft: Optional[float] = None) -> pd.DataFrame:
    return _synthesize(profile, kop_md_ft, azimuth_deg, ds_ft, build_rate,
                       theta_hold_deg, target_md_ft, hold_length_ft,
                       drop_rate, final_inc_after_drop_deg, lateral_length_ft)

def mcm_positions(df_md_inc_az: pd.DataFrame) -> pd.DataFrame:
    """Apply Minimum Curvature to MD/Inc/Az to get TVD/N/E and DLS."""
    md = df_md_inc_az["MD_ft"].to_numpy()
    inc = df_md_inc_az["Inc_deg"].to_numpy()
    az  = df_md_inc_az["Az_deg"].to_numpy()

    north = [0.0]; east = [0.0]; tvd = [0.0]; dls = [0.0]

    for i in range(1, len(md)):
        n2, e2, t2, dls_i = _mcm_step(md[i-1], inc[i-1], az[i-1],
                                      md[i],   inc[i],   az[i],
                                      north[-1], east[-1], tvd[-1])
        north.append(n2); east.append(e2); tvd.append(t2); dls.append(dls_i)

    out = df_md_inc_az.copy()
    out["TVD_ft"] = tvd
    out["North_ft"] = north
    out["East_ft"]  = east
    out["DLS_deg_per_100ft"] = dls
    return out
