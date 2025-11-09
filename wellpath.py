# wellpath.py
# Pure math and survey builders for J, S, and Horizontal profiles using
# the Minimum Curvature Method (MCM). ASCII-only to avoid Unicode issues.

from __future__ import annotations
import math
import numpy as np
import pandas as pd

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def _rf(dogs: float) -> float:
    # Ratio factor for minimum curvature (dogs in radians)
    if abs(dogs) < 1e-12:
        return 1.0
    return 2.0 / dogs * math.tan(0.5 * dogs)

def _min_curve_step(md1, inc1_deg, az1_deg, md2, inc2_deg, az2_deg,
                    n1, e1, tvd1):
    # One MCM step from station 1 to 2
    ds = md2 - md1
    inc1 = inc1_deg * DEG2RAD
    inc2 = inc2_deg * DEG2RAD
    az1  = az1_deg  * DEG2RAD
    az2  = az2_deg  * DEG2RAD

    cos_dog = math.cos(inc1)*math.cos(inc2) + math.sin(inc1)*math.sin(inc2)*math.cos(az2-az1)
    # Clamp for safety
    cos_dog = max(-1.0, min(1.0, cos_dog))
    dogs = math.acos(cos_dog)

    rf = _rf(dogs)

    dN = 0.5 * ds * (math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2)) * rf
    dE = 0.5 * ds * (math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2)) * rf
    dT = 0.5 * ds * (math.cos(inc1) + math.cos(inc2)) * rf

    n2 = n1 + dN
    e2 = e1 + dE
    tvd2 = tvd1 + dT

    dls_deg_per_100 = 0.0
    if ds > 0:
        dls_deg_per_100 = dogs * RAD2DEG / ds * 100.0

    return n2, e2, tvd2, dls_deg_per_100

def _build_inc(inc_deg, sign, rate_deg_per_100ft, ds_ft):
    return inc_deg + sign * rate_deg_per_100ft * (ds_ft / 100.0)

def _to_metric_if_needed(arr_ft, metric):
    if not metric:
        return np.array(arr_ft, dtype=float)
    # Convert ft->m
    return np.array(arr_ft, dtype=float) * 0.3048

def _from_metric_if_needed(val_m, metric):
    if not metric:
        return val_m
    return val_m / 0.3048

def _advance_profile(md, inc, az, ds, metric, md_stop=None, tvd_stop=None):
    # Returns one new station after stepping ds along hole.
    # If metric, MD and ds are in meters internally; convert at IO.
    return md + ds, inc, az

def _synthesize_survey(well_type: str,
                       kop_md_ft: float,
                       azimuth_deg: float,
                       ds_ft: float,
                       build_rate: float,
                       theta_hold_deg: float | None,
                       target_md_ft: float | None,
                       target_tvd_ft: float | None,
                       hold_length_ft: float | None,
                       drop_rate: float | None,
                       final_inc_after_drop_deg: float | None,
                       lateral_length_ft: float | None) -> pd.DataFrame:
    """
    Build a synthetic survey MD,Inc,Az for the chosen profile.
    Strategy: step along MD in ds_ft increments and update inclination by
    build/drop rates when in the corresponding section.
    Azimuth is held constant per your spec.
    """

    md = [0.0]
    inc = [0.0]
    az  = [azimuth_deg]
    # Track where we are relative to sections
    current_md = 0.0

    def done():
        if target_md_ft is not None and current_md >= target_md_ft - 1e-6:
            return True
        return False

    # Helper flags
    after_kop = False
    in_hold = False
    in_drop = False
    built_to_theta = False
    drop_finished = False
    hold_md_remaining = hold_length_ft if hold_length_ft is not None else None
    lateral_md_remaining = lateral_length_ft if lateral_length_ft is not None else None

    while True:
        if done():
            break

        ds = ds_ft
        next_md = current_md + ds

        prev_inc = inc[-1]
        next_inc = prev_inc

        if not after_kop and next_md >= kop_md_ft:
            after_kop = True

        if after_kop:
            if well_type == "Build & Hold" or well_type == "Horizontal (Continuous Build + Lateral)":
                # Build up to theta_hold_deg (or 90 for horizontal) then hold/lateral
                target_theta = theta_hold_deg if well_type == "Build & Hold" else 90.0
                if not built_to_theta:
                    next_inc = min(prev_inc + build_rate * (ds/100.0), target_theta)
                    if abs(next_inc - target_theta) < 1e-6:
                        built_to_theta = True
                else:
                    # hold or lateral
                    next_inc = target_theta
                    if well_type == "Horizontal (Continuous Build + Lateral)":
                        if lateral_md_remaining is not None:
                            lateral_md_remaining -= ds
                            if lateral_md_remaining <= 0:
                                # Stop once lateral is drilled
                                next_md = current_md + (ds + lateral_md_remaining)
                                md.append(next_md)
                                inc.append(next_inc)
                                az.append(azimuth_deg)
                                break

            elif well_type == "Build & Hold & Drop":
                # Build to theta_hold_deg, hold for hold_length_ft (if provided), then drop
                if not built_to_theta:
                    next_inc = min(prev_inc + build_rate * (ds/100.0), theta_hold_deg)
                    if abs(next_inc - theta_hold_deg) < 1e-6:
                        built_to_theta = True
                        in_hold = True if (hold_md_remaining is not None and hold_md_remaining > 0) else False
                elif in_hold:
                    next_inc = theta_hold_deg
                    if hold_md_remaining is not None:
                        hold_md_remaining -= ds
                        if hold_md_remaining <= 0:
                            in_hold = False
                            in_drop = True
                elif in_drop:
                    # Drop toward final_inc_after_drop_deg (often 0)
                    next_inc = max(prev_inc - (drop_rate or 0.0) * (ds/100.0), final_inc_after_drop_deg or 0.0)
                    if abs(next_inc - (final_inc_after_drop_deg or 0.0)) < 1e-6:
                        in_drop = False
                        drop_finished = True
                else:
                    # After drop finished, maintain final inclination
                    next_inc = final_inc_after_drop_deg or 0.0

        md.append(next_md)
        inc.append(next_inc)
        az.append(azimuth_deg)
        current_md = next_md

        if target_tvd_ft is not None:
            # We stop when TVD reaches target; to check that, we need positions.
            # Defer stopping on TVD to the conversion phase in the app.

            pass

        if target_md_ft is not None and current_md >= target_md_ft - 1e-6:
            break

        # Safety to avoid infinite loops
        if len(md) > 20000:
            break

    df = pd.DataFrame({"MD_ft": md, "Inc_deg": inc, "Az_deg": az})
    return df

def build_survey(profile: str,
                 kop_md_ft: float,
                 azimuth_deg: float,
                 ds_ft: float,
                 build_rate: float,
                 theta_hold_deg: float | None = None,
                 target_md_ft: float | None = None,
                 target_tvd_ft: float | None = None,
                 hold_length_ft: float | None = None,
                 drop_rate: float | None = None,
                 final_inc_after_drop_deg: float | None = None,
                 lateral_length_ft: float | None = None) -> pd.DataFrame:
    return _synthesize_survey(profile,
                              kop_md_ft, azimuth_deg, ds_ft,
                              build_rate, theta_hold_deg,
                              target_md_ft, target_tvd_ft,
                              hold_length_ft, drop_rate, final_inc_after_drop_deg,
                              lateral_length_ft)

def mcm_positions(df_md_inc_az: pd.DataFrame) -> pd.DataFrame:
    # Compute TVD/N/E and DLS via minimum curvature from the synthetic survey
    md = df_md_inc_az["MD_ft"].to_numpy()
    inc = df_md_inc_az["Inc_deg"].to_numpy()
    az  = df_md_inc_az["Az_deg"].to_numpy()

    n = [0.0]; e = [0.0]; tvd = [0.0]; dls = [0.0]

    for i in range(1, len(md)):
        n2, e2, t2, dls_i = _min_curve_step(md[i-1], inc[i-1], az[i-1],
                                            md[i],   inc[i],   az[i],
                                            n[-1], e[-1], tvd[-1])
        n.append(n2); e.append(e2); tvd.append(t2); dls.append(dls_i)

    out = df_md_inc_az.copy()
    out["TVD_ft"] = tvd
    out["North_ft"] = n
    out["East_ft"] = e
    out["DLS_deg_per_100ft"] = dls
    return out
