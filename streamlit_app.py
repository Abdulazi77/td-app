from __future__ import annotations
import math
from typing import Dict, Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────── Page ─────────────────────────────
st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")

# ─────────────────────────── Constants ──────────────────────────
DEG2RAD = math.pi / 180.0
IN2FT   = 1.0/12.0
LBF_TO_KN = 4.4482216152605 / 1000.0        # lbf → kN
LBF_FT_TO_KNM = 1.3558179483314 / 1000.0    # lbf-ft → kN·m

def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def bf_from_mw(mw_ppg: float) -> float: return (65.5 - mw_ppg)/65.5  # steel ~65.5 ppg
def I_in4(od_in: float, id_in: float) -> float: return (math.pi/64.0)*(od_in**4 - id_in**4)
def Z_in3(od_in: float, id_in: float) -> float: return I_in4(od_in, id_in) / (od_in/2.0)

# extra section properties
def A_in2(od_in: float, id_in: float) -> float: return (math.pi/4.0)*(od_in**2 - id_in**2)
def J_in4(od_in: float, id_in: float) -> float: return (math.pi/32.0)*(od_in**4 - id_in**4)

def EI_lbf_ft2_from_in4(Epsi_lbf_in2: float, I_in4_val: float) -> float:
    """Convert E (psi) * I (in^4) to EI (lbf·ft^2)."""
    EI_lbf_in2 = Epsi_lbf_in2 * I_in4_val
    return EI_lbf_in2 / (12.0**2)

# ────────────────────── Minimal standards DBs ───────────────────
CASING_DB = {
    "13-3/8": {"weights": {48.0: 12.415, 54.5: 12.347, 61.0: 12.107}},
    "9-5/8":  {"weights": {29.3: 8.921, 36.0: 8.535, 40.0: 8.321}},
    "7":      {"weights": {20.0: 6.366, 23.0: 6.059, 26.0: 5.920}},
    "5-1/2":  {"weights": {17.0: 4.778, 20.0: 4.670, 23.0: 4.560}},
}

TOOL_JOINT_DB = {
    "NC38": {"od": 4.75, "id": 2.25, "T_makeup_ftlbf": 12000, "F_tensile_lbf": 350000, "T_yield_ftlbf": 20000},
    "NC40": {"od": 5.00, "id": 2.25, "T_makeup_ftlbf": 16000, "F_tensile_lbf": 420000, "T_yield_ftlbf": 25000},
    "NC50": {"od": 6.63, "id": 3.00, "T_makeup_ftlbf": 30000, "F_tensile_lbf": 650000, "T_yield_ftlbf": 45000},
}

# ───────────────────────── Survey builders ──────────────────────
def synth_build_hold(kop_md, build_rate_deg_per_100ft, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * (build_rate_deg_per_100ft / 100.0))
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate / 100.0
    dr = drop_rate / 100.0
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * br)
    start_drop = 0.75 * target_md
    theta = np.maximum(0.0, theta - np.maximum(0.0, md - start_drop) * dr)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate / 100.0
    theta = np.minimum(theta_max, np.maximum(0.0, md - kop_md) * br)
    idx = np.where(theta >= theta_max - 1e-6)[0]
    if len(idx):
        m_h = md[idx[0]]
        md_end = max(target_md, m_h + lateral_length)
        md = np.arange(0.0, md_end + ds, ds)
        theta = np.where(md <= m_h, np.minimum(theta_max, np.maximum(0.0, md - kop_md) * br), theta_max)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def mincurv_positions(md, inc_deg, az_deg):
    ds = np.diff(md)
    n = len(md)
    N = np.zeros(n); E = np.zeros(n); TVD = np.zeros(n); DLS = np.zeros(n)
    for i in range(1, n):
        I1 = inc_deg[i-1]*DEG2RAD; A1 = az_deg[i-1]*DEG2RAD
        I2 = inc_deg[i]*DEG2RAD;   A2 = az_deg[i]*DEG2RAD
        dmd = ds[i-1]
        cos_dl = clamp(math.cos(I1)*math.cos(I2) + math.sin(I1)*math.sin(I2)*math.cos(A2 - A1), -1.0, 1.0)
        dpsi = math.acos(cos_dl)
        RF = 1.0 if dpsi < 1e-12 else (2.0/dpsi)*math.tan(dpsi/2.0)
        dN = 0.5*dmd*(math.sin(I1)*math.cos(A1)+math.sin(I2)*math.cos(A2))*RF
        dE = 0.5*dmd*(math.sin(I1)*math.sin(A1)+math.sin(I2)*math.sin(A2))*RF
        dZ = 0.5*dmd*(math.cos(I1)+math.cos(I2))*RF
        N[i] = N[i-1]+dN; E[i] = E[i-1]+dE; TVD[i] = TVD[i-1]+dZ
        DLS[i] = (dpsi/DEG2RAD)/dmd*100.0 if dmd>0 else 0.0
    return N, E, TVD, DLS

# ───────────────────── Soft-string (Johancsik) ──────────────────
def soft_string_stepper(
    md: Iterable[float],
    inc_deg: Iterable[float],
    kappa_rad_per_ft: Iterable[float],
    cased_mask: Iterable[bool],
    comp_along_depth: Iterable[str],
    comp_props: Dict[str, Dict[str, float]],
    mu_slide_cased: float, mu_slide_open: float,
    mu_rot_cased: float,   mu_rot_open: float,
    mw_ppg: float,
    scenario: str = "slackoff",
    WOB_lbf: float = 0.0, Mbit_ftlbf: float = 0.0,
    tortuosity_mode: str = "off",    # "off" | "kappa" | "mu"
    tau: float = 0.0,                # 0.0 .. 0.5 typ
    mu_open_boost: float = 0.0,      # hole cleaning booster
):
    """
    Δs = 1 ft soft-string integration, bit -> surface.
    scenario: "pickup" | "slackoff" | "rotate_off" | "onbottom"
    tortuosity_mode: inflate kappa or mu by (1+tau) in OPEN-HOLE segments.
    """
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg)
    nseg = len(md) - 1
    if nseg <= 0: raise ValueError("Trajectory is empty.")
    inc = np.deg2rad(inc_deg[:-1])

    kappa_all = np.asarray(kappa_rad_per_ft)
    kappa_seg = kappa_all[:-1] if len(kappa_all) == len(md) else kappa_all
    if len(kappa_seg) != nseg: kappa_seg = np.resize(kappa_seg, nseg)

    cased_seg = np.asarray(cased_mask)[:nseg]
    comp_arr  = np.asarray(list(comp_along_depth))[:nseg]

    # per-segment properties
    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s = np.zeros(nseg); mu_r = np.zeros(nseg)
    BF = bf_from_mw(mw_ppg)

    for i in range(nseg):
        comp = comp_arr[i]
        od_in = float(comp_props[comp]['od_in']); id_in = float(comp_props[comp]['id_in']); w_air_ft = float(comp_props[comp]['w_air_lbft'])
        w_air[i] = w_air_ft; w_b[i] = w_air_ft*BF
        r_eff_ft[i] = 0.5*od_in*IN2FT
        if cased_seg[i]:
            mu_s[i] = mu_slide_cased; mu_r[i] = mu_rot_cased
        else:
            mu_s[i] = mu_slide_open + mu_open_boost
            mu_r[i] = mu_rot_open   + mu_open_boost

        # tortuosity penalties
        if not cased_seg[i]:
            if tortuosity_mode == "kappa":
                kappa_seg[i] *= (1.0 + tau)
            elif tortuosity_mode == "mu":
                mu_s[i] *= (1.0 + tau); mu_r[i] *= (1.0 + tau)

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)   # axial force, torque
    dT = np.zeros(nseg);  dM = np.zeros(nseg);   N_side = np.zeros(nseg)

    # boundary condition
    if scenario == "onbottom":
        T[0] = -float(WOB_lbf)       # compressive at bit → negative
        M[0] = float(Mbit_ftlbf)     # motor / bit torque allowed

    # axial gravity sign by scenario
    if scenario == "onbottom":
        sgn_ax = -1.0                # like slackoff but clamped at bit
    else:
        sgn_ax = {"pickup": +1.0, "slackoff": -1.0}.get(scenario, 0.0)

    # only sliding scenarios include axial friction term
    is_sliding = scenario in ("pickup", "slackoff", "onbottom")

    for i in range(nseg):
        N_raw = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        N_side[i] = max(0.0, N_raw)

        axial_drag = mu_s[i]*N_side[i] if is_sliding else 0.0

        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + axial_drag)*ds
        M_next = M[i] + (mu_r[i]*N_side[i]*r_eff_ft[i])*ds

        dT[i] = T_next - T[i]; dM[i] = M_next - M[i]
        T[i+1] = T_next; M[i+1] = M_next

    df = pd.DataFrame({
        "md_top_ft": md[:-1], "md_bot_ft": md[1:], "ds_ft": 1.0,
        "inc_deg": inc_deg[:-1], "kappa_rad_ft": kappa_seg,
        "w_air_lbft": w_air, "w_b_lbft": w_b,
        "mu_slide": mu_s, "mu_rot": mu_r,
        "N_lbf": N_side, "dT_lbf": dT, "T_next_lbf": T[1:],
        "dM_lbf_ft": dM, "M_next_lbf_ft": M[1:], "cased?": cased_seg,
        "comp": comp_arr
    })
    return df, T, M

# ───────────────────── Helper diagnostics ────────────────────────
def neutral_point_md(md: np.ndarray, T_arr: np.ndarray) -> float:
    """Return MD where axial force crosses zero (bit→surface array). NaN if none."""
    if len(md) < 2 or len(T_arr) < 2: return float('nan')
    for i in range(len(T_arr)-1):
        t1, t2 = T_arr[i], T_arr[i+1]
        if t1 == 0: return md[i]
        if t1*t2 < 0:
            frac = abs(t1)/(abs(t1)+abs(t2)+1e-9)
            return md[i] + frac*(md[i+1]-md[i])
    return float('nan')

def grid_calibrate_mu(
    md, inc_deg, kappa, cased_mask, comp_along, comp_props, mw_ppg,
    depth_for_fit: float,
    measured_pickup_hl: Optional[float], measured_slackoff_hl: Optional[float],
    measured_rotate_hl: Optional[float], measured_surface_torque: Optional[float],
    mu_ranges: Dict[str, Tuple[float,float,float]],
):
    """Very simple grid search across μ ranges; returns best μ dict or None."""
    targets = []
    if measured_pickup_hl is not None:   targets.append("pickup")
    if measured_slackoff_hl is not None: targets.append("slackoff")
    if measured_rotate_hl is not None:   targets.append("rotate_off")
    if len(targets) == 0: return None

    md = np.asarray(md)
    idx = np.searchsorted(md, depth_for_fit, side="right")
    md_fit = md[:idx+1]; inc_fit = np.asarray(inc_deg)[:idx+1]; kappa_fit = np.asarray(kappa)[:idx+1]
    cased_fit = np.asarray(cased_mask)[:idx]
    comp_fit  = np.asarray(list(comp_along))[:len(md_fit)-1]

    best = None; best_err = 1e99
    mu_c_s_rng = np.arange(*mu_ranges["mu_c_s"])
    mu_o_s_rng = np.arange(*mu_ranges["mu_o_s"])
    mu_c_r_rng = np.arange(*mu_ranges["mu_c_r"])
    mu_o_r_rng = np.arange(*mu_ranges["mu_o_r"])

    for mu_c_s in mu_c_s_rng:
        for mu_o_s in mu_o_s_rng:
            for mu_c_r in mu_c_r_rng:
                for mu_o_r in mu_o_r_rng:
                    err2 = 0.0
                    for scen in targets:
                        df_tmp, T_tmp, M_tmp = soft_string_stepper(
                            md_fit, inc_fit, kappa_fit, cased_fit, comp_fit, comp_props,
                            mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
                            scenario=scen, WOB_lbf=0.0, Mbit_ftlbf=0.0
                        )
                        HL = abs(T_tmp[-1])  # magnitude of surface tension
                        if scen == "pickup":   err2 += (HL - measured_pickup_hl)**2
                        if scen == "slackoff": err2 += (HL - measured_slackoff_hl)**2
                        if scen == "rotate_off" and measured_rotate_hl is not None:
                            err2 += (HL - measured_rotate_hl)**2
                        if measured_surface_torque is not None:
                            err2 += (abs(M_tmp[-1]) - measured_surface_torque)**2
                    if err
