# streamlit_app.py
# -*- coding: utf-8 -*-
"""
PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
Single-tab linked workflow with calibration, NP, 0.8×MU, BSR/SR, tortuosity, motor-BT,
rig-limit margins, hole-cleaning booster, and calibrated μ overlay (dashed).

Libraries needed: streamlit, plotly, numpy, pandas
"""

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

def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def bf_from_mw(mw_ppg: float) -> float: return (65.5 - mw_ppg)/65.5  # steel ~65.5 ppg
def I_in4(od_in: float, id_in: float) -> float: return (math.pi/64.0)*(od_in**4 - id_in**4)
def Z_in3(od_in: float, id_in: float) -> float: return I_in4(od_in, id_in) / (od_in/2.0)

# NEW: extra section properties
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

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)   # axial tension, torque
    dT = np.zeros(nseg);  dM = np.zeros(nseg);   N_side = np.zeros(nseg)

    # boundary condition
    if scenario == "onbottom":
        T[0] = -float(WOB_lbf)       # compressive at bit
        M[0] = float(Mbit_ftlbf)     # motor / bit torque allowed

    sgn_ax = {"pickup": +1.0, "slackoff": -1.0}.get(scenario, 0.0)

    for i in range(nseg):
        # side-force cannot be negative
        N_raw = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        N_side[i] = max(0.0, N_raw)

        # axial
        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*N_side[i])*ds
        # torque
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
                        HL = max(0.0, -T_tmp[-1])
                        if scen == "pickup":   err2 += (HL - measured_pickup_hl)**2
                        if scen == "slackoff": err2 += (HL - measured_slackoff_hl)**2
                        if scen == "rotate_off" and measured_rotate_hl is not None:
                            err2 += (HL - measured_rotate_hl)**2
                        if measured_surface_torque is not None:
                            err2 += (abs(M_tmp[-1]) - measured_surface_torque)**2
                    if err2 < best_err:
                        best_err = err2
                        best = dict(mu_c_s=mu_c_s, mu_o_s=mu_o_s, mu_c_r=mu_c_r, mu_o_r=mu_o_r, SSE=best_err)
    return best

# ─────────────────────────────── UI ──────────────────────────────
(tab,) = st.tabs(["Wellpath + Torque & Drag (linked)"])

with tab:
    st.header("Wellpath + Torque & Drag (Δs = 1 ft) — Linked")

    # ───────── Trajectory
    st.subheader("Trajectory & 3D schematic (Minimum Curvature)")
    c1, c2, c3, c4 = st.columns(4)
    profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
    kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
    build   = c3.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.0, 0.1)
    az_deg  = c4.number_input("Azimuth (deg, clockwise from North)", 0.0, 360.0, 0.0, 1.0)

    r1, r2, r3 = st.columns(3)
    if profile == "Build & Hold":
        theta_hold = r1.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 0.5)
        target_md  = r2.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_build_hold(kop_md, build, theta_hold, target_md, az_deg)
    elif profile == "Build–Hold–Drop":
        theta_hold = r1.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 0.5)
        drop_rate  = r2.number_input("Drop rate (deg/100 ft)", 0.0, 30.0, 2.0, 0.1)
        target_md  = r3.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_build_hold_drop(kop_md, build, theta_hold, drop_rate, target_md, az_deg)
    else:
        lateral    = r1.number_input("Lateral length (ft)", 0.0, 30000.0, 2000.0, 100.0)
        target_md  = r2.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_horizontal(kop_md, build, lateral, target_md, az_deg)

    N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

    # ───────── Casing / OH
    st.subheader("Casing / Open-hole (simple, last string + open hole)")
    md_end = float(md[-1])
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    nominal = cc1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), index=1)  # default 9-5/8
    weight  = cc2.selectbox("lb/ft (standards only)", list(CASING_DB[nominal]["weights"].keys()))
    casing_id_in = float(CASING_DB[nominal]["weights"][weight])
    cc3.text_input("Casing ID (in, locked)", f"{casing_id_in:.3f}", disabled=True)

    if 'shoe_md' not in st.session_state:
        st.session_state['shoe_md'] = min(3000.0, md_end)
    st.session_state['shoe_md'] = clamp(float(st.session_state['shoe_md']), 0.0, md_end)
    st.session_state['shoe_md'] = cc4.slider("Shoe MD (ft)", 0.0, md_end, float(st.session_state['shoe_md']), 50.0)

    hole_diam_in = cc5.number_input("Open-hole diameter (in)", 4.0, 20.0, 8.50, 0.01)
    shoe_md = float(st.session_state['shoe_md'])
    cased_mask = md <= shoe_md

    # ───────── 3D split by shoe
    idx = int(np.searchsorted(md, shoe_md, side='right'))
    fig3d = go.Figure()
    if idx > 1:
        fig3d.add_trace(go.Scatter3d(x=E[:idx], y=N[:idx], z=TVD[:idx],
                                     mode="lines", line=dict(width=6, color="#4cc9f0"), name="Cased"))
    if idx < len(md):
        fig3d.add_trace(go.Scatter3d(x=E[idx-1:], y=N[idx-1:], z=TVD[idx-1:],
                                     mode="lines", line=dict(width=4, color="#a97142"), name="Open-hole"))
    fig3d.update_layout(height=420, scene=dict(
        xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)",
        zaxis=dict(autorange="reversed")
    ), legend=dict(orientation="h"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3d, use_container_width=True)

    # 2D TVD-VS
    st.subheader("2D Wellbore Profile — TVD vs Vertical Section")
    vs_ref = st.number_input("VS reference azimuth (deg)", 0.0, 360.0, float(az[0] if len(az) else az_deg), 1.0)
    VS = N*np.cos(vs_ref*DEG2RAD) + E*np.sin(vs_ref*DEG2RAD)
    fig2d = go.Figure()
    if idx > 1: fig2d.add_trace(go.Scatter(x=VS[:idx], y=TVD[:idx], mode="lines", line=dict(width=6, color="#4cc9f0"), name="Cased"))
    if idx < len(md): fig2d.add_trace(go.Scatter(x=VS[idx-1:], y=TVD[idx-1:], mode="lines", line=dict(width=4, color="#a97142"), name="Open-hole"))
    fig2d.update_layout(xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)",
                        yaxis=dict(autorange="reversed"), height=360, legend=dict(orientation="h"),
                        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2d, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "MD (ft)": md[:11], "Inc (deg)": inc_deg[:11], "Az (deg)": az[:11],
        "TVD (ft)": TVD[:11], "North (ft)": N[:11], "East (ft)": E[:11],
        "VS (ft)": VS[:11], "DLS (deg/100 ft)": DLS[:11]
    }), use_container_width=True)

    # ───────── T&D Inputs
    st.subheader("Soft-string Torque & Drag — Johancsik (linked to survey above)")
    with st.expander("Typical μ starting ranges (lecture hints)"):
        st.markdown("""
- **Casing (WBM):** 0.15 – 0.25 (sliding/rotating similar)  
- **Open-hole (WBM):** 0.25 – 0.40 (can be higher when cleaning is poor)  
- **OBM/SBM:** often lower than WBM  
> Use **history matching** to calibrate for your well/section.
""")

    mu_cased_slide = st.number_input("μ in casing (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_open_slide  = st.number_input("μ in open-hole (sliding)", 0.05, 0.80, 0.35, 0.01)
    mu_cased_rot   = st.number_input("μ in casing (rotating)", 0.05, 0.80, 0.25, 0.01)
    mu_open_rot    = st.number_input("μ in open-hole (rotating)", 0.05, 0.80, 0.35, 0.01)
    mw_ppg         = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    hc1, hc2 = st.columns(2)
    cleaning_mode = hc1.selectbox("Hole-cleaning condition (open-hole μ booster)", ["Good (0)", "Fair (+0.03)", "Poor (+0.08)"], index=0)
    mu_boost = {"Good (0)":0.0, "Fair (+0.03)":0.03, "Poor (+0.08)":0.08}[cleaning_mode]
    tort_mode = hc2.selectbox("Tortuosity penalty mode", ["off", "kappa", "mu"], index=0)
    tau = st.slider("Penalty factor τ (for open-hole only)", 0.0, 0.5, 0.2, 0.05) if tort_mode != "off" else 0.0

    st.markdown("**Drillstring (bit up)**")
    d1, d2, d3 = st.columns(3)
    dc_len  = d1.number_input("DC length (ft)", 0.0, 10000.0, 600.0, 10.0)
    hwdp_len= d2.number_input("HWDP length (ft)", 0.0, 20000.0, 1000.0, 10.0)
    dp_len  = d3.number_input("DP length (ft)", 0.0, 50000.0, max(0.0, float(md[-1])-(dc_len+hwdp_len)), 10.0)

    e1, e2, e3 = st.columns(3)
    dc_od = e1.number_input("DC OD (in)", 3.0, 12.0, 8.0, 0.1)
    hwdp_od = e2.number_input("HWDP OD (in)", 2.0, 8.0, 3.5, 0.1)
    dp_od = e3.number_input("DP OD (in)", 3.0, 6.625, 5.0, 0.01)

    i1, i2, i3 = st.columns(3)
    dc_id = i1.number_input("DC ID (in)", 0.5, 6.0, 2.81, 0.01)
    hwdp_id = i2.number_input("HWDP ID (in)", 0.5, 4.0, 2.0, 0.01)
    dp_id = i3.number_input("DP ID (in)", 1.5, 5.0, 4.28, 0.01)

    w1, w2, w3 = st.columns(3)
    dc_w   = w1.number_input("DC weight (air, lb/ft)",   30.0, 150.0, 66.7, 0.1)
    hwdp_w = w2.number_input("HWDP weight (air, lb/ft)",  5.0,  40.0, 16.0, 0.1)
    dp_w   = w3.number_input("DP weight (air, lb/ft)",    8.0,  40.0, 19.5, 0.1)

    # map string along depth
    nseg = len(md) - 1
    comp_along = np.empty(nseg, dtype=object)
    for i in range(nseg):
        from_bit = float(md[-1]) - md[i]
        if from_bit <= dc_len: comp_along[i] = "DC"
        elif from_bit <= dc_len + hwdp_len: comp_along[i] = "HWDP"
        else: comp_along[i] = "DP"

    comp_props = {
        "DC":   {"od_in": dc_od,   "id_in": dc_id,   "w_air_lbft": dc_w},
        "HWDP": {"od_in": hwdp_od, "id_in": hwdp_id, "w_air_lbft": hwdp_w},
        "DP":   {"od_in": dp_od,   "id_in": dp_id,   "w_air_lbft": dp_w},
    }

    # NEW: per-segment geometry arrays for severity calcs
    comp_map = {"DC": (dc_od, dc_id), "HWDP": (hwdp_od, hwdp_id), "DP": (dp_od, dp_id)}
    comp_od_in = np.array([comp_map[c][0] for c in comp_along])
    comp_id_in = np.array([comp_map[c][1] for c in comp_along])

    # curvature per-ft
    _, _, _, DLS = mincurv_positions(md, inc_deg, az)
    kappa = (DLS*DEG2RAD)/100.0

    # scenario + bit torque
    scen = st.selectbox("Scenario", ["Slack-off (RIH)","Pickup (POOH)","Rotate off-bottom","Rotate on-bottom"])
    scenario = {"Slack-off (RIH)":"slackoff","Pickup (POOH)":"pickup","Rotate off-bottom":"rotate_off","Rotate on-bottom":"onbottom"}[scen]
    wob  = st.number_input("WOB (lbf) for on-bottom", 0.0, 150000.0, 6000.0, 100.0)
    mcol1, mcol2 = st.columns(2)
    motor_mode = mcol1.checkbox("Motor on-bottom (bit torque from ΔP)", value=False)
    K_tbit = mcol2.number_input("Motor torque factor K (lbf-ft/psi)", 0.0, 10_000.0, 2.5, 0.1) if motor_mode else 0.0
    deltaP = st.number_input("Motor ΔP (psi)", 0.0, 5000.0, 0.0, 10.0) if motor_mode else 0.0
    Mbit = K_tbit * deltaP if motor_mode and scenario == "onbottom" else 0.0

    # main run
    df_itr, T_arr, M_arr = soft_string_stepper(
