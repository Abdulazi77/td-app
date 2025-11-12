# streamlit_app.py
# -*- coding: utf-8 -*-
"""
PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
Linked, single-tab workflow with survey import, calibration, μ-sweep risk curves,
0.8×MU gate, TJ combined-load envelope, buckling indicators, NP check, motor torque,
and rig-limit margins. Rotating off-bottom HL = average(PU, SO) per standard practice.

Libraries: streamlit, plotly, numpy, pandas
"""

from __future__ import annotations
import io
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

# Minimal dimensional DBs (extend as needed)
CASING_DB = {
    "13-3/8": {"weights": {48.0: 12.415, 54.5: 12.347, 61.0: 12.107}},
    "9-5/8":  {"weights": {29.3: 8.921, 36.0: 8.535, 40.0: 8.321}},
    "7":      {"weights": {20.0: 6.366, 23.0: 6.059, 26.0: 5.920}},
    "5-1/2":  {"weights": {17.0: 4.778, 20.0: 4.670, 23.0: 4.560}},
}
TOOL_JOINT_DB = {
    # name : od(in), id(in), make-up torque(ft-lbf), tension rating(lbf), yield torque(ft-lbf) (approx)
    "NC31": {"od": 4.06, "id": 2.25, "T_makeup_ftlbf":  9000, "F_tensile_lbf": 260000, "T_yield_ftlbf": 15000},
    "NC35": {"od": 4.50, "id": 2.50, "T_makeup_ftlbf": 11000, "F_tensile_lbf": 300000, "T_yield_ftlbf": 18000},
    "NC38": {"od": 4.75, "id": 2.25, "T_makeup_ftlbf": 12000, "F_tensile_lbf": 350000, "T_yield_ftlbf": 20000},
    "NC40": {"od": 5.00, "id": 2.25, "T_makeup_ftlbf": 16000, "F_tensile_lbf": 420000, "T_yield_ftlbf": 25000},
    "NC46": {"od": 5.75, "id": 2.75, "T_makeup_ftlbf": 22000, "F_tensile_lbf": 530000, "T_yield_ftlbf": 35000},
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
    # simple drop start (change if exact geometry is needed)
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
    md = np.asarray(md); inc_deg = np.asarray(inc_deg); az_deg = np.asarray(az_deg)
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
    tortuosity_mode: str = "off",  # "off" | "kappa" | "mu"
    tau: float = 0.0,              # 0.0 .. 0.5 typ
    mu_open_boost: float = 0.0,    # hole cleaning booster
    include_slide_friction: bool = True,  # set False for rotating HL/torque
):
    """
    Δs = 1 ft soft-string integration, bit -> surface.

    Key behaviours (small but important fixes):
    - 'rotate_off' (off-bottom rotation) should ignore axial weight transport and sliding friction.
      We do this by setting sgn_weight=0 and include_slide_friction=False.
    - For torque, we always sum μ_rot * N * r_eff along the string.
    """
    ds = 1.0
    md = np.asarray(md); inc_full = np.asarray(inc_deg)
    nseg = len(md) - 1
    if nseg <= 0: raise ValueError("Trajectory is empty.")
    inc = np.deg2rad(inc_full[:-1])

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
        r_eff_ft[i] = 0.5*od_in*IN2FT  # pipe radius as lever arm
        if cased_seg[i]:
            mu_s[i] = mu_slide_cased; mu_r[i] = mu_rot_cased
        else:
            mu_s[i] = mu_slide_open + mu_open_boost
            mu_r[i] = mu_rot_open   + mu_open_boost

        # tortuosity penalties (open-hole only)
        if not cased_seg[i]:
            if tortuosity_mode == "kappa":
                kappa_seg[i] *= (1.0 + tau)
            elif tortuosity_mode == "mu":
                mu_s[i] *= (1.0 + tau); mu_r[i] *= (1.0 + tau)

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)   # axial tension, torque
    dT = np.zeros(nseg);  dM = np.zeros(nseg);   N_side = np.zeros(nseg)

    # boundary conditions
    if scenario == "onbottom":
        T[0] = -float(WOB_lbf)       # compressive at bit
        M[0] = float(Mbit_ftlbf)     # motor / bit torque allowed
    else:
        T[0] = 0.0
        M[0] = 0.0

    # axial weight transport
    if scenario == "pickup":
        sgn_weight = +1.0
    elif scenario == "slackoff":
        sgn_weight = -1.0
    elif scenario == "rotate_off":
        # >>> FIX: for off-bottom rotation, ignore axial weight transport entirely
        sgn_weight = 0.0
        include_slide_friction = False
    else:  # onbottom
        sgn_weight = -1.0

    for i in range(nseg):
        # side force (Johancsik): buoyed weight projection + curvature drag (T * kappa)
        N_side[i] = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]

        # axial increment
        dT_weight = sgn_weight*w_b[i]*math.cos(inc[i])
        dT_fric   = (mu_s[i]*N_side[i]) if include_slide_friction else 0.0
        T_next = T[i] + (dT_weight + dT_fric)*ds

        # torque increment (elemental torque per ft)
        dM_i = (mu_r[i]*N_side[i]*r_eff_ft[i])*ds
        M_next = M[i] + dM_i

        dT[i] = T_next - T[i]
        dM[i] = dM_i
        T[i+1] = T_next
        M[i+1] = M_next

    df = pd.DataFrame({
        "md_top_ft": md[:-1], "md_bot_ft": md[1:], "ds_ft": 1.0,
        "inc_deg": inc_full[:-1], "kappa_rad_ft": kappa_seg,
        "w_air_lbft": w_air, "w_b_lbft": w_b,
        "mu_slide": mu_s, "mu_rot": mu_r,
        "N_lbf": N_side,
        "dT_lbf": dT, "T_next_lbf": T[1:],
        "dM_lbf_ft": dM, "M_next_lbf_ft": M[1:],  # cumulative M and elemental dM saved
        "cased?": cased_seg, "comp": comp_arr
    })
    return df, T, M

# ───────────────────── Helper diagnostics ────────────────────────
def neutral_point_md(md: np.ndarray, T_profile_bit_to_surface: np.ndarray) -> float:
    """
    Improved NP finder:
    - Use on-bottom axial profile (bit->surface).
    - Look for sign change from compression (−) to tension (+) and linearly interpolate.
    """
    md = np.asarray(md); T = np.asarray(T_profile_bit_to_surface)
    if len(md) != len(T):  # allow T to be len(md); if not, adjust
        # if T is segment endpoints (nseg+1) and md has npts, they should match
        n = min(len(md), len(T))
        md = md[:n]; T = T[:n]
    for i in range(len(T)-1):
        t1, t2 = T[i], T[i+1]
        if t1 == 0.0:
            return float(md[i])
        if t1 * t2 < 0.0:
            # linear interpolation
            frac = abs(t1) / (abs(t1) + abs(t2) + 1e-12)
            return float(md[i] + frac * (md[i+1] - md[i]))
    return float('nan')

def grid_calibrate_mu(
    md, inc_deg, kappa, cased_mask, comp_along, comp_props, mw_ppg,
    depth_for_fit: float,
    measured_pickup_hl: Optional[float], measured_slackoff_hl: Optional[float],
    measured_rotate_hl: Optional[float], measured_surface_torque: Optional[float],
    mu_ranges: Dict[str, Tuple[float,float,float]],
):
    """Grid search across μ ranges; returns best μ dict or None."""
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
                    # pickup & slackoff (HL)
                    _, T_pick, _ = soft_string_stepper(
                        md_fit, inc_fit, kappa_fit, cased_fit, comp_fit, comp_props,
                        mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
                        scenario="pickup", include_slide_friction=True
                    )
                    _, T_so, _ = soft_string_stepper(
                        md_fit, inc_fit, kappa_fit, cased_fit, comp_fit, comp_props,
                        mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
                        scenario="slackoff", include_slide_friction=True
                    )
                    HL_pick = abs(T_pick[-1]); HL_so = abs(T_so[-1]); HL_rot = 0.5*(HL_pick + HL_so)
                    if measured_pickup_hl is not None:   err2 += (HL_pick - measured_pickup_hl)**2
                    if measured_slackoff_hl is not None: err2 += (HL_so   - measured_slackoff_hl)**2
                    if measured_rotate_hl is not None:   err2 += (HL_rot  - measured_rotate_hl)**2

                    # torque (off-bottom)
                    if measured_surface_torque is not None:
                        _, _, M_rot = soft_string_stepper(
                            md_fit, inc_fit, kappa_fit, cased_fit, comp_fit, comp_props,
                            mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
                            scenario="rotate_off", include_slide_friction=False
                        )
                        err2 += (abs(M_rot[-1]) - measured_surface_torque)**2

                    if err2 < best_err:
                        best_err = err2
                        best = dict(mu_c_s=mu_c_s, mu_o_s=mu_o_s, mu_c_r=mu_c_r, mu_o_r=mu_o_r, SSE=best_err)
    return best

# ─────────────────────────────── UI ──────────────────────────────
(tab,) = st.tabs(["Wellpath + Torque & Drag (linked)"])

with tab:
    st.header("Wellpath + Torque & Drag (Δs = 1 ft) — Linked")
    st.caption("Minimum-curvature survey; Johancsik soft-string; Δs=1 ft piecewise integration. "
               "Rotating off-bottom hookload = average of pickup & slack-off; "
               "off-bottom torque is computed with pure rotation (no axial weight transport).")

    # ───────── Survey: import or synthesize
    st.subheader("Trajectory — import survey (CSV) or synthesize (Minimum Curvature)")
    up = st.file_uploader("Upload survey CSV with columns: MD (ft), INC (deg), AZI (deg). Leave empty to synthesize.", type=["csv"])
    if up is not None:
        try:
            df_svy = pd.read_csv(up)
            if not set(["MD","INC","AZI"]).issubset(df_svy.columns.str.upper()):
                cols = {c.upper(): c for c in df_svy.columns}
                md = df_svy[cols["MD"]].to_numpy(dtype=float)
                inc_deg = df_svy[cols["INC"]].to_numpy(dtype=float)
                az_deg  = df_svy[cols["AZI"]].to_numpy(dtype=float)
            else:
                md     = df_svy[[c for c in df_svy.columns if c.upper()=="MD"][0]].to_numpy(dtype=float)
                inc_deg= df_svy[[c for c in df_svy.columns if c.upper()=="INC"][0]].to_numpy(dtype=float)
                az_deg = df_svy[[c for c in df_svy.columns if c.upper()=="AZI"][0]].to_numpy(dtype=float)
            # resample to Δs=1 ft
            md_grid = np.arange(float(md.min()), float(md.max())+1.0, 1.0)
            inc_grid = np.interp(md_grid, md, inc_deg)
            az_grid  = np.interp(md_grid, md, az_deg)
            md, inc_deg, az = md_grid, inc_grid, az_grid
            st.success(f"Survey loaded: {len(md)} points at 1-ft resolution.")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            md = np.array([]); inc_deg = np.array([]); az = np.array([])
    else:
        c1, c2, c3, c4 = st.columns(4)
        profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
        kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
        build   = c3.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.0, 0.1)
        az0     = c4.number_input("Azimuth (deg, clockwise from North)", 0.0, 360.0, 0.0, 1.0)

        r1, r2, r3 = st.columns(3)
        if profile == "Build & Hold":
            theta_hold = r1.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 0.5)
            target_md  = r2.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
            md, inc_deg, az = synth_build_hold(kop_md, build, theta_hold, target_md, az0)
        elif profile == "Build–Hold–Drop":
            theta_hold = r1.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 0.5)
            drop_rate  = r2.number_input("Drop rate (deg/100 ft)", 0.0, 30.0, 2.0, 0.1)
            target_md  = r3.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
            md, inc_deg, az = synth_build_hold_drop(kop_md, build, theta_hold, drop_rate, target_md, az0)
        else:
            lateral    = r1.number_input("Lateral length (ft)", 0.0, 30000.0, 2000.0, 100.0)
            target_md  = r2.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
            md, inc_deg, az = synth_horizontal(kop_md, build, lateral, target_md, az0)

    if len(md) < 3:
        st.stop()

    N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

    # ───────── Casing / OH (simple last string)
    st.subheader("Casing / Open-hole (simple: last string + open hole)")
    md_end = float(md[-1])
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    nominal = cc1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), index=1)  # default 9-5/8
    weight  = cc2.selectbox("lb/ft (standards only)", list(CASING_DB[nominal]["weights"].keys()))
    casing_id_in = float(CASING_DB[nominal]["weights"][weight])
    cc3.text_input("Casing ID (in, locked)", f"{casing_id_in:.3f}", disabled=True)
    shoe_md = cc4.slider("Shoe MD (ft)", 0.0, md_end, min(3000.0, md_end), 50.0)
    hole_diam_in = cc5.number_input("Open-hole diameter (in)", 4.0, 20.0, 8.50, 0.01)
    cased_mask = md <= shoe_md
    idx = int(np.searchsorted(md, shoe_md, side='right'))

    # ───────── 3D split by shoe
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
    vs_ref = st.number_input("VS reference azimuth (deg)", 0.0, 360.0, float(az[0] if len(az) else 0.0), 1.0)
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
    st.subheader("Soft-string Torque & Drag — Johancsik (linked)")
    with st.expander("Typical μ starting ranges (lecture hints)"):
        st.markdown("""
- **Casing (WBM):** 0.15 – 0.25 (sliding/rotating similar)  
- **Open-hole (WBM):** 0.25 – 0.40 (higher when cleaning is poor)  
- **OBM/SBM:** often lower than WBM  
> Calibrate with **history matching** for your section.
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
    tau = st.slider("Penalty factor τ (open-hole only)", 0.0, 0.5, 0.2, 0.05) if tort_mode != "off" else 0.0

    st.markdown("**Drillstring (bit up)**")
    d1, d2, d3 = st.columns(3)
    dc_len  = d1.number_input("DC length (ft)",   0.0, 10000.0, 600.0, 10.0)
    hwdp_len= d2.number_input("HWDP length (ft)", 0.0, 20000.0, 1000.0, 10.0)
    dp_len  = d3.number_input("DP length (ft)",   0.0, 50000.0, max(0.0, float(md[-1])-(dc_len+hwdp_len)), 10.0)

    e1, e2, e3 = st.columns(3)
    dc_od   = e1.number_input("DC OD (in)",   3.0, 12.0, 8.0, 0.1)
    hwdp_od = e2.number_input("HWDP OD (in)", 2.0, 8.0, 3.5, 0.1)
    dp_od   = e3.number_input("DP OD (in)",   3.0, 6.625, 5.0, 0.01)

    i1, i2, i3 = st.columns(3)
    dc_id   = i1.number_input("DC ID (in)",   0.5, 6.0, 2.81, 0.01)
    hwdp_id = i2.number_input("HWDP ID (in)", 0.5, 4.0, 2.0, 0.01)
    dp_id   = i3.number_input("DP ID (in)",   1.5, 5.0, 4.28, 0.01)

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

    # main runs (PU/SO for HL; rotate_off for torque)
    df_pick, T_pick, _ = soft_string_stepper(
        md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props,
        mu_cased_slide, mu_open_slide, mu_cased_rot, mu_open_rot,
        mw_ppg, scenario="pickup", WOB_lbf=0.0, Mbit_ftlbf=0.0,
        tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost,
        include_slide_friction=True
    )
    df_so, T_so, _ = soft_string_stepper(
        md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props,
        mu_cased_slide, mu_open_slide, mu_cased_rot, mu_open_rot,
        mw_ppg, scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0,
        tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost,
        include_slide_friction=True
    )
    df_rot_tq, _, M_rot = soft_string_stepper(
        md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props,
        mu_cased_slide, mu_open_slide, mu_cased_rot, mu_open_rot,
        mw_ppg, scenario="rotate_off", WOB_lbf=0.0, Mbit_ftlbf=0.0,
        tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost,
        include_slide_friction=False  # pure rotation torque
    )
    # on-bottom (for NP)
    df_on, T_on, _ = soft_string_stepper(
        md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props,
        mu_cased_slide, mu_open_slide, mu_cased_rot, mu_open_rot,
        mw_ppg, scenario="onbottom", WOB_lbf=wob, Mbit_ftlbf=Mbit,
        tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost,
        include_slide_friction=True
    )

    depth = df_pick["md_bot_ft"].to_numpy()
    HL_pick = abs(T_pick[-1]); HL_so = abs(T_so[-1])
    HL_rot  = 0.5*(HL_pick + HL_so)
    surf_torque = abs(M_rot[-1])

    st.success(f"Surface HL — Pickup: {HL_pick:,.0f} lbf | Slack-off: {HL_so:,.0f} lbf | Rotating (avg): {HL_rot:,.0f} lbf")
    st.info(f"Surface torque (rotating off-bottom): {surf_torque:,.0f} lbf-ft")

    # ───────── Safety & limits
    st.subheader("Safety & limits")
    s1, s2, s3 = st.columns(3)
    tj_name = s1.selectbox("Tool-joint size", list(TOOL_JOINT_DB.keys()), index=2)  # default NC38
    sf_joint = s2.number_input("Safety factor (tool-joint)", 1.00, 2.00, 1.10, 0.05)
    sf_tension = s3.number_input("SF for pipe body tension", 1.00, 2.00, 1.15, 0.05)

    rig_torque_lim = st.number_input("Top-drive torque limit (lbf-ft)", 10000, 150000, 60000, 1000)
    rig_pull_lim   = st.number_input("Rig max hookload (lbf)", 50000, 1500000, 500000, 5000)

    # 0.8×MU rule and margins
    T_makeup = TOOL_JOINT_DB[tj_name]['T_makeup_ftlbf']
    T80 = 0.8*T_makeup
    torque_margin = T80 - surf_torque
    st.info(f"0.8×Make-up = {T80/1000:.1f} k lbf-ft — Surface torque = {surf_torque/1000:.2f} k → "
            f"{'PASS ✅' if torque_margin >= 0 else 'FAIL ❌'} (margin {torque_margin/1000:.2f} k)")

    # Neutral point from on-bottom run (robust)
    np_depth = neutral_point_md(md, np.array(T_on))
    if math.isnan(np_depth):
        st.warning("Neutral point not found (no sign change in axial on-bottom).")
    else:
        NP_from_bit = md[-1] - np_depth
        in_DCs = NP_from_bit <= dc_len
        st.write(f"Neutral point from bit ≈ **{NP_from_bit:,.0f} ft** → {'✅ inside DCs' if in_DCs else '❌ not inside DCs'}")
        if not in_DCs:
            st.warning("Increase DC length so NP sits inside the collars for current WOB.")

    # BSR & SR quick checks
    st.subheader("Connection checks — BSR & SR (rule-of-thumb)")
    Z_dp, Z_h, Z_dc = Z_in3(dp_od, dp_id), Z_in3(hwdp_od, hwdp_id), Z_in3(dc_od, dc_id)
    I_dp, I_h, I_dc = I_in4(dp_od, dp_id), I_in4(hwdp_od, hwdp_id), I_in4(dc_od, dc_id)
    BSR_dp_h  = Z_h / Z_dp if Z_dp>0 else float('inf')
    BSR_h_dc  = Z_dc/ Z_h if Z_h>0 else float('inf')
    SR_dp_h   = I_h / I_dp if I_dp>0 else float('inf')
    SR_h_dc   = I_dc/ I_h if I_h>0 else float('inf')
    c1m, c2m, c3m, c4m = st.columns(4)
    c1m.metric("BSR DP→HWDP", f"{BSR_dp_h:.2f}", help="Bending Strength Ratio. Flag if <~1.0–1.1")
    c2m.metric("BSR HWDP→DC", f"{BSR_h_dc:.2f}")
    c3m.metric("SR DP→HWDP", f"{SR_dp_h:.2f}", help="Stiffness Ratio (I ratio). Big jumps (>~1.4) raise twist-off risk.")
    c4m.metric("SR HWDP→DC", f"{SR_h_dc:.2f}")

    # ───────── Simple/Advanced charting
    simple_mode = st.checkbox("Use classic simple view (hide safety overlays & μ-sweep)", value=False)

    # overlay toggle for calibrated μ curves
    overlay_calibrated = False
    mu_fit = st.session_state.get("μ_fit")
    if mu_fit is not None and not simple_mode:
        overlay_calibrated = st.checkbox("Overlay calibrated μ curves (dashed)", value=True)
        if overlay_calibrated:
            st.caption(
                f"Overlay uses fitted μ: casing(slide)={mu_fit['mu_c_s']:.2f}, open(slide)={mu_fit['mu_o_s']:.2f}, "
                f"casing(rot)={mu_fit['mu_c_r']:.2f}, open(rot)={mu_fit['mu_o_r']:.2f}"
            )

    if simple_mode:
        # Torque vs Depth (rotating, cumulative)
        figT = go.Figure(go.Scatter(x=np.abs(df_rot_tq["M_next_lbf_ft"].to_numpy()),
                                    y=df_rot_tq["md_bot_ft"], mode="lines", name="Torque (cum)"))
        figT.update_xaxes(title_text="Torque (lbf-ft)")
        figT.update_yaxes(title_text="Depth (ft)", autorange="reversed")

        # Hookload magnitude vs depth (PU, SO, Rot avg)
        figH = go.Figure()
        figH.add_trace(go.Scatter(x=np.abs(df_pick["T_next_lbf"].to_numpy()), y=df_pick["md_bot_ft"], mode="lines", name="Pickup"))
        figH.add_trace(go.Scatter(x=np.abs(df_so["T_next_lbf"].to_numpy()),   y=df_so["md_bot_ft"],   mode="lines", name="Slack-off"))
        figH.add_trace(go.Scatter(x=0.5*(np.abs(df_pick["T_next_lbf"].to_numpy())+np.abs(df_so["T_next_lbf"].to_numpy())),
                                  y=df_so["md_bot_ft"],   mode="lines", name="Rotating (avg)", line=dict(dash="dot")))
        figH.update_xaxes(title_text="Hookload (lbf)")
        figH.update_yaxes(title_text="Depth (ft)", autorange="reversed")

        c1x, c2x = st.columns(2)
        with c1x: st.plotly_chart(figT, use_container_width=True)
        with c2x: st.plotly_chart(figH, use_container_width=True)
    else:
        st.markdown("### T&D Model Charts — Risk curves and limits")

        mu_band = st.multiselect("μ sweep for off-bottom risk curves (rotating torque)", [0.15,0.20,0.25,0.30,0.35,0.40], default=[0.20,0.25,0.30,0.35])

        T_makeup_sf = TOOL_JOINT_DB[tj_name]['T_makeup_ftlbf']/sf_joint
        T_yield_sf  = TOOL_JOINT_DB[tj_name]['T_yield_ftlbf']/sf_joint
        F_tensile_sf= TOOL_JOINT_DB[tj_name]['F_tensile_lbf']/sf_tension

        def run_td_off_bottom(mu_slide: float, mu_rot: float):
            # pure rotation (no axial weight transport, no sliding friction)
            df_tmp, _, M_tmp = soft_string_stepper(
                md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props,
                mu_slide, mu_slide, mu_rot, mu_rot, mw_ppg,
                scenario="rotate_off",
                tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost,
                include_slide_friction=False
            )
            return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy()), np.abs(df_tmp["dM_lbf_ft"].to_numpy())

        # LEFT: μ-sweep off-bottom torque vs depth (cumulative)
        fig_left = go.Figure()
        for mu in mu_band:
            dmu, Mcum, _ = run_td_off_bottom(mu, mu)
            fig_left.add_trace(go.Scatter(x=Mcum/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"))
        fig_left.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash",
                           annotation_text="Make-up/SF", annotation_position="top right")
        fig_left.add_vline(x=(0.8*TOOL_JOINT_DB[tj_name]['T_makeup_ftlbf'])/1000.0, line_color="#00d5ff", line_dash="dot",
                           annotation_text="0.8×MU", annotation_position="bottom right")
        fig_left.update_yaxes(autorange="reversed", title_text="Depth (ft)")
        fig_left.update_xaxes(title_text="Off-bottom torque (k lbf-ft)")
        fig_left.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

        # RIGHT: **Elemental torque** (per-ft dM) + limits (this matches typical “U-friction topside high” look)
        fig_right = go.Figure()
        for mu in mu_band:
            dmu, _, dM_pf = run_td_off_bottom(mu, mu)
            fig_right.add_trace(go.Scatter(x=dM_pf/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"))
        fig_right.add_vline(x=T_makeup_sf/1000.0,  line_color="#00d5ff", line_dash="dash", annotation_text="Make-up/SF")
        fig_right.add_vline(x=rig_torque_lim/1000.0, line_color="magenta", line_dash="dot", annotation_text="Top-drive limit")
        fig_right.update_yaxes(autorange="reversed", title_text="Depth (ft)")
        fig_right.update_xaxes(title_text="Elemental torque dM (k lbf-ft/ft)")
        fig_right.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

        cL, cR = st.columns(2)
        with cL: st.plotly_chart(fig_left, use_container_width=True)
        with cR: st.plotly_chart(fig_right, use_container_width=True)

        # Envelope & Hookload diagnostics (buckling plots & rig limit)
        Epsi = 30.0e6
        Iin4_dp = I_in4(dp_od, dp_id)
        rbore_ft = 0.5*hole_diam_in*IN2FT; rpipe_ft = 0.5*dp_od*IN2FT
        clearance_ft = max(1e-3, rbore_ft - rpipe_ft)
        theta = np.deg2rad(np.maximum(0.0, df_pick['inc_deg'].to_numpy()))
        Fs = 2.0*np.sqrt(Epsi*Iin4_dp * df_pick['w_b_lbft'].to_numpy()*np.sin(theta)/clearance_ft)
        Fh = 1.6*Fs  # indicator

        fig_env = go.Figure()
        F_env = np.linspace(0, TOOL_JOINT_DB[tj_name]['F_tensile_lbf']/sf_tension, 100)
        T_env = (TOOL_JOINT_DB[tj_name]['T_yield_ftlbf']/sf_joint)*np.sqrt(np.clip(1.0 - (F_env/np.maximum(F_env.max(),1.0))**2, 0.0, 1.0))
        fig_env.add_trace(go.Scatter(x=T_env/1000.0, y=F_env/1000.0, mode="lines", name="API-style envelope (approx)"))
        fig_env.add_vline(x=0.8*T_makeup/1000.0, line_color="#00d5ff", line_dash="dash", annotation_text="0.8×MU")
        fig_env.add_trace(go.Scatter(x=[surf_torque/1000.0], y=[HL_rot/1000.0], mode="markers",
                                     name="Operating point (rotating)", marker=dict(size=10, color="orange")))
        fig_env.update_xaxes(title_text="Torque (k lbf-ft)")
        fig_env.update_yaxes(title_text="Tension (k lbf)")
        fig_env.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))

        fig_hl = go.Figure()
        # Hookloads vs depth
        fig_hl.add_trace(go.Scatter(x=np.abs(df_pick['T_next_lbf'].to_numpy())/1000.0, y=depth,
                                    mode="lines", name="HL Pickup"))
        fig_hl.add_trace(go.Scatter(x=np.abs(df_so['T_next_lbf'].to_numpy())/1000.0, y=depth,
                                    mode="lines", name="HL Slack-off"))
        fig_hl.add_trace(go.Scatter(x=(0.5*(np.abs(df_pick['T_next_lbf'].to_numpy())+np.abs(df_so['T_next_lbf'].to_numpy())))/1000.0, y=depth,
                                    mode="lines", name="HL Rotating (avg)", line=dict(dash="dot")))
        fig_hl.add_vline(x=rig_pull_lim/1000.0, line_color="magenta", line_dash="dot", annotation_text="Rig pull limit")
        # Buckling indicators
        fig_hl.add_trace(go.Scatter(x=Fs/1000.0, y=depth, name="Sinusoidal Fs (indic.)", line=dict(dash="dash")))
        fig_hl.add_trace(go.Scatter(x=Fh/1000.0, y=depth, name="Helical Fh (indic.)", line=dict(dash="dot")))
        fig_hl.update_yaxes(autorange="reversed", title_text="Depth (ft)")
        fig_hl.update_xaxes(title_text="Force / Hookload (k lbf)")
        fig_hl.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(fig_env, use_container_width=True)
        with c4: st.plotly_chart(fig_hl,  use_container_width=True)

    # ───────── Friction calibration (history match)
    st.subheader("Friction calibration (history match at a depth)")
    cal1, cal2, cal3, cal4 = st.columns(4)
    fit_depth = cal1.number_input("Depth to fit (MD ft)", 0.0, float(md[-1]), float(min(4000.0, md[-1])), 50.0)
    meas_pick = cal2.number_input("Measured Pickup HL (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
    meas_slack= cal3.number_input("Measured Slack-off HL (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
    meas_rot  = cal4.number_input("Measured Rotating off-bottom HL (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
    meas_torque = st.number_input("Measured surface torque (lbf-ft, optional)", 0.0, 200_000.0, 0.0, 100.0)

    with st.expander("Calibration search ranges (μ)"):
        cA, cB, cC, cD = st.columns(4)
        mu_c_s_rng = cA.text_input("μ_casing_slide: start,stop,step", "0.15,0.35,0.05")
        mu_o_s_rng = cB.text_input("μ_open_slide: start,stop,step", "0.25,0.45,0.05")
        mu_c_r_rng = cC.text_input("μ_casing_rot: start,stop,step", "0.15,0.35,0.05")
        mu_o_r_rng = cD.text_input("μ_open_rot: start,stop,step", "0.20,0.40,0.05")

    if st.button("Run μ calibration (grid search)"):
        try:
            parse = lambda s: tuple(float(x.strip()) for x in s.split(","))
            mu_ranges = {
                "mu_c_s": parse(mu_c_s_rng), "mu_o_s": parse(mu_o_s_rng),
                "mu_c_r": parse(mu_c_r_rng), "mu_o_r": parse(mu_o_r_rng),
            }
            best = grid_calibrate_mu(
                md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props, mw_ppg,
                depth_for_fit=fit_depth,
                measured_pickup_hl=meas_pick if meas_pick>0 else None,
                measured_slackoff_hl=meas_slack if meas_slack>0 else None,
                measured_rotate_hl=meas_rot if meas_rot>0 else None,
                measured_surface_torque=meas_torque if meas_torque>0 else None,
                mu_ranges=mu_ranges
            )
            if best is None:
                st.warning("Provide at least one measurement (pickup/slackoff/rotate).")
            else:
                st.session_state["μ_fit"] = best
                st.success(f"Fitted μ: casing(slide)={best['mu_c_s']:.2f}, open(slide)={best['mu_o_s']:.2f}, "
                           f"casing(rot)={best['mu_c_r']:.2f}, open(rot)={best['mu_o_r']:.2f} (SSE={best['SSE']:.1f})")
                st.caption("Overlay toggle (above) compares baseline vs calibrated curves; "
                           "or copy values into inputs and re-run.")
        except Exception as e:
            st.error(f"Calibration failed: {e}")

    # Download elemental results
    st.markdown("### Iteration trace (first 12 rows) + download")
    st.dataframe(df_pick.head(12), use_container_width=True)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        df_pick.to_excel(xw, index=False, sheet_name="Pickup")
        df_so.to_excel(xw,   index=False, sheet_name="Slackoff")
        df_rot_tq.to_excel(xw, index=False, sheet_name="Rotate_TorqueOnly")
    st.download_button("Download elemental results (xlsx)", data=out.getvalue(),
                       file_name="TD_iteration_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.caption("Johancsik soft-string (Δs=1 ft). Survey → shoe → T&D are linked. Defaults: last casing 9-5/8, OH 8.50 in. "
               "Tools include history-matching μ, calibrated overlay, NP check, 0.8×MU gate, BSR/SR, tortuosity penalty, motor bit torque, "
               "and rig-limit margins. Rotating off-bottom HL = average(PU, SO).")
