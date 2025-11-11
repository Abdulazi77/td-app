# app/streamlit_app.py
# -*- coding: utf-8 -*-
"""
PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
Linked single-tab app (survey → shoe → T&D) aligned with the lecture.
Adds μ history-match, neutral-point/DC hint, 0.8× MU gate, BSR/SR checks,
tortuosity bump, motor ΔP→Tbit, and rig-limit margins.
"""

from __future__ import annotations
import io, math, base64, itertools
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------ Page ------------------------------------------
st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")

# ------------------------------ Constants/helpers -----------------------------
DEG2RAD, IN2FT = math.pi/180.0, 1.0/12.0
def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg) / 65.5              # steel ~65.5 ppg
def I_moment(od_in, id_in): return (math.pi/64.0)*(od_in**4 - id_in**4)
def J_polar(od_in, id_in):  return (math.pi/32.0)*(od_in**4 - id_in**4)  # polar area moment (torsion)

def png_download_button(fig: go.Figure, label: str, filename: str):
    buf = io.BytesIO(fig.to_image(format="png", scale=2))
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")

# Minimal casing & tool-joint sets (extend as needed)
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

# ------------------------------ Survey builders -------------------------------
def synth_build_hold(kop_md, build_rate_deg_per_100ft, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * (build_rate_deg_per_100ft/100.0))
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br, dr = build_rate/100.0, drop_rate/100.0
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * br)
    theta = np.maximum(0.0, theta - np.maximum(0.0, md - 0.75*target_md) * dr)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate/100.0
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
        I1, A1 = inc_deg[i-1]*DEG2RAD, az_deg[i-1]*DEG2RAD
        I2, A2 = inc_deg[i]*DEG2RAD,   az_deg[i]*DEG2RAD
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

# ------------------------------ Soft-string (Johancsik) -----------------------
def soft_string_stepper(md, inc_deg, kappa_rad_per_ft, cased_mask,
                        comp_along_depth, comp_props,
                        mu_slide_cased, mu_slide_open, mu_rot_cased, mu_rot_open,
                        mw_ppg, scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0):
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg)
    nseg = len(md) - 1
    if nseg <= 0: raise ValueError("Trajectory is empty.")
    inc = np.deg2rad(inc_deg[:-1])

    kappa_all = np.asarray(kappa_rad_per_ft)
    kappa_seg = kappa_all[:-1] if len(kappa_all) == len(md) else kappa_all
    if len(kappa_seg) != nseg: kappa_seg = np.resize(kappa_seg, nseg)

    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s = np.zeros(nseg); mu_r = np.zeros(nseg); BF = bf_from_mw(mw_ppg)

    cased_seg = np.asarray(cased_mask)[:nseg]
    comp_arr  = np.asarray(list(comp_along_depth))[:nseg]

    for i in range(nseg):
        props = comp_props[comp_arr[i]]
        od_in, id_in, w_air_ft = float(props["od_in"]), float(props["id_in"]), float(props["w_air_lbft"])
        w_air[i] = w_air_ft; w_b[i] = w_air_ft*BF
        r_eff_ft[i] = 0.5*od_in*IN2FT
        mu_s[i], mu_r[i] = (mu_slide_cased, mu_rot_cased) if cased_seg[i] else (mu_slide_open, mu_rot_open)

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)
    dT = np.zeros(nseg);  dM = np.zeros(nseg);  N_side = np.zeros(nseg)

    if scenario == "onbottom":
        T[0] = -float(WOB_lbf)               # on-bottom: enforce WOB at bit
        M[0] = float(Mbit_ftlbf)

    sgn_ax = {"pickup": +1.0, "slackoff": -1.0}.get(scenario, 0.0)

    for i in range(nseg):
        N_side[i] = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*N_side[i])*ds
        M_next = M[i] + (mu_r[i]*N_side[i]*r_eff_ft[i])*ds
        dT[i] = T_next - T[i];  dM[i] = M_next - M[i]
        T[i+1] = T_next;        M[i+1] = M_next

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

# Envelope & buckling helpers
def api7g_envelope_points(F_lim, T_lim, n=80):
    Fmax = max(float(F_lim), 1.0)
    F = np.linspace(0.0, Fmax, n)
    T = float(T_lim)*np.sqrt(np.clip(1.0 - (F/Fmax)**2, 0.0, 1.0))
    return F, T

def Fs_sinusoidal(Epsi, Iin4, w_b_lbf_ft, inc_deg, clearance_ft):
    theta = np.deg2rad(np.maximum(0.0, inc_deg)); r = max(1e-6, clearance_ft)
    return 2.0*np.sqrt(Epsi*Iin4 * w_b_lbf_ft*np.sin(theta)/r)

def Fh_helical(Fs): return 1.6*Fs

# ------------------------------ μ history-match (grid LSQ) --------------------
def history_match_mu(md, inc_deg, kappa, cased_mask, comp_along, comp_props, mw_ppg,
                     target_pick_lbf=None, target_slack_lbf=None, target_rot_lbf=None,
                     target_surf_torque=None,
                     mu_bounds=(0.10, 0.45), coarse_step=0.05, fine_step=0.02, bump_open_pct=0.0):
    """
    Simple least-squares fit of four μ's (μs_cased, μs_open, μr_cased, μr_open).
    Coarse grid then refined local grid around best point.
    """
    lo, hi = mu_bounds
    def cost(mu_s_c, mu_s_o, mu_r_c, mu_r_o):
        # apply tortuosity bump in open-hole
        mu_s_o_eff = mu_s_o*(1.0 + bump_open_pct/100.0)
        mu_r_o_eff = mu_r_o*(1.0 + bump_open_pct/100.0)
        # predict 3 states
        def run(scenario):
            df, T, M = soft_string_stepper(md, inc_deg, kappa, cased_mask, comp_along, comp_props,
                                           mu_s_c, mu_s_o_eff, mu_r_c, mu_r_o_eff, mw_ppg,
                                           scenario=scenario, WOB_lbf=0.0, Mbit_ftlbf=0.0)
            return max(0.0, -T[-1]), abs(M[-1])  # hookload up, surface torque
        J = 0.0; n = 0
        if target_pick_lbf is not None:
            pred, _ = run("pickup"); J += (pred - target_pick_lbf)**2; n+=1
        if target_slack_lbf is not None:
            pred, _ = run("slackoff"); J += (pred - target_slack_lbf)**2; n+=1
        if target_rot_lbf is not None:
            pred, _ = run("rotate_off"); J += (pred - target_rot_lbf)**2; n+=1
        if target_surf_torque is not None:
            _, tor = run("rotate_off"); J += (tor - target_surf_torque)**2; n+=1
        return J/(n if n>0 else 1.0)

    # coarse grid
    grid = np.arange(lo, hi+1e-9, coarse_step)
    best = None; best_tuple = None
    for mu_s_c, mu_s_o, mu_r_c, mu_r_o in itertools.product(grid, grid, grid, grid):
        J = cost(mu_s_c, mu_s_o, mu_r_c, mu_r_o)
        if (best is None) or (J < best):
            best = J; best_tuple = (mu_s_c, mu_s_o, mu_r_c, mu_r_o)

    # refine around best
    mu_s_c, mu_s_o, mu_r_c, mu_r_o = best_tuple
    def refine(val): return np.clip(np.arange(val-2*fine_step, val+2.001*fine_step, fine_step), lo, hi)
    best2 = best; best2_tuple = best_tuple
    for ms_c in refine(mu_s_c):
        for ms_o in refine(mu_s_o):
            for mr_c in refine(mu_r_c):
                for mr_o in refine(mu_r_o):
                    J = cost(ms_c, ms_o, mr_c, mr_o)
                    if J < best2:
                        best2, best2_tuple = J, (ms_c, ms_o, mr_c, mr_o)
    return best2_tuple, best2

# ------------------------------ UI --------------------------------------------
(tab,) = st.tabs(["Wellpath + Torque & Drag (linked)"])
with tab:
    # ================== Trajectory ==================
    st.subheader("Trajectory & 3D schematic (Minimum Curvature)")
    c1, c2, c3, c4 = st.columns(4)
    profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"],
                           help="Synthetic survey generator; swap to real surveys later if needed.")
    kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2100.0, 50.0)
    build   = c3.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.40, 0.10)
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
        lateral    = r1.number_input("Lateral length (ft)", 0.0, 30000.0, 5000.0, 100.0)
        target_md  = r2.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_horizontal(kop_md, build, lateral, target_md, az_deg)

    N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

    # ================== Casing / Open hole ==================
    st.subheader("Casing / Open-hole (simple, last string + open hole)")
    md_end = float(md[-1])
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    nominal = cc1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), index=1)
    weight  = cc2.selectbox("lb/ft (standards only)", list(CASING_DB[nominal]["weights"].keys()))
    casing_id_in = float(CASING_DB[nominal]["weights"][weight])
    cc3.text_input("Casing ID (in)", f"{casing_id_in:.3f}", disabled=True)

    # shoe bounded
    if 'shoe_md' not in st.session_state: st.session_state['shoe_md'] = min(3000.0, md_end)
    st.session_state['shoe_md'] = clamp(float(st.session_state['shoe_md']), 0.0, md_end)
    st.session_state['shoe_md'] = cc4.slider("Shoe MD (ft)", 0.0, md_end, float(st.session_state['shoe_md']), 50.0,
                                             help="Depth of last casing shoe; above is 'cased', below is 'open hole'.")
    hole_diam_in = cc5.number_input("Open-hole diameter (in)", 4.0, 20.0, 8.50, 0.01)
    shoe_md = float(st.session_state['shoe_md'])
    cased_mask = md <= shoe_md

    # 3D wellpath (downward TVD)
    idx = int(np.searchsorted(md, shoe_md, side='right'))
    fig3d = go.Figure()
    if idx > 1:
        fig3d.add_trace(go.Scatter3d(x=E[:idx], y=N[:idx], z=TVD[:idx], mode="lines",
                                     line=dict(width=6, color="#4cc9f0"), name="Cased"))
    if idx < len(md):
        fig3d.add_trace(go.Scatter3d(x=E[idx-1:], y=N[idx-1:], z=TVD[idx-1:], mode="lines",
                                     line=dict(width=4, color="#a97142"), name="Open-hole"))
    fig3d.update_layout(height=420, scene=dict(
        xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)",
        zaxis=dict(autorange="reversed")
    ), legend=dict(orientation="h"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3d, use_container_width=True)

    # 2D VS profile
    st.subheader("2D Wellbore Profile — TVD vs Vertical Section")
    vs_ref = st.number_input("VS reference azimuth (deg)", 0.0, 360.0, float(az[0] if len(az) else az_deg), 1.0)
    VS = N*np.cos(vs_ref*DEG2RAD) + E*np.sin(vs_ref*DEG2RAD)
    fig2d = go.Figure()
    if idx > 1:
        fig2d.add_trace(go.Scatter(x=VS[:idx], y=TVD[:idx], mode="lines",
                                   line=dict(width=6, color="#4cc9f0"), name="Cased"))
    if idx < len(md):
        fig2d.add_trace(go.Scatter(x=VS[idx-1:], y=TVD[idx-1:], mode="lines",
                                   line=dict(width=4, color="#a97142"), name="Open-hole"))
    fig2d.update_layout(xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)",
                        yaxis=dict(autorange="reversed"),
                        height=360, legend=dict(orientation="h"),
                        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2d, use_container_width=True)

    survey_df = pd.DataFrame({
        "MD (ft)": md, "Inc (deg)": inc_deg, "Az (deg)": az,
        "TVD (ft)": TVD, "North (ft)": N, "East (ft)": E, "VS (ft)": VS, "DLS (deg/100 ft)": DLS
    })
    st.dataframe(survey_df.head(15), use_container_width=True)
    st.download_button("Download survey CSV", survey_df.to_csv(index=False).encode(),
                       file_name="survey.csv", mime="text/csv")

    # ================== T&D inputs ==================
    st.subheader("Soft-string Torque & Drag — Johancsik (linked to survey above)")
    c_mu1, c_mu2, c_mu3 = st.columns(3)
    mu_cased_slide = c_mu1.number_input("μ in casing (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_open_slide  = c_mu2.number_input("μ in open-hole (sliding)", 0.05, 0.80, 0.35, 0.01)
    micro_bump     = c_mu3.slider("Tortuosity bump on open-hole μ (%)", 0, 50, 0,
                                  help="Inflates μ_open (sliding & rotating) to mimic micro-DLS/tortuosity.")

    mu_cased_rot   = st.number_input("μ in casing (rotating)", 0.05, 0.80, 0.25, 0.01)
    mu_open_rot    = st.number_input("μ in open-hole (rotating)", 0.05, 0.80, 0.35, 0.01)
    mw_ppg         = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    st.markdown("**Drillstring (bit up)**")
    d1, d2, d3 = st.columns(3)
    dc_len  = d1.number_input("DC length (ft)", 0.0, 10000.0, 600.0, 10.0)
    hwdp_len= d2.number_input("HWDP length (ft)", 0.0, 20000.0, 1000.0, 10.0)
    default_dp = max(0.0, float(md[-1])-(dc_len+hwdp_len))
    dp_len  = d3.number_input("DP length (ft)", 0.0, 50000.0, default_dp, 10.0,
                              help="Auto-filled so DC + HWDP + DP ≈ total MD.")

    e1, e2, e3 = st.columns(3)
    dc_od  = e1.number_input("DC OD (in)",   3.0, 12.0, 8.0, 0.1)
    hwdp_od= e2.number_input("HWDP OD (in)", 2.0,  8.0, 3.5, 0.1)
    dp_od  = e3.number_input("DP OD (in)",   3.0, 6.625, 5.0, 0.01)

    i1, i2, i3 = st.columns(3)
    dc_id  = i1.number_input("DC ID (in)",   0.5, 6.0, 2.81, 0.01)
    hwdp_id= i2.number_input("HWDP ID (in)", 0.5, 4.0, 2.00, 0.01)
    dp_id  = i3.number_input("DP ID (in)",   1.5, 5.0, 4.28, 0.01)

    w1, w2, w3 = st.columns(3)
    dc_w   = w1.number_input("DC weight (air, lb/ft)",   30.0, 150.0, 66.7, 0.1)
    hwdp_w = w2.number_input("HWDP weight (air, lb/ft)",  5.0,  40.0, 16.0, 0.1)
    dp_w   = w3.number_input("DP weight (air, lb/ft)",    8.0,  40.0, 19.5, 0.1)

    if dc_len + hwdp_len + dp_len > md_end + 1e-6:
        st.warning("DC + HWDP + DP length exceeds total MD. Consider reducing one of them.")

    # map along depth
    nseg = len(md) - 1
    comp_along = np.empty(nseg, dtype=object)
    for i in range(nseg):
        from_bit = float(md[-1]) - md[i]
        comp_along[i] = "DC" if from_bit <= dc_len else ("HWDP" if from_bit <= dc_len + hwdp_len else "DP")
    comp_props = {
        "DC":   {"od_in": dc_od,   "id_in": dc_id,   "w_air_lbft": dc_w},
        "HWDP": {"od_in": hwdp_od, "id_in": hwdp_id, "w_air_lbft": hwdp_w},
        "DP":   {"od_in": dp_od,   "id_in": dp_id,   "w_air_lbft": dp_w},
    }
    kappa = (DLS*DEG2RAD)/100.0

    # Scenario (+ optional motor torque on-bottom)
    scen = st.selectbox("Scenario", ["Slack-off (RIH)","Pickup (POOH)","Rotate off-bottom","Rotate on-bottom"])
    scenario = {"Slack-off (RIH)":"slackoff","Pickup (POOH)":"pickup",
                "Rotate off-bottom":"rotate_off","Rotate on-bottom":"onbottom"}[scen]
    cmot1, cmot2 = st.columns(2)
    wob  = cmot1.number_input("WOB (lbf) for on-bottom", 0.0, 120000.0, 6000.0, 100.0)
    use_motor = cmot2.checkbox("Motor mode (ΔP → bit torque)", value=False)
    if use_motor:
        dP = st.number_input("Motor ΔP (psi)", 0.0, 2000.0, 0.0, 10.0,
                             help="Simple linear map to bit torque; for RSS leave 0.")
        k_t = st.number_input("Torque coefficient (lbf-ft/psi)", 0.0, 10.0, 0.5, 0.05)
        mbit = dP * k_t
    else:
        mbit = st.number_input("Bit torque (lbf-ft) for on-bottom", 0.0, 50000.0, 0.0, 100.0)

    # Apply tortuosity bump to open-hole μ’s
    mu_open_slide_eff = mu_open_slide*(1.0 + micro_bump/100.0)
    mu_open_rot_eff   = mu_open_rot*(1.0 + micro_bump/100.0)

    df_itr, T_arr, M_arr = soft_string_stepper(
        md, inc_deg, kappa, cased_mask, comp_along, comp_props,
        mu_cased_slide, mu_open_slide_eff, mu_cased_rot, mu_open_rot_eff,
        mw_ppg, scenario=scenario, WOB_lbf=wob if scenario=="onbottom" else 0.0, Mbit_ftlbf=mbit if scenario=="onbottom" else 0.0
    )
    depth = df_itr["md_bot_ft"].to_numpy()
    surf_hookload = max(0.0, -T_arr[-1]); surf_torque = abs(M_arr[-1])

    # Neutral point & DC sufficiency (on-bottom view)
    np_txt = ""
    if scenario == "onbottom":
        # find first index from bit where axial tension crosses zero (compression below)
        T_seg = df_itr["T_next_lbf"].to_numpy()
        cross = np.where(T_seg >= 0.0)[0]
        if len(cross) == 0:
            NP = float(md[-1])   # all in compression
        else:
            NP = depth[cross[0]]
        np_txt = f"Neutral point ≈ **{NP:,.0f} ft** from surface"
        in_DC = NP >= (md[-1]-dc_len)
        if not in_DC:
            need_dc = md[-1] - NP + 50.0  # ~50 ft margin
            st.warning(f"Neutral point not inside collars. Suggest DC length ≥ ~{need_dc:,.0f} ft to keep NP within DCs.")
        else:
            st.info("Neutral point is within drill collars ✓")
    st.success(f"Surface hookload: {surf_hookload:,.0f} lbf — Surface torque: {surf_torque:,.0f} lbf-ft"
               + (f" — {np_txt}" if np_txt else ""))

    # ================== Safety & limits (Dr.-style charts) ==================
    st.subheader("Safety & limits")
    s1, s2, s3 = st.columns(3)
    tj_name   = s1.selectbox("Tool-joint size", list(TOOL_JOINT_DB.keys()), index=2)
    sf_joint  = s2.number_input("Safety factor (tool-joint)", 1.00, 2.00, 1.10, 0.05)
    sf_tension= s3.number_input("SF for pipe body tension", 1.00, 2.00, 1.15, 0.05)
    rig_torque_lim = st.number_input("Top-drive torque limit (lbf-ft)", 10000, 150000, 60000, 1000)
    rig_pull_lim   = st.number_input("Rig max hookload (lbf)",        50000, 1500000, 500000, 5000)
    mu_band = st.multiselect("μ sweep for off-bottom risk curves", [0.15,0.20,0.25,0.30,0.35,0.40],
                             default=[0.20,0.25,0.30,0.35])

    tj = TOOL_JOINT_DB[tj_name]
    T_makeup    = tj['T_makeup_ftlbf']
    T_makeup_sf = T_makeup/sf_joint
    T_yield_sf  = tj['T_yield_ftlbf']/sf_joint
    F_tensile_sf= tj['F_tensile_lbf']/sf_tension

    # 0.8× make-up rule and margins
    T_eighty = 0.8*T_makeup
    margin_torque_to_80 = (T_eighty - surf_torque)/1000.0
    margin_torque_to_topdrive = (rig_torque_lim - surf_torque)/1000.0
    margin_pull = (rig_pull_lim - surf_hookload)/1000.0
    st.caption(f"Margins: torque to 0.8×MU = **{margin_torque_to_80:,.1f} kft-lbf**, "
               f"to top-drive = **{margin_torque_to_topdrive:,.1f} kft-lbf**; "
               f"pull to rig limit = **{margin_pull:,.1f} klbf**.")
    if surf_torque > T_eighty:
        st.error("Surface torque exceeds 0.8× make-up torque rule (lecture guidance).")

    # helper for μ-sweep
    def run_td_off_bottom(mu):
        df_tmp, _, _ = soft_string_stepper(
            md, inc_deg, kappa, cased_mask, comp_along, comp_props,
            mu, mu*(1.0+micro_bump/100.0), mu, mu*(1.0+micro_bump/100.0), mw_ppg,
            scenario="rotate_off", WOB_lbf=0.0, Mbit_ftlbf=0.0
        )
        return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy())

    # LEFT (risk curves)
    fig_left = go.Figure()
    for mu in mu_band:
        dmu, tmu = run_td_off_bottom(mu)
        fig_left.add_trace(go.Scatter(x=tmu/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"))
    fig_left.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash", annotation_text="Make-up/SF")
    fig_left.add_vline(x=T_eighty/1000.0,    line_color="#ffaa00", line_dash="dot", annotation_text="0.8× MU")
    fig_left.update_yaxes(autorange="reversed", title_text="Depth (ft)")
    fig_left.update_xaxes(title_text="Off-bottom torque (k lbf-ft)")
    fig_left.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

    # RIGHT (elemental torque + combined-load limit)
    fig_right = go.Figure()
    for mu in mu_band:
        dmu, tmu = run_td_off_bottom(mu)
        fig_right.add_trace(go.Scatter(x=tmu/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"))
    fig_right.add_vline(x=T_makeup_sf/1000.0,  line_color="#00d5ff", line_dash="dash", annotation_text="Make-up/SF")
    fig_right.add_vline(x=T_eighty/1000.0,     line_color="#ffaa00", line_dash="dot", annotation_text="0.8× MU")
    fig_right.add_vline(x=rig_torque_lim/1000.0, line_color="magenta", line_dash="dot", annotation_text="Top-drive")

    # Combined-load line at pickup tension
    df_pick, _, _ = soft_string_stepper(
        md, inc_deg, kappa, cased_mask, comp_along, comp_props,
        mu_cased_slide, mu_open_slide_eff, mu_cased_rot, mu_open_rot_eff,
        mw_ppg, scenario="pickup", WOB_lbf=0.0, Mbit_ftlbf=0.0
    )
    F_ax = np.maximum(0.0, df_pick["T_next_lbf"].to_numpy())
    T_allow = T_yield_sf * np.sqrt(np.clip(1.0 - (F_ax / np.maximum(F_tensile_sf, 1.0))**2, 0.0, 1.0))
    fig_right.add_trace(go.Scatter(x=T_allow/1000.0, y=df_pick["md_bot_ft"].to_numpy(),
                                   mode="lines", name="TJ combined-load", line=dict(dash="dot")))
    fig_right.update_yaxes(autorange="reversed", title_text="Depth (ft)")
    fig_right.update_xaxes(title_text="Elemental torque (k lbf-ft)")
    fig_right.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

    st.markdown("### T&D Model Chart — Risk curves and limits")
    cL, cR = st.columns(2)
    with cL:
        st.plotly_chart(fig_left, use_container_width=True)
        png_download_button(fig_left, "Save left chart (PNG)", "risk_curves.png")
    with cR:
        st.plotly_chart(fig_right, use_container_width=True)
        png_download_button(fig_right, "Save right chart (PNG)", "elemental_torque.png")

    # ================== Envelope & Hookload diagnostics ==================
    Epsi = 30.0e6; Iin4 = I_moment(dp_od, dp_id)
    rbore_ft, rpipe_ft = 0.5*hole_diam_in*IN2FT, 0.5*dp_od*IN2FT
    clearance_ft = max(1e-3, rbore_ft - rpipe_ft)
    Fs = Fs_sinusoidal(Epsi, Iin4, df_itr['w_b_lbft'].to_numpy(), df_itr['inc_deg'].to_numpy(), clearance_ft)
    Fh = Fh_helical(Fs)
    F_env, T_env = api7g_envelope_points(F_tensile_sf, T_yield_sf, n=100)
    hookload = np.maximum(0.0, -df_itr['T_next_lbf'].to_numpy())

    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(x=(T_env)/1000.0, y=(F_env)/1000.0, mode="lines", name="API 7G envelope (approx)"))
    fig_env.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash", annotation_text="Make-up/SF")
    fig_env.add_trace(go.Scatter(x=[abs(df_itr['M_next_lbf_ft'].to_numpy()[-1])/1000.0], y=[hookload[-1]/1000.0],
                                 mode="markers", name="Operating point", marker=dict(size=10, color="orange")))
    fig_env.update_xaxes(title_text="Torque (k lbf-ft)")
    fig_env.update_yaxes(title_text="Tension (k lbf)")
    fig_env.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))

    fig_hl = go.Figure()
    fig_hl.add_trace(go.Scatter(x=hookload/1000.0, y=depth, mode="lines", name="Hookload"))
    fig_hl.add_vline(x=rig_pull_lim/1000.0, line_color="magenta", line_dash="dot", annotation_text="Rig pull limit")
    fig_hl.add_trace(go.Scatter(x=Fs/1000.0, y=depth, name="Sinusoidal Fs", line=dict(dash="dash")))
    fig_hl.add_trace(go.Scatter(x=Fh/1000.0, y=depth, name="Helical Fh", line=dict(dash="dot")))
    fig_hl.update_yaxes(autorange="reversed", title_text="Depth (ft)")
    fig_hl.update_xaxes(title_text="Force / Hookload (k lbf)")
    fig_hl.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))

    st.markdown("### Envelope & Hookload diagnostics")
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(fig_env, use_container_width=True)
    with c4: st.plotly_chart(fig_hl,  use_container_width=True)

    # ================== BSR / SR quick checks ==================
    st.subheader("Connection checks — BSR & stiffness ratio")
    b1, b2, b3, b4 = st.columns(4)
    up_od = b1.number_input("Upstream OD (in)", 3.5, 9.0, 5.0, 0.01,
                            help="OD above the connection / transition")
    up_id = b2.number_input("Upstream ID (in)", 1.5, 5.0, 4.28, 0.01)
    dn_od = b3.number_input("Downstream OD (in)", 3.5, 9.0, 6.63, 0.01,
                            help="OD at tool-joint/connection or below transition")
    dn_id = b4.number_input("Downstream ID (in)", 1.5, 5.0, 3.00, 0.01)
    J_up, J_dn = J_polar(up_od, up_id), J_polar(dn_od, dn_id)
    I_up, I_dn = I_moment(up_od, up_id), I_moment(dn_od, dn_id)
    BSR = J_dn/max(J_up,1e-9)
    SR  = I_dn/max(I_up,1e-9)
    st.caption(f"BSR = J_down/J_up = **{BSR:0.2f}**;  SR = I_down/I_up = **{SR:0.2f}**")
    if BSR < 1.0:
        st.warning("BSR < 1.0 → potential twist-off risk at the connection (lecture note).")
    if SR < 0.8 or SR > 1.2:
        st.warning("Stiffness jump (SR outside ~0.8–1.2) may concentrate stress / torque.")

    # ================== μ ranges crib + history matching ==================
    with st.expander("Typical μ ranges (starting points) + History match"):
        st.table(pd.DataFrame({
            "Environment":["Cased WBM (sliding)","Cased (rotating)","Open-hole WBM (sliding)","Open-hole (rotating)"],
            "Typical μ":[ "0.15–0.20","0.18–0.25","0.25–0.40","0.25–0.35" ],
        }))
        st.markdown("**History match (single depth trio):** enter measured surface hookload for Pickup, Slack-off, Rotate off-bottom (and optional surface torque).")
        hm1, hm2, hm3, hm4 = st.columns(4)
        m_pick  = hm1.number_input("Measured pickup (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
        m_slack = hm2.number_input("Measured slack-off (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
        m_rot   = hm3.number_input("Measured rotate off-bottom (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
        m_tor   = hm4.number_input("Measured surface torque (lbf-ft, optional)", 0.0, 150_000.0, 0.0, 100.0)
        cc = st.columns(3)
        coarse = cc[0].selectbox("Coarse step", [0.05,0.04,0.03], index=0)
        fine   = cc[1].selectbox("Fine step",   [0.02,0.015,0.01], index=0)
        mu_lo, mu_hi = cc[2].slider("Bounds for μ", 0.05, 0.60, (0.10, 0.45))
        if st.button("Fit μ from measurements"):
            (ms_c, ms_o, mr_c, mr_o), J = history_match_mu(
                md, inc_deg, kappa, cased_mask, comp_along, comp_props, mw_ppg,
                target_pick_lbf= m_pick if m_pick>0 else None,
                target_slack_lbf=m_slack if m_slack>0 else None,
                target_rot_lbf=  m_rot if m_rot>0 else None,
                target_surf_torque= m_tor if m_tor>0 else None,
                mu_bounds=(mu_lo, mu_hi), coarse_step=float(coarse), fine_step=float(fine),
                bump_open_pct=micro_bump
            )
            st.success(f"Fitted μ values — casing(slide)={ms_c:.2f}, open(slide)={ms_o:.2f}, "
                       f"casing(rot)={mr_c:.2f}, open(rot)={mr_o:.2f}  (mean squared error={J:,.0f})")
            st.info("Click above μ inputs to copy the values manually (kept explicit for transparency).")

    # ================== Iteration trace + export ==================
    st.markdown("### Iteration trace (first 15 rows)")
    st.dataframe(df_itr.head(15), use_container_width=True)
    st.download_button("Download iteration trace CSV",
                       df_itr.to_csv(index=False).encode(), file_name="td_iteration_trace.csv", mime="text/csv")

    st.caption(
        "Soft-string, Δs = 1 ft. Linked flow: survey → shoe → cased/open-hole masks → T&D. "
        "Left chart = μ-sweep risk curves; Right = elemental torque with make-up/top-drive and combined-load line. "
        "0.8× MU gate and rig-limit margins shown per lecture guidance."
    )
