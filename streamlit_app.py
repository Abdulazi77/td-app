# -*- coding: utf-8 -*-
# PEGN 517 – Wellpath + Torque & Drag (Δs = 1 ft)
# Default view = Dr-style “Drag Risk Off-Bottom Torque” panel + Elemental torque with safety lines.

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")

# ------------------------------ constants/helpers ------------------------------
IN2FT   = 1.0/12.0
DEG2RAD = math.pi/180.0
def in2ft(v_in): return v_in*IN2FT
def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5  # steel ~ 65.5 ppg

# minimal casing DB (ID locked to standards)
CASING_DB = {
    "13-3/8": {"weights": {48.0: 12.415, 54.5: 12.347, 61.0: 12.107}, "grades": ["J55","K55","L80","P110"]},
    "9-5/8":  {"weights": {29.3: 8.921, 36.0: 8.535, 40.0: 8.321},    "grades": ["J55","L80","P110"]},
    "7":      {"weights": {20.0: 6.366, 23.0: 6.059, 26.0: 5.920},    "grades": ["J55","L80","P110"]},
    "5-1/2":  {"weights": {17.0: 4.778, 20.0: 4.670, 23.0: 4.560},    "grades": ["J55","L80","P110"]},
}

# tool-joint mini DB (for safety lines)
TOOL_JOINT_DB = {
    'NC38': {'od': 4.75, 'id': 2.25, 'T_makeup_ftlbf': 12000, 'F_tensile_lbf': 350000, 'T_yield_ftlbf': 20000},
    'NC40': {'od': 5.00, 'id': 2.25, 'T_makeup_ftlbf': 16000, 'F_tensile_lbf': 420000, 'T_yield_ftlbf': 25000},
    'NC50': {'od': 6.63, 'id': 3.00, 'T_makeup_ftlbf': 30000, 'F_tensile_lbf': 650000, 'T_yield_ftlbf': 45000},
}

# ------------------------------ synthetic survey builders ----------------------
def synth_build_hold(kop_md, build_rate_deg_per_100ft, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    theta = np.zeros_like(md)
    az = np.full_like(md, az_deg, dtype=float)
    brad = build_rate_deg_per_100ft/100.0
    for i, m in enumerate(md):
        theta[i] = 0.0 if m < kop_md else min(theta_hold_deg, (m - kop_md)*brad)
    return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    theta = np.zeros_like(md)
    az = np.full_like(md, az_deg, dtype=float)
    brad = build_rate/100.0
    drad = drop_rate/100.0
    for i, m in enumerate(md):
        theta[i] = 0.0 if m < kop_md else min(theta_hold_deg, (m - kop_md)*brad)
    start_drop = 0.75*target_md
    for i, m in enumerate(md):
        if m > start_drop and theta[i] > 0:
            theta[i] = max(0.0, theta[i] - (m - start_drop)*drad)
    return md, theta, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    theta = np.zeros_like(md)
    az = np.full_like(md, az_deg, dtype=float)
    brad = build_rate/100.0
    for i, m in enumerate(md):
        theta[i] = 0.0 if m < kop_md else min(theta_max, (m - kop_md)*brad)
    idx_h = np.where(theta >= theta_max - 1e-6)[0]
    if len(idx_h):
        m_h = md[idx_h[0]]
        md_end = max(m_h + lateral_length, target_md)
        md = np.arange(0, md_end + ds, ds)
        theta = np.where(md <= m_h, np.minimum(theta_max, np.maximum(0, (md - kop_md)*brad)), theta_max)
        az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

# Minimum curvature positions + DLS
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

# ------------------------------ Johancsik soft-string ----------------------
def soft_string_stepper(md, inc_deg, kappa_rad_per_ft, cased_mask,
                        comp_along_depth, comp_props,
                        mu_slide_cased, mu_slide_open, mu_rot_cased, mu_rot_open,
                        mw_ppg, scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0):
    """
    Δs=1 ft soft-string integration (bit→surface), returning per-segment trace and cumulative T/M arrays.
    """
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg)
    # station κ → per-seg κ
    kappa_arr = np.asarray(kappa_rad_per_ft)
    kappa_seg = kappa_arr[:-1] if kappa_arr.shape[0] == md.shape[0] else kappa_arr

    nseg = md.shape[0] - 1
    if nseg <= 0: raise ValueError("Need at least two MD stations for T&D.")
    inc = np.deg2rad(inc_deg[:-1])

    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s  = np.zeros(nseg);    mu_r = np.zeros(nseg)
    BF = bf_from_mw(mw_ppg);   BF_arr = np.full(nseg, BF, dtype=float)
    cased_seg = np.asarray(cased_mask)[:nseg]

    for i in range(nseg):
        comp = comp_along_depth[i]
        od_in = comp_props[comp]['od_in']; id_in = comp_props[comp]['id_in']; w_air_ft = comp_props[comp]['w_air_lbft']
        w_air[i] = w_air_ft; w_b[i] = w_air_ft*BF; r_eff_ft[i] = 0.5*in2ft(od_in)
        if cased_seg[i]: mu_s[i] = mu_slide_cased; mu_r[i] = mu_rot_cased
        else:            mu_s[i] = mu_slide_open;  mu_r[i] = mu_rot_open

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)
    if scenario == "onbottom": T[0] = -float(WOB_lbf); M[0] = float(Mbit_ftlbf)
    sgn_ax = +1.0 if scenario == "pickup" else -1.0 if scenario == "slackoff" else 0.0

    dT = np.zeros(nseg); dM = np.zeros(nseg); N_side = np.zeros(nseg)
    for i in range(nseg):
        N_side[i] = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*N_side[i])*ds; dT[i]=T_next-T[i]; T[i+1]=T_next
        M_next = M[i] + (mu_r[i]*N_side[i]*r_eff_ft[i])*ds;                   dM[i]=M_next-M[i]; M[i+1]=M_next

    df = pd.DataFrame({
        "md_top_ft": md[:-1], "md_bot_ft": md[1:], "ds_ft": np.full(nseg, 1.0),
        "inc_deg": inc_deg[:-1], "kappa_rad_ft": kappa_seg,
        "w_air_lbft": w_air, "BF": BF_arr, "w_b_lbft": w_b,
        "mu_slide": mu_s, "mu_rot": mu_r, "N_lbf": N_side,
        "dT_lbf": dT, "T_next_lbf": T[1:], "r_eff_ft": r_eff_ft,
        "dM_lbf_ft": dM, "M_next_lbf_ft": M[1:], "cased?": cased_seg,
        "comp": np.asarray(comp_along_depth)[:nseg]
    })
    return df, T, M

# safety helpers
def api7g_envelope_points(F_lim, T_lim, n=80):
    F = np.linspace(0, max(F_lim, 1.0), n)
    T = T_lim*np.sqrt(np.clip(1.0 - (F/max(F_lim,1.0))**2, 0.0, 1.0))
    return F, T

# ------------------------------ UI ---------------------------------------------------
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")
tab1, tab2 = st.tabs(["Trajectory & 3D schematic", "Torque & Drag"])

# ---------- Trajectory tab ----------
with tab1:
    st.subheader("Synthetic survey (Minimum Curvature)")
    c1, c2, c3, c4 = st.columns(4)
    profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
    kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
    az_quick = c3.selectbox("Quick azimuth", ["North (0)","East (90)","South (180)","West (270)"], index=0)
    az_map = {"North (0)":0.0,"East (90)":90.0,"South (180)":180.0,"West (270)":270.0}
    az_deg_precise = c4.number_input("Azimuth (deg from North, clockwise)", 0.0, 359.99, float(az_map[az_quick]), 0.5)

    br = st.number_input("Build rate (deg/100 ft)", 0.0, 20.0, 3.0, 0.1)
    c5, c6, c7 = st.columns(3)
    if profile == "Build & Hold":
        theta_hold = c5.number_input("Final inclination (deg)", 0.0, 90.0, 38.5, 0.5)
        target_md  = c6.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_build_hold(kop_md, br, theta_hold, target_md, az_deg_precise)
    elif profile == "Build–Hold–Drop":
        theta_hold = c5.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 0.5)
        drop_rate  = c6.number_input("Drop rate (deg/100 ft)", 0.0, 20.0, 2.0, 0.1)
        target_md  = c7.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_build_hold_drop(kop_md, br, theta_hold, drop_rate, target_md, az_deg_precise)
    else:
        lateral   = c5.number_input("Lateral length (ft)", 0.0, 30000.0, 2000.0, 100.0)
        target_md = c6.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_horizontal(kop_md, br, lateral, target_md, az_deg_precise)

    N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

    # Preview shoe (for 3D coloring) – default 9-5/8 @ 3000 ft, 8.5" OH
    p1, p2, p3 = st.columns(3)
    shoe_md_prev = p1.number_input("Preview shoe MD (ft)", 0.0, float(md[-1]), 3000.0, 50.0)
    last_casing  = p2.selectbox("Preview last casing nominal OD", list(CASING_DB.keys()), index=1)  # 9-5/8 default
    weight_prev  = p3.selectbox("Preview lb/ft (standards)", list(CASING_DB[last_casing]["weights"].keys()), index=1 if last_casing=="9-5/8" else 0)
    casing_id_prev = CASING_DB[last_casing]["weights"][weight_prev]

    cased_mask_prev = (md <= shoe_md_prev)

    # 3D with cased vs open-hole (brown)
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=E[cased_mask_prev], y=N[cased_mask_prev], z=-TVD[cased_mask_prev],
        mode="lines", line=dict(width=6, color="#4cc9f0"), name="Cased"
    ))
    fig3d.add_trace(go.Scatter3d(
        x=E[~cased_mask_prev], y=N[~cased_mask_prev], z=-TVD[~cased_mask_prev],
        mode="lines", line=dict(width=4, color="#8b572a"), name="Open-hole"
    ))
    fig3d.update_layout(height=480, scene=dict(
        xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    survey_df = pd.DataFrame({
        "MD (ft)": md, "Inc (deg)": inc_deg, "Az (deg)": az,
        "TVD (ft)": TVD, "North (ft)": N, "East (ft)": E, "DLS (deg/100 ft)": DLS
    })
    st.dataframe(survey_df.head(11), use_container_width=True)

    # share with T&D tab
    st.session_state["md"] = md; st.session_state["inc_deg"] = inc_deg
    st.session_state["az_deg"] = az; st.session_state["DLS"] = DLS
    st.session_state["shoe_preview"] = float(shoe_md_prev)

# ---------- T&D tab ----------
with tab2:
    if not all(k in st.session_state for k in ("md","inc_deg","az_deg","DLS")):
        st.warning("Define the trajectory first in the previous tab.")
        st.stop()

    md = st.session_state["md"]; inc_deg = st.session_state["inc_deg"]; az = st.session_state["az_deg"]
    # curvature per ft from DLS
    kappa = (st.session_state["DLS"]*DEG2RAD)/100.0

    st.subheader("Casing / Open-hole (simple)")
    cc1, cc2, cc3, cc4, cc5 = st.columns([1,1,1,1,1])
    nominal = cc1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), index=1)  # 9-5/8 default
    weight  = cc2.selectbox("lb/ft (standards only)", list(CASING_DB[nominal]["weights"].keys()),
                             index=(list(CASING_DB["9-5/8"]["weights"].keys()).index(36.0) if nominal=="9-5/8" else 0))
    casing_id_in = CASING_DB[nominal]["weights"][weight]
    cc3.text_input("Casing ID (in, locked)", f"{casing_id_in:.3f}", disabled=True)
    shoe_md = cc4.number_input("Deepest shoe MD (ft)", 0.0, float(md[-1]), max(3000.0, 0.0), 50.0)
    hole_diam_in = cc5.number_input("Open-hole diameter (in)", 4.0, 20.0, 8.5, 0.1)
    cased_mask = md <= shoe_md

    st.subheader("Friction & mud")
    c1, c2, c3, c4 = st.columns(4)
    mu_cased_slide = c1.number_input("μ in casing (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_open_slide  = c2.number_input("μ in open-hole (sliding)", 0.05, 0.80, 0.35, 0.01)
    mu_cased_rot   = c3.number_input("μ in casing (rotating)", 0.05, 0.80, 0.25, 0.01)
    mu_open_rot    = c4.number_input("μ in open-hole (rotating)", 0.05, 0.80, 0.35, 0.01)
    mw_ppg         = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    st.markdown("#### Drillstring (bit up)")
    d1, d2, d3 = st.columns(3)
    # DC
    dc_len = d1.number_input("DC length (ft)", 0.0, 10000.0, 600.0, 10.0)
    dc_od  = d2.number_input("DC OD (in)", 3.0, 12.0, 8.0, 0.1)
    dc_id  = d3.number_input("DC ID (in)", 0.5, 6.0, 2.81, 0.01)
    dc_w   = st.number_input("DC weight (air, lb/ft)", 30.0, 150.0, 66.7, 0.1)
    # HWDP
    h1, h2, h3 = st.columns(3)
    hwdp_len = h1.number_input("HWDP length (ft)", 0.0, 20000.0, 1000.0, 10.0)
    hwdp_od  = h2.number_input("HWDP OD (in)", 2.0, 8.0, 3.5, 0.1)
    hwdp_id  = h3.number_input("HWDP ID (in)", 0.5, 4.0, 2.0, 0.01)
    hwdp_w   = st.number_input("HWDP weight (air, lb/ft)", 5.0, 40.0, 16.0, 0.1)
    # DP
    p1, p2, p3 = st.columns(3)
    dp_len = p1.number_input("DP length (ft)", 0.0, 50000.0, max(0.0, float(md[-1]) - (dc_len + hwdp_len)), 10.0)
    dp_od  = p2.number_input("DP OD (in)", 3.0, 6.625, 5.0, 0.01)
    dp_id  = p3.number_input("DP ID (in)", 1.5, 5.0, 4.28, 0.01)
    dp_w   = st.number_input("DP weight (air, lb/ft)", 10.0, 30.0, 19.5, 0.1)

    comp_props = {"DC":{"od_in":dc_od,"id_in":dc_id,"w_air_lbft":dc_w},
                  "HWDP":{"od_in":hwdp_od,"id_in":hwdp_id,"w_air_lbft":hwdp_w},
                  "DP":{"od_in":dp_od,"id_in":dp_id,"w_air_lbft":dp_w}}
    # map component along depth (exactly nseg)
    nseg = len(md)-1
    comp_along = np.empty(nseg, dtype=object)
    for i in range(nseg):
        depth_from_bit = float(md[-1]) - md[i]
        if depth_from_bit <= dc_len: comp_along[i]="DC"
        elif depth_from_bit <= dc_len + hwdp_len: comp_along[i]="HWDP"
        else: comp_along[i]="DP"

    # scenario & bit BCs
    scen = st.selectbox("Scenario", ["Slack-off (RIH)","Pickup (POOH)","Rotate off-bottom","Rotate on-bottom"])
    if "Slack-off" in scen: scenario="slackoff"
    elif "Pickup" in scen:  scenario="pickup"
    elif "Rotate off-bottom" in scen: scenario="rotate_off"
    else: scenario="onbottom"
    wob  = st.number_input("WOB (lbf) for on-bottom", 0.0, 100000.0, 6000.0, 100.0)
    mbit = st.number_input("Bit torque (lbf-ft) for on-bottom", 0.0, 50000.0, 0.0, 100.0)

    # main run (current μ)
    df_itr, T_arr, M_arr = soft_string_stepper(
        md, inc_deg, kappa, cased_mask, comp_along, comp_props,
        mu_cased_slide, mu_open_slide, mu_cased_rot, mu_open_rot,
        mw_ppg, scenario=scenario, WOB_lbf=wob, Mbit_ftlbf=mbit
    )
    depth = df_itr["md_bot_ft"].to_numpy()
    surf_hookload = max(0.0, -T_arr[-1]); surf_torque = abs(M_arr[-1])
    st.success(f"Surface hookload: {surf_hookload:,.0f} lbf — Surface torque: {surf_torque:,.0f} lbf-ft")

    # --- Dr-style risk view (DEFAULT) ---
    st.subheader("T&D Model Chart — Risk curves and limits")

    s1, s2, s3 = st.columns(3)
    tj_name = s1.selectbox("Tool-joint size", list(TOOL_JOINT_DB.keys()), index=2)  # NC50 default
    sf_joint = s2.number_input("Safety factor (tool-joint)", 1.00, 2.00, 1.10, 0.05)
    rig_torque_lim = s3.number_input("Top-drive torque limit (lbf-ft)", 10000, 150000, 60000, 1000)

    mu_band = st.multiselect("μ sweep for off-bottom risk curves",
                             [0.15,0.20,0.25,0.30,0.35,0.40],
                             default=[0.15,0.20,0.25,0.30,0.35])

    tj = TOOL_JOINT_DB[tj_name]
    T_makeup_sf = tj['T_makeup_ftlbf']/sf_joint
    T_yield_sf  = tj['T_yield_ftlbf']/sf_joint

    # μ-sweep helper
    def run_td_off_bottom(mu):
        df_tmp, _, _ = soft_string_stepper(
            md, inc_deg, kappa, cased_mask, comp_along, comp_props,
            mu, mu, mu, mu, mw_ppg,
            scenario="rotate_off", WOB_lbf=0.0, Mbit_ftlbf=0.0
        )
        return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy())

    # Left: μ-sweep off-bottom torque vs depth
    fig_left = go.Figure()
    for mu in mu_band:
        dmu, tmu = run_td_off_bottom(mu)
        fig_left.add_trace(go.Scatter(x=tmu/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"))
    fig_left.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash",
                       annotation_text="Make-up torque / SF", annotation_position="top")
    fig_left.add_vline(x=rig_torque_lim/1000.0, line_color="magenta", line_dash="dot",
                       annotation_text="Top-drive limit", annotation_position="top")
    fig_left.update_yaxes(autorange="reversed", title_text="Depth (ft)")
    fig_left.update_xaxes(title_text="Off-bottom torque (k lbf-ft)")
    fig_left.update_layout(height=680, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))

    # Right: elemental torque (current μ) + lines
    comp = df_itr['comp'].to_numpy(); dM = df_itr['dM_lbf_ft'].to_numpy()
    tor_dc = np.cumsum(np.where(comp=='DC', dM, 0.0))
    tor_hwdp = np.cumsum(np.where(comp=='HWDP', dM, 0.0))
    tor_dp = np.cumsum(np.where(comp=='DP', dM, 0.0))
    tor_total = np.cumsum(dM)

    fig_right = go.Figure()
    fig_right.add_trace(go.Scatter(x=tor_dc/1000.0,   y=depth, name="DC",   mode="lines"))
    fig_right.add_trace(go.Scatter(x=tor_hwdp/1000.0, y=depth, name="HWDP", mode="lines"))
    fig_right.add_trace(go.Scatter(x=tor_dp/1000.0,   y=depth, name="DP",   mode="lines"))
    fig_right.add_trace(go.Scatter(x=tor_total/1000.0,y=depth, name="Total", mode="lines", line=dict(width=3)))
    fig_right.add_vline(x=T_makeup_sf/1000.0,  line_color="#00d5ff", line_dash="dash",
                        annotation_text="Make-up / SF", annotation_position="top")
    fig_right.add_vline(x=rig_torque_lim/1000.0, line_color="magenta", line_dash="dot",
                        annotation_text="Top-drive limit", annotation_position="top")
    fig_right.update_yaxes(autorange="reversed", title_text="Depth (ft)")
    fig_right.update_xaxes(title_text="Elemental torque (k lbf-ft)")
    fig_right.update_layout(height=680, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))

    cL, cR = st.columns(2)
    with cL: st.plotly_chart(fig_left, use_container_width=True)
    with cR: st.plotly_chart(fig_right, use_container_width=True)

    st.markdown("### Iteration trace (first 12 rows)")
    st.dataframe(df_itr.head(12), use_container_width=True)

    st.caption("Johancsik soft-string (Δs=1 ft): N ≈ w_b·sinθ + T·κ; BF=(65.5−MW)/65.5. "
               "Charts follow Dr-style risk view: μ-sweep (off-bottom) and elemental torque with safety lines. "
               "Trajectory and T&D are linked; 3D schematic uses depth-down convention with cased (blue) vs open-hole (brown).")
