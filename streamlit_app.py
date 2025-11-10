# -*- coding: utf-8 -*-
# PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
# One-tab, fully linked (survey → casing/open-hole → T&D).
# Plots/behavior are the SAME as your last good version.

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")

# ------------------------------ constants/helpers ------------------------------
DEG2RAD = math.pi/180.0
IN2FT   = 1.0/12.0

def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5  # steel ~65.5 ppg

def I_moment(od_in, id_in):
    return (math.pi/64.0)*(od_in**4 - id_in**4)  # in^4

# ------------------------------ minimal casing table ---------------------------
CASING_DB = {
    "13-3/8": {"weights": {48.0: 12.415, 54.5: 12.347, 61.0: 12.107}},
    "9-5/8":  {"weights": {29.3: 8.921, 36.0: 8.535, 40.0: 8.321}},
    "7":      {"weights": {20.0: 6.366, 23.0: 6.059, 26.0: 5.920}},
    "5-1/2":  {"weights": {17.0: 4.778, 20.0: 4.670, 23.0: 4.560}},
}

TOOL_JOINT_DB = {
    'NC38': {'od': 4.75, 'id': 2.25, 'T_makeup_ftlbf': 12000, 'F_tensile_lbf': 350000, 'T_yield_ftlbf': 20000},
    'NC40': {'od': 5.00, 'id': 2.25, 'T_makeup_ftlbf': 16000, 'F_tensile_lbf': 420000, 'T_yield_ftlbf': 25000},
    'NC50': {'od': 6.63, 'id': 3.00, 'T_makeup_ftlbf': 30000, 'F_tensile_lbf': 650000, 'T_yield_ftlbf': 45000},
}

# ------------------------------ survey builders --------------------------------
def synth_build_hold(kop_md, build_rate_deg_per_100ft, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md-kop_md)*(build_rate_deg_per_100ft/100.0))
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    br = build_rate/100.0
    dr = drop_rate/100.0
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md-kop_md)*br)
    start_drop = 0.75*target_md
    theta = np.maximum(0.0, theta - np.maximum(0.0, md-start_drop)*dr)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    br = build_rate/100.0
    theta = np.minimum(theta_max, np.maximum(0.0, md-kop_md)*br)
    idx = np.where(theta >= theta_max-1e-6)[0]
    if len(idx):
        m_h = md[idx[0]]
        md_end = max(target_md, m_h + lateral_length)
        md = np.arange(0, md_end + ds, ds)
        theta = np.where(md <= m_h, np.minimum(theta_max, np.maximum(0.0, md-kop_md)*br), theta_max)
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

# ------------------------------ soft-string (Johancsik) ------------------------
def soft_string_stepper(md, inc_deg, kappa_rad_per_ft, cased_mask,
                        comp_along_depth, comp_props,
                        mu_slide_cased, mu_slide_open, mu_rot_cased, mu_rot_open,
                        mw_ppg, scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0):
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg)
    nseg = len(md) - 1
    inc = np.deg2rad(inc_deg[:-1])

    kappa_seg = kappa_rad_per_ft[:-1] if len(kappa_rad_per_ft)==len(md) else kappa_rad_per_ft

    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s = np.zeros(nseg); mu_r = np.zeros(nseg); BF = bf_from_mw(mw_ppg)

    cased_seg = np.asarray(cased_mask)[:nseg]
    for i in range(nseg):
        comp = comp_along_depth[i]
        od_in = comp_props[comp]['od_in']; id_in = comp_props[comp]['id_in']; w_air_ft = comp_props[comp]['w_air_lbft']
        w_air[i] = w_air_ft; w_b[i] = w_air_ft*BF
        r_eff_ft[i] = 0.5*od_in*IN2FT
        if cased_seg[i]: mu_s[i] = mu_slide_cased; mu_r[i] = mu_rot_cased
        else:           mu_s[i] = mu_slide_open;  mu_r[i] = mu_rot_open

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)  # 0 at bit
    dT = np.zeros(nseg);  dM = np.zeros(nseg);  N_side = np.zeros(nseg)
    if scenario == "onbottom": T[0] = -float(WOB_lbf); M[0] = float(Mbit_ftlbf)
    sgn_ax = +1.0 if scenario=="pickup" else -1.0 if scenario=="slackoff" else 0.0

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
        "comp": np.asarray(comp_along_depth)[:nseg]
    })
    return df, T, M

# ------------------------------ API 7G envelope (surrogate) --------------------
def api7g_envelope_points(F_lim, T_lim, n=80):
    F = np.linspace(0, max(F_lim, 1.0), n)
    T = T_lim*np.sqrt(np.clip(1.0 - (F/max(F_lim,1.0))**2, 0.0, 1.0))
    return F, T

def Fs_sinusoidal(Epsi, Iin4, w_b_lbf_ft, inc_deg, clearance_ft):
    theta = np.deg2rad(np.maximum(0.0, inc_deg))
    r = np.maximum(1e-6, clearance_ft)
    return 2.0*np.sqrt(Epsi*Iin4 * w_b_lbf_ft*np.sin(theta)/r)

def Fh_helical(Fs): return 1.6*Fs

# ------------------------------ one tab, everything inside ---------------------
(tab,) = st.tabs(["Wellpath + Torque & Drag (linked)"])

with tab:
    # ---------- TRAJECTORY ----------
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

    # store survey so later widgets read the latest values
    st.session_state['survey'] = dict(md=md, inc=inc_deg, az=az, N=N, E=E, TVD=TVD, DLS=DLS)

    # ---------- CASING / OPEN-HOLE (linked) ----------
    st.subheader("Casing / Open-hole (simple, last string + open hole)")
    md_end = float(md[-1])
    cc1, cc2, cc3, cc4, cc5 = st.columns([1,1,1,1,1])
    nominal = cc1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), index=1)  # default 9-5/8
    weight  = cc2.selectbox("lb/ft (standards only)", list(CASING_DB[nominal]["weights"].keys()))
    casing_id_in = CASING_DB[nominal]["weights"][weight]
    cc3.text_input("Casing ID (in, locked)", f"{casing_id_in:.3f}", disabled=True)

    # persistent shoe depth that clamps if trajectory changes
    if 'shoe_md' not in st.session_state:
        st.session_state['shoe_md'] = min(3000.0, md_end)
    st.session_state['shoe_md'] = clamp(st.session_state['shoe_md'], 0.0, md_end)
    st.session_state['shoe_md'] = cc4.slider("Shoe MD (ft)", 0.0, md_end, float(st.session_state['shoe_md']), 50.0)

    hole_diam_in = cc5.number_input("Open-hole diameter (in)", 4.0, 20.0, 8.50, 0.01)
    shoe_md = st.session_state['shoe_md']
    cased_mask = md <= shoe_md

    # 3D split by shoe (same look as before)
    idx = int(np.searchsorted(md, shoe_md, side='right'))
    fig3d = go.Figure()
    if idx > 1:
        fig3d.add_trace(go.Scatter3d(x=E[:idx], y=N[:idx], z=-TVD[:idx],
                                     mode="lines", line=dict(width=6, color="#4cc9f0"), name="Cased"))
    if idx < len(md):
        fig3d.add_trace(go.Scatter3d(x=E[idx-1:], y=N[idx-1:], z=-TVD[idx-1:],
                                     mode="lines", line=dict(width=4, color="#a97142"), name="Open-hole"))
    fig3d.update_layout(height=420, scene=dict(
        xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)",
        zaxis=dict(autorange="reversed")
    ), legend=dict(orientation="h"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3d, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "MD (ft)": md[:11], "Inc (deg)": inc_deg[:11], "Az (deg)": az[:11],
        "TVD (ft)": TVD[:11], "North (ft)": N[:11], "East (ft)": E[:11],
        "DLS (deg/100 ft)": DLS[:11]
    }), use_container_width=True)

    # ---------- T&D (same plots/logic as before) ----------
    st.subheader("Soft-string Torque & Drag — Johancsik (linked to survey above)")
    mu_cased_slide = st.number_input("μ in casing (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_open_slide  = st.number_input("μ in open-hole (sliding)", 0.05, 0.80, 0.35, 0.01)
    mu_cased_rot   = st.number_input("μ in casing (rotating)", 0.05, 0.80, 0.25, 0.01)
    mu_open_rot    = st.number_input("μ in open-hole (rotating)", 0.05, 0.80, 0.35, 0.01)
    mw_ppg         = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    st.markdown("**Drillstring (bit up)**")
    d1, d2, d3 = st.columns(3)
    dc_len = d1.number_input("DC length (ft)", 0.0, 10000.0, 600.0, 10.0)
    hwdp_len = d2.number_input("HWDP length (ft)", 0.0, 20000.0, 1000.0, 10.0)
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

    # curvature from DLS
    kappa = (DLS*DEG2RAD)/100.0

    simple_mode = st.checkbox("Use classic simple view (hide safety overlays & μ-sweep)", value=True)

    scen = st.selectbox("Scenario", ["Slack-off (RIH)","Pickup (POOH)","Rotate off-bottom","Rotate on-bottom"])
    scenario = {"Slack-off (RIH)":"slackoff","Pickup (POOH)":"pickup",
                "Rotate off-bottom":"rotate_off","Rotate on-bottom":"onbottom"}[scen]
    wob  = st.number_input("WOB (lbf) for on-bottom", 0.0, 100000.0, 6000.0, 100.0)
    mbit = st.number_input("Bit torque (lbf-ft) for on-bottom", 0.0, 50000.0, 0.0, 100.0)

    df_itr, T_arr, M_arr = soft_string_stepper(
        md, inc_deg, kappa, cased_mask, comp_along, comp_props,
        mu_cased_slide, mu_open_slide, mu_cased_rot, mu_open_rot,
        mw_ppg, scenario=scenario, WOB_lbf=wob, Mbit_ftlbf=mbit
    )

    depth = df_itr["md_bot_ft"].to_numpy()
    surf_hookload = max(0.0, -T_arr[-1]); surf_torque = abs(M_arr[-1])
    st.success(f"Surface hookload: {surf_hookload:,.0f} lbf — Surface torque: {surf_torque:,.0f} lbf-ft")

    if simple_mode:
        figT = go.Figure(go.Scatter(x=df_itr["md_bot_ft"], y=np.abs(df_itr["M_next_lbf_ft"]),
                                    mode="lines", name="Torque"))
        figT.update_xaxes(title_text="MD (ft)"); figT.update_yaxes(title_text="Torque (lbf-ft)")
        figH = go.Figure(go.Scatter(x=df_itr["md_bot_ft"], y=np.maximum(0.0, -df_itr["T_next_lbf"]),
                                    mode="lines", name="Hookload"))
        figH.update_xaxes(title_text="MD (ft)"); figH.update_yaxes(title_text="Hookload (lbf)")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(figT, use_container_width=True)
        with c2: st.plotly_chart(figH, use_container_width=True)

    if not simple_mode:
        st.subheader("Safety & limits")
        s1, s2, s3 = st.columns(3)
        tj_name = s1.selectbox("Tool-joint size", list(TOOL_JOINT_DB.keys()), index=2)
        sf_joint = s2.number_input("Safety factor (tool-joint)", 1.00, 2.00, 1.10, 0.05)
        sf_tension = s3.number_input("SF for pipe body tension", 1.00, 2.00, 1.15, 0.05)

        rig_torque_lim = st.number_input("Top-drive torque limit (lbf-ft)", 10000, 150000, 60000, 1000)
        rig_pull_lim   = st.number_input("Rig max hookload (lbf)", 50000, 1500000, 500000, 5000)

        mu_band = st.multiselect("μ sweep for off-bottom risk curves",
                                 [0.15,0.20,0.25,0.30,0.35,0.40],
                                 default=[0.20,0.25,0.30,0.35])

        tj = TOOL_JOINT_DB[tj_name]
        T_makeup_sf = tj['T_makeup_ftlbf']/sf_joint
        T_yield_sf  = tj['T_yield_ftlbf']/sf_joint
        F_tensile_sf= tj['F_tensile_lbf']/sf_joint
        F_env, T_env = api7g_envelope_points(F_tensile_sf, T_yield_sf, n=100)

        def run_td_off_bottom(mu):
            df_tmp, _, _ = soft_string_stepper(
                md, inc_deg, kappa, cased_mask, comp_along, comp_props,
                mu, mu, mu, mu, mw_ppg, scenario="rotate_off", WOB_lbf=0.0, Mbit_ftlbf=0.0
            )
            return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy())

        fig_left = go.Figure()
        for mu in mu_band:
            dmu, tmu = run_td_off_bottom(mu)
            fig_left.add_trace(go.Scatter(x=tmu/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"))
        fig_left.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash",
                           annotation_text="Make-up torque / SF", annotation_position="top")
        fig_left.update_yaxes(autorange="reversed", title_text="Depth (ft)")
        fig_left.update_xaxes(title_text="Off-bottom torque (k lbf-ft)")
        fig_left.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

        comp = df_itr['comp'].to_numpy(); dM = df_itr['dM_lbf_ft'].to_numpy()
        tor_dc   = np.cumsum(np.where(comp=='DC',   dM, 0.0))
        tor_hwdp = np.cumsum(np.where(comp=='HWDP', dM, 0.0))
        tor_dp   = np.cumsum(np.where(comp=='DP',   dM, 0.0))
        tor_total= np.cumsum(dM)
        hookload = np.maximum(0.0, -df_itr['T_next_lbf'].to_numpy())

        fig_right = go.Figure()
        fig_right.add_trace(go.Scatter(x=tor_dc/1000.0,   y=depth, name="DC",   mode="lines"))
        fig_right.add_trace(go.Scatter(x=tor_hwdp/1000.0, y=depth, name="HWDP", mode="lines"))
        fig_right.add_trace(go.Scatter(x=tor_dp/1000.0,   y=depth, name="DP",   mode="lines"))
        fig_right.add_trace(go.Scatter(x=tor_total/1000.0,y=depth, name="Total", mode="lines", line=dict(width=3)))
        fig_right.add_vline(x=T_makeup_sf/1000.0,  line_color="#00d5ff", line_dash="dash", annotation_text="Make-up / SF")
        fig_right.add_vline(x=rig_torque_lim/1000.0, line_color="magenta", line_dash="dot", annotation_text="Top-drive limit")
        fig_right.update_yaxes(autorange="reversed", title_text="Depth (ft)")
        fig_right.update_xaxes(title_text="Elemental torque (k lbf-ft)")
        fig_right.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

        st.markdown("### T&D Model Chart — Risk curves and limits")
        cL, cR = st.columns(2)
        with cL: st.plotly_chart(fig_left, use_container_width=True)
        with cR: st.plotly_chart(fig_right, use_container_width=True)

        # Envelope & Hookload diagnostics (same as before)
        Epsi = 30.0e6
        Iin4 = I_moment(dp_od, dp_id)
        rbore_ft = 0.5*hole_diam_in*IN2FT; rpipe_ft = 0.5*dp_od*IN2FT
        clearance_ft = max(1e-3, rbore_ft - rpipe_ft)
        Fs = Fs_sinusoidal(Epsi, Iin4, df_itr['w_b_lbft'].to_numpy(), df_itr['inc_deg'].to_numpy(), clearance_ft)
        Fh = Fh_helical(Fs)

        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(x=(T_env)/1000.0, y=(F_env)/1000.0, mode="lines", name="API 7G envelope (approx)"))
        fig_env.add_vline(x=T_makeup_sf/1000.0,  line_color="#00d5ff", line_dash="dash", annotation_text="Make-up / SF")
        fig_env.add_vline(x=rig_torque_lim/1000.0, line_color="magenta", line_dash="dot", annotation_text="Top-drive limit")
        fig_env.add_trace(go.Scatter(x=[abs(tor_total[-1])/1000.0], y=[hookload[-1]/1000.0], mode="markers",
                                     name="Operating point", marker=dict(size=10, color="orange")))
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

    st.markdown("### Iteration trace (first 12 rows)")
    st.dataframe(df_itr.head(12), use_container_width=True)

    st.caption("Johancsik soft-string (Δs = 1 ft). Survey → casing shoe → T&D are linked. Defaults: last casing 9-5/8, OH 8.50 in.")
