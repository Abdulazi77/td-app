# -*- coding: utf-8 -*-
# PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
# Single-page app with live-linked survey → casing/open-hole → T&D.

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

def bf_from_mw(mw_ppg):
    # buoyancy factor for steel ≈ 65.5 ppg
    return (65.5 - mw_ppg)/65.5

def mincurv_positions(md, inc_deg, az_deg):
    """Minimum curvature with ratio factor; returns N,E,TVD,DLS (deg/100 ft)."""
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

# ------------------------------ simple synthetic surveys -----------------------
def synth_build_hold(kop_md, build_rate, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    inc = np.zeros_like(md)
    az  = np.full_like(md, az_deg, dtype=float)
    brad = build_rate/100.0
    after_kop = np.maximum(0.0, md - kop_md)
    inc = np.minimum(theta_hold_deg, after_kop*brad)
    return md, inc, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    inc = np.zeros_like(md)
    az  = np.full_like(md, az_deg, dtype=float)
    br = build_rate/100.0
    dr = drop_rate/100.0
    after_kop = np.maximum(0.0, md - kop_md)
    inc = np.minimum(theta_hold_deg, after_kop*br)
    # drop in last 25% of MD
    start_drop = 0.75*target_md
    drop_amt = np.maximum(0.0, md - start_drop)*dr
    inc = np.maximum(0.0, inc - drop_amt)
    return md, inc, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0, target_md + ds, ds)
    br = build_rate/100.0
    inc = np.minimum(theta_max, np.maximum(0.0, md - kop_md)*br)
    # extend lateral if needed
    idx = np.where(inc >= theta_max - 1e-6)[0]
    if len(idx):
        m_h = md[idx[0]]
        md_end = max(target_md, m_h + lateral_length)
        md = np.arange(0, md_end + ds, ds)
        inc = np.where(md <= m_h, np.minimum(theta_max, np.maximum(0.0, md - kop_md)*br), theta_max)
    az = np.full_like(md, az_deg, dtype=float)
    return md, inc, az

# ------------------------------ soft-string (Johancsik) ------------------------
def soft_string_stepper(md, inc_deg, kappa_rad_per_ft, cased_mask,
                        comp_along_depth, comp_props,
                        mu_slide_cased, mu_slide_open, mu_rot_cased, mu_rot_open,
                        mw_ppg, scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0):
    ds = 1.0
    md = np.asarray(md)
    inc_deg = np.asarray(inc_deg)
    nseg = len(md) - 1
    inc = np.deg2rad(inc_deg[:-1])
    # per-segment curvature
    kappa_seg = kappa_rad_per_ft[:-1] if len(kappa_rad_per_ft)==len(md) else kappa_rad_per_ft
    # arrays
    r_eff_ft = np.zeros(nseg)
    w_air = np.zeros(nseg)
    w_b   = np.zeros(nseg)
    mu_s  = np.zeros(nseg)
    mu_r  = np.zeros(nseg)
    BF = bf_from_mw(mw_ppg)

    cased_seg = np.asarray(cased_mask)[:nseg]
    for i in range(nseg):
        comp = comp_along_depth[i]
        od_in = comp_props[comp]['od_in']
        id_in = comp_props[comp]['id_in']
        w_air_ft = comp_props[comp]['w_air_lbft']
        w_air[i] = w_air_ft
        w_b[i]   = w_air_ft * BF
        r_eff_ft[i] = 0.5 * od_in * IN2FT
        if cased_seg[i]:
            mu_s[i] = mu_slide_cased; mu_r[i] = mu_rot_cased
        else:
            mu_s[i] = mu_slide_open;  mu_r[i] = mu_rot_open

    T = np.zeros(nseg+1)   # axial; index 0 at bit
    M = np.zeros(nseg+1)   # torque
    dT = np.zeros(nseg); dM = np.zeros(nseg); N_side = np.zeros(nseg)

    if scenario == "onbottom":
        T[0] = -float(WOB_lbf); M[0] = float(Mbit_ftlbf)

    sgn_ax = +1.0 if scenario=="pickup" else -1.0 if scenario=="slackoff" else 0.0

    for i in range(nseg):
        N_side[i] = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*N_side[i])*ds
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
        "comp": np.asarray(comp_along_depth)[:nseg]
    })
    return df, T, M

# ------------------------------ minimal casing table ---------------------------
CASING_DB = {
    "13-3/8": {"weights": {48.0: 12.415, 54.5: 12.347, 61.0: 12.107}},
    "9-5/8":  {"weights": {29.3: 8.921, 36.0: 8.535, 40.0: 8.321}},
    "7":      {"weights": {20.0: 6.366, 23.0: 6.059, 26.0: 5.920}},
    "5-1/2":  {"weights": {17.0: 4.778, 20.0: 4.670, 23.0: 4.560}},
}

# ------------------------------ UI starts -------------------------------------
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

# ---- TRAJECTORY (drives everything) ------------------------------------------
with st.expander("① Trajectory (minimum curvature) — drives casing & T&D", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
    kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
    build   = c3.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.0, 0.1)
    az_deg  = c4.number_input("Azimuth (deg, clockwise from North)", 0.0, 360.0, 0.0, 1.0)

    r1, r2, r3 = st.columns(3)
    if profile == "Build & Hold":
        theta_hold = r1.number_input("Final inclination (deg)", 0.0, 90.0, 38.5, 0.5)
        target_md  = r2.number_input("Target MD (ft)", 100.0, 50000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_build_hold(kop_md, build, theta_hold, target_md, az_deg)
    elif profile == "Build–Hold–Drop":
        theta_hold = r1.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 0.5)
        drop_rate  = r2.number_input("Drop rate (deg/100 ft)", 0.0, 30.0, 2.0, 0.1)
        target_md  = r3.number_input("Target MD (ft)", 100.0, 50000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_build_hold_drop(kop_md, build, theta_hold, drop_rate, target_md, az_deg)
    else:
        lateral    = r1.number_input("Lateral length (ft)", 0.0, 50000.0, 2000.0, 50.0)
        target_md  = r2.number_input("Target MD (ft)", 100.0, 50000.0, 10000.0, 100.0)
        md, inc_deg, az = synth_horizontal(kop_md, build, lateral, target_md, az_deg)

    N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

    # store in session so later widgets always "see" the latest survey
    st.session_state['survey'] = dict(md=md, inc=inc_deg, az=az, N=N, E=E, TVD=TVD, DLS=DLS)

    # quick mini-plot 3D skeleton (colored later after shoe is chosen)
    fig3d = go.Figure()
    fig3d.update_layout(height=420, scene=dict(
        xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)",
        zaxis=dict(autorange="reversed")
    ), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3d, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "MD (ft)": md[:11], "Inc (deg)": inc_deg[:11], "Az (deg)": az[:11],
        "TVD (ft)": TVD[:11], "North (ft)": N[:11], "East (ft)": E[:11], "DLS (deg/100 ft)": DLS[:11]
    }), use_container_width=True)

# ---- CASING / OPEN-HOLE (linked to current MD) --------------------------------
with st.expander("② Casing / open-hole (simple, last string + OH)", expanded=True):
    md = st.session_state['survey']['md']  # latest from step ①
    md_end = float(md[-1])

    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    nominal = c1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), index=1)  # default 9-5/8
    weight  = c2.selectbox("lb/ft (standards)", list(CASING_DB[nominal]["weights"].keys()))
    casing_id_in = CASING_DB[nominal]["weights"][weight]
    c3.text_input("Casing ID (in, locked)", f"{casing_id_in:.3f}", disabled=True)

    # keep a persistent shoe depth; clamp if survey changed
    if 'shoe_md' not in st.session_state:
        st.session_state['shoe_md'] = min(3000.0, md_end)
    st.session_state['shoe_md'] = clamp(st.session_state['shoe_md'], 0.0, md_end)

    st.session_state['shoe_md'] = c4.slider("Shoe MD (ft)", 0.0, md_end, float(st.session_state['shoe_md']), 50.0)
    hole_diam_in = c5.number_input("Open-hole diameter (in)", 4.0, 20.0, 8.50, 0.01)

    # cased above shoe, open-hole below
    shoe_md = st.session_state['shoe_md']
    cased_mask = md <= shoe_md

    # update 3D plot with colored segments
    N = st.session_state['survey']['N']; E = st.session_state['survey']['E']; TVD = st.session_state['survey']['TVD']
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

# ---- T&D (uses the current survey + casing mask) ------------------------------
with st.expander("③ Soft-string Torque & Drag (Johancsik) — linked to the survey above", expanded=True):
    inc_deg = st.session_state['survey']['inc']; az = st.session_state['survey']['az']
    _, _, _, DLS = st.session_state['survey']['DLS'], st.session_state['survey']['DLS'], st.session_state['survey']['DLS'], st.session_state['survey']['DLS']
    # curvature (rad/ft) from DLS
    kappa = (st.session_state['survey']['DLS']*DEG2RAD)/100.0

    # friction & mud
    f1, f2, f3, f4, f5 = st.columns(5)
    mu_cased_slide = f1.number_input("μ in casing (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_open_slide  = f2.number_input("μ in open-hole (sliding)", 0.05, 0.80, 0.35, 0.01)
    mu_cased_rot   = f3.number_input("μ in casing (rotating)", 0.05, 0.80, 0.25, 0.01)
    mu_open_rot    = f4.number_input("μ in open-hole (rotating)", 0.05, 0.80, 0.35, 0.01)
    mw_ppg         = f5.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    # simple drillstring (bit up)
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

    # map component along depth (exactly nseg entries)
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

    # Classic “Doctor-style” plots (depth down)
    fig_left = go.Figure(go.Scatter(x=np.abs(df_itr["M_next_lbf_ft"])/1000.0, y=depth,
                                    mode="lines", name="Torque"))
    fig_left.update_yaxes(autorange="reversed", title="Depth (ft)")
    fig_left.update_xaxes(title="Elemental torque (k lbf-ft)")
    fig_left.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))

    fig_right = go.Figure(go.Scatter(x=np.maximum(0.0, -df_itr["T_next_lbf"])/1000.0, y=depth,
                                     mode="lines", name="Hookload"))
    fig_right.update_yaxes(autorange="reversed", title="Depth (ft)")
    fig_right.update_xaxes(title="Hookload (k lbf)")
    fig_right.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))

    cL, cR = st.columns(2)
    with cL: st.plotly_chart(fig_left, use_container_width=True)
    with cR: st.plotly_chart(fig_right, use_container_width=True)

    st.markdown("**Iteration trace (first 12 rows)**")
    st.dataframe(df_itr.head(12), use_container_width=True)

st.caption("Everything above is live-linked: Changing the survey immediately changes the casing shoe range and the T&D results.")
