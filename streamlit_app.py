# -*- coding: utf-8 -*-
# PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
# Shared survey state across tabs + azimuth controls + Dr.-style T&D plots.

import math, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Wellpath + Torque & Drag", layout="wide")

# -------------------------- Utils & constants --------------------------
DEG2RAD = math.pi/180.0
def clamp(x,a,b): return max(a, min(b,x))
def in2ft(x_in): return x_in/12.0
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5  # steel ≈ 65.5 ppg
def api_envelope(F_lim, T_lim, n=120):
    F = np.linspace(0, max(F_lim,1.0), n)
    T = T_lim*np.sqrt(np.clip(1.0-(F/max(F_lim,1.0))**2, 0.0, 1.0))
    return F, T

# -------------------------- Standards (subset) --------------------------
CASING_DB = {
    "13-3/8": {"weights": {48.0:12.415, 54.5:12.347, 61.0:12.107}, "grades":["J55","L80","P110"]},
    "9-5/8":  {"weights": {29.3:8.921, 36.0:8.535, 40.0:8.321},  "grades":["J55","L80","P110"]},
    "7":      {"weights": {20.0:6.366, 23.0:6.059, 26.0:5.920},  "grades":["J55","L80","P110"]},
    "5-1/2":  {"weights": {17.0:4.778, 20.0:4.670, 23.0:4.560},  "grades":["J55","L80","P110"]},
}

TOOL_JOINT_DB = {
    'NC38': {'od': 4.75, 'id': 2.25, 'T_makeup_ftlbf': 12000, 'F_tensile_lbf': 350000, 'T_yield_ftlbf': 20000},
    'NC40': {'od': 5.00, 'id': 2.25, 'T_makeup_ftlbf': 16000, 'F_tensile_lbf': 420000, 'T_yield_ftlbf': 25000},
    'NC50': {'od': 6.63, 'id': 3.00, 'T_makeup_ftlbf': 30000, 'F_tensile_lbf': 650000, 'T_yield_ftlbf': 45000},
}

# -------------------------- Session state --------------------------
def init_state():
    s = st.session_state
    s.setdefault("profile", "Build & Hold")
    s.setdefault("kop_md", 1000.0)
    s.setdefault("br_deg_100", 3.0)
    s.setdefault("theta_hold_deg", 30.0)
    s.setdefault("drop_deg_100", 2.0)
    s.setdefault("lateral_ft", 2000.0)
    s.setdefault("target_md", 10000.0)
    s.setdefault("az_deg", 0.0)              # NEW — shared azimuth (deg from North, cw)
    s.setdefault("shoe_md", 3000.0)
    s.setdefault("hole_od_in", 8.5)
    s.setdefault("last_casing_nominal", "7")
    s.setdefault("last_casing_weight", 20.0)

init_state()

def set_and_rerun(key, val):
    st.session_state[key] = val
    st.rerun()  # modern rerun API (experimental_rerun is deprecated)

# -------------------------- Synthetic surveys --------------------------
def synth_build_hold(kop_md, br_deg_100, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md+ds, ds)
    br = br_deg_100/100.0
    inc = np.where(md<=kop_md, 0.0, np.minimum(theta_hold_deg, (md-kop_md)*br))
    az = np.full_like(md, az_deg, dtype=float)
    return md, inc, az

def synth_build_hold_drop(kop_md, br_deg_100, theta_hold_deg, drop_deg_100, target_md, az_deg):
    ds = 1.0
    md = np.arange(0, target_md+ds, ds)
    br = br_deg_100/100.0
    inc = np.where(md<=kop_md, 0.0, np.minimum(theta_hold_deg, (md-kop_md)*br))
    # simple drop over last 25% MD
    start_drop = 0.75*target_md
    dr = drop_deg_100/100.0
    mask = md > start_drop
    inc[mask] = np.maximum(0.0, inc[mask] - (md[mask]-start_drop)*dr)
    az = np.full_like(md, az_deg, dtype=float)
    return md, inc, az

def synth_horizontal(kop_md, br_deg_100, lateral_ft, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0, target_md+ds, ds)
    br = br_deg_100/100.0
    inc = np.where(md<=kop_md, 0.0, np.minimum(theta_max, (md-kop_md)*br))
    idx = np.where(inc >= theta_max-1e-6)[0]
    if len(idx):
        m_h = md[idx[0]]
        md_end = max(m_h + lateral_ft, target_md)
        md = np.arange(0, md_end+ds, ds)
        inc = np.where(md <= m_h, np.minimum(theta_max, np.maximum(0, (md-kop_md)*br)), theta_max)
    az = np.full_like(md, az_deg, dtype=float)
    return md, inc, az

# Minimum Curvature positions + DLS (deg/100 ft)
def mincurv_positions(md, inc_deg, az_deg):
    ds = np.diff(md)
    n = len(md); N = np.zeros(n); E = np.zeros(n); Z = np.zeros(n); DLS = np.zeros(n)
    for i in range(1, n):
        I1, A1 = inc_deg[i-1]*DEG2RAD, az_deg[i-1]*DEG2RAD
        I2, A2 = inc_deg[i]*DEG2RAD,   az_deg[i]*DEG2RAD
        dmd = ds[i-1]
        cos_dl = clamp(math.cos(I1)*math.cos(I2) + math.sin(I1)*math.sin(I2)*math.cos(A2-A1), -1.0, 1.0)
        dpsi = math.acos(cos_dl)
        RF = 1.0 if dpsi < 1e-12 else (2.0/dpsi)*math.tan(dpsi/2.0)
        dN = 0.5*dmd*(math.sin(I1)*math.cos(A1)+math.sin(I2)*math.cos(A2))*RF
        dE = 0.5*dmd*(math.sin(I1)*math.sin(A1)+math.sin(I2)*math.sin(A2))*RF
        dZ = 0.5*dmd*(math.cos(I1)+math.cos(I2))*RF
        N[i], E[i], Z[i] = N[i-1]+dN, E[i-1]+dE, Z[i-1]+dZ
        DLS[i] = (dpsi/DEG2RAD)/dmd*100.0 if dmd>0 else 0.0
    return N, E, Z, DLS

# One place to build the survey from session_state (used by both tabs)
def build_survey_from_state():
    s = st.session_state
    if s.profile == "Build & Hold":
        md, inc, az = synth_build_hold(s.kop_md, s.br_deg_100, s.theta_hold_deg, s.target_md, s.az_deg)
    elif s.profile == "Build–Hold–Drop":
        md, inc, az = synth_build_hold_drop(s.kop_md, s.br_deg_100, s.theta_hold_deg, s.drop_deg_100, s.target_md, s.az_deg)
    else:
        md, inc, az = synth_horizontal(s.kop_md, s.br_deg_100, s.lateral_ft, s.target_md, s.az_deg)
    N, E, TVD, DLS = mincurv_positions(md, inc, az)
    return md, inc, az, N, E, TVD, DLS

# -------------------------- Soft-string (Johancsik) --------------------------
def soft_string_stepper(md, inc_deg, kappa_rad_per_ft, cased_mask,
                        comp_along_depth, comp_props,
                        mu_slide_cased, mu_slide_open, mu_rot_cased, mu_rot_open,
                        mw_ppg, scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0):
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg)
    kappa = np.asarray(kappa_rad_per_ft)
    if kappa.shape[0] == md.shape[0]:
        kappa_seg = kappa[:-1]
    else:
        kappa_seg = kappa
    nseg = md.shape[0]-1
    inc = np.deg2rad(inc_deg[:-1])
    BF = bf_from_mw(mw_ppg)

    # per-segment props
    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s = np.zeros(nseg); mu_r = np.zeros(nseg); cased_seg = np.asarray(cased_mask)[:nseg]
    for i in range(nseg):
        comp = comp_along_depth[i]
        od = comp_props[comp]['od_in']; w_air[i] = comp_props[comp]['w_air_lbft']
        w_b[i] = w_air[i]*BF; r_eff_ft[i] = 0.5*in2ft(od)
        if cased_seg[i]: mu_s[i], mu_r[i] = mu_slide_cased, mu_rot_cased
        else:            mu_s[i], mu_r[i] = mu_slide_open,  mu_rot_open

    # integrate bit->surface
    T = np.zeros(nseg+1); M = np.zeros(nseg+1)
    if scenario == "onbottom": T[0], M[0] = -float(WOB_lbf), float(Mbit_ftlbf)
    sgn_ax = +1.0 if scenario=="pickup" else -1.0 if scenario=="slackoff" else 0.0

    dT = np.zeros(nseg); dM = np.zeros(nseg); Nside = np.zeros(nseg)
    for i in range(nseg):
        Nside[i] = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*Nside[i])*ds
        M_next = M[i] + (mu_r[i]*Nside[i]*r_eff_ft[i])*ds
        dT[i], dM[i] = T_next-T[i], M_next-M[i]
        T[i+1], M[i+1] = T_next, M_next

    df = pd.DataFrame({
        "md_top_ft": md[:-1], "md_bot_ft": md[1:], "ds_ft": np.full(nseg,1.0),
        "inc_deg": inc_deg[:-1], "kappa_rad_ft": kappa_seg,
        "w_air_lbft": w_air, "BF": np.full(nseg, BF), "w_b_lbft": w_b,
        "mu_slide": mu_s, "mu_rot": mu_r, "N_lbf": Nside,
        "dT_lbf": dT, "T_next_lbf": T[1:], "r_eff_ft": r_eff_ft,
        "dM_lbf_ft": dM, "M_next_lbf_ft": M[1:], "cased?": cased_seg,
        "comp": np.asarray(comp_along_depth)[:nseg]
    })
    return df, T, M

# -------------------------- UI --------------------------
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

tab1, tab2 = st.tabs(["Trajectory & 3D schematic", "Torque & Drag"])

# ---- Tab 1: Survey controls (write into session_state) ----
with tab1:
    s = st.session_state
    st.subheader("Synthetic survey (Minimum Curvature)")

    c1, c2, c3, c4 = st.columns(4)
    s.profile = c1.selectbox("Profile", ["Build & Hold","Build–Hold–Drop","Horizontal (build + lateral)"],
                             index=["Build & Hold","Build–Hold–Drop","Horizontal (build + lateral)"].index(s.profile), key="profile")
    s.kop_md = c2.number_input("KOP MD (ft)", 0.0, 50000.0, s.kop_md, 50.0, key="kop_md")

    # NEW — quick azimuth + numeric azimuth, both write the same 'az_deg' key
    def pick_az(preset):
        mapping = {"North (0)":0.0,"East (90)":90.0,"South (180)":180.0,"West (270)":270.0}
        set_and_rerun("az_deg", mapping[preset])
    c3.selectbox("Quick azimuth", ["North (0)","East (90)","South (180)","West (270)"],
                 index={0.0:0,90.0:1,180.0:2,270.0:3}.get(s.az_deg,0), on_change=pick_az,
                 args=("North (0)" if s.az_deg==0 else "East (90)" if s.az_deg==90 else
                       "South (180)" if s.az_deg==180 else "West (270)",))
    s.az_deg = c4.number_input("Azimuth (deg from North, cw)", -360.0, 720.0, s.az_deg, 1.0, key="az_deg")

    c5, c6, c7 = st.columns(3)
    s.br_deg_100 = c5.number_input("Build rate (deg/100 ft)", 0.0, 20.0, s.br_deg_100, 0.1, key="br_deg_100")
    if s.profile == "Build & Hold":
        s.theta_hold_deg = c6.number_input("Final inclination (deg)", 0.0, 90.0, s.theta_hold_deg, 0.5, key="theta_hold_deg")
        s.target_md = c7.number_input("Target MD (ft)", 100.0, 100000.0, s.target_md, 100.0, key="target_md")
    elif s.profile == "Build–Hold–Drop":
        s.theta_hold_deg = c6.number_input("Hold inclination (deg)", 0.0, 90.0, s.theta_hold_deg, 0.5, key="theta_hold_deg")
        s.drop_deg_100   = c7.number_input("Drop rate (deg/100 ft)", 0.0, 20.0, s.drop_deg_100, 0.1, key="drop_deg_100")
        s.target_md = st.number_input("Target MD (ft)", 100.0, 100000.0, s.target_md, 100.0, key="target_md")
    else:
        s.lateral_ft = c6.number_input("Lateral length (ft)", 0.0, 30000.0, s.lateral_ft, 100.0, key="lateral_ft")
        s.target_md  = c7.number_input("Target MD (ft)", 100.0, 100000.0, s.target_md, 100.0, key="target_md")

    md, inc_deg, az, N, E, TVD, DLS = build_survey_from_state()

    # Cased vs open-hole mask for plot (from session shoe_md)
    cased_mask_plot = md <= s.shoe_md

    # 3D plot: depth down (z reversed), cased vs open-hole
    def add_colored_segments(fig, x, y, z, mask, name_true, name_false,
                             color_true="#1f77b4", color_false="#8B4513",
                             width_true=8, width_false=5):
        idx = np.arange(len(x))
        for which, color, width, nm in [(True,color_true,width_true,name_true),
                                        (False,color_false,width_false,name_false)]:
            m = (mask == which)
            if not m.any(): continue
            runs = np.split(idx[m], np.where(np.diff(idx[m]) != 1)[0]+1)
            for run in runs:
                fig.add_trace(go.Scatter3d(
                    x=x[run], y=y[run], z=z[run], mode="lines",
                    line=dict(color=color, width=width), name=nm, showlegend=True
                ))

    fig3d = go.Figure()
    add_colored_segments(fig3d, E, N, TVD, cased_mask_plot, "Cased", "Open-hole")
    fig3d.update_layout(height=520,
        scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)",
                   zaxis=dict(autorange="reversed")),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.02))
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("**Survey and calculated positions (first 11 rows)**")
    survey_df = pd.DataFrame({"MD (ft)":md, "Inc (deg)":inc_deg, "Az (deg)":az,
                              "TVD (ft)":TVD, "North (ft)":N, "East (ft)":E, "DLS (deg/100 ft)":DLS})
    st.dataframe(survey_df.head(11), use_container_width=True)
    st.caption("Minimum Curvature with ratio factor; azimuth is deg from North, clockwise.")

# ---- Tab 2: T&D (reads & writes the same session_state) ----
with tab2:
    s = st.session_state
    st.subheader("Casing / Open-hole (simple)")
    cc1, cc2, cc3, cc4, cc5 = st.columns([1,1,1,1,1])
    s.last_casing_nominal = cc1.selectbox("Last casing nominal OD", list(CASING_DB.keys()),
                                          index=list(CASING_DB.keys()).index(s.last_casing_nominal), key="last_casing_nominal")
    s.last_casing_weight  = cc2.selectbox("lb/ft (standards only)",
                                          list(CASING_DB[s.last_casing_nominal]["weights"].keys()),
                                          index=list(CASING_DB[s.last_casing_nominal]["weights"].keys()).index(s.last_casing_weight),
                                          key="last_casing_weight")
    casing_id_in = CASING_DB[s.last_casing_nominal]["weights"][s.last_casing_weight]
    cc3.text_input("Casing ID (in, locked)", f"{casing_id_in:.3f}", disabled=True)
    s.shoe_md = cc4.number_input("Deepest shoe MD (ft)", 0.0, float(md[-1]), min(s.shoe_md, float(md[-1])), 50.0, key="shoe_md")
    s.hole_od_in = cc5.number_input("Open-hole diameter (in)", 4.0, 20.0, s.hole_od_in, 0.1, key="hole_od_in")

    st.subheader("Soft-string T&D — inputs")
    mu_cased_slide = st.number_input("μ in casing (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_open_slide  = st.number_input("μ in open-hole (sliding)", 0.05, 0.80, 0.35, 0.01)
    mu_cased_rot   = st.number_input("μ in casing (rotating)", 0.05, 0.80, 0.25, 0.01)
    mu_open_rot    = st.number_input("μ in open-hole (rotating)", 0.05, 0.80, 0.35, 0.01)
    mw_ppg         = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    # (Optional) also expose azimuth here using the SAME key so you can adjust from this tab too
    st.markdown("**Well azimuth** (same as survey tab)")
    s.az_deg = st.number_input("Azimuth (deg from North, cw)", -360.0, 720.0, s.az_deg, 1.0, key="az_deg_td")

    st.markdown("**Drillstring (bit up)**")
    d1, d2, d3 = st.columns(3)
    dc_len = d1.number_input("DC length (ft)", 0.0, 10000.0, 600.0, 10.0)
    dc_od  = d2.number_input("DC OD (in)", 3.0, 12.0, 8.0, 0.1)
    dc_id  = d3.number_input("DC ID (in)", 0.5, 6.0, 2.81, 0.01)
    dc_w   = st.number_input("DC weight (air, lb/ft)", 30.0, 150.0, 66.7, 0.1)

    h1, h2, h3 = st.columns(3)
    hwdp_len = h1.number_input("HWDP length (ft)", 0.0, 20000.0, 1000.0, 10.0)
    hwdp_od  = h2.number_input("HWDP OD (in)", 2.0, 8.0, 3.5, 0.1)
    hwdp_id  = h3.number_input("HWDP ID (in)", 0.5, 4.0, 2.0, 0.01)
    hwdp_w   = st.number_input("HWDP weight (air, lb/ft)", 5.0, 40.0, 16.0, 0.1)

    p1, p2, p3 = st.columns(3)
    dp_len = p1.number_input("DP length (ft)", 0.0, 50000.0, max(0.0, float(md[-1]) - (dc_len + hwdp_len)), 10.0)
    dp_od  = p2.number_input("DP OD (in)", 3.0, 6.625, 5.0, 0.01)
    dp_id  = p3.number_input("DP ID (in)", 1.5, 5.0, 4.28, 0.01)
    dp_w   = st.number_input("DP weight (air, lb/ft)", 10.0, 30.0, 19.5, 0.1)

    # Build survey again from session (ensures T&D always uses current trajectory incl. azimuth)
    md, inc_deg, az, N, E, TVD, DLS = build_survey_from_state()
    cased_mask = md <= s.shoe_md
    kappa = (DLS*DEG2RAD)/100.0

    # map components along depth
    nseg = len(md)-1
    comp_along = np.empty(nseg, dtype=object)
    for i in range(nseg):
        dist_up = float(md[-1]) - md[i]
        comp_along[i] = "DC" if dist_up <= dc_len else "HWDP" if dist_up <= dc_len+hwdp_len else "DP"
    comp_props = {"DC":{"od_in":dc_od,"id_in":dc_id,"w_air_lbft":dc_w},
                  "HWDP":{"od_in":hwdp_od,"id_in":hwdp_id,"w_air_lbft":hwdp_w},
                  "DP":{"od_in":dp_od,"id_in":dp_id,"w_air_lbft":dp_w}}

    scen = st.selectbox("Scenario", ["Slack-off (RIH)","Pickup (POOH)","Rotate off-bottom","Rotate on-bottom"])
    scenario = "slackoff" if "Slack-off" in scen else "pickup" if "Pickup" in scen else \
               "rotate_off" if "Rotate off-bottom" in scen else "onbottom"
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

    # Doctor-style risk plots (defaults)
    tj_name = st.selectbox("Tool-joint size", list(TOOL_JOINT_DB.keys()), index=2)
    sf_joint = st.number_input("Safety factor (tool-joint)", 1.00, 2.00, 1.10, 0.05)
    rig_torque = st.number_input("Top-drive torque limit (lbf-ft)", 10000, 150000, 60000, 1000)
    mu_band = st.multiselect("μ sweep for off-bottom risk curves", [0.15,0.20,0.25,0.30,0.35,0.40],
                             default=[0.20,0.25,0.30,0.35])

    tj = TOOL_JOINT_DB[tj_name]
    T_makeup_sf = tj['T_makeup_ftlbf']/sf_joint
    T_yield_sf  = tj['T_yield_ftlbf']/sf_joint
    F_tens_sf   = tj['F_tensile_lbf']/sf_joint
    _, _ = api_envelope(F_tens_sf, T_yield_sf, n=120)

    def run_offbottom(mu):
        df_tmp, _, _ = soft_string_stepper(
            md, inc_deg, kappa, cased_mask, comp_along, comp_props,
            mu, mu, mu, mu, mw_ppg, scenario="rotate_off", WOB_lbf=0.0, Mbit_ftlbf=0.0
        )
        return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy())

    # Left: μ curves (off-bottom torque risk)
    fig_left = go.Figure()
    for mu in mu_band:
        d_mu, t_mu = run_offbottom(mu)
        fig_left.add_trace(go.Scatter(x=t_mu/1000.0, y=d_mu, mode="lines", name=f"μ={mu:.2f}"))
    fig_left.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash",
                       annotation_text="Make-up / SF", annotation_position="top")
    fig_left.add_vline(x=rig_torque/1000.0, line_color="magenta", line_dash="dot",
                       annotation_text="Top-drive", annotation_position="top")
    fig_left.update_xaxes(title_text="Off-bottom Torque (k lbf-ft)")
    fig_left.update_yaxes(title_text="Depth (ft)", autorange="reversed")
    fig_left.update_layout(height=700, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))

    # Right: elemental torque contribution + limits
    comp = df_itr['comp'].to_numpy(); dM = df_itr['dM_lbf_ft'].to_numpy()
    tor_dc = np.cumsum(np.where(comp=="DC", dM, 0.0))
    tor_hw = np.cumsum(np.where(comp=="HWDP", dM, 0.0))
    tor_dp = np.cumsum(np.where(comp=="DP", dM, 0.0))
    tor_total = np.cumsum(dM)
    fig_right = go.Figure()
    fig_right.add_trace(go.Scatter(x=tor_dc/1000.0,   y=depth, name="DC",   mode="lines"))
    fig_right.add_trace(go.Scatter(x=tor_hw/1000.0,   y=depth, name="HWDP", mode="lines"))
    fig_right.add_trace(go.Scatter(x=tor_dp/1000.0,   y=depth, name="DP",   mode="lines"))
    fig_right.add_trace(go.Scatter(x=tor_total/1000.0,y=depth, name="Total", mode="lines", line=dict(width=3)))
    fig_right.add_vline(x=T_makeup_sf/1000.0, line_color="#00d5ff", line_dash="dash",
                        annotation_text="Make-up / SF")
    fig_right.add_vline(x=rig_torque/1000.0, line_color="magenta", line_dash="dot",
                        annotation_text="Top-drive")
    fig_right.update_xaxes(title_text="Elemental Torque (k lbf-ft)")
    fig_right.update_yaxes(title_text="Depth (ft)", autorange="reversed")
    fig_right.update_layout(height=700, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))

    cL, cR = st.columns(2)
    cL.plotly_chart(fig_left,  use_container_width=True)
    cR.plotly_chart(fig_right, use_container_width=True)

    st.markdown("### Iteration trace (first 12 rows)")
    st.dataframe(df_itr.head(12), use_container_width=True)
    st.caption("Trajectory and friction are linked via shared session state; "
               "survey recomputes with azimuth & geometry on every rerun. "
               "State across tabs uses Streamlit session_state & rerun callbacks.")
