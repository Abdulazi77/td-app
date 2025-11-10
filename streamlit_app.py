# td_app.py
# Wellpath + Torque&Drag (Δs = 1 ft) — three synthetic profiles, classic plots restored,
# depth-down 3D, brown open-hole vs blue cased, safe session_state usage.

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------- session_state helpers ---------------------------
def get_state(key, default=None):
    return st.session_state.get(key, default)

def set_az_from_quick():
    mapping = {"North (0)": 0.0, "East (90)": 90.0, "South (180)": 180.0, "West (270)": 270.0}
    sel = st.session_state.get("quick_az", "North (0)")
    st.session_state["az_deg"] = mapping.get(sel, 0.0)
    st.rerun()

def push_td_az_into_trajectory():
    st.session_state["az_deg"] = st.session_state.get("az_deg_td", 0.0)
    st.rerun()

# ------------------------------ standards -----------------------------------
CASING_DB = {
    "13-3/8": {48.0: 12.415, 54.5: 12.347, 61.0: 12.300},
    "9-5/8":  {36.0:  8.921, 40.0:  8.535, 43.5:  8.347},
    "7":      {20.0:  6.538, 23.0:  6.276, 26.0:  6.094, 29.0:  5.921},
}

# --------------------------- geometry utilities -----------------------------
def rad(d): return np.deg2rad(d)
def deg(r): return np.rad2deg(r)

def minimum_curvature(md, inc_deg, az_deg):
    md = np.asarray(md, float)
    inc = rad(np.asarray(inc_deg, float))
    az  = rad(np.asarray(az_deg,  float))

    n = len(md)
    tvd = np.zeros(n); north = np.zeros(n); east = np.zeros(n)
    for i in range(1, n):
        dmd = md[i]-md[i-1]
        inc1, inc2 = inc[i-1], inc[i]
        az1,  az2  = az[i-1],  az[i]
        cos_dl = np.clip(np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1)+np.cos(inc1)*np.cos(inc2), -1.0, 1.0)
        dl = np.arccos(cos_dl)
        rf = 1.0 if dl < 1e-12 else (2/dl)*np.tan(dl/2)

        tvd[i]   = tvd[i-1]   + 0.5*dmd*(np.cos(inc1)+np.cos(inc2))*rf
        north[i] = north[i-1] + 0.5*dmd*(np.sin(inc1)*np.cos(az1)+np.sin(inc2)*np.cos(az2))*rf
        east[i]  = east[i-1]  + 0.5*dmd*(np.sin(inc1)*np.sin(az1)+np.sin(inc2)*np.sin(az2))*rf

    dls = np.zeros(n)
    for i in range(1, n):
        dmd = md[i]-md[i-1]
        if dmd > 0:
            dls[i] = deg(np.arccos(np.sin(inc[i-1])*np.sin(inc[i])*np.cos(az[i]-az[i-1]) + np.cos(inc[i-1])*np.cos(inc[i]))) * 100/dmd
    return tvd, north, east, dls

# ------------------------ synthetic profiles (Δs=1 ft) ----------------------
def synth_build_hold(kop, br_deg100, theta_hold, target_md, az_deg, ds=1.0):
    md = np.arange(0.0, float(target_md)+ds, ds)
    inc = np.zeros_like(md)
    az  = np.zeros_like(md) + float(az_deg)
    br_perft = br_deg100/100.0
    for i, m in enumerate(md):
        if m < kop: inc[i] = 0.0
        else:       inc[i] = min(theta_hold, max(0.0, (m-kop)*br_perft))
    tvd, n, e, dls = minimum_curvature(md, inc, az)
    return md, inc, az, tvd, n, e, dls

def synth_build_hold_drop(kop, br_deg100, theta_hold, drop_start_md, drop_deg100, final_inc, target_md, az_deg, ds=1.0):
    md = np.arange(0.0, float(target_md)+ds, ds)
    inc = np.zeros_like(md)
    az  = np.zeros_like(md) + float(az_deg)
    br_perft   = br_deg100/100.0
    drop_perft = drop_deg100/100.0  # positive rate; we subtract during drop
    for i, m in enumerate(md):
        if m < kop:
            inc[i] = 0.0
        elif m < drop_start_md:
            inc[i] = min(theta_hold, (m-kop)*br_perft)
        else:
            # start dropping from whatever angle we reached at drop_start_md
            inc_drop0 = min(theta_hold, (drop_start_md-kop)*br_perft)
            inc[i] = max(final_inc, inc_drop0 - (m-drop_start_md)*drop_perft)
    tvd, n, e, dls = minimum_curvature(md, inc, az)
    return md, inc, az, tvd, n, e, dls

def synth_horizontal(kop, br_deg100, lateral_length, target_md, az_deg, ds=1.0):
    md = np.arange(0.0, float(target_md)+ds, ds)
    inc = np.zeros_like(md)
    az  = np.zeros_like(md) + float(az_deg)
    br_perft = br_deg100/100.0
    # build to 90 if possible, then hold (lateral)
    for i, m in enumerate(md):
        if m < kop:
            inc[i] = 0.0
        else:
            inc[i] = min(90.0, (m-kop)*br_perft)
    # ensure there is at least `lateral_length` at near-horizontal if target allows
    # (this simply shapes the curve; detailed lateral control would use segmenting)
    tvd, n, e, dls = minimum_curvature(md, inc, az)
    return md, inc, az, tvd, n, e, dls

# ------------------------ Soft-string T&D stepper ---------------------------
def soft_string_stepper(md, inc_deg, dls_deg100, cased_mask, mw_ppg,
                        casing_id_in, open_hole_diam_in,
                        mu_c_slide, mu_oh_slide, mu_c_rot, mu_oh_rot,
                        dc_len, dc_od, dc_w_air,
                        hwdp_len, hwdp_od, hwdp_w_air,
                        dp_len, dp_od, dp_w_air,
                        scenario="Slack-off (RIH)", wob_lbf=0.0, bit_tq=0.0):
    BF = (65.5 - float(mw_ppg))/65.5

    md = np.asarray(md, float)
    inc = rad(np.asarray(inc_deg, float))
    dls = np.asarray(dls_deg100, float)
    kappa = np.deg2rad(dls/100.0)

    depth_from_bit = (md[-1] - md)  # 0 at bit
    w_air = np.zeros_like(md) + dp_w_air
    od_in = np.zeros_like(md) + dp_od
    mask_hwdp = depth_from_bit <= (hwdp_len + dc_len)
    w_air[mask_hwdp] = hwdp_w_air; od_in[mask_hwdp] = hwdp_od
    mask_dc = depth_from_bit <= dc_len
    w_air[mask_dc] = dc_w_air;     od_in[mask_dc] = dc_od

    w_b = w_air * BF
    r_eff_ft = np.where(cased_mask, (casing_id_in/2)/12.0, (open_hole_diam_in/2)/12.0)
    mu_slide = np.where(cased_mask, mu_c_slide, mu_oh_slide)
    mu_rot   = np.where(cased_mask, mu_c_rot,   mu_oh_rot)

    sigma = -1.0 if scenario.lower().startswith("slack") or scenario.lower().startswith("rih") else +1.0
    T = -float(wob_lbf)  # negative on-bottom WOB; 0 off-bottom
    M = float(bit_tq)

    T_list, M_list, N_list = [T], [M], []

    for i in range(len(md)-1, 0, -1):  # bit -> surface
        ds = md[i]-md[i-1]
        N = w_b[i]*abs(np.sin(inc[i])) + abs(T)*abs(kappa[i])
        dT = (sigma*w_b[i]*np.cos(inc[i]) + mu_slide[i]*N) * ds
        dM = (mu_rot[i]*N*r_eff_ft[i]) * ds
        T, M = T + dT, M + dM
        N_list.append(N); T_list.append(T); M_list.append(M)

    T_arr = np.array(T_list[::-1]); M_arr = np.array(M_list[::-1])
    N_arr = np.array(N_list[::-1]); N_arr = np.pad(N_arr, (1,0), constant_values=N_arr[0] if N_arr.size else 0)

    df = pd.DataFrame(dict(
        md_ft=md, inc_deg=deg(inc), dls_deg100=dls,
        w_air_lbft=w_air, w_b_lbft=w_b,
        mu_slide=mu_slide, mu_rot=mu_rot,
        N_lbf=N_arr, T_lbf=T_arr, M_lbf_ft=M_arr, cased=cased_mask.astype(bool)
    ))
    return df, float(T_arr[0]), float(M_arr[0])

# ------------------------------- UI -----------------------------------------
st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

tabs = st.tabs(["Trajectory & 3D schematic", "Torque & Drag"])

# === TRAJECTORY TAB ===
with tabs[0]:
    st.subheader("Synthetic survey (Minimum Curvature, Δs = 1 ft)")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"], key="profile")
    c2.number_input("KOP MD (ft)", min_value=0.0, step=100.0, key="kop_md", value=get_state("kop_md", 1000.0))
    c3.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], key="quick_az", on_change=set_az_from_quick)
    c4.number_input("Azimuth (deg from North, clockwise)", min_value=-360.0, max_value=720.0, step=1.0,
                    key="az_deg", value=get_state("az_deg", 0.0))

    # profile-specific controls
    prof = get_state("profile", "Build & Hold")
    if prof == "Build & Hold":
        a,b,c = st.columns(3)
        a.number_input("Build rate (deg/100 ft)", min_value=0.0, step=0.1, key="br", value=get_state("br", 3.0))
        b.number_input("Final inclination (deg)", min_value=0.0, step=0.1, key="theta_hold", value=get_state("theta_hold", 30.0))
        c.number_input("Target MD (ft)", min_value=0.0, step=100.0, key="target_md", value=get_state("target_md", 10000.0))
    elif prof == "Build–Hold–Drop":
        a,b,c = st.columns(3)
        a.number_input("Build rate (deg/100 ft)", min_value=0.0, step=0.1, key="br", value=get_state("br", 3.0))
        b.number_input("Hold angle (deg)", min_value=0.0, step=0.1, key="theta_hold", value=get_state("theta_hold", 30.0))
        c.number_input("Target MD (ft)", min_value=0.0, step=100.0, key="target_md", value=get_state("target_md", 10000.0))
        d,e,f = st.columns(3)
        d.number_input("Drop starts at MD (ft)", min_value=0.0, step=100.0, key="drop_start_md", value=get_state("drop_start_md", 7000.0))
        e.number_input("Drop rate (deg/100 ft)", min_value=0.0, step=0.1, key="drop_rate", value=get_state("drop_rate", 2.0))
        f.number_input("Final inclination after drop (deg)", min_value=0.0, step=0.1, key="final_inc", value=get_state("final_inc", 0.0))
    else:  # Horizontal
        a,b,c = st.columns(3)
        a.number_input("Build rate (deg/100 ft)", min_value=0.0, step=0.1, key="br", value=get_state("br", 8.0))
        b.number_input("Lateral length (ft)", min_value=0.0, step=100.0, key="lat_len", value=get_state("lat_len", 2000.0))
        c.number_input("Target MD (ft)", min_value=0.0, step=100.0, key="target_md", value=get_state("target_md", 10000.0))

    # Casing/Open-hole simple
    st.markdown("### Casing / Open-hole (simple)")
    x1,x2,x3,x4 = st.columns([1.2,1.2,1.2,1.2])
    x1.selectbox("Last casing nominal OD", list(CASING_DB.keys()), key="last_nom_od", index=2)
    weights = sorted(CASING_DB[get_state("last_nom_od")].keys())
    x2.selectbox("lb/ft (standards only)", weights, key="last_lbft",
                 index=min(1, len(weights)-1))
    casing_id_in = CASING_DB[get_state("last_nom_od")][get_state("last_lbft")]
    x3.text_input("Casing ID (in, locked)", value=f"{casing_id_in:.3f}", key="last_casing_id_locked", disabled=True)
    x4.number_input("Deepest shoe MD (ft)", min_value=0.0, step=50.0, key="shoe_md", value=get_state("shoe_md", 3000.0))
    st.number_input("Open hole diameter (in)", min_value=4.75, max_value=26.0, step=0.25, key="oh_diam_in", value=get_state("oh_diam_in", 8.50))

    # ---------- compute trajectory ----------
    kop     = float(get_state("kop_md", 1000.0))
    az_deg  = float(get_state("az_deg", 0.0))
    br      = float(get_state("br", 3.0))
    target  = float(get_state("target_md", 10000.0))
    theta_h = float(get_state("theta_hold", 30.0))
    shoe_md = float(get_state("shoe_md", 3000.0))
    oh_in   = float(get_state("oh_diam_in", 8.5))

    if prof == "Build & Hold":
        md,inc,az,tvd,n,e,dls = synth_build_hold(kop, br, theta_h, target, az_deg)
    elif prof == "Build–Hold–Drop":
        md,inc,az,tvd,n,e,dls = synth_build_hold_drop(
            kop, br, theta_h,
            float(get_state("drop_start_md", 7000.0)),
            float(get_state("drop_rate", 2.0)),
            float(get_state("final_inc", 0.0)),
            target, az_deg
        )
    else:
        md,inc,az,tvd,n,e,dls = synth_horizontal(
            kop, br, float(get_state("lat_len", 2000.0)), target, az_deg
        )

    cased_mask = md <= shoe_md

    st.session_state["traj"] = dict(md=md, inc=inc, az=az, tvd=tvd, north=n, east=e, dls=dls)
    st.session_state["traj_meta"] = dict(cased_mask=cased_mask, casing_id_in=casing_id_in, open_hole_in=oh_in)

    # ---------- classic plots (default) ----------
    plot3d = go.Figure()
    # segment list
    segs = []; cur = cased_mask[0]; s0 = 0
    for i in range(1, len(md)):
        if cased_mask[i] != cur:
            segs.append((cur, s0, i)); s0=i; cur=cased_mask[i]
    segs.append((cur, s0, len(md)-1))

    for is_cased, i0, i1 in segs:
        color = "steelblue" if is_cased else "saddlebrown"
        width = 6 if is_cased else 5
        plot3d.add_trace(go.Scatter3d(
            x=e[i0:i1+1], y=n[i0:i1+1], z=-tvd[i0:i1+1], mode="lines",
            line=dict(color=color, width=width),
            name="Cased" if is_cased else "Open hole"
        ))
    plot3d.update_layout(height=360, margin=dict(l=0,r=0,t=0,b=0),
        scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
        legend=dict(orientation="h", x=0.0, y=1.02)
    )

    prof_fig = go.Figure()
    for is_cased, i0, i1 in segs:
        color = "steelblue" if is_cased else "saddlebrown"
        prof_fig.add_trace(go.Scatter(x=md[i0:i1+1], y=tvd[i0:i1+1], mode="lines", line=dict(color=color, width=3),
                                      name="Cased" if is_cased else "Open hole"))
    prof_fig.update_yaxes(autorange="reversed")  # depth down
    prof_fig.update_layout(height=300, margin=dict(l=30,r=10,t=10,b=30),
                           xaxis_title="MD (ft)", yaxis_title="TVD (ft)")

    plan_fig = go.Figure()
    for is_cased, i0, i1 in segs:
        color = "steelblue" if is_cased else "saddlebrown"
        plan_fig.add_trace(go.Scatter(x=e[i0:i1+1], y=n[i0:i1+1], mode="lines", line=dict(color=color, width=3),
                                      name="Cased" if is_cased else "Open hole"))
    plan_fig.update_layout(height=300, margin=dict(l=30,r=10,t=10,b=30),
                           xaxis_title="East (ft)", yaxis_title="North (ft)")

    # layout: 3D + (profile & plan)
    cA, cB = st.columns([1.2, 1.2])
    with cA:
        st.plotly_chart(plot3d, use_container_width=True)
    with cB:
        cB1, cB2 = st.columns(2)
        with cB1: st.plotly_chart(prof_fig, use_container_width=True)
        with cB2: st.plotly_chart(plan_fig, use_container_width=True)

    st.markdown("**Survey preview (first 11 rows)**")
    df_svy = pd.DataFrame(dict(MD_ft=md, Inc_deg=inc, Az_deg=az, TVD_ft=tvd, North_ft=n, East_ft=e, DLS_deg100=dls))
    st.dataframe(df_svy.head(11), use_container_width=True)

# === T&D TAB ===
with tabs[1]:
    st.subheader("Soft-string Torque & Drag (Johancsik) — with buoyancy")

    st.number_input("Azimuth for T&D (deg)", min_value=-360.0, max_value=720.0, step=1.0,
                    value=get_state("az_deg", 0.0), key="az_deg_td", on_change=push_td_az_into_trajectory)

    r1,r2,r3,r4 = st.columns(4)
    r1.number_input("μ in casing (sliding)",  min_value=0.0, max_value=1.0, step=0.01, key="mu_c_slide", value=get_state("mu_c_slide", 0.25))
    r2.number_input("μ in open hole (sliding)",min_value=0.0, max_value=1.0, step=0.01, key="mu_oh_slide", value=get_state("mu_oh_slide", 0.35))
    r3.number_input("μ in casing (rotating)", min_value=0.0, max_value=1.0, step=0.01, key="mu_c_rot",   value=get_state("mu_c_rot", 0.25))
    r4.number_input("μ in open hole (rotating)",min_value=0.0, max_value=1.0, step=0.01, key="mu_oh_rot", value=get_state("mu_oh_rot", 0.35))

    st.number_input("Mud weight (ppg)", min_value=0.0, step=0.1, key="mw_ppg", value=get_state("mw_ppg", 10.0))

    st.markdown("**Drillstring (bit up)**")
    g1,g2,g3 = st.columns(3)
    with g1:
        st.number_input("DC length (ft)", min_value=0.0, step=10.0, key="dc_len", value=get_state("dc_len", 600.0))
        st.number_input("DC weight (air, lb/ft)", min_value=0.0, step=0.1, key="dc_w_air", value=get_state("dc_w_air", 66.7))
        st.number_input("DC OD (in)", min_value=1.0, step=0.01, key="dc_od", value=get_state("dc_od", 8.00))
    with g2:
        st.number_input("HWDP length (ft)", min_value=0.0, step=10.0, key="hwdp_len", value=get_state("hwdp_len", 1000.0))
        st.number_input("HWDP weight (air, lb/ft)", min_value=0.0, step=0.1, key="hwdp_w_air", value=get_state("hwdp_w_air", 16.0))
        st.number_input("HWDP OD (in)", min_value=1.0, step=0.01, key="hwdp_od", value=get_state("hwdp_od", 3.50))
    with g3:
        st.number_input("DP length (ft)", min_value=0.0, step=10.0, key="dp_len", value=get_state("dp_len", 7000.0))
        st.number_input("DP weight (air, lb/ft)", min_value=0.0, step=0.1, key="dp_w_air", value=get_state("dp_w_air", 19.5))
        st.number_input("DP OD (in)", min_value=1.0, step=0.01, key="dp_od", value=get_state("dp_od", 5.00))

    st.selectbox("Scenario", ["Slack-off (RIH)", "Pickup (POOH)"], key="scenario")
    a,b = st.columns(2)
    a.number_input("WOB (lbf) for on-bottom", min_value=0.0, step=100.0, key="wob_lbf", value=get_state("wob_lbf", 6000.0))
    b.number_input("Bit torque (lbf-ft) for on-bottom", min_value=0.0, step=10.0, key="bit_tq", value=get_state("bit_tq", 0.0))

    traj = get_state("traj"); meta = get_state("traj_meta")
    if not traj or not meta:
        st.warning("Define the wellpath in the Trajectory tab first.")
        st.stop()

    df, hook, tq = soft_string_stepper(
        traj["md"], traj["inc"], traj["dls"], meta["cased_mask"], get_state("mw_ppg", 10.0),
        meta["casing_id_in"], meta["open_hole_in"],
        get_state("mu_c_slide", 0.25), get_state("mu_oh_slide", 0.35),
        get_state("mu_c_rot", 0.25),   get_state("mu_oh_rot", 0.35),
        get_state("dc_len", 600.0), get_state("dc_od", 8.0), get_state("dc_w_air", 66.7),
        get_state("hwdp_len", 1000.0), get_state("hwdp_od", 3.5), get_state("hwdp_w_air", 16.0),
        get_state("dp_len", 7000.0), get_state("dp_od", 5.0), get_state("dp_w_air", 19.5),
        scenario=get_state("scenario", "Slack-off (RIH)"),
        wob_lbf=get_state("wob_lbf", 6000.0), bit_tq=get_state("bit_tq", 0.0)
    )

    st.success(f"Surface hookload: **{hook:,.0f} lbf** — Surface torque: **{tq:,.0f} lbf·ft**")

    c1,c2 = st.columns(2)
    with c1:
        figT = go.Figure()
        figT.add_trace(go.Scatter(x=df["md_ft"], y=df["T_lbf"], mode="lines", name="Tension/Hookload"))
        figT.update_layout(xaxis_title="MD (ft)", yaxis_title="lbf", height=300, margin=dict(l=30,r=10,t=10,b=30))
        st.plotly_chart(figT, use_container_width=True)
    with c2:
        figM = go.Figure()
        figM.add_trace(go.Scatter(x=df["md_ft"], y=df["M_lbf_ft"], mode="lines", name="Torque"))
        figM.update_layout(xaxis_title="MD (ft)", yaxis_title="lbf·ft", height=300, margin=dict(l=30,r=10,t=10,b=30))
        st.plotly_chart(figM, use_container_width=True)

    st.markdown("**Iteration trace (bit → surface)**")
    st.dataframe(df[["md_ft","inc_deg","dls_deg100","w_air_lbft","w_b_lbft","mu_slide","mu_rot","N_lbf","T_lbf","M_lbf_ft","cased"]].head(250),
                 use_container_width=True)
