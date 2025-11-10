# td_app.py
# Wellpath + Torque&Drag (Δs = 1 ft) — fixed session_state flow, linked tabs, depth-down 3D
# Author: (Your team) — PEGN 517
# Streamlit callbacks & rerun pattern per docs; depth-down via z = -TVD.  See refs in README.

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Utilities: safe state read + callbacks
# -----------------------------------------------------------------------------
def get_state(key: str, default=None):
    return st.session_state.get(key, default)

def set_az_from_quick():
    mapping = {"North (0)": 0.0, "East (90)": 90.0, "South (180)": 180.0, "West (270)": 270.0}
    sel = st.session_state.get("quick_az", "North (0)")
    st.session_state["az_deg"] = mapping.get(sel, 0.0)
    st.rerun()  # rerun after programmatic state change (legal & supported)

def push_td_az_into_trajectory():
    st.session_state["az_deg"] = st.session_state.get("az_deg_td", 0.0)
    st.rerun()

# -----------------------------------------------------------------------------
# API-ish constants (subset of API 5CT typical combos; extend as needed)
# Nominal OD -> { weight_lbft: ID_in }
# -----------------------------------------------------------------------------
CASING_DB = {
    "13-3/8": {
        48.0: 12.415,
        54.5: 12.347,
        61.0: 12.300,
    },
    "9-5/8": {
        36.0: 8.921,
        40.0: 8.535,
        43.5: 8.347,
    },
    "7": {
        20.0: 6.538,
        23.0: 6.276,
        26.0: 6.094,
        29.0: 5.921,
    },
}

# -----------------------------------------------------------------------------
# Geometry helpers (minimum curvature)
# -----------------------------------------------------------------------------
def rad(deg): return np.deg2rad(deg)
def deg(radv): return np.rad2deg(radv)

def minimum_curvature(md, inc_deg, az_deg):
    """Compute TVD/N/E using minimum curvature. Inputs are arrays (same length)."""
    md = np.asarray(md)
    inc = rad(np.asarray(inc_deg))
    az = rad(np.asarray(az_deg))

    n = len(md)
    tvd = np.zeros(n)
    north = np.zeros(n)
    east = np.zeros(n)

    for i in range(1, n):
        dmd = md[i] - md[i-1]
        inc1, inc2 = inc[i-1], inc[i]
        az1, az2 = az[i-1], az[i]

        cos_dl = (np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1) + np.cos(inc1)*np.cos(inc2))
        cos_dl = np.clip(cos_dl, -1.0, 1.0)
        dl = np.arccos(cos_dl)  # dogleg in radians

        if dl < 1e-12:
            rf = 1.0
        else:
            rf = (2/ dl) * np.tan(dl/2)

        tvd[i]   = tvd[i-1]   + 0.5 * dmd * (np.cos(inc1) + np.cos(inc2)) * rf
        north[i] = north[i-1] + 0.5 * dmd * (np.sin(inc1)*np.cos(az1) + np.sin(inc2)*np.cos(az2)) * rf
        east[i]  = east[i-1]  + 0.5 * dmd * (np.sin(inc1)*np.sin(az1) + np.sin(inc2)*np.sin(az2)) * rf

    # DLS deg/100ft for info
    dls_deg_per100 = np.zeros(n)
    for i in range(1, n):
        dmd = md[i] - md[i-1]
        if dmd > 0:
            dls_deg_per100[i] = deg(np.arccos(
                np.sin(inc[i-1])*np.sin(inc[i])*np.cos(az[i]-az[i-1]) + np.cos(inc[i-1])*np.cos(inc[i])
            )) * 100 / dmd

    return tvd, north, east, dls_deg_per100

def synth_build_hold(kop_md, build_rate_deg_per100, theta_hold_deg, target_md, az_deg, ds=1.0):
    """Simple synthetic Build & Hold path (inc increases from KOP at constant build, then holds)."""
    md = np.arange(0.0, float(target_md) + ds, ds)
    inc = np.zeros_like(md)
    az = np.zeros_like(md) + float(az_deg)

    # build until theta_hold or target
    build_rate_deg_perft = build_rate_deg_per100 / 100.0
    for i, m in enumerate(md):
        if m < kop_md:
            inc[i] = 0.0
        else:
            inc[i] = min(theta_hold_deg, max(0.0, (m - kop_md) * build_rate_deg_perft))

    tvd, north, east, dls = minimum_curvature(md, inc, az)
    return md, inc, az, tvd, north, east, dls

# -----------------------------------------------------------------------------
# Soft-string T&D (Johancsik-style stepper)
# -----------------------------------------------------------------------------
def soft_string_stepper(md, inc_deg, dls_deg100, cased_mask, mw_ppg,
                        # hole geometry (for torque radius)
                        casing_id_in, open_hole_diam_in,
                        # friction
                        mu_c_slide, mu_oh_slide, mu_c_rot, mu_oh_rot,
                        # string from bit up
                        dc_len, dc_od, dc_w_air,
                        hwdp_len, hwdp_od, hwdp_w_air,
                        dp_len, dp_od, dp_w_air,
                        # boundary / scenario
                        scenario="RIH", wob_lbf=0.0, bit_torque_lbf_ft=0.0):
    """
    Returns DataFrame with per-step forces/torque and overall surface hookload & torque.

    Notes:
      - Buoyancy: BF=(65.5-MW)/65.5 (ppg)
      - Normal force per unit length N = w_b*sin(theta) + T*kappa
      - Axial: T_next = T + (sigma*w_b*cos(theta) + mu*N)*ds, sigma = +1 POOH, -1 RIH
      - Torque: M_next = M + mu_rot*N*r_eff*ds ; r_eff = (ID/2) for casing or (OH/2) in open hole
    """
    mw = float(mw_ppg)
    BF = (65.5 - mw)/65.5  # buoyancy factor (ppg) — standard office formula

    md = np.asarray(md)
    inc = rad(np.asarray(inc_deg))
    dls = np.asarray(dls_deg100)             # deg/100ft
    ds = np.gradient(md)                     # may vary but we target 1 ft
    kappa = np.deg2rad(dls/100.0)            # rad/ft

    # Build per-foot air weights and OD along the string measured from bit (bottom) to surface.
    total_len = float(dc_len + hwdp_len + dp_len)
    # Map each MD to distance from bit:
    depth_from_bit = (md[-1] - md)  # 0 at bit, increasing up to surface
    w_air_perft = np.zeros_like(md) + dp_w_air
    od_in = np.zeros_like(md) + dp_od
    # HWDP segment:
    mask_hwdp = depth_from_bit <= (hwdp_len + dc_len)
    w_air_perft[mask_hwdp] = hwdp_w_air
    od_in[mask_hwdp] = hwdp_od
    # DC segment:
    mask_dc = depth_from_bit <= dc_len
    w_air_perft[mask_dc] = dc_w_air
    od_in[mask_dc] = dc_od

    # Buoyed weight per ft
    w_b = w_air_perft * BF

    # Effective radius for torque (cased vs open):
    r_eff_ft = np.where(cased_mask, (casing_id_in/2.0)/12.0, (open_hole_diam_in/2.0)/12.0)

    # Friction factors (sliding for axial; rotating for torque)
    mu_slide = np.where(cased_mask, mu_c_slide, mu_oh_slide)
    mu_rot   = np.where(cased_mask, mu_c_rot,   mu_oh_rot)

    # Motion sign
    sigma = -1.0 if scenario.lower().startswith("rih") else +1.0  # friction opposes motion

    # Boundary at bit:
    T = 0.0 - float(wob_lbf)  # on-bottom use negative WOB; off-bottom wob=0
    M = float(bit_torque_lbf_ft)

    # Iteration containers
    T_list, M_list = [T], [M]
    N_list, BF_list = [], []  # store for debugging/trace

    for i in range(len(md)-1, 0, -1):  # integrate from bit (end) to surface (start)
        # use angle at "i" (closer to bit) for gravity components
        theta = inc[i]
        ds_i  = md[i] - md[i-1]
        k_i   = kappa[i]
        mu_s  = mu_slide[i]
        mu_r  = mu_rot[i]
        w_b_i = w_b[i]

        # normal per unit length
        N = w_b_i * abs(np.sin(theta)) + abs(T) * abs(k_i)

        dT = (sigma * w_b_i * np.cos(theta) + mu_s * N) * ds_i
        T_next = T + dT

        dM = (mu_r * N * r_eff_ft[i]) * ds_i
        M_next = M + dM

        N_list.append(N)
        BF_list.append(BF)
        T_list.append(T_next); M_list.append(M_next)
        T, M = T_next, M_next

    # reverse lists to align from surface->bit
    T_arr = np.array(T_list[::-1])
    M_arr = np.array(M_list[::-1])
    N_arr = np.array(N_list[::-1])
    # pad N to same length
    N_arr = np.pad(N_arr, (1, 0), constant_values=N_arr[0] if N_arr.size else 0.0)

    df = pd.DataFrame(dict(
        md_ft = md,
        inc_deg = np.degrees(inc),
        dls_deg100 = dls,
        w_air_lbft = w_air_perft,
        w_b_lbft = w_b,
        mu_slide = mu_slide,
        mu_rot = mu_rot,
        N_lbf = N_arr,
        T_lbf = T_arr,
        M_lbf_ft = M_arr,
        cased = cased_mask.astype(bool),
    ))
    return df, float(T_arr[0]), float(M_arr[0])

# -----------------------------------------------------------------------------
# UI – Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

tabs = st.tabs(["Trajectory & 3D schematic", "Torque & Drag"])

# --- TRAJECTORY TAB ---
with tabs[0]:
    st.subheader("Synthetic survey (Minimum Curvature)")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Profile", ["Build & Hold"], key="profile")
    c2.number_input("KOP MD (ft)", min_value=0.0, step=100.0, key="kop_md", value=get_state("kop_md", 1000.0))
    c3.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"],
                 key="quick_az", on_change=set_az_from_quick)
    c4.number_input("Azimuth (deg from North, clockwise)", min_value=-360.0, max_value=720.0, step=1.0,
                    key="az_deg", value=get_state("az_deg", 0.0))

    c5, c6, c7 = st.columns(3)
    c5.number_input("Build rate (deg/100 ft)", min_value=0.0, step=0.1, key="build_rate", value=get_state("build_rate", 3.0))
    c6.number_input("Final inclination (deg)", min_value=0.0, step=0.1, key="theta_hold", value=get_state("theta_hold", 30.0))
    c7.number_input("Target MD (ft)", min_value=0.0, step=100.0, key="target_md", value=get_state("target_md", 10000.0))

    # Casing / open hole simple inputs
    st.markdown("### Casing / Open-hole (simple)")
    cA, cB, cC, cD = st.columns([1.2,1.2,1.2,1.2])
    cA.selectbox("Last casing nominal OD", list(CASING_DB.keys()), key="last_nom_od", index=2)
    # weights depend on nominal OD
    weights = sorted(CASING_DB[get_state("last_nom_od")].keys())
    cB.selectbox("lb/ft (standards only)", weights, key="last_lbft", index=1)
    casing_id = CASING_DB[get_state("last_nom_od")][get_state("last_lbft")]
    cC.text_input("Casing ID (in, locked)", value=f"{casing_id:.3f}", key="last_casing_id_locked", disabled=True)
    cD.number_input("Deepest shoe MD (ft)", min_value=0.0, step=50.0, key="shoe_md", value=get_state("shoe_md", 3000.0))

    st.number_input("Open hole diameter (in)", min_value=4.75, max_value=26.0, step=0.25, key="oh_diam_in",
                    value=get_state("oh_diam_in", 8.50))

    # READ all values (safe)
    kop_md    = float(get_state("kop_md", 1000.0))
    az_deg    = float(get_state("az_deg", 0.0))
    brate     = float(get_state("build_rate", 3.0))
    theta_hold= float(get_state("theta_hold", 30.0))
    target_md = float(get_state("target_md", 10000.0))
    shoe_md   = float(get_state("shoe_md", 3000.0))
    oh_in     = float(get_state("oh_diam_in", 8.5))
    casing_id_in = float(casing_id)

    # Build synthetic path
    md, inc, az, tvd, north, east, dls = synth_build_hold(kop_md, brate, theta_hold, target_md, az_deg, ds=1.0)
    cased_mask = md <= shoe_md

    # Save the trajectory for the T&D tab
    st.session_state["traj"] = dict(md=md, inc=inc, az=az, tvd=tvd, north=north, east=east, dls=dls)
    st.session_state["traj_meta"] = dict(cased_mask=cased_mask, casing_id_in=casing_id_in, open_hole_in=oh_in)

    # 3D plot (depth-down via z = -TVD).  Separate traces for cased vs open hole.
    fig = go.Figure()
    # Segment indices where cased changes to open
    segments = []
    current = cased_mask[0]
    start = 0
    for i in range(1, len(md)):
        if cased_mask[i] != current:
            segments.append((current, start, i))
            start = i
            current = cased_mask[i]
    segments.append((current, start, len(md)-1))

    for is_cased, i0, i1 in segments:
        color = "steelblue" if is_cased else "saddlebrown"
        width = 6 if is_cased else 5
        fig.add_trace(go.Scatter3d(
            x=east[i0:i1+1], y=north[i0:i1+1], z=-tvd[i0:i1+1],  # depth-down
            mode="lines", line=dict(color=color, width=width),
            name="Cased" if is_cased else "Open hole"
        ))

    fig.update_layout(
        height=360, margin=dict(l=0,r=0,t=0,b=0),
        scene=dict(
            xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)",
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", x=0.0, y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Survey table (first few rows)
    df_svy = pd.DataFrame(dict(MD_ft=md, Inc_deg=inc, Az_deg=az, TVD_ft=tvd, North_ft=north, East_ft=east, DLS_deg100=dls))
    df_svy["Inc_deg"] = df_svy["Inc_deg"].round(3)
    df_svy["Az_deg"] = df_svy["Az_deg"].round(3)
    df_svy["DLS_deg100"] = df_svy["DLS_deg100"].round(3)
    st.markdown("**Survey and calculated positions (first 11 rows)**")
    st.dataframe(df_svy.head(11), use_container_width=True)

    st.caption("Δs = 1 ft; minimum curvature with ratio factor; industry depth-down 3D (using z=−TVD).")

# --- T&D TAB ---
with tabs[1]:
    st.subheader("Soft-string Torque & Drag (Johancsik) — with buoyancy")

    # Quick link: azimuth control for T&D (writes back using callback)
    st.number_input("Azimuth for T&D (deg)", min_value=-360.0, max_value=720.0, step=1.0,
                    value=get_state("az_deg", 0.0), key="az_deg_td", on_change=push_td_az_into_trajectory)

    # Friction & MW
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("μ in casing (sliding)", min_value=0.0, max_value=1.0, step=0.01, key="mu_c_slide", value=get_state("mu_c_slide", 0.25))
    c2.number_input("μ in open hole (sliding)", min_value=0.0, max_value=1.0, step=0.01, key="mu_oh_slide", value=get_state("mu_oh_slide", 0.35))
    c3.number_input("μ in casing (rotating)", min_value=0.0, max_value=1.0, step=0.01, key="mu_c_rot", value=get_state("mu_c_rot", 0.25))
    c4.number_input("μ in open hole (rotating)", min_value=0.0, max_value=1.0, step=0.01, key="mu_oh_rot", value=get_state("mu_oh_rot", 0.35))

    st.number_input("Mud weight (ppg)", min_value=0.0, step=0.1, key="mw_ppg", value=get_state("mw_ppg", 10.0))

    # Drillstring inputs (bit up)
    st.markdown("**Drillstring (bit up)**")
    g1, g2, g3 = st.columns(3)
    # DC
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

    # Scenario
    st.selectbox("Scenario", ["Slack-off (RIH)", "Pickup (POOH)"], key="scenario")
    c1, c2 = st.columns(2)
    c1.number_input("WOB (lbf) for on-bottom", min_value=0.0, step=100.0, key="wob_lbf", value=get_state("wob_lbf", 6000.0))
    c2.number_input("Bit torque (lbf-ft) for on-bottom", min_value=0.0, step=10.0, key="bit_tq", value=get_state("bit_tq", 0.0))

    # Pull trajectory from state
    traj = get_state("traj")
    meta = get_state("traj_meta")
    if not traj or not meta:
        st.warning("Define the wellpath in the Trajectory tab first.")
        st.stop()

    md    = traj["md"]; inc = traj["inc"]; az = traj["az"]
    dls   = traj["dls"]
    cased_mask = meta["cased_mask"]
    casing_id_in = meta["casing_id_in"]
    open_hole_in = meta["open_hole_in"]

    # Compute
    df, hook_surf, torque_surf = soft_string_stepper(
        md, inc, dls, cased_mask,
        get_state("mw_ppg", 10.0),
        casing_id_in, open_hole_in,
        get_state("mu_c_slide", 0.25), get_state("mu_oh_slide", 0.35),
        get_state("mu_c_rot", 0.25),   get_state("mu_oh_rot", 0.35),
        get_state("dc_len", 600.0), get_state("dc_od", 8.00), get_state("dc_w_air", 66.7),
        get_state("hwdp_len", 1000.0), get_state("hwdp_od", 3.50), get_state("hwdp_w_air", 16.0),
        get_state("dp_len", 7000.0), get_state("dp_od", 5.00), get_state("dp_w_air", 19.5),
        scenario=get_state("scenario", "Slack-off (RIH)"),
        wob_lbf=get_state("wob_lbf", 6000.0), bit_torque_lbf_ft=get_state("bit_tq", 0.0)
    )

    st.success(f"Surface hookload: **{hook_surf:,.0f} lbf** — Surface torque: **{torque_surf:,.0f} lbf·ft**")

    # Plots
    c1, c2 = st.columns(2)
    with c1:
        figT = go.Figure()
        figT.add_trace(go.Scatter(x=df["md_ft"], y=df["T_lbf"], mode="lines", name="Tension / Hookload"))
        figT.update_layout(xaxis_title="MD (ft)", yaxis_title="lbf", height=300, margin=dict(l=30,r=10,t=10,b=30))
        st.plotly_chart(figT, use_container_width=True)
    with c2:
        figM = go.Figure()
        figM.add_trace(go.Scatter(x=df["md_ft"], y=df["M_lbf_ft"], mode="lines", name="Torque"))
        figM.update_layout(xaxis_title="MD (ft)", yaxis_title="lbf·ft", height=300, margin=dict(l=30,r=10,t=10,b=30))
        st.plotly_chart(figM, use_container_width=True)

    # Iteration trace (compact)
    st.markdown("**Iteration trace (bit → surface)**")
    cols = ["md_ft","inc_deg","dls_deg100","w_air_lbft","w_b_lbft","mu_slide","mu_rot","N_lbf","T_lbf","M_lbf_ft","cased"]
    st.dataframe(df[cols].head(200), use_container_width=True)
