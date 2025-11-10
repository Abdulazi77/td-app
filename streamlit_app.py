# td_app.py — Wellpath + Torque & Drag (Δs = 1 ft)
# Default visuals mirror "Drag Risk Off-Bottom Torque":
#   Left: μ-sweep off-bottom torque vs depth (multiple curves)
#   Right: Elemental torque vs depth with safety-factor reference lines
#
# Methods:
# - Minimum curvature survey with ratio factor (RF) and DLS.
# - Johancsik soft-string: N ≈ w_b*sin(θ) + T*κ; dT = (σ_ax*w_b*cosθ + μ_slide*N) ds; dM = μ_rot*N*r_eff ds
# - Buoyancy factor: BF = (65.5 - MW)/65.5
# - API RP 7G torque references shown as vertical lines with a safety factor divider.
#
# Notes:
# - T&D uses the CURRENT trajectory from the Trajectory tab via st.session_state.
# - Depth-down convention for charts; 3D schematic color-codes cased vs open hole.
# - Default last casing 9-5/8" with 8.5" OH as requested.

import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")

# ---------- constants & helpers ----------
IN2FT   = 1.0 / 12.0
DEG2RAD = math.pi / 180.0

def buoyancy_factor_ppg(mw_ppg: float) -> float:
    # BF per standard petroleum refs
    return (65.5 - mw_ppg) / 65.5

def min_curv_step(inc1_deg, az1_deg, inc2_deg, az2_deg, ds_ft):
    """Minimum curvature step with ratio factor (RF). Returns ΔN, ΔE, ΔTVD, κ, DLS."""
    inc1 = inc1_deg * DEG2RAD
    inc2 = inc2_deg * DEG2RAD
    az1  = az1_deg  * DEG2RAD
    az2  = az2_deg  * DEG2RAD

    cos_dpsi = math.sin(inc1)*math.sin(inc2)*math.cos(az2-az1) + math.cos(inc1)*math.cos(inc2)
    cos_dpsi = max(-1.0, min(1.0, cos_dpsi))
    dpsi = math.acos(cos_dpsi)

    if abs(dpsi) < 1e-12:
        rf = 1.0
    else:
        rf = 2.0/dpsi * math.tan(dpsi/2.0)

    dN = 0.5*ds_ft*(math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2))*rf
    dE = 0.5*ds_ft*(math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2))*rf
    dTVD = 0.5*ds_ft*(math.cos(inc1) + math.cos(inc2))*rf

    kappa = dpsi/ds_ft if ds_ft > 0 else 0.0
    dls = (dpsi/DEG2RAD) / (ds_ft/100.0) if ds_ft > 0 else 0.0
    return dN, dE, dTVD, kappa, dls

def build_hold_survey(kop_md, bur_deg_per_100ft, theta_hold_deg, target_md, az_deg,
                      ds_ft=1.0):
    """
    Build & Hold synthetic survey:
      - Surface to KOP vertical,
      - Build at BUR until theta_hold,
      - Hold to TD (target_md).
    """
    md_list   = [0.0]
    inc_list  = [0.0]
    az_list   = [az_deg]
    tvd_list  = [0.0]
    north_list= [0.0]
    east_list = [0.0]
    dls_list  = [0.0]
    kappa_list= [0.0]

    md, inc, az = 0.0, 0.0, az_deg

    # vertical to KOP
    while md < kop_md:
        step = min(ds_ft, kop_md - md)
        md_next, inc_next, az_next = md + step, 0.0, az_deg
        dN, dE, dTVD, kappa, dls = min_curv_step(inc, az, inc_next, az_next, step)
        north_list.append(north_list[-1] + dN)
        east_list.append(east_list[-1] + dE)
        tvd_list.append(tvd_list[-1] + dTVD)
        md_list.append(md_next); inc_list.append(inc_next); az_list.append(az_next)
        dls_list.append(dls); kappa_list.append(kappa)
        md, inc, az = md_next, inc_next, az_next

    # build to hold angle
    BUR = bur_deg_per_100ft
    while inc < theta_hold_deg and md < target_md:
        step = min(ds_ft, target_md - md)
        inc_next = min(theta_hold_deg, inc + BUR*(step/100.0))
        az_next = az_deg
        md_next = md + step
        dN, dE, dTVD, kappa, dls = min_curv_step(inc, az, inc_next, az_next, step)
        north_list.append(north_list[-1] + dN)
        east_list.append(east_list[-1] + dE)
        tvd_list.append(tvd_list[-1] + dTVD)
        md_list.append(md_next); inc_list.append(inc_next); az_list.append(az_next)
        dls_list.append(dls); kappa_list.append(kappa)
        md, inc, az = md_next, inc_next, az_next

    # hold to TD
    while md < target_md:
        step = min(ds_ft, target_md - md)
        md_next, inc_next, az_next = md + step, inc, az
        dN, dE, dTVD, kappa, dls = min_curv_step(inc, az, inc_next, az_next, step)
        north_list.append(north_list[-1] + dN)
        east_list.append(east_list[-1] + dE)
        tvd_list.append(tvd_list[-1] + dTVD)
        md_list.append(md_next); inc_list.append(inc_next); az_list.append(az_next)
        dls_list.append(dls); kappa_list.append(kappa)
        md, inc, az = md_next, inc_next, az_next

    return pd.DataFrame({
        "md_ft": md_list,
        "inc_deg": inc_list,
        "az_deg": az_list,
        "tvd_ft": tvd_list,
        "north_ft": north_list,
        "east_ft": east_list,
        "dls_deg_per_100ft": dls_list,
        "kappa_rad_per_ft": kappa_list
    })

# Subset of standard casing table (extend as needed)
# key: nominal; value: {lb/ft: ID inch}
CASING_TABLE = {
    "9-5/8":  {"36.0": 8.921, "40.0": 8.835, "43.5": 8.681},
    "13-3/8": {"54.5": 12.415, "61.0": 12.347}
}

def casing_id_std(nom_od: str, lbft: str) -> float:
    return float(CASING_TABLE[nom_od][lbft])

# ---------- soft-string engine ----------
def soft_string_stepper(df_svy: pd.DataFrame,
                        mu_slide_cased: float, mu_slide_open: float,
                        mu_rot_cased: float,  mu_rot_open: float,
                        mw_ppg: float,
                        dc_len_ft: float, dc_od_in: float, dc_w_air_lbft: float,
                        hwdp_len_ft: float, hwdp_od_in: float, hwdp_w_air_lbft: float,
                        dp_len_ft: float, dp_od_in: float, dp_w_air_lbft: float,
                        shoe_md_ft: float,
                        scenario: str = "SLACKOFF",
                        wob_lbf: float = 0.0, bit_torque_lbf_ft: float = 0.0,
                        off_bottom_only: bool = False) -> pd.DataFrame:
    """
    Integrate bit -> surface along the current trajectory (1-ft resolution from df_svy).
    Arrays are reversed so index 0 = bit depth. Returns per-depth T and M.
    """

    # reverse to go from bit to surface
    md    = df_svy["md_ft"].to_numpy()[::-1]
    inc_r = (df_svy["inc_deg"].to_numpy() * DEG2RAD)[::-1]
    kap_r = df_svy["kappa_rad_per_ft"].to_numpy()[::-1]

    n = len(md)
    if n < 2:
        return pd.DataFrame({"md_ft":[md[0]], "T_lbf":[-wob_lbf], "M_lbf_ft":[bit_torque_lbf_ft], "cased":[md[0] <= shoe_md_ft]})

    # buoyed weights
    BF = buoyancy_factor_ppg(mw_ppg)
    w_dc   = dc_w_air_lbft   * BF
    w_hwdp = hwdp_w_air_lbft * BF
    w_dp   = dp_w_air_lbft   * BF

    r_dc_ft   = (dc_od_in  /2.0)*IN2FT
    r_hwdp_ft = (hwdp_od_in/2.0)*IN2FT
    r_dp_ft   = (dp_od_in  /2.0)*IN2FT

    # axial sign
    if off_bottom_only:
        sigma_ax = 0.0
    else:
        sigma_ax = -1.0 if scenario.upper().startswith("SLACK") else +1.0

    # starting boundary at bit
    T = -wob_lbf if not off_bottom_only else 0.0
    M =  bit_torque_lbf_ft
    Ts = [T]; Ms = [M]; md_list = [md[0]]; cased_list = [md[0] <= shoe_md_ft]

    cum_from_bit = 0.0
    L_DC   = dc_len_ft
    L_HWDP = dc_len_ft + hwdp_len_ft
    # dp beyond

    for i in range(1, n):
        ds = abs(md[i-1] - md[i])  # step length up the string
        if ds <= 0.0:
            continue
        cum_from_bit += ds

        # choose section
        if cum_from_bit <= L_DC:
            w_b = w_dc;   r_eff = r_dc_ft
        elif cum_from_bit <= L_HWDP:
            w_b = w_hwdp; r_eff = r_hwdp_ft
        else:
            w_b = w_dp;   r_eff = r_dp_ft

        is_cased = (md[i] <= shoe_md_ft)
        mu_rot   = (mu_rot_cased   if is_cased else mu_rot_open)
        mu_slide = 0.0 if off_bottom_only else (mu_slide_cased if is_cased else mu_slide_open)

        # side force per length at this step
        Ni = w_b*math.sin(inc_r[i]) + T*kap_r[i]
        dT = (sigma_ax * w_b*math.cos(inc_r[i]) + mu_slide*Ni) * ds
        dM = (mu_rot * Ni * r_eff) * ds

        T += dT; M += dM
        Ts.append(T); Ms.append(M); md_list.append(md[i]); cased_list.append(is_cased)

    return pd.DataFrame({"md_ft": md_list, "T_lbf": Ts, "M_lbf_ft": Ms,
                         "cased": cased_list, "BF": [BF]*len(md_list)})

# ---------- plotting ----------
def plot_3d(df_svy: pd.DataFrame, shoe_md_ft: float):
    seg_cased = df_svy["md_ft"] <= shoe_md_ft
    x = df_svy["east_ft"]; y = df_svy["north_ft"]; z = -df_svy["tvd_ft"]  # negative for depth-down
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x[seg_cased], y=y[seg_cased], z=z[seg_cased],
        mode="lines", line=dict(color="#4ea1ff", width=6), name="Cased"
    ))
    fig.add_trace(go.Scatter3d(
        x=x[~seg_cased], y=y[~seg_cased], z=z[~seg_cased],
        mode="lines", line=dict(color="#8b572a", width=4), name="Open-hole"
    ))
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)")
    )
    return fig

def plot_mu_sweep(df_svy, shoe_md, mw_ppg, mu_values, dc, hwdp, dp,
                  tube_torque_kft=None, combined_kft=None, sf=1.10):
    fig = go.Figure()
    for mu in mu_values:
        df = soft_string_stepper(df_svy,
                                 mu_slide_cased=0.0, mu_slide_open=0.0,
                                 mu_rot_cased=mu, mu_rot_open=mu,
                                 mw_ppg=mw_ppg,
                                 dc_len_ft=dc["L"], dc_od_in=dc["OD"], dc_w_air_lbft=dc["w"],
                                 hwdp_len_ft=hwdp["L"], hwdp_od_in=hwdp["OD"], hwdp_w_air_lbft=hwdp["w"],
                                 dp_len_ft=dp["L"], dp_od_in=dp["OD"], dp_w_air_lbft=dp["w"],
                                 shoe_md_ft=shoe_md,
                                 scenario="SLACKOFF",
                                 wob_lbf=0.0, bit_torque_lbf_ft=0.0,
                                 off_bottom_only=True)
        fig.add_trace(go.Scatter(
            x=df["M_lbf_ft"]/1000.0, y=df["md_ft"],
            mode="lines", name=f"μ={mu:.2f}", line=dict(width=2)
        ))
    if tube_torque_kft is not None:
        fig.add_vline(x=tube_torque_kft/sf, line=dict(color="#66e0ff", width=2, dash="dash"),
                      annotation_text=f"Tube torque @ SF {sf}", annotation_position="top left")
    if combined_kft is not None:
        fig.add_vline(x=combined_kft/sf, line=dict(color="#00c2ff", width=2),
                      annotation_text=f"Combined @ SF {sf}", annotation_position="top left")

    fig.update_layout(
        height=600, margin=dict(l=10, r=10, b=10, t=30),
        xaxis_title="Off-bottom surface torque (k lbf·ft)",
        yaxis_title="Depth / MD (ft)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.02),
        title="Drag Risk Off-Bottom Torque (μ sweep)"
    )
    return fig

def plot_elem_torque(df_elem, tube_torque_kft=None, combined_kft=None, sf=1.10):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_elem["M_lbf_ft"]/1000.0, y=df_elem["md_ft"],
        mode="lines", line=dict(color="#4ea1ff", width=2), name="Elemental torque"
    ))
    if tube_torque_kft is not None:
        fig.add_vline(x=tube_torque_kft/sf, line=dict(color="#66e0ff", width=2, dash="dash"),
                      annotation_text=f"Tube torque @ SF {sf}", annotation_position="top left")
    if combined_kft is not None:
        fig.add_vline(x=combined_kft/sf, line=dict(color="#00c2ff", width=2),
                      annotation_text=f"Combined @ SF {sf}", annotation_position="top left")

    fig.update_layout(
        height=600, margin=dict(l=10, r=10, b=10, t=30),
        xaxis_title="Elemental torque (k lbf·ft)",
        yaxis_title="Depth / MD (ft)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.02),
        title="Elemental Torque @ current μ"
    )
    return fig

# ---------- UI ----------
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")
tab_trj, tab_td = st.tabs(["Trajectory & 3D schematic", "Torque & Drag"])

with tab_trj:
    st.subheader("Synthetic survey (Minimum Curvature)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kop_md = st.number_input("KOP MD (ft)", value=2000.0, min_value=0.0, step=50.0)
    with c2:
        quick_az = st.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], index=0)
        az_map = {"North (0)":0.0, "East (90)":90.0, "South (180)":180.0, "West (270)":270.0}
    with c3:
        az_deg_custom = st.number_input("Azimuth (deg, from North, clockwise)", value=az_map[quick_az], min_value=0.0, max_value=359.99, step=0.5)
    with c4:
        bur = st.number_input("Build rate (deg/100 ft)", value=3.0, min_value=0.0, step=0.5)

    c5, c6, c7 = st.columns(3)
    with c5:
        theta_hold = st.number_input("Final inclination (deg)", value=38.5, min_value=0.0, max_value=90.0, step=0.5)
    with c6:
        target_md = st.number_input("Target MD (ft)", value=10000.0, min_value=100.0, step=100.0)
    with c7:
        ds_ft = st.number_input("Step length (ft)", value=1.0, min_value=0.5, max_value=25.0, step=0.5)

    st.markdown("#### Casing / Open-hole (simple)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # default to 9-5/8''
        nom_od = st.selectbox("Last casing nominal OD", options=list(CASING_TABLE.keys()), index=0)
    with c2:
        lbft = st.selectbox("Casing lb/ft (standards)", options=list(CASING_TABLE[nom_od].keys()), index=0)
    with c3:
        cid_in = casing_id_std(nom_od, lbft)
        st.number_input("Casing ID (in) — from table", value=cid_in, format="%.3f", disabled=True)
    with c4:
        shoe_md = st.number_input("Deepest shoe MD (ft)", value=3000.0, min_value=0.0, step=100.0)

    c5, c6 = st.columns(2)
    with c5:
        oh_diam_in = st.number_input("Open-hole diameter (in)", value=8.5, min_value=4.0, step=0.25)

    # Build survey and store in session
    df_svy = build_hold_survey(kop_md, bur, theta_hold, target_md, az_deg_custom, ds_ft=ds_ft)
    st.session_state["svy_df"] = df_svy
    st.session_state["shoe_md"] = float(shoe_md)

    st.plotly_chart(plot_3d(df_svy, shoe_md), use_container_width=True)
    st.markdown("##### Survey sample (first 11 rows)")
    st.dataframe(df_svy.head(11), use_container_width=True)
    st.caption("Minimum curvature with ratio factor and DLS (deg/100 ft). 3D uses depth-down convention.")

with tab_td:
    st.subheader("Soft-string T&D (Johancsik) — uses current trajectory")

    if "svy_df" not in st.session_state:
        st.warning("Define a trajectory first (see the Trajectory tab).")
        st.stop()

    df_svy = st.session_state["svy_df"]
    shoe_md = float(st.session_state.get("shoe_md", 3000.0))

    # friction + mud
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mu_c_slide = st.number_input("μ in casing (sliding)", value=0.25, min_value=0.0, step=0.01)
    with c2:
        mu_o_slide = st.number_input("μ in open hole (sliding)", value=0.35, min_value=0.0, step=0.01)
    with c3:
        mu_c_rot = st.number_input("μ in casing (rotating)", value=0.25, min_value=0.0, step=0.01)
    with c4:
        mu_o_rot = st.number_input("μ in open hole (rotating)", value=0.35, min_value=0.0, step=0.01)

    c5, c6 = st.columns(2)
    with c5:
        mw_ppg = st.number_input("Mud weight (ppg)", value=10.0, min_value=5.0, max_value=20.0, step=0.1)
    with c6:
        st.metric("Buoyancy factor BF", f"{buoyancy_factor_ppg(mw_ppg):.3f}")

    # drillstring
    st.markdown("#### Drillstring (bit up)")
    c1, c2, c3 = st.columns(3)
    with c1:
        dc_len = st.number_input("DC length (ft)", value=600.0, min_value=0.0, step=50.0)
        dc_w   = st.number_input("DC weight (air, lb/ft)", value=66.70, min_value=1.0, step=0.1)
        dc_od  = st.number_input("DC OD (in)", value=8.0, min_value=2.0, step=0.1)
    with c2:
        hwdp_len = st.number_input("HWDP length (ft)", value=1000.0, min_value=0.0, step=50.0)
        hwdp_w   = st.number_input("HWDP weight (air, lb/ft)", value=16.0, min_value=1.0, step=0.1)
        hwdp_od  = st.number_input("HWDP OD (in)", value=3.5, min_value=2.0, step=0.1)
    with c3:
        dp_len = st.number_input("DP length (ft)", value=7000.0, min_value=0.0, step=100.0)
        dp_w   = st.number_input("DP weight (air, lb/ft)", value=19.50, min_value=1.0, step=0.1)
        dp_od  = st.number_input("DP OD (in)", value=5.0, min_value=2.375, step=0.125)

    dc   = {"L":dc_len,   "OD":dc_od,   "w":dc_w}
    hwdp = {"L":hwdp_len, "OD":hwdp_od, "w":hwdp_w}
    dp   = {"L":dp_len,   "OD":dp_od,   "w":dp_w}

    # Scenario & bit BCs
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario = st.selectbox("Scenario", ["Slack-off (RIH)", "Pickup (POOH)"], index=0)
    with c2:
        wob_lbf = st.number_input("WOB (lbf) for on-bottom", value=6000.0, min_value=0.0, step=500.0)
    with c3:
        bit_torque = st.number_input("Bit torque (lbf·ft) for on-bottom", value=0.0, min_value=0.0, step=100.0)

    # API 7G style torque lines (user-supplied values with safety factor)
    st.markdown("#### Reference torque lines (API RP 7G style)")
    c1, c2, c3 = st.columns(3)
    with c1:
        tube_torque_kft = st.number_input("Tube torque limit (k lbf·ft)", value=16.0, min_value=1.0, step=0.5)
    with c2:
        combined_kft = st.number_input("Combined load limit (k lbf·ft)", value=20.0, min_value=1.0, step=0.5)
    with c3:
        sf = st.number_input("Safety factor", value=1.10, min_value=1.0, step=0.01)

    # Current elemental (right chart)
    df_elem = soft_string_stepper(
        df_svy,
        mu_slide_cased=mu_c_slide, mu_slide_open=mu_o_slide,
        mu_rot_cased=mu_c_rot,   mu_rot_open=mu_o_rot,
        mw_ppg=mw_ppg,
        dc_len_ft=dc["L"], dc_od_in=dc["OD"], dc_w_air_lbft=dc["w"],
        hwdp_len_ft=hwdp["L"], hwdp_od_in=hwdp["OD"], hwdp_w_air_lbft=hwdp["w"],
        dp_len_ft=dp["L"], dp_od_in=dp["OD"], dp_w_air_lbft=dp["w"],
        shoe_md_ft=shoe_md,
        scenario=scenario,
        wob_lbf=wob_lbf, bit_torque_lbf_ft=bit_torque,
        off_bottom_only=False
    )
    T_surf = float(df_elem["T_lbf"].iloc[-1])
    M_surf = float(df_elem["M_lbf_ft"].iloc[-1])
    st.success(f"Surface hookload: {T_surf:,.0f} lbf — Surface torque: {M_surf:,.0f} lbf·ft")

    # μ-sweep (left chart)
    mu_default = [0.15, 0.20, 0.25, 0.30, 0.35]
    mu_values = st.multiselect("μ sweep for off-bottom risk curves", mu_default, default=mu_default)
    if not mu_values:
        mu_values = [0.25]

    colL, colR = st.columns(2)
    with colL:
        fig_mu = plot_mu_sweep(df_svy, shoe_md, mw_ppg, mu_values, dc, hwdp, dp,
                               tube_torque_kft, combined_kft, sf)
        st.plotly_chart(fig_mu, use_container_width=True)
    with colR:
        fig_elem = plot_elem_torque(df_elem, tube_torque_kft, combined_kft, sf)
        st.plotly_chart(fig_elem, use_container_width=True)
