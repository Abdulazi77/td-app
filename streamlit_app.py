# streamlit_app.py  —  ASCII only
# 3D Well Trajectory (Build & Hold, Build & Hold & Drop, Horizontal + Lateral)
# Uses Minimum Curvature Method (MCM): dogleg angle, ratio factor, ΔN/ΔE/ΔTVD.

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="3D Well Trajectory — J / S / Horizontal", layout="wide")
st.title("3D Well Trajectory (J / S / Horizontal) — Minimum Curvature")

# ------------------------ MCM utilities (self-contained) ------------------------
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def _rf(dogs_rad: float) -> float:
    # Ratio Factor for minimum curvature: RF = 2/dogs * tan(dogs/2)
    if abs(dogs_rad) < 1e-12:
        return 1.0
    return 2.0 / dogs_rad * math.tan(0.5 * dogs_rad)

def _mcm_step(md1, inc1_deg, az1_deg, md2, inc2_deg, az2_deg,
              n1, e1, tvd1) -> Tuple[float, float, float, float]:
    ds = md2 - md1
    inc1 = inc1_deg * DEG2RAD; inc2 = inc2_deg * DEG2RAD
    az1  = az1_deg  * DEG2RAD; az2  = az2_deg  * DEG2RAD

    cos_dog = (math.cos(inc1)*math.cos(inc2) +
               math.sin(inc1)*math.sin(inc2)*math.cos(az2-az1))
    cos_dog = max(-1.0, min(1.0, cos_dog))
    dogs = math.acos(cos_dog)  # dogleg angle (rad)
    rf = _rf(dogs)

    dN = 0.5 * ds * (math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2)) * rf
    dE = 0.5 * ds * (math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2)) * rf
    dT = 0.5 * ds * (math.cos(inc1) + math.cos(inc2)) * rf

    n2 = n1 + dN; e2 = e1 + dE; tvd2 = tvd1 + dT
    dls = 0.0 if ds <= 0 else dogs * RAD2DEG / ds * 100.0
    return n2, e2, tvd2, dls

# ------------------------ Synthetic survey builders ------------------------
def synth_build_hold(kop_md, build_rate_deg_per_100, theta_hold_deg,
                     target_md, ds, azimuth_deg) -> pd.DataFrame:
    md = [0.0]; inc = [0.0]; az = [azimuth_deg]
    cur_md = 0.0; built = False
    while True:
        if target_md is not None and cur_md >= target_md - 1e-6:
            break
        step = ds
        next_md = cur_md + step
        prev_inc = inc[-1]
        nxt_inc = prev_inc
        if (not built) and next_md >= kop_md:
            # in build
            if prev_inc < theta_hold_deg:
                nxt_inc = min(prev_inc + build_rate_deg_per_100 * (step/100.0), theta_hold_deg)
                if abs(nxt_inc - theta_hold_deg) < 1e-9:
                    built = True
            else:
                built = True
        if built:
            nxt_inc = theta_hold_deg
        md.append(next_md); inc.append(nxt_inc); az.append(azimuth_deg)
        cur_md = next_md
        if len(md) > 25000: break
    return pd.DataFrame({"MD_ft": md, "Inc_deg": inc, "Az_deg": az})

def synth_build_hold_drop(kop_md, build_rate_deg_per_100, theta_hold_deg,
                          hold_len_ft, drop_rate_deg_per_100, final_inc_deg,
                          target_md, ds, azimuth_deg) -> pd.DataFrame:
    md = [0.0]; inc = [0.0]; az = [azimuth_deg]
    cur_md = 0.0; built = False; in_hold = False; in_drop = False
    hold_left = hold_len_ft
    while True:
        if target_md is not None and cur_md >= target_md - 1e-6:
            break
        step = ds
        next_md = cur_md + step
        prev_inc = inc[-1]; nxt_inc = prev_inc
        if next_md >= kop_md:
            if not built:
                nxt_inc = min(prev_inc + build_rate_deg_per_100 * (step/100.0), theta_hold_deg)
                if abs(nxt_inc - theta_hold_deg) < 1e-9:
                    built = True
                    in_hold = hold_left is not None and hold_left > 0
            elif in_hold:
                nxt_inc = theta_hold_deg
                if hold_left is not None:
                    hold_left -= step
                    if hold_left <= 0:
                        in_hold = False; in_drop = True
            elif in_drop:
                nxt_inc = max(prev_inc - drop_rate_deg_per_100 * (step/100.0), final_inc_deg)
                if abs(nxt_inc - final_inc_deg) < 1e-9:
                    in_drop = False
            else:
                nxt_inc = final_inc_deg
        md.append(next_md); inc.append(nxt_inc); az.append(azimuth_deg)
        cur_md = next_md
        if len(md) > 25000: break
    return pd.DataFrame({"MD_ft": md, "Inc_deg": inc, "Az_deg": az})

def synth_horizontal(kop_md, build_rate_deg_per_100, lateral_len_ft,
                     target_md, ds, azimuth_deg) -> pd.DataFrame:
    md = [0.0]; inc = [0.0]; az = [azimuth_deg]
    cur_md = 0.0; built_to_90 = False; lateral_left = lateral_len_ft
    while True:
        if target_md is not None and cur_md >= target_md - 1e-6:
            break
        step = ds
        next_md = cur_md + step
        prev_inc = inc[-1]; nxt_inc = prev_inc
        if next_md >= kop_md:
            if not built_to_90:
                nxt_inc = min(prev_inc + build_rate_deg_per_100 * (step/100.0), 90.0)
                built_to_90 = abs(nxt_inc - 90.0) < 1e-9
            else:
                nxt_inc = 90.0
                if lateral_left is not None:
                    lateral_left -= step
                    if lateral_left <= 0:
                        next_md = cur_md + (step + lateral_left)
                        md.append(next_md); inc.append(nxt_inc); az.append(azimuth_deg)
                        break
        md.append(next_md); inc.append(nxt_inc); az.append(azimuth_deg)
        cur_md = next_md
        if len(md) > 25000: break
    return pd.DataFrame({"MD_ft": md, "Inc_deg": inc, "Az_deg": az})

def mcm_positions(df_md_inc_az: pd.DataFrame) -> pd.DataFrame:
    md = df_md_inc_az["MD_ft"].to_numpy()
    inc = df_md_inc_az["Inc_deg"].to_numpy()
    az  = df_md_inc_az["Az_deg"].to_numpy()
    north = [0.0]; east = [0.0]; tvd = [0.0]; dls = [0.0]
    for i in range(1, len(md)):
        n2, e2, t2, dls_i = _mcm_step(md[i-1], inc[i-1], az[i-1],
                                      md[i],   inc[i],   az[i],
                                      north[-1], east[-1], tvd[-1])
        north.append(n2); east.append(e2); tvd.append(t2); dls.append(dls_i)
    out = df_md_inc_az.copy()
    out["TVD_ft"] = tvd; out["North_ft"] = north; out["East_ft"] = east
    out["DLS_deg_per_100ft"] = dls
    return out

# ------------------------ Sidebar UI ------------------------
with st.sidebar:
    st.header("Well profile")
    profile = st.selectbox("Select profile", [
        "Build & Hold",
        "Build & Hold & Drop",
        "Horizontal (Continuous Build + Lateral)"
    ], index=0)

    st.header("Survey setup")
    ds_ft = st.selectbox("Course length (step, ft)", [10, 20, 30, 50, 100], index=2)
    kop_md = st.number_input("KOP MD (ft)", min_value=0.0, value=1000.0, step=50.0)

    az_label = st.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], index=0)
    az_map = {"North (0)": 0.0, "East (90)": 90.0, "South (180)": 180.0, "West (270)": 270.0}
    azimuth_deg = st.number_input("Azimuth (deg from North, clockwise)",
                                  min_value=0.0, max_value=360.0,
                                  value=az_map[az_label], step=1.0)

    st.header("Build/Drop parameters")
    build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)

    # Per-profile inputs
    theta_hold = None
    target_md = None
    hold_length = None
    drop_rate = None
    final_inc_after_drop = None
    lateral_length = None

    if profile == "Build & Hold":
        theta_hold = st.number_input("Final inclination (deg)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
        target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=10000.0, step=100.0)

    elif profile == "Build & Hold & Drop":
        theta_hold = st.number_input("Hold inclination (deg)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
        hold_length = st.number_input("Hold length (ft)", min_value=0.0, value=1000.0, step=100.0)
        drop_rate = st.selectbox("Drop rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
        final_inc_after_drop = st.number_input("Final inclination after drop (deg)", min_value=0.0, max_value=90.0, value=0.0, step=1.0)
        target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=12000.0, step=100.0)

    else:  # Horizontal
        lateral_length = st.number_input("Lateral length (ft)", min_value=0.0, value=2000.0, step=100.0)
        target_md = st.number_input("Target MD (ft, optional, 0 = auto)", min_value=0.0, value=0.0, step=100.0)
        if target_md == 0.0:
            target_md = None

    go_btn = st.button("Compute trajectory")

# ------------------------ Run ------------------------
if go_btn:
    if profile == "Build & Hold":
        survey = synth_build_hold(kop_md, build_rate, theta_hold, target_md, ds_ft, azimuth_deg)
    elif profile == "Build & Hold & Drop":
        survey = synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_length,
                                       drop_rate, final_inc_after_drop, target_md, ds_ft, azimuth_deg)
    else:
        survey = synth_horizontal(kop_md, build_rate, lateral_length, target_md, ds_ft, azimuth_deg)

    pos = mcm_positions(survey)

    # 3D plot
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=pos["East_ft"], y=pos["North_ft"], z=-pos["TVD_ft"],
        mode="lines", line=dict(width=6), name="Well path"
    ))
    fig3d.update_layout(
        title="3D Well Trajectory",
        scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    c1, c2 = st.columns([2,1])
    with c1:
        st.plotly_chart(fig3d, use_container_width=True)

    # 2D schematics
    prof = go.Figure()
    prof.add_trace(go.Scatter(x=pos["MD_ft"], y=pos["TVD_ft"], mode="lines", name="Profile"))
    prof.update_yaxes(autorange="reversed")
    prof.update_layout(title="Profile: TVD vs MD", xaxis_title="MD (ft)", yaxis_title="TVD (ft)")
    plan = go.Figure()
    plan.add_trace(go.Scatter(x=pos["East_ft"], y=pos["North_ft"], mode="lines", name="Plan"))
    plan.update_layout(title="Plan: East vs North", xaxis_title="East (ft)", yaxis_title="North (ft)")
    with c2:
        st.plotly_chart(prof, use_container_width=True)
        st.plotly_chart(plan, use_container_width=True)

    # Table + download
    nice = pos.rename(columns={
        "MD_ft":"MD (ft)", "Inc_deg":"Inc (deg)", "Az_deg":"Az (deg)",
        "TVD_ft":"TVD (ft)", "North_ft":"North (ft)", "East_ft":"East (ft)",
        "DLS_deg_per_100ft":"DLS (deg/100 ft)"
    })
    st.subheader("Survey and calculated positions")
    st.dataframe(nice, height=420, use_container_width=True)
    st.download_button("Download trajectory CSV",
                       data=nice.to_csv(index=False).encode("utf-8"),
                       file_name="trajectory.csv", mime="text/csv")

    # Equations
    st.markdown("---")
    st.subheader("Minimum Curvature — equations")
    st.latex(r"\cos\Delta\sigma=\cos\theta_1\cos\theta_2+\sin\theta_1\sin\theta_2\cos(\phi_2-\phi_1)")
    st.latex(r"\mathrm{RF}=\frac{2}{\Delta\sigma}\tan\left(\frac{\Delta\sigma}{2}\right)")
    st.latex(r"\Delta N=\frac{\Delta s}{2}\left(\sin\theta_1\cos\phi_1+\sin\theta_2\cos\phi_2\right)\mathrm{RF}")
    st.latex(r"\Delta E=\frac{\Delta s}{2}\left(\sin\theta_1\sin\phi_1+\sin\theta_2\sin\phi_2\right)\mathrm{RF}")
    st.latex(r"\Delta \mathrm{TVD}=\frac{\Delta s}{2}\left(\cos\theta_1+\cos\theta_2\right)\mathrm{RF}")
    st.caption("Standard survey method used in industry.")
