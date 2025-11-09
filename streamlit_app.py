# streamlit_app.py  —  ASCII only
# 3D Well Trajectory Builder for: Build & Hold, Build & Hold & Drop, Horizontal
# Uses Minimum Curvature Method (MCM) for TVD/N/E and DLS.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wellpath import build_survey, mcm_positions

st.set_page_config(page_title="3D Well Trajectory — J/S/Horizontal", layout="wide")
st.title("3D Well Trajectory Builder (J / S / Horizontal)")

# ---------------- Sidebar: inputs ----------------
with st.sidebar:
    st.header("Well Profile")
    profile = st.selectbox("Select profile", [
        "Build & Hold",
        "Build & Hold & Drop",
        "Horizontal (Continuous Build + Lateral)"
    ], index=0)

    st.header("Survey setup")
    ds_ft = st.selectbox("Course length (step, ft)", [10, 20, 30, 50, 100], index=2)
    kop_md = st.number_input("KOP MD (ft)", min_value=0.0, value=1000.0, step=50.0)

    # quick azimuth pick + explicit override
    az_default = st.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], index=0)
    quick_map = {"North (0)": 0.0, "East (90)": 90.0, "South (180)": 180.0, "West (270)": 270.0}
    azimuth_deg = st.number_input("Azimuth (deg from North, clockwise)",
                                  min_value=0.0, max_value=360.0,
                                  value=quick_map[az_default], step=1.0)

    st.header("Build/Drop parameters")
    build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)

    # Profile-specific inputs
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

    st.header("Run")
    go_btn = st.button("Compute trajectory")

# ---------------- Run & outputs ----------------
if go_btn:
    # Build synthetic survey
    survey = build_survey(
        profile=profile,
        kop_md_ft=float(kop_md),
        azimuth_deg=float(azimuth_deg),
        ds_ft=float(ds_ft),
        build_rate=float(build_rate),
        theta_hold_deg=theta_hold,
        target_md_ft=target_md,
        hold_length_ft=hold_length,
        drop_rate=drop_rate,
        final_inc_after_drop_deg=final_inc_after_drop,
        lateral_length_ft=lateral_length
    )

    # Convert to positions (MCM)
    pos = mcm_positions(survey)

    # 3D line
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=pos["East_ft"], y=pos["North_ft"], z=-pos["TVD_ft"],
        mode="lines", name="Well path", line=dict(width=6)
    ))
    fig3d.update_layout(
        title="3D Well Trajectory",
        scene=dict(
            xaxis_title="East (ft)",
            yaxis_title="North (ft)",
            zaxis_title="TVD (ft, down)"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig3d, use_container_width=True)

    # 2D schematics: Profile (TVD vs MD) and Plan (East vs North)
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

    # Table
    st.subheader("Survey and calculated positions")
    nice = pos.rename(columns={
        "MD_ft":"MD (ft)", "Inc_deg":"Inc (deg)", "Az_deg":"Az (deg)",
        "TVD_ft":"TVD (ft)", "North_ft":"North (ft)", "East_ft":"East (ft)",
        "DLS_deg_per_100ft":"DLS (deg/100 ft)"
    })
    st.dataframe(nice, height=420, use_container_width=True)

    # Download CSV
    st.download_button("Download trajectory CSV",
                       data=nice.to_csv(index=False).encode("utf-8"),
                       file_name="trajectory.csv",
                       mime="text/csv")

    # Column definitions & equations
    st.markdown("---")
    st.subheader("Column definitions")
    st.markdown(
        "- **MD (ft)**: Measured Depth along wellbore.\n"
        "- **Inc (deg)**: inclination from vertical.\n"
        "- **Az (deg)**: azimuth from North, clockwise.\n"
        "- **TVD (ft)**: True Vertical Depth.\n"
        "- **North/East (ft)**: map displacements.\n"
        "- **DLS (deg/100 ft)**: dogleg severity per 100 ft."
    )

    st.subheader("Minimum Curvature Method (formulas)")
    st.latex(r"\cos\Delta\sigma=\cos\theta_1\cos\theta_2+\sin\theta_1\sin\theta_2\cos(\phi_2-\phi_1)")
    st.latex(r"\text{RF}=\frac{2}{\Delta\sigma}\tan\left(\frac{\Delta\sigma}{2}\right)")
    st.latex(r"\Delta N=\frac{\Delta s}{2}\left(\sin\theta_1\cos\phi_1+\sin\theta_2\cos\phi_2\right)\text{RF}")
    st.latex(r"\Delta E=\frac{\Delta s}{2}\left(\sin\theta_1\sin\phi_1+\sin\theta_2\sin\phi_2\right)\text{RF}")
    st.latex(r"\Delta \mathrm{TVD}=\frac{\Delta s}{2}\left(\cos\theta_1+\cos\theta_2\right)\text{RF}")
    st.caption("These are the standard MCM survey equations used in directional drilling.")
