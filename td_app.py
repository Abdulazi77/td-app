# td_app.py
# Streamlit UI for 3D trajectory of J, S, and Horizontal wells using MCM.
# ASCII-only. Requires: wellpath.py, standard_options.json

import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wellpath import build_survey, mcm_positions

st.set_page_config(page_title="Well Trajectory Builder (J, S, Horizontal)", layout="wide")

st.title("Well Trajectory Builder — J, S, Horizontal")

# Load standard options (fallback if JSON is missing)
try:
    import json
    with open("standard_options.json", "r") as f:
        OPT = json.load(f)
except Exception:
    OPT = {
        "units": ["field (ft, deg)"],
        "well_types": ["Build & Hold", "Build & Hold & Drop", "Horizontal (Continuous Build + Lateral)"],
        "build_rates_deg_per_100ft": [0.5,1,1.5,2,3,4,6,8,10],
        "drop_rates_deg_per_100ft": [0.5,1,1.5,2,3,4,6,8,10],
        "course_lengths_ft": [10,20,30,50,100],
        "quick_azimuths": [
            {"label":"North (0)","deg":0},
            {"label":"East (90)","deg":90},
            {"label":"South (180)","deg":180},
            {"label":"West (270)","deg":270}
        ]
    }

colA, colB = st.columns(2)
with colA:
    units = st.selectbox("Units", OPT["units"], index=0)
with colB:
    ds_ft = st.selectbox("Course length (ft)", OPT["course_lengths_ft"], index=2)

# Simple, reliable quick-azimuth selection
label_list = [x["label"] for x in OPT["quick_azimuths"]]
label = st.selectbox("Quick azimuth", label_list, index=0)
label_to_deg = {x["label"]: x["deg"] for x in OPT["quick_azimuths"]}
default_deg = float(label_to_deg[label])
azimuth_deg = st.number_input("Azimuth (deg from North, clockwise)",
                              min_value=0.0, max_value=360.0,
                              value=default_deg, step=1.0)


# Common inputs
kop_md = st.number_input("KOP MD (ft)", min_value=0.0, value=1000.0, step=50.0)
build_rate = st.selectbox("Build rate (deg/100 ft)", OPT["build_rates_deg_per_100ft"], index=4)

theta_hold = None
target_md = None
target_tvd = None
hold_length = None
drop_rate = None
final_inc_after_drop = None
lateral_length = None

if profile == "Build & Hold":
    theta_hold = st.number_input("Final inclination (deg)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
    colA, colB = st.columns(2)
    with colA:
        use_target_md = st.checkbox("Stop at Target MD", value=True)
    if use_target_md:
        target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=10000.0, step=100.0)
    else:
        target_tvd = st.number_input("Target TVD (ft)", min_value=0.0, value=8000.0, step=50.0)

elif profile == "Build & Hold & Drop":
    theta_hold = st.number_input("Hold inclination (deg)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
    hold_length = st.number_input("Hold length (ft)", min_value=0.0, value=1000.0, step=100.0)
    drop_rate = st.selectbox("Drop rate (deg/100 ft)", OPT["drop_rates_deg_per_100ft"], index=4)
    final_inc_after_drop = st.number_input("Final inclination after drop (deg)", min_value=0.0, max_value=90.0, value=0.0, step=1.0)
    colA, colB = st.columns(2)
    with colA:
        use_target_md = st.checkbox("Stop at Target MD", value=True)
    if use_target_md:
        target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=12000.0, step=100.0)
    else:
        target_tvd = st.number_input("Target TVD (ft)", min_value=0.0, value=9000.0, step=50.0)

else:  # Horizontal (Continuous Build + Lateral)
    lateral_length = st.number_input("Lateral length (ft)", min_value=0.0, value=2000.0, step=100.0)
    colA, colB = st.columns(2)
    with colA:
        target_md = st.number_input("Target MD (ft) (optional)", min_value=0.0, value=0.0, step=100.0)
        if target_md == 0.0:
            target_md = None
    theta_hold = None  # implied 90 deg

st.markdown("---")
if st.button("Compute trajectory"):
    df = build_survey(
        profile=profile,
        kop_md_ft=float(kop_md),
        azimuth_deg=float(azimuth_deg),
        ds_ft=float(ds_ft),
        build_rate=float(build_rate),
        theta_hold_deg=theta_hold,
        target_md_ft=target_md,
        target_tvd_ft=target_tvd,
        hold_length_ft=hold_length,
        drop_rate=drop_rate,
        final_inc_after_drop_deg=final_inc_after_drop,
        lateral_length_ft=lateral_length
    )
    # Convert to positions using MCM
    df_pos = mcm_positions(df)

    # If user picked Target TVD stop, trim where we pass it
    if target_tvd is not None:
        mask = df_pos["TVD_ft"] <= float(target_tvd) + 0.5 * ds_ft
        if mask.any():
            df_pos = df_pos.loc[mask]

    # 3D plot (Plotly offline)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df_pos["East_ft"], y=df_pos["North_ft"], z=-df_pos["TVD_ft"],
        mode="lines",
        line=dict(width=6),
        name="Well path"
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="East (ft)",
            yaxis_title="North (ft)",
            zaxis_title="TVD (ft, down)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="3D Well Trajectory"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Survey and calculated positions")
    nice = df_pos.rename(columns={
        "MD_ft": "MD (ft)",
        "Inc_deg": "Inc (deg)",
        "Az_deg": "Az (deg)",
        "TVD_ft": "TVD (ft)",
        "North_ft": "North (ft)",
        "East_ft": "East (ft)",
        "DLS_deg_per_100ft": "DLS (deg/100 ft)"
    })
    st.dataframe(nice, height=400)

    # Download
    csv = nice.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="trajectory.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("What the columns mean")
    st.markdown(
        "- **MD (ft)**: measured depth along the wellbore.\n"
        "- **Inc (deg)**: inclination from vertical.\n"
        "- **Az (deg)**: azimuth from North, clockwise.\n"
        "- **TVD (ft)**: true vertical depth.\n"
        "- **North/East (ft)**: map displacements.\n"
        "- **DLS (deg/100 ft)**: dogleg severity per interval."
    )

    st.subheader("Equations used (Minimum Curvature)")
    st.latex(r"\cos \Delta\sigma = \cos\theta_1\cos\theta_2 + \sin\theta_1\sin\theta_2\cos(\phi_2-\phi_1)")
    st.latex(r"\text{RF} = \frac{2}{\Delta\sigma}\tan\left(\frac{\Delta\sigma}{2}\right)")
    st.latex(r"\Delta N = \frac{\Delta s}{2}(\sin\theta_1\cos\phi_1 + \sin\theta_2\cos\phi_2)\,\text{RF}")
    st.latex(r"\Delta E = \frac{\Delta s}{2}(\sin\theta_1\sin\phi_1 + \sin\theta_2\sin\phi_2)\,\text{RF}")
    st.latex(r"\Delta \text{TVD} = \frac{\Delta s}{2}(\cos\theta_1 + \cos\theta_2)\,\text{RF}")
    st.caption("Formulas per the standard Minimum Curvature Method used in directional drilling.")
