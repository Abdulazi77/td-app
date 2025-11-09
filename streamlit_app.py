# streamlit_app.py — 3D J/S/Horizontal well path using Minimum Curvature (MCM)
# Minimal imports for fast cold starts; ASCII-only to avoid syntax errors.

import math
from typing import Tuple
import streamlit as st

st.set_page_config(page_title="3D Well Trajectory — J / S / Horizontal", layout="wide")
st.title("3D Well Trajectory (J / S / Horizontal) — Minimum Curvature")

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ---------- Minimum Curvature core ----------
def _rf(dogs_rad: float) -> float:
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
    dogs = math.acos(cos_dog)
    rf = _rf(dogs)
    dN = 0.5 * ds * (math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2)) * rf
    dE = 0.5 * ds * (math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2)) * rf
    dT = 0.5 * ds * (math.cos(inc1) + math.cos(inc2)) * rf
    n2, e2, tvd2 = n1 + dN, e1 + dE, tvd1 + dT
    dls = 0.0 if ds <= 0 else dogs * RAD2DEG / ds * 100.0
    return n2, e2, tvd2, dls

# ---------- Synthetic survey builders (J / S / Horizontal) ----------
def synth_build_hold(kop_md, build_rate, theta_hold, target_md, ds, az_deg):
    md, inc, az = [0.0], [0.0], [az_deg]
    cur_md, built = 0.0, False
    while True:
        if target_md is not None and cur_md >= target_md - 1e-6: break
        step, next_md = ds, cur_md + ds
        prev_inc, nxt_inc = inc[-1], inc[-1]
        if (not built) and next_md >= kop_md:
            nxt_inc = min(prev_inc + build_rate * (step/100.0), theta_hold)
            built = abs(nxt_inc - theta_hold) < 1e-9
        if built: nxt_inc = theta_hold
        md.append(next_md); inc.append(nxt_inc); az.append(az_deg)
        cur_md = next_md
        if len(md) > 25000: break
    return md, inc, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len, drop_rate,
                          final_inc, target_md, ds, az_deg):
    md, inc, az = [0.0], [0.0], [az_deg]
    cur_md, built, in_hold, in_drop = 0.0, False, False, False
    hold_left = hold_len
    while True:
        if target_md is not None and cur_md >= target_md - 1e-6: break
        step, next_md = ds, cur_md + ds
        prev_inc, nxt_inc = inc[-1], inc[-1]
        if next_md >= kop_md:
            if not built:
                nxt_inc = min(prev_inc + build_rate * (step/100.0), theta_hold)
                built = abs(nxt_inc - theta_hold) < 1e-9
                in_hold = built and (hold_left is not None) and (hold_left > 0)
            elif in_hold:
                nxt_inc = theta_hold
                if hold_left is not None:
                    hold_left -= step
                    if hold_left <= 0: in_hold, in_drop = False, True
            elif in_drop:
                nxt_inc = max(prev_inc - drop_rate * (step/100.0), final_inc)
                if abs(nxt_inc - final_inc) < 1e-9: in_drop = False
            else:
                nxt_inc = final_inc
        md.append(next_md); inc.append(nxt_inc); az.append(az_deg)
        cur_md = next_md
        if len(md) > 25000: break
    return md, inc, az

def synth_horizontal(kop_md, build_rate, lateral_len, target_md, ds, az_deg):
    md, inc, az = [0.0], [0.0], [az_deg]
    cur_md, built90 = 0.0, False
    lat_left = lateral_len
    while True:
        if target_md is not None and cur_md >= target_md - 1e-6: break
        step, next_md = ds, cur_md + ds
        prev_inc, nxt_inc = inc[-1], inc[-1]
        if next_md >= kop_md:
            if not built90:
                nxt_inc = min(prev_inc + build_rate * (step/100.0), 90.0)
                built90 = abs(nxt_inc - 90.0) < 1e-9
            else:
                nxt_inc = 90.0
                if lat_left is not None:
                    lat_left -= step
                    if lat_left <= 0:
                        next_md = cur_md + (step + lat_left)
                        md.append(next_md); inc.append(nxt_inc); az.append(az_deg)
                        break
        md.append(next_md); inc.append(nxt_inc); az.append(az_deg)
        cur_md = next_md
        if len(md) > 25000: break
    return md, inc, az

def mcm_positions(md, inc, az):
    north, east, tvd, dls = [0.0], [0.0], [0.0], [0.0]
    for i in range(1, len(md)):
        n2, e2, t2, d = _mcm_step(md[i-1], inc[i-1], az[i-1], md[i], inc[i], az[i],
                                  north[-1], east[-1], tvd[-1])
        north.append(n2); east.append(e2); tvd.append(t2); dls.append(d)
    return north, east, tvd, dls

# ---------------- Sidebar UI (lightweight) ----------------
with st.sidebar:
    st.header("Well profile")
    profile = st.selectbox("Select profile", [
        "Build & Hold",
        "Build & Hold & Drop",
        "Horizontal (Continuous Build + Lateral)"
    ], index=0)
    st.header("Survey setup")
    ds_ft = st.selectbox("Course length (ft)", [10, 20, 30, 50, 100], index=2)
    kop_md = st.number_input("KOP MD (ft)", min_value=0.0, value=1000.0, step=50.0)
    az_label = st.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], index=0)
    az_map = {"North (0)": 0.0, "East (90)": 90.0, "South (180)": 180.0, "West (270)": 270.0}
    az_deg = st.number_input("Azimuth (deg from North, clockwise)",
                             min_value=0.0, max_value=360.0,
                             value=az_map[az_label], step=1.0)
    st.header("Build/Drop parameters")
    build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)

    theta_hold = hold_len = drop_rate = final_inc = lat_len = target_md = None
    if profile == "Build & Hold":
        theta_hold = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 1.0)
        target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=10000.0, step=100.0)
    elif profile == "Build & Hold & Drop":
        theta_hold = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 1.0)
        hold_len = st.number_input("Hold length (ft)", 0.0, None, 1000.0, 100.0)
        drop_rate = st.selectbox("Drop rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
        final_inc = st.number_input("Final inclination after drop (deg)", 0.0, 90.0, 0.0, 1.0)
        target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=12000.0, step=100.0)
    else:
        lat_len = st.number_input("Lateral length (ft)", 0.0, None, 2000.0, 100.0)
        target_md = st.number_input("Target MD (ft, optional; 0 = auto)", 0.0, None, 0.0, 100.0)
        if target_md == 0.0:
            target_md = None
    go_btn = st.button("Compute trajectory")

# ---------------- Run & render ----------------
if go_btn:
    # Build synthetic survey
    if profile == "Build & Hold":
        md, inc, az = synth_build_hold(kop_md, build_rate, theta_hold, target_md, ds_ft, az_deg)
    elif profile == "Build & Hold & Drop":
        md, inc, az = synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len,
                                            drop_rate, final_inc, target_md, ds_ft, az_deg)
    else:
        md, inc, az = synth_horizontal(kop_md, build_rate, lat_len, target_md, ds_ft, az_deg)

    # Positions via Minimum Curvature
    north, east, tvd, dls = mcm_positions(md, inc, az)

    # Lazy import Plotly only when needed (faster first paint)
    import plotly.graph_objects as go

    # 3D line
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=east, y=north, z=[-x for x in tvd],
                                 mode="lines", line=dict(width=6), name="Well path"))
    fig3d.update_layout(title="3D Well Trajectory",
                        scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
                        margin=dict(l=0, r=0, t=40, b=0))
    c1, c2 = st.columns([2,1])
    with c1:
        st.plotly_chart(fig3d, use_container_width=True)

    # 2D schematics
    prof = go.Figure()
    prof.add_trace(go.Scatter(x=md, y=tvd, mode="lines", name="Profile"))
    prof.update_yaxes(autorange="reversed")
    prof.update_layout(title="Profile: TVD vs MD", xaxis_title="MD (ft)", yaxis_title="TVD (ft)")
    plan = go.Figure()
    plan.add_trace(go.Scatter(x=east, y=north, mode="lines", name="Plan"))
    plan.update_layout(title="Plan: East vs North", xaxis_title="East (ft)", yaxis_title="North (ft)")
    with c2:
        st.plotly_chart(prof, use_container_width=True)
        st.plotly_chart(plan, use_container_width=True)

    # Table + CSV (no pandas)
    rows = [{"MD (ft)": md[i], "Inc (deg)": inc[i], "Az (deg)": az[i],
             "TVD (ft)": tvd[i], "North (ft)": north[i], "East (ft)": east[i],
             "DLS (deg/100 ft)": dls[i]} for i in range(len(md))]
    st.subheader("Survey and calculated positions")
    st.dataframe(rows, use_container_width=True, height=420)

    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)
    st.download_button("Download trajectory CSV", data=buf.getvalue().encode("utf-8"),
                       file_name="trajectory.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Minimum Curvature — equations used")
    st.latex(r"\cos\Delta\sigma=\cos\theta_1\cos\theta_2+\sin\theta_1\sin\theta_2\cos(\phi_2-\phi_1)")
    st.latex(r"\mathrm{RF}=\frac{2}{\Delta\sigma}\tan\left(\frac{\Delta\sigma}{2}\right)")
    st.latex(r"\Delta N=\frac{\Delta s}{2}\left(\sin\theta_1\cos\phi_1+\sin\theta_2\cos\phi_2\right)\mathrm{RF}")
    st.latex(r"\Delta E=\frac{\Delta s}{2}\left(\sin\theta_1\sin\phi_1+\sin\theta_2\sin\phi_2\right)\mathrm{RF}")
    st.latex(r"\Delta \mathrm{TVD}=\frac{\Delta s}{2}\left(\cos\theta_1+\cos\theta_2\right)\mathrm{RF}")
    st.caption("Standard Minimum Curvature Method used for directional survey calculations.")
