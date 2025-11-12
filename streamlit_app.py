import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Utilities & data structures
# -----------------------------
STEEL_SG = 7.85  # ~ density/SG of steel

@dataclass
class Pipe:
    od_in: float
    id_in: float
    weight_lbft_air: float

@dataclass
class Hole:
    open_hole_d_in: float
    csg_d_in: float

@dataclass
class Params:
    step_ft: float
    mud_weight_ppg: float
    friction_factors: List[float]
    wob_klbf: float
    bit_torque_ftlbf: float
    rotation_reduction_factor: float  # multiplicative reduction of μ during "rotating weight" case


# -----------------------------
# Geometry builders (profiles)
# -----------------------------
def build_hold(md_total, md_kop, build_rate_deg_per_100ft, hold_inc_deg, step):
    """
    Build to a target inclination, then hold; azimuth assumed constant (0) for simplicity.
    Outputs arrays of MD, TVD, VS, INC (deg), AZI (deg).
    """
    n = int(np.floor(md_total/step))+1
    md = np.linspace(0, md_total, n)
    inc = np.zeros_like(md)
    azi = np.zeros_like(md)
    inc_target = hold_inc_deg

    # segment 1: vertical to KOP (inc=0)
    # segment 2: build from 0 to inc_target with specified rate
    build_rate_deg_per_ft = build_rate_deg_per_100ft/100.0
    build_length = inc_target / build_rate_deg_per_ft if build_rate_deg_per_ft > 0 else 0.0
    build_start = md_kop
    build_end = min(md_kop + build_length, md_total)

    for i, d in enumerate(md):
        if d < build_start:
            inc[i] = 0.0
        elif d <= build_end:
            inc[i] = (d - build_start) * build_rate_deg_per_ft
        else:
            inc[i] = inc_target

    # compute TVD & VS (minimum curvature approximation with small-step simplification)
    # For small steps, TVD ≈ Σ step * cos(inc), Vertical Section (VS) ≈ Σ step * sin(inc)
    inc_rad = np.deg2rad(inc)
    dTVD = np.diff(md, prepend=0) * np.cos(inc_rad)
    dVS  = np.diff(md, prepend=0) * np.sin(inc_rad)
    tvd = np.cumsum(dTVD)
    vs = np.cumsum(dVS)

    return md, tvd, vs, inc, azi


def build_hold_drop(md_total, md_kop, build_rate_deg_per_100ft,
                    target_inc_deg, md_eoc, drop_rate_deg_per_100ft, step):
    """
    Build to target_inc, hold to MD_eoc, then drop back toward vertical.
    """
    n = int(np.floor(md_total/step))+1
    md = np.linspace(0, md_total, n)
    inc = np.zeros_like(md)
    azi = np.zeros_like(md)

    br_ft = build_rate_deg_per_100ft/100
    dr_ft = drop_rate_deg_per_100ft/100

    # Build phase
    build_len = target_inc_deg / br_ft if br_ft > 0 else 0
    build_end = md_kop + build_len

    # Hold phase to md_eoc (end of curve in assignment sense)
    # Then drop until vertical or MD end
    # Determine drop start at md_eoc; estimate needed length to drop to 0°
    drop_len_needed = target_inc_deg / dr_ft if dr_ft > 0 else 0
    drop_end = md_eoc + drop_len_needed

    for i, d in enumerate(md):
        if d < md_kop:
            inc[i] = 0.0
        elif d <= build_end:
            inc[i] = (d - md_kop) * br_ft
        elif d <= md_eoc:
            inc[i] = target_inc_deg
        elif d <= drop_end:
            inc[i] = max(0.0, target_inc_deg - (d - md_eoc) * dr_ft)
        else:
            inc[i] = 0.0

    inc_rad = np.deg2rad(inc)
    dTVD = np.diff(md, prepend=0) * np.cos(inc_rad)
    dVS  = np.diff(md, prepend=0) * np.sin(inc_rad)
    tvd = np.cumsum(dTVD)
    vs = np.cumsum(dVS)
    return md, tvd, vs, inc, azi


def horizontal_continuous_build(md_total, md_kop, build_rate_deg_per_100ft, lateral_len_ft, step):
    """
    Continuous build until horizontal (90°), then lateral section of specified length.
    """
    n = int(np.floor(md_total/step))+1
    md = np.linspace(0, md_total, n)
    inc = np.zeros_like(md); azi = np.zeros_like(md)
    br_ft = build_rate_deg_per_100ft/100
    build_len = 90.0 / br_ft if br_ft > 0 else 0.0
    build_end = md_kop + build_len
    lateral_end = build_end + lateral_len_ft

    for i, d in enumerate(md):
        if d < md_kop:
            inc[i] = 0.0
        elif d <= build_end:
            inc[i] = (d - md_kop) * br_ft
        elif d <= lateral_end:
            inc[i] = 90.0
        else:
            inc[i] = 90.0  # beyond lateral_end, keep 90 if MD allows (or well just ends)

    inc_rad = np.deg2rad(inc)
    dTVD = np.diff(md, prepend=0) * np.cos(inc_rad)
    dVS  = np.diff(md, prepend=0) * np.sin(inc_rad)
    tvd = np.cumsum(dTVD)
    vs = np.cumsum(dVS)
    return md, tvd, vs, inc, azi


# -----------------------------
# Mechanics (soft-string model)
# -----------------------------
def buoyancy_factor(mud_ppg):
    """
    BF ≈ 1 - ρ_mud / ρ_steel  (with ρ as SG). Using SG ~ ppg / 8.33
    """
    sg_mud = mud_ppg / 8.33
    return max(0.0, 1.0 - sg_mud / STEEL_SG)


def dogleg_severity_deg_per_100ft(inc_deg):
    """
    Simple DLS estimate from local curvature of inclination sequence (no azimuth change).
    """
    inc = np.array(inc_deg)
    dinc = np.diff(inc, prepend=inc[0])
    # Convert per-step to per-100ft rate
    return np.abs(dinc)  # caller will scale by step to deg per ft then to deg/100 ft


def soft_string_forces(md, inc_deg, pipe_w_air_lbft, mud_ppg, step_ft,
                       mu, motion: str, contact_radius_in):
    """
    Returns arrays of tension (lb) and torque (ft-lbf) along the string for the chosen friction μ and motion.
    motion in {"pickup","slackoff","rotating"}
    """
    BF = buoyancy_factor(mud_ppg)
    w_b = pipe_w_air_lbft * BF  # buoyed weight per ft
    inc_rad = np.deg2rad(inc_deg)

    # Normal force per segment:
    # Straight-well soft-string baseline: N ≈ W*sin(inc)
    # Add curvature term: N_add ≈ T * κ  (κ = dogleg (rad/ft))
    # We'll compute iteratively from bit to surface.
    n = len(md)
    T = np.zeros(n)     # axial tension
    Torque = np.zeros(n)

    # Direction of drag:
    # pickup (POOH): drag opposes upward motion => adds to W along the string
    # slackoff (RIH): drag opposes downward motion => subtracts from W
    # rotating: reduce μ by factor for axial drag & torque to emulate friction relief while rotating (project bonus)
    mu_eff = mu
    if motion == "rotating":
        mu_eff = mu * st.session_state.get("rot_reduction_factor", 0.3)

    # geometry
    R_ft = (contact_radius_in / 12.0)

    # precompute curvature κ from DLS
    dls_deg_step = dogleg_severity_deg_per_100ft(inc_deg)  # deg per step
    dls_deg_per_ft = dls_deg_step / step_ft
    kappa_rad_per_ft = np.deg2rad(dls_deg_per_ft)

    # march from bit (index 0 assumed bottom) to surface
    # assume bit tension known from WOB (downward compression at bit; use magnitude to start above bit as 0)
    T[0] = 0.0  # just above the bit, start from zero and integrate upward

    for i in range(1, n):
        # segment i-1 -> i
        Wseg = w_b * step_ft  # buoyed weight of segment
        N_straight = Wseg * np.sin(inc_rad[i])  # axial -> normal from inclination
        N_curve = T[i-1] * kappa_rad_per_ft[i] * step_ft  # curvature-induced normal
        N = N_straight + N_curve

        Ff = mu_eff * N  # friction magnitude

        if motion == "pickup":
            # Going up: tension at top = tension at bottom + Wseg + Ff
            T[i] = T[i-1] + Wseg + Ff
        elif motion == "slackoff":
            # Going down: tension at top = tension at bottom + Wseg - Ff
            T[i] = T[i-1] + Wseg - Ff
        else:  # rotating (while holding axial), approximate axial drag reduced by μ_eff:
            T[i] = T[i-1] + Wseg + (mu_eff * N)

        # torque increment: dT = μ*N*R  (contact radius)
        dTorque = mu_eff * N * R_ft
        Torque[i] = Torque[i-1] + dTorque

    return T, Torque


def compute_all_cases(md, tvd, vs, inc, pipe: Pipe, mud_ppg, step_ft,
                      mu_list: List[float], wob_klbf, bit_torque_ftlbf, contact_radius_in, rot_reduction_factor):
    results = {}
    st.session_state["rot_reduction_factor"] = rot_reduction_factor

    for mu in mu_list:
        case = {}
        # Three motion states
        for motion in ["pickup", "slackoff", "rotating"]:
            T, Trq = soft_string_forces(
                md=md,
                inc_deg=inc,
                pipe_w_air_lbft=pipe.weight_lbft_air,
                mud_ppg=mud_ppg,
                step_ft=step_ft,
                mu=mu,
                motion=motion,
                contact_radius_in=contact_radius_in,
            )
            # add WOB at bit (compressive); for surface hookload we care about top node:
            # Hookload ≈ T_surface (ft-lbf not included). Using pounds.
            # Bit torque adds to torque profile baseline:
            Trq_total = Trq + (bit_torque_ftlbf if bit_torque_ftlbf else 0.0)
            case[motion] = {"tension_lb": T, "torque_ftlbf": Trq_total}
        results[mu] = case

    df_geom = pd.DataFrame({
        "MD (ft)": md,
        "TVD (ft)": tvd,
        "Vertical Section (ft)": vs,
        "Inclination (deg)": inc
    })
    return results, df_geom


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Torque & Drag Soft-String Simulator", layout="wide")

st.title("Torque & Drag Soft-String Simulator")
st.caption("Interactive T&D model for directional wells (soft-string). Inputs/outputs align with the project handout. "
           "Profiles: Build-Hold, Build-Hold-Drop, Horizontal with 2000-ft lateral. 1-ft step. "
           "Plots: TVD vs Vertical Section, Pickup/Slack-off/Rotating weights vs MD for multiple μ, and Torque vs Depth."
           "\nRefs: Project brief (required inputs/outputs) and course T&D notes on force balance & torque. "
           "See sidebar for parameters.")

with st.sidebar:
    st.header("Model Inputs")
    # Well geometry choice
    profile = st.selectbox("Well Profile", ["Build & Hold", "Build & Hold & Drop", "Horizontal (continuous build + 2000 ft lateral)"])

    md_total = st.number_input("Total Measured Depth, MD (ft)",  value=12000, min_value=1000, step=500)
    step_ft = st.number_input("Calculation Step (ft)", value=1, min_value=1, max_value=50, help="Use 1 ft as per assignment.")

    # Common trajectory parameters
    md_kop = st.number_input("KOP (ft)", value=2000, min_value=0, step=100)
    build_rate = st.number_input("Build Rate (deg/100 ft)", value=8.0, min_value=0.0, max_value=30.0, step=0.5)

    target_inc = st.number_input("Target Inclination (deg)", value=60.0, min_value=0.0, max_value=90.0, step=1.0)

    if profile == "Build & Hold & Drop":
        md_eoc = st.number_input("End of Hold (start of Drop) MD (ft)", value=8000, min_value=0, step=100)
        drop_rate = st.number_input("Drop Rate (deg/100 ft)", value=8.0, min_value=0.0, max_value=30.0, step=0.5)
    else:
        md_eoc, drop_rate = None, None

    if profile == "Horizontal (continuous build + 2000 ft lateral)":
        lateral_len = st.number_input("Lateral Length (ft)", value=2000, min_value=0, step=100)
    else:
        lateral_len = 0

    st.markdown("---")
    st.subheader("Drilling Parameters")
    mud_ppg = st.number_input("Mud Weight (ppg)", value=10.0, min_value=5.0, max_value=20.0, step=0.1)
    od_in = st.number_input("Pipe OD (in)", value=5.0, min_value=2.0, max_value=8.0, step=0.125)
    id_in = st.number_input("Pipe ID (in)", value=4.0, min_value=1.0, max_value=7.5, step=0.125)
    weight_air = st.number_input("Pipe Weight in Air (lb/ft)", value=19.5, min_value=5.0, max_value=70.0, step=0.5)

    hole_d_in = st.number_input("Open Hole Diameter (in)", value=8.5, min_value=3.5, max_value=17.5, step=0.5)
    csg_d_in = st.number_input("Casing ID (in)", value=0.0, help="Optional (0 if not applicable)", min_value=0.0, max_value=20.0, step=0.5)

    # Friction factors list
    mu_str = st.text_input("Friction factors (comma-separated between 0 and 1)", value="0.2, 0.3, 0.4")
    mu_list = [max(0.0, min(1.0, float(x.strip()))) for x in mu_str.split(",") if x.strip()]

    wob_klbf = st.number_input("WOB (k lbf) [optional]", value=0.0, min_value=0.0, step=1.0)
    bit_torque = st.number_input("Bit Torque (ft-lbf) [optional]", value=0.0, min_value=0.0, step=100.0)
    rot_reduction_factor = st.slider("Rotation friction reduction factor (0–1)", value=0.3, min_value=0.0, max_value=1.0)

    st.markdown("---")
    st.subheader("Advanced")
    contact_radius_in = st.number_input("Effective contact radius for torque (in)", value=od_in/2.0, min_value=0.1, max_value=hole_d_in/2.0, step=0.1)
    st.caption("Torque increment dT ≈ μ·N·R (R in ft). As a first approximation, R ≈ pipe radius.")

# Build geometry
if profile == "Build & Hold":
    md, tvd, vs, inc, azi = build_hold(md_total, md_kop, build_rate, target_inc, step_ft)
elif profile == "Build & Hold & Drop":
    md, tvd, vs, inc, azi = build_hold_drop(md_total, md_kop, build_rate, target_inc, md_eoc, drop_rate, step_ft)
else:
    md, tvd, vs, inc, azi = horizontal_continuous_build(md_total, md_kop, build_rate, lateral_len, step_ft)

pipe = Pipe(od_in=od_in, id_in=id_in, weight_lbft_air=weight_air)
results, df_geom = compute_all_cases(
    md=md, tvd=tvd, vs=vs, inc=inc, pipe=pipe, mud_ppg=mud_ppg, step_ft=step_ft,
    mu_list=mu_list, wob_klbf=wob_klbf, bit_torque_ftlbf=bit_torque,
    contact_radius_in=contact_radius_in, rot_reduction_factor=rot_reduction_factor
)

st.subheader("1) 2D Wellbore Profile — TVD vs Vertical Section")
fig_prof = px.line(df_geom, x="Vertical Section (ft)", y="TVD (ft)")
fig_prof.update_yaxes(autorange="reversed")
st.plotly_chart(fig_prof, use_container_width=True)

st.subheader("2) Drag Profiles — Pickup / Slack-off / Rotating Weights vs MD (multiple μ)")
tabs = st.tabs([f"μ = {mu:.2f}" for mu in mu_list])
for tab, mu in zip(tabs, mu_list):
    with tab:
        df = pd.DataFrame({"MD (ft)": md})
        for motion, label in [("pickup","Pickup"), ("slackoff","Slack-off"), ("rotating","Rotating")]:
            df[f"{label} Tension (lb)"] = results[mu][motion]["tension_lb"]
        fig = px.line(df, x="MD (ft)", y=[c for c in df.columns if c.endswith("(lb)")])
        st.plotly_chart(fig, use_container_width=True)

st.subheader("3) Torque Profile — Surface & Downhole Torque vs Depth (per μ)")
tabs2 = st.tabs([f"μ = {mu:.2f}" for mu in mu_list])
for tab, mu in zip(tabs2, mu_list):
    with tab:
        df = pd.DataFrame({
            "MD (ft)": md,
            "Torque (ft-lbf) — pickup": results[mu]["pickup"]["torque_ftlbf"],
            "Torque (ft-lbf) — slackoff": results[mu]["slackoff"]["torque_ftlbf"],
            "Torque (ft-lbf) — rotating": results[mu]["rotating"]["torque_ftlbf"],
        })
        fig = px.line(df, x="MD (ft)", y=df.columns[1:])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Surface torque is the value at MD = total depth on each curve.")

with st.expander("Bonus: 3D Path (MD–TVD–Inclination)"):
    df3 = pd.DataFrame({"MD (ft)": md, "TVD (ft)": tvd, "VS (ft)": vs, "Inclination (deg)": inc})
    fig3d = px.line_3d(df3, x="VS (ft)", y="MD (ft)", z="TVD (ft)", color="Inclination (deg)")
    fig3d.update_traces(line=dict(width=6))
    fig3d.update_scenes(zaxis_autorange="reversed")
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.subheader("Notes & References")
st.markdown(
    "- **Assignment requirements**: configurable inputs (well profile, mud weight, pipe, friction factors ≥3, hole/casing), 1-ft step; "
    "outputs include TVD–VS profile, Pickup/Slack-off/Rotating weights vs MD for multiple μ, and Torque vs depth; "
    "interactive/slider bonus & 3D display supported. :contentReference[oaicite:2]{index=2}\n"
    "- **Soft-string T&D mechanics**: force balance in an inclined/curved wellbore, normal force N from weight & curvature, "
    "friction F=μN; torque increment dT≈μ·N·R; rotating case reduces effective μ (bonus simplification). :contentReference[oaicite:3]{index=3}"
)
