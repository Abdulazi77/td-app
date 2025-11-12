# -*- coding: utf-8 -*-
# Streamlit Torque & Drag + Buckling (soft-string + buckling augmentation)
# References:
# - Project requirements (inputs/outputs/1-ft step/visuals) :contentReference[oaicite:4]{index=4}
# - Buckling theory & coupling with torque/drag (Menand et al., SPE 102850) :contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Constants & helpers
# -----------------------------
STEEL_SG = 7.85  # specific gravity of steel (≈ 65 ppg / 8.33)
E_STEEL_PSI = 30_000_000.0  # Young's modulus, psi

IN_PER_FT = 12.0
PI = math.pi

def in2_to_ft2(x_lbf_in2):   # convert lbf·in^2 -> lbf·ft^2
    return x_lbf_in2 / (IN_PER_FT**2)

def inches_to_ft(x):
    return x / IN_PER_FT

def BF_from_ppg(mw_ppg):
    sg_mud = mw_ppg / 8.33
    return max(0.0, 1.0 - sg_mud / STEEL_SG)

def I_circular_in4(OD_in, ID_in):
    return (PI/64.0) * (OD_in**4 - ID_in**4)

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class StringComp:
    name: str
    length_ft: float
    od_in: float
    id_in: float
    wt_air_lbft: float

@dataclass
class HoleSec:
    top_ft: float
    bot_ft: float
    hole_d_in: float
    casing_id_in: float  # 0 if open hole

# -----------------------------
# Trajectory builders (as per project brief) :contentReference[oaicite:8]{index=8}
# -----------------------------
def build_hold(md_total, md_kop, build_rate_deg_per_100ft, hold_inc_deg, step):
    n = int(np.floor(md_total/step))+1
    md = np.linspace(0, md_total, n)
    inc = np.zeros_like(md)
    br_ft = build_rate_deg_per_100ft/100.0
    inc_target = hold_inc_deg
    build_len = inc_target / br_ft if br_ft>0 else 0.0
    build_end = min(md_kop + build_len, md_total)
    for i, d in enumerate(md):
        if d < md_kop:
            inc[i] = 0.0
        elif d <= build_end:
            inc[i] = (d - md_kop) * br_ft
        else:
            inc[i] = inc_target
    inc_rad = np.deg2rad(inc)
    dL = np.diff(md, prepend=0)
    tvd = np.cumsum(dL * np.cos(inc_rad))
    vs  = np.cumsum(dL * np.sin(inc_rad))
    azi = np.zeros_like(md)
    return md, tvd, vs, inc, azi

def build_hold_drop(md_total, md_kop, build_rate, target_inc_deg, md_eoc, drop_rate, step):
    n = int(np.floor(md_total/step))+1
    md = np.linspace(0, md_total, n)
    inc = np.zeros_like(md)
    br_ft = build_rate/100.0
    dr_ft = drop_rate/100.0
    build_len = target_inc_deg / br_ft if br_ft>0 else 0.0
    build_end = md_kop + build_len
    drop_len = target_inc_deg / dr_ft if dr_ft>0 else 0.0
    drop_end = md_eoc + drop_len
    for i, d in enumerate(md):
        if d < md_kop:
            inc[i] = 0.0
        elif d <= build_end:
            inc[i] = (d - md_kop)*br_ft
        elif d <= md_eoc:
            inc[i] = target_inc_deg
        elif d <= drop_end:
            inc[i] = max(0.0, target_inc_deg - (d - md_eoc)*dr_ft)
        else:
            inc[i] = 0.0
    inc_rad = np.deg2rad(inc)
    dL = np.diff(md, prepend=0)
    tvd = np.cumsum(dL * np.cos(inc_rad))
    vs  = np.cumsum(dL * np.sin(inc_rad))
    azi = np.zeros_like(md)
    return md, tvd, vs, inc, azi

def horizontal_continuous_build(md_total, md_kop, build_rate, lateral_len, step):
    n = int(np.floor(md_total/step))+1
    md = np.linspace(0, md_total, n)
    inc = np.zeros_like(md)
    br_ft = build_rate/100.0
    build_len = 90.0 / br_ft if br_ft>0 else 0.0
    build_end = md_kop + build_len
    lateral_end = build_end + lateral_len
    for i, d in enumerate(md):
        if d < md_kop:
            inc[i] = 0.0
        elif d <= build_end:
            inc[i] = (d - md_kop)*br_ft
        elif d <= lateral_end:
            inc[i] = 90.0
        else:
            inc[i] = 90.0
    inc_rad = np.deg2rad(inc)
    dL = np.diff(md, prepend=0)
    tvd = np.cumsum(dL * np.cos(inc_rad))
    vs  = np.cumsum(dL * np.sin(inc_rad))
    azi = np.zeros_like(md)
    return md, tvd, vs, inc, azi

# -----------------------------
# Buckling formulas (Menand et al.) :contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}
# -----------------------------
def critical_loads_sin_hel(EI_lbf_ft2, w_b_lbft, inc_deg, r_ft, lam_hel=2.83):
    """
    F_sin = (2 EI ω) / (r * sin(Inc))
    F_hel = (λ EI ω) / (r * sin(Inc))
    Units consistent: EI [lbf·ft^2], ω [lbf/ft], r [ft] => F [lbf]
    """
    inc_rad = np.deg2rad(np.maximum(inc_deg, 1e-6))  # avoid zero division when vertical
    denom = np.maximum(r_ft * np.sin(inc_rad), 1e-9)
    F_sin = (2.0 * EI_lbf_ft2 * w_b_lbft) / denom
    F_hel = (lam_hel * EI_lbf_ft2 * w_b_lbft) / denom
    return F_sin, F_hel

def mitchell_contact_force_add(F_comp_lbf, EI_lbf_ft2, r_ft):
    """
    Mitchell (1986) helical contact force approximation (adds to normal):
    N_add ≈ (F^2 * r) / (4 * E I)  :contentReference[oaicite:11]{index=11}
    """
    return (F_comp_lbf**2) * r_ft / (4.0 * max(EI_lbf_ft2, 1e-9))

# -----------------------------
# Mechanics (soft-string core + buckling augmentation)
# -----------------------------
def compute_td_with_buckling(
    md, inc_deg, dstep_ft,
    string_table: pd.DataFrame,
    hole_table: pd.DataFrame,
    mud_ppg: float,
    mu_slide: float,
    mu_rot_factor: float,
    mode: str,          # "pickup", "slackoff", "rotating"
    wob_klbf: float,
    bit_torque_ftlbf: float,
    lam_hel: float = 2.83,
    rot_crit_factor: float = 0.78  # rotating critical load ≈ 78% of non-rotating :contentReference[oaicite:12]{index=12}
):
    """
    Returns per-depth arrays for tension (lb), torque (ft-lbf), and buckling state.
    - string_table columns: name, length_ft, od_in, id_in, wt_air_lbft
    - hole_table columns: top_ft, bot_ft, hole_d_in, casing_id_in (0 if open hole)
    """
    BF = BF_from_ppg(mud_ppg)
    n = len(md)
    inc_rad = np.deg2rad(inc_deg)
    # survey spacing
    dL = np.full(n, dstep_ft)
    dL[0] = 0.0

    # Build cumulative length from bit upward to map string components from the bit
    total_string_len = float(string_table["length_ft"].sum())
    # Map each MD depth to local position from bit (assuming bit at deepest MD)
    md0 = md[-1]
    pos_from_bit = (md0 - md).clip(min=0.0)

    # Build per-depth pipe properties (piecewise from the bit)
    od_in = np.zeros(n); id_in = np.zeros(n); wt_air = np.zeros(n); name = np.array([""]*n)
    cum = 0.0
    for _, row in string_table.iterrows():
        L = float(row["length_ft"])
        mask = (pos_from_bit >= cum) & (pos_from_bit < cum + L)
        od_in[mask] = float(row["od_in"])
        id_in[mask] = float(row["id_in"])
        wt_air[mask] = float(row["wt_air_lbft"])
        name[mask] = str(row["name"])
        cum += L
    # If above top of string (pos_from_bit > total_len), keep zeros; won't contribute.

    # Hole / casing properties per depth
    hole_d_in = np.zeros(n); casing_id_in = np.zeros(n)
    hole_table = hole_table.sort_values(["top_ft","bot_ft"])
    for _, row in hole_table.iterrows():
        mask = (md >= float(row["top_ft"])) & (md <= float(row["bot_ft"]))
        hole_d_in[mask] = float(row["hole_d_in"])
        casing_id_in[mask] = float(row["casing_id_in"])

    # Derived mechanical properties
    I_in4 = I_circular_in4(od_in, id_in)
    EI_lbf_in2 = E_STEEL_PSI * I_in4
    EI_lbf_ft2 = in2_to_ft2(EI_lbf_in2)

    w_b = wt_air * BF  # buoyed weight (lb/ft)
    # Clearance r (ft): contact taken against hole wall (open hole or casing ID if present)
    bore_d_in = np.where(casing_id_in>0, casing_id_in, hole_d_in)
    r_in = np.maximum((bore_d_in - od_in)/2.0, 1e-4)
    r_ft = inches_to_ft(r_in)

    # friction coefficient for this mode
    mu = mu_slide
    if mode == "rotating":
        mu = mu_slide * mu_rot_factor

    # Arrays
    T = np.zeros(n)         # axial tension (+tension / -compression)
    Torque = np.zeros(n)
    Fsin = np.zeros(n); Fhel = np.zeros(n); Fsin_eff = np.zeros(n); Fhel_eff = np.zeros(n)
    buckled = np.zeros(n, dtype=int)  # 0 none, 1 sinusoidal (pre-helix), 2 helical

    # marching from bit (index n-1) to surface (index 0) in our MD array order
    # We'll integrate from bottom (n-1) towards 0, then reverse to align with md ascending
    order = np.arange(n-1, -1, -1)

    for k in range(1, len(order)):
        i_bot = order[k-1]   # deeper node
        i_top = order[k]     # shallower node

        # segment properties at top node for simplicity
        seg_w = w_b[i_top] * dL[i_top]   # lbs
        # normal force (soft-string baseline): N0 ≈ W*sin(inc)
        N0 = seg_w * np.sin(inc_rad[i_top])

        # compute buckling thresholds at top node
        Fsin[i_top], Fhel[i_top] = critical_loads_sin_hel(
            EI_lbf_ft2[i_top], w_b[i_top], inc_deg[i_top], r_ft[i_top], lam_hel=lam_hel
        )
        # Rotation lowers critical loads (Menand referencing Mitchell 2007) :contentReference[oaicite:13]{index=13}
        if mode == "rotating":
            Fsin_eff[i_top] = Fsin[i_top] * rot_crit_factor
            Fhel_eff[i_top] = Fhel[i_top] * rot_crit_factor
        else:
            Fsin_eff[i_top] = Fsin[i_top]
            Fhel_eff[i_top] = Fhel[i_top]

        # Decide friction direction and compute tension transfer
        # drag force magnitude on segment
        Ff = mu * N0

        if mode == "pickup":
            # Upward motion: friction adds to weight
            T[i_top] = T[i_bot] + seg_w + Ff
        elif mode == "slackoff":
            # Downward motion: friction subtracts from weight
            T[i_top] = T[i_bot] + seg_w - Ff
        else:  # rotating "axial hold" approx (drag reduced via mu_rot_factor)
            T[i_top] = T[i_bot] + seg_w + Ff

        # If axial force is compressive at top node (T<0), check buckling
        if T[i_top] < 0 and EI_lbf_ft2[i_top] > 0.0 and r_ft[i_top] > 0:
            Fcomp = abs(T[i_top])
            if Fcomp >= Fhel_eff[i_top]:
                buckled[i_top] = 2  # helical
            elif Fcomp >= Fsin_eff[i_top]:
                buckled[i_top] = 1  # sinusoidal

            if buckled[i_top] == 2:
                # helical: add Mitchell contact force to normal (augments drag & torque) :contentReference[oaicite:14]{index=14}
                N_add = mitchell_contact_force_add(Fcomp, EI_lbf_ft2[i_top], r_ft[i_top])
                # recompute with augmented normal force (one-step correction)
                Ff_aug = mu * (N0 + N_add)
                if mode == "pickup":
                    T[i_top] = T[i_bot] + seg_w + Ff_aug
                elif mode == "slackoff":
                    T[i_top] = T[i_bot] + seg_w - Ff_aug
                else:
                    T[i_top] = T[i_bot] + seg_w + Ff_aug

                # torque increment with augmented N
                dTorque = mu * (N0 + N_add) * r_ft[i_top]
            else:
                # sinusoidal or none: baseline torque
                dTorque = mu * N0 * r_ft[i_top]
        else:
            # tension or no buckling: baseline torque
            dTorque = mu * N0 * r_ft[i_top]

        Torque[i_top] = Torque[i_bot] + dTorque

    # Add bit torque uniformly (baseline shift)
    if bit_torque_ftlbf:
        Torque += bit_torque_ftlbf

    # reorder arrays to ascending md
    # (they are already aligned with md ascending because we wrote into i_top indices)
    return T, Torque, Fsin_eff, Fhel_eff, buckled

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Advanced Torque & Drag + Buckling", layout="wide")
st.title("Advanced Torque & Drag + Buckling (Soft-String with Buckling Augmentation)")

st.caption(
    "Implements soft-string T&D with 1-ft step, multi-component string, hole/casing intervals, "
    "buckling thresholds (sinusoidal & helical) and contact-force augmentation when buckled. "
    "Plots: TVD-VS, drag (pickup/slack-off/rotating) vs MD, torque vs MD, and a buckling envelope. "
    "Based on your project specification:contentReference[oaicite:15]{index=15} and Menand et al. (SPE 102850):contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}."
)

with st.sidebar:
    st.header("Well Path (choose one)")

    profile = st.selectbox("Profile", ["Build & Hold", "Build & Hold & Drop", "Horizontal + Lateral"])
    md_total = st.number_input("Total MD (ft)", value=12000, min_value=1000, step=500)
    step_ft  = st.number_input("Step (ft) — use 1 as required", value=1, min_value=1, max_value=50)

    md_kop     = st.number_input("KOP (ft)", value=2000, min_value=0, max_value=30000, step=100)
    build_rate = st.number_input("Build rate (deg/100 ft)", value=8.0, min_value=0.0, max_value=30.0, step=0.5)
    target_inc = st.number_input("Target inclination (deg)", value=60.0, min_value=0.0, max_value=90.0, step=1.0)

    md_eoc = 0; drop_rate = 0.0; lateral_len = 0
    if profile == "Build & Hold & Drop":
        md_eoc    = st.number_input("End of Hold (start Drop) MD (ft)", value=8000, min_value=0, max_value=50000, step=100)
        drop_rate = st.number_input("Drop rate (deg/100 ft)", value=8.0, min_value=0.0, max_value=30.0, step=0.5)
    elif profile == "Horizontal + Lateral":
        lateral_len = st.number_input("Lateral length (ft)", value=2000, min_value=0, max_value=20000, step=100)

    st.markdown("---")
    st.subheader("Drilling Parameters")
    mud_ppg = st.number_input("Mud weight (ppg)", value=10.0, min_value=5.0, max_value=20.0, step=0.1)

    st.markdown("**Friction**")
    mu_csv = st.text_input("Sliding friction μ (comma-sep; ≥3 values)", "0.20,0.30,0.40")
    mu_list = [max(0.0, min(1.0, float(x.strip()))) for x in mu_csv.split(",") if x.strip()]
    mu_rot_factor = st.slider("Rotating μ reduction factor", min_value=0.1, max_value=1.0, value=0.30, step=0.05)

    wob_klbf   = st.number_input("WOB (klbf) [optional]", value=0.0, min_value=0.0, max_value=500.0, step=1.0)
    bit_torque = st.number_input("Bit torque (ft-lbf) [optional]", value=0.0, min_value=0.0, max_value=100000.0, step=100.0)

    st.markdown("---")
    st.subheader("Buckling Parameters")
    lam_hel = st.slider("λ for helical critical load (2.83–5.65)", min_value=2.83, max_value=5.65, value=2.83, step=0.01)
    rot_crit_factor = st.slider("Rotating critical-load factor (≈0.78)", min_value=0.5, max_value=1.0, value=0.78, step=0.01)

    st.markdown("---")
    st.subheader("Drillstring Assembly (from bit up)")
    st.caption("Edit lengths/OD/ID/weight. Add rows for collars/HWDP/DP.")
    default_string = pd.DataFrame([
        {"name":"Bit sub", "length_ft":30, "od_in":6.5, "id_in":2.25, "wt_air_lbft":30.0},
        {"name":"DC",      "length_ft":600, "od_in":6.5, "id_in":2.75, "wt_air_lbft":50.0},
        {"name":"HWDP",    "length_ft":1000, "od_in":5.0, "id_in":3.0, "wt_air_lbft":26.0},
        {"name":"DP",      "length_ft":8000, "od_in":5.0, "id_in":4.0, "wt_air_lbft":19.5},
    ])
    string_table = st.data_editor(default_string, num_rows="dynamic", use_container_width=True)

    st.subheader("Hole / Casing Intervals")
    st.caption("Define intervals with hole diameter and casing ID (0 = open hole).")
    default_hole = pd.DataFrame([
        {"top_ft":0, "bot_ft":md_total, "hole_d_in":8.5, "casing_id_in":0.0},
    ])
    hole_table = st.data_editor(default_hole, num_rows="dynamic", use_container_width=True)


# Build trajectory
if profile == "Build & Hold":
    md, tvd, vs, inc, azi = build_hold(md_total, md_kop, build_rate, target_inc, step_ft)
elif profile == "Build & Hold & Drop":
    md, tvd, vs, inc, azi = build_hold_drop(md_total, md_kop, build_rate, target_inc, md_eoc, drop_rate, step_ft)
else:
    md, tvd, vs, inc, azi = horizontal_continuous_build(md_total, md_kop, build_rate, lateral_len, step_ft)

# Geometry table
df_geom = pd.DataFrame({"MD (ft)": md, "TVD (ft)": tvd, "Vertical Section (ft)": vs, "Inclination (deg)": inc})

# Compute for each μ and each motion
motions = [("pickup","Pickup"), ("slackoff","Slack-off"), ("rotating","Rotating")]
results: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}

for mu in mu_list:
    cases = {}
    for mcode, mlabel in motions:
        T, Trq, Fsin, Fhel, buck = compute_td_with_buckling(
            md=md, inc_deg=inc, dstep_ft=step_ft,
            string_table=string_table, hole_table=hole_table,
            mud_ppg=mud_ppg, mu_slide=mu, mu_rot_factor=mu_rot_factor,
            mode=mcode, wob_klbf=wob_klbf, bit_torque_ftlbf=bit_torque,
            lam_hel=lam_hel, rot_crit_factor=rot_crit_factor
        )
        cases[mcode] = {
            "tension_lb": T,
            "torque_ftlbf": Trq,
            "Fsin_lbf": Fsin,
            "Fhel_lbf": Fhel,
            "buckling_state": buck
        }
    results[mu] = cases

# -----------------------------
# Plots / Outputs (per handout) :contentReference[oaicite:21]{index=21}
# -----------------------------
st.subheader("1) 2D Wellbore Profile — TVD vs Vertical Section")
fig_prof = px.line(df_geom, x="Vertical Section (ft)", y="TVD (ft)")
fig_prof.update_yaxes(autorange="reversed")
st.plotly_chart(fig_prof, use_container_width=True)

st.subheader("2) Drag Profiles — Pickup / Slack-off / Rotating (multiple μ)")
tabs = st.tabs([f"μ = {mu:.2f}" for mu in mu_list])
for tab, mu in zip(tabs, mu_list):
    with tab:
        df = pd.DataFrame({"MD (ft)": md})
        for mcode, mlabel in motions:
            df[f"{mlabel} tension (lb)"] = results[mu][mcode]["tension_lb"]
        fig = px.line(df, x="MD (ft)", y=[c for c in df.columns if c.endswith("(lb)")])
        st.plotly_chart(fig, use_container_width=True)

st.subheader("3) Torque Profile — Surface & Downhole Torque vs Depth (per μ)")
tabs2 = st.tabs([f"μ = {mu:.2f}" for mu in mu_list])
for tab, mu in zip(tabs2, mu_list):
    with tab:
        df = pd.DataFrame({
            "MD (ft)": md,
            "Torque — pickup (ft-lbf)": results[mu]["pickup"]["torque_ftlbf"],
            "Torque — slack-off (ft-lbf)": results[mu]["slackoff"]["torque_ftlbf"],
            "Torque — rotating (ft-lbf)": results[mu]["rotating"]["torque_ftlbf"],
        })
        fig = px.line(df, x="MD (ft)", y=df.columns[1:])
        st.plotly_chart(fig, use_container_width=True)

st.subheader("4) Buckling Envelope — Critical Loads and Buckled Zones")
st.caption("Shows effective sinusoidal/helical critical loads (with rotation factor if applicable) and where the string is predicted to buckle.")
tabs3 = st.tabs([f"μ = {mu:.2f}" for mu in mu_list])
for tab, mu in zip(tabs3, mu_list):
    with tab:
        # Pick one mode to display envelope (rotating most interesting)
        Fsin = results[mu]["rotating"]["Fsin_lbf"]
        Fhel = results[mu]["rotating"]["Fhel_lbf"]
        T_ax  = results[mu]["rotating"]["tension_lb"]

        comp = -np.minimum(T_ax, 0.0)  # compressive magnitude
        df = pd.DataFrame({
            "MD (ft)": md,
            "Compression (lb)": comp,
            "F_sin, eff (lb)": Fsin,
            "F_hel, eff (lb)": Fhel,
        })
        fig = px.line(df, x="MD (ft)", y=["Compression (lb)", "F_sin, eff (lb)", "F_hel, eff (lb)"])
        st.plotly_chart(fig, use_container_width=True)

        # highlight buckled intervals
        buck = results[mu]["rotating"]["buckling_state"]
        buck_str = np.where(buck==2, "Helical", np.where(buck==1,"Sinusoidal","None"))
        st.dataframe(pd.DataFrame({"MD (ft)": md, "Buckling state (rotating)": buck_str}))

with st.expander("Bonus: 3D well path (VS–MD–TVD)"):
    df3 = pd.DataFrame({"MD (ft)": md, "TVD (ft)": tvd, "VS (ft)": vs, "Inclination (deg)": inc})
    fig3d = px.line_3d(df3, x="VS (ft)", y="MD (ft)", z="TVD (ft)", color="Inclination (deg)")
    fig3d.update_scenes(zaxis_autorange="reversed")
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.subheader("Notes")
st.markdown(
    "- **Project deliverables satisfied**: 1-ft step; inputs for profile, mud weight, pipe size, μ (≥3), hole/casing; "
    "outputs include TVD–VS, pickup/slack-off/rotating weights vs MD, torque vs depth; interactive/3D views; "
    "plus buckling threshold estimation (envelope):contentReference[oaicite:22]{index=22}.\n"
    "- **Buckling theory**: sinusoidal & helical thresholds (inclination & clearance dependent), rotation lowering critical loads, "
    "and Mitchell’s helical contact augmentation to normal force — all per Menand et al. with classic references:contentReference[oaicite:23]{index=23}:contentReference[oaicite:24]{index=24}:contentReference[oaicite:25]{index=25}.\n"
    "- **Coupling**: Because buckling ↑ contact force → ↑ drag/torque → ↑ compression, we apply a one-step augmentation within each segment; "
    "full simultaneous coupling as in ABIS is iterative and heavy, but this captures first-order field effects:contentReference[oaicite:26]{index=26}."
)
