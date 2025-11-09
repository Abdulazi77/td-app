# Part 1: Imports + UI Layout for “Soft-String Torque & Drag” App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import the engine from the torque_drag package
import torque_drag as td
# If you also use a trajectory/ well path library:
# import welleng as we

# ===== Sidebar Inputs =====
st.set_page_config(page_title="Torque & Drag — Soft-String Model", layout="wide")
st.title("Torque & Drag — Soft-String Model (1-ft steps)")

with st.sidebar:
    st.header("Team & Report Info")
    team_names = st.text_input("Team names (comma-separated)", "")
    methodology = st.text_area("Methodology", 
                               "Soft-string model (Johancsik) using 1-ft discretization bottom→top.")
    assumptions = st.text_area("Assumptions", 
                               "Uniform friction factor μ along each interval; pipe fully fluid–filled; no buckling considered.")
    insights_user = st.text_area("Your insights (to include in report)", 
                                 "E.g., drag increases with inclination & mud weight; torque sensitivity to μ.")

    st.header("1) Survey / Trajectory")
    src = st.radio("Trajectory source", ["Upload CSV", "Synthetic"], horizontal=True)
    if src == "Upload CSV":
        st.caption("CSV must include columns: md_ft, inc_deg, azi_deg")
        f = st.file_uploader("Upload survey CSV", type=["csv"])
        if f:
            survey_df = pd.read_csv(f)
        else:
            survey_df = None
    else:
        synth_type = st.selectbox("Synthetic trajectory type", ["Build & Hold", "S-Curve", "Horizontal + Lateral"])
        # Additional synthetic-profile inputs (will fill later)
        survey_df = None  # placeholder

    st.header("2) String / Fluid / Geometry")
    # Pipe OD drop-list
    pipe_od = st.selectbox("Pipe OD (inches)", [2.375, 2.875, 3.5, 4.0, 4.5, 5.0, 6.625], index=4)
    pipe_id = st.number_input("Pipe ID (inches)", 1.0, float(pipe_od)-0.1, float(pipe_od)-0.5, step=0.01, format="%.3f")
    # Casing/Hole geometry
    casing_od = st.selectbox("Casing OD / Hole OD (inches)", [7.0, 8.5, 9.625, 12.25, 16.0], index=1)
    casing_id = st.number_input("Casing ID / Hole ID (inches)", 3.0, float(casing_od)-0.1, float(casing_od)-0.5, step=0.01, format="%.3f")

    mud_ppg = st.number_input("Mud weight (ppg)", 6.0, 20.0, 10.0, step=0.1)
    steel_sg = st.number_input("Steel SG", 7.6, 8.2, 7.85, step=0.01)

    st.header("3) Friction / Loads")
    mu_base = st.number_input("Base friction factor μ", 0.05, 0.80, 0.28, step=0.01)
    mu_overlay_on = st.checkbox("Overlay multiple μ values", value=True)
    mu_overlay_str = st.text_input("List of μ values (comma-separated)", "0.20,0.30,0.40", disabled=not mu_overlay_on)
    mode = st.selectbox("Case mode", ["PU", "SL", "ROB"])
    wob_klbf = st.number_input("WOB at bit (klbf)", 0.0, 300.0, 50.0, step=1.0)
    tbit_kftlbf = st.number_input("Bit torque (kft-lbf)", 0.0, 100.0, 0.0, step=0.5)

    st.header("4) Options & Run")
    show_trace = st.checkbox("Show iteration trace (selected rows)", value=True)
    show_deriv = st.checkbox("Include numeric derivations in report", value=True)
    do_calibrate = st.checkbox("Calibrate μ to match surface hookload?", value=False)
    target_hook_lbf = st.number_input("Target surface hookload (lbf)", 0, 1_000_000, 0, step=1000, disabled=not do_calibrate)
    mu_lo = st.number_input("μ lower bound (calibration)", 0.01, 0.80, 0.10, step=0.01, disabled=not do_calibrate)
    mu_hi = st.number_input("μ upper bound (calibration)", 0.02, 0.90, 0.50, step=0.01, disabled=not do_calibrate)
    run_btn = st.button("Run T&D Model")
