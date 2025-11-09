
---

# 3) `td_app.py`  (full app)

> Paste this whole file as `td_app.py`. If Streamlit Cloud still spins, set **Main file path** to `td_app.py` in **Manage app → Settings** or re-deploy the app with this file as the entrypoint. :contentReference[oaicite:4]{index=4}

```python
# Torque & Drag — Soft-String App (Johancsik + torque_drag)
# Sources: torque_drag.calc API (SPE-11380-PA model) and well_profile trajectory loader.
# torque_drag docs: https://torque-drag.readthedocs.io/en/latest/calculations.html
# well_profile load CSV/DataFrame: https://well-profile.readthedocs.io/en/latest/load_profile.html

import streamlit as st
st.set_page_config(page_title="Torque & Drag — Soft-String", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# External engines
import torque_drag as td   # pip: torque-drag
import well_profile as wp  # pip: well-profile

# ------------------------ constants & helpers ------------------------

LBF_PER_kN = 224.8089431
KFTLBF_PER_kN_M = 737.5621493
kN_PER_KLBF = 4.4482216153
kN_M_PER_KFTLBF = 1.3558179483
STEEL_DENS_LB_PER_IN3 = 0.283  # ~7.85 g/cc

def buoyancy_factor_from_mw_ppg(mw_ppg: float) -> float:
    # BF = (65.5 - MW)/65.5  (English ppg)
    return (65.5 - mw_ppg) / 65.5

def metal_area_in2(od_in, id_in) -> float:
    return np.pi/4.0 * (od_in**2 - id_in**2)

def w_air_lbf_per_ft(od_in, id_in) -> float:
    # weight/ft = volume_per_ft * density
    # volume_per_ft [in^3/ft] = area_in2 * 12
    vol_in3_per_ft = metal_area_in2(od_in, id_in) * 12.0
    return vol_in3_per_ft * STEEL_DENS_LB_PER_IN3

def resample_survey(md_ft, inc_deg, azi_deg, step_ft=1.0):
    """Resample survey to ~1-ft increments for smoother soft-string integration."""
    md_min, md_max = float(md_ft.min()), float(md_ft.max())
    grid = np.arange(md_min, md_max + step_ft, step_ft)
    inc_i = np.interp(grid, md_ft, inc_deg)
    azi_i = np.interp(grid, md_ft, azi_deg)
    return grid, inc_i, azi_i

def dogleg_and_kappa(md, inc_deg, azi_deg):
    """Compute dogleg angle (rad) and curvature kappa (rad/ft) between points."""
    inc = np.radians(inc_deg)
    azi = np.radians(azi_deg)
    dmd = np.diff(md)
    # Minimum curvature dogleg angle
    cos_dog = np.sin(inc[:-1])*np.sin(inc[1:])*np.cos(azi[1:]-azi[:-1]) + np.cos(inc[:-1])*np.cos(inc[1:])
    cos_dog = np.clip(cos_dog, -1.0, 1.0)
    dog = np.arccos(cos_dog)  # radians
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.zeros_like(dog)
        kappa[dmd_nonzero := dmd > 0] = dog[dmd_nonzero] / dmd[dmd_nonzero]
    return dog, kappa, dmd

def build_mu_profile(md_grid, mu_cased, mu_open, shoe_md_ft):
    mu = np.where(md_grid <= shoe_md_ft, mu_cased, mu_open)
    return mu

def eff_radius_ft(casing_id_in, pipe_od_in):
    # approximate wall contact radius for annular friction torque lever arm
    clr_in = max(casing_id_in - pipe_od_in, 0.0)
    return 0.5 * clr_in / 12.0

def map_case(mode_str):
    # UI -> torque_drag
    return {"PU": "hoisting", "SL": "lowering", "ROB": "static"}[mode_str]

# ------------------------ internal soft-string step solver ------------------------

def soft_string_iterate(md_grid, inc_deg, azi_deg, pipe_od_in, pipe_id_in,
                        casing_id_in, bf, mu_prof, mode, wob_klbf, tbit_kftlbf):
    """
    Johancsik soft-string per-step integration bottom->top:
      N = Wb*sin(theta) + T*kappa
      dF = (sgn*Wb*cos(theta) + mu*N)*ds,   sgn=+1(PU), -1(SL), 0(ROB for friction term)
      dT = mu*N*r_eff*ds
    Returns DataFrame 'trace' with per-row values and summary dict.
    """

    # prepare arrays
    dog, kappa, dmd = dogleg_and_kappa(md_grid, inc_deg, azi_deg)
    theta = np.radians(inc_deg)
    # weight per ft (air) & buoyed
    w_air = w_air_lbf_per_ft(pipe_od_in, pipe_id_in)
    w_b = w_air * bf
    r_eff_ft = eff_radius_ft(casing_id_in, pipe_od_in)

    # boundary at bit (last node)
    wob_lbf = wob_klbf * 1000.0
    tbit_lbf_ft = tbit_kftlbf * 1000.0
    if mode == "ROB":  # rotating on bottom
        T_prev = -wob_lbf  # axial sign convention: compressive at bit
        M_prev = tbit_lbf_ft
        friction_term_on = False
        sgn = +1.0  # only weight term (axial) sign doesn't matter for ROB frictionless axial; keep +1 for clarity
    elif mode == "PU":
        T_prev = 0.0
        M_prev = 0.0
        friction_term_on = True
        sgn = +1.0
    elif mode == "SL":
        T_prev = 0.0
        M_prev = 0.0
        friction_term_on = True
        sgn = -1.0
    else:
        raise ValueError("mode must be 'PU','SL','ROB'")

    rows = []
    # iterate bottom (end) -> top (start)
    for i in range(len(md_grid)-1, 0, -1):
        ds = md_grid[i] - md_grid[i-1]
        th = float((theta[i] + theta[i-1]) / 2.0)          # avg inclination
        kap = float(kappa[i-1])
        mu = float((mu_prof[i] + mu_prof[i-1]) / 2.0)

        # side/normal force per unit length
        N_per_ft = w_b*np.sin(th) + T_prev*kap
        # axial increment per unit length
        dF_per_ft = sgn*w_b*np.cos(th) + (mu*N_per_ft if friction_term_on else 0.0)
        dF = dF_per_ft * ds
        F_next = T_prev + dF

        # torque increment
        dT_per_ft = (mu*N_per_ft*r_eff_ft if friction_term_on else 0.0)
        dT = dT_per_ft * ds
        T_next = M_prev + dT

        rows.append({
            "md_to_ft": md_grid[i-1],
            "md_from_ft": md_grid[i],
            "ds_ft": ds,
            "theta_deg": np.degrees(th),
            "kappa_rad_per_ft": kap,
            "mu": mu,
            "Wb_lbf_per_ft": w_b,
            "N_lbf_per_ft": N_per_ft,
            "dF_lbf": dF,
            "F_next_lbf": F_next,  # axial at this node going up
            "dT_lbf_ft": dT,
            "T_next_lbf_ft": T_next
        })

        # advance
        T_prev = F_next
        M_prev = T_next

    trace = pd.DataFrame(rows[::-1])  # reorder top->bottom
    # surface values are last updated (at top)
    hookload_surf_lbf = T_prev
    torque_surf_lbf_ft = M_prev

    summary = {
        "hookload_lbf": hookload_surf_lbf,
        "torque_lbf_ft": torque_surf_lbf_ft,
        "w_air_lbf_per_ft": w_air,
        "w_buoy_lbf_per_ft": w_b,
        "bf": bf,
        "r_eff_ft": r_eff_ft
    }
    return trace, summary

# ------------------------ UI ------------------------

st.title("Torque & Drag — Soft-String (Johancsik)")

with st.sidebar:
    st.header("Report Header")
    team_names = st.text_input("Team names (comma-separated)", "")
    methodology = st.text_area("Methodology", "Soft-string model (Johancsik SPE-11380) + torque_drag cross-check.")
    assumptions = st.text_area("Assumptions", "Pipe fully fluid-filled; constant μ per interval; no buckling.")
    insights_user = st.text_area("Your insights", "Drag ↑ with inclination and μ; torque lever arm set by annular clearance.")

    st.header("1) Trajectory")
    src = st.radio("Source", ["Upload CSV", "Synthetic"], horizontal=True)
    survey_df = None
    if src == "Upload CSV":
        st.caption("CSV columns (headers flexible): md_ft, inc_deg, azi_deg")
        file = st.file_uploader("Upload survey CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            # normalize headers
            rename_map = {c: c.strip().lower() for c in df.columns}
            df.columns = rename_map.values()
            # common aliases
            alias = {}
            if "md_ft" not in df.columns and "md" in df.columns: alias["md"] = "md_ft"
            if "inc_deg" not in df.columns and "inc" in df.columns: alias["inc"] = "inc_deg"
            if "azi_deg" not in df.columns and "azi" in df.columns: alias["azi"] = "azi_deg"
            df = df.rename(columns=alias)
            survey_df = df[["md_ft", "inc_deg", "azi_deg"]].copy()
    else:
        st.caption("Quick synthetic profile (well_profile.get)")
        prof = st.selectbox("Type", ["J", "S", "H1 (Horizontal single curve)"], index=0)
        mdt = st.number_input("Target MD (ft)", 1000.0, 30000.0, 10000.0, step=500.0)
        build_angle = st.number_input("Build angle (°)", 1.0, 90.0, 30.0, step=1.0)
        kop = st.number_input("KOP (ft)", 0.0, 20000.0, 2000.0, step=100.0)
        eob = st.number_input("EOB (ft)", 0.0, 30000.0, 4000.0, step=100.0)
        sod = st.number_input("SOD (ft)", 0.0, 30000.0, 0.0, step=100.0, help="For S-type/H2 only")
        eod = st.number_input("EOD (ft)", 0.0, 30000.0, 0.0, step=100.0, help="For S-type/H2 only")
        points = st.slider("Points (resolution)", 100, 5000, 1200, 100)
        # Create via well_profile.get, then extract md/inc/azi from returned object
        wtmp = wp.get(mdt, profile=('H1' if prof.startswith('H1') else prof),
                      build_angle=build_angle, kop=kop, eob=eob, sod=sod, eod=eod,
                      points=points, set_start={'north':0, 'east':0})
        # The well object has a dataframe property
        wdf = pd.DataFrame({
            "md_ft": wtmp.trajectory['md'], 
            "inc_deg": np.degrees(wtmp.trajectory['inc']),
            "azi_deg": np.degrees(wtmp.trajectory['azi'])
        })
        survey_df = wdf

    st.header("2) Pipe / Casing / Fluid")
    pipe_od = st.selectbox("Pipe OD (in)", [2.375, 2.875, 3.5, 4.0, 4.5, 5.0, 5.5, 6.625], index=4)
    pipe_id = st.number_input("Pipe ID (in)", 1.0, float(pipe_od)-0.05, float(pipe_od)-0.5, step=0.01, format="%.3f")
    casing_od = st.selectbox("Casing/Hole OD (in)", [7.0, 8.5, 9.625, 12.25, 16.0], index=1)
    casing_id = st.number_input("Casing/Hole ID (in)", 3.0, float(casing_od)-0.05, float(casing_od)-0.5, step=0.01, format="%.3f")
    shoe_md_ft = st.number_input("Casing shoe depth (ft)", 0.0, 40000.0, 4000.0, step=100.0)

    mud_ppg = st.number_input("Mud weight (ppg)", 6.0, 20.0, 10.0, step=0.1)
    steel_sg = st.number_input("Steel SG", 7.6, 8.2, 7.85, step=0.01)

    st.header("3) Friction & Loads")
    mu_cased = st.number_input("μ in cased hole", 0.05, 0.80, 0.25, step=0.01)
    mu_open = st.number_input("μ in open hole", 0.05, 0.80, 0.30, step=0.01)
    overlay_on = st.checkbox("Overlay extra μ curves (comma list)", value=True)
    mu_overlay_str = st.text_input("Extra μ values", "0.20,0.30,0.40", disabled=not overlay_on)
    mode = st.selectbox("Case", ["PU", "SL", "ROB"])  # hoisting / lowering / static
    wob_klbf = st.number_input("WOB at bit (klbf)", 0.0, 300.0, 40.0, step=1.0)
    tbit_kftlbf = st.number_input("Bit torque (kft-lbf)", 0.0, 100.0, 0.0, step=0.5)

    st.header("4) Output Options")
    show_trace = st.checkbox("Show iteration trace (sample rows)", value=True)
    show_deriv = st.checkbox("Include numeric derivations in report", value=True)
    excel_export = st.checkbox("Export Excel workbook", value=True)
    run_btn = st.button("Run model")

# ------------------------ main run ------------------------

if run_btn:
    if survey_df is None or survey_df.empty:
        st.error("No survey data. Upload a CSV or generate a synthetic path.")
        st.stop()

    # Load trajectory into well_profile (units: English/feet)
    well = wp.load(survey_df.rename(columns={"md_ft":"md","inc_deg":"inclination","azi_deg":"azimuth"}),
                   set_info={'units':'english'}, set_start={'north':0,'east':0})
    # For plotting 3D, generate TVD/NE with our resample
    md_grid, inc_i, azi_i = resample_survey(survey_df["md_ft"].values,
                                            survey_df["inc_deg"].values,
                                            survey_df["azi_deg"].values, step_ft=1.0)
    # Friction profile by interval
    bf = buoyancy_factor_from_mw_ppg(mud_ppg)
    mu_prof = build_mu_profile(md_grid, mu_cased, mu_open, shoe_md_ft)

    # ----- internal soft-string (equations & numbers) -----
    trace, summary = soft_string_iterate(md_grid, inc_i, azi_i,
                                         pipe_od, pipe_id, casing_id,
                                         bf, mu_prof, mode, wob_klbf, tbit_kftlbf)

    # ----- torque_drag engine (cross-check & overlays) -----
    dims = {'pipe': {'od': float(pipe_od), 'id': float(pipe_id), 'shoe': float(shoe_md_ft)},
            'odAnn': float(casing_od)}
    dens = {'rhof': float(mud_ppg/8.33), 'rhod': float(steel_sg)}  # SG
    td_case = map_case(mode)

    def run_td(mu_val):
        res = td.calc(well.trajectory, dims, densities=dens,
                      case=td_case, fric=float(mu_val),
                      wob=wob_klbf * kN_PER_KLBF,
                      tbit=tbit_kftlbf * kN_M_PER_KFTLBF,
                      torque_calc=True)
        force_kN = res.force[td_case]
        torque_kN_m = res.torque[td_case]
        # build a DataFrame vs md (align lengths with md_grid if possible)
        # use simple index for md; engines typically output per-point arrays.
        md_for_td = np.linspace(md_grid.min(), md_grid.max(), num=len(force_kN))
        return pd.DataFrame({
            "md_ft": md_for_td,
            "F_lbf": np.array(force_kN) * LBF_PER_kN,
            "T_lbf_ft": np.array(torque_kN_m) * KFTLBF_PER_kN_M
        })

    overlay_mus = []
    if overlay_on and mu_overlay_str.strip():
        overlay_mus = [float(x) for x in mu_overlay_str.split(",") if x.strip()]

    td_curves = {"base": run_td(mu_open)}  # base uses open-hole μ nominal (visual)
    for mu_extra in overlay_mus:
        td_curves[f"μ={mu_extra:.2f}"] = run_td(mu_extra)

    # -------------------- PLOTS --------------------

    col1, col2 = st.columns(2)

    with col1:
        figF = go.Figure()
        figF.add_trace(go.Scatter(x=trace["md_to_ft"], y=trace["F_next_lbf"],
                                  name="Internal model (axial)", mode="lines"))
        for name, dfc in td_curves.items():
            figF.add_trace(go.Scatter(x=dfc["md_ft"], y=dfc["F_lbf"],
                                      name=f"torque_drag {name}", mode="lines"))
        figF.update_layout(title="Hookload / Axial vs MD",
                           xaxis_title="MD (ft)", yaxis_title="Force (lbf)")
        st.plotly_chart(figF, use_container_width=True)

    with col2:
        figT = go.Figure()
        figT.add_trace(go.Scatter(x=trace["md_to_ft"], y=trace["T_next_lbf_ft"],
                                  name="Internal model (torque)", mode="lines"))
        for name, dfc in td_curves.items():
            figT.add_trace(go.Scatter(x=dfc["md_ft"], y=dfc["T_lbf_ft"],
                                      name=f"torque_drag {name}", mode="lines"))
        figT.update_layout(title="Surface Torque vs MD",
                           xaxis_title="MD (ft)", yaxis_title="Torque (lbf·ft)")
        st.plotly_chart(figT, use_container_width=True)

    # 3D well path (simple minimum-curvature forward; uses resampled grid)
    # Compute incremental NE/TVD
    inc_rad = np.radians(inc_i)
    azi_rad = np.radians(azi_i)
    dmd = np.diff(md_grid, prepend=md_grid[0])
    tvd = np.cumsum(dmd * np.cos(inc_rad))
    north = np.cumsum(dmd * np.sin(inc_rad) * np.cos(azi_rad))
    east = np.cumsum(dmd * np.sin(inc_rad) * np.sin(azi_rad))

    fig3d = go.Figure(data=[go.Scatter3d(
        x=east, y=north, z=-tvd, mode="lines", name="Well path")])
    fig3d.update_layout(title="3D Well Path", scene=dict(
        xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)"))
    st.plotly_chart(fig3d, use_container_width=True)

    # -------------------- TABLES --------------------
    st.subheader("Iteration trace (selected rows)")
    if show_trace:
        st.dataframe(trace.iloc[::max(1, len(trace)//30)].round(5), use_container_width=True)

    # -------------------- REPORT --------------------
    report = []
    report.append("# Torque & Drag — Soft-String Report\n")
    report.append(f"**Team**: {team_names}\n")
    report.append("## Methodology & Sources\n")
    report.append("- Soft-string model per Johancsik SPE-11380; friction force = μ×(Wb·sinθ + T·κ).\n")
    report.append("- Well path loaded with `well_profile.load` (English units), T&D cross-check with `torque_drag.calc`.\n")
    report.append("## Inputs (key)\n")
    report.append(f"- Pipe: OD {pipe_od:.3f} in, ID {pipe_id:.3f} in; Casing/Hole ID {casing_id:.3f} in (OD {casing_od:.3f} in), Shoe {shoe_md_ft:.1f} ft\n")
    report.append(f"- Mud {mud_ppg:.2f} ppg → BF {bf:.4f}; μ(cased/open) = {mu_cased:.2f}/{mu_open:.2f}\n")
    report.append(f"- Case {mode}; WOB {wob_klbf:.1f} klbf; Bit torque {tbit_kftlbf:.2f} kft-lbf\n")
    report.append("## Core Equations\n")
    report.append("Side force per ft: N = Wb·sinθ + T·κ\n")
    report.append("Axial: F(i+1) = F(i) + [ sgn·Wb·cosθ + μ·N ]·Δs (sgn=+1 PU, −1 SL; ROB→friction term canceled)\n")
    report.append("Torque: M(i+1) = M(i) + μ·N·r_eff·Δs, r_eff = (ID_casing − OD_pipe)/2\n")
    report.append("## Results (internal model)\n")
    report.append(f"- Surface hookload ≈ {summary['hookload_lbf']:.0f} lbf; Surface torque ≈ {summary['torque_lbf_ft']:.0f} lbf·ft\n")
    if show_deriv:
        report.append("## Sample Derivations (first few steps)\n")
        for _, r in trace.head(6).iterrows():
            report.append(
                f"MD {r['md_to_ft']:.1f}→{r['md_from_ft']:.1f}: θ={r['theta_deg']:.2f}°, κ={r['kappa_rad_per_ft']:.5e} rad/ft, "
                f"N={r['N_lbf_per_ft']:.1f} lbf/ft, dF={r['dF_lbf']:.1f} lbf, F_next={r['F_next_lbf']:.1f} lbf; "
                f"dT={r['dT_lbf_ft']:.1f} lbf·ft, T_next={r['T_next_lbf_ft']:.1f} lbf·ft")
    report_txt = "\n".join(report)
    st.download_button("Download report (txt)", data=report_txt, file_name="TD_Report.txt", mime="text/plain")

    # -------------------- EXCEL EXPORT --------------------
    if excel_export:
        with pd.ExcelWriter("TD_Output.xlsx", engine="xlsxwriter") as xw:
            survey_df.to_excel(xw, index=False, sheet_name="Inputs_Survey")
            pd.DataFrame({
                "PipeOD_in":[pipe_od], "PipeID_in":[pipe_id],
                "CsgOD_in":[casing_od], "CsgID_in":[casing_id],
                "Shoe_ft":[shoe_md_ft], "Mud_ppg":[mud_ppg],
                "mu_cased":[mu_cased], "mu_open":[mu_open],
                "Case":[mode], "WOB_klbf":[wob_klbf], "Tbit_kftlbf":[tbit_kftlbf]
            }).to_excel(xw, index=False, sheet_name="Inputs_Strings")
            trace.round(6).to_excel(xw, index=False, sheet_name="Iteration_Trace")
            # Curves
            internal = pd.DataFrame({"md_ft": trace["md_to_ft"],
                                     "F_lbf": trace["F_next_lbf"],
                                     "T_lbf_ft": trace["T_next_lbf_ft"]})
            internal.to_excel(xw, index=False, sheet_name="Curves_Internal")
            for name, dfc in td_curves.items():
                dfc.to_excel(xw, index=False, sheet_name=f"Curves_{name[:25]}")
            # Report
            pd.DataFrame({"Report": report}).to_excel(xw, index=False, sheet_name="Report")
        with open("TD_Output.xlsx", "rb") as f:
            st.download_button("Download Excel (XLSX)", f, file_name="TD_Output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # -------------------- FOOTER --------------------
    st.info("Engine check: torque_drag calc signature and usage per docs (SPE-11380-PA). Well path loading per well_profile docs.")
