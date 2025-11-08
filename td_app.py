# td_app.py
import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ===== Units & helpers =====
IN2M = 0.0254
FT2M = 0.3048
G = 9.80665
N2LBF = 1 / 4.4482216
Mode = Literal["PU", "SL", "ROB"]

@dataclass
class Segment:
    md_ft: float
    inc_deg: float
    azi_deg: float
    dmd_ft: float
    dinc_rad: float
    dazi_rad: float

@dataclass
class StringSection:
    top_md_ft: float
    shoe_md_ft: float
    od_in: float
    id_in: float
    ann_od_in: float
    mud_ppg: float
    steel_sg: float = 7.85
    mu: float = 0.28

@dataclass
class LoadCase:
    mode: Mode
    wob_klbf: float = 50.0
    tbit_kftlbf: float = 0.0

def ppg_to_sg(ppg: float) -> float:
    return ppg / 8.33  # 8.33 ppg ≈ SG 1.0

def areas_m2(od_in, id_in, ann_od_in):
    ro = (od_in * IN2M) / 2
    ri = (id_in * IN2M) / 2
    r_ann = (ann_od_in * IN2M) / 2
    A_m = math.pi * (ro**2 - ri**2)   # metal area
    A_od = math.pi * (ro**2)          # displaced outside
    A_id = math.pi * (ri**2)          # displaced inside
    A_ann = math.pi * (r_ann**2)
    return ro, A_m, A_od, A_id, A_ann

def buoyant_weight_N_per_m(steel_sg, mud_sg, ro_m, A_m, A_od, A_id):
    # pipe immersed in fluid and filled with fluid (inside + outside buoyancy)
    rho_s = steel_sg * 1000.0
    rho_f = mud_sg * 1000.0
    return G * (rho_s * A_m - rho_f * A_od - rho_f * A_id)

# ===== Survey handling =====
def resample_to_step(survey_df: pd.DataFrame, step_ft: float = 1.0) -> List[Segment]:
    """Linearly resample MD/Inc/Azi to 1-ft spacing and build per-foot segments."""
    s = survey_df.sort_values("md_ft").reset_index(drop=True)
    md_min = int(round(s["md_ft"].iloc[0]))
    md_max = int(round(s["md_ft"].iloc[-1]))
    grid = np.arange(md_min, md_max + 1, step_ft)
    md = s["md_ft"].values
    inc = s["inc_deg"].values
    azi = s["azi_deg"].values
    inc_i = np.interp(grid, md, inc)
    azi_i = np.interp(grid, md, azi)
    segs: List[Segment] = []
    for i in range(1, len(grid)):
        dmd = grid[i] - grid[i - 1]
        dinc = math.radians(inc_i[i] - inc_i[i - 1])
        dazi = math.radians(azi_i[i] - azi_i[i - 1])
        segs.append(Segment(grid[i], float(inc_i[i]), float(azi_i[i]), float(dmd), dinc, dazi))
    return segs

def min_curvature_xyz(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Compute TVD, North, East using minimum curvature (ft/deg inputs)."""
    s = survey_df.sort_values("md_ft").reset_index(drop=True).copy()
    s["tvd_ft"] = 0.0; s["north_ft"] = 0.0; s["east_ft"] = 0.0
    for i in range(1, len(s)):
        md1, md2 = s.loc[i-1,"md_ft"], s.loc[i,"md_ft"]
        inc1, inc2 = math.radians(s.loc[i-1,"inc_deg"]), math.radians(s.loc[i,"inc_deg"])
        azi1, azi2 = math.radians(s.loc[i-1,"azi_deg"]), math.radians(s.loc[i,"azi_deg"])
        dmd = md2 - md1
        cos_dog = (math.sin(inc1)*math.sin(inc2)*math.cos(azi2-azi1) + math.cos(inc1)*math.cos(inc2))
        cos_dog = max(min(cos_dog, 1.0), -1.0)
        dog = math.acos(cos_dog)
        rf = (2/dog)*math.tan(dog/2) if dog > 1e-9 else 1.0
        north = 0.5*dmd*(math.sin(inc1)*math.cos(azi1) + math.sin(inc2)*math.cos(azi2))*rf
        east  = 0.5*dmd*(math.sin(inc1)*math.sin(azi1) + math.sin(inc2)*math.sin(azi2))*rf
        tvd   = 0.5*dmd*(math.cos(inc1) + math.cos(inc2))*rf
        s.loc[i,"north_ft"] = s.loc[i-1,"north_ft"] + north
        s.loc[i,"east_ft"]  = s.loc[i-1,"east_ft"] + east
        s.loc[i,"tvd_ft"]   = s.loc[i-1,"tvd_ft"] + tvd
    ref_az = math.atan2(s["east_ft"].iloc[-1], s["north_ft"].iloc[-1])
    s["vs_ft"] = s["north_ft"]*math.cos(ref_az) + s["east_ft"]*math.sin(ref_az)
    return s

# ===== Soft-string (Johancsik) with trace & μ-calibration =====
def solve_soft_string(
    segs: List[Segment],
    sec: StringSection,
    case: LoadCase,
    trace: bool = True,
    trace_strategy: Literal["ends", "all", "none"] = "ends",
    trace_rows: int = 12
) -> Dict:
    ro_m, A_m, A_od, A_id, _ = areas_m2(sec.od_in, sec.id_in, sec.ann_od_in)
    mud_sg = ppg_to_sg(sec.mud_ppg)
    w_N_per_m = buoyant_weight_N_per_m(sec.steel_sg, mud_sg, ro_m, A_m, A_od, A_id)
    w_lbf_per_ft = (w_N_per_m / FT2M) * N2LBF

    F_next = case.wob_klbf * 1000.0
    T_next = case.tbit_kftlbf * 1000.0
    F_profile, T_profile, md = [F_next], [T_next], [segs[-1].md_ft]
    rows = []

    for s in reversed(segs):  # bottom → top
        Ibar = math.radians(s.inc_deg)
        W = w_lbf_per_ft * s.dmd_ft
        term1 = F_next * s.dazi_rad * math.sin(Ibar)
        term2 = F_next * s.dinc_rad + W * math.sin(Ibar)
        FN = math.hypot(term1, term2)
        if case.mode == "PU":
            dF = W*math.cos(Ibar) + sec.mu*FN
        elif case.mode == "SL":
            dF = W*math.cos(Ibar) - sec.mu*FN
        else:
            dF = W*math.cos(Ibar)
        F_n = F_next + dF
        dT = sec.mu * FN * (ro_m / FT2M)
        T_n = T_next + dT

        if trace and trace_strategy != "none":
            rows.append({
                "md_to_ft": s.md_ft, "Ibar_deg": s.inc_deg,
                "dI_rad": s.dinc_rad, "dPsi_rad": s.dazi_rad,
                "W_lbf": W, "F_next_lbf": F_next, "FN_lbf": FN,
                "mu": sec.mu, "deltaF_lbf": dF, "F_n_lbf": F_n,
                "T_next_lbf_ft": T_next, "dT_lbf_ft": dT, "T_n_lbf_ft": T_n
            })

        F_profile.append(F_n); T_profile.append(T_n); md.append(s.md_ft)
        F_next, T_next = F_n, T_n

    F_profile, T_profile, md = list(reversed(F_profile)), list(reversed(T_profile)), list(reversed(md))
    shown_rows = []
    if rows:
        rows_surf_to_bit = list(reversed(rows))
        if trace_strategy == "all":
            shown_rows = rows_surf_to_bit
        elif trace_strategy == "ends":
            k = max(1, trace_rows // 2)
            shown_rows = rows_surf_to_bit[:k] + [{"...": "..."}] + rows_surf_to_bit[-k:]
    return {
        "md_ft": md, "F_lbf": F_profile, "T_lbf_ft": T_profile,
        "hookload_lbf": F_profile[0], "surface_torque_lbf_ft": T_profile[0],
        "unit_w_lbf_per_ft": w_lbf_per_ft, "ro_in": sec.od_in / 2, "trace_rows": shown_rows
    }

def calibrate_mu_bisection(
    segs: List[Segment],
    sec: StringSection,
    case: LoadCase,
    hookload_target_lbf: float,
    mu_lo: float = 0.10, mu_hi: float = 0.50,
    tol_lbf: float = 50.0, max_iter: int = 20
) -> Tuple[float, List[dict]]:
    log = []
    for it in range(1, max_iter + 1):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        sec.mu = mu_mid
        out = solve_soft_string(segs, sec, case, trace=False, trace_strategy="none")
        err = out["hookload_lbf"] - hookload_target_lbf
        log.append({"iter": it, "mu": mu_mid, "hookload_pred_lbf": out["hookload_lbf"], "error_lbf": err})
        if abs(err) <= tol_lbf:
            return mu_mid, log
        if err > 0: mu_hi = mu_mid
        else: mu_lo = mu_mid
    return mu_mid, log

# ===== Synthetic wells =====
def gen_build_hold(total_md_ft=12000, build_rate_deg_per100ft=3.0, target_inc_deg=60.0, azimuth_deg=0.0):
    md = [0.0]; inc = [0.0]; azi = [azimuth_deg]; step = 100.0
    while md[-1] < total_md_ft:
        md_next = min(md[-1] + step, total_md_ft)
        inc_next = min(inc[-1] + build_rate_deg_per100ft, target_inc_deg) if inc[-1] < target_inc_deg else inc[-1]
        md.append(md_next); inc.append(inc_next); azi.append(azimuth_deg)
    return pd.DataFrame({"md_ft": md, "inc_deg": inc, "azi_deg": azi})

def gen_s_curve(total_md_ft=12000, build_rate=3.0, drop_rate=3.0, max_inc_deg=40.0, azimuth_deg=0.0):
    md = [0.0]; inc = [0.0]; azi = [azimuth_deg]; step = 100.0; phase = "build"
    while md[-1] < total_md_ft:
        md_next = min(md[-1] + step, total_md_ft)
        inc_next = min(inc[-1] + build_rate, max_inc_deg) if phase=="build" else max(inc[-1]-drop_rate, 0.0)
        if inc_next >= max_inc_deg: phase = "drop"
        md.append(md_next); inc.append(inc_next); azi.append(azimuth_deg)
    return pd.DataFrame({"md_ft": md, "inc_deg": inc, "azi_deg": azi})

def gen_horizontal_lateral(kop_ft=2000, build_rate=3.0, lateral_ft=2000, azimuth_deg=0.0):
    md = [0.0]; inc = [0.0]; azi = [azimuth_deg]; step = 100.0
    build_len = 90.0 / build_rate * 100.0
    total_md = kop_ft + build_len + lateral_ft
    while md[-1] < total_md:
        md_next = min(md[-1] + step, total_md)
        if md_next <= kop_ft: inc_next = 0.0
        elif md[-1] < kop_ft + build_len: inc_next = min(90.0, inc[-1] + build_rate)
        else: inc_next = 90.0
        md.append(md_next); inc.append(inc_next); azi.append(azimuth_deg)
    return pd.DataFrame({"md_ft": md, "inc_deg": inc, "azi_deg": azi})

# ===== Report text =====
def make_report_text(sec: StringSection, case: LoadCase, out: Dict, mu_log: Optional[List[dict]] = None) -> str:
    ro_m, A_m, A_od, A_id, _ = areas_m2(sec.od_in, sec.id_in, sec.ann_od_in)
    mud_sg = ppg_to_sg(sec.mud_ppg)
    wNpm = buoyant_weight_N_per_m(sec.steel_sg, mud_sg, ro_m, A_m, A_od, A_id)
    wNpf = wNpm / FT2M
    lines = [
        "=== Labeled Inputs ===",
        f"Step = 1.0 ft | Mode = {case.mode}",
        f"OD={sec.od_in:.3f} in, ID={sec.id_in:.3f} in, AnnOD={sec.ann_od_in:.3f} in",
        f"Mud = {sec.mud_ppg:.2f} ppg (SG={mud_sg:.3f}) | Steel SG={sec.steel_sg:.2f} | μ={sec.mu:.3f}",
        f"WOB = {case.wob_klbf:.2f} klbf | Tbit = {case.tbit_kftlbf:.2f} kft-lbf",
        "",
        "=== Documented Formulas (Johancsik soft-string) ===",
        "F_N = √[(F_{n+1} Δψ sinĪ)^2 + (F_{n+1} ΔI + W sinĪ)^2]",
        "PU:  F_n = F_{n+1} + W cosĪ + μ F_N",
        "SL:  F_n = F_{n+1} + W cosĪ − μ F_N",
        "ROB: F_n = F_{n+1} + W cosĪ",
        "ΔM = μ F_N r_o",
        f"Buoyant unit weight ≈ {wNpf*N2LBF:.2f} lbf/ft",
        "",
        "=== Results (surface) ===",
        f"Hookload = {out['hookload_lbf']:.0f} lbf",
        f"Surface torque = {out['surface_torque_lbf_ft']:.0f} lbf·ft",
        "",
        "=== Iteration Trace (bottom → top, selected rows) ===",
        "md_to(ft)  Ī(deg)  ΔI(rad)  Δψ(rad)   W(lbf)   F_next   F_N     μ    ΔF     F_n     T_next   ΔT     T_n"
    ]
    for r in out["trace_rows"]:
        if "..." in r:
            lines.append("  ... (rows omitted) ...")
        else:
            lines.append(f"{r['md_to_ft']:8.1f} {r['Ibar_deg']:7.2f} {r['dI_rad']:7.4f} {r['dPsi_rad']:7.4f} "
                         f"{r['W_lbf']:7.1f} {r['F_next_lbf']:7.0f} {r['FN_lbf']:7.0f} {r['mu']:4.2f} "
                         f"{r['deltaF_lbf']:7.0f} {r['F_n_lbf']:7.0f} {r['T_next_lbf_ft']:8.0f} "
                         f"{r['dT_lbf_ft']:6.1f} {r['T_n_lbf_ft']:8.0f}")
    if mu_log:
        lines += ["", "=== μ-calibration iteration (bisection) ===", "iter   μ        hookload_pred(lbf)   error(lbf)"]
        for j in mu_log:
            lines.append(f"{j['iter']:>3d}  {j['mu']:.4f}   {j['hookload_pred_lbf']:>10.0f}         {j['error_lbf']:>+8.0f}")
    return "\n".join(lines)

# ===== Streamlit UI =====
st.set_page_config(page_title="Soft-String Torque & Drag (Johancsik)", layout="wide")
st.title("Soft-String Torque & Drag — 1-ft steps (Johancsik)")

with st.sidebar:
    st.header("1) Survey")
    src = st.radio("Source", ["Upload CSV", "Synthetic"], horizontal=True)
    if src == "Upload CSV":
        st.caption("CSV columns required: md_ft, inc_deg, azi_deg")
        f = st.file_uploader("Upload survey CSV", type=["csv"])
        survey_df = pd.read_csv(f) if f else None
    else:
        synth_type = st.selectbox("Synthetic type", ["Build & Hold", "S-curve", "Horizontal + Lateral"])
        if synth_type == "Build & Hold":
            total_md = st.number_input("Total MD (ft)", 5000, 30000, 12000, 100)
            build = st.number_input("Build rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            target_inc = st.number_input("Target inclination (°)", 5.0, 90.0, 60.0, 1.0)
            az = st.number_input("Azimuth (°)", 0.0, 360.0, 0.0, 1.0)
            survey_df = gen_build_hold(total_md, build, target_inc, az)
        elif synth_type == "S-curve":
            total_md = st.number_input("Total MD (ft)", 5000, 30000, 12000, 100)
            build = st.number_input("Build rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            drop = st.number_input("Drop rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            max_inc = st.number_input("Max inclination (°)", 5.0, 80.0, 40.0, 1.0)
            az = st.number_input("Azimuth (°)", 0.0, 360.0, 0.0, 1.0)
            survey_df = gen_s_curve(total_md, build, drop, max_inc, az)
        else:
            kop = st.number_input("Kickoff MD (ft)", 500, 5000, 2000, 50)
            build = st.number_input("Build rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            lat = st.number_input("Lateral length (ft)", 500, 5000, 2000, 50)
            az = st.number_input("Azimuth (°)", 0.0, 360.0, 0.0, 1.0)
            survey_df = gen_horizontal_lateral(kop, build, lat, az)

    st.header("2) String / Fluid")
    od = st.number_input("Pipe OD (in)", 2.0, 8.0, 5.0, 0.001, format="%.3f")
    id_ = st.number_input("Pipe ID (in)", 1.0, 7.5, 4.276, 0.001, format="%.3f")
    ann = st.number_input("Annulus OD (hole/casing ID) (in)", 3.0, 20.0, 6.500, 0.001, format="%.3f")
    mud_ppg = st.number_input("Mud weight (ppg)", 6.0, 20.0, 10.0, 0.1)
    steel_sg = st.number_input("Steel SG", 7.6, 8.1, 7.85, 0.01)
    mu = st.number_input("Friction factor μ", 0.05, 0.80, 0.28, 0.01)

    st.header("3) Load Case")
    mode = st.selectbox("Mode", ["PU", "SL", "ROB"])
    wob = st.number_input("WOB (klbf) at bit", 0.0, 200.0, 50.0, 1.0)
    tbit = st.number_input("Bit torque (kft-lbf)", 0.0, 50.0, 0.0, 0.5)

    st.header("4) Options")
    show_trace = st.checkbox("Show per-foot iteration trace", value=True)
    do_cal = st.checkbox("Calibrate μ to match a measured surface hookload?", value=False)
    target_hook = st.number_input("Target hookload (lbf)", 0, 500000, 180000, 1000, disabled=not do_cal)
    mu_lo = st.number_input("μ lower bound", 0.01, 0.80, 0.10, 0.01, disabled=not do_cal)
    mu_hi = st.number_input("μ upper bound", 0.02, 0.90, 0.50, 0.01, disabled=not do_cal)

    run_btn = st.button("Run T&D")

# ===== Main run =====
if run_btn:
    if survey_df is None or survey_df.empty:
        st.error("Please provide a survey (upload CSV or use a synthetic profile).")
        st.stop()
    req_cols = {"md_ft", "inc_deg", "azi_deg"}
    if not req_cols.issubset(set(survey_df.columns)):
        st.error("CSV must include columns: md_ft, inc_deg, azi_deg")
        st.stop()

    segs = resample_to_step(survey_df, step_ft=1.0)

    sec = StringSection(top_md_ft=float(survey_df["md_ft"].min()),
                        shoe_md_ft=float(survey_df["md_ft"].max()),
                        od_in=od, id_in=id_, ann_od_in=ann, mud_ppg=mud_ppg,
                        steel_sg=steel_sg, mu=mu)
    case = LoadCase(mode=mode, wob_klbf=wob, tbit_kftlbf=tbit)

    out = solve_soft_string(segs, sec, case, trace=show_trace, trace_strategy="ends", trace_rows=12)

    mu_log = None
    if do_cal:
        mu_star, mu_log = calibrate_mu_bisection(segs, sec, case, hookload_target_lbf=target_hook,
                                                 mu_lo=mu_lo, mu_hi=mu_hi)
        st.info(f"Calibrated μ ≈ {mu_star:.4f} (to match {target_hook} lbf). Re-running with μ* …")
        sec.mu = mu_star
        out = solve_soft_string(segs, sec, case, trace=show_trace, trace_strategy="ends", trace_rows=12)

    # Trajectory plots
    survey_xyz = min_curvature_xyz(survey_df)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(survey_xyz, x="vs_ft", y="tvd_ft", title="Trajectory: TVD vs Vertical Section (VS)",
                       labels={"vs_ft": "VS (ft)", "tvd_ft": "TVD (ft)"})
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=survey_df["md_ft"], y=survey_df["inc_deg"], name="Inc (deg)"))
        fig2.add_trace(go.Scatter(x=survey_df["md_ft"], y=survey_df["azi_deg"], name="Azi (deg)"))
        fig2.update_layout(title="Inc/Azi vs MD", xaxis_title="MD (ft)", yaxis_title="Degrees")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(x=out["md_ft"], y=out["F_lbf"], title=f"{mode} Hookload vs MD",
                   labels={"x":"MD (ft)","y":"Force (lbf)"})
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.line(x=out["md_ft"], y=out["T_lbf_ft"], title=f"{mode} Torque vs MD",
                   labels={"x":"MD (ft)","y":"Torque (lbf·ft)"})
    st.plotly_chart(fig4, use_container_width=True)

    if show_trace and out["trace_rows"]:
        st.subheader("Iteration trace (selected rows)")
        st.dataframe(pd.DataFrame(out["trace_rows"]))

    st.success(f"Surface Hookload ({mode}) = {out['hookload_lbf']:.0f} lbf   |   "
               f"Surface Torque = {out['surface_torque_lbf_ft']:.0f} lbf·ft")

    report_text = make_report_text(sec, case, out, mu_log=mu_log)
    st.download_button("Download report (txt)", report_text.encode("utf-8"), file_name="td_report.txt")
    curves = pd.DataFrame({"md_ft": out["md_ft"], "F_lbf": out["F_lbf"], "T_lbf_ft": out["T_lbf_ft"]})
    st.download_button("Download curves (CSV)", curves.to_csv(index=False).encode("utf-8"), file_name="td_curves.csv")

# Footer help
st.markdown("---")
with st.expander("How to use this app"):
    st.markdown("""
**Step 1 — Survey**: Upload CSV with `md_ft, inc_deg, azi_deg` or pick a Synthetic profile.  
**Step 2 — String/Fluid**: Enter pipe OD/ID, annulus OD, mud (ppg), steel SG, friction μ.  
**Step 3 — Load Case**: Choose PU/SL/ROB, set WOB (klbf) and bit torque (kft-lbf).  
**Step 4 — Run**: See trajectory, Hookload/Torque, iteration trace; optionally calibrate μ.
""")
