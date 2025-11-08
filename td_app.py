# td_app.py — Soft-String Torque & Drag (Johancsik), 1-ft steps
# Adds: step-by-step derivations, technical report, 3D path, multi-sheet Excel export.

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict
from datetime import datetime
from io import BytesIO

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
    rho_s = steel_sg * 1000.0
    rho_f = mud_sg * 1000.0
    return G * (rho_s * A_m - rho_f * A_od - rho_f * A_id)

# ===== Survey handling (Minimum Curvature) =====
def resample_to_step(survey_df: pd.DataFrame, step_ft: float = 1.0) -> List[Segment]:
    s = survey_df.sort_values("md_ft").reset_index(drop=True)
    md_min = int(round(s["md_ft"].iloc[0]))
    md_max = int(round(s["md_ft"].iloc[-1]))
    grid = np.arange(md_min, md_max + 1, step_ft)
    inc_i = np.interp(grid, s["md_ft"], s["inc_deg"])
    azi_i = np.interp(grid, s["md_ft"], s["azi_deg"])
    segs: List[Segment] = []
    for i in range(1, len(grid)):
        dmd = grid[i] - grid[i - 1]
        dinc = math.radians(inc_i[i] - inc_i[i - 1])
        dazi = math.radians(azi_i[i] - azi_i[i - 1])
        segs.append(Segment(grid[i], float(inc_i[i]), float(azi_i[i]), float(dmd), dinc, dazi))
    return segs

def min_curvature_xyz(survey_df: pd.DataFrame) -> pd.DataFrame:
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

# ===== Soft-string (Johancsik) with trace =====
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
        else:  # ROB
            dF = W*math.cos(Ibar)
        F_n = F_next + dF
        dT = sec.mu * FN * (ro_m / FT2M)
        T_n = T_next + dT

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
    rows_surf_to_bit = list(reversed(rows))

    shown_rows = []
    if trace and rows_surf_to_bit:
        if trace_strategy == "all": shown_rows = rows_surf_to_bit
        else:
            k = max(1, trace_rows // 2)
            shown_rows = rows_surf_to_bit[:k] + [{"...": "..."}] + rows_surf_to_bit[-k:]

    return {
        "md_ft": md, "F_lbf": F_profile, "T_lbf_ft": T_profile,
        "hookload_lbf": F_profile[0], "surface_torque_lbf_ft": T_profile[0],
        "unit_w_lbf_per_ft": w_lbf_per_ft, "ro_in": sec.od_in / 2,
        "trace_rows": shown_rows, "trace_rows_full": rows_surf_to_bit
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

# ===== Sanity checks (prove correctness) =====
def run_self_checks(od, id_, ann, mud_ppg, steel_sg, base_mu):
    df_vert = pd.DataFrame({"md_ft": [0, 10000], "inc_deg": [0, 0], "azi_deg": [0, 0]})
    segs = resample_to_step(df_vert, 1.0)
    sec = StringSection(0, 10000, od, id_, ann, mud_ppg, steel_sg, base_mu)
    case = LoadCase("PU", 50.0, 0.0)
    out = solve_soft_string(segs, sec, case, trace=False)
    ro_m, A_m, A_od, A_id, _ = areas_m2(od, id_, ann)
    mud_sg = ppg_to_sg(mud_ppg)
    wNpm = buoyant_weight_N_per_m(steel_sg, mud_sg, ro_m, A_m, A_od, A_id)
    w_lbf_ft = (wNpm/FT2M)*N2LBF
    theory = 50_000 + w_lbf_ft*10000
    err1 = abs(out["hookload_lbf"] - theory)/max(theory, 1)*100.0
    sec.mu = 0.0
    out_pu = solve_soft_string(segs, sec, LoadCase("PU", 50.0, 0.0), trace=False)
    out_sl = solve_soft_string(segs, sec, LoadCase("SL", 50.0, 0.0), trace=False)
    diff = abs(out_pu["hookload_lbf"] - out_sl["hookload_lbf"])
    return {"vertical_err_pct": err1, "frictionless_diff_lbf": diff}

# ===== Derivations (step-by-step text) =====
def build_derivations(rows_surface_to_bit: List[dict], mode: str, n_head: int = 5, n_tail: int = 5) -> str:
    lines = ["Step-by-step numeric derivations (surface → bit):"]
    sel = rows_surface_to_bit[:n_head] + ([{"...":"..."}] if len(rows_surface_to_bit)>n_head+n_tail else []) + rows_surface_to_bit[-n_tail:]
    for r in sel:
        if "..." in r: lines.append("  ... (many similar 1-ft steps omitted) ..."); continue
        Ibar = r["Ibar_deg"]; dI = r["dI_rad"]; dPsi = r["dPsi_rad"]; W = r["W_lbf"]
        Fnext = r["F_next_lbf"]; FN = r["FN_lbf"]; mu = r["mu"]; dF = r["deltaF_lbf"]; Fn = r["F_n_lbf"]
        dT = r["dT_lbf_ft"]; Tnext = r["T_next_lbf_ft"]; Tn = r["T_n_lbf_ft"]; md_to = r["md_to_ft"]
        lines += [
            f"MD→ {md_to:.1f} ft | Ī={Ibar:.2f}°, ΔI={dI:.4f} rad, Δψ={dPsi:.4f} rad",
            f"  F_N = √[(F_next·Δψ·sinĪ)^2 + (F_next·ΔI + W·sinĪ)^2]",
            f"      = √[({Fnext:.1f}*{dPsi:.4f}*sin{math.radians(Ibar):.4f})^2 + "
            f"({Fnext:.1f}*{dI:.4f} + {W:.1f}*sin{math.radians(Ibar):.4f})^2] = {FN:.1f} lbf",
        ]
        if mode == "PU":
            lines.append(f"  ΔF = W·cosĪ + μ·F_N = {W:.1f}*cos{math.radians(Ibar):.4f} + {mu:.3f}*{FN:.1f} = {dF:.1f} lbf")
        elif mode == "SL":
            lines.append(f"  ΔF = W·cosĪ − μ·F_N = {W:.1f}*cos{math.radians(Ibar):.4f} − {mu:.3f}*{FN:.1f} = {dF:.1f} lbf")
        else:
            lines.append(f"  ΔF = W·cosĪ = {W:.1f}*cos{math.radians(Ibar):.4f} = {dF:.1f} lbf")
        lines += [
            f"  F_n = F_next + ΔF = {Fnext:.1f} + {dF:.1f} = {Fn:.1f} lbf",
            f"  ΔM = μ·F_N·r_o = {mu:.3f}*{FN:.1f}*r_o  →  ΔM = {dT:.1f} lbf·ft",
            f"  M_n = M_next + ΔM = {Tnext:.1f} + {dT:.1f} = {Tn:.1f} lbf·ft",
            ""
        ]
    return "\n".join(lines)

# ===== Report builder =====
def make_report_text(team_names: str,
                     methodology: str,
                     assumptions: str,
                     insights_user: str,
                     sec: StringSection, case: LoadCase, out: Dict,
                     survey_xyz: pd.DataFrame,
                     header_help: Dict[str, str],
                     mu_log: Optional[List[dict]] = None,
                     derivations_text: Optional[str] = None,
                     overlays: Optional[List[Dict]] = None) -> str:

    ro_m, A_m, A_od, A_id, _ = areas_m2(sec.od_in, sec.id_in, sec.ann_od_in)
    mud_sg = ppg_to_sg(sec.mud_ppg)
    wNpm = buoyant_weight_N_per_m(sec.steel_sg, mud_sg, ro_m, A_m, A_od, A_id)
    wNpf = wNpm / FT2M

    # Auto insights (base + overlay sensitivity)
    auto = []
    auto.append(f"Base μ={sec.mu:.2f}: surface hookload {out['hookload_lbf']:.0f} lbf; surface torque {out['surface_torque_lbf_ft']:.0f} lbf·ft.")
    if overlays:
        hooks = [(o["mu"], o["out"]["hookload_lbf"]) for o in overlays]
        torqs = [(o["mu"], o["out"]["surface_torque_lbf_ft"]) for o in overlays]
        hooks_sorted = sorted(hooks); torqs_sorted = sorted(torqs)
        auto.append(f"Hookload sensitivity μ: {hooks_sorted[0][0]:.2f}→{hooks_sorted[-1][0]:.2f} gives {hooks_sorted[0][1]:.0f}→{hooks_sorted[-1][1]:.0f} lbf at surface.")
        auto.append(f"Torque sensitivity μ: {torqs_sorted[0][0]:.2f}→{torqs_sorted[-1][0]:.2f} gives {torqs_sorted[0][1]:.0f}→{torqs_sorted[-1][1]:.0f} lbf·ft.")

    EQUATIONS = [
        "Trajectory: Minimum Curvature (TVD/N/E/VS).",
        "Buoyant unit weight:   w_b = g(ρ_s A_m − ρ_f A_od − ρ_f A_id).",
        "Normal force:          F_N = √[(F_{n+1} Δψ sinĪ)^2 + (F_{n+1} ΔI + W sinĪ)^2].",
        "Axial recursion (PU):  F_n = F_{n+1} + W cosĪ + μ F_N.",
        "Axial recursion (SL):  F_n = F_{n+1} + W cosĪ − μ F_N.",
        "ROB (static):          F_n = F_{n+1} + W cosĪ.",
        "Torque increment:      ΔM = μ F_N r_o."
    ]

    lines = []
    lines += [
        f"Soft-String Torque & Drag — Technical Report   ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        f"Team: {team_names}",
        "",
        "A. Results & Discussion",
        "\n".join(auto),
        insights_user if insights_user.strip() else "",
        "",
        "B. Engineering Insights (trends & sensitivities)",
        "- PU vs SL spread grows with inclination and μ (sliding friction adds on PU, subtracts on SL).",
        "- Higher mud weight reduces buoyant weight and thus hookload; friction trend depends on normal force × μ.",
        "- Torque increases with inclination and μ; lateral sections dominate torque growth.",
        "",
        "C. Methodology",
        methodology,
        "",
        "D. Assumptions",
        assumptions,
        "",
        "E. Equations Used",
        *EQUATIONS,
        f"\nMetal area A_m = π/4(OD^2−ID^2) = {A_m/(IN2M**2):.3f} in²; Buoyant unit weight = {wNpm:.1f} N/m = {wNpf*N2LBF:.2f} lbf/ft.",
        "",
        "F. Inputs Summary",
        f"Mode={case.mode} | WOB={case.wob_klbf:.2f} klbf | Tbit={case.tbit_kftlbf:.2f} kft-lbf | μ={sec.mu:.3f}",
        f"Pipe OD={sec.od_in:.3f} in, ID={sec.id_in:.3f} in, Annulus OD={sec.ann_od_in:.3f} in; Mud={sec.mud_ppg:.2f} ppg (SG={mud_sg:.3f}); Steel SG={sec.steel_sg:.2f}",
        f"MD range: {survey_xyz['md_ft'].min():.0f}–{survey_xyz['md_ft'].max():.0f} ft; Max inc: {survey_xyz['inc_deg'].max():.1f}°",
        "",
        "G. Surface Results",
        f"Hookload = {out['hookload_lbf']:.0f} lbf;  Surface torque = {out['surface_torque_lbf_ft']:.0f} lbf·ft.",
        "",
        "H. Iteration Trace (columns explained)",
        ", ".join([f"{k}={v}" for k,v in TRACE_HEADER_HELP.items()]),
        "",
        "I. Step-by-step numeric derivations (subset)",
        derivations_text if derivations_text else "(enable in UI)",
    ]

    if mu_log:
        lines += ["", "J. μ-calibration log (bisection)", "iter   μ        hookload_pred(lbf)   error(lbf)"]
        for j in mu_log:
            lines.append(f"{j['iter']:>3d}  {j['mu']:.4f}   {j['hookload_pred_lbf']:>10.0f}         {j['error_lbf']:>+8.0f}")

    return "\n".join(lines)

# ===== Trace header help (also appended in report) =====
TRACE_HEADER_HELP = {
    "md_to_ft": "End MD (ft) of the current 1-ft segment (we march bottom→top).",
    "Ibar_deg": "Segment inclination used for this step (deg).",
    "dI_rad": "Change in inclination over the 1-ft step (radians).",
    "dPsi_rad": "Change in azimuth over the 1-ft step (radians).",
    "W_lbf": "Buoyant segment weight (lbf) for this foot.",
    "F_next_lbf": "Axial force at the lower node (toward bit) before this step (lbf).",
    "FN_lbf": "Normal (side) force from Johancsik discrete relation (lbf).",
    "mu": "Friction factor (dimensionless).",
    "deltaF_lbf": "Axial force change across the step (lbf). PU:+μFN; SL:−μFN; ROB:0.",
    "F_n_lbf": "Axial force at the upper node (toward surface) after this step (lbf).",
    "T_next_lbf_ft": "Torque at the lower node (lbf·ft) before this step.",
    "dT_lbf_ft": "Torque increase across the step (lbf·ft).",
    "T_n_lbf_ft": "Torque at the upper node (lbf·ft) after this step."
}

# ===== UI =====
st.set_page_config(page_title="Soft-String Torque & Drag (Johancsik)", layout="wide")
st.title("Soft-String Torque & Drag — 1-ft steps (Johancsik)")

with st.sidebar:
    st.header("0) Team & Report")
    team_names = st.text_input("Team names (comma-separated)", "")
    methodology = st.text_area("Methodology (auto-fill allowed)", 
        "Soft-string model (Johancsik). 1-ft discretization bottom→top. "
        "Trajectory by Minimum Curvature. μ constant per run; optional bisection calibration to match measured hookload.")
    assumptions = st.text_area("Assumptions (edit)", 
        "Uniform μ along string; Coulomb friction with side-force F_N; pipe fully fluid-filled; "
        "buoyancy inside & outside; no soft/hard string transition; no buckling included.")
    insights_user = st.text_area("Your insights to show on top of report (optional)", 
        "PU–SL spread increases with μ and inclination; higher mud weight reduces buoyant weight and thus lowers hookload.")

    st.header("1) Survey")
    src = st.radio("Source", ["Upload CSV", "Synthetic"], horizontal=True)
    if src == "Upload CSV":
        st.caption("CSV required: md_ft, inc_deg, azi_deg")
        f = st.file_uploader("Upload survey CSV", type=["csv"])
        survey_df = pd.read_csv(f) if f else None
    else:
        synth_type = st.selectbox("Synthetic type", ["Build & Hold", "S-curve", "Horizontal + Lateral"])
        if synth_type == "Build & Hold":
            total_md = st.number_input("Total MD (ft)", 5000, 30000, 12000, 100)
            build = st.number_input("Build rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            target_inc = st.number_input("Target inclination (°)", 5.0, 90.0, 60.0, 1.0)
            az = st.number_input("Azimuth (°)", 0.0, 360.0, 0.0, 1.0)
            md=[0.0]; inc=[0.0]; azi=[az]; step=100.0
            while md[-1] < total_md:
                md_next=min(md[-1]+step,total_md)
                inc_next=min(inc[-1]+build, target_inc) if inc[-1] < target_inc else inc[-1]
                md.append(md_next); inc.append(inc_next); azi.append(az)
            survey_df=pd.DataFrame({"md_ft":md,"inc_deg":inc,"azi_deg":azi})
        elif synth_type == "S-curve":
            total_md = st.number_input("Total MD (ft)", 5000, 30000, 12000, 100)
            build = st.number_input("Build rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            drop = st.number_input("Drop rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            max_inc = st.number_input("Max inclination (°)", 5.0, 80.0, 40.0, 1.0)
            az = st.number_input("Azimuth (°)", 0.0, 360.0, 0.0, 1.0)
            md=[0.0]; inc=[0.0]; azi=[az]; step=100.0; phase="build"
            while md[-1] < total_md:
                md_next=min(md[-1]+step,total_md)
                inc_next=min(inc[-1]+build,max_inc) if phase=="build" else max(inc[-1]-drop,0.0)
                if inc_next>=max_inc: phase="drop"
                md.append(md_next); inc.append(inc_next); azi.append(az)
            survey_df=pd.DataFrame({"md_ft":md,"inc_deg":inc,"azi_deg":azi})
        else:
            kop = st.number_input("Kickoff MD (ft)", 500, 5000, 2000, 50)
            build = st.number_input("Build rate (°/100 ft)", 1.0, 10.0, 3.0, 0.5)
            lat = st.number_input("Lateral length (ft)", 500, 8000, 2000, 50)
            az = st.number_input("Azimuth (°)", 0.0, 360.0, 0.0, 1.0)
            build_len = 90.0/build*100.0
            total_md = kop + build_len + lat
            md=[0.0]; inc=[0.0]; azi=[az]; step=100.0
            while md[-1] < total_md:
                md_next=min(md[-1]+step,total_md)
                if md_next <= kop: inc_next=0.0
                elif md[-1] < kop+build_len: inc_next=min(90.0, inc[-1]+build)
                else: inc_next=90.0
                md.append(md_next); inc.append(inc_next); azi.append(az)
            survey_df=pd.DataFrame({"md_ft":md,"inc_deg":inc,"azi_deg":azi})

    st.header("2) String / Fluid")
    od = st.number_input("Pipe OD (in)", 2.0, 8.0, 5.0, 0.001, format="%.3f")
    id_ = st.number_input("Pipe ID (in)", 1.0, 7.5, 4.276, 0.001, format="%.3f")
    ann = st.number_input("Annulus OD (hole/casing ID) (in)", 3.0, 20.0, 6.500, 0.001, format="%.3f")
    mud_ppg = st.number_input("Mud weight (ppg)", 6.0, 20.0, 10.0, 0.1)
    steel_sg = st.number_input("Steel SG", 7.6, 8.1, 7.85, 0.01)

    st.header("3) Friction / Loads")
    mu_base = st.number_input("Base friction μ (for trace/report)", 0.05, 0.80, 0.28, 0.01)
    mu_overlay_on = st.checkbox("Overlay multiple μ values (comma-separated)", value=True)
    mu_overlay_str = st.text_input("μ list (e.g., 0.20,0.30,0.40)", "0.20,0.30,0.40", disabled=not mu_overlay_on)
    mode = st.selectbox("Mode", ["PU", "SL", "ROB"])
    wob = st.number_input("WOB (klbf) at bit", 0.0, 200.0, 50.0, 1.0)
    tbit = st.number_input("Bit torque (kft-lbf)", 0.0, 50.0, 0.0, 0.5)

    st.header("4) Options")
    show_trace = st.checkbox("Show per-foot iteration trace (selected rows)", value=True)
    show_deriv = st.checkbox("Include step-by-step numeric derivations in report", value=True)
    deriv_head = st.number_input("Derivation head rows", 1, 50, 6, 1, disabled=not show_deriv)
    deriv_tail = st.number_input("Derivation tail rows", 1, 50, 6, 1, disabled=not show_deriv)
    do_cal = st.checkbox("Calibrate μ to match a measured surface hookload?", value=False)
    target_hook = st.number_input("Target hookload (lbf)", 0, 500000, 180000, 1000, disabled=not do_cal)
    mu_lo = st.number_input("μ lower bound", 0.01, 0.80, 0.10, 0.01, disabled=not do_cal)
    mu_hi = st.number_input("μ upper bound", 0.02, 0.90, 0.50, 0.01, disabled=not do_cal)
    run_checks = st.checkbox("Run built-in accuracy self-checks", value=False)

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

    # 1-ft segments
    segs = resample_to_step(survey_df, step_ft=1.0)

    # Base case (used for trace & report)
    sec = StringSection(top_md_ft=float(survey_df["md_ft"].min()),
                        shoe_md_ft=float(survey_df["md_ft"].max()),
                        od_in=od, id_in=id_, ann_od_in=ann, mud_ppg=mud_ppg,
                        steel_sg=steel_sg, mu=mu_base)
    case = LoadCase(mode=mode, wob_klbf=wob, tbit_kftlbf=tbit)

    mu_log = None
    if do_cal:
        mu_star, mu_log = calibrate_mu_bisection(segs, sec, case, hookload_target_lbf=target_hook,
                                                 mu_lo=mu_lo, mu_hi=mu_hi)
        st.info(f"Calibrated μ ≈ {mu_star:.4f} (to match {target_hook} lbf). Re-running with μ* …")
        sec.mu = mu_star

    out_base = solve_soft_string(segs, sec, case, trace=show_trace, trace_strategy="ends", trace_rows=12)

    # μ overlays
    overlays = []
    if mu_overlay_on and mu_overlay_str.strip():
        try:
            mu_list = [float(x) for x in mu_overlay_str.split(",")]
            mu_list = [m for m in mu_list if 0 <= m <= 1]
            for m in mu_list:
                sec_tmp = StringSection(sec.top_md_ft, sec.shoe_md_ft, od, id_, ann, mud_ppg, steel_sg, m)
                out_m = solve_soft_string(segs, sec_tmp, case, trace=False, trace_strategy="none")
                overlays.append({"mu": m, "out": out_m})
        except Exception:
            st.warning("Could not parse μ list; showing base case only.")
            overlays = []

    # 3D well path
    survey_xyz = min_curvature_xyz(survey_df)
    fig3d = go.Figure(data=[go.Scatter3d(
        x=survey_xyz["east_ft"], y=survey_xyz["north_ft"], z=survey_xyz["tvd_ft"],
        mode="lines",
        line=dict(width=6, color=survey_xyz["inc_deg"], colorscale="Viridis"),
        name="Well path"
    )])
    fig3d.update_layout(title="3D Well Path (colored by inclination)",
                        scene=dict(
                            xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)",
                            zaxis=dict(autorange="reversed")
                        ))

    # 2D trajectory & surveys
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(survey_xyz, x="vs_ft", y="tvd_ft", title="Trajectory: TVD vs Vertical Section (VS)",
                       labels={"vs_ft":"VS (ft)", "tvd_ft":"TVD (ft)"})
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=survey_df["md_ft"], y=survey_df["inc_deg"], name="Inc (deg)"))
        fig2.add_trace(go.Scatter(x=survey_df["md_ft"], y=survey_df["azi_deg"], name="Azi (deg)"))
        fig2.update_layout(title="Inc/Azi vs MD", xaxis_title="MD (ft)", yaxis_title="Degrees")
        st.plotly_chart(fig2, use_container_width=True)

    st.plotly_chart(fig3d, use_container_width=True)

    # Hookload vs MD (with overlays)
    figF = go.Figure()
    figF.add_trace(go.Scatter(x=out_base["md_ft"], y=out_base["F_lbf"], name=f"{mode} μ={sec.mu:.2f}", mode="lines"))
    for o in overlays:
        figF.add_trace(go.Scatter(x=o["out"]["md_ft"], y=o["out"]["F_lbf"], name=f"{mode} μ={o['mu']:.2f}", mode="lines"))
    figF.update_layout(title=f"{mode} Hookload vs MD", xaxis_title="MD (ft)", yaxis_title="Force (lbf)")
    st.plotly_chart(figF, use_container_width=True)

    # Torque vs MD (with overlays)
    figT = go.Figure()
    figT.add_trace(go.Scatter(x=out_base["md_ft"], y=out_base["T_lbf_ft"], name=f"{mode} μ={sec.mu:.2f}", mode="lines"))
    for o in overlays:
        figT.add_trace(go.Scatter(x=o["out"]["md_ft"], y=o["out"]["T_lbf_ft"], name=f"{mode} μ={o['mu']:.2f}", mode="lines"))
    figT.update_layout(title=f"{mode} Torque vs MD", xaxis_title="MD (ft)", yaxis_title="Torque (lbf·ft)")
    st.plotly_chart(figT, use_container_width=True)

    # Trace table (selected rows)
    if show_trace and out_base["trace_rows"]:
        st.subheader("Iteration trace (selected rows)")
        st.dataframe(pd.DataFrame(out_base["trace_rows"]))

    # Step-by-step derivations
    deriv_text = ""
    if show_deriv:
        deriv_text = build_derivations(out_base["trace_rows_full"], case.mode, deriv_head, deriv_tail)
        with st.expander("Step-by-step numeric derivations (printable)"):
            st.code(deriv_text, language="text")
        st.download_button("Download derivations (txt)", deriv_text.encode("utf-8"),
                           file_name="td_derivations.txt")

    # Summary
    st.success(
        f"Surface Hookload ({case.mode}) = {out_base['hookload_lbf']:.0f} lbf   |   "
        f"Surface Torque = {out_base['surface_torque_lbf_ft']:.0f} lbf·ft"
    )

    # Self checks
    if run_checks:
        chk = run_self_checks(od, id_, ann, mud_ppg, steel_sg, sec.mu)
        if chk["vertical_err_pct"] < 0.5:
            st.info(f"Vertical-well check: OK (error {chk['vertical_err_pct']:.2f}%).")
        else:
            st.warning(f"Vertical-well check: large error ({chk['vertical_err_pct']:.2f}%).")
        if chk["frictionless_diff_lbf"] < 5.0:
            st.info(f"Frictionless check (μ=0): PU ≈ SL (Δ {chk['frictionless_diff_lbf']:.2f} lbf).")
        else:
            st.warning(f"Frictionless check: PU-SL difference {chk['frictionless_diff_lbf']:.1f} lbf (expected ~0).")

    # Build technical report text
    report_text = make_report_text(team_names, methodology, assumptions, insights_user,
                                   sec, case, out_base, survey_xyz, TRACE_HEADER_HELP,
                                   mu_log=mu_log, derivations_text=deriv_text, overlays=overlays)
    st.download_button("Download technical report (txt)", report_text.encode("utf-8"), file_name="td_report.txt")

    # CSV exports
    curves = pd.DataFrame({"md_ft": out_base["md_ft"], "F_lbf": out_base["F_lbf"], "T_lbf_ft": out_base["T_lbf_ft"]})
    st.download_button("Download curves (CSV)", curves.to_csv(index=False).encode("utf-8"), file_name="td_curves.csv")
    trace_full_df = pd.DataFrame(out_base["trace_rows_full"])
    st.download_button("Download full iteration trace (CSV)", trace_full_df.to_csv(index=False).encode("utf-8"),
                       file_name="td_trace_full.csv")

    # Excel workbook (multi-sheet)
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            # Inputs
            pd.DataFrame({
                "Parameter":["Team","Mode","WOB_klbf","Tbit_kftlbf","μ","Pipe_OD_in","Pipe_ID_in","Annulus_OD_in",
                             "Mud_ppg","Steel_SG","MD_min_ft","MD_max_ft","Max_inc_deg"],
                "Value":[team_names, case.mode, case.wob_klbf, case.tbit_kftlbf, sec.mu, od, id_, ann,
                         mud_ppg, steel_sg, survey_xyz["md_ft"].min(), survey_xyz["md_ft"].max(),
                         survey_xyz["inc_deg"].max()]
            }).to_excel(writer, sheet_name="Inputs", index=False)

            # Equations
            eq = pd.DataFrame({"Equation":[
                "Minimum Curvature for TVD/N/E/VS",
                "w_b = g(ρ_s A_m − ρ_f A_od − ρ_f A_id)",
                "F_N = √[(F_{n+1} Δψ sinĪ)^2 + (F_{n+1} ΔI + W sinĪ)^2]",
                "PU:  F_n = F_{n+1} + W cosĪ + μ F_N",
                "SL:  F_n = F_{n+1} + W cosĪ − μ F_N",
                "ROB: F_n = F_{n+1} + W cosĪ",
                "ΔM = μ F_N r_o"
            ]})
            eq.to_excel(writer, sheet_name="Equations", index=False)

            # Curves & Trace
            curves.to_excel(writer, sheet_name="Curves", index=False)
            trace_full_df.to_excel(writer, sheet_name="Trace", index=False)

            # SelfChecks
            sc = run_checks and run_self_checks(od, id_, ann, mud_ppg, steel_sg, sec.mu) or {}
            pd.DataFrame([sc]).to_excel(writer, sheet_name="SelfChecks", index=False)

            # Report (single cell)
            ws = writer.book.add_worksheet("Report")
            ws.write_string(0, 0, report_text)
            ws.set_column(0, 0, 110)
            ws.set_row(0, 200)

        st.download_button("Download Excel workbook (.xlsx)", buffer.getvalue(),
                           file_name="td_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Footer / Help
st.markdown("---")
with st.expander("How to use this app"):
    st.markdown("""
**Step 0 — Team & Report**: Fill team names and (optionally) edit methodology/assumptions/insights.  
**Step 1 — Survey**: Upload `md_ft, inc_deg, azi_deg` or choose a Synthetic profile.  
**Step 2 — String/Fluid**: Pipe OD/ID, annulus OD (hole/casing ID), mud (ppg), steel SG.  
**Step 3 — Friction & Loads**: Select μ (and optional overlays), PU/SL/ROB, WOB, bit torque.  
**Step 4 — Options**: Show trace, include step-by-step derivations, enable μ-calibration and self-checks.  
**Run**: Get 2D & 3D trajectory, Hookload/Torque vs MD, trace table, derivations, and downloads (TXT/CSV/XLSX).
""")
st.caption("Method: Johancsik soft-string (SPE-11380) with 1-ft steps; trajectory via Minimum Curvature.")
