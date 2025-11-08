# td_app.py — Soft-String Torque & Drag (Johancsik), 1-ft steps
# Upgrades: multi-μ overlays, 3D well path, detailed report w/ equations & worked example,
# full iteration trace CSV, and built-in accuracy self-checks.

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict
from datetime import datetime

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
    """Compute TVD, North, East via minimum curvature (standard in wellpath calcs)."""
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
        else:  # ROB (rotating off bottom)
            dF = W*math.cos(Ibar)
        F_n = F_next + dF
        dT = sec.mu * FN * (ro_m / FT2M)
        T_n = T_next + dT

        # Store full trace row (we’ll show selected rows later)
        rows.append({
            "md_to_ft": s.md_ft, "Ibar_deg": s.inc_deg,
            "dI_rad": s.dinc_rad, "dPsi_rad": s.dazi_rad,
            "W_lbf": W, "F_next_lbf": F_next, "FN_lbf": FN,
            "mu": sec.mu, "deltaF_lbf": dF, "F_n_lbf": F_n,
            "T_next_lbf_ft": T_next, "dT_lbf_ft": dT, "T_n_lbf_ft": T_n
        })

        F_profile.append(F_n); T_profile.append(T_n); md.append(s.md_ft)
        F_next, T_next = F_n, T_n

    # Reverse arrays to surface→bit
    F_profile, T_profile, md = list(reversed(F_profile)), list(reversed(T_profile)), list(reversed(md))
    rows_surf_to_bit = list(reversed(rows))

    # Pick selected rows to display
    shown_rows = []
    if trace and rows_surf_to_bit:
        if trace_strategy == "all":
            shown_rows = rows_surf_to_bit
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

# ===== Built-in accuracy self-checks =====
def run_self_checks(od, id_, ann, mud_ppg, steel_sg, base_mu):
    # vertical well: inc=0 → friction term → 0 → hookload = WOB + buoyed weight
    df_vert = pd.DataFrame({"md_ft": [0, 10000], "inc_deg": [0, 0], "azi_deg": [0, 0]})
    segs = resample_to_step(df_vert, 1.0)
    sec = StringSection(0, 10000, od, id_, ann, mud_ppg, steel_sg, base_mu)
    case = LoadCase("PU", 50.0, 0.0)
    out = solve_soft_string(segs, sec, case, trace=False)
    # theory for vertical: WOB + w_b * L
    ro_m, A_m, A_od, A_id, _ = areas_m2(od, id_, ann)
    mud_sg = ppg_to_sg(mud_ppg)
    wNpm = buoyant_weight_N_per_m(steel_sg, mud_sg, ro_m, A_m, A_od, A_id)
    w_lbf_ft = (wNpm/FT2M)*N2LBF
    theory = 50_000 + w_lbf_ft*10000
    err1 = abs(out["hookload_lbf"] - theory)/max(theory, 1)*100.0

    # frictionless check: μ=0 → PU ~= SL
    sec.mu = 0.0
    out_pu = solve_soft_string(segs, sec, LoadCase("PU", 50.0, 0.0), trace=False)
    out_sl = solve_soft_string(segs, sec, LoadCase("SL", 50.0, 0.0), trace=False)
    diff = abs(out_pu["hookload_lbf"] - out_sl["hookload_lbf"])

    return {"vertical_err_pct": err1, "frictionless_diff_lbf": diff}

# ===== Report builder =====
def make_report_text(sec: StringSection, case: LoadCase, out: Dict,
                     survey_xyz: pd.DataFrame,
                     header_help: Dict[str, str],
                     mu_log: Optional[List[dict]] = None) -> str:
    ro_m, A_m, A_od, A_id, _ = areas_m2(sec.od_in, sec.id_in, sec.ann_od_in)
    mud_sg = ppg_to_sg(sec.mud_ppg)
    wNpm = buoyant_weight_N_per_m(sec.steel_sg, mud_sg, ro_m, A_m, A_od, A_id)
    wNpf = wNpm / FT2M

    # choose one representative step (surface-most) for numeric substitution
    sample = next((r for r in out["trace_rows"] if "..." not in r), None)

    lines = []
    lines += [
        f"Soft-String Torque & Drag — Report   ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "",
        "=== Labeled Inputs ===",
        f"Step = 1.0 ft | Mode = {case.mode}",
        f"Pipe: OD={sec.od_in:.3f} in, ID={sec.id_in:.3f} in, Annulus OD={sec.ann_od_in:.3f} in",
        f"Fluids/Materials: Mud = {sec.mud_ppg:.2f} ppg (SG={mud_sg:.3f}), Steel SG={sec.steel_sg:.2f}, μ={sec.mu:.3f}",
        f"Loads at bit: WOB = {case.wob_klbf:.2f} klbf, Tbit = {case.tbit_kftlbf:.2f} kft-lbf",
        f"Trajectory: MD range = {survey_xyz['md_ft'].min():.0f}–{survey_xyz['md_ft'].max():.0f} ft, max inc = {survey_xyz['inc_deg'].max():.1f}°",
        "",
        "=== Documented Formulas (Johancsik soft-string) ===",
        "Buoyant unit weight:    w_b = g(ρ_s A_m − ρ_f A_od − ρ_f A_id)    [N/m]",
        f" → A_m = π/4 (OD^2 − ID^2) = {A_m/(IN2M**2):.3f} in^2   (converted internally to m^2)",
        f" → w_b = {wNpm:.1f} N/m = {wNpf*N2LBF:.2f} lbf/ft",
        "Normal force (per foot): F_N = √[(F_{n+1} Δψ sinĪ)^2 + (F_{n+1} ΔI + W sinĪ)^2]",
        "Axial recursion (PU):    F_n = F_{n+1} + W cosĪ + μ F_N",
        "Axial recursion (SL):    F_n = F_{n+1} + W cosĪ − μ F_N",
        "Static/ROB:              F_n = F_{n+1} + W cosĪ",
        "Torque increment:        ΔM = μ F_N r_o",
        "",
        "=== Worked example (one 1-ft step) ==="
    ]
    if sample and "..." not in sample:
        Ibar = sample['Ibar_deg']; dI = sample['dI_rad']; dPsi = sample['dPsi_rad']
        W = sample['W_lbf']; Fnext = sample['F_next_lbf']; mu = sample['mu']
        FN = sample['FN_lbf']; dF = sample['deltaF_lbf']; Fn = sample['F_n_lbf']
        dT = sample['dT_lbf_ft']; Tnext = sample['T_next_lbf_ft']
        lines += [
            f"Ī = {Ibar:.2f}°, ΔI = {dI:.4f} rad, Δψ = {dPsi:.4f} rad, W = {W:.1f} lbf, F_next = {Fnext:.0f} lbf, μ = {mu:.2f}",
            f"F_N = √[(F_next·Δψ·sin Ī)^2 + (F_next·ΔI + W·sin Ī)^2] = {FN:.1f} lbf",
            f"{case.mode}: ΔF = W·cos Ī {'+' if case.mode=='PU' else '-' if case.mode=='SL' else ''}{' μ·F_N' if case.mode!='ROB' else ''} = {dF:.1f} lbf → F_n = {Fn:.1f} lbf",
            f"ΔM = μ·F_N·r_o = {dT:.1f} lbf·ft;  M_n = M_next + ΔM = {Tnext + dT:.1f} lbf·ft"
        ]
    else:
        lines += ["(Trace unavailable for sample substitution.)"]

    lines += [
        "",
        "=== Results (surface) ===",
        f"Hookload = {out['hookload_lbf']:.0f} lbf",
        f"Surface torque = {out['surface_torque_lbf_ft']:.0f} lbf·ft",
        "",
        "=== Iteration Trace (bottom → top, selected rows) ===",
        "Columns: " + ", ".join(header_help.keys())
    ]
    # Include a few selected rows (already prepared)
    for r in out["trace_rows"]:
        if "..." in r:
            lines.append("  ... (rows omitted) ...")
        else:
            lines.append(
                f"{r['md_to_ft']:8.1f} | Ī={r['Ibar_deg']:6.2f}° | ΔI={r['dI_rad']:7.4f} | Δψ={r['dPsi_rad']:7.4f} | "
                f"W={r['W_lbf']:7.1f} | F_next={r['F_next_lbf']:7.0f} | F_N={r['FN_lbf']:7.0f} | μ={r['mu']:4.2f} | "
                f"ΔF={r['deltaF_lbf']:7.0f} | F_n={r['F_n_lbf']:7.0f} | M_next={r['T_next_lbf_ft']:8.0f} | "
                f"ΔM={r['dT_lbf_ft']:6.1f} | M_n={r['T_n_lbf_ft']:8.0f}"
            )
    if mu_log:
        lines += ["", "=== μ-calibration iteration (bisection) ===", "iter   μ        hookload_pred(lbf)   error(lbf)"]
        for j in mu_log:
            lines.append(f"{j['iter']:>3d}  {j['mu']:.4f}   {j['hookload_pred_lbf']:>10.0f}         {j['error_lbf']:>+8.0f}")

    lines += ["", "=== Column header explanations ==="]
    for k, v in header_help.items():
        lines.append(f"{k}: {v}")

    return "\n".join(lines)

# ===== Header explanations for the trace table =====
TRACE_HEADER_HELP = {
    "md_to_ft": "End MD (ft) of the current 1‑ft segment (we march bottom→top).",
    "Ibar_deg": "Segment inclination used for this step (deg).",
    "dI_rad": "Change in inclination over the 1‑ft step (radians).",
    "dPsi_rad": "Change in azimuth over the 1‑ft step (radians).",
    "W_lbf": "Buoyant segment weight (lbf) for this foot.",
    "F_next_lbf": "Axial force at the lower node (toward bit) before applying this step (lbf).",
    "FN_lbf": "Normal (side) force from Johancsik discrete relation (lbf).",
    "mu": "Friction factor used at this step (dimensionless).",
    "deltaF_lbf": "Axial force change across the step (lbf). Mode‑dependent sign.",
    "F_n_lbf": "Axial force at the upper node (toward surface) after this step (lbf).",
    "T_next_lbf_ft": "Torque at the lower node (lbf·ft) before this step.",
    "dT_lbf_ft": "Torque increase across the step (lbf·ft).",
    "T_n_lbf_ft": "Torque at the upper node (lbf·ft) after this step."
}

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
            survey_df = pd.DataFrame({"md_ft":[0.0], "inc_deg":[0.0], "azi_deg":[az]})
            # build segment
            md = [0.0]; inc=[0.0]; azi=[az]; step=100.0
            while md[-1] < total_md:
                md_next = min(md[-1]+step, total_md)
                inc_next = min(inc[-1]+build, target_inc) if inc[-1] < target_inc else inc[-1]
                md.append(md_next); inc.append(inc_next); azi.append(az)
            survey_df = pd.DataFrame({"md_ft":md,"inc_deg":inc,"azi_deg":azi})
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
            # build to 90°, then hold
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
    mu_base = st.number_input("Base friction factor μ (for trace/report)", 0.05, 0.80, 0.28, 0.01)
    mu_overlay_on = st.checkbox("Overlay multiple μ values (comma-separated)", value=True)
    mu_overlay_str = st.text_input("μ list (e.g., 0.20,0.30,0.40)", "0.20,0.30,0.40", disabled=not mu_overlay_on)
    mode = st.selectbox("Mode", ["PU", "SL", "ROB"])
    wob = st.number_input("WOB (klbf) at bit", 0.0, 200.0, 50.0, 1.0)
    tbit = st.number_input("Bit torque (kft-lbf)", 0.0, 50.0, 0.0, 0.5)

    st.header("4) Options")
    show_trace = st.checkbox("Show per-foot iteration trace (selected rows)", value=True)
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

    # Base (trace/report) case
    sec = StringSection(top_md_ft=float(survey_df["md_ft"].min()),
                        shoe_md_ft=float(survey_df["md_ft"].max()),
                        od_in=od, id_in=id_, ann_od_in=ann, mud_ppg=mud_ppg,
                        steel_sg=steel_sg, mu=mu_base)
    case = LoadCase(mode=mode, wob_klbf=wob, tbit_kftlbf=tbit)

    # Optional μ calibration
    mu_log = None
    if do_cal:
        mu_star, mu_log = calibrate_mu_bisection(segs, sec, case, hookload_target_lbf=target_hook,
                                                 mu_lo=mu_lo, mu_hi=mu_hi)
        st.info(f"Calibrated μ ≈ {mu_star:.4f} (to match {target_hook} lbf). Re-running with μ* …")
        sec.mu = mu_star

    # Solve base
    out_base = solve_soft_string(segs, sec, case, trace=show_trace, trace_strategy="ends", trace_rows=12)

    # Multi-μ overlays
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

    # 3D well path (TVD vs North/East)
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
                            zaxis=dict(autorange="reversed")  # depth downwards
                        ))
    # Plots: 2D trajectory, Inc/Azi, Hookload, Torque
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(survey_xyz, x="vs_ft", y="tvd_ft",
                       title="Trajectory: TVD vs Vertical Section (VS)",
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

    # Hookload vs MD
    figF = go.Figure()
    figF.add_trace(go.Scatter(x=out_base["md_ft"], y=out_base["F_lbf"],
                              name=f"{mode} μ={sec.mu:.2f}", mode="lines"))
    for o in overlays:
        figF.add_trace(go.Scatter(x=o["out"]["md_ft"], y=o["out"]["F_lbf"],
                                  name=f"{mode} μ={o['mu']:.2f}", mode="lines"))
    figF.update_layout(title=f"{mode} Hookload vs MD", xaxis_title="MD (ft)", yaxis_title="Force (lbf)")
    st.plotly_chart(figF, use_container_width=True)

    # Torque vs MD
    figT = go.Figure()
    figT.add_trace(go.Scatter(x=out_base["md_ft"], y=out_base["T_lbf_ft"],
                              name=f"{mode} μ={sec.mu:.2f}", mode="lines"))
    for o in overlays:
        figT.add_trace(go.Scatter(x=o["out"]["md_ft"], y=o["out"]["T_lbf_ft"],
                                  name=f"{mode} μ={o['mu']:.2f}", mode="lines"))
    figT.update_layout(title=f"{mode} Torque vs MD", xaxis_title="MD (ft)", yaxis_title="Torque (lbf·ft)")
    st.plotly_chart(figT, use_container_width=True)

    # Trace table (selected rows)
    if show_trace and out_base["trace_rows"]:
        st.subheader("Iteration trace (selected rows)")
        st.dataframe(pd.DataFrame(out_base["trace_rows"]))

    # Results summary
    st.success(
        f"Surface Hookload ({mode}) = {out_base['hookload_lbf']:.0f} lbf   |   "
        f"Surface Torque = {out_base['surface_torque_lbf_ft']:.0f} lbf·ft"
    )

    # Optional self-checks
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

    # Report text
    report_text = make_report_text(sec, case, out_base, survey_xyz, TRACE_HEADER_HELP, mu_log=mu_log)
    st.download_button("Download report (txt)", report_text.encode("utf-8"), file_name="td_report.txt")

    # Exports
    curves = pd.DataFrame({"md_ft": out_base["md_ft"],
                           "F_lbf": out_base["F_lbf"],
                           "T_lbf_ft": out_base["T_lbf_ft"]})
    st.download_button("Download curves (CSV)", curves.to_csv(index=False).encode("utf-8"), file_name="td_curves.csv")

    # Full trace CSV (surface→bit)
    trace_full_df = pd.DataFrame(out_base["trace_rows_full"])
    st.download_button("Download full iteration trace (CSV)", trace_full_df.to_csv(index=False).encode("utf-8"),
                       file_name="td_trace_full.csv")

# Footer / Help
st.markdown("---")
with st.expander("How to use this app"):
    st.markdown("""
**Step 1 — Survey**: Upload CSV with `md_ft, inc_deg, azi_deg` **or** pick a Synthetic profile.  
**Step 2 — String/Fluid**: Enter pipe OD/ID, annulus OD (hole/casing ID), mud (ppg), steel SG.  
**Step 3 — Friction & Loads**: Pick μ (base), optionally add multiple μ values to overlay; select PU/SL/ROB; set WOB, bit torque.  
**Step 4 — Run**: See 2D & 3D trajectory, Hookload/Torque, iteration trace; optionally calibrate μ to match a measured surface hookload; download report & CSVs.
""")
st.caption("Method: Johancsik soft-string (SPE-11380). 1-ft discretization. Trajectory via Minimum Curvature.")
