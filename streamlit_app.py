# streamlit_app.py  — PEGN517 Wellpath + Torque & Drag (Δs = 1 ft)
# Key features
# - Minimum Curvature survey (ratio factor) at 1 ft step
# - VS (Vertical Section) plot, Profile (TVD-MD), and Plan (E-N)
# - 3D schematic with legend; cased vs open-hole colors; line width ~ ID/hole OD
# - Standards-only simple inputs: deepest shoe + last casing size + open-hole (bit) size
# - Detailed Casing/Liner/Open-hole editor (optional)
# - Soft-string Torque & Drag (Johancsik): buoyancy, cased vs open μ, on/off-bottom
# - μ-sweep overlay for quick sensitivity
#
# References used in the design (see README/report):
#   Minimum Curvature + ratio factor: industry standard for surveys.  (cf. Maxwellsci PDF; DirectionalDrillingArt) 
#   Vertical Section definition: project onto a vertical plane of reference azimuth. 
#   Buoyancy factor BF = (65.5 − MW)/65.5 (steel ≈ 65.5 ppg).
#   Johancsik soft-string step-wise axial/torque accumulation; sliding friction dominates in good holes.
#   API RP 7G combined Tension–Torque envelope (for later overlay).
#
# NOTE: All piecewise calculations use Δs = 1 ft.

import math
from typing import Tuple, List, Dict
import io
import csv

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Wellpath + Torque & Drag — PEGN517", layout="wide")

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ---------------------------
# Standards mini-library
# ---------------------------
# API-like casing subsets (Nominal OD -> (lb/ft -> ID))
# Values representative of API 5CT tables commonly used in planning.
CASING_DB: Dict[str, Dict[str, List[Dict[str, float]]]] = {
    "13-3/8": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf": 48.0, "id": 12.415},
            {"ppf": 54.5, "id": 12.347},
            {"ppf": 61.0,  "id": 12.115},
            {"ppf": 68.0,  "id": 11.965},
        ],
    },
    "9-5/8": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf": 36.0, "id": 8.835},
            {"ppf": 40.0, "id": 8.535},
            {"ppf": 43.5, "id": 8.405},
            {"ppf": 47.0, "id": 8.311},
            {"ppf": 53.5, "id": 8.097},
        ],
    },
    "7-5/8": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf": 39.0,  "id": 6.625},
            {"ppf": 42.8,  "id": 6.501},
            {"ppf": 45.3,  "id": 6.435},
            {"ppf": 47.1,  "id": 6.375},
            {"ppf": 51.2,  "id": 6.249},
            {"ppf": 52.8,  "id": 6.201},
            {"ppf": 55.75, "id": 6.176},
        ],
    },
    "7": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf": 23.0, "id": 6.366},
            {"ppf": 26.0, "id": 6.276},
            {"ppf": 29.0, "id": 6.184},
            {"ppf": 32.0, "id": 6.094},
            {"ppf": 35.0, "id": 6.004},
            {"ppf": 38.0, "id": 5.938},
            {"ppf": 41.0, "id": 5.921},
        ],
    },
    "5-1/2": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf": 14.0, "id": 4.892},
            {"ppf": 17.0, "id": 4.778},
            {"ppf": 20.0, "id": 4.670},
            {"ppf": 23.0, "id": 4.563},
            {"ppf": 26.0, "id": 4.494},
        ],
    },
    "4-1/2": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf": 9.5,  "id": 4.090},
            {"ppf": 10.5, "id": 4.052},
            {"ppf": 11.6, "id": 4.000},
            {"ppf": 13.5, "id": 3.920},
        ],
    },
}

# Common bit/hole sizes (open hole OD)
HOLE_SIZES = [
    "36","26","24","20",
    "17-1/2","16","14-3/4","12-1/4",
    "10-5/8","9-7/8","8-3/4","8-1/2",
    "6-1/2","6-1/8","6","5-7/8","4-3/4"
]

def parse_inch_str(label: str) -> float:
    """Convert '13-3/8' or '12-1/4' to float inches."""
    if "-" in label:
        whole, frac = label.split("-")
        n, d = frac.split("/")
        return float(whole) + float(n)/float(d)
    return float(label)

# ---------------------------
# Minimum Curvature (Δs = 1)
# ---------------------------
def _ratio_factor(dogs_rad: float) -> float:
    """Minimum Curvature ratio factor with small-angle limit."""
    if abs(dogs_rad) < 1e-12:
        return 1.0
    return 2.0 / dogs_rad * math.tan(0.5 * dogs_rad)

def _mcm_step(md1, inc1_deg, az1_deg, md2, inc2_deg, az2_deg, n1, e1, tvd1):
    ds = md2 - md1
    inc1 = math.radians(inc1_deg); inc2 = math.radians(inc2_deg)
    az1  = math.radians(az1_deg);  az2  = math.radians(az2_deg)
    cosd = (math.cos(inc1) * math.cos(inc2)
            + math.sin(inc1) * math.sin(inc2) * math.cos(az2 - az1))
    cosd = max(-1.0, min(1.0, cosd))
    dogs = math.acos(cosd)
    rf   = _ratio_factor(dogs)
    dN = 0.5 * ds * (math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2)) * rf
    dE = 0.5 * ds * (math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2)) * rf
    dT = 0.5 * ds * (math.cos(inc1) + math.cos(inc2)) * rf
    n2, e2, tvd2 = n1+dN, e1+dE, tvd1+dT
    dls = 0.0 if ds <= 0 else math.degrees(dogs)/ds * 100.0
    return n2, e2, tvd2, dls

def mcm_positions(md, inc, az):
    north, east, tvd, dls = [0.0], [0.0], [0.0], [0.0]
    for i in range(1, len(md)):
        n2, e2, tvd2, d = _mcm_step(md[i-1], inc[i-1], az[i-1], md[i], inc[i], az[i],
                                    north[-1], east[-1], tvd[-1])
        north.append(n2); east.append(e2); tvd.append(tvd2); dls.append(d)
    return north, east, tvd, dls

def _gen_md(target_md: float, ds: float = 1.0) -> List[float]:
    steps = int(max(0.0, target_md) // ds)
    md = [i*ds for i in range(steps + 1)]
    if md[-1] < target_md:
        md.append(target_md)
    return md

def synth_build_hold(kop_md, br, theta_hold, target_md, az_deg):
    ds = 1.0
    md = _gen_md(target_md, ds)
    inc = []
    for m in md:
        if m < kop_md:
            inc.append(0.0)
        else:
            prev = inc[-1] if inc else 0.0
            inc.append(min(prev + br*(ds/100.0), theta_hold))
    return md, inc, [az_deg]*len(md)

def synth_build_hold_drop(kop_md, br, theta_hold, hold_len, dr, final_inc, target_md, az_deg):
    ds=1.0; md=_gen_md(target_md, ds); inc=[]
    built=False; in_hold=False; in_drop=False; H = hold_len or 0.0
    for m in md:
        if m < kop_md:
            inc.append(0.0); continue
        prev = inc[-1] if inc else 0.0
        if not built:
            nxt = min(prev + br*(ds/100.0), theta_hold)
            built = abs(nxt - theta_hold) < 1e-12
            if built and H > 0: in_hold = True
            inc.append(nxt)
        elif in_hold:
            inc.append(theta_hold); H -= ds
            if H <= 0: in_hold, in_drop = False, True
        elif in_drop:
            nxt = max(prev - dr*(ds/100.0), final_inc)
            inc.append(nxt)
            if abs(nxt - final_inc) < 1e-12: in_drop = False
        else:
            inc.append(final_inc)
    return md, inc, [az_deg]*len(md)

def synth_horizontal(kop_md, br, lat_len, target_md, az_deg):
    ds=1.0
    if not target_md or target_md <= 0:
        build_len = math.ceil(90.0/(br/100.0))
        target_md = kop_md + build_len + (lat_len or 0.0)
    md=_gen_md(target_md, ds); inc=[]; built=False
    for m in md:
        if m < kop_md:
            inc.append(0.0); continue
        prev = inc[-1] if inc else 0.0
        if not built:
            nxt = min(prev + br*(ds/100.0), 90.0)
            built = abs(nxt-90.0) < 1e-12
            inc.append(nxt)
        else:
            inc.append(90.0)
    return md, inc, [az_deg]*len(md)

# ---------------------------
# Soft-string Torque & Drag
# ---------------------------
def buoyancy_factor_ppg(mw_ppg: float) -> float:
    # BF = (65.5 − MW)/65.5  (steel ≈ 65.5 ppg)
    return max(0.0, min(1.0, (65.5 - float(mw_ppg)) / 65.5))

def kappa_from_dls(dls_deg_100: float) -> float:
    return math.radians(dls_deg_100) / 100.0

def solve_torque_drag(md, inc_deg, dls_deg_100,
                      dc_len, dc_od, dc_id, dc_w,
                      hwdp_len, hwdp_od, hwdp_id, hwdp_w,
                      dp_len, dp_od, dp_id, dp_w,
                      cased_intervals: List[Tuple[float, float]],
                      mu_cased_slide, mu_open_slide,
                      mu_cased_rot=None, mu_open_rot=None,
                      mw_ppg=10.0, wob_lbf=0.0, bit_torque_lbf_ft=0.0,
                      scenario="Pickup"):
    n = len(md)
    kappa = [kappa_from_dls(x) for x in dls_deg_100]
    mu_rot_cased = mu_cased_rot if mu_cased_rot is not None else mu_cased_slide
    mu_rot_open  = mu_open_rot  if mu_open_rot  is not None else mu_open_slide

    bit_depth = md[-1]
    use_dc   = max(0.0, min(dc_len, bit_depth))
    use_hwdp = max(0.0, min(hwdp_len, bit_depth - use_dc))
    use_dp   = max(0.0, min(dp_len, bit_depth - use_dc - use_hwdp))

    def comp_at_md(m):
        if m > (bit_depth - use_dc):                 return dc_w, dc_od
        elif m > (bit_depth - use_dc - use_hwdp):    return hwdp_w, hwdp_od
        else:                                        return dp_w, dp_od

    def is_cased(middle_md: float) -> bool:
        for a,b in cased_intervals:
            if middle_md >= a - 1e-9 and middle_md <= b + 1e-9:
                return True
        return False

    BF = buoyancy_factor_ppg(mw_ppg)
    sigma_ax = +1.0 if scenario.lower().startswith("pick") else -1.0

    # Boundary at bit
    if scenario.lower().startswith("rotate on"):
        T = [-float(wob_lbf)]
        M = [float(bit_torque_lbf_ft)]
    else:
        T = [0.0]
        M = [0.0]

    rows = []
    # integrate bit -> surface
    for i in range(n-1, 0, -1):
        md2, md1 = md[i], md[i-1]; ds = md2 - md1
        theta = math.radians(inc_deg[i]); kap = kappa[i]
        w_air, od_in = comp_at_md(md2)
        w_b = w_air * BF
        r_eff_ft = (od_in/2.0)/12.0
        mid_md = 0.5*(md1 + md2)
        in_cased = is_cased(mid_md)
        mu_slide = mu_cased_slide if in_cased else mu_open_slide
        mu_rot   = mu_rot_cased  if in_cased else mu_open_rot

        # Normal force per unit length
        N = w_b*math.sin(theta) + T[-1]*kap

        # Axial & torque increments (Johancsik)
        dT = (sigma_ax*w_b*math.cos(theta) + mu_slide*N) * ds
        dM = (mu_rot * N * r_eff_ft) * ds

        T_next = T[-1] + dT
        M_next = M[-1] + dM

        rows.append({
            "md_top_ft": md1, "md_bot_ft": md2, "ds_ft": ds,
            "inc_deg": inc_deg[i], "kappa_rad_ft": kap,
            "w_air_lbft": w_air, "BF": BF, "w_b_lbft": w_b,
            "mu_slide": mu_slide, "mu_rot": mu_rot,
            "N_lbf": N, "dT_lbf": dT, "T_next_lbf": T_next,
            "r_eff_ft": r_eff_ft, "dM_lbf_ft": dM, "M_next_lbf_ft": M_next,
            "cased?": in_cased
        })
        T.append(T_next); M.append(M_next)

    T.reverse(); M.reverse(); rows.reverse()
    return {
        "T_lbf_along": T,
        "M_lbf_ft_along": M,
        "trace_rows": rows,
        "surface_hookload_lbf": T[0],
        "surface_torque_lbf_ft": M[0] + (bit_torque_lbf_ft if scenario.lower().startswith("rotate on") else 0.0),
    }

# -------------------------------------------
# Program helpers (detailed mode intervals)
# -------------------------------------------
def intervals_from_program(rows, td) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    cased, openh = [], []
    for r in rows:
        typ = r["type"]
        if typ == "Surface casing":
            cased.append((0.0, min(r["shoe_md"], td)))
        elif typ == "Liner":
            top = min(r["top_md"], td); bot = min(r["shoe_md"], td)
            if bot > top: cased.append((top, bot))
        elif typ == "Open hole":
            top = min(r["top_md"], td); bot = min(r["bot_md"], td)
            if bot > top: openh.append((top, bot))
    return cased, openh

# --------------- UI ---------------
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

tab1, tab2 = st.tabs(["Trajectory & 3D schematic", "Torque & Drag"])

# ---------- TAB 1: TRAJECTORY ----------
with tab1:
    st.subheader("Synthetic survey (Minimum Curvature)")

    colA, colB, colC = st.columns(3)
    with colA:
        profile = st.selectbox("Profile", ["Build & Hold", "Build & Hold & Drop", "Horizontal (Build + Lateral)"])
        kop_md  = st.number_input("KOP MD (ft)", 0.0, None, 1000.0, 50.0)
    with colB:
        quick = st.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], index=0)
        az_map = {"North (0)":0.0, "East (90)":90.0, "South (180)":180.0, "West (270)":270.0}
        az_deg = st.number_input("Azimuth (deg from North, clockwise)", 0.0, 360.0, az_map[quick], 1.0)
        build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
    with colC:
        theta_hold=hold_len=drop_rate=final_inc=lat_len=target_md=None
        if profile == "Build & Hold":
            theta_hold = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 1.0)
            target_md  = st.number_input("Target MD (ft)", kop_md, None, 10000.0, 100.0)
        elif profile == "Build & Hold & Drop":
            theta_hold = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 1.0)
            hold_len   = st.number_input("Hold length (ft)", 0.0, None, 1000.0, 50.0)
            drop_rate  = st.selectbox("Drop rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
            final_inc  = st.number_input("Final inclination after drop (deg)", 0.0, 90.0, 0.0, 1.0)
            target_md  = st.number_input("Target MD (ft)", kop_md, None, 12000.0, 100.0)
        else:
            lat_len   = st.number_input("Lateral length (ft)", 0.0, None, 2000.0, 100.0)
            target_md = st.number_input("Target MD (0 = auto)", 0.0, None, 0.0, 100.0) or None

    st.markdown("### Casing / Liner / Open-hole program (optional)")
    st.caption("Use simple mode (deepest shoe + last casing size + open-hole OD) or build a detailed program. All calcs use Δs = 1 ft.")

    use_simple = st.checkbox("Use simple inputs (deepest shoe + last casing size + open-hole diameter)", value=True)

    simple_sizes = None
    program = None

    if use_simple:
        s1, s2, s3 = st.columns(3)
        with s1:
            simple_shoe_md = st.number_input("Deepest shoe MD (ft)", 0.0, None, 3000.0, 50.0)
        with s2:
            last_od = st.selectbox("Last casing — Nominal OD", list(CASING_DB.keys()))
        with s3:
            opt = st.selectbox("Last casing — lb/ft (ID auto)", CASING_DB[last_od]["options"],
                               format_func=lambda o: f"{o['ppf']:.2f} #  →  ID {o['id']:.3f} in")
        s4, s5 = st.columns(2)
        with s4:
            hole_sel = st.selectbox("Open-hole diameter (standards)", HOLE_SIZES,
                                    index=HOLE_SIZES.index("12-1/4") if "12-1/4" in HOLE_SIZES else 0)
        with s5:
            st.write("")
            st.write(f"Selected open-hole OD = **{parse_inch_str(hole_sel):.3f} in**")
        simple_sizes = {
            "shoe_md": simple_shoe_md,
            "casing_id_in": float(opt["id"]),
            "casing_od_in": parse_inch_str(last_od),
            "openhole_od_in": parse_inch_str(hole_sel),
        }
    else:
        if "prog_rows" not in st.session_state:
            st.session_state["prog_rows"] = 1
        hdr = st.columns([1.1,1.1,0.9,0.9,1.1,1.1,1.1,1.1])
        hdr[0].markdown("**Type**"); hdr[1].markdown("**Nominal OD**"); hdr[2].markdown("**lb/ft**"); hdr[3].markdown("**ID (in)**")
        hdr[4].markdown("**Grade**"); hdr[5].markdown("**Top MD (ft)**"); hdr[6].markdown("**Shoe/Bottom MD (ft)**"); hdr[7].markdown("**Actions**")
        program=[]
        for i in range(st.session_state["prog_rows"]):
            c0,c1,c2,c3,c4,c5,c6,c7 = st.columns([1.1,1.1,0.9,0.9,1.1,1.1,1.1,1.1])
            typ = c0.selectbox("", ["Surface casing","Liner","Open hole"], key=f"type_{i}")
            if typ == "Open hole":
                top_md = c5.number_input("", 0.0, None, 0.0, 50.0, key=f"top_{i}")
                bot_md = c6.number_input("", 0.0, None, 0.0, 50.0, key=f"bot_{i}")
                if c7.button("Remove", key=f"rm_{i}"):
                    st.session_state["prog_rows"] = max(0, st.session_state["prog_rows"]-1)
                program.append({"type":typ, "top_md":top_md, "bot_md":bot_md})
                continue
            od = c1.selectbox("", list(CASING_DB.keys()), key=f"od_{i}")
            opts = CASING_DB[od]["options"]
            sel_opt = c2.selectbox("", opts, key=f"ppf_{i}",
                                   format_func=lambda o: f"{o['ppf']:.2f} #  →  ID {o['id']:.3f} in")
            c3.text_input("", f"{sel_opt['id']:.3f}", key=f"id_{i}", disabled=True)
            grade = c4.selectbox("", CASING_DB[od]["grades"], key=f"gr_{i}")
            if typ == "Surface casing":
                top_md = 0.0; shoe_md = c6.number_input("", 0.0, None, 3000.0, 50.0, key=f"shoe_{i}")
                c5.text_input("", "—", key=f"top_txt_{i}", disabled=True)
            else:
                top_md = c5.number_input("", 0.0, None, 7000.0, 50.0, key=f"top_{i}")
                shoe_md = c6.number_input("", 0.0, None, 9000.0, 50.0, key=f"shoe_{i}")
            if c7.button("Remove", key=f"rmc_{i}"):
                st.session_state["prog_rows"] = max(0, st.session_state["prog_rows"]-1)
            program.append({
                "type":typ, "od":od, "ppf":float(sel_opt["ppf"]), "id":sel_opt["id"], "grade":grade,
                "top_md":top_md, "shoe_md":shoe_md
            })
        add = st.columns([7,1])
        if add[1].button("Add interval"):
            st.session_state["prog_rows"] = min(12, st.session_state["prog_rows"]+1)

    # ---- Compute trajectory button ----
    if st.button("Compute trajectory & 3D schematic"):
        if profile == "Build & Hold":
            md, inc, az = synth_build_hold(kop_md, build_rate, theta_hold, target_md, az_deg)
        elif profile == "Build & Hold & Drop":
            md, inc, az = synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len, drop_rate, final_inc, target_md, az_deg)
        else:
            md, inc, az = synth_horizontal(kop_md, build_rate, lat_len, target_md, az_deg)

        north, east, tvd, dls = mcm_positions(md, inc, az)

        # VS (project onto reference azimuth = first azimuth by default)
        ref_az_deg = az[0]
        ref_rad = math.radians(ref_az_deg)
        vs = [ n*math.cos(ref_rad) + e*math.sin(ref_rad) for n,e in zip(north, east) ]

        st.session_state["last_traj"] = (md, inc, az, dls, tvd, north, east, vs)
        TD = md[-1]

        # Intervals for schematic & T&D
        if use_simple:
            cased_intervals = [(0.0, min(simple_sizes["shoe_md"], TD))]
            openhole_intervals = [(min(simple_sizes["shoe_md"], TD), TD)]
            st.session_state["cased_intervals"] = cased_intervals
            st.session_state["simple_sizes"] = simple_sizes
        else:
            cased_intervals, openhole_intervals = intervals_from_program(program, TD)
            st.session_state["cased_intervals"] = cased_intervals
            st.session_state["program"] = program
            st.session_state.pop("simple_sizes", None)

        # ----- 3D schematic -----
        fig3d = go.Figure()

        def seg_idx(top, bot):
            return [k for k, m in enumerate(md) if m >= top - 1e-9 and m <= bot + 1e-9]

        # Cased
        for (top, bot) in st.session_state["cased_intervals"]:
            idx = seg_idx(top, bot)
            if len(idx) < 2: continue
            if "simple_sizes" in st.session_state:
                width_px = max(2, int(2 + st.session_state["simple_sizes"]["casing_id_in"]/4.0))
                name = f"Cased: {int(top)}–{int(bot)} ft (ID {st.session_state['simple_sizes']['casing_id_in']:.3f} in)"
            else:
                # use first csg ID as width hint
                row = next((r for r in st.session_state.get("program", []) if r.get("id")), None)
                id_in = float(row["id"]) if row else 6.5
                width_px = max(2, int(2 + id_in/4.0))
                name = f"Cased: {int(top)}–{int(bot)} ft"
            fig3d.add_trace(go.Scatter3d(
                x=[east[k] for k in idx], y=[north[k] for k in idx], z=[-tvd[k] for k in idx],
                mode="lines", line=dict(width=width_px, color="#4C78A8"), name=name
            ))

        # Open-hole
        if use_simple:
            openhole_intervals = [(min(simple_sizes["shoe_md"], TD), TD)]
        for (top, bot) in openhole_intervals:
            idx = seg_idx(top, bot)
            if len(idx) < 2: continue
            width_px = 4
            label = f"Open hole: {int(top)}–{int(bot)} ft"
            if use_simple:
                width_px = max(2, int(2 + st.session_state["simple_sizes"]['openhole_od_in']/4.0))
                label += f" (OD {st.session_state['simple_sizes']['openhole_od_in']:.2f} in)"
            fig3d.add_trace(go.Scatter3d(
                x=[east[k] for k in idx], y=[north[k] for k in idx], z=[-tvd[k] for k in idx],
                mode="lines", line=dict(width=width_px, color="#8B4513"), name=label
            ))

        fig3d.update_layout(
            title="3D Trajectory — Cased vs Open-hole",
            scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
            legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0)
        )

        # ----- 2D plots -----
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(x=md, y=tvd, mode="lines", name="TVD vs MD"))
        fig_prof.update_yaxes(autorange="reversed")
        fig_prof.update_layout(title="Profile (TVD vs MD)", xaxis_title="MD (ft)", yaxis_title="TVD (ft)")

        fig_plan = go.Figure()
        fig_plan.add_trace(go.Scatter(x=east, y=north, mode="lines", name="Plan"))
        fig_plan.update_layout(title="Plan (East vs North)", xaxis_title="East (ft)", yaxis_title="North (ft)")

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=vs, y=tvd, mode="lines", name="VS profile"))
        fig_vs.update_yaxes(autorange="reversed")
        fig_vs.update_layout(title="Vertical Section profile", xaxis_title="VS (ft)", yaxis_title="TVD (ft)")

        A, B = st.columns([2,1])
        A.plotly_chart(fig3d, use_container_width=True)
        B.plotly_chart(fig_prof, use_container_width=True)
        B.plotly_chart(fig_plan, use_container_width=True)
        B.plotly_chart(fig_vs, use_container_width=True)

        # Survey table + CSV
        df = pd.DataFrame({
            "MD (ft)": md,
            "Inc (deg)": inc,
            "Az (deg)": az,
            "TVD (ft)": tvd,
            "North (ft)": north,
            "East (ft)": east,
            "VS (ft)": vs,
            "DLS (deg/100 ft)": dls
        })
        st.subheader("Survey and calculated positions")
        st.dataframe(df, use_container_width=True, height=420)
        csv_buf = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download trajectory CSV", data=csv_buf, file_name="trajectory.csv", mime="text/csv")

# ---------- TAB 2: T&D ----------
with tab2:
    st.subheader("Soft-string Torque & Drag (Johancsik) — with buoyancy")

    L, R = st.columns(2)
    with L:
        mw_ppg = st.number_input("Mud weight (ppg)", 6.0, 20.0, 10.0, 0.1)
        mu_cased_slide = st.number_input("μ in casing (sliding)", 0.05, 0.60, 0.25, 0.01)
        mu_open_slide  = st.number_input("μ in open hole (sliding)", 0.05, 0.60, 0.35, 0.01)
        mu_cased_rot   = st.number_input("μ in casing (rotating)", 0.00, 1.00, 0.25, 0.01)
        mu_open_rot    = st.number_input("μ in open hole (rotating)", 0.00, 1.00, 0.35, 0.01)
    with R:
        wob_lbf    = st.number_input("WOB (lbf) for on-bottom", 0.0, None, 0.0, 500.0)
        bit_torque = st.number_input("Bit torque (lbf-ft) for on-bottom", 0.0, None, 0.0, 50.0)
        scenario   = st.selectbox("Scenario", ["Pickup (POOH)", "Slack-off (RIH)", "Rotate off-bottom", "Rotate on-bottom"])

    st.markdown("#### Drillstring (bit up)")
    c1,c2,c3 = st.columns(3)
    with c1:
        dc_len = st.number_input("DC length (ft)", 0.0, None, 600.0, 10.0)
        dc_od  = st.number_input("DC OD (in)", 1.0, 12.0, 8.0, 0.5)
        dc_id  = st.number_input("DC ID (in)", 0.0, 10.0, 2.81, 0.01)
        dc_w   = st.number_input("DC weight (air, lb/ft)", 0.0, None, 66.7, 0.1)
    with c2:
        hwdp_len = st.number_input("HWDP length (ft)", 0.0, None, 1000.0, 10.0)
        hwdp_od  = st.number_input("HWDP OD (in)", 1.0, 7.0, 3.5, 0.25)
        hwdp_id  = st.number_input("HWDP ID (in)", 0.0, 6.0, 2.0, 0.01)
        hwdp_w   = st.number_input("HWDP weight (air, lb/ft)", 0.0, None, 16.0, 0.1)
    with c3:
        dp_len = st.number_input("DP length (ft)", 0.0, None, 7000.0, 10.0)
        dp_od  = st.number_input("DP OD (in)", 1.0, 6.0, 5.0, 0.25)
        dp_id  = st.number_input("DP ID (in)", 0.0, 5.5, 4.276, 0.01)
        dp_w   = st.number_input("DP weight (air, lb/ft)", 0.0, None, 19.5, 0.1)

    # μ sweep controls
    st.markdown("#### Optional: μ sweep (overlay)")
    sweep_on = st.checkbox("Add μ-sweep overlay", value=False)
    mu_min, mu_max, mu_steps = 0.20, 0.40, 3
    if sweep_on:
        colS1, colS2 = st.columns(2)
        mu_min = colS1.number_input("μ sweep start", 0.05, 1.00, 0.20, 0.01)
        mu_max = colS2.number_input("μ sweep end",   0.05, 1.00, 0.40, 0.01)
        mu_steps = st.slider("Number of curves", 2, 6, 3)

    if st.button("Run Torque & Drag"):
        if "last_traj" not in st.session_state:
            st.error("No trajectory found. Compute in Tab 1 first.")
        else:
            md, inc, az, dls, tvd, north, east, vs = st.session_state["last_traj"]
            cased_intervals = st.session_state.get("cased_intervals", [(0.0, 0.0)])

            # Base case
            base_out = solve_torque_drag(
                md, inc, dls,
                dc_len, dc_od, dc_id, dc_w,
                hwdp_len, hwdp_od, hwdp_id, hwdp_w,
                dp_len, dp_od, dp_id, dp_w,
                cased_intervals,
                mu_cased_slide, mu_open_slide,
                mu_cased_rot,  mu_open_rot,
                mw_ppg=mw_ppg, wob_lbf=wob_lbf, bit_torque_lbf_ft=bit_torque,
                scenario=("Pickup" if scenario.startswith("Pickup") else
                          "Slack-off" if scenario.startswith("Slack") else
                          "Rotate on-bottom" if scenario.endswith("on-bottom") else
                          "Rotate off-bottom")
            )

            st.success(
                f"{'PUW' if scenario.startswith('Pickup') else 'SOW' if scenario.startswith('Slack') else 'Hookload'} at surface: "
                f"{abs(base_out['surface_hookload_lbf']):,.0f} lbf  —  "
                f"Surface torque: {base_out['surface_torque_lbf_ft']:,.0f} lbf-ft"
            )

            # Charts
            fig_T = go.Figure()
            fig_T.add_trace(go.Scatter(x=md, y=base_out["T_lbf_along"], mode="lines",
                                       name=f"Tension ({scenario}) μc={mu_cased_slide:.2f}/μo={mu_open_slide:.2f}"))
            fig_T.update_layout(title="Tension / Hookload vs MD", xaxis_title="MD (ft)", yaxis_title="lbf")

            fig_M = go.Figure()
            fig_M.add_trace(go.Scatter(x=md, y=base_out["M_lbf_ft_along"], mode="lines",
                                       name=f"Torque ({scenario})"))
            fig_M.update_layout(title="Torque vs MD", xaxis_title="MD (ft)", yaxis_title="lbf-ft")

            # μ sweep overlays
            if sweep_on:
                if mu_steps < 2: mu_steps = 2
                if mu_max < mu_min: mu_max = mu_min
                grid = [mu_min + i*(mu_max-mu_min)/(mu_steps-1) for i in range(mu_steps)]
                for mu in grid:
                    out = solve_torque_drag(
                        md, inc, dls,
                        dc_len, dc_od, dc_id, dc_w,
                        hwdp_len, hwdp_od, hwdp_id, hwdp_w,
                        dp_len, dp_od, dp_id, dp_w,
                        cased_intervals,
                        mu, mu, mu, mu,
                        mw_ppg=mw_ppg, wob_lbf=wob_lbf, bit_torque_lbf_ft=bit_torque,
                        scenario=("Pickup" if scenario.startswith("Pickup") else
                                  "Slack-off" if scenario.startswith("Slack") else
                                  "Rotate on-bottom" if scenario.endswith("on-bottom") else
                                  "Rotate off-bottom")
                    )
                    fig_T.add_trace(go.Scatter(x=md, y=out["T_lbf_along"], mode="lines",
                                               name=f"μ={mu:.2f}", line=dict(width=1, dash="dot")))
                    fig_M.add_trace(go.Scatter(x=md, y=out["M_lbf_ft_along"], mode="lines",
                                               name=f"μ={mu:.2f}", line=dict(width=1, dash="dot")))

            A,B = st.columns(2)
            A.plotly_chart(fig_T, use_container_width=True)
            B.plotly_chart(fig_M, use_container_width=True)

            rows = base_out["trace_rows"]
            if rows:
                st.subheader("Iteration trace (bit → surface)")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
                st.download_button("Download iteration trace (CSV)",
                    data=buf.getvalue().encode("utf-8"),
                    file_name="td_iteration_trace.csv",
                    mime="text/csv")

st.markdown("---")
st.caption(
    "Trajectory: Minimum Curvature with ratio factor; VS = projection on a vertical plane of reference azimuth. "
    "T&D: Johancsik soft-string with buoyancy and cased/open μ. Δs = 1 ft throughout."
)
