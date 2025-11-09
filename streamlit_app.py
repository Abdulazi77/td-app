# streamlit_app.py — PEGN517 Wellpath + T&D
# - Minimum Curvature (Δs = 1 ft)
# - Casing program (OD -> lb/ft -> ID, Grade, Shoe MD)
# - 3D "schematic": wellpath colored by casing intervals, thicker for larger (shallower) strings
# - Soft-string Torque/Drag (Johancsik) with cased vs open-hole friction split by deepest shoe
#
# Python 3.12, Streamlit Cloud friendly (minimal deps).
# References (see app footer):
#   - Minimum Curvature & RF (ratio factor), DLS definitions
#   - Johancsik soft-string torque/drag
#   - Buoyancy factor BF = (65.5 - MW)/65.5
#   - API RP 7G (envelope guidance)
#   - Streamlit st.rerun()

import math
from typing import Tuple, List, Dict, Optional
import io, csv
import streamlit as st

st.set_page_config(page_title="Wellpath + Torque & Drag — PEGN517", layout="wide")

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ------------------------------
# Small, standard casing library
# ------------------------------
# OD label -> [{"ppf": .., "id": ..}, ...] ; Grade list (typical availability)
CASING_DB: Dict[str, Dict[str, List[Dict[str, float]]]] = {
    "13-3/8": {
        "grades": ["J55", "K55", "N80", "L80", "P110"],
        "options": [
            {"ppf": 48.0,  "id": 12.415},
            {"ppf": 54.5,  "id": 12.347},
            {"ppf": 61.0,  "id": 12.115},
            {"ppf": 68.0,  "id": 11.965},
        ],
    },
    "9-5/8": {
        "grades": ["J55", "K55", "N80", "L80", "P110"],
        "options": [
            {"ppf": 36.0,  "id": 8.835},
            {"ppf": 40.0,  "id": 8.535},
            {"ppf": 43.5,  "id": 8.405},
            {"ppf": 47.0,  "id": 8.311},
            {"ppf": 53.5,  "id": 8.097},
        ],
    },
    "7-5/8": {
        "grades": ["J55", "K55", "N80", "L80", "P110"],
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
        "grades": ["J55", "K55", "N80", "L80", "P110"],
        "options": [
            {"ppf": 23.0,  "id": 6.366},
            {"ppf": 26.0,  "id": 6.276},  # common
            {"ppf": 29.0,  "id": 6.184},
            {"ppf": 32.0,  "id": 6.094},
            {"ppf": 35.0,  "id": 6.004},
            {"ppf": 38.0,  "id": 5.938},
            {"ppf": 41.0,  "id": 5.921},
        ],
    },
    "5-1/2": {
        "grades": ["J55", "K55", "N80", "L80", "P110"],
        "options": [
            {"ppf": 14.0,  "id": 4.892},
            {"ppf": 17.0,  "id": 4.778},
            {"ppf": 20.0,  "id": 4.670},
            {"ppf": 23.0,  "id": 4.563},
            {"ppf": 26.0,  "id": 4.494},
        ],
    },
    "4-1/2": {
        "grades": ["J55", "K55", "N80", "L80", "P110"],
        "options": [
            {"ppf": 9.5,   "id": 4.090},
            {"ppf": 10.5,  "id": 4.052},
            {"ppf": 11.6,  "id": 4.000},
            {"ppf": 13.5,  "id": 3.920},
        ],
    },
}

# --------------------------
# Utility: parse OD label
# --------------------------
def parse_od_inch(label: str) -> float:
    # "9-5/8" -> 9.625, "7" -> 7.0
    if "-" in label:
        whole, frac = label.split("-")
        num, den = frac.split("/")
        return float(whole) + float(num) / float(den)
    return float(label)

# =========================================================
# Minimum Curvature Method (Δs = 1 ft piecewise generation)
# =========================================================
def _rf(dogs_rad: float) -> float:
    # Ratio Factor RF = 2/dogs * tan(dogs/2)
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

def mcm_positions(md: List[float], inc: List[float], az: List[float]):
    north, east, tvd, dls = [0.0], [0.0], [0.0], [0.0]
    for i in range(1, len(md)):
        n2, e2, t2, d = _mcm_step(md[i-1], inc[i-1], az[i-1], md[i], inc[i], az[i],
                                  north[-1], east[-1], tvd[-1])
        north.append(n2); east.append(e2); tvd.append(t2); dls.append(d)
    return north, east, tvd, dls

def _gen_md(target_md: float, ds: float = 1.0) -> List[float]:
    steps = int(max(0.0, target_md) // ds)
    md = [i * ds for i in range(steps + 1)]
    if md[-1] < target_md:
        md.append(target_md)
    return md

# Synthetic well builders (Δs fixed to 1 ft)
def synth_build_hold(kop_md, build_rate, theta_hold, target_md, az_deg):
    ds = 1.0
    md = _gen_md(target_md, ds)
    inc = []
    for m in md:
        if m < kop_md:
            inc.append(0.0)
        else:
            prev = inc[-1] if inc else 0.0
            inc.append(min(prev + build_rate * (ds/100.0), theta_hold))
    az = [az_deg] * len(md)
    return md, inc, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len, drop_rate,
                          final_inc, target_md, az_deg):
    ds = 1.0
    md = _gen_md(target_md, ds)
    inc = []
    built = False; in_hold = False; in_drop = False; hold_left = hold_len or 0.0
    for m in md:
        if m < kop_md:
            inc.append(0.0); continue
        prev = inc[-1] if inc else 0.0
        if not built:
            nxt = min(prev + build_rate * (ds/100.0), theta_hold)
            built = abs(nxt - theta_hold) < 1e-12
            if built and hold_left > 0: in_hold = True
            inc.append(nxt)
        elif in_hold:
            inc.append(theta_hold)
            hold_left -= ds
            if hold_left <= 0: in_hold, in_drop = False, True
        elif in_drop:
            nxt = max(prev - drop_rate * (ds/100.0), final_inc)
            if abs(nxt - final_inc) < 1e-12: in_drop = False
            inc.append(nxt)
        else:
            inc.append(final_inc)
    az = [az_deg] * len(md)
    return md, inc, az

def synth_horizontal(kop_md, build_rate, lateral_len, target_md, az_deg):
    ds = 1.0
    if target_md is None:
        build_len = math.ceil(90.0 / (build_rate / 100.0))
        target_md = kop_md + build_len + (lateral_len or 0.0)
    md = _gen_md(target_md, ds)
    inc, built90 = [], False
    for m in md:
        if m < kop_md:
            inc.append(0.0); continue
        prev = inc[-1] if inc else 0.0
        if not built90:
            nxt = min(prev + build_rate * (ds/100.0), 90.0)
            built90 = abs(nxt - 90.0) < 1e-12
            inc.append(nxt)
        else:
            inc.append(90.0)
    az = [az_deg] * len(md)
    return md, inc, az

# ==============================================
# Soft-string Torque & Drag (Johancsik stepwise)
# ==============================================
def buoyancy_factor_ppg(mw_ppg: float) -> float:
    # BF = (65.5 - MW) / 65.5
    return max(0.0, min(1.0, (65.5 - float(mw_ppg)) / 65.5))

def kappa_from_dls(dls_deg_per_100ft: float) -> float:
    # κ [rad/ft] from DLS [deg/100 ft]
    return (dls_deg_per_100ft * DEG2RAD) / 100.0

def solve_torque_drag(md: List[float], inc_deg: List[float], dls_deg_100: List[float],
                      # drillstring (bit up): lengths in ft, dimensions in in, weights lb/ft
                      dc_len, dc_od_in, dc_id_in, dc_w_air,
                      hwdp_len, hwdp_od_in, hwdp_id_in, hwdp_w_air,
                      dp_len, dp_od_in, dp_id_in, dp_w_air,
                      # well & operation
                      casing_shoes_md: List[float],  # multiple casings
                      mu_cased_slide, mu_open_slide,
                      mu_cased_rot=None, mu_open_rot=None,
                      mw_ppg=10.0, wob_lbf=0.0, bit_torque_lbf_ft=0.0,
                      scenario="Pickup"):
    n = len(md)
    theta = inc_deg
    kappa = [kappa_from_dls(d) for d in dls_deg_100]
    mu_rot_cased = mu_cased_rot if mu_cased_rot is not None else mu_cased_slide
    mu_rot_open  = mu_open_rot  if mu_open_rot  is not None else mu_open_slide

    bit_depth = md[-1]
    use_dc   = max(0.0, min(dc_len, bit_depth))
    use_hwdp = max(0.0, min(hwdp_len, bit_depth - use_dc))
    use_dp   = max(0.0, min(dp_len, bit_depth - use_dc - use_hwdp))

    def comp_at_md(m: float):
        # Return (w_air, od_in). From bit upward: DC -> HWDP -> DP
        if m > (bit_depth - use_dc):              return dc_w_air, dc_od_in
        elif m > (bit_depth - use_dc - use_hwdp): return hwdp_w_air, hwdp_od_in
        else:                                     return dp_w_air, dp_od_in

    BF = buoyancy_factor_ppg(mw_ppg)
    sigma_ax = +1.0 if scenario.lower().startswith("pick") else -1.0
    T = [-float(wob_lbf) if wob_lbf > 0 and scenario.lower().startswith("rotate on") else 0.0]  # at bit
    M = [float(bit_torque_lbf_ft) if scenario.lower().startswith("rotate on") else 0.0]

    rows: List[Dict] = []
    deepest_shoe = max(casing_shoes_md) if casing_shoes_md else 0.0

    for i in range(n-1, 0, -1):
        md2, md1 = md[i], md[i-1]
        ds = md2 - md1
        th = theta[i] * DEG2RAD
        kap = kappa[i]
        w_air, od_in = comp_at_md(md2)
        w_b = w_air * BF
        r_eff_ft = (od_in / 2.0) / 12.0

        in_open = (md2 > deepest_shoe)
        mu_slide = (mu_open_slide if in_open else mu_cased_slide)
        mu_rot   = (mu_rot_open  if in_open else mu_rot_cased)

        # Normal force per length: N = w_b sinθ + T κ   (Johancsik)
        N = w_b * math.sin(th) + T[-1] * kap

        # Axial update: T_{i+1} = T_i + (σ_ax w_b cosθ + μ N) Δs
        dT = (sigma_ax * w_b * math.cos(th) + mu_slide * N) * ds
        T_next = T[-1] + dT

        # Torque update: M_{i+1} = M_i + μ_rot N r_eff Δs
        dM = (mu_rot * N * r_eff_ft) * ds
        M_next = M[-1] + dM

        rows.append({
            "md_top_ft": md1, "md_bot_ft": md2, "ds_ft": ds,
            "inc_deg": theta[i], "kappa_rad_ft": kap,
            "w_air_lbft": w_air, "BF": BF, "w_b_lbft": w_b,
            "mu_slide": mu_slide, "mu_rot": mu_rot,
            "N_lbf": N, "dT_lbf": dT, "T_next_lbf": T_next,
            "r_eff_ft": r_eff_ft, "dM_lbf_ft": dM, "M_next_lbf_ft": M_next,
            "in_open": in_open
        })

        T.append(T_next); M.append(M_next)

    T = list(reversed(T)); M = list(reversed(M)); rows.reverse()
    surf_hookload = T[0]
    surf_torque   = M[0] + (bit_torque_lbf_ft if scenario.lower().startswith("rotate on") else 0.0)
    return {"T_lbf_along": T, "M_lbf_ft_along": M, "trace_rows": rows,
            "surface_hookload_lbf": surf_hookload, "surface_torque_lbf_ft": surf_torque}

# ------------
# Plot helpers
# ------------
def decimate(xs: List[float], factor: int) -> List[float]:
    k = max(1, int(factor))
    return xs[::k]

def segment_indices(md: List[float], top_md: float, bot_md: float) -> List[int]:
    idx = []
    for i, m in enumerate(md):
        if m >= top_md - 1e-9 and m <= bot_md + 1e-9:
            idx.append(i)
    return idx

# =====================  UI  =====================
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

tab1, tab2 = st.tabs(["1) Trajectory & 3D Casing Schematic", "2) Torque & Drag"])

# -------------------- Tab 1 --------------------
with tab1:
    st.subheader("Synthetic survey (Minimum Curvature)")

    c1, c2, c3 = st.columns(3)
    with c1:
        profile = st.selectbox("Profile", [
            "Build & Hold",
            "Build & Hold & Drop",
            "Horizontal (Continuous Build + Lateral)"
        ])
        kop_md = st.number_input("KOP MD (ft)", min_value=0.0, value=1000.0, step=50.0)
    with c2:
        az_label = st.selectbox("Quick azimuth", ["North (0)","East (90)","South (180)","West (270)"], index=0)
        az_map = {"North (0)":0.0,"East (90)":90.0,"South (180)":180.0,"West (270)":270.0}
        az_deg = st.number_input("Azimuth (deg from North, clockwise)",
                                 min_value=0.0, max_value=360.0, value=az_map[az_label], step=1.0)
        build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
    with c3:
        theta_hold = hold_len = drop_rate = final_inc = lat_len = target_md = None
        if profile == "Build & Hold":
            theta_hold = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 1.0)
            target_md = st.number_input("Target MD (ft)", min_value=kop_md, value=10000.0, step=100.0)
        elif profile == "Build & Hold & Drop":
            theta_hold = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 1.0)
            hold_len   = st.number_input("Hold length (ft)", 0.0, None, 1000.0, 100.0)
            drop_rate  = st.selectbox("Drop rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
            final_inc  = st.number_input("Final inclination after drop (deg)", 0.0, 90.0, 0.0, 1.0)
            target_md  = st.number_input("Target MD (ft)", min_value=kop_md, value=12000.0, step=100.0)
        else:
            lat_len   = st.number_input("Lateral length (ft)", 0.0, None, 2000.0, 100.0)
            target_md = st.number_input("Target MD (ft, optional; 0 = auto)", 0.0, None, 0.0, 100.0)
            if target_md == 0.0: target_md = None

    # ----------- Casing program (linked dropdowns) -----------
    st.markdown("### Casing program (optional)")
    st.caption("Pick nominal OD → choose a weight (lb/ft) → ID fills automatically. Select grade and shoe MD (ft).")
    if "casing_rows" not in st.session_state:
        st.session_state["casing_rows"] = 1  # start with one string

    cols = st.columns([1.2, 1.2, 1.0, 1.0, 1.2, 1.2])
    cols[0].markdown("**Nominal OD**"); cols[1].markdown("**lb/ft**"); cols[2].markdown("**ID (in)**")
    cols[3].markdown("**Grade**"); cols[4].markdown("**Setting depth (MD ft)**"); cols[5].markdown("**Actions**")

    casing_rows: List[Dict] = []
    for i in range(st.session_state["casing_rows"]):
        c0,c1,c2,c3,c4,c5 = st.columns([1.2,1.2,1.0,1.0,1.2,1.2])
        with c0: od = st.selectbox("", list(CASING_DB.keys()), key=f"od_{i}")
        with c1:
            opts = CASING_DB[od]["options"]
            weights = [f"{o['ppf']:.2f}".rstrip('0').rstrip('.') for o in opts]
            ppf = st.selectbox("", weights, key=f"ppf_{i}")
        with c2:
            sel = next(x for x in CASING_DB[od]["options"] if f"{x['ppf']:.2f}".rstrip('0').rstrip('.') == ppf)
            st.text_input("", f"{sel['id']:.3f}", key=f"id_{i}", disabled=True)
        with c3:
            grade = st.selectbox("", CASING_DB[od]["grades"], key=f"grade_{i}")
        with c4:
            depth = st.number_input("", min_value=0.0, value=3000.0, step=50.0, key=f"depth_{i}")
        with c5:
            if st.button("➖ Remove", key=f"rem_{i}"):
                st.session_state["casing_rows"] = max(0, st.session_state["casing_rows"]-1)
                st.rerun()
        casing_rows.append({"od": od, "ppf": float(ppf), "id": sel["id"], "grade": grade, "shoe_md": depth})

    add_cols = st.columns([6,1])
    with add_cols[1]:
        if st.button("➕ Add string"):
            st.session_state["casing_rows"] = min(6, st.session_state["casing_rows"]+1)
            st.rerun()

    st.caption("Using Δs = 1 ft for all piecewise MCM calculations.")

    go_traj = st.button("Compute trajectory & 3D schematic")

    if go_traj:
        # Build the survey
        if profile == "Build & Hold":
            md, inc, az = synth_build_hold(kop_md, build_rate, theta_hold, target_md, az_deg)
        elif profile == "Build & Hold & Drop":
            md, inc, az = synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len, drop_rate, final_inc, target_md, az_deg)
        else:
            md, inc, az = synth_horizontal(kop_md, build_rate, lat_len, target_md, az_deg)

        north, east, tvd, dls = mcm_positions(md, inc, az)
        st.session_state["last_traj"] = (md, inc, az, dls, tvd, north, east)
        st.session_state["casing_program"] = casing_rows

        # 3D wellpath split by casing intervals (legend, thickness by OD; open hole brown)
        import plotly.graph_objects as go

        # Sort casing by shoe depth (ascending = shallower first)
        cas_sorted = sorted([r for r in casing_rows if r["shoe_md"] > 0], key=lambda x: x["shoe_md"])
        colors = ["#4C78A8", "#F58518", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6"]  # casing palette
        openhole_color = "#8B4513"  # brown

        fig3d = go.Figure()
        start_md = 0.0
        for idx, row in enumerate(cas_sorted):
            end_md = row["shoe_md"]
            inds = [k for k,m in enumerate(md) if m >= start_md-1e-9 and m <= end_md+1e-9]
            if len(inds) >= 2:
                od_in = parse_od_inch(row["od"])
                width_px = max(2, int(2 + od_in/3.0))  # thicker for larger OD
                fig3d.add_trace(go.Scatter3d(
                    x=[east[k] for k in inds],
                    y=[north[k] for k in inds],
                    z=[-tvd[k] for k in inds],
                    mode="lines",
                    line=dict(width=width_px, color=colors[idx % len(colors)]),
                    name=f"{row['od']} {row['ppf']}# {row['grade']} to {int(end_md)} ft"
                ))
            start_md = end_md

        # Open hole from deepest shoe to TD
        deepest = cas_sorted[-1]["shoe_md"] if cas_sorted else 0.0
        inds_oh = [k for k,m in enumerate(md) if m >= deepest-1e-9]
        if len(inds_oh) >= 2:
            fig3d.add_trace(go.Scatter3d(
                x=[east[k] for k in inds_oh],
                y=[north[k] for k in inds_oh],
                z=[-tvd[k] for k in inds_oh],
                mode="lines",
                line=dict(width=4, color=openhole_color),
                name=f"Open hole (below {int(deepest)} ft)"
            ))

        fig3d.update_layout(
            title="3D Trajectory — Casing Schematic by Interval",
            scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0,r=0,t=40,b=0)
        )

        # 2D quick views (undecimated for accuracy; plots stay light with WebGL)
        prof = go.Figure()
        prof.add_trace(go.Scatter(x=md, y=tvd, mode="lines", name="TVD vs MD"))
        prof.update_yaxes(autorange="reversed")
        prof.update_layout(title="Profile", xaxis_title="MD (ft)", yaxis_title="TVD (ft)")

        plan = go.Figure()
        plan.add_trace(go.Scatter(x=east, y=north, mode="lines", name="Plan"))
        plan.update_layout(title="Plan", xaxis_title="East (ft)", yaxis_title="North (ft)")

        colA, colB = st.columns([2,1])
        with colA: st.plotly_chart(fig3d, use_container_width=True)
        with colB:
            st.plotly_chart(prof, use_container_width=True)
            st.plotly_chart(plan, use_container_width=True)

        # Survey table + CSV
        rows = [{"MD (ft)": md[i], "Inc (deg)": inc[i], "Az (deg)": az[i],
                 "TVD (ft)": tvd[i], "North (ft)": north[i], "East (ft)": east[i],
                 "DLS (deg/100 ft)": dls[i]} for i in range(len(md))]
        st.subheader("Survey and calculated positions")
        st.dataframe(rows, use_container_width=True, height=420)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
        st.download_button("Download trajectory CSV", data=buf.getvalue().encode("utf-8"),
                           file_name="trajectory.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Minimum Curvature — key equations")
        st.latex(r"\cos\Delta\sigma=\cos\theta_1\cos\theta_2+\sin\theta_1\sin\theta_2\cos(\phi_2-\phi_1)")
        st.latex(r"\mathrm{RF}=\frac{2}{\Delta\sigma}\tan\left(\frac{\Delta\sigma}{2}\right)")
        st.latex(r"\Delta N=\frac{\Delta s}{2}\left(\sin\theta_1\cos\phi_1+\sin\theta_2\cos\phi_2\right)\mathrm{RF}")
        st.latex(r"\Delta E=\frac{\Delta s}{2}\left(\sin\theta_1\sin\phi_1+\sin\theta_2\sin\phi_2\right)\mathrm{RF}")
        st.latex(r"\Delta \mathrm{TVD}=\frac{\Delta s}{2}\left(\cos\theta_1+\cos\theta_2\right)\mathrm{RF}")
        st.caption("DLS is reported in deg/100 ft of course length (Δs).")

# -------------------- Tab 2 --------------------
with tab2:
    st.subheader("Soft-string Torque & Drag — Johancsik")

    colA, colB = st.columns(2)
    with colA:
        mw_ppg = st.number_input("Mud weight (ppg)", min_value=6.0, max_value=20.0, value=10.0, step=0.1)
        mu_cased_slide = st.number_input("Friction factor in casing (sliding)", 0.05, 0.60, 0.25, 0.01)
        mu_open_slide  = st.number_input("Friction factor in open hole (sliding)", 0.05, 0.60, 0.35, 0.01)
        mu_cased_rot   = st.number_input("Friction factor in casing (rotating, optional)", 0.0, 1.0, 0.25, 0.01)
        mu_open_rot    = st.number_input("Friction factor in open hole (rotating, optional)", 0.0, 1.0, 0.35, 0.01)
    with colB:
        wob_lbf = st.number_input("WOB (lbf) for on-bottom case", min_value=0.0, value=0.0, step=1000.0)
        bit_torque = st.number_input("Bit torque (lbf-ft) for on-bottom case", min_value=0.0, value=0.0, step=100.0)
        scenario = st.selectbox("Scenario", [
            "Pickup (POOH, sliding)", "Slack-off (RIH, sliding)",
            "Rotate off-bottom", "Rotate on-bottom"
        ])

    st.markdown("#### Drillstring (bit up)")
    c1, c2, c3 = st.columns(3)
    with c1:
        dc_len = st.number_input("DC length (ft)", 0.0, None, 600.0, 10.0)
        dc_od  = st.number_input("DC OD (in)", 1.0, 12.0, 8.0, 0.5)
        dc_id  = st.number_input("DC ID (in)", 0.0, 10.0, 2.813, 0.001)
        dc_w   = st.number_input("DC weight in air (lb/ft)", 0.0, None, 66.7, 0.1)
    with c2:
        hwdp_len = st.number_input("HWDP length (ft)", 0.0, None, 1000.0, 10.0)
        hwdp_od  = st.number_input("HWDP OD (in)", 1.0, 7.0, 3.5, 0.25)
        hwdp_id  = st.number_input("HWDP ID (in)", 0.0, 6.0, 2.0, 0.01)
        hwdp_w   = st.number_input("HWDP weight in air (lb/ft)", 0.0, None, 16.0, 0.1)
    with c3:
        dp_len = st.number_input("DP length (ft)", 0.0, None, 7000.0, 10.0)
        dp_od  = st.number_input("DP OD (in)", 1.0, 6.0, 5.0, 0.25)
        dp_id  = st.number_input("DP ID (in)", 0.0, 5.5, 4.276, 0.01)
        dp_w   = st.number_input("DP weight in air (lb/ft)", 0.0, None, 19.5, 0.1)

    if st.button("Run Torque & Drag"):
        if "last_traj" not in st.session_state:
            st.error("No trajectory found. Please generate in Tab 1 first.")
        else:
            md, inc, az, dls, tvd, north, east = st.session_state["last_traj"]
            shoes = [row["shoe_md"] for row in st.session_state.get("casing_program", []) if row["shoe_md"] > 0]

            out = solve_torque_drag(
                md, inc, dls,
                dc_len, dc_od, dc_id, dc_w,
                hwdp_len, hwdp_od, hwdp_id, hwdp_w,
                dp_len, dp_od, dp_id, dp_w,
                shoes,
                mu_cased_slide, mu_open_slide,
                mu_cased_rot, mu_open_rot,
                mw_ppg=mw_ppg, wob_lbf=wob_lbf, bit_torque_lbf_ft=bit_torque,
                scenario=("Pickup" if scenario.startswith("Pickup") else
                          "Slack-off" if scenario.startswith("Slack") else
                          "Rotate on-bottom" if scenario.endswith("on-bottom") else
                          "Rotate off-bottom")
            )

            T = out["T_lbf_along"]; M = out["M_lbf_ft_along"]
            surf_hook = out["surface_hookload_lbf"]; surf_tq = out["surface_torque_lbf_ft"]
            st.success(f"Surface hookload: {surf_hook:,.0f} lbf — Surface torque: {surf_tq:,.0f} lbf-ft")

            import plotly.graph_objects as go
            fig_hl = go.Figure()
            fig_hl.add_trace(go.Scatter(x=md, y=T, mode="lines", name="Tension/Hookload"))
            fig_hl.update_layout(title="Tension / Hookload vs MD", xaxis_title="MD (ft)", yaxis_title="Tension (lbf)")
            fig_tq = go.Figure()
            fig_tq.add_trace(go.Scatter(x=md, y=M, mode="lines", name="Torque"))
            fig_tq.update_layout(title="Torque build-up vs MD", xaxis_title="MD (ft)", yaxis_title="Torque (lbf-ft)")
            cA, cB = st.columns(2)
            with cA: st.plotly_chart(fig_hl, use_container_width=True)
            with cB: st.plotly_chart(fig_tq, use_container_width=True)

            rows = out["trace_rows"]
            if rows:
                st.subheader("Iteration trace (bit → surface)")
                st.dataframe(rows, use_container_width=True, height=380)
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
                st.download_button("Download iteration trace (CSV)",
                                   data=buf.getvalue().encode("utf-8"),
                                   file_name="td_iteration_trace.csv",
                                   mime="text/csv")

            st.markdown("---")
            st.subheader("Equations used (soft-string)")
            st.latex(r"N = w_b \sin\theta + T\,\kappa")
            st.latex(r"T_{i+1}=T_i + \big(\sigma_{\mathrm{ax}}\,w_b\cos\theta + \mu\,N\big)\,\Delta s")
            st.latex(r"M_{i+1}=M_i + \mu_{\mathrm{rot}}\,N\,r_{\mathrm{eff}}\,\Delta s")
            st.markdown(
                "- $w_b = w_{air}\\times \\text{BF}$ with $\\text{BF}=(65.5-\\text{MW})/65.5$.\n"
                "- $\\kappa$ from DLS: $\\kappa = (\\mathrm{DLS}\\,[\\deg/100\\,\\mathrm{ft}])\\,\\pi/(180\\times100)$.\n"
                "- $\\sigma_{ax}=+1$ pickup (POOH), $-1$ slack-off (RIH). On-bottom uses WOB and bit torque."
            )

# Footer references (citations appear in the app page)
st.markdown("---")
st.caption(
    "Minimum Curvature equations & RF (ratio factor) per standard directional-survey references; "
    "Johancsik soft-string model assumes sliding friction with normal force from buoyed weight and tension-through-curvature; "
    "Buoyancy factor BF = (65.5 − MW)/65.5; API RP 7G for combined tension–torque guidance; "
    "Streamlit st.rerun() is the supported rerun API."
)
