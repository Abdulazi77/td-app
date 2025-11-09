# streamlit_app.py — Full version (ASCII-only)
# Tabs:
#   1) Trajectory (3D) — Minimum Curvature Method (MCM)
#   2) Torque & Drag (soft-string) — Johancsik stepwise model
#
# Designed for Streamlit Community Cloud, Python 3.12. Keep requirements lean.

import math
from typing import Tuple, List, Dict
import io, csv
import streamlit as st

# -------- Page config --------
st.set_page_config(page_title="Wellpath + Torque & Drag — PEGN517", layout="wide")

# =========================
#  Minimum Curvature (MCM)
# =========================
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def _rf(dogs_rad: float) -> float:
    """Ratio Factor for Minimum Curvature: RF = 2/dogs * tan(dogs/2)."""
    if abs(dogs_rad) < 1e-12:
        return 1.0
    return 2.0 / dogs_rad * math.tan(0.5 * dogs_rad)

def _mcm_step(md1, inc1_deg, az1_deg, md2, inc2_deg, az2_deg,
              n1, e1, tvd1) -> Tuple[float, float, float, float]:
    """One MCM step. Returns (N2, E2, TVD2, DLS_deg_per_100ft)."""
    ds = md2 - md1
    inc1 = inc1_deg * DEG2RAD; inc2 = inc2_deg * DEG2RAD
    az1  = az1_deg  * DEG2RAD; az2  = az2_deg  * DEG2RAD

    cos_dog = (math.cos(inc1)*math.cos(inc2) +
               math.sin(inc1)*math.sin(inc2)*math.cos(az2-az1))
    cos_dog = max(-1.0, min(1.0, cos_dog))
    dogs = math.acos(cos_dog)  # dogleg angle (rad)
    rf = _rf(dogs)

    dN = 0.5 * ds * (math.sin(inc1)*math.cos(az1) + math.sin(inc2)*math.cos(az2)) * rf
    dE = 0.5 * ds * (math.sin(inc1)*math.sin(az1) + math.sin(inc2)*math.sin(az2)) * rf
    dT = 0.5 * ds * (math.cos(inc1) + math.cos(inc2)) * rf

    n2, e2, tvd2 = n1 + dN, e1 + dE, tvd1 + dT
    dls = 0.0 if ds <= 0 else dogs * RAD2DEG / ds * 100.0
    return n2, e2, tvd2, dls

def mcm_positions(md: List[float], inc: List[float], az: List[float]):
    """Apply MCM to MD/Inc/Az to get TVD/N/E and DLS arrays."""
    north, east, tvd, dls = [0.0], [0.0], [0.0], [0.0]
    for i in range(1, len(md)):
        n2, e2, t2, d = _mcm_step(md[i-1], inc[i-1], az[i-1], md[i], inc[i], az[i],
                                  north[-1], east[-1], tvd[-1])
        north.append(n2); east.append(e2); tvd.append(t2); dls.append(d)
    return north, east, tvd, dls

# -------------------------
# Synthetic well profiles
# -------------------------
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
        if len(md) > 50000: break
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
        if len(md) > 50000: break
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
        if len(md) > 50000: break
    return md, inc, az

# ================================
# Torque & Drag — soft-string T&D
# ================================
def buoyancy_factor_ppg(mw_ppg: float) -> float:
    # BF = (65.5 - MW)/65.5
    return max(0.0, min(1.0, (65.5 - float(mw_ppg)) / 65.5))

def kappa_from_dls(dls_deg_per_100ft: float) -> float:
    # Curvature κ [rad/ft] from DLS [deg/100 ft]
    return (dls_deg_per_100ft * DEG2RAD) / 100.0

def solve_torque_drag(md: List[float], inc_deg: List[float], dls_deg_100: List[float],
                      # drillstring (bit up): lengths in ft, dimensions in in, weights lb/ft
                      dc_len, dc_od_in, dc_id_in, dc_w_air,
                      hwdp_len, hwdp_od_in, hwdp_id_in, hwdp_w_air,
                      dp_len, dp_od_in, dp_id_in, dp_w_air,
                      # well & operation
                      shoe_md_ft, mu_cased_slide, mu_open_slide,
                      mu_cased_rot=None, mu_open_rot=None,
                      mw_ppg=10.0, wob_lbf=0.0, bit_torque_lbf_ft=0.0,
                      scenario="Pickup"):
    """
    Johancsik soft-string stepwise accumulation from bit -> surface.
    Axial update per segment i:
        T_{i+1} = T_i + (sigma_ax * w_b * cosθ + μ * (w_b * sinθ + T_i * κ)) * Δs
        where sigma_ax = +1 (Pickup/POOH), -1 (Slack-off/RIH).
    Tension is positive in tension; on-bottom boundary uses T_bit = -WOB.
    Torque update:
        M_{i+1} = M_i + μ_rot * (w_b * sinθ + T_i * κ) * r_eff * Δs
        r_eff ≈ OD/2 (ft). Boundary: M_bit = M_b (bit torque if on-bottom).
    """
    n = len(md)
    # Inputs per station
    theta = inc_deg
    kappa = [kappa_from_dls(d) for d in dls_deg_100]

    # Piecewise friction (cased vs open hole) and mu_rot defaults
    mu_rot_cased = mu_cased_rot if mu_cased_rot is not None else mu_cased_slide
    mu_rot_open  = mu_open_rot  if mu_open_rot  is not None else mu_open_slide

    # Drillstring stack from bit up
    # Assign each MD station to a component: DC -> HWDP -> DP
    total_len = dc_len + hwdp_len + dp_len
    bit_depth = md[-1]
    # Effective lengths actually in hole (truncate to bit depth)
    use_dc   = max(0.0, min(dc_len, bit_depth))
    use_hwdp = max(0.0, min(hwdp_len, bit_depth - use_dc))
    use_dp   = max(0.0, min(dp_len, bit_depth - use_dc - use_hwdp))

    # Component properties lookup (w_air lb/ft, OD in)
    def comp_at_md(m: float):
        if m > bit_depth: m = bit_depth
        if m > (bit_depth - use_dc):
            return dc_w_air, dc_od_in
        elif m > (bit_depth - use_dc - use_hwdp):
            return hwdp_w_air, hwdp_od_in
        else:
            return dp_w_air, dp_od_in

    BF = buoyancy_factor_ppg(mw_ppg)

    # Boundary conditions
    sigma_ax = +1.0 if scenario.lower().startswith("pick") else -1.0
    T = [-float(wob_lbf) if wob_lbf > 0 and scenario.lower().startswith("rotate on") else 0.0]  # at bit
    M = [float(bit_torque_lbf_ft) if scenario.lower().startswith("rotate on") else 0.0]          # at bit

    # Iteration trace
    rows: List[Dict] = []

    # Integrate from bit (high MD) toward surface (low MD)
    # Our arrays are from 0 -> end, so step backwards.
    for i in range(n-1, 0, -1):
        md2, md1 = md[i], md[i-1]
        ds = md2 - md1
        th = theta[i] * DEG2RAD
        dls = dls_deg_100[i]
        kap = kappa[i]
        w_air, od_in = comp_at_md(md2)
        w_b = w_air * BF  # buoyed weight per ft
        r_eff_ft = (od_in / 2.0) / 12.0

        # Which friction (cased vs open hole)?
        in_open = (md2 > shoe_md_ft)
        mu_slide = (mu_open_slide if in_open else mu_cased_slide)
        mu_rot   = (mu_rot_open  if in_open else mu_rot_cased)

        # Normal force per unit length
        N = w_b * math.sin(th) + T[-1] * kap

        # Axial (drag) update
        dT = (sigma_ax * w_b * math.cos(th) + mu_slide * N) * ds
        T_next = T[-1] + dT

        # Torque update
        dM = (mu_rot * N * r_eff_ft) * ds
        M_next = M[-1] + dM

        rows.append({
            "md_top_ft": md1, "md_bot_ft": md2, "ds_ft": ds,
            "inc_deg": theta[i], "dls_deg_100ft": dls, "kappa_rad_ft": kap,
            "w_air_lbft": w_air, "BF": BF, "w_b_lbft": w_b,
            "mu_slide": mu_slide, "mu_rot": mu_rot,
            "N_lbf": N, "dT_lbf": dT, "T_next_lbf": T_next,
            "r_eff_ft": r_eff_ft, "dM_lbf_ft": dM, "M_next_lbf_ft": M_next
        })

        T.append(T_next); M.append(M_next)

    # The arrays were accumulated bit->surface; reverse for surface->bit alignment
    T = list(reversed(T))
    M = list(reversed(M))
    rows.reverse()

    surf_hookload = T[0]  # add block weight outside if desired
    surf_torque   = M[0] + (bit_torque_lbf_ft if scenario.lower().startswith("rotate on") else 0.0)

    return {
        "T_lbf_along": T,
        "M_lbf_ft_along": M,
        "trace_rows": rows,
        "surface_hookload_lbf": surf_hookload,
        "surface_torque_lbf_ft": surf_torque
    }

# =====================
#        UI
# =====================
st.title("Wellpath + Torque & Drag (soft-string)")

tab1, tab2 = st.tabs(["1) Trajectory (3D)", "2) Torque & Drag"])

# -------------------------------
# Tab 1 — Trajectory (3D MCM)
# -------------------------------
with tab1:
    st.subheader("Synthetic survey & 3D trajectory")

    col1, col2, col3 = st.columns(3)
    with col1:
        profile = st.selectbox("Profile", [
            "Build & Hold",
            "Build & Hold & Drop",
            "Horizontal (Continuous Build + Lateral)"
        ])
        ds_ft = st.selectbox("Course length (ft)", [10, 20, 30, 50, 100], index=2)
        kop_md = st.number_input("KOP MD (ft)", min_value=0.0, value=1000.0, step=50.0)
    with col2:
        az_label = st.selectbox("Quick azimuth", ["North (0)", "East (90)", "South (180)", "West (270)"], index=0)
        az_map = {"North (0)": 0.0, "East (90)": 90.0, "South (180)": 180.0, "West (270)": 270.0}
        az_deg = st.number_input("Azimuth (deg from North, clockwise)",
                                 min_value=0.0, max_value=360.0,
                                 value=az_map[az_label], step=1.0)
        build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
    with col3:
        theta_hold = None; hold_len = None; drop_rate = None
        final_inc = None; lat_len = None; target_md = None
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
            target_md = st.number_input("Target MD (ft, optional; 0=auto)", 0.0, None, 0.0, 100.0)
            if target_md == 0.0: target_md = None

    if st.button("Compute trajectory", key="go_traj"):
        if profile == "Build & Hold":
            md, inc, az = synth_build_hold(kop_md, build_rate, theta_hold, target_md, ds_ft, az_deg)
        elif profile == "Build & Hold & Drop":
            md, inc, az = synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len,
                                                drop_rate, final_inc, target_md, ds_ft, az_deg)
        else:
            md, inc, az = synth_horizontal(kop_md, build_rate, lat_len, target_md, ds_ft, az_deg)

        north, east, tvd, dls = mcm_positions(md, inc, az)

        # 3D plot (lazy import for faster loads)
        import plotly.graph_objects as go
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=east, y=north, z=[-x for x in tvd],
            mode="lines", line=dict(width=6), name="Well path"
        ))
        fig3d.update_layout(
            title="3D Well Trajectory",
            scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        c1, c2 = st.columns([2,1])
        with c1: st.plotly_chart(fig3d, use_container_width=True)

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

        # Table + CSV
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
        st.subheader("Equations (Minimum Curvature)")
        st.latex(r"\cos\Delta\sigma=\cos\theta_1\cos\theta_2+\sin\theta_1\sin\theta_2\cos(\phi_2-\phi_1)")
        st.latex(r"\mathrm{RF}=\frac{2}{\Delta\sigma}\tan\left(\frac{\Delta\sigma}{2}\right)")
        st.latex(r"\Delta N=\frac{\Delta s}{2}\left(\sin\theta_1\cos\phi_1+\sin\theta_2\cos\phi_2\right)\mathrm{RF}")
        st.latex(r"\Delta E=\frac{\Delta s}{2}\left(\sin\theta_1\sin\phi_1+\sin\theta_2\sin\phi_2\right)\mathrm{RF}")
        st.latex(r"\Delta \mathrm{TVD}=\frac{\Delta s}{2}\left(\cos\theta_1+\cos\theta_2\right)\mathrm{RF}")
        st.caption("Standard Minimum Curvature Method (MCM). See references in this page.")

# -------------------------------------
# Tab 2 — Torque & Drag (soft-string)
# -------------------------------------
with tab2:
    st.subheader("Soft-string T&D (Johancsik) — cased vs open-hole")

    colA, colB = st.columns(2)
    with colA:
        shoe_md = st.number_input("Casing shoe MD (ft)", min_value=0.0, value=4000.0, step=50.0)
        mw_ppg = st.number_input("Mud weight (ppg)", min_value=6.0, max_value=20.0, value=10.0, step=0.1)
        mu_cased_slide = st.number_input("Friction factor in casing (sliding)", 0.05, 0.6, 0.25, 0.01)
        mu_open_slide  = st.number_input("Friction factor in open hole (sliding)", 0.05, 0.6, 0.35, 0.01)
        mu_cased_rot   = st.number_input("Friction factor in casing (rotating, optional)", 0.0, 1.0, 0.25, 0.01)
        mu_open_rot    = st.number_input("Friction factor in open hole (rotating, optional)", 0.0, 1.0, 0.35, 0.01)
    with colB:
        wob_lbf = st.number_input("WOB (lbf) for on-bottom case", min_value=0.0, value=0.0, step=1000.0)
        bit_torque = st.number_input("Bit torque (lbf-ft) for on-bottom case", min_value=0.0, value=0.0, step=100.0)
        scenario = st.selectbox("Scenario",
                                ["Pickup (POOH, sliding)", "Slack-off (RIH, sliding)",
                                 "Rotate off-bottom", "Rotate on-bottom"])

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

    st.markdown("#### Survey source")
    survey_src = st.radio("Use the trajectory from Tab 1 (recommended) or create a quick profile here?",
                          ["Use Tab 1 (last run)", "Quick synthetic here"], horizontal=True)

    # To share state across tabs, cache in session_state the last trajectory
    if "last_md" not in st.session_state:
        st.session_state["last_md"] = None

    if survey_src == "Quick synthetic here":
        colQ1, colQ2, colQ3 = st.columns(3)
        with colQ1:
            profile2 = st.selectbox("Profile (T&D tab)", [
                "Build & Hold",
                "Build & Hold & Drop",
                "Horizontal (Continuous Build + Lateral)"
            ], key="td_prof")
            ds2 = st.selectbox("Course length (ft)", [10,20,30,50,100], index=2, key="td_ds")
            kop2 = st.number_input("KOP MD (ft)", 0.0, None, 1000.0, 50.0, key="td_kop")
        with colQ2:
            az2_label = st.selectbox("Quick azimuth", ["North (0)","East (90)","South (180)","West (270)"], index=0, key="td_azlab")
            az2_map = {"North (0)":0.0,"East (90)":90.0,"South (180)":180.0,"West (270)":270.0}
            az2 = st.number_input("Azimuth (deg)", 0.0, 360.0, az2_map[az2_label], 1.0, key="td_az")
            br2 = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4, key="td_br")
        with colQ3:
            th2=hl2=dr2=fi2=ll2=tm2=None
            if profile2=="Build & Hold":
                th2 = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 1.0, key="td_th")
                tm2 = st.number_input("Target MD (ft)", min_value=kop2, value=10000.0, step=100.0, key="td_tm")
            elif profile2=="Build & Hold & Drop":
                th2 = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 1.0, key="td_ths")
                hl2 = st.number_input("Hold length (ft)", 0.0, None, 1000.0, 100.0, key="td_hl")
                dr2 = st.selectbox("Drop rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4, key="td_dr")
                fi2 = st.number_input("Final inclination after drop (deg)", 0.0, 90.0, 0.0, 1.0, key="td_fi")
                tm2 = st.number_input("Target MD (ft)", min_value=kop2, value=12000.0, step=100.0, key="td_tmd")
            else:
                ll2 = st.number_input("Lateral length (ft)", 0.0, None, 2000.0, 100.0, key="td_ll")
                tm2 = st.number_input("Target MD (ft, 0=auto)", 0.0, None, 0.0, 100.0, key="td_tm0")
                if tm2 == 0.0: tm2 = None

        go2 = st.button("Build survey (for T&D)")
        if go2:
            if profile2=="Build & Hold":
                md, inc, az = synth_build_hold(kop2, br2, th2, tm2, ds2, az2)
            elif profile2=="Build & Hold & Drop":
                md, inc, az = synth_build_hold_drop(kop2, br2, th2, hl2, dr2, fi2, tm2, ds2, az2)
            else:
                md, inc, az = synth_horizontal(kop2, br2, ll2, tm2, ds2, az2)
            north, east, tvd, dls = mcm_positions(md, inc, az)
            st.session_state["last_md"] = (md, inc, az, dls)
    else:
        # If user already ran Tab 1 in this session
        if "last_md_traj" in st.session_state:
            md, inc, az, dls = st.session_state["last_md_traj"]
            st.write("Using trajectory from Tab 1.")
            st.session_state["last_md"] = (md, inc, az, dls)
        else:
            st.info("Run Tab 1 first to generate a trajectory, or switch to 'Quick synthetic here'.")

    # Allow Tab 1 to save survey automatically
def _store_traj_on_switch(md, inc, az, dls):
    st.session_state["last_md_traj"] = (md, inc, az, dls)

# Attach hook: if tab1 computed, save it
# (We can’t detect from here; but Tab 1 users can refresh to re-use.)

    st.markdown("---")
    if st.button("Run Torque & Drag"):
        if st.session_state.get("last_md") is None:
            st.error("No survey available. Build one above or run Tab 1 first.")
        else:
            md, inc, az, dls = st.session_state["last_md"]

            # Solve soft-string T&D
            out = solve_torque_drag(
                md, inc, dls,
                dc_len, dc_od, dc_id, dc_w,
                hwdp_len, hwdp_od, hwdp_id, hwdp_w,
                dp_len, dp_od, dp_id, dp_w,
                shoe_md, mu_cased_slide, mu_open_slide,
                mu_cased_rot, mu_open_rot,
                mw_ppg=mw_ppg, wob_lbf=wob_lbf, bit_torque_lbf_ft=bit_torque,
                scenario=("Pickup" if scenario.startswith("Pickup") else
                          "Slack-off" if scenario.startswith("Slack") else
                          "Rotate on-bottom" if scenario.endswith("on-bottom") else
                          "Rotate off-bottom")
            )

            T = out["T_lbf_along"]; M = out["M_lbf_ft_along"]
            surf_hook = out["surface_hookload_lbf"]
            surf_tq   = out["surface_torque_lbf_ft"]

            st.success(f"Surface hookload: {surf_hook:,.0f} lbf — Surface torque: {surf_tq:,.0f} lbf-ft")

            # Plots (lazy import)
            import plotly.graph_objects as go
            fig_hl = go.Figure()
            fig_hl.add_trace(go.Scatter(x=md, y=T, mode="lines", name="Hookload/Tension"))
            fig_hl.update_layout(title="Tension / Hookload vs MD", xaxis_title="MD (ft)", yaxis_title="Tension (lbf)")
            fig_tq = go.Figure()
            fig_tq.add_trace(go.Scatter(x=md, y=M, mode="lines", name="Torque"))
            fig_tq.update_layout(title="Surface torque build-up vs MD", xaxis_title="MD (ft)", yaxis_title="Torque (lbf-ft)")
            cA, cB = st.columns(2)
            with cA: st.plotly_chart(fig_hl, use_container_width=True)
            with cB: st.plotly_chart(fig_tq, use_container_width=True)

            # Iteration trace table
            rows = out["trace_rows"]
            if rows:
                st.subheader("Iteration trace (bit → surface)")
                st.dataframe(rows, use_container_width=True, height=380)

                # CSV
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
                st.download_button("Download iteration trace (CSV)",
                                   data=buf.getvalue().encode("utf-8"),
                                   file_name="td_iteration_trace.csv",
                                   mime="text/csv")

            # Equations recap for report
            st.markdown("---")
            st.subheader("Equations used (soft-string)")
            st.latex(r"N = w_b \sin\theta + T\,\kappa")
            st.latex(r"T_{i+1}=T_i + \big(\sigma_{\mathrm{ax}}\,w_b\cos\theta + \mu\,N\big)\,\Delta s")
            st.latex(r"M_{i+1}=M_i + \mu_{\mathrm{rot}}\,N\,r_{\mathrm{eff}}\,\Delta s")
            st.markdown(
                "- $w_b = w_{air}\\times \\text{BF}$, with $\\text{BF}=(65.5-\\text{MW})/65.5$.\n"
                "- $\\kappa$ from DLS: $\\kappa = (\\mathrm{DLS}\\,[\\deg/100\\,\\mathrm{ft}])\\,\\pi/(180\\times100)$.\n"
                "- $\\sigma_{ax}=+1$ pickup (POOH), $-1$ slack-off (RIH). $M_{bit}=0$ off-bottom; on-bottom uses WOB and bit torque."
            )

# Footer references for your report (citations shown on the app page)
st.markdown("---")
st.caption(
    "Minimum Curvature (MCM) equations and ratio factor are standard in directional survey calculations; "
    "soft-string T&D follows Johancsik’s stepwise method; API RP 7G provides combined tension–torque envelope guidance."
)
