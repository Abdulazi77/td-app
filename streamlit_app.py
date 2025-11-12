# streamlit_app.py
# -*- coding: utf-8 -*-
"""
PEGN 517 — Wellpath + Torque & Drag (Δs = 1 ft)
Linked workflow with calibration, NP, 0.8×MU, BSR/SR, tortuosity, motor-BT,
rig-limit margins, Menand-style buckling & severity, and enhanced visuals.
"""

from __future__ import annotations
import math
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ───────────────────────────── Page & Styles ─────────────────────────────
st.set_page_config(page_title="Wellpath + Torque & Drag (Δs = 1 ft)", layout="wide")

# Colour / style constants (you can tweak)
COLOUR_CASED     = "#4cc9f0"
COLOUR_OH        = "#a97142"
COLOUR_LIMIT     = "#ff0000"
COLOUR_SAFE_ZONE = "#d0f0d0"
LINE_DASH_LIMIT  = "dash"
LINE_DOT_LIMIT   = "dot"

# ─────────────────────────── Constants ──────────────────────────
DEG2RAD = math.pi / 180.0
IN2FT   = 1.0/12.0

def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def bf_from_mw(mw_ppg: float) -> float: return (65.5 - mw_ppg)/65.5  # steel ~65.5 ppg
def I_in4(od_in: float, id_in: float) -> float: return (math.pi/64.0)*(od_in**4 - id_in**4)
def J_in4(od_in: float, id_in: float) -> float: return (math.pi/32.0)*(od_in**4 - id_in**4)
def Z_in3(od_in: float, id_in: float) -> float: return I_in4(od_in, id_in) / (od_in/2.0)
def A_in2(od_in: float, id_in: float) -> float: return (math.pi/4.0)*(od_in**2 - id_in**2)

# ────────────────────── Minimal standards DBs ───────────────────
CASING_DB = {
    "13-3/8": {"weights": {48.0: 12.415, 54.5: 12.347, 61.0: 12.107}},
    "9-5/8":  {"weights": {29.3: 8.921, 36.0: 8.535, 40.0: 8.321}},
    "7":      {"weights": {20.0: 6.366, 23.0: 6.059, 26.0: 5.920}},
    "5-1/2":  {"weights": {17.0: 4.778, 20.0: 4.670, 23.0: 4.560}},
}

TOOL_JOINT_DB = {
    "NC38": {"od": 4.75, "id": 2.25, "T_makeup_ftlbf": 12000, "F_tensile_lbf": 350000, "T_yield_ftlbf": 20000},
    "NC40": {"od": 5.00, "id": 2.25, "T_makeup_ftlbf": 16000, "F_tensile_lbf": 420000, "T_yield_ftlbf": 25000},
    "NC50": {"od": 6.63, "id": 3.00, "T_makeup_ftlbf": 30000, "F_tensile_lbf": 650000, "T_yield_ftlbf": 45000},
}

# ───────────────────────── Survey builders ──────────────────────
def synth_build_hold(kop_md, build_rate_deg_per_100ft, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * (build_rate_deg_per_100ft / 100.0))
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate / 100.0
    dr = drop_rate / 100.0
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * br)
    start_drop = 0.75 * target_md
    theta = np.maximum(0.0, theta - np.maximum(0.0, md - start_drop) * dr)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate / 100.0
    theta = np.minimum(theta_max, np.maximum(0.0, md - kop_md) * br)
    idx = np.where(theta >= theta_max - 1e-6)[0]
    if len(idx):
        m_h = md[idx[0]]
        md_end = max(target_md, m_h + lateral_length)
        md = np.arange(0.0, md_end + ds, ds)
        theta = np.where(md <= m_h,
                         np.minimum(theta_max, np.maximum(0.0, md - kop_md) * br),
                         theta_max)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def mincurv_positions(md, inc_deg, az_deg):
    ds = np.diff(md)
    n = len(md)
    N = np.zeros(n); E = np.zeros(n); TVD = np.zeros(n); DLS = np.zeros(n)
    for i in range(1, n):
        I1 = inc_deg[i-1]*DEG2RAD; A1 = az_deg[i-1]*DEG2RAD
        I2 = inc_deg[i]*DEG2RAD;   A2 = az_deg[i]*DEG2RAD
        dmd = ds[i-1]
        cos_dl = clamp(math.cos(I1)*math.cos(I2) + math.sin(I1)*math.sin(I2)*math.cos(A2 - A1), -1.0, 1.0)
        dpsi = math.acos(cos_dl)
        RF = 1.0 if dpsi < 1e-12 else (2.0/dpsi)*math.tan(dpsi/2.0)
        dN = 0.5*dmd*(math.sin(I1)*math.cos(A1)+math.sin(I2)*math.cos(A2))*RF
        dE = 0.5*dmd*(math.sin(I1)*math.sin(A1)+math.sin(I2)*math.sin(A2))*RF
        dZ = 0.5*dmd*(math.cos(I1)+math.cos(I2))*RF
        N[i] = N[i-1]+dN; E[i] = E[i-1]+dE; TVD[i] = TVD[i-1]+dZ
        DLS[i] = (dpsi/DEG2RAD)/dmd*100.0 if dmd>0 else 0.0
    return N, E, TVD, DLS

# ───────────────────── Soft-string (Johancsik) ──────────────────
def soft_string_stepper(
    md: Iterable[float],
    inc_deg: Iterable[float],
    kappa_rad_per_ft: Iterable[float],
    cased_mask: Iterable[bool],
    comp_along_depth: Iterable[str],
    comp_props: Dict[str, Dict[str, float]],
    mu_slide_cased: float, mu_slide_open: float,
    mu_rot_cased: float,   mu_rot_open: float,
    mw_ppg: float,
    scenario: str = "slackoff",
    WOB_lbf: float = 0.0, Mbit_ftlbf: float = 0.0,
    tortuosity_mode: str = "off",    # "off" | "kappa" | "mu"
    tau: float = 0.0,                # 0.0 .. 0.5 typ
    mu_open_boost: float = 0.0,      # hole cleaning booster
):
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg)
    nseg = len(md) - 1
    if nseg <= 0: raise ValueError("Trajectory is empty.")
    inc = np.deg2rad(inc_deg[:-1])

    kappa_all = np.asarray(kappa_rad_per_ft)
    kappa_seg = kappa_all[:-1] if len(kappa_all) == len(md) else kappa_all
    if len(kappa_seg) != nseg: kappa_seg = np.resize(kappa_seg, nseg)

    cased_seg = np.asarray(cased_mask)[:nseg]
    comp_arr  = np.asarray(list(comp_along_depth))[:nseg]

    # per-segment properties
    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s = np.zeros(nseg); mu_r = np.zeros(nseg)
    BF = bf_from_mw(mw_ppg)

    for i in range(nseg):
        comp = comp_arr[i]
        od_in = float(comp_props[comp]['od_in']); id_in = float(comp_props[comp]['id_in']); w_air_ft = float(comp_props[comp]['w_air_lbft'])
        w_air[i] = w_air_ft; w_b[i] = w_air_ft*BF
        r_eff_ft[i] = 0.5*od_in*IN2FT
        if cased_seg[i]:
            mu_s[i] = mu_slide_cased; mu_r[i] = mu_rot_cased
        else:
            mu_s[i] = mu_slide_open + mu_open_boost
            mu_r[i] = mu_rot_open   + mu_open_boost

        if not cased_seg[i]:
            if tortuosity_mode == "kappa":
                kappa_seg[i] *= (1.0 + tau)
            elif tortuosity_mode == "mu":
                mu_s[i] *= (1.0 + tau); mu_r[i] *= (1.0 + tau)

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)
    dT = np.zeros(nseg); dM = np.zeros(nseg); N_side = np.zeros(nseg)

    if scenario == "onbottom":
        T[0] = -float(WOB_lbf)
        M[0] = float(Mbit_ftlbf)

    sgn_ax = {"pickup": +1.0, "slackoff": -1.0}.get(scenario, 0.0)

    for i in range(nseg):
        # PHYSICS FIX 1: clamp side force to ≥0
        N_raw = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        N_side[i] = max(0.0, N_raw)

        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*N_side[i])*ds
        M_next = M[i] + (mu_r[i]*N_side[i]*r_eff_ft[i])*ds
        dT[i] = T_next - T[i]
        dM[i] = M_next - M[i]
        T[i+1] = T_next; M[i+1] = M_next

    df = pd.DataFrame({
        "md_top_ft": md[:-1],
        "md_bot_ft": md[1:],
        "ds_ft": 1.0,
        "inc_deg": inc_deg[:-1],
        "kappa_rad_ft": kappa_seg,
        "w_air_lbft": w_air,
        "w_b_lbft": w_b,
        "mu_slide": mu_s,
        "mu_rot": mu_r,
        "N_lbf": N_side,
        "dT_lbf": dT,
        "T_next_lbf": T[1:],
        "dM_lbf_ft": dM,
        "M_next_lbf_ft": M[1:],
        "cased?": cased_seg,
        "comp": comp_arr
    })
    return df, T, M

# ───────────────────── Helper diagnostics ────────────────────────
def neutral_point_md(md: np.ndarray, T_arr: np.ndarray) -> float:
    if len(md) < 2 or len(T_arr) < 2: return float('nan')
    for i in range(len(T_arr)-1):
        t1, t2 = T_arr[i], T_arr[i+1]
        if t1 == 0: return md[i]
        if t1*t2 < 0:
            frac = abs(t1)/(abs(t1)+abs(t2)+1e-9)
            return md[i] + frac*(md[i+1]-md[i])
    return float('nan')

def grid_calibrate_mu(
    md, inc_deg, kappa, cased_mask, comp_along, comp_props, mw_ppg,
    depth_for_fit: float,
    measured_pickup_hl: Optional[float], measured_slackoff_hl: Optional[float],
    measured_rotate_hl: Optional[float], measured_surface_torque: Optional[float],
    mu_ranges: Dict[str, Tuple[float,float,float]],
):
    targets = []
    if measured_pickup_hl is not None:   targets.append("pickup")
    if measured_slackoff_hl is not None:   targets.append("slackoff")
    if measured_rotate_hl is not None:   targets.append("rotate_off")
    if len(targets) == 0: return None

    md = np.asarray(md)
    idx = np.searchsorted(md, depth_for_fit, side="right")
    md_fit = md[:idx+1]; inc_fit = np.asarray(inc_deg)[:idx+1]; kappa_fit = np.asarray(kappa)[:idx+1]
    cased_fit = np.asarray(cased_mask)[:idx]
    comp_fit  = np.asarray(list(comp_along))[:len(md_fit)-1]

    best = None; best_err = 1e99
    mu_c_s_rng = np.arange(*mu_ranges["mu_c_s"])
    mu_o_s_rng = np.arange(*mu_ranges["mu_o_s"])
    mu_c_r_rng = np.arange(*mu_ranges["mu_c_r"])
    mu_o_r_rng = np.arange(*mu_ranges["mu_o_r"])

    for mu_c_s in mu_c_s_rng:
        for mu_o_s in mu_o_s_rng:
            for mu_c_r in mu_c_r_rng:
                for mu_o_r in mu_o_r_rng:
                    err2 = 0.0
                    for scen in targets:
                        df_tmp, T_tmp, M_tmp = soft_string_stepper(
                            md_fit, inc_fit, kappa_fit, cased_fit, comp_fit, comp_props,
                            mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
                            scenario=scen, WOB_lbf=0.0, Mbit_ftlbf=0.0
                        )
                        HL = max(0.0, -T_tmp[-1])
                        if scen == "pickup":   err2 += (HL - measured_pickup_hl)**2
                        if scen == "slackoff": err2 += (HL - measured_slackoff_hl)**2
                        if scen == "rotate_off" and measured_rotate_hl is not None:
                            err2 += (HL - measured_rotate_hl)**2
                        if measured_surface_torque is not None:
                            err2 += (abs(M_tmp[-1]) - measured_surface_torque)**2
                    if err2 < best_err:
                        best_err = err2
                        best = dict(mu_c_s=mu_c_s, mu_o_s=mu_o_s, mu_c_r=mu_c_r, mu_o_r=mu_o_r, SSE=best_err)
    return best

def EI_lbf_ft2_from_in4(Epsi_lbf_in2: float, I_in4_val: float) -> float:
    EI_lbf_in2 = Epsi_lbf_in2 * I_in4_val
    return EI_lbf_in2 / (12.0**2)

# ─────────────────────────────── UI ──────────────────────────────
(tab,) = st.tabs(["Wellpath + Torque & Drag (linked)"])

with tab:
    st.header("Wellpath + Torque & Drag (Δs = 1 ft) — Linked")

    # … (trajectory inputs etc) …
    # [Keep the same UI blocks you used previously up to the main run]

    # (For brevity I’m not repeating every UI input block again here — assume you copy-paste your existing
    # input section as shown earlier, unchanged.)

    # After computing df_itr, T_arr, M_arr:
    depth = df_itr["md_bot_ft"].to_numpy()
    surf_hookload = max(0.0, -T_arr[-1]); surf_torque = abs(M_arr[-1])
    st.success(f"Surface hookload: {surf_hookload:,.0f} lbf — Surface torque: {surf_torque:,.0f} lbf-ft")

    # Safety & limits UI
    # … (unchanged) …

    # Classic/simple toggle
    simple_mode = st.checkbox("Use classic simple view (hide μ-sweep & severity)", value=False)

    # Overlay toggle for calibrated μ curves
    overlay_calibrated = False
    mu_fit = st.session_state.get("μ_fit")
    if mu_fit is not None and not simple_mode:
        overlay_calibrated = st.checkbox("Overlay calibrated μ curves (dashed)", value=True)
        if overlay_calibrated:
            st.caption(
                f"Overlay uses fitted μ: casing(slide)={mu_fit['mu_c_s']:.2f}, open(slide)={mu_fit['mu_o_s']:.2f}, "
                f"casing(rot)={mu_fit['mu_c_r']:.2f}, open(rot)={mu_fit['mu_o_r']:.2f}"
            )

    if simple_mode:
        figT = go.Figure(go.Scatter(
            x=df_itr["md_bot_ft"], y=np.abs(df_itr["M_next_lbf_ft"]),
            mode="lines", name="Torque"
        ))
        figT.update_xaxes(title_text="MD (ft)"); figT.update_yaxes(title_text="Torque (lbf-ft)")
        figH = go.Figure(go.Scatter(
            x=df_itr["md_bot_ft"], y=np.maximum(0.0, -df_itr["T_next_lbf"]),
            mode="lines", name="Hookload"
        ))
        figH.update_xaxes(title_text="MD (ft)"); figH.update_yaxes(title_text="Hookload (lbf)")
        c1x, c2x = st.columns(2)
        with c1x: st.plotly_chart(figT, use_container_width=True, key="simple-torque")
        with c2x: st.plotly_chart(figH, use_container_width=True, key="simple-hook")
    else:
        st.markdown("### T&D Model Chart — μ sweep & buckling severity")

        mu_band = st.multiselect("μ sweep (off-bottom torque vs depth)", [0.15,0.20,0.25,0.30,0.35,0.40], default=[0.20,0.25,0.30,0.35])

        T_makeup_sf = TOOL_JOINT_DB[tj_name]['T_makeup_ftlbf']/sf_joint
        rig_torque_lim = rig_torque_lim  # already from inputs
        rig_pull_lim   = rig_pull_lim    # already from inputs

        def run_td_off_bottom(mu_slide: float, mu_rot: float):
            df_tmp, T_tmp, M_tmp = soft_string_stepper(
                md, inc_deg, kappa, (md<=shoe_md), comp_along, comp_props,
                mu_slide, mu_slide, mu_rot, mu_rot, mw_ppg,
                scenario="rotate_off", tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost
            )
            return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy())

        # PHYSICS BLOCK: severity, buckling etc
        hole_d_profile = np.where(
            df_itr["cased?"].to_numpy(),
            2.0*casing_id_in,
            hole_diam_in + (overgage_rot_in if "rotate" in scenario else 0.0)
        )

        Epsi       = 30.0e6
        od_local   = comp_od_in
        id_local   = comp_id_in
        I_local    = np.array([I_in4(o, i) for o, i in zip(od_local, id_local)])
        Z_local    = np.array([Z_in3(o, i) for o, i in zip(od_local, id_local)])
        A_local    = np.array([A_in2(o, i) for o, i in zip(od_local, id_local)])
        J_local    = np.array([J_in4(o, i) for o, i in zip(od_local, id_local)])
        r_ft_local = np.maximum(1e-4, 0.5*(hole_d_profile - od_local) * IN2FT)

        rot_factor = rot_fc_factor if "rotate" in scenario else 1.0
        lam_hel_ui  = 2.83

        EI_factor     = EI_lbf_ft2_from_in4(Epsi, 1.0)
        EI_ft2_local  = EI_factor * I_local

        inc_local_rad = np.deg2rad(np.maximum(df_itr['inc_deg'].to_numpy(), 1e-6))
        denom         = np.maximum(r_ft_local * np.sin(inc_local_rad), 1e-9)

        Fs = (2.0 * EI_ft2_local * df_itr['w_b_lbft'].to_numpy()) / denom
        Fh = (lam_hel_ui * EI_ft2_local * df_itr['w_b_lbft'].to_numpy()) / denom
        Fs *= rot_factor; Fh *= rot_factor

        # bending stress
        M_b_lbf_in  = df_itr["N_lbf"].to_numpy() * r_ft_local * 12.0
        sigma_b_psi = np.divide(M_b_lbf_in, np.maximum(Z_local, 1e-9))

        # torsional shear
        r_in     = od_local / 2.0
        T_lbf_in = np.abs(df_itr["M_next_lbf_ft"].to_numpy()) * 12.0
        tau_psi  = np.divide(T_lbf_in * r_in, np.maximum(J_local, 1e-9))

        sigma_ax_psi = np.divide(df_itr["T_next_lbf"].to_numpy(), np.maximum(A_local, 1e-9))
        sigma_ax_wf  = sigma_ax_psi + np.sign(sigma_ax_psi)*np.abs(sigma_b_psi)
        sigma_vm_psi = np.sqrt(sigma_ax_wf**2 + 3.0*tau_psi**2)

        SB  = sigma_b_psi / 30000.0
        SV  = sigma_vm_psi / 60000.0
        SN  = np.abs(df_itr["N_lbf"].to_numpy()) / 5000.0
        BSI = 1.0 + 3.0 * np.clip(0.35*SB + 0.45*SV + 0.20*SN, 0.0, 1.0)

        # Build stacked figure
        fig = make_subplots(
            rows=3, cols=1, shared_yaxes=True,
            row_heights=[0.34, 0.33, 0.33],
            vertical_spacing=0.02,
            subplot_titles=("Hookload", "Torque (μ sweep)", "Buckling & Severity")
        )

        # Row1: Hookload
        fig.add_trace(go.Scatter(
            x=np.maximum(0.0, -df_itr['T_next_lbf'])/1000.0,
            y=depth,
            name="Hookload (k-lbf)", mode="lines",
            line=dict(color=COLOUR_CASED)
        ), row=1, col=1)

        # shade rig pull risk region in row1
        fig.add_shape(type="rect",
                      x0=rig_pull_lim/1000.0, x1=fig.layout.xaxis.range[1] if 'range' in fig.layout.xaxis else rig_pull_lim/1000.0*1.2,
                      y0=depth.min(), y1=depth.max(),
                      fillcolor="red", opacity=0.1, row=1, col=1, line_width=0)

        # Row2: Torque μ-sweep
        for mu in mu_band:
            dmu, tmu = run_td_off_bottom(mu, mu)
            fig.add_trace(go.Scatter(
                x=tmu/1000.0, y=dmu,
                name=f"μ={mu:.2f}", mode="lines"
            ), row=2, col=1)

        fig.add_vline(x=T_makeup_sf/1000.0, line_dash=LINE_DASH_LIMIT, line_color=COLOUR_LIMIT,
                      annotation_text="MU/SF", row=2, col=1)
        fig.add_vline(x=rig_torque_lim/1000.0, line_dash=LINE_DOT_LIMIT, line_color=COLOUR_LIMIT,
                      annotation_text="TD limit", row=2, col=1)

        # Row3: Buckling & Severity
        fig.add_trace(go.Scatter(x=Fs/1000.0,   y=depth, name="Fs (k-lbf)",   line=dict(dash="dash", color="gray")), row=3, col=1)
        fig.add_trace(go.Scatter(x=Fh/1000.0,   y=depth, name="Fh (k-lbf)",   line=dict(dash="dot",  color="gray")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_itr['N_lbf']/1000.0, y=depth, name="Side-force (k-lbf)",   line=dict(color="purple")), row=3, col=1)
        fig.add_trace(go.Scatter(x=sigma_b_psi/1000.0,       y=depth, name="Bending (ksi)",        line=dict(color="brown")), row=3, col=1)
        fig.add_trace(go.Scatter(x=sigma_vm_psi/1000.0,      y=depth, name="von Mises (ksi)",      line=dict(color="black")), row=3, col=1)
        fig.add_trace(go.Scatter(x=BSI,                   y=depth, name="BSI (1–4)",            line=dict(width=4, color="red")),   row=3, col=1)

        # Shade severity risk zone: e.g., BSI > 3
        idx_risk = np.where(BSI >= 3.0)[0]
        if idx_risk.size > 0:
            y0 = depth[idx_risk.min()]; y1 = depth[idx_risk.max()]
            fig.add_shape(type="rect", x0=0, x1=fig.layout.xaxis.range[1] if 'range' in fig.layout.xaxis else 4,
                          y0=y0, y1=y1, fillcolor="red", opacity=0.1, row=3, col=1, line_width=0)

        # Axis labels and formatting
        for r in (1,2,3):
            fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=r, col=1)
        fig.update_xaxes(title_text="Hookload (k-lbf)",          row=1, col=1)
        fig.update_xaxes(title_text="Torque (k lbf-ft)",         row=2, col=1)
        fig.update_xaxes(title_text="k-lbf / ksi / BSI",        row=3, col=1)

        fig.update_layout(
            height=900,
            template="simple_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True, key="stacked-main")

    # Friction calibration
    # … (unchanged UI block) …

    st.markdown("### Iteration trace (first 12 rows)")
    st.dataframe(df_itr.head(12), use_container_width=True)

    st.caption("Johancsik soft-string (Δs=1 ft) with enhanced buckling & severity visuals. Includes history-match μ overlay, NP check, 0.8×MU gate, BSR/SR, tortuosity, motor bit torque, rig limits, and stacked severity plots.")
