# streamlit_app.py — PEGN517 Wellpath + T&D (Δs = 1 ft)
# - Minimum Curvature (1 ft pieces)
# - Casing/liner/open-hole intervals with real API IDs
# - 3D trajectory colored by interval; line thickness ~ OD
# - Soft-string Torque & Drag with buoyancy & cased/open split
# - Streamlit 1.3x+: uses st.rerun()

import math
from typing import Tuple, List, Dict
import io, csv
import streamlit as st

st.set_page_config(page_title="Wellpath + Torque & Drag — PEGN517", layout="wide")
DEG2RAD = math.pi/180.0; RAD2DEG = 180.0/math.pi

# ---------------------------
# API-like casing mini library
# ---------------------------
# Each OD has a list of (ppf, id) objects. (IDs from API 5CT/World Oil charts.)
CASING_DB: Dict[str, Dict[str, List[Dict[str, float]]]] = {
    "13-3/8": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf":48.0, "id":12.415},
            {"ppf":54.5, "id":12.347},
            {"ppf":61.0, "id":12.115},
            {"ppf":68.0, "id":11.965},
        ],
    },
    "9-5/8": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf":36.0, "id":8.835},
            {"ppf":40.0, "id":8.535},
            {"ppf":43.5, "id":8.405},
            {"ppf":47.0, "id":8.311},
            {"ppf":53.5, "id":8.097},
        ],
    },
    "7-5/8": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf":39.0,  "id":6.625},
            {"ppf":42.8,  "id":6.501},
            {"ppf":45.3,  "id":6.435},
            {"ppf":47.1,  "id":6.375},
            {"ppf":51.2,  "id":6.249},
            {"ppf":52.8,  "id":6.201},
            {"ppf":55.75, "id":6.176},
        ],
    },
    "7": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf":23.0, "id":6.366},
            {"ppf":26.0, "id":6.276},
            {"ppf":29.0, "id":6.184},
            {"ppf":32.0, "id":6.094},
            {"ppf":35.0, "id":6.004},
            {"ppf":38.0, "id":5.938},
            {"ppf":41.0, "id":5.921},
        ],
    },
    "5-1/2": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf":14.0, "id":4.892},
            {"ppf":17.0, "id":4.778},
            {"ppf":20.0, "id":4.670},
            {"ppf":23.0, "id":4.563},
            {"ppf":26.0, "id":4.494},
        ],
    },
    "4-1/2": {
        "grades": ["J55","K55","N80","L80","P110"],
        "options": [
            {"ppf":9.5,  "id":4.090},
            {"ppf":10.5, "id":4.052},
            {"ppf":11.6, "id":4.000},
            {"ppf":13.5, "id":3.920},
        ],
    },
}

def parse_od_inch(label: str) -> float:
    if "-" in label:
        whole, frac = label.split("-"); num, den = frac.split("/")
        return float(whole) + float(num)/float(den)
    return float(label)

# ======================================
# Minimum Curvature (Δs = 1 ft) builders
# ======================================
def _rf(dogs_rad: float)->float:
    if abs(dogs_rad)<1e-12: return 1.0
    return 2.0/dogs_rad*math.tan(0.5*dogs_rad)

def _mcm_step(md1,inc1_deg,az1_deg,md2,inc2_deg,az2_deg,n1,e1,tvd1):
    ds = md2-md1
    inc1=inc1_deg*DEG2RAD; inc2=inc2_deg*DEG2RAD
    az1=az1_deg*DEG2RAD;   az2=az2_deg*DEG2RAD
    cosd = (math.cos(inc1)*math.cos(inc2)
            + math.sin(inc1)*math.sin(inc2)*math.cos(az2-az1))
    cosd = max(-1.0,min(1.0,cosd)); dogs = math.acos(cosd)
    rf=_rf(dogs)
    dN=0.5*ds*(math.sin(inc1)*math.cos(az1)+math.sin(inc2)*math.cos(az2))*rf
    dE=0.5*ds*(math.sin(inc1)*math.sin(az1)+math.sin(inc2)*math.sin(az2))*rf
    dT=0.5*ds*(math.cos(inc1)+math.cos(inc2))*rf
    return n1+dN, e1+dE, tvd1+dT, (0 if ds<=0 else dogs*RAD2DEG/ds*100.0)

def mcm_positions(md,inc,az):
    N,E,TVD,DLS=[0.0],[0.0],[0.0],[0.0]
    for i in range(1,len(md)):
        n2,e2,t2,d = _mcm_step(md[i-1],inc[i-1],az[i-1],md[i],inc[i],az[i],N[-1],E[-1],TVD[-1])
        N.append(n2); E.append(e2); TVD.append(t2); DLS.append(d)
    return N,E,TVD,DLS

def _gen_md(target_md: float, ds: float=1.0)->List[float]:
    steps = int(max(0.0,target_md)//ds); md=[i*ds for i in range(steps+1)]
    if md[-1]<target_md: md.append(target_md); return md

def synth_build_hold(kop, br, theta_hold, target_md, az):
    ds=1.0; md=_gen_md(target_md,ds); inc=[]
    for m in md:
        if m<kop: inc.append(0.0)
        else: inc.append(min((inc[-1] if inc else 0.0)+br*(ds/100.0), theta_hold))
    return md, inc, [az]*len(md)

def synth_build_hold_drop(kop, br, theta_hold, hold_len, dr, final_inc, target_md, az):
    ds=1.0; md=_gen_md(target_md,ds); inc=[]; built=False; in_hold=False; in_drop=False; H=hold_len or 0.0
    for m in md:
        if m<kop: inc.append(0.0); continue
        prev=inc[-1] if inc else 0.0
        if not built:
            nxt=min(prev+br*(ds/100.0),theta_hold); built=abs(nxt-theta_hold)<1e-12
            if built and H>0: in_hold=True; inc.append(nxt)
        elif in_hold:
            inc.append(theta_hold); H-=ds; 
            if H<=0: in_hold=False; in_drop=True
        elif in_drop:
            nxt=max(prev-dr*(ds/100.0),final_inc); inc.append(nxt)
            if abs(nxt-final_inc)<1e-12: in_drop=False
        else: inc.append(final_inc)
    return md, inc, [az]*len(md)

def synth_horizontal(kop, br, lat_len, target_md, az):
    ds=1.0
    if target_md is None:
        build_len=math.ceil(90.0/(br/100.0)); target_md=kop+build_len+(lat_len or 0.0)
    md=_gen_md(target_md,ds); inc=[]; built=False
    for m in md:
        if m<kop: inc.append(0.0); continue
        prev=inc[-1] if inc else 0.0
        if not built:
            nxt=min(prev+br*(ds/100.0),90.0); built=abs(nxt-90.0)<1e-12; inc.append(nxt)
        else: inc.append(90.0)
    return md, inc, [az]*len(md)

# ==========================================
# Soft-string Torque & Drag (Johancsik)
# ==========================================
def buoyancy_factor_ppg(MW_ppg: float)->float:
    # BF = (65.5 - MW)/65.5   (steel ≈ 65.5 ppg)
    BF = (65.5 - float(MW_ppg))/65.5
    return max(0.0, min(1.0, BF))

def kappa_from_dls(dls_deg_100: float)->float:
    return (dls_deg_100*DEG2RAD)/100.0

def solve_torque_drag(md, inc_deg, dls_deg_100,
                      dc_len, dc_od, dc_id, dc_w,
                      hwdp_len, hwdp_od, hwdp_id, hwdp_w,
                      dp_len, dp_od, dp_id, dp_w,
                      # intervals: list of (top_md, bot_md) for cased segments
                      cased_intervals: List[Tuple[float,float]],
                      mu_cased_slide, mu_open_slide,
                      mu_cased_rot=None, mu_open_rot=None,
                      mw_ppg=10.0, wob_lbf=0.0, bit_torque_lbf_ft=0.0,
                      scenario="Pickup"):
    n=len(md); kappa=[kappa_from_dls(d) for d in dls_deg_100]
    mu_rot_cased = mu_cased_rot if mu_cased_rot is not None else mu_cased_slide
    mu_rot_open  = mu_open_rot  if mu_open_rot  is not None else mu_open_slide

    bit_depth = md[-1]
    use_dc =  max(0.0, min(dc_len, bit_depth))
    use_hwdp = max(0.0, min(hwdp_len, bit_depth-use_dc))
    use_dp =  max(0.0, min(dp_len, bit_depth-use_dc-use_hwdp))

    def comp_at_md(m):
        if m>(bit_depth-use_dc): return dc_w, dc_od
        elif m>(bit_depth-use_dc-use_hwdp): return hwdp_w, hwdp_od
        else: return dp_w, dp_od

    def is_cased(middle_md: float)->bool:
        for a,b in cased_intervals:
            if middle_md>=a-1e-9 and middle_md<=b+1e-9:
                return True
        return False

    BF = buoyancy_factor_ppg(mw_ppg)
    sigma_ax = +1.0 if scenario.lower().startswith("pick") else -1.0
    T=[-float(wob_lbf) if wob_lbf>0 and scenario.lower().startswith("rotate on") else 0.0]
    M=[float(bit_torque_lbf_ft) if scenario.lower().startswith("rotate on") else 0.0]
    rows=[]

    # integrate bit -> surface
    for i in range(n-1,0,-1):
        md2, md1 = md[i], md[i-1]; ds = md2-md1
        theta = inc_deg[i]*DEG2RAD; kap = kappa[i]
        w_air, od_in = comp_at_md(md2); w_b = w_air*BF; r_eff_ft=(od_in/2.0)/12.0
        mid_md = 0.5*(md1+md2)
        in_cased = is_cased(mid_md)
        mu_slide = mu_cased_slide if in_cased else mu_open_slide
        mu_rot   = mu_rot_cased  if in_cased else mu_rot_open

        N = w_b*math.sin(theta) + T[-1]*kap
        dT = (sigma_ax*w_b*math.cos(theta) + mu_slide*N)*ds
        dM = (mu_rot*N*r_eff_ft)*ds
        T_next = T[-1]+dT; M_next = M[-1]+dM

        rows.append({
            "md_top_ft":md1,"md_bot_ft":md2,"ds_ft":ds,
            "inc_deg":inc_deg[i],"kappa_rad_ft":kap,
            "w_air_lbft":w_air,"BF":BF,"w_b_lbft":w_b,
            "mu_slide":mu_slide,"mu_rot":mu_rot,
            "N_lbf":N,"dT_lbf":dT,"T_next_lbf":T_next,
            "r_eff_ft":r_eff_ft,"dM_lbf_ft":dM,"M_next_lbf_ft":M_next,
            "cased?":in_cased
        })
        T.append(T_next); M.append(M_next)

    T=list(reversed(T)); M=list(reversed(M)); rows.reverse()
    return {"T_lbf_along":T,"M_lbf_ft_along":M,
            "trace_rows":rows,
            "surface_hookload_lbf":T[0],
            "surface_torque_lbf_ft":M[0] + (bit_torque_lbf_ft if scenario.lower().startswith("rotate on") else 0.0)}

# ----------
# UI helpers
# ----------
def decimate(xs, factor:int): return xs[::max(1,int(factor))]

def intervals_from_program(rows, td)->Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    """Return (cased_intervals, openhole_intervals) as [ (top,bot), ... ] within [0, TD]."""
    cased=[]; openh=[]
    for r in rows:
        typ=r["type"]
        if typ=="Surface casing":
            cased.append((0.0, min(r["shoe_md"], td)))
        elif typ=="Liner":
            top=min(r["top_md"], td); bot=min(r["shoe_md"], td)
            if bot>top: cased.append((top, bot))
        elif typ=="Open hole":
            top=min(r["top_md"], td); bot=min(r["bot_md"], td)
            if bot>top: openh.append((top, bot))
    # any remaining gaps below TD are open hole implicitly; but explicit open-hole rows keep their label
    return cased, openh

# ============== APP ==============
st.title("Wellpath + Torque & Drag (Δs = 1 ft)")

tab1, tab2 = st.tabs(["1) Trajectory & 3D schematic", "2) Torque & Drag"])

with tab1:
    st.subheader("Synthetic survey (Minimum Curvature)")
    a,b,c = st.columns(3)
    with a:
        profile = st.selectbox("Profile", ["Build & Hold","Build & Hold & Drop","Horizontal (Build + Lateral)"])
        kop_md  = st.number_input("KOP MD (ft)", 0.0, None, 1000.0, 50.0)
    with b:
        az_quick = st.selectbox("Quick azimuth", ["North (0)","East (90)","South (180)","West (270)"], index=0)
        az_map={"North (0)":0.0,"East (90)":90.0,"South (180)":180.0,"West (270)":270.0}
        az_deg = st.number_input("Azimuth (deg)", 0.0, 360.0, az_map[az_quick], 1.0)
        build_rate = st.selectbox("Build rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
    with c:
        theta_hold=hold_len=drop_rate=final_inc=lat_len=target_md=None
        if profile=="Build & Hold":
            theta_hold = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 1.0)
            target_md  = st.number_input("Target MD (ft)", kop_md, None, 10000.0, 100.0)
        elif profile=="Build & Hold & Drop":
            theta_hold = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 1.0)
            hold_len   = st.number_input("Hold length (ft)", 0.0, None, 1000.0, 50.0)
            drop_rate  = st.selectbox("Drop rate (deg/100 ft)", [0.5,1,1.5,2,3,4,6,8,10], index=4)
            final_inc  = st.number_input("Final inclination after drop (deg)", 0.0, 90.0, 0.0, 1.0)
            target_md  = st.number_input("Target MD (ft)", kop_md, None, 12000.0, 100.0)
        else:
            lat_len   = st.number_input("Lateral length (ft)", 0.0, None, 2000.0, 100.0)
            target_md = st.number_input("Target MD (0 = auto)", 0.0, None, 0.0, 100.0) or None

    # ----- Casing / Liner / Open-hole program -----
    st.markdown("### Casing / Liner / Open-hole program")
    st.caption("Select interval **Type**. For casing: Nominal OD → Weight (lb/ft) (ID auto) → Grade → depths. For open-hole: define Top/Bottom MD.")
    if "prog_rows" not in st.session_state: st.session_state["prog_rows"]=1

    hdr = st.columns([1.2,1.2,1.0,1.0,1.2,1.2,1.2,1.2])
    hdr[0].markdown("**Type**"); hdr[1].markdown("**Nominal OD**"); hdr[2].markdown("**lb/ft**"); hdr[3].markdown("**ID (in)**")
    hdr[4].markdown("**Grade**"); hdr[5].markdown("**Top MD (ft)**"); hdr[6].markdown("**Shoe/Bottom MD (ft)**"); hdr[7].markdown("**Actions**")

    program=[]
    for i in range(st.session_state["prog_rows"]):
        c0,c1,c2,c3,c4,c5,c6,c7 = st.columns([1.2,1.2,1.0,1.0,1.2,1.2,1.2,1.2])
        typ = c0.selectbox("", ["Surface casing","Liner","Open hole"], key=f"type_{i}")

        if typ=="Open hole":
            top_md = c5.number_input("", 0.0, None, 0.0, 50.0, key=f"top_{i}")
            bot_md = c6.number_input("", 0.0, None, 0.0, 50.0, key=f"bot_{i}")
            if c7.button("➖ Remove", key=f"rm_{i}"):
                st.session_state["prog_rows"]=max(0,st.session_state["prog_rows"]-1); st.rerun()
            program.append({"type":typ, "top_md":top_md, "bot_md":bot_md})
            continue

        od = c1.selectbox("", list(CASING_DB.keys()), key=f"od_{i}")
        # options as objects to avoid any string-matching bugs
        opts = CASING_DB[od]["options"]
        sel_opt = c2.selectbox("", opts, key=f"ppf_{i}",
                               format_func=lambda o: f"{o['ppf']:.2f}".rstrip("0").rstrip("."))
        c3.text_input("", f"{sel_opt['id']:.3f}", key=f"id_{i}", disabled=True)
        grade = c4.selectbox("", CASING_DB[od]["grades"], key=f"gr_{i}")

        if typ=="Surface casing":
            top_md = 0.0; shoe_md = c6.number_input("", 0.0, None, 3000.0, 50.0, key=f"shoe_{i}")
            c5.text_input("", "—", key=f"top_txt_{i}", disabled=True)
        else:  # Liner
            top_md = c5.number_input("", 0.0, None, 7000.0, 50.0, key=f"top_{i}")
            shoe_md = c6.number_input("", 0.0, None, 9000.0, 50.0, key=f"shoe_{i}")

        if c7.button("➖ Remove", key=f"rm_{i}"):
            st.session_state["prog_rows"]=max(0,st.session_state["prog_rows"]-1); st.rerun()

        program.append({
            "type":typ, "od":od, "ppf":float(sel_opt["ppf"]), "id":sel_opt["id"], "grade":grade,
            "top_md":top_md, "shoe_md":shoe_md
        })

    add = st.columns([7,1]); 
    if add[1].button("➕ Add interval"): st.session_state["prog_rows"]=min(12, st.session_state["prog_rows"]+1); st.rerun()

    st.caption("All piecewise calculations use **Δs = 1 ft**.")

    if st.button("Compute trajectory & 3D schematic"):
        # survey
        if profile=="Build & Hold":
            md,inc,az = synth_build_hold(kop_md, build_rate, theta_hold, target_md, az_deg)
        elif profile=="Build & Hold & Drop":
            md,inc,az = synth_build_hold_drop(kop_md, build_rate, theta_hold, hold_len, drop_rate, final_inc, target_md, az_deg)
        else:
            md,inc,az = synth_horizontal(kop_md, build_rate, lat_len, target_md, az_deg)

        north,east,tvd,dls = mcm_positions(md,inc,az)
        st.session_state["last_traj"] = (md,inc,az,dls,tvd,north,east)
        st.session_state["program"] = program

        # prep intervals for plotting & T&D
        TD = md[-1]; cased_int, openh_int = intervals_from_program(program, TD)

        # 3D schematic: colored traces per interval (thickness ~ OD)
        import plotly.graph_objects as go
        fig3d = go.Figure()
        palette = ["#4C78A8","#F58518","#54A24B","#EECA3B","#B279A2","#FF9DA6","#9C755F","#E45756"]
        # Cased intervals first
        for j, r in enumerate([x for x in program if x["type"]!="Open hole"]):
            top = 0.0 if r["type"]=="Surface casing" else r["top_md"]
            bot = r["shoe_md"]
            idx = [k for k,m in enumerate(md) if m>=top-1e-9 and m<=bot+1e-9]
            if len(idx) < 2: continue
            od_in = parse_od_inch(r["od"])
            width_px = max(2, int(2 + od_in/3.0))
            fig3d.add_trace(go.Scatter3d(
                x=[east[k] for k in idx], y=[north[k] for k in idx], z=[-tvd[k] for k in idx],
                mode="lines", line=dict(width=width_px, color=palette[j%len(palette)]),
                name=f"{r['type']}: {r['od']} {r['ppf']}# {r['grade']} ({int(top)}–{int(bot)} ft)"
            ))
        # Explicit open-hole intervals
        for r in [x for x in program if x["type"]=="Open hole"]:
            top, bot = r["top_md"], r["bot_md"]
            idx = [k for k,m in enumerate(md) if m>=top-1e-9 and m<=bot+1e-9]
            if len(idx)<2: continue
            fig3d.add_trace(go.Scatter3d(
                x=[east[k] for k in idx], y=[north[k] for k in idx], z=[-tvd[k] for k in idx],
                mode="lines", line=dict(width=4, color="#8B4513"),
                name=f"Open hole: {int(top)}–{int(bot)} ft"
            ))
        # Any leftover below deepest casing is also open hole (implicit)
        # (The T&D solver classifies per-segment anyway.)

        fig3d.update_layout(title="3D Trajectory — Casing/Liner/Open-hole Schematic",
                            scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft, down)"),
                            legend=dict(itemsizing="constant"), margin=dict(l=0,r=0,t=40,b=0))
        prof = go.Figure(); prof.add_trace(go.Scatter(x=md, y=tvd, mode="lines", name="Profile"))
        prof.update_yaxes(autorange="reversed"); prof.update_layout(title="Profile", xaxis_title="MD (ft)", yaxis_title="TVD (ft)")
        plan = go.Figure(); plan.add_trace(go.Scatter(x=east, y=north, mode="lines", name="Plan"))
        plan.update_layout(title="Plan", xaxis_title="East (ft)", yaxis_title="North (ft)")

        A,B = st.columns([2,1]); A.plotly_chart(fig3d, use_container_width=True); B.plotly_chart(prof, use_container_width=True); B.plotly_chart(plan, use_container_width=True)

        # survey table + CSV
        tbl=[{"MD (ft)":md[i],"Inc (deg)":inc[i],"Az (deg)":az[i],"TVD (ft)":tvd[i],"North (ft)":north[i],"East (ft)":east[i],"DLS (deg/100 ft)":dls[i]} for i in range(len(md))]
        st.subheader("Survey and calculated positions"); st.dataframe(tbl, use_container_width=True, height=420)
        buf=io.StringIO(); w=csv.DictWriter(buf, fieldnames=list(tbl[0].keys())); w.writeheader(); w.writerows(tbl)
        st.download_button("Download trajectory CSV", data=buf.getvalue().encode("utf-8"), file_name="trajectory.csv", mime="text/csv")

with tab2:
    st.subheader("Soft-string Torque & Drag (Johancsik) — with buoyancy")

    L,R = st.columns(2)
    with L:
        mw_ppg = st.number_input("Mud weight (ppg)", 6.0, 20.0, 10.0, 0.1)
        mu_cased_slide = st.number_input("μ in casing (sliding)", 0.05, 0.60, 0.25, 0.01)
        mu_open_slide  = st.number_input("μ in open hole (sliding)", 0.05, 0.60, 0.35, 0.01)
        mu_cased_rot   = st.number_input("μ in casing (rotating)", 0.00, 1.00, 0.25, 0.01)
        mu_open_rot    = st.number_input("μ in open hole (rotating)", 0.00, 1.00, 0.35, 0.01)
    with R:
        wob_lbf = st.number_input("WOB (lbf) for on-bottom", 0.0, None, 0.0, 1000.0)
        bit_torque = st.number_input("Bit torque (lbf-ft) for on-bottom", 0.0, None, 0.0, 100.0)
        scenario = st.selectbox("Scenario", ["Pickup (POOH)","Slack-off (RIH)","Rotate off-bottom","Rotate on-bottom"])

    st.markdown("#### Drillstring (bit up)")
    c1,c2,c3 = st.columns(3)
    with c1:
        dc_len = st.number_input("DC length (ft)", 0.0, None, 600.0, 10.0)
        dc_od  = st.number_input("DC OD (in)", 1.0, 12.0, 8.0, 0.5)
        dc_id  = st.number_input("DC ID (in)", 0.0, 10.0, 2.813, 0.001)
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

    if st.button("Run Torque & Drag"):
        if "last_traj" not in st.session_state:
            st.error("No trajectory found. Compute in Tab 1 first.")
        else:
            md,inc,az,dls,tvd,north,east = st.session_state["last_traj"]
            TD = md[-1]
            prog = st.session_state.get("program", [])
            cased_int, _ = intervals_from_program(prog, TD)

            out = solve_torque_drag(
                md,inc,dls,
                dc_len,dc_od,dc_id,dc_w,
                hwdp_len,hwdp_od,hwdp_id,hwdp_w,
                dp_len,dp_od,dp_id,dp_w,
                cased_int,
                mu_cased_slide, mu_open_slide,
                mu_cased_rot,  mu_open_rot,
                mw_ppg=mw_ppg, wob_lbf=wob_lbf, bit_torque_lbf_ft=bit_torque,
                scenario=("Pickup" if scenario.startswith("Pickup") else
                          "Slack-off" if scenario.startswith("Slack") else
                          "Rotate on-bottom" if scenario.endswith("on-bottom") else
                          "Rotate off-bottom")
            )
            import plotly.graph_objects as go
            T,M = out["T_lbf_along"], out["M_lbf_ft_along"]
            st.success(f"Surface hookload: {out['surface_hookload_lbf']:,.0f} lbf — "
                       f"Surface torque: {out['surface_torque_lbf_ft']:,.0f} lbf-ft")

            fig1=go.Figure(); fig1.add_trace(go.Scatter(x=md,y=T,mode="lines",name="Tension/Hookload"))
            fig1.update_layout(title="Tension / Hookload vs MD", xaxis_title="MD (ft)", yaxis_title="lbf")
            fig2=go.Figure(); fig2.add_trace(go.Scatter(x=md,y=M,mode="lines",name="Torque"))
            fig2.update_layout(title="Torque vs MD", xaxis_title="MD (ft)", yaxis_title="lbf-ft")
            A,B = st.columns(2); A.plotly_chart(fig1, use_container_width=True); B.plotly_chart(fig2, use_container_width=True)

            rows = out["trace_rows"]
            if rows:
                st.subheader("Iteration trace (bit → surface)")
                st.dataframe(rows, use_container_width=True, height=380)
                buf=io.StringIO(); w=csv.DictWriter(buf, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
                st.download_button("Download iteration trace (CSV)", data=buf.getvalue().encode("utf-8"),
                                   file_name="td_iteration_trace.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "Casing IDs/weights from API 5CT/World Oil charts; e.g., 13⅜-61→ID 12.115 in, 9⅝-36→ID 8.835 in, 7-23→ID 6.366 in. "
    "Buoyancy factor BF = (65.5 − MW)/65.5 applied to all component weights. "
    "Soft-string (Johancsik) equations with per-segment cased/open classification."
)
