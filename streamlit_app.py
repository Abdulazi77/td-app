# =========================
# File: main.py
# =========================
import well_profile
import torque_n_drag
import io
import contextlib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Soft String Torque and Drag", layout="wide")
st.markdown(" ## Soft String Torque and Drag")
st.markdown(" ### PEGN 517 Advanced Drilling Engineering 2025")
st.markdown("Noah Perkovich, Abdulaziz Alzahrani, Shaolin Jahan Eidee")
st.markdown("###### press 'enter' after parameter input then click run, (torque history takes several seconds to run with default step size of 1ft)")

# ---- Layout: Sidebar for inputs, main area for plot ----
st.sidebar.header("Trajectory and Hole Parameters")

with st.form("input_form"):
    # trajectory inputs
    KOP = st.sidebar.number_input("Kick-Off Point (ft)", value=2000.0, step=10.0)
    BUR = st.sidebar.number_input("Build-Up Rate (deg/100ft)", value=3.0, step=0.5)
    theta1 = st.sidebar.number_input("Hold Inclination (deg)", value=60.0, step=1.0)
    L1 = st.sidebar.number_input("Hold Length (ft)", value=2000.0, step=10.0)
    azimuth = st.sidebar.number_input("Azimuth (deg)", value=45.0, step=1.0)
    delta = st.sidebar.number_input("Delta s (ft)", value=1.0, step=0.1)

    # Optional trajectory inputs
    BDR_input = st.sidebar.text_input("Build-Down Rate (deg/100ft, optional) - required for build-hold-drop", value="")
    BDR = float(BDR_input) if BDR_input.strip() != "" else None
    theta2_input = st.sidebar.text_input("End Inclination (deg, optional) - required for build-hold-drop", value="")
    theta2 = float(theta2_input) if theta2_input.strip() != "" else None
    L2_input = st.sidebar.text_input("End Hold length (ft, optional) - to hold after drop", value="")
    L2 = float(L2_input) if L2_input.strip() != "" else None
    max_md_input = st.sidebar.text_input("Max MD (ft, optional)", value="")
    max_md = float(max_md_input) if max_md_input.strip() != "" else None
    max_tvd_input = st.sidebar.text_input("Max TVD (ft, optional)", value="")
    max_tvd = float(max_tvd_input) if max_tvd_input.strip() != "" else None
    horizontal_target_input = st.sidebar.text_input("Horizontal Target (ft, optional) - for horizontal to land at this depth", value="")
    horizontal_target = float(horizontal_target_input) if horizontal_target_input.strip() != "" else None

    # hole inputs
    c_md = st.sidebar.number_input("Casing Depth (ft)", value=2000.0, step=10.0)
    c_id = st.sidebar.number_input("Casing ID (in)", value=6.5, step=0.01)
    oh_id = st.sidebar.number_input("Hole ID (in)", value=6.0, step=0.01)

    # Fluid & Joint Input
    MW = st.sidebar.number_input("Mud Weight (ppg)", value=10.0, step=0.1)
    joint_length = st.sidebar.number_input("Joint Length (ft)", value=30.0, step=0.1)

    # Drill String 
    st.markdown("Drill String Components (Bottom to Top, exclude BHA, leave last (n) = None)")
    default_ds = pd.DataFrame({
        'component':        ['DC', 'HWDP', 'DP'],
        'OD (in)':          [6.0,    5.0,   4.0 ],
        'ID (in)':          [2.25,   3.0,   3.34],
        'Wt (lb/ft)':       [90.1,   49.0,  14.0],
        'YP (kpsi)':        [100.0,  100.0, 75.0],
        'number of joints': [10,     20,    None]  # last value left blank for auto calculation
    })
    ds_params = st.data_editor(default_ds, num_rows="dynamic")

    # friction param inputs
    mu_c = st.sidebar.number_input("Friction Coefieient in Casing (f/f)", value=0.3, step=0.05)
    mu_o = st.sidebar.number_input("Friction Coefieient in Open Hole (f/f)", value=0.3, step=0.05)
    mu_b = st.sidebar.number_input("Friction Coefieient at the Bit (f/f) - this is probably a poor parameter and should be a function of WOB", value=0.5, step=0.05)
    fric_coefs = {'mu_case': mu_c, 'mu_open': mu_o, 'mu_bit': mu_b}

    WOB = st.sidebar.number_input("Weight on Bit (lb) for rotating ON bottom", value=20000.0, step=10000.0)

    submitted = st.form_submit_button("Run Calculations")

if submitted:
    try:
        # Capture print output
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # problem setup
            hole = well_profile.build_hold_drop(
                KOP=KOP, BUR=BUR, BDR=BDR, theta1=theta1, theta2=theta2, L1=L1,
                azimuth=azimuth, delta=delta, max_md=max_md, max_tvd=max_tvd,
                horizontal_target=horizontal_target, L2=L2
            )
            hole_df = well_profile.hole_profile(hole, c_md=c_md, c_id=c_id, oh_id=oh_id)
            ds_df = well_profile.drill_string(ds_params, joint_length, hole['MD'])

        # Plot well profile (3-pane: includes a small drill string profile)
        fig = well_profile.plot_well_profile(hole_df, ds_df)
        st.plotly_chart(fig, use_container_width=True)

        # --- Show log output, if any
        output = buffer.getvalue().strip()
        st.markdown(f"```\n{output}\n```" if output else "")

        # === Classic Oct-14 style graphs (now always visible) ===
        inner_col1, inner_col2 = st.columns(2)

        with inner_col1:
            # Tension profiles at TMD (RiH, PooH, RoB, ROB)
            T_RiH = torque_n_drag.tension(hole_df, ds_df, fric_coefs, MW, Op='RiH', WOB=0)
            T_PooH = torque_n_drag.tension(hole_df, ds_df, fric_coefs, MW, Op='PooH', WOB=0)
            T_RoB = torque_n_drag.tension(hole_df, ds_df, fric_coefs, MW, Op='RoB', WOB=0)
            T_ROB = torque_n_drag.tension(hole_df, ds_df, fric_coefs, MW, Op='RoB', WOB=WOB)
            fig_tension = torque_n_drag.plot_TD_tensions(T_RiH, T_PooH, T_RoB, T_ROB, hole['MD'])
            st.plotly_chart(fig_tension, use_container_width=True)

        with inner_col2:
            # Surface torque vs depth while rotating off bottom
            Tq_bh = torque_n_drag.torque_plan(hole_df, ds_df, fric_coefs, MW)
            fig_torque = torque_n_drag.plot_surface_torques(Tq_bh, hole['MD'][-len(Tq_bh):])
            st.plotly_chart(fig_torque, use_container_width=True)

        # Standalone Drill String Profile (classic single stick)
        fig_ds = well_profile.plot_string_profile(ds_df)
        st.plotly_chart(fig_ds, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# =========================
# File: torque_n_drag.py
# =========================
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go

def buoy_factor(mw, steel_density=65.5):
    return (steel_density - mw) / steel_density

def normal_force(weight, inclination):
    return weight * np.sin(np.radians(inclination))

def axial_weight(weight, inclination):
    return weight * np.cos(np.radians(inclination))

def tension(hole_df, ds_df, fric_coefs, mw, Op='RiH', WOB=0):
    """Axial tension along string at TD for basic ops."""
    bf = buoy_factor(mw)
    md = ds_df['MD']
    wt_per_ft = ds_df['Wt'] * bf
    inc = hole_df['INC']
    wall = hole_df['WALL']

    ds = np.diff(md, prepend=md[0])
    mu_case = fric_coefs['mu_case']
    mu_open = fric_coefs['mu_open']
    mu = np.where(wall == 'Steel', mu_case, mu_open)

    W_seg = wt_per_ft * ds
    W_axial = axial_weight(W_seg, inc)
    N = normal_force(W_seg, inc)

    if Op == 'RiH':
        Ff = -mu * N; WOB = 0
    elif Op == 'PooH':
        Ff = mu * N;  WOB = 0
    elif Op == 'RoB':
        Ff = np.zeros_like(N)
    else:
        raise ValueError("Unknown Op")

    T = np.cumsum((W_axial + Ff)[::-1])[::-1]
    T -= WOB
    return np.array(T)

def torque(hole_df, ds_df, fric_coefs, mw, Op='RiH', WOB=0):
    """Elemental torque along string."""
    bf = buoy_factor(mw)
    md = ds_df['MD']
    wt_per_ft = ds_df['Wt'] * bf
    inc = hole_df['INC']
    wall = hole_df['WALL']

    ds = np.diff(md, prepend=md[0])
    mu_case = fric_coefs['mu_case']
    mu_open = fric_coefs['mu_open']
    mu_bit = fric_coefs['mu_bit']

    mu = np.where(wall == 'Steel', mu_case, mu_open)
    W_seg = wt_per_ft * ds
    N = normal_force(W_seg, inc)
    Ff = mu * N

    if Op in ['RiH', 'PooH']:
        WOB = 0
    elif Op == 'RoB':
        pass
    else:
        raise ValueError("Unknown Op")

    Tq = np.cumsum((Ff / 2 / 12)[::-1])[::-1]
    Tq += WOB * mu_bit * hole_df['HOLE_ID'].iloc[-1] / 2 / 12
    return np.array(Tq)

def torque_plan(hole_df, ds_df, fric_coefs, mw):
    """Surface torque vs depth while rotating off bottom (progressive TD)."""
    open_hole_idx = hole_df.index[hole_df['WALL'] == 'Open']
    n_steps = len(open_hole_idx)
    surface_torque_RoB = np.zeros(n_steps)

    for j, i_td in enumerate(tqdm(open_hole_idx)):
        hole_slice = hole_df.iloc[:i_td + 1].reset_index(drop=True)
        n_rows = len(hole_slice)
        ds_slice = ds_df.iloc[-n_rows:].copy().reset_index(drop=True)
        ds_slice['MD'] = hole_slice['MD'].values
        Tq_RoB = torque(hole_slice, ds_slice, fric_coefs, mw, Op='RoB', WOB=0)
        surface_torque_RoB[j] = Tq_RoB[0]

    return surface_torque_RoB

def plot_TD_tensions(T_RiH, T_PooH, T_RoB, T_ROB, md):
    """Classic tension profile plot (4 ops) vs MD."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_RiH, y=md, mode='lines', name='RiH', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=T_PooH, y=md, mode='lines', name='PooH', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=T_RoB, y=md, mode='lines', name='RoB', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=T_ROB, y=md, mode='lines', name='ROB', line=dict(color='orange')))
    fig.update_layout(
        yaxis=dict(autorange='reversed', title='MD (ft)', showgrid=True),
        xaxis=dict(title='Tension (lb)', showgrid=True),
        title='Tension profile along drill string at TMD',
        legend=dict(title='Activity'),
        height=700
    )
    return fig

def plot_surface_torques(Tq_bh, md):
    """Classic surface torque vs MD while RoB (open hole)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Tq_bh, y=md, mode='lines', name='RoB', line=dict(color='green')))
    fig.update_layout(
        yaxis=dict(autorange='reversed', title='MD (ft)', showgrid=True),
        xaxis=dict(title='Torque (ft-lb)', showgrid=True),
        title='Surface Torques Rotating off Bottom  \n while drilling open hole section',
        legend=dict(title='Activity'),
        height=700
    )
    return fig


# =========================
# File: well_profile.py
# =========================
import numpy as np
import pandas as pd

def dogleg(i1, i2, a1, a2):
    i1, i2, a1, a2 = np.radians([i1, i2, a1, a2])
    dl = np.arccos(np.cos(i2 - i1) - (np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))))
    return np.degrees(dl)

def dxdydz(i1, i2, a1, a2, delta_m, rf):
    i1, i2, a1, a2 = np.radians([i1, i2, a1, a2])
    dx = (delta_m / 2) * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    dy = (delta_m / 2) * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    dz = (delta_m / 2) * (np.cos(i1) + np.cos(i2)) * rf
    return dx, dy, dz

def ratio_factor(dl):
    dl = np.radians(dl)
    return  2 / dl * np.tan(dl / 2) if dl != 0 else 1

def stepper(md, tvd, n, e, i, a, delta_m, delta_i, delta_a):
    dl = dogleg(i, i+delta_i, a, a + delta_a)
    rf = ratio_factor(dl)
    delta_e, delta_n, delta_tvd = dxdydz(i, i+delta_i, a, a + delta_a, delta_m, rf)
    return md + delta_m, tvd + delta_tvd, n + delta_n, e + delta_e, i + delta_i, a + delta_a

def reached_max(md_val, tvd_val, max_md, max_tvd):
    if max_md is not None and md_val >= max_md: return True
    if max_tvd is not None and tvd_val >= max_tvd: return True
    return False

def build_hold_drop(KOP, BUR, theta1, L1, azimuth, delta=1,
                    max_md=None, max_tvd=None, horizontal_target=None, BDR=None, theta2=None, L2=None):
    """Generate synthetic well survey (build/hold/drop + optional horizontal)."""
    md = [0.0]; tvd = [0.0]; north = [0.0]; east = [0.0]; inc = [0.0]; az = [azimuth]; dls = [0.0]
    next_md, next_tvd, next_n, next_e, next_i, next_a = md[-1], tvd[-1], north[-1], east[-1], inc[-1], az[-1]

    # vertical to KOP
    while next_md < KOP:
        if reached_max(next_md, next_tvd, max_md, max_tvd): break
        next_md, next_tvd, next_n, next_e, next_i, next_a = stepper(next_md, next_tvd, next_n, next_e, next_i, next_a, delta, 0, 0)
        md.append(next_md); tvd.append(next_tvd); north.append(next_n); east.append(next_e); inc.append(next_i); az.append(next_a)
        dls.append(dogleg(inc[-2], next_i, az[-2], next_a) / (delta / 100))

    # build to theta1
    while next_i < theta1:
        if reached_max(next_md, next_tvd, max_md, max_tvd): break
        delta_i_step = min(BUR * delta / 100, theta1 - next_i)
        next_md, next_tvd, next_n, next_e, next_i, next_a = stepper(next_md, next_tvd, next_n, next_e, next_i, next_a, delta, delta_i_step, 0)
        md.append(next_md); tvd.append(next_tvd); north.append(next_n); east.append(next_e); inc.append(next_i); az.append(next_a)
        dls.append(dogleg(inc[-2], next_i, az[-2], next_a) / (delta / 100))

    # horizontal landing if requested
    if horizontal_target is not None:
        # simple landing at target TVD; build to 90 then lateral
        step_md = delta
        while next_i < 90.0:
            if reached_max(next_md, next_tvd, max_md, max_tvd): break
            delta_i_step = min(BUR * step_md / 100, 90.0 - next_i)
            next_md, next_tvd, next_n, next_e, next_i, next_a = stepper(next_md, next_tvd, next_n, next_e, next_i, next_a, step_md, delta_i_step, 0)
            md.append(next_md); tvd.append(next_tvd); north.append(next_n); east.append(next_e); inc.append(next_i); az.append(next_a)
            dls.append(dogleg(inc[-2], next_i, az[-2], next_a) / (step_md / 100))

        shift = horizontal_target - tvd[-1]
        if shift < 0:
            raise ValueError(
                f"Cannot land this horizontal at {horizontal_target:.1f} ft TVD with BUR={BUR}°/100ft! "
                "Try increasing BUR or make this a deeper target."
            )
        tvd = [t + shift for t in tvd]
        md = [m + shift for m in md]

        n_prep = int(np.round(shift/delta,0))
        prep_md = [delta * (i) for i in range(n_prep)]
        md = prep_md + md; tvd = prep_md + tvd
        north = [0.0]*n_prep + north; east = [0.0]*n_prep + east
        inc = [0.0]*n_prep + inc; az = [0.0]*n_prep + az; dls = [0.0]*n_prep + dls

        next_md = md[-1]; next_tvd = tvd[-1]

    # hold to L1
    md_eob = md[-1]
    while next_md - md_eob < L1:
        if reached_max(next_md, next_tvd, max_md, max_tvd): break
        next_md, next_tvd, next_n, next_e, next_i, next_a = stepper(next_md, next_tvd, next_n, next_e, next_i, next_a, delta, 0, 0)
        md.append(next_md); tvd.append(next_tvd); north.append(next_n); east.append(next_e); inc.append(next_i); az.append(next_a)
        dls.append(dogleg(inc[-2], next_i, az[-2], next_a) / (delta / 100))

    # optional drop to theta2 and second hold L2
    if None not in (BDR, theta2):
        while next_i > theta2:
            if reached_max(next_md, next_tvd, max_md, max_tvd): break
            step_md = delta
            delta_i_step = theta2 - next_i if next_i - BDR * step_md / 100 < theta2 else -BDR * step_md / 100
            next_md, next_tvd, next_n, next_e, next_i, next_a = stepper(next_md, next_tvd, next_n, next_e, next_i, next_a, step_md, delta_i_step, 0)
            md.append(next_md); tvd.append(next_tvd); north.append(next_n); east.append(next_e); inc.append(next_i); az.append(next_a)
            dls.append(dogleg(inc[-2], next_i, az[-2], next_a) / (step_md / 100))

        if L2 is not None:
            md_eob = md[-1]
            while next_md - md_eob < L2:
                if reached_max(next_md, next_tvd, max_md, max_tvd): break
                next_md, next_tvd, next_n, next_e, next_i, next_a = stepper(next_md, next_tvd, next_n, next_e, next_i, next_a, delta, 0, 0)
                md.append(next_md); tvd.append(next_tvd); north.append(next_n); east.append(next_e); inc.append(next_i); az.append(next_a)
                dls.append(dogleg(inc[-2], next_i, az[-2], next_a) / (step_md / 100))

    return pd.DataFrame({"MD": md, "TVD": tvd, "NORTH": north, "EAST": east, "INC": inc, "AZIM": az, "DLS": dls})

def hole_profile(df, c_md, c_id, oh_id):
    df = df.copy()
    df['HOLE_ID'] = np.where(df['MD'] <= c_md, c_id, oh_id)
    df['WALL'] = np.where(df['MD'] <= c_md, 'Steel', 'Open')
    return df

def drill_string(ds_df, joint_length, md):
    """Map DS components (OD, ID, Wt, YP, component) along MD array `md`."""
    total_length = md.max()
    fixed_length = (ds_df['number of joints'].fillna(0) * joint_length).sum()
    if fixed_length > total_length: raise ValueError("You entered too many joints ")

    # num DP joints at top
    last_row = ds_df.iloc[-1]
    if pd.isna(last_row['number of joints']):
        last_fixed = ds_df.iloc[:-1]['number of joints'].fillna(0).sum() * joint_length
        remaining = max(total_length - last_fixed, 0)
        num_joints = int(np.ceil(remaining / joint_length))
        ds_df.at[last_row.name, 'number of joints'] = num_joints

    # expand downhole list
    rows = []
    cur_md = 0.0
    for _, r in ds_df.iterrows():
        n = int(r['number of joints'])
        for _ in range(n):
            top = cur_md
            bot = min(cur_md + joint_length, total_length)
            cur_md = bot
            rows.append([bot, r['OD (in)'], r['ID (in)'], r['Wt (lb/ft)'], r['YP (kpsi)'], r['component']])
            if cur_md >= total_length: break
        if cur_md >= total_length: break

    df = pd.DataFrame(rows, columns=['MD', 'OD', 'ID', 'Wt', 'YP', 'component'])
    return df

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_well_profile(hole_df, ds_df):
    """Three-pane: (1) drill string profile, (2) 2D well profile, (3) 3D well profile."""
    has_wall = 'WALL' in hole_df.columns
    has_hole_id = 'HOLE_ID' in hole_df.columns

    hole_df = hole_df.copy()
    hole_df['ds'] = np.sqrt(hole_df['EAST']**2 + hole_df['NORTH']**2)

    hover_cols = ['MD', 'TVD', 'INC', 'AZIM', 'DLS']
    hover_text = (
        'MD: %{customdata[0]:.1f} ft<br>'
        'TVD: %{customdata[1]:.1f} ft<br>'
        'INC: %{customdata[2]:.1f}°<br>'
        'AZIM: %{customdata[3]:.1f}°<br>'
        'DLS: %{customdata[4]:.3f}°/100ft'
    )

    if has_hole_id:
        hover_cols.append('HOLE_ID'); hover_text += f'<br>HOLE ID: %{{customdata[{len(hover_cols)-1}]:.2f}} in'
    if has_wall:
        hover_cols.append('WALL');    hover_text += f'<br>WALL: %{{customdata[{len(hover_cols)-1}]}}'

    colors = {'Open': 'blue', 'Steel': 'gray'}

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'scatter3d'}]],
        column_widths=[0.25, 0.35, 0.40],
        horizontal_spacing=0.05,
        subplot_titles=("Drill String Profile", "2D Well Profile", "3D Well Profile")
    )

    # (1) drill string profile (compact left pane)
    scale_factor = 2.0
    colors_cycle = ['#999', '#666', '#444']
    color_map = {comp: colors_cycle[i % len(colors_cycle)] for i, comp in enumerate(ds_df['component'].unique())}

    for comp, group in ds_df.groupby('component'):
        fig.add_trace(
            go.Scatter(
                x=[0]*len(group),
                y=group['MD'],
                mode='lines',
                line=dict(width=group['OD'].mean()*scale_factor, color=color_map[comp], shape='hv'),
                hovertemplate=(f"Component: {comp}<br>"
                               "MD: %{y:.1f} ft<br>"
                               "OD: %{customdata[0]:.2f} in<br>"
                               "ID: %{customdata[1]:.2f} in<br>"
                               "Wt: %{customdata[2]:.1f} lb/ft<br>"
                               "YP: %{customdata[3]:.1f} kpsi"),
                customdata=group[['OD','ID','Wt','YP']],
                name=comp
            ),
            row=1, col=1
        )
    fig.update_yaxes(title='Measured Depth (ft)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title='', visible=False, row=1, col=1)

    # (2) 2D well profile
    col_idx = 2
    if has_wall:
        for wall, group in hole_df.groupby('WALL'):
            fig.add_trace(
                go.Scatter(x=group['ds'], y=group['TVD'], mode='lines',
                           line=dict(color=colors.get(wall,'blue')),
                           name=f"{wall}", showlegend=False),
                row=1, col=col_idx
            )
    else:
        fig.add_trace(go.Scatter(x=hole_df['ds'], y=hole_df['TVD'], mode='lines', name='Wellbore', showlegend=False), row=1, col=col_idx)
    fig.update_yaxes(title='TVD (ft)', autorange='reversed', row=1, col=col_idx)
    fig.update_xaxes(title='Distance (ft)', row=1, col=col_idx)

    # (3) 3D well profile
    fig.add_trace(
        go.Scatter3d(
            x=hole_df['EAST'], y=hole_df['NORTH'], z=hole_df['TVD'],
            mode='lines', line=dict(width=6), name='Wellpath', showlegend=False
        ),
        row=1, col=3
    )
    fig.update_scenes(zaxis=dict(autorange='reversed'), row=1, col=3)
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def plot_string_profile(ds_df, title="Drill String Profile"):
    """Standalone vertical drill string profile (single tall stick)."""
    import plotly.graph_objects as go
    if ds_df is None or len(ds_df) == 0:
        fig = go.Figure(); fig.update_layout(title=title); return fig

    comps_order = ['DP', 'HWDP', 'DC']
    comp_colors = {'DP': '#a0a0a0', 'HWDP': '#7a7a7a', 'DC': '#555555'}
    md_top = float(ds_df['MD'].min()); md_bot = float(ds_df['MD'].max())
    fig = go.Figure()
    for comp in comps_order:
        if comp not in ds_df['component'].unique(): continue
        g = ds_df[ds_df['component'] == comp]
        y0 = float(g['MD'].min()); y1 = float(g['MD'].max())
        width = max(1.0, float(g['OD'].mean()) * 6.0)  # why: make thickness visually clear
        fig.add_trace(go.Scatter(
            x=[0.0, 0.0], y=[y0, y1], mode='lines',
            line=dict(width=width, color=comp_colors.get(comp, '#888')),
            name=comp,
            hovertemplate=("Component: %s<br>Top MD: %%{y[0]:.0f} ft<br>Bottom MD: %%{y[1]:.0f} ft<br>OD(avg): %.2f in<extra></extra>"
                           % (comp, float(g['OD'].mean())))
        ))
    fig.update_layout(title=title, showlegend=False, height=800, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title="Measured Depth (ft)", autorange='reversed', range=[md_bot, md_top])
    fig.update_xaxes(visible=False)
    return fig
