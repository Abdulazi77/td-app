from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=1, shared_yaxes=True,
    row_heights=[0.34, 0.33, 0.33],
    vertical_spacing=0.02,
    subplot_titles=("Hookload", "Torque (μ sweep)", "Buckling & Severity")
)

# Row 1: Hookload
fig.add_trace(go.Scatter(x=np.maximum(0.0, -df_itr['T_next_lbf'])/1000.0, y=depth,
                         name="Hookload (k-lbf)", mode="lines"), row=1, col=1)
fig.add_vline(x=rig_pull_lim/1000.0, line_dash="dot", annotation_text="Rig pull", row=1, col=1)

# Row 2: Torque μ-bands
for mu in mu_band:
    dmu, tmu = run_td_off_bottom(mu, mu)
    fig.add_trace(go.Scatter(x=tmu/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"), row=2, col=1)
fig.add_vline(x=T_makeup_sf/1000.0, line_dash="dash", annotation_text="MU/SF", row=2, col=1)
fig.add_vline(x=rig_torque_lim/1000.0, line_dash="dot", annotation_text="TD limit", row=2, col=1)

# Row 3: Buckling & severity
fig.add_trace(go.Scatter(x=Fs/1000.0, y=depth, name="Fs (k-lbf)", line=dict(dash="dash")), row=3, col=1)
fig.add_trace(go.Scatter(x=Fh/1000.0, y=depth, name="Fh (k-lbf)", line=dict(dash="dot")),  row=3, col=1)
fig.add_trace(go.Scatter(x=df_itr['N_lbf']/1000.0, y=depth, name="Side-force (k-lbf)"),     row=3, col=1)
fig.add_trace(go.Scatter(x=sigma_b_psi/1000.0, y=depth, name="Bending (ksi)"),             row=3, col=1)
fig.add_trace(go.Scatter(x=sigma_vm_psi/1000.0, y=depth, name="von Mises (ksi)"),          row=3, col=1)
fig.add_trace(go.Scatter(x=BSI, y=depth, name="BSI (1–4)", line=dict(width=4)),            row=3, col=1)
 
# Axis labels, formatting
for r in (1,2,3):
    fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=r, col=1)
fig.update_xaxes(title_text="k-lbf",  row=1, col=1)
fig.update_xaxes(title_text="k lbf-ft", row=2, col=1)
fig.update_xaxes(title_text="k-lbf / ksi / BSI", row=3, col=1)

fig.update_layout(
    height=900, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig, use_container_width=True, key="stacked-main")
