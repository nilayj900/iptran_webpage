#!/usr/bin/env python3
"""
moc_ui.py  —  CNG Pipeline MOC  |  Streamlit Dashboard
=======================================================

Run with:
    streamlit run moc_ui.py

Tabs
----
  1. Simulation   — upload upstream PT data, set calibration, run MOC, view P_dn
  2. Optimisation — run diameter optimisation against downstream field data
  3. Results      — download diameter profile, export plots

Author : Bharat Flow Analytics
Date   : 2026-02-20
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import io, time

from moc_engine      import (run_cng_moc_v5, run_cng_moc_v4,
                              identify_valve_Cv,
                              LinearizedCNG, CoolPropCNG,
                              COOLPROP_AVAILABLE)
from moc_optimization import (run_diameter_optimisation, plot_comparison,
                               save_diameter_profile)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CNG Pipeline MOC",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    code, .stCode              { font-family: 'IBM Plex Mono', monospace; }

    /* Header bar */
    .header-bar {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.4rem 2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .header-bar h1 { color: #e8f4f8; margin: 0; font-size: 1.6rem; font-weight: 600; }
    .header-bar p  { color: #90b8c8; margin: 0.2rem 0 0; font-size: 0.85rem; }

    /* Metric cards */
    .metric-card {
        background: #1a2b38;
        border: 1px solid #2d4a5e;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { color: #7fb3c8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .value { color: #e8f4f8; font-size: 1.5rem; font-weight: 600; font-family: 'IBM Plex Mono'; }
    .metric-card .unit  { color: #90b8c8; font-size: 0.8rem; }

    /* Section headings */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #5ba3c9;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #2d4a5e;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }

    /* Status badges */
    .badge-ok  { background:#1a3d2b; color:#4caf7d; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-warn{ background:#3d2e1a; color:#f0a84c; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-err { background:#3d1a1a; color:#e05555; padding:2px 10px; border-radius:12px; font-size:0.8rem; }

    div[data-testid="stExpander"] { border: 1px solid #2d4a5e; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label, value, unit=''):
    return (f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="unit">{unit}</div>'
            f'</div>')


def load_csv_pressure(uploaded_file, time_col='time', pressure_col='pressure_bar'):
    """
    Load a pressure CSV.  Accepts:
      - two named columns (time_col, pressure_col)
      - two unnamed columns → treated as [time, pressure]
      - single unnamed column → treated as pressures at 100 Hz
    Returns (t_array, p_array)
    """
    df = pd.read_csv(uploaded_file)
    if time_col in df.columns and pressure_col in df.columns:
        return df[time_col].values, df[pressure_col].values
    elif df.shape[1] == 2:
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    elif df.shape[1] == 1:
        p = df.iloc[:, 0].values
        t = np.arange(len(p)) * 0.01   # assume 100 Hz
        return t, p
    else:
        raise ValueError(
            f"Cannot parse CSV with {df.shape[1]} columns. "
            "Expected: [time, pressure_bar] or single pressure column."
        )


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def moc_plot(results, t_field=None, p_field=None, bc_mode=None):
    """
    2×2 matplotlib figure (or 2×3 when elevation profile is active):
      Row 1: Pressure time series | Error panel
      Row 2: Flow time series     | Diameter profile  [| Elevation profile]

    bc_mode : 'case1' or 'case2' (read from results dict if not supplied)
    """
    if bc_mode is None:
        bc_mode = results.get('bc_mode', 'case1')

    elev_mode = results.get('elevation_mode', 'flat')
    z_nodes   = results.get('z_nodes', None)
    show_elev = (elev_mode == 'profile' and z_nodes is not None
                 and np.any(z_nodes != 0))

    # Label logic based on BC mode
    if bc_mode == 'case2':
        prescribed_key  = 'P_dn'
        simulated_key   = 'P_up'
        prescribed_lbl  = 'P_dn (prescribed / field input)'
        simulated_lbl   = 'P_up (simulated)'
        field_lbl       = 'P_up (field)'
        error_title_pfx = 'P_up'
        no_field_msg    = 'Upload upstream\nfield data to see error'
    else:
        prescribed_key  = 'P_up'
        simulated_key   = 'P_dn'
        prescribed_lbl  = 'P_up (prescribed / field input)'
        simulated_lbl   = 'P_dn (simulated)'
        field_lbl       = 'P_dn (field)'
        error_title_pfx = 'P_dn'
        no_field_msg    = 'Upload downstream\nfield data to see error'

    ncols = 3 if show_elev else 2
    fig, axes = plt.subplots(2, ncols,
                             figsize=(7 * ncols, 7),
                             facecolor='#0d1b24',
                             constrained_layout=True)

    for ax in axes.flat:
        ax.set_facecolor('#0d1b24')
        ax.tick_params(colors='#90b8c8', labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#2d4a5e')
        ax.grid(alpha=0.15, color='#2d4a5e')

    # ── Pressures ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(results['time'], results[prescribed_key], lw=1.5,
            color='#5ba3c9', label=prescribed_lbl)
    ax.plot(results['time'], results[simulated_key],  lw=1.5,
            color='#e07b4f', label=simulated_lbl)
    if t_field is not None and p_field is not None:
        ax.plot(t_field, p_field, 'o', ms=1.2, alpha=0.5,
                color='#4caf7d', label=field_lbl)
    ax.set_ylabel('Pressure (bar)', color='#90b8c8')
    ax.set_xlabel('Time (s)', color='#90b8c8')
    case_tag  = 'Case 2' if bc_mode == 'case2' else 'Case 1'
    elev_tag  = ' | Elevation profile' if show_elev else ' | Flat pipe'
    ax.set_title(f'Pressure Time Series ({case_tag}{elev_tag})',
                 color='#e8f4f8', fontsize=11)
    leg = ax.legend(fontsize=8, facecolor='#1a2b38', edgecolor='#2d4a5e')
    for t in leg.get_texts(): t.set_color('#90b8c8')

    # ── Error ─────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if t_field is not None and p_field is not None:
        p_sim_i = np.interp(t_field, results['time'], results[simulated_key])
        err     = (p_sim_i - p_field) * 1000
        ax.plot(t_field, err, lw=0.8, color='#e05555', alpha=0.9)
        ax.fill_between(t_field, err, alpha=0.2, color='#e05555')
        ax.axhline(0, color='#2d4a5e', lw=0.8, ls='--')
        rmse = np.sqrt(np.mean(err**2))
        ax.set_title(f'{error_title_pfx} Error   RMSE={rmse:.2f} mbar',
                     color='#e8f4f8', fontsize=11)
        ax.set_ylabel('Error (mbar)', color='#90b8c8')
        ax.set_xlabel('Time (s)', color='#90b8c8')
    else:
        ax.text(0.5, 0.5, no_field_msg,
                ha='center', va='center', color='#5ba3c9',
                transform=ax.transAxes, fontsize=10)
        ax.set_title(f'{error_title_pfx} Error', color='#e8f4f8', fontsize=11)

    # ── Flows ─────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(results['time'], results['Q_up'], lw=1.5, color='#5ba3c9', label='Q_up')
    ax.plot(results['time'], results['Q_dn'], lw=1.5, color='#e07b4f', label='Q_dn')
    ax.set_ylabel('Flow (Scmh)', color='#90b8c8')
    ax.set_xlabel('Time (s)', color='#90b8c8')
    ax.set_title('Flow Time Series', color='#e8f4f8', fontsize=11)
    leg = ax.legend(fontsize=8, facecolor='#1a2b38', edgecolor='#2d4a5e')
    for t in leg.get_texts(): t.set_color('#90b8c8')

    # ── Diameter profile ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(results['x_segments'], results['D_segments'] * 1000,
            '-o', lw=2, ms=4, color='#f0a84c')
    ax.set_ylabel('Diameter (mm)', color='#90b8c8')
    ax.set_xlabel('Position (m)', color='#90b8c8')
    ax.set_title('Diameter Profile', color='#e8f4f8', fontsize=11)

    # ── Elevation profile (only when active) ──────────────────────────────────
    if show_elev:
        ax_top  = axes[0, 2]
        ax_bot  = axes[1, 2]
        x_nodes = results['x_nodes']

        # Top: elevation terrain
        ax_top.plot(x_nodes, z_nodes, lw=2, color='#4caf7d', label='Elevation z(x)')
        ax_top.fill_between(x_nodes, z_nodes, alpha=0.18, color='#4caf7d')
        ax_top.set_ylabel('Elevation (m)', color='#90b8c8')
        ax_top.set_xlabel('Distance (m)', color='#90b8c8')
        ax_top.set_title('Pipe Elevation Profile', color='#e8f4f8', fontsize=11)
        leg = ax_top.legend(fontsize=8, facecolor='#1a2b38', edgecolor='#2d4a5e')
        for t in leg.get_texts(): t.set_color('#90b8c8')

        # Bottom: hydraulic grade line at t=0 (initial piezometric head)
        # P_up and P_dn at t=0, interpolate linearly across pipe for illustration
        P_up0 = results['P_up'][0]
        P_dn0 = results['P_dn'][0]
        p_linear = P_up0 + (P_dn0 - P_up0) * (x_nodes / x_nodes[-1])
        # Convert to pressure head (bar) and add elevation for piezometric head line
        ax_bot.plot(x_nodes, p_linear, lw=1.5, color='#5ba3c9',
                    label='Approx. pressure (bar)')
        ax_bot.set_ylabel('Pressure (bar)', color='#90b8c8')
        ax_bot2 = ax_bot.twinx()
        ax_bot2.plot(x_nodes, z_nodes, lw=1.2, color='#4caf7d',
                     ls='--', label='Elevation (m)')
        ax_bot2.set_ylabel('Elevation (m)', color='#4caf7d')
        ax_bot2.tick_params(colors='#4caf7d', labelsize=9)
        ax_bot.set_xlabel('Distance (m)', color='#90b8c8')
        ax_bot.set_title('Pressure + Elevation along Pipe', color='#e8f4f8', fontsize=11)
        lines1, labels1 = ax_bot.get_legend_handles_labels()
        lines2, labels2 = ax_bot2.get_legend_handles_labels()
        leg = ax_bot.legend(lines1 + lines2, labels1 + labels2,
                            fontsize=8, facecolor='#1a2b38', edgecolor='#2d4a5e')
        for t in leg.get_texts(): t.set_color('#90b8c8')
        ax_bot2.set_facecolor('#0d1b24')
        for sp in ax_bot2.spines.values(): sp.set_color('#2d4a5e')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

for key, default in {
    'sim_results' : None,
    'opt_results' : None,
    't_up'        : None,
    'p_up'        : None,
    't_dn_field'  : None,
    'p_dn_field'  : None,
    'Cv'          : None,
    'Ca'          : None,
    'bc_mode'     : 'case1',
    'elev_mode'   : 'flat',
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-bar">
  <h1>🔧 CNG Pipeline — MOC Analyser</h1>
  <p>Field Calibration Mode  ·  Segmented Diameter Optimisation  ·  v5.0 — CoolProp Real Gas</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — pipe & gas parameters
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown('<div class="section-title">Pipe Geometry</div>', unsafe_allow_html=True)
    L        = st.number_input("Pipeline length L (m)",      value=2100.0, step=100.0)
    dx       = st.number_input("Segment length dx (m)",      value=100.0,  step=10.0)
    D_init   = st.number_input("Initial diameter (mm)",      value=254.0,  step=1.0)
    eps_mm   = st.number_input("Roughness ε (mm)",           value=0.36,   step=0.01, format="%.3f")

    st.markdown('<div class="section-title" style="margin-top:1rem">Gas Properties</div>',
                unsafe_allow_html=True)
    T_op     = st.number_input("Temperature (°C)",           value=30.0,   step=1.0)
    P_op     = st.number_input("Operating pressure (bar)",   value=13.0,   step=0.5)
    P_init   = st.number_input("Initial pressure (bar)",     value=13.65,  step=0.05)
    Q_init   = st.number_input("Initial flow (Scmh)",        value=600.0,  step=50.0)
    f_tune   = st.slider("Friction tuning factor",           0.5, 1.5, 0.9, 0.05)

    st.markdown('<div class="section-title" style="margin-top:1rem">Gas Model</div>',
                unsafe_allow_html=True)
    if COOLPROP_AVAILABLE:
        use_coolprop = st.toggle("Use CoolProp real-gas model", value=False,
                                  help="Enables real-gas properties via CoolProp (~5–15% "
                                       "better accuracy). Adds ~5–10 s table build time.")
        if use_coolprop:
            cp_fluid   = st.selectbox("CoolProp fluid",
                                      ['Methane', 'HEOS::Methane', 'CNG', 'NaturalGas'],
                                      index=0)
            cp_npoints = st.slider("Table grid resolution", 10, 50, 25, 5,
                                   help="Higher = more accurate but slower at startup.")
        else:
            cp_fluid   = 'Methane'
            cp_npoints = 25
    else:
        use_coolprop = False
        cp_fluid     = 'Methane'
        cp_npoints   = 25
        st.info("CoolProp not installed. Using linearised gas model.\n\n"
                "`pip install coolprop`  to enable real-gas mode.")

    st.markdown('<div class="section-title" style="margin-top:1rem">Downstream Valve</div>',
                unsafe_allow_html=True)
    tau_cal  = st.slider("τ at calibration (opening fraction)", 0.01, 1.0, 1.0, 0.01)
    Q_cal    = st.number_input("Q at calibration (Scmh)",    value=600.0,  step=50.0)
    P_cal    = st.number_input("P_dn at calibration (bar)",  value=13.20,  step=0.05)

    st.markdown('<div class="section-title" style="margin-top:1rem">Boundary Condition Mode</div>',
                unsafe_allow_html=True)
    bc_mode = st.radio(
        "Select Case",
        options=['case1', 'case2'],
        format_func=lambda x: (
            "Case 1 — Upstream transient (P_up prescribed, downstream valve open)"
            if x == 'case1' else
            "Case 2 — Downstream transient (P_dn prescribed, upstream valve open)"
        ),
        index=0,
        help=(
            "Case 1: Upstream pressure time-series drives the simulation. "
            "Downstream valve is fully open (Cv from calibration). "
            "Simulates transients originating at the upstream end.\n\n"
            "Case 2: Downstream pressure time-series drives the simulation. "
            "Upstream valve is fully open (Cv from calibration). "
            "Simulates transients originating at the downstream end."
        )
    )
    st.session_state['bc_mode'] = bc_mode

    if bc_mode == 'case2':
        st.markdown('<div class="section-title" style="margin-top:0.5rem">Upstream Valve (Case 2)</div>',
                    unsafe_allow_html=True)
        P_up_cal = st.number_input("P_up at calibration (bar)", value=13.65, step=0.05,
                                   help="Upstream pressure at the calibration instant "
                                        "used to identify the upstream valve Cv.")
    else:
        P_up_cal = None

    # ── Elevation ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1rem">Elevation Profile</div>',
                unsafe_allow_html=True)
    elev_mode = st.radio(
        "Elevation case",
        options=['flat', 'profile'],
        format_func=lambda x: (
            "Case 1 — Flat pipe (no elevation correction)"
            if x == 'flat' else
            "Case 2 — With elevation profile"
        ),
        index=0,
        help=(
            "Case 1 (Flat): No elevation term in MOC equations. "
            "H[i] = pressure head only. Default behaviour.\n\n"
            "Case 2 (Profile): Pipe elevation z(x) included as piezometric head. "
            "MOC characteristics carry the gravity head change between nodes. "
            "Upload a CSV with columns [distance_m, elevation_m] or use the "
            "built-in dummy profile."
        )
    )
    st.session_state['elev_mode'] = elev_mode

    elev_data_array = None   # default: None → flat zeros inside engine

    if elev_mode == 'profile':
        st.markdown(
            '<div style="font-size:0.78rem;color:#90b8c8;margin-bottom:4px">'
            'Upload elevation CSV  <span style="color:#5ba3c9">'
            '[distance_m, elevation_m]</span> or use dummy profile below.</div>',
            unsafe_allow_html=True
        )
        elev_file = st.file_uploader(
            "Elevation CSV (optional)",
            type=['csv'], key='elev_file',
            help="Two columns: distance_m (0 → L) and elevation_m. "
                 "Values are interpolated to MOC node positions."
        )

        if elev_file is not None:
            try:
                elev_df = pd.read_csv(elev_file, header=None)
                elev_df.columns = ['distance_m', 'elevation_m']
                elev_data_array = elev_df.values   # (M,2) array
                st.markdown(
                    f'<span class="badge-ok">✓ {len(elev_df)} elevation points  '
                    f'{elev_df["elevation_m"].min():.1f}–'
                    f'{elev_df["elevation_m"].max():.1f} m</span>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Could not load elevation file: {e}")
                elev_data_array = None

        else:
            # ── Dummy elevation profile ─────────────────────────────────────
            # Represents a realistic CNG trunk pipeline:
            #   0–500 m   : gentle rise out of compressor station (+15 m)
            #   500–900 m : plateau at ~+15 m
            #   900–1400 m: ridge crossing — rises to +38 m peak at 1200 m
            #   1400–1800 m: descent back down to +12 m
            #   1800–2100 m: slight dip into receiving station at +5 m
            _x_elev = np.array([   0,  300,  500,  700,  900,
                                 1100, 1200, 1400, 1600, 1800,
                                 2000, 2100], dtype=float)
            _z_elev = np.array([   0,   8,   15,   16,   18,
                                   30,   38,   28,   18,   12,
                                    7,    5], dtype=float)
            elev_data_array = np.column_stack([_x_elev, _z_elev])
            st.caption(
                "Using built-in dummy profile: gentle rise → ridge at 1200 m "
                "(+38 m peak) → descent to receiving station (+5 m)."
            )
            # Mini preview plot
            fig_elev, ax_elev = plt.subplots(figsize=(5, 1.8), facecolor='#0d1b24')
            ax_elev.set_facecolor('#0d1b24')
            ax_elev.plot(_x_elev, _z_elev, '-o', lw=1.5, ms=3, color='#4caf7d')
            ax_elev.fill_between(_x_elev, _z_elev, alpha=0.15, color='#4caf7d')
            ax_elev.set_xlabel('Distance (m)', color='#90b8c8', fontsize=7)
            ax_elev.set_ylabel('Elev (m)', color='#90b8c8', fontsize=7)
            ax_elev.tick_params(colors='#90b8c8', labelsize=6)
            for sp in ax_elev.spines.values(): sp.set_color('#2d4a5e')
            st.pyplot(fig_elev, use_container_width=True)
            plt.close(fig_elev)

    # Live Cv preview
    if st.button("Preview Cv"):
        try:
            if use_coolprop and COOLPROP_AVAILABLE:
                T_min_K = T_op - 10 + 273.15
                T_max_K = T_op + 10 + 273.15
                P_min_Pa = max((P_op - 3) * 1e5, 1e5)
                P_max_Pa = (P_op + 3) * 1e5
                gas_prev = CoolPropCNG(
                    T_range=(T_min_K, T_max_K),
                    P_range=(P_min_Pa, P_max_Pa),
                    n_points=cp_npoints,
                    fluid=cp_fluid,
                    silent=True,
                )
            else:
                gas_prev = LinearizedCNG(T_celsius=T_op, P_op_bar=P_op, silent=True)
            D_last   = D_init * 1e-3
            cv, ca, _ = identify_valve_Cv(Q_cal, P_cal, gas_prev,
                                          D_last, tau_cal, verbose=False)
            st.session_state['Cv'] = cv
            st.session_state['Ca'] = ca
            st.success(f"Cv = {cv:.4e} m³/s")
            st.info(f"Ca = g·A/a = {ca:.4e} m²/s")
        except Exception as e:
            st.error(f"Cv error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_sim, tab_opt, tab_results = st.tabs([
    "▶  Simulation",
    "⚙  Optimisation",
    "📊  Results & Export",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_sim:

    # ── BC mode banner ────────────────────────────────────────────────────────
    bc_mode_now = st.session_state.get('bc_mode', 'case1')
    if bc_mode_now == 'case1':
        st.info("**Case 1 — Upstream transient:** Upload upstream P(t). "
                "Downstream valve fully open. Simulated output: P_dn(t).")
    else:
        st.info("**Case 2 — Downstream transient:** Upload downstream P(t). "
                "Upstream valve fully open. Simulated output: P_up(t).")

    col_up, col_dn = st.columns(2)

    with col_up:
        if bc_mode_now == 'case1':
            st.markdown('<div class="section-title">Upstream PT Data — prescribed P(t)</div>',
                        unsafe_allow_html=True)
            up_file = st.file_uploader(
                "Upload upstream pressure CSV",
                type=['csv'], key='up_file',
                help="Columns: [time, pressure_bar]  or  single pressure column at 100 Hz"
            )
        else:
            st.markdown('<div class="section-title">Upstream PT Data (optional — comparison only)</div>',
                        unsafe_allow_html=True)
            up_file = st.file_uploader(
                "Upload upstream pressure CSV (comparison only)",
                type=['csv'], key='up_file',
                help="Not used as BC in Case 2. Upload to overlay on results plot."
            )

        dt_field = st.number_input("Field sampling interval (s)", value=0.01,
                                   format="%.4f", step=0.001)

        if up_file:
            try:
                t_up, p_up = load_csv_pressure(up_file)
                st.session_state['t_up'] = t_up
                st.session_state['p_up'] = p_up
                st.markdown(
                    f'<span class="badge-ok">✓ {len(p_up)} samples  '
                    f'{t_up[0]:.2f}–{t_up[-1]:.2f} s</span>',
                    unsafe_allow_html=True
                )
                fig_prev, ax_prev = plt.subplots(figsize=(5, 2), facecolor='#0d1b24')
                ax_prev.set_facecolor('#0d1b24')
                ax_prev.plot(t_up, p_up, lw=0.8, color='#5ba3c9')
                ax_prev.tick_params(colors='#90b8c8', labelsize=7)
                for sp in ax_prev.spines.values(): sp.set_color('#2d4a5e')
                ax_prev.set_ylabel('bar', color='#90b8c8', fontsize=8)
                st.pyplot(fig_prev, use_container_width=True)
                plt.close(fig_prev)
            except Exception as e:
                st.error(f"Could not load file: {e}")

    with col_dn:
        if bc_mode_now == 'case2':
            st.markdown('<div class="section-title">Downstream PT Data — prescribed P(t)</div>',
                        unsafe_allow_html=True)
            dn_file = st.file_uploader(
                "Upload downstream pressure CSV",
                type=['csv'], key='dn_file',
                help="Columns: [time, pressure_bar]  or  single pressure column at 100 Hz"
            )
        else:
            st.markdown('<div class="section-title">Downstream PT Data (optional — comparison only)</div>',
                        unsafe_allow_html=True)
            dn_file = st.file_uploader(
                "Upload downstream pressure CSV (comparison only)",
                type=['csv'], key='dn_file',
                help="Used for error comparison and optimisation"
            )

        if dn_file:
            try:
                t_dn, p_dn = load_csv_pressure(dn_file)
                st.session_state['t_dn_field'] = t_dn
                st.session_state['p_dn_field'] = p_dn
                st.markdown(
                    f'<span class="badge-ok">✓ {len(p_dn)} samples  '
                    f'{t_dn[0]:.2f}–{t_dn[-1]:.2f} s</span>',
                    unsafe_allow_html=True
                )
                fig_prev2, ax_prev2 = plt.subplots(figsize=(5, 2), facecolor='#0d1b24')
                ax_prev2.set_facecolor('#0d1b24')
                ax_prev2.plot(t_dn, p_dn, lw=0.8, color='#e07b4f')
                ax_prev2.tick_params(colors='#90b8c8', labelsize=7)
                for sp in ax_prev2.spines.values(): sp.set_color('#2d4a5e')
                ax_prev2.set_ylabel('bar', color='#90b8c8', fontsize=8)
                st.pyplot(fig_prev2, use_container_width=True)
                plt.close(fig_prev2)
            except Exception as e:
                st.error(f"Could not load file: {e}")

    st.divider()

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_sim = st.button("▶  Run Simulation", type='primary', use_container_width=True)

    if run_sim:
        bc = st.session_state.get('bc_mode', 'case1')
        p_up_data = st.session_state.get('p_up')
        p_dn_data = st.session_state.get('p_dn_field')

        if bc == 'case1' and p_up_data is None:
            st.error("Case 1: Please upload upstream PT data first.")
        elif bc == 'case2' and p_dn_data is None:
            st.error("Case 2: Please upload downstream PT data first.")
        else:
            with st.spinner("Running MOC simulation…"):
                t0 = time.time()
                try:
                    results = run_cng_moc_v5(
                        L                        = L,
                        dx                       = dx,
                        D                        = D_init * 1e-3,
                        eps                      = eps_mm * 1e-3,
                        T_celsius                = T_op,
                        P_operating_bar          = P_op,
                        use_coolprop             = use_coolprop,
                        coolprop_fluid           = cp_fluid,
                        coolprop_n_points        = cp_npoints,
                        P_initial_bar            = P_init,
                        Q_initial_scmh           = Q_init,
                        bc_mode                  = bc,
                        upstream_pressure_data   = p_up_data if bc == 'case1' else None,
                        downstream_pressure_data = p_dn_data if bc == 'case2' else None,
                        dt_field                 = dt_field,
                        Q_calib_scmh             = Q_cal,
                        P_dn_calib_bar           = P_cal,
                        P_up_calib_bar           = P_up_cal,
                        tau_calib                = tau_cal,
                        T_total                  = None,
                        friction_tuning          = f_tune,
                        elevation_mode           = elev_mode,
                        elevation_data           = elev_data_array,
                        verbose                  = False,
                    )
                    st.session_state['sim_results'] = results
                    elapsed = time.time() - t0
                    gas_label  = results.get('gas_model_type', 'LinearizedCNG')
                    bc_label   = results.get('bc_mode', bc)
                    elev_label = results.get('elevation_mode', 'flat')
                    st.success(
                        f"Simulation complete in {elapsed:.2f} s  "
                        f"({gas_label}, {bc_label}, elevation={elev_label})"
                    )
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    st.exception(e)

    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        bc_res = res.get('bc_mode', 'case1')

        # ── Metrics row ───────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        sim_out = res['P_up'] if bc_res == 'case2' else res['P_dn']
        sim_label = 'P_up' if bc_res == 'case2' else 'P_dn'
        with m1:
            st.markdown(metric_card(f"{sim_label} min", f"{sim_out.min():.3f}", "bar"),
                        unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card(f"{sim_label} max", f"{sim_out.max():.3f}", "bar"),
                        unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("Cv", f"{res['Cv']:.3e}", "m³/s"),
                        unsafe_allow_html=True)
        with m4:
            N_seg = len(res['D_segments'])
            st.markdown(metric_card("Segments", str(N_seg), ""),
                        unsafe_allow_html=True)

        st.markdown("")

        # ── Main plot ─────────────────────────────────────────────────────────
        # In Case 2 the "field" comparison series is P_up (upstream), not P_dn
        if bc_res == 'case2':
            t_field_plot = st.session_state.get('t_up')
            p_field_plot = st.session_state.get('p_up')
        else:
            t_field_plot = st.session_state.get('t_dn_field')
            p_field_plot = st.session_state.get('p_dn_field')

        fig_main = moc_plot(
            res,
            t_field=t_field_plot,
            p_field=p_field_plot,
            bc_mode=bc_res,
        )
        st.pyplot(fig_main, use_container_width=True)
        plt.close(fig_main)

        # ── Error metrics — compare simulated output vs the "other" field PT ──
        # Case 1: simulated P_dn vs uploaded downstream field
        # Case 2: simulated P_up vs uploaded upstream field
        if bc_res == 'case2':
            t_cmp = st.session_state.get('t_up')
            p_cmp = st.session_state.get('p_up')
            cmp_lbl = 'P_up'
        else:
            t_cmp = st.session_state.get('t_dn_field')
            p_cmp = st.session_state.get('p_dn_field')
            cmp_lbl = 'P_dn'

        if t_cmp is not None and p_cmp is not None:
            sim_key  = 'P_up' if bc_res == 'case2' else 'P_dn'
            p_sim_i  = np.interp(t_cmp, res['time'], res[sim_key])
            err      = p_sim_i - p_cmp
            rmse     = np.sqrt(np.mean(err**2)) * 1000
            mae      = np.mean(np.abs(err)) * 1000
            bias     = err.mean() * 1000

            e1, e2, e3 = st.columns(3)
            with e1:
                st.markdown(metric_card(f"{cmp_lbl} RMSE", f"{rmse:.3f}", "mbar"),
                            unsafe_allow_html=True)
            with e2:
                st.markdown(metric_card("MAE",  f"{mae:.3f}", "mbar"),
                            unsafe_allow_html=True)
            with e3:
                st.markdown(metric_card("Bias", f"{bias:.3f}", "mbar"),
                            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_opt:

    st.markdown('<div class="section-title">Optimisation Settings</div>',
                unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)

    with oc1:
        D_min  = st.number_input("D min (mm)", value=150.0, step=10.0)
        D_max  = st.number_input("D max (mm)", value=350.0, step=10.0)

    with oc2:
        method = st.selectbox("Method",
                              ['differential_evolution', 'L-BFGS-B'])
        maxiter = st.number_input("Max iterations", value=200, step=50)

    with oc3:
        popsize = st.number_input("Population size (DE only)", value=12, step=2)
        seed    = st.number_input("Random seed", value=42, step=1)

    st.divider()

    run_opt_col, _ = st.columns([1, 3])
    with run_opt_col:
        run_opt = st.button("⚙  Run Optimisation", type='primary',
                            use_container_width=True)

    if run_opt:
        p_up = st.session_state.get('p_up')
        t_dn = st.session_state.get('t_dn_field')
        p_dn = st.session_state.get('p_dn_field')
        bc   = st.session_state.get('bc_mode', 'case1')

        if p_up is None and bc == 'case1':
            st.error("Upload upstream PT data first (Simulation tab).")
        elif p_dn is None:
            st.error("Upload downstream field PT data first (Simulation tab).")
        else:
            sim_params = dict(
                L                        = L,
                dx                       = dx,
                eps                      = eps_mm * 1e-3,
                T_celsius                = T_op,
                P_operating_bar          = P_op,
                P_initial_bar            = P_init,
                Q_initial_scmh           = Q_init,
                upstream_pressure_data   = p_up if bc == 'case1' else None,
                downstream_pressure_data = p_dn if bc == 'case2' else None,
                dt_field                 = dt_field if 'dt_field' in dir() else 0.01,
                Q_calib_scmh             = Q_cal,
                P_dn_calib_bar           = P_cal,
                P_up_calib_bar           = P_up_cal,
                tau_calib                = tau_cal,
                friction_tuning          = f_tune,
                bc_mode                  = bc,
                elevation_mode           = elev_mode,
                elevation_data           = elev_data_array,
                # CoolProp real-gas options
                use_coolprop             = use_coolprop,
                coolprop_fluid           = cp_fluid,
                coolprop_n_points        = cp_npoints,
            )

            progress_bar = st.progress(0, text="Initialising optimiser…")
            log_box      = st.empty()
            log_lines    = []

            class StreamlitCallback:
                def __init__(self, total):
                    self.total  = total
                    self.count  = 0
                def __call__(self, xk, convergence=None):
                    self.count += 1
                    pct = min(int(100 * self.count / self.total), 99)
                    _sim_key = 'P_up' if bc == 'case2' else 'P_dn'
                    _res1 = run_cng_moc_v5(**{**sim_params, 'D': xk, 'verbose': False})
                    rmse_now = np.sqrt(np.mean(
                        (np.interp(t_dn, _res1['time'], _res1[_sim_key]) - p_dn)**2
                    )) * 1000
                    log_lines.append(
                        f"iter {self.count:4d}  RMSE={rmse_now:.3f} mbar  "
                        f"D: {xk.min()*1000:.1f}–{xk.max()*1000:.1f} mm"
                    )
                    progress_bar.progress(pct,
                        text=f"Iteration {self.count}  RMSE={rmse_now:.3f} mbar")
                    log_box.code('\n'.join(log_lines[-15:]), language='')

            with st.spinner("Optimising diameter profile…"):
                t0 = time.time()
                try:
                    opt = run_diameter_optimisation(
                        sim_params     = sim_params,
                        t_dn_field     = t_dn,
                        p_dn_field     = p_dn,
                        D_min_mm       = D_min,
                        D_max_mm       = D_max,
                        method         = method,
                        maxiter        = int(maxiter),
                        popsize        = int(popsize),
                        seed           = int(seed),
                        D_init_mm      = D_init,
                        callback_every = 1,
                        verbose        = False,
                    )
                    st.session_state['opt_results'] = opt
                    progress_bar.progress(100, text="Optimisation complete!")
                    elapsed = time.time() - t0
                    st.success(
                        f"Done in {elapsed:.1f} s  —  "
                        f"RMSE = {opt['rmse_mbar']:.3f} mbar"
                    )
                except Exception as e:
                    st.error(f"Optimisation failed: {e}")
                    st.exception(e)

    if st.session_state['opt_results'] is not None:
        opt = st.session_state['opt_results']
        res = opt['sim_results']
        t_dn = st.session_state['t_dn_field']
        p_dn = st.session_state['p_dn_field']

        oa, ob, oc_ = st.columns(3)
        with oa:
            st.markdown(metric_card("RMSE (opt)", f"{opt['rmse_mbar']:.3f}", "mbar"),
                        unsafe_allow_html=True)
        with ob:
            st.markdown(metric_card("D range",
                                    f"{opt['D_opt_mm'].min():.1f}–{opt['D_opt_mm'].max():.1f}",
                                    "mm"),
                        unsafe_allow_html=True)
        with oc_:
            st.markdown(metric_card("D mean", f"{opt['D_opt_mm'].mean():.1f}", "mm"),
                        unsafe_allow_html=True)

        st.markdown("")

        fig_opt = moc_plot(res, t_field=t_dn, p_field=p_dn)
        st.pyplot(fig_opt, use_container_width=True)
        plt.close(fig_opt)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESULTS & EXPORT
# ─────────────────────────────────────────────────────────────────────────────

with tab_results:

    st.markdown('<div class="section-title">Download Results</div>',
                unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3)

    # ── Simulated P_dn CSV ────────────────────────────────────────────────────
    with dl1:
        st.markdown("**Simulated pressure trace**")
        res = st.session_state.get('sim_results') or \
              (st.session_state.get('opt_results') or {}).get('sim_results')
        if res is not None:
            df_out = pd.DataFrame({
                'time_s'    : res['time'],
                'P_up_bar'  : res['P_up'],
                'P_dn_bar'  : res['P_dn'],
                'Q_up_scmh' : res['Q_up'],
                'Q_dn_scmh' : res['Q_dn'],
            })
            csv_bytes = df_out.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download P_dn.csv",
                data     = csv_bytes,
                file_name= "simulated_pressures.csv",
                mime     = "text/csv",
                use_container_width=True,
            )
        else:
            st.info("Run a simulation first.")

    # ── Optimised diameter profile CSV ────────────────────────────────────────
    with dl2:
        st.markdown("**Optimised diameter profile**")
        opt = st.session_state.get('opt_results')
        if opt is not None:
            N_seg = len(opt['D_opt_m'])
            dx_r  = L / N_seg
            df_d  = pd.DataFrame({
                'segment'  : np.arange(N_seg),
                'x_mid_m'  : np.arange(N_seg) * dx_r + dx_r / 2,
                'D_opt_mm' : opt['D_opt_mm'],
                'D_opt_m'  : opt['D_opt_m'],
            })
            csv_d = df_d.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download diameter_profile.csv",
                data     = csv_d,
                file_name= "diameter_profile.csv",
                mime     = "text/csv",
                use_container_width=True,
            )
        else:
            st.info("Run optimisation first.")

    # ── Figure PNG ────────────────────────────────────────────────────────────
    with dl3:
        st.markdown("**Results figure (PNG)**")
        res_for_fig = None
        if opt is not None:
            res_for_fig = opt.get('sim_results')
        elif st.session_state['sim_results'] is not None:
            res_for_fig = st.session_state['sim_results']

        if res_for_fig is not None:
            fig_dl = moc_plot(
                res_for_fig,
                t_field=st.session_state.get('t_dn_field'),
                p_field=st.session_state.get('p_dn_field'),
            )
            png_bytes = fig_to_bytes(fig_dl)
            plt.close(fig_dl)
            st.download_button(
                "⬇ Download figure.png",
                data     = png_bytes,
                file_name= "moc_results.png",
                mime     = "image/png",
                use_container_width=True,
            )
        else:
            st.info("Run a simulation first.")

    # ── Convergence history ───────────────────────────────────────────────────
    if opt is not None and len(opt.get('history_rmse', [])) > 1:
        st.markdown('<div class="section-title" style="margin-top:1.5rem">Convergence History</div>',
                    unsafe_allow_html=True)
        fig_conv, ax_conv = plt.subplots(figsize=(10, 3), facecolor='#0d1b24')
        ax_conv.set_facecolor('#0d1b24')
        ax_conv.semilogy(np.arange(1, len(opt['history_rmse']) + 1),
                         np.array(opt['history_rmse']) * 1000,
                         '-o', lw=1.5, ms=4, color='#5ba3c9')
        ax_conv.set_xlabel('Callback iteration', color='#90b8c8')
        ax_conv.set_ylabel('RMSE (mbar)', color='#90b8c8')
        ax_conv.set_title('Optimisation Convergence', color='#e8f4f8')
        ax_conv.tick_params(colors='#90b8c8')
        for sp in ax_conv.spines.values(): sp.set_color('#2d4a5e')
        ax_conv.grid(alpha=0.15, color='#2d4a5e')
        st.pyplot(fig_conv, use_container_width=True)
        plt.close(fig_conv)

    # ── Diameter profile table ────────────────────────────────────────────────
    if opt is not None:
        st.markdown('<div class="section-title" style="margin-top:1.5rem">Segment Diameter Table</div>',
                    unsafe_allow_html=True)
        N_seg  = len(opt['D_opt_m'])
        dx_r   = L / N_seg
        df_tbl = pd.DataFrame({
            'Segment'   : np.arange(N_seg),
            'x_start_m' : np.arange(N_seg) * dx_r,
            'x_end_m'   : np.arange(N_seg) * dx_r + dx_r,
            'D_opt_mm'  : np.round(opt['D_opt_mm'], 2),
        })
        st.dataframe(df_tbl, use_container_width=True, height=300)
