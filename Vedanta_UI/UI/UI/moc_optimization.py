#!/usr/bin/env python3
"""
moc_optimization.py  —  Diameter Profile Optimiser
====================================================

Wraps moc_engine.run_cng_moc_v4 inside a scipy optimiser to find the
per-segment diameter array D[0..N-1] that minimises the RMSE between
simulated and field-measured downstream pressure.

Workflow
--------
  1. Load upstream field PT data  → upstream_pressure_data
  2. Load downstream field PT data → t_dn_field, p_dn_field
  3. Identify valve Cv from one calibration instant
  4. Call run_diameter_optimisation()
  5. Inspect results, plot, save

Author : Bharat Flow Analytics
Date   : 2026-02-20
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d

from moc_engine import (
    run_cng_moc_v5, run_cng_moc_v4,   # v5 = real-gas capable; v4 = legacy alias
    identify_valve_Cv,
    LinearizedCNG, CoolPropCNG,
    COOLPROP_AVAILABLE,
)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def moc_objective(D_flat, sim_params, t_dn_field, p_dn_field):
    """
    Objective function for scipy optimiser.

    sim_params optional keys (in addition to existing):
      'bc_mode'                  : 'case1' (default) or 'case2'
      'downstream_pressure_data' : field P(t) at downstream end (case2)
      'P_up_calib_bar'           : upstream calib pressure (case2)
    """
    try:
        results = run_cng_moc_v5(
            L                        = sim_params['L'],
            dx                       = sim_params['dx'],
            D                        = D_flat,
            eps                      = sim_params['eps'],
            gas_model                = sim_params.get('gas_model'),
            T_celsius                = sim_params['T_celsius'],
            P_operating_bar          = sim_params['P_operating_bar'],
            use_coolprop             = sim_params.get('use_coolprop', False),
            coolprop_fluid           = sim_params.get('coolprop_fluid', 'Methane'),
            coolprop_n_points        = sim_params.get('coolprop_n_points', 25),
            P_initial_bar            = sim_params['P_initial_bar'],
            Q_initial_scmh           = sim_params['Q_initial_scmh'],
            bc_mode                  = sim_params.get('bc_mode', 'case1'),
            upstream_pressure_data   = sim_params.get('upstream_pressure_data'),
            downstream_pressure_data = sim_params.get('downstream_pressure_data'),
            dt_field                 = sim_params['dt_field'],
            Q_calib_scmh             = sim_params.get('Q_calib_scmh'),
            P_dn_calib_bar           = sim_params.get('P_dn_calib_bar'),
            P_up_calib_bar           = sim_params.get('P_up_calib_bar'),
            tau_calib                = sim_params.get('tau_calib', 1.0),
            Cv_known                 = sim_params.get('Cv_known'),
            T_total                  = sim_params.get('T_total'),
            friction_tuning          = sim_params.get('friction_tuning', 0.9),
            elevation_mode           = sim_params.get('elevation_mode', 'flat'),
            elevation_data           = sim_params.get('elevation_data'),
            verbose                  = False,
        )
        # For case2 the "measured" output to match is P_up (simulated upstream)
        if sim_params.get('bc_mode', 'case1') == 'case2':
            p_sim = np.interp(t_dn_field, results['time'], results['P_up'])
        else:
            p_sim = np.interp(t_dn_field, results['time'], results['P_dn'])
        rmse = np.sqrt(np.mean((p_sim - p_dn_field)**2))
        return rmse

    except Exception:
        return 1e6


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMISATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_diameter_optimisation(
        sim_params,
        t_dn_field,
        p_dn_field,
        D_min_mm=150.0,
        D_max_mm=350.0,
        method='differential_evolution',
        maxiter=300,
        popsize=12,
        tol=1e-4,
        seed=42,
        D_init_mm=None,
        callback_every=10,
        verbose=True
):
    """
    Optimise per-segment diameters to match measured downstream pressure.

    Parameters
    ----------
    sim_params : dict
        All fixed inputs passed to run_cng_moc_v5.
        Required keys:
          L, dx, eps, T_celsius, P_operating_bar,
          P_initial_bar, Q_initial_scmh,
          upstream_pressure_data, dt_field,
          AND one of:
            (a)  Q_calib_scmh + P_dn_calib_bar  [+ tau_calib]
            (b)  Cv_known
        Optional keys (CoolProp real-gas model):
          gas_model        : pre-built LinearizedCNG or CoolPropCNG instance
          use_coolprop     : bool   — build CoolPropCNG automatically (default False)
          coolprop_fluid   : str    — CoolProp fluid name  (default 'Methane')
          coolprop_n_points: int    — property table grid resolution (default 25)
        Other optional keys:
          T_total, friction_tuning

    t_dn_field  : 1-D array — time axis of downstream field PT data  (s)
    p_dn_field  : 1-D array — downstream field pressure  (bar)

    D_min_mm, D_max_mm : float — diameter search bounds  (mm)
    method      : 'differential_evolution'  or  'L-BFGS-B'
    maxiter     : max iterations / generations
    popsize     : population size multiplier (differential_evolution only)
    tol         : convergence tolerance
    seed        : random seed for reproducibility
    D_init_mm   : float or 1-D array — initial guess in mm
    callback_every : print progress every N iterations

    Returns
    -------
    opt_result : dict
        'D_opt_m'      : np.ndarray  — optimised diameters  (m)
        'D_opt_mm'     : np.ndarray  — optimised diameters  (mm)
        'rmse_opt'     : float       — final RMSE  (bar)
        'rmse_mbar'    : float       — final RMSE  (mbar)
        'scipy_result' : scipy OptimizeResult object
        'history_rmse' : list of RMSE values per iteration (DE only)
        'sim_results'  : dict from run_cng_moc_v5 at optimised D
    """

    N = int(round(sim_params['L'] / sim_params['dx']))
    bounds = [(D_min_mm * 1e-3, D_max_mm * 1e-3)] * N

    if D_init_mm is None:
        D_init = np.full(N, 0.5 * (D_min_mm + D_max_mm) * 1e-3)
    elif np.isscalar(D_init_mm):
        D_init = np.full(N, D_init_mm * 1e-3)
    else:
        D_init = np.asarray(D_init_mm) * 1e-3

    history_rmse = []
    iteration_count = [0]

    def callback_de(xk, convergence):
        iteration_count[0] += 1
        if iteration_count[0] % callback_every == 0:
            rmse_now = moc_objective(xk, sim_params, t_dn_field, p_dn_field)
            history_rmse.append(rmse_now)
            if verbose:
                print(f"  iter {iteration_count[0]:4d}  RMSE = {rmse_now*1000:.4f} mbar  "
                      f"D range: {xk.min()*1000:.1f}–{xk.max()*1000:.1f} mm")

    def callback_lbfgsb(xk):
        iteration_count[0] += 1
        if iteration_count[0] % callback_every == 0:
            rmse_now = moc_objective(xk, sim_params, t_dn_field, p_dn_field)
            history_rmse.append(rmse_now)
            if verbose:
                print(f"  iter {iteration_count[0]:4d}  RMSE = {rmse_now*1000:.4f} mbar")

    if verbose:
        print("\n" + "=" * 60)
        print("DIAMETER OPTIMISATION")
        print("=" * 60)
        print(f"  Segments N    : {N}")
        print(f"  Bounds        : {D_min_mm:.1f} – {D_max_mm:.1f} mm")
        print(f"  Method        : {method}")
        print(f"  Max iterations: {maxiter}")
        rmse_init = moc_objective(D_init, sim_params, t_dn_field, p_dn_field)
        print(f"  RMSE (initial): {rmse_init*1000:.4f} mbar")
        print()

    # ── Run optimiser ─────────────────────────────────────────────────────────
    if method == 'differential_evolution':
        scipy_result = differential_evolution(
            func       = moc_objective,
            bounds     = bounds,
            args       = (sim_params, t_dn_field, p_dn_field),
            maxiter    = maxiter,
            popsize    = popsize,
            tol        = tol,
            seed       = seed,
            callback   = callback_de,
            init       = 'latinhypercube',
            polish     = True,
            workers    = 1,
        )

    elif method == 'L-BFGS-B':
        scipy_result = minimize(
            fun      = moc_objective,
            x0       = D_init,
            args     = (sim_params, t_dn_field, p_dn_field),
            method   = 'L-BFGS-B',
            bounds   = bounds,
            options  = {'maxiter': maxiter, 'ftol': tol**2, 'gtol': tol},
            callback = callback_lbfgsb,
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'differential_evolution' or 'L-BFGS-B'.")

    D_opt = scipy_result.x
    rmse_opt = scipy_result.fun

    if verbose:
        print(f"\n=== Optimisation complete ===")
        print(f"  RMSE (optimised) : {rmse_opt*1000:.4f} mbar")
        print(f"  D range (opt)    : {D_opt.min()*1000:.2f} – {D_opt.max()*1000:.2f} mm")
        print(f"  D mean  (opt)    : {D_opt.mean()*1000:.2f} mm")
        print(f"  Success          : {scipy_result.success}")
        if hasattr(scipy_result, 'message'):
            print(f"  Message          : {scipy_result.message}")

    # Run one final simulation at optimal D to get full results
    sim_results = run_cng_moc_v5(
        L                        = sim_params['L'],
        dx                       = sim_params['dx'],
        D                        = D_opt,
        eps                      = sim_params['eps'],
        gas_model                = sim_params.get('gas_model'),
        T_celsius                = sim_params['T_celsius'],
        P_operating_bar          = sim_params['P_operating_bar'],
        use_coolprop             = sim_params.get('use_coolprop', False),
        coolprop_fluid           = sim_params.get('coolprop_fluid', 'Methane'),
        coolprop_n_points        = sim_params.get('coolprop_n_points', 25),
        P_initial_bar            = sim_params['P_initial_bar'],
        Q_initial_scmh           = sim_params['Q_initial_scmh'],
        bc_mode                  = sim_params.get('bc_mode', 'case1'),
        upstream_pressure_data   = sim_params.get('upstream_pressure_data'),
        downstream_pressure_data = sim_params.get('downstream_pressure_data'),
        dt_field                 = sim_params['dt_field'],
        Q_calib_scmh             = sim_params.get('Q_calib_scmh'),
        P_dn_calib_bar           = sim_params.get('P_dn_calib_bar'),
        P_up_calib_bar           = sim_params.get('P_up_calib_bar'),
        tau_calib                = sim_params.get('tau_calib', 1.0),
        Cv_known                 = sim_params.get('Cv_known'),
        T_total                  = sim_params.get('T_total'),
        friction_tuning          = sim_params.get('friction_tuning', 0.9),
        elevation_mode           = sim_params.get('elevation_mode', 'flat'),
        elevation_data           = sim_params.get('elevation_data'),
        verbose                  = True,
    )

    return {
        'D_opt_m'      : D_opt,
        'D_opt_mm'     : D_opt * 1000,
        'rmse_opt'     : rmse_opt,
        'rmse_mbar'    : rmse_opt * 1000,
        'scipy_result' : scipy_result,
        'history_rmse' : history_rmse,
        'sim_results'  : sim_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOT  (simulated vs field downstream pressure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(sim_results, t_dn_field, p_dn_field,
                    history_rmse=None, save_path='optimisation_result.png'):
    """
    4-panel figure:
      1. P_dn simulated vs field
      2. Error trace
      3. Diameter profile
      4. Convergence history (if history_rmse supplied)

    Parameters
    ----------
    sim_results  : dict from run_cng_moc_v4
    t_dn_field   : field time axis  (s)
    p_dn_field   : field downstream pressure  (bar)
    history_rmse : list of RMSE per callback iteration (optional)
    save_path    : output PNG filename

    Returns
    -------
    metrics : dict  {rmse, mae, max_error, bias}  all in bar
    """
    p_sim = np.interp(t_dn_field, sim_results['time'], sim_results['P_dn'])
    err   = p_sim - p_dn_field

    rmse     = np.sqrt(np.mean(err**2))
    mae      = np.mean(np.abs(err))
    max_err  = np.max(np.abs(err))
    bias     = err.mean()

    print(f"\n=== Downstream Pressure Metrics ===")
    print(f"  RMSE      : {rmse*1000:.3f} mbar")
    print(f"  MAE       : {mae*1000:.3f} mbar")
    print(f"  Max error : {max_err*1000:.3f} mbar")
    print(f"  Bias      : {bias*1000:.3f} mbar")

    n_panels = 4 if (history_rmse and len(history_rmse) > 1) else 3
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 4 * n_panels),
                             constrained_layout=True)

    # Panel 1: pressure comparison
    ax = axes[0]
    ax.plot(t_dn_field, p_dn_field, 'o', ms=1.5, alpha=0.5,
            color='steelblue', label='Field P_dn')
    ax.plot(sim_results['time'], sim_results['P_dn'], '-', lw=1.5,
            color='crimson', label='MOC P_dn (simulated)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title(f'Downstream Pressure — RMSE = {rmse*1000:.2f} mbar', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Panel 2: error
    ax = axes[1]
    ax.plot(t_dn_field, err * 1000, lw=0.8, color='crimson', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.fill_between(t_dn_field, err * 1000, alpha=0.2, color='crimson')
    ax.set_ylabel('Error (mbar)')
    ax.set_title(f'Error — MAE={mae*1000:.2f} mbar   Bias={bias*1000:.2f} mbar')
    ax.grid(alpha=0.3)

    # Panel 3: diameter profile
    ax = axes[2]
    ax.plot(sim_results['x_segments'], sim_results['D_segments'] * 1000,
            '-o', lw=2, ms=4, color='darkorange')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Diameter (mm)')
    ax.set_title('Optimised Diameter Profile')
    ax.grid(alpha=0.3)

    # Panel 4: convergence
    if n_panels == 4:
        ax = axes[3]
        ax.semilogy(np.arange(1, len(history_rmse) + 1),
                    np.array(history_rmse) * 1000,
                    '-o', lw=1.5, ms=4, color='teal')
        ax.set_xlabel('Callback iteration')
        ax.set_ylabel('RMSE (mbar)')
        ax.set_title('Optimisation Convergence')
        ax.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved → {save_path}")
    plt.show()

    return {'rmse': rmse, 'mae': mae, 'max_error': max_err, 'bias': bias}


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD OPTIMISED DIAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def save_diameter_profile(opt_result, sim_params, filepath='diameter_profile.csv'):
    """Save segment diameters and pipe geometry to CSV."""
    N  = int(round(sim_params['L'] / sim_params['dx']))
    dx = sim_params['L'] / N
    x_mid = np.arange(N) * dx + dx / 2

    df = pd.DataFrame({
        'segment'   : np.arange(N),
        'x_mid_m'   : x_mid,
        'D_opt_mm'  : opt_result['D_opt_mm'],
        'D_opt_m'   : opt_result['D_opt_m'],
    })
    df.to_csv(filepath, index=False)
    print(f"Diameter profile saved → {filepath}")
    return df


def load_diameter_profile(filepath):
    """Load diameter profile saved by save_diameter_profile."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} segment diameters from {filepath}")
    return df['D_opt_m'].values


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Synthetic upstream data ───────────────────────────────────────────────
    fs     = 100
    T_demo = 60.0
    t_up   = np.arange(0, T_demo, 1.0 / fs)
    p_up   = np.where(t_up < 10, 13.65,
             np.where(t_up < 20, 13.65 - 0.25*(t_up-10)/10, 13.40))

    # ── Fixed simulation parameters ───────────────────────────────────────────
    sim_params = dict(
        L                      = 2100.0,
        dx                     = 100.0,
        eps                    = 8 * 45e-3,
        T_celsius              = 30.0,
        P_operating_bar        = 13.0,
        P_initial_bar          = 13.65,
        Q_initial_scmh         = 600.0,
        upstream_pressure_data = p_up,
        dt_field               = 1.0 / fs,
        Q_calib_scmh           = 600.0,
        P_dn_calib_bar         = 13.20,
        tau_calib              = 1.0,
        T_total                = T_demo,
        friction_tuning        = 0.9,
        # ── CoolProp real-gas (optional) ──────────────────────────────────────
        # Set use_coolprop=True to enable.  Requires:  pip install coolprop
        use_coolprop           = False,
        coolprop_fluid         = 'Methane',
        coolprop_n_points      = 25,
    )

    # ── Synthetic downstream "field" data ─────────────────────────────────────
    ref = run_cng_moc_v5(**{k: v for k, v in sim_params.items()
                            if k != 'T_total'},
                         T_total=T_demo, verbose=False)
    rng        = np.random.default_rng(0)
    t_dn_field = ref['time']
    p_dn_field = ref['P_dn'] + rng.normal(0, 0.002, len(ref['time']))

    # ── Run optimisation ──────────────────────────────────────────────────────
    opt = run_diameter_optimisation(
        sim_params    = sim_params,
        t_dn_field    = t_dn_field,
        p_dn_field    = p_dn_field,
        D_min_mm      = 200.0,
        D_max_mm      = 300.0,
        method        = 'differential_evolution',
        maxiter       = 50,
        popsize       = 8,
        D_init_mm     = 254.0,
        callback_every= 5,
        verbose       = True,
    )

    # ── Plot & save ───────────────────────────────────────────────────────────
    metrics = plot_comparison(
        opt['sim_results'], t_dn_field, p_dn_field,
        history_rmse=opt['history_rmse'],
    )

    save_diameter_profile(opt, sim_params)
