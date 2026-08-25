
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from moc_core import run_moc
import sys
import os
import json

# ============================================================
# OPTIMIZATION ENGINE V2 (Single Valve)
# ============================================================

def run_optimization(
    pt_data_file,
    elevation_file,
    L=69500.0,
    dx=500.0,
    soundspeed=820.0,
    pipe_od_inch=10.75,
    wall_thk=0.0071,
    fix_start_km=5.0,
    fix_end_km=0.5,
    block_size_km=3.0,
    max_iter=40,
    status_callback=None,
    iteration_callback=None, # New callback for per-iteration saving
    **kwargs # Accept extra arguments like Q0, H_up, etc. to be compatible with UI
):
    """
    Runs the pipeline diameter optimization using moc_core logic.
    Progress is saved after each iteration via iteration_callback.
    """
    
    def log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    log("Starting optimization process (v2 / Single-Valve logic)...")

    # ============================================================
    # 1. LOAD PT DATA
    # ============================================================
    log(f"Loading PT data from: {pt_data_file}")
    try:
        pt = pd.read_csv(pt_data_file)
        t_pt = pt.iloc[:, 0].values
        p_pt = pt.iloc[:, 1].values
    except Exception as e:
        log(f"Error loading PT data: {e}")
        raise e

    # ============================================================
    # 2. PIPE GEOMETRY & GRID
    # ============================================================
    PIPE_OD = pipe_od_inch * 0.0254
    PIPE_ID = PIPE_OD - 2 * wall_thk
    
    N = int(L / dx)
    
    x_km = np.linspace(0, L, N + 1) / 1000.0
    
    # Diameter bounds
    D_min = max(PIPE_ID - 10.0 * wall_thk, 0.05) # At least 50mm
    D_max = PIPE_ID + 0.05 * wall_thk
    D_nom = PIPE_ID
    
    log(f"Geometry: L={L}m, dx={dx}m, N={N}")
    log(f"Pipe ID (Nominal): {D_nom:.4f} m")

    # ============================================================
    # 3. FIXED / FREE SECTIONS
    # ============================================================
    i_fix_start = int(fix_start_km * 1000 / dx)
    i_fix_end = int(N - fix_end_km * 1000 / dx)
    
    # ============================================================
    # 4. BLOCK DIAMETER SETUP
    # ============================================================
    nodes_per_block = int(block_size_km * 1000 / dx)
    if nodes_per_block < 1: 
        nodes_per_block = 1
        
    n_opt_nodes = i_fix_end - i_fix_start
    if n_opt_nodes <= 0:
        n_blocks = 0
    else:
        n_blocks = int(np.ceil(n_opt_nodes / nodes_per_block))
    
    log(f"Block size: {block_size_km} km ({nodes_per_block} nodes)")
    log(f"Optimized blocks: {n_blocks}")

    # ============================================================
    # 5. BUILD FULL DIAMETER ARRAY FROM BLOCKS
    # ============================================================
    def build_full_diameter(D_blocks):
        D_full = np.full(N + 1, D_nom)
        if n_blocks == 0:
            return D_full
            
        idx = i_fix_start
        for b in range(len(D_blocks)):
            val = D_blocks[b]
            i_end = min(idx + nodes_per_block, i_fix_end)
            D_full[idx:i_end] = val
            idx = i_end
            if idx >= i_fix_end:
                break
        return D_full

    # ============================================================
    # 6. INITIAL MOC RUN
    # ============================================================
    if n_blocks > 0:
        D0_blocks = np.full(n_blocks, D_nom)
    else:
        D0_blocks = np.array([])
        
    log("Running Initial MOC...")
    D0_full = build_full_diameter(D0_blocks)
    
    # Single valve moc_core.run_moc usually has fewer params than 2valve
    # We try to pass common params if they exist in kwargs
    moc_params = {
        "L": L,
        "dx": dx,
        "a": soundspeed,
        "elevation_file": elevation_file
    }
    # Optional params for moc_core.run_moc
    if "Q0" in kwargs: moc_params["Q0"] = kwargs["Q0"]
    if "H_up" in kwargs: moc_params["H_up"] = kwargs["H_up"]
    if "H_ref" in kwargs: moc_params["H_ref"] = kwargs["H_ref"]
    if "T_total" in kwargs: moc_params["T_total"] = kwargs["T_total"]
    if "T_stable" in kwargs: moc_params["T_stable"] = kwargs["T_stable"]
    if "viscosity" in kwargs: moc_params["nu"] = kwargs["viscosity"]
    if "density" in kwargs: moc_params["rho"] = kwargs["density"]

    t0, p0 = run_moc(D0_full, **moc_params)

    # ============================================================
    # 7. COST FUNCTION
    # ============================================================
    iter_counter = {"i": 0}
    lambda_smooth = 100
    lambda_grad = 1000
    max_dD = 0.01 * D_nom

    def cost_fn(D_blocks):
        D_full = build_full_diameter(D_blocks)
        t, p = run_moc(D_full, **moc_params)
        p_interp = np.interp(t_pt, t, p)
        
        data_misfit = np.linalg.norm(p_interp - p_pt)**2
        if len(D_blocks) > 1:
            smooth_penalty = np.sum(np.diff(D_blocks)**2)
            dD = np.abs(np.diff(D_blocks))
            grad_violation = np.maximum(dD - max_dD, 0.0)
            grad_penalty = np.sum(grad_violation**2)
        else:
            smooth_penalty = 0
            grad_penalty = 0
            
        return data_misfit + lambda_smooth * smooth_penalty + lambda_grad * grad_penalty

    def opt_callback(xk):
        iter_counter["i"] += 1
        dmax = np.max(np.abs(xk - D_nom)) * 1000
        log(f"Iter {iter_counter['i']:3d} | Max ΔD = {dmax:.3f} mm")
        
        if iteration_callback:
            D_full = build_full_diameter(xk)
            t, p = run_moc(D_full, **moc_params)
            
            iter_results = {
                "iteration": iter_counter["i"],
                "t_opt": t,
                "p_opt": p,
                "t_pt": t_pt,
                "p_pt": p_pt,
                "x_km": x_km,
                "D_full": D_full,
                "max_delta_D_mm": dmax
            }
            iteration_callback(iter_results)

    # ============================================================
    # 8. OPTIMIZATION
    # ============================================================
    if n_blocks > 0 and max_iter > 0:
        log("Starting block-diameter optimization...")
        bounds = [(D_min, D_max)] * len(D0_blocks)
        
        res = minimize(
            cost_fn,
            D0_blocks,
            method="L-BFGS-B",
            bounds=bounds,
            callback=opt_callback,
            options={"maxiter": max_iter}
        )
        
        log(f"Optimization complete. Success: {res.success}, Cost: {res.fun}")
        D_opt_full = build_full_diameter(res.x)
    else:
        log("Skipping optimization (no blocks or max_iter=0).")
        D_opt_full = D0_full

    # ============================================================
    # 9. FINAL RESULTS
    # ============================================================
    t_opt, p_opt = run_moc(D_opt_full, **moc_params)
    
    return {
        "t_pt": t_pt,
        "p_pt": p_pt,
        "t0": t0,
        "p0": p0,
        "t_opt": t_opt,
        "p_opt": p_opt,
        "x_km": x_km,
        "D_opt_full": D_opt_full
    }
