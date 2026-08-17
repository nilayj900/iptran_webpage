import numpy as np
import pandas as pd
from scipy.signal import detrend
import logging

# Setup logging (aligned for EC2 deployment)
# logging.basicConfig(
#     filename="/home/ubuntu/bfa/results/logs/production_run_leak_results.log",
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

WINSIZE = 2500 # Moving average window size

#Pressure Conversion Constants
V_MIN = 0.72
V_MAX = 3.60
P_MIN = 0.0
P_MAX = 100.0
D_P_D_V_Factor = 1.0197 

# Pressure Conversion Function

def volt_to_bar_if_needed(v: np.ndarray) -> np.ndarray:
        V_min = V_MIN
        V_max = V_MAX
        P_max = P_MAX
        P_min = P_MIN
        dP_dV = (D_P_D_V_Factor)* (P_max - P_min) / (V_max - V_min)
        p_bar= (v - V_min) * dP_dV
        # p_bar = np.maximum(p_bar, 0.00)
        return p_bar


def convert_to_pt(t, v=None, p=None):
    # --- FIX: handle Unix timestamps properly ---
    t = pd.to_numeric(t, errors='coerce')

    # Detect unit automatically (robust approach)
    if np.nanmedian(t) > 1e12:
        # milliseconds
        t = pd.to_datetime(t, unit='ms', errors='coerce')
    else:
        # seconds
        t = pd.to_datetime(t, unit='s', errors='coerce')

    t = t.to_numpy()

    CAL_SLOPE = 1.0818
    CAL_INTERCEPT = -0.0086
    V_MIN, V_MAX = 0.72, 3.60

    if v is not None:
        v = pd.to_numeric(v, errors='coerce').astype(float)

        nan_count = np.isnan(v).sum()
        if nan_count > 0:
            logging.warning(f"{nan_count} non-numeric Voltage values before fill")

        v_series = pd.Series(v).ffill()
        if np.isnan(v_series.iloc[0]):
            logging.warning("Leading NaNs detected, applying backfill")
            v_series = v_series.bfill()

        v = v_series.to_numpy()

        corrected_v = CAL_SLOPE * v + CAL_INTERCEPT
        corrected_v = np.clip(corrected_v, V_MIN, V_MAX)

        p_out = volt_to_bar_if_needed(corrected_v)

    elif p is not None:
        p_out = pd.to_numeric(p, errors='coerce').astype(float)

        nan_count = np.isnan(p_out).sum()
        if nan_count > 0:
            logging.warning(f"{nan_count} non-numeric Pressure values before fill")

        p_series = pd.Series(p_out).ffill()
        if np.isnan(p_series.iloc[0]):
            logging.warning("Leading NaNs in pressure, applying backfill")
            p_series = p_series.bfill()

        p_out = p_series.to_numpy()

    else:
        raise ValueError("Either voltage or pressure must be provided")

    if len(t) != len(p_out):
        raise ValueError("Timestamp and pressure length mismatch")

    return t, p_out

# Filtering functions

def fill_outliers_linear(signal):
    """
    Replicates MATLAB's filloutliers(..., 'linear', 'quartiles').
    
    It identifies outliers based on the interquartile range (IQR) and replaces
    them with values interpolated from their non-outlier neighbors.
    """
    # Calculate Q1, Q3, and IQR
    q1 = np.percentile(signal, 25)
    q3 = np.percentile(signal, 75)
    iqr = q3 - q1
    
    # Define outlier thresholds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find indices of outliers
    outlier_indices = np.where((signal < lower_bound) | (signal > upper_bound))[0]
    good_indices = np.where((signal >= lower_bound) & (signal <= upper_bound))[0]

    if outlier_indices.size == 0 or good_indices.size < 2:
        # No outliers or not enough good points to interpolate
        return signal

    # Create a copy to modify
    signal_filled = np.copy(signal)
    
    # Interpolate using non-outlier points
    # `np.interp` is a simple 1D linear interpolation function
    signal_filled[outlier_indices] = np.interp(
        outlier_indices, 
        good_indices, 
        signal[good_indices]
    )
    
    return signal_filled

def filter_v1(signal, Q=0.01, R=1, win_size=WINSIZE):
        try:
            # Ensure input is a numpy array
            signal = np.array(signal, dtype=float)
            n1 = len(signal)

            # 1. Outlier removal (equivalent to filloutliers)
            # signal = fill_outliers_linear(signal)

            # 2. Mirror pad the signal
            pad_before = (win_size - 1) // 2
            pad_after = win_size // 2
            framelen = 36000
            N1 = len(signal)
            padlen = framelen // 2

            sig_padded=np.pad(signal, (padlen, padlen), 'reflect')

            sig_padded_med = pd.Series(sig_padded).rolling(window=win_size, min_periods=1, center=True).median().reset_index(drop=True).to_numpy()

            sig_padded_filt = pd.Series(sig_padded_med).rolling(window=win_size, min_periods=1, center=True).mean().reset_index(drop=True).to_numpy()

            if len(sig_padded_filt) > 36000:
                sig_trimmed = sig_padded_filt[padlen : padlen + N1]
            else:
                sig_trimmed = sig_padded_filt


            return sig_trimmed

            # return sig_padded_filt  
        except Exception as e:
            print(f"An error occurred in filter_v1: {e}")
            # Return an empty array or handle as appropriate
            return np.array([])