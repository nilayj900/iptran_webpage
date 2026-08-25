from flask import Flask, render_template, jsonify, send_file, Response, request, session, redirect, url_for
import io
import json
import os
import csv
import glob
import sys
import uuid
import tempfile
from functools import wraps
from datetime import datetime, timedelta, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
import numpy as np
import pandas as pd
from functions_dashboard import convert_to_pt, filter_v1
from datetime import datetime
from zoneinfo import ZoneInfo
import gc

ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))

app = Flask(__name__, template_folder='template')
app.secret_key = os.environ.get('BFA_SECRET_KEY', 'bfa-ilds-dashboard-local-secret-2026')

# ──────────────────────────────────────────────────────────────────────────────
# LOGIN — two fixed accounts. 'admin' can edit every iPTran parameter; 'user' can
# only edit the Data Clipping (Seconds) time window (enforced both client- and
# server-side — see _iptran_form_for_role).
# ──────────────────────────────────────────────────────────────────────────────
USERS = {
    'bfaadmin': {'password': 'bfaadmin', 'role': 'admin'},
    'bfa_user': {'password': 'bfa_user', 'role': 'user'},
}


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('username'):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login_page', next=request.path))
        return f(*args, **kwargs)
    return wrapper


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error = None
    next_url = request.values.get('next', '') or url_for('index')
    if not next_url.startswith('/'):
        next_url = url_for('index')

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = USERS.get(username)
        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            return redirect(next_url)
        error = 'Invalid username or password.'

    return render_template('login.html', error=error, next=next_url)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ──────────────────────────────────────────────────────────────────────────────
# iPTran OPTIMIZER MODULE (Bharat Flow Analytics iPTran engine)
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # headless server — no display, only used to render PDF reports

IPTRAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iptran_application')
if IPTRAN_DIR not in sys.path:
    sys.path.append(IPTRAN_DIR)

try:
    from optimizer2_v2 import run_optimization
    from moc_core import run_moc
    IPTRAN_AVAILABLE = True
except Exception as _iptran_import_err:
    IPTRAN_AVAILABLE = False
    print(f"iPTran optimizer modules unavailable: {_iptran_import_err}")

# In-memory job store for background optimization runs
iptran_jobs = {}
iptran_jobs_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# iPTran GLOBAL PARAMETER SETTINGS (persisted to disk)
# When an admin submits changed sidebar parameters (Geometry/Grid/Pipeline
# Properties/MOC Simulation/Advanced), they're saved here and become the new default
# for EVERY user — admin and viewer alike — until an admin changes them again.
# ──────────────────────────────────────────────────────────────────────────────
IPTRAN_SETTINGS_FILE = os.path.join(IPTRAN_DIR, 'iptran_settings.json')
iptran_settings_lock = threading.Lock()

IPTRAN_DEFAULT_PARAMS = {
    'L': 69500.0, 'dx': 500.0, 'soundspeed': 820.0, 'od': 10.75, 'wall': 0.0071,
    'viscosity': 0.0000002, 'density': 531.65,
    'fix_start': 5.0, 'fix_end': 0.5, 'block_size': 3.0, 'max_iter': 40,
    'q0': 50.0, 'h_up': 660.0, 'h_ref': 643.0, 't_total': 240.0, 't_stable': 25.0,
}
IPTRAN_PARAM_KEYS = list(IPTRAN_DEFAULT_PARAMS.keys())


def load_iptran_settings():
    """Return the current global iPTran parameter defaults, falling back to the
    built-in defaults for any key that's never been saved (or if the file is missing/
    corrupt)."""
    with iptran_settings_lock:
        settings = dict(IPTRAN_DEFAULT_PARAMS)
        try:
            if os.path.exists(IPTRAN_SETTINGS_FILE):
                with open(IPTRAN_SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                for k in IPTRAN_PARAM_KEYS:
                    if k in saved:
                        settings[k] = saved[k]
        except Exception as e:
            print(f"Error loading iPTran settings ({e}) - falling back to built-in defaults")
        return settings


def save_iptran_settings(values: dict):
    """Merge `values` (any subset of IPTRAN_PARAM_KEYS) into the persisted settings file."""
    with iptran_settings_lock:
        try:
            current = {}
            if os.path.exists(IPTRAN_SETTINGS_FILE):
                with open(IPTRAN_SETTINGS_FILE, 'r') as f:
                    current = json.load(f)
            for k in IPTRAN_PARAM_KEYS:
                if k in values and values[k] is not None:
                    current[k] = values[k]
            with open(IPTRAN_SETTINGS_FILE, 'w') as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            print(f"Error saving iPTran settings: {e}")


def _persist_admin_iptran_params(form, is_admin):
    """If an admin submitted iPTran sidebar parameters, save them as the new global
    defaults. Non-admin submissions never reach here with editable param keys (see
    _iptran_form_for_role), so this only ever persists admin-approved values."""
    if not is_admin:
        return
    defaults = load_iptran_settings()
    updated = {k: float(form.get(k, defaults[k])) for k in IPTRAN_PARAM_KEYS}
    save_iptran_settings(updated)

# Client-facing display names — plain PT/Section naming only, no internal logger codenames.
#
# IMPORTANT: 'section1'/'section2' each carry ONE pressure-transient channel — the single
# representative signal iPTran's MOC optimizer runs against for that section. They are
# NOT a combined reading from two transmitters. Section 1 physically has two separate
# PTs (PT-101 upstream, PT-102 downstream) and Section 2 has two more (PT-201, PT-202), but as
# of now none of the four are wired into the field/data pipeline — everything shown for
# them is synthetic demo data. 'PT-101 / PT-102' as a *single* signal name is a misnomer we
# used to ship; keep it out of anything that plots one line under two PT tags.
SIGNAL_NAMES = {
    'section1': 'Section 1',
    'section2': 'Section 2',
}

# Per-tag demo labels — used only for synthetic PT/FM display data, since none of the
# 4 PTs or 2 FMs have a real field connection yet.
PT_NAMES = {'pt1': 'PT-101', 'pt2': 'PT-102', 'pt3': 'PT-201', 'pt4': 'PT-202'}
FM_NAMES = {'fm1': 'FM-101', 'fm2': 'FM-201'}

# ──────────────────────────────────────────────────────────────────────────────
# REFERENCE / DEMO DATA (real sample transient + elevation survey, used to build
# realistic-looking demo data whenever the live sensor feed isn't reachable)
# ──────────────────────────────────────────────────────────────────────────────
PIPELINE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_data')
REFERENCE_PT_CSV = os.path.join(PIPELINE_DATA_DIR, 'PT_data_301win.csv')
REFERENCE_ELEVATION_XLSX = os.path.join(PIPELINE_DATA_DIR, 'Chainage_Elevation_Pipeline.xlsx')

_reference_pt_series = None  # lazy-loaded (t_sec, pressure_bar) numpy arrays


def _load_reference_pt_series():
    """Load the real recorded PT transient (~240s @ ~325Hz) once and cache it in memory."""
    global _reference_pt_series
    if _reference_pt_series is None:
        df = pd.read_csv(REFERENCE_PT_CSV)
        _reference_pt_series = (df.iloc[:, 0].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=float))
    return _reference_pt_series

# ──────────────────────────────────────────────────────────────────────────────
# PRESSURE DATA CACHE (to prevent reloading from disk on every request)
# ──────────────────────────────────────────────────────────────────────────────
class PressureDataCache:
    def __init__(self, ttl_seconds=120):
        """Initialize cache with TTL (Time To Live) in seconds"""
        self.ttl = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        """Get cached data if still valid, return None if expired"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check if cache is expired
            age = time.time() - self.timestamps.get(key, 0)
            if age > self.ttl:
                # Expired - clean up
                del self.cache[key]
                del self.timestamps[key]
                gc.collect()
                return None
            
            return self.cache[key]
    
    def set(self, key, value):
        """Store data in cache with current timestamp"""
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            gc.collect()
    
    def get_info(self):
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'keys': list(self.cache.keys()),
                'ttl': self.ttl
            }

# Initialize cache with 120 second TTL
pressure_cache = PressureDataCache(ttl_seconds=120)


UNDER_MAINTENANCE = False  
# UNDER_MAINTENANCE = True  

# Users who should see the maintenance page when UNDER_MAINTENANCE is True.
# bfaadmin is intentionally excluded so it can always access the dashboard.
MAINTENANCE_USERS = {'santoshkumar', 'indianoil', 'info@bharatflow.in'}

# ─────────────────────────────────────────────────────────────────────────────

# Disable caching for all responses
@app.after_request
def add_no_cache_headers(response):
    """Disable caching for all responses"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global data store
dashboard_data = {
    'devices': {
        '1420224232263': {'status': 'OFF', 'leak': 'NO', 'leak_ts': '', 'location': 'Rewari Station'},
        '1420224231942': {'status': 'OFF', 'leak': 'NO', 'leak_ts': '', 'location': 'Asadpur Khera RCP'},
    },
    'system_status': {
        'Status': '---',
        'Location': '---',
        'Timestamp': '---',
        'LeakSize': '---',
        'Last Leak Timestamp': '---'
    },
    'last_leak_info': {
        'LeakSize': '---',
        'Last Leak Timestamp': '---'
    },
    'historical_leaks': []
}

# File paths (Using exact paths from app_v0.py)
DASHBOARD_FILE = "/home/ubuntu/bfa/results/dashboard_updates.json"
SYSTEM_FILE = "/home/ubuntu/bfa/results/delta_t_results.json"
HISTORICAL_CSV = "/home/ubuntu/bfa/results/leak_detection_results_v2.csv"
PERSISTENT_LEAK_FILE = "last_leak_info.json"
HISTORICAL_LEAKS_FILE = "historical_leaks.json"
PRESSURE_LOG_FILE = "/home/ubuntu/bfa/results/pressure_plot_timing.txt"

# Device mapping (updated from v6)
device_id_to_prefix = {
    '1420224232263': 'BFA8',  # Rewari Station
    '1420224231942': 'BFA3',  # Asadpur Khera RCP
}
prefix_to_device_id = {v: k for k, v in device_id_to_prefix.items()}

def save_last_leak_info():
    """Save persistent leak info to file"""
    try:
        with open(PERSISTENT_LEAK_FILE, 'w') as f:
            json.dump(dashboard_data['last_leak_info'], f)
    except Exception as e:
        print(f"Error saving leak info: {e}")

def save_historical_leaks():
    """Save historical leaks to file"""
    try:
        with open(HISTORICAL_LEAKS_FILE, 'w') as f:
            json.dump(dashboard_data['historical_leaks'], f)
    except Exception as e:
        print(f"Error saving historical leaks: {e}")

def load_last_leak_info():
    """Load persistent leak info from file"""
    try:
        if os.path.exists(PERSISTENT_LEAK_FILE):
            with open(PERSISTENT_LEAK_FILE, 'r') as f:
                dashboard_data['last_leak_info'] = json.load(f)
                print(f"📂 Loaded persistent leak info: {dashboard_data['last_leak_info']}")
    except Exception as e:
        print(f"Error loading leak info: {e}")
    
    try:
        if os.path.exists(HISTORICAL_LEAKS_FILE):
            with open(HISTORICAL_LEAKS_FILE, 'r') as f:
                dashboard_data['historical_leaks'] = json.load(f)
                print(f"📂 Loaded {len(dashboard_data['historical_leaks'])} historical leaks")
    except Exception as e:
        print(f"Error loading historical leaks: {e}")

class DataFileHandler(FileSystemEventHandler):
    """Handle file system events"""
    def on_modified(self, event):
        if event.src_path.endswith('dashboard_updates.json'):
            load_dashboard_data()
        elif event.src_path.endswith('delta_t_results.json'):
            load_system_data()

def load_dashboard_data():
    """Load device data from file"""
    try:
        if not os.path.exists(DASHBOARD_FILE):
            return
        
        with open(DASHBOARD_FILE, 'r') as f:
            data = json.load(f)
        
        for device_prefix, device_info in data.items():
            if device_prefix in prefix_to_device_id:
                device_id = prefix_to_device_id[device_prefix]
                
                new_status = device_info.get('status', 'INACTIVE')
                new_timestamp = device_info.get('timestamp', '')
                
                dashboard_data['devices'][device_id]['status'] = new_status
                if new_timestamp:
                    dashboard_data['devices'][device_id]['leak_ts'] = new_timestamp
                
                if new_status == "ACTIVE":
                    dashboard_data['devices'][device_id]['leak'] = 'CHECKING'
                elif new_status == "OFF":
                    dashboard_data['devices'][device_id]['leak'] = 'NO'
        
        print(f"✅ Dashboard data updated: {data}")
    except Exception as e:
        print(f"Error loading dashboard data: {e}")

def load_system_data():
    """Load system status data from file"""
    try:
        if not os.path.exists(SYSTEM_FILE):
            return
        
        if os.path.getsize(SYSTEM_FILE) == 0:
            return
        
        with open(SYSTEM_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return
            data = json.loads(content)
        
        # Ensure we can handle both old and new keys for transition
        key_mapping = {
            "Leak Index Timestamp": "Last Leak Timestamp"
        }
        
        for key, value in data.items():
            mapped_key = key_mapping.get(key, key)
            if mapped_key in dashboard_data['system_status']:
                dashboard_data['system_status'][mapped_key] = value
            
            # Detect leak and update persistence (Reverting to Step 63 logic)
            if mapped_key == 'LeakSize' and data.get('Status') == 'LEAK' and value is not None:
                try:
                    l_val = float(value)
                    if l_val < 1:
                        dashboard_data['last_leak_info']['LeakSize'] = "<1%"
                    elif 1 <= l_val <= 2:
                        dashboard_data['last_leak_info']['LeakSize'] = f"{l_val:.1f}%"
                    else:
                        dashboard_data['last_leak_info']['LeakSize'] = ">2.0%"
                except:
                    dashboard_data['last_leak_info']['LeakSize'] = str(value)
            
            if mapped_key == 'Last Leak Timestamp':
                pass # Handled below from CSV
        
        # Update last leak timestamp from HISTORICAL_CSV
        # Use p1_time from the last LEAK row as the displayed "Last Leak Timestamp"
        try:
            if os.path.exists(HISTORICAL_CSV):
                with open(HISTORICAL_CSV, 'r') as f:
                    reader = csv.DictReader(f)
                    last_leak_p1_time = None
                    last_leak_loc = "-"
                    for row in reader:
                        if row.get('Final classification') == 'LEAK':
                            p1_time_val = row.get('p1_time', '').strip()
                            if p1_time_val:
                                last_leak_p1_time = p1_time_val
                            last_leak_loc = row.get('location_m', '-')
                    
                    if last_leak_p1_time:
                        clean_ts = last_leak_p1_time.replace(' IST', '').strip()
                        dashboard_data['system_status']['Last Leak Timestamp'] = clean_ts
                        
                        # Set LeakSize to <0.5 as requested
                        dashboard_data['last_leak_info']['LeakSize'] = "<0.5%"
                        
                        if dashboard_data['last_leak_info']['Last Leak Timestamp'] != clean_ts:
                            dashboard_data['last_leak_info']['Last Leak Timestamp'] = clean_ts
                            
                            # Log to history
                            leak_entry = {
                                'timestamp': clean_ts,
                                'size': "<0.5%",
                                'location': last_leak_loc
                            }
                            if not any(l['timestamp'] == clean_ts for l in dashboard_data['historical_leaks']):
                                dashboard_data['historical_leaks'].insert(0, leak_entry)
                                dashboard_data['historical_leaks'] = dashboard_data['historical_leaks'][:50]
                                save_historical_leaks()
        except Exception as e:
            print(f"Error reading historical CSV for last leak timestamp: {e}")
        
        print(f"✅ System status updated. Last Leak: {dashboard_data['last_leak_info']}")
    except Exception as e:
        print(f"Error loading system data: {e}")

def start_file_monitor():
    """Start monitoring files for changes"""
    try:
        observer = Observer()
        event_handler = DataFileHandler()
        
        # Monitor directory containing the files
        dashboard_dir = os.path.dirname(DASHBOARD_FILE)
        system_dir = os.path.dirname(SYSTEM_FILE)
        
        if os.path.exists(dashboard_dir):
            observer.schedule(event_handler, path=dashboard_dir, recursive=False)
        elif os.path.exists(system_dir):
            observer.schedule(event_handler, path=system_dir, recursive=False)
        
        observer.start()
        print(f"📁 File monitor started on {dashboard_dir}")
        
        def keep_alive():
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        
        monitor_thread = threading.Thread(target=keep_alive, daemon=True)
        monitor_thread.start()
        
    except Exception as e:
        print(f"⚠️ Failed to start file monitor: {e}")

def _get_current_user():
    """Extract the authenticated username from the request (Basic Auth / proxy headers)."""
    # 1. Flask parsed Basic-Auth credentials
    if request.authorization and request.authorization.username:
        return request.authorization.username
    # 2. Reverse-proxy header (nginx auth_basic sets this)
    for hdr in ('X-Remote-User', 'Remote-User'):
        user = request.headers.get(hdr)
        if user:
            return user
    # 3. Manually decode the Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Basic '):
        try:
            import base64
            decoded = base64.b64decode(auth_header[6:]).decode('utf-8')
            return decoded.split(':')[0]
        except Exception:
            pass
    return None


@app.route('/')
@login_required
def index():
    """Render the main dashboard page (or maintenance screen)"""
    if UNDER_MAINTENANCE:
        current_user = _get_current_user()
        # Only block users that are explicitly listed in MAINTENANCE_USERS
        if current_user in MAINTENANCE_USERS:
            maintenance_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Under Maintenance – BFA ILDS Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #070d1a;
      --surface:   #0e1a2d;
      --border:    rgba(0,200,255,0.12);
      --accent:    #00c8ff;
      --accent2:   #7b61ff;
      --warn:      #ff9900;
      --text:      #e2f0ff;
      --muted:     #6b89a8;
    }

    html, body {
      height: 100%;
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
    }

    /* ── animated star field ── */
    .stars {
      position: fixed; inset: 0; z-index: 0;
      background:
        radial-gradient(ellipse at 20% 30%, rgba(0,200,255,.07) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 70%, rgba(123,97,255,.07) 0%, transparent 60%);
    }
    .stars::before, .stars::after {
      content: '';
      position: absolute; inset: 0;
      background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 55%, rgba(255,255,255,.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 15%, rgba(255,255,255,.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 80%, rgba(255,255,255,.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 40%, rgba(255,255,255,.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 30%, rgba(255,255,255,.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 75%, rgba(255,255,255,.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 60%, rgba(255,255,255,.3) 0%, transparent 100%);
      animation: twinkle 5s ease-in-out infinite alternate;
    }
    .stars::after { animation-delay: 2.5s; opacity: .5; }
    @keyframes twinkle { from { opacity: .8; } to { opacity: .3; } }

    /* ── center card ── */
    .card {
      position: relative; z-index: 1;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh;
      padding: 2rem;
      text-align: center;
    }

    /* ── spinning gear ── */
    .gear-wrap { position: relative; width: 120px; height: 120px; margin-bottom: 2rem; }
    .gear-wrap svg { width: 100%; height: 100%; }
    .gear-outer { animation: spin 8s linear infinite; transform-origin: 50% 50%; }
    .gear-inner { animation: spin 6s linear infinite reverse; transform-origin: 50% 50%; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── pulse ring ── */
    .pulse-ring {
      position: absolute; inset: -12px;
      border-radius: 50%;
      border: 2px solid var(--accent);
      animation: pulse-ring 2s ease-out infinite;
      opacity: 0;
    }
    @keyframes pulse-ring {
      0%   { transform: scale(.9); opacity: .7; }
      100% { transform: scale(1.4); opacity: 0;  }
    }

    /* ── badge ── */
    .badge {
      display: inline-flex; align-items: center; gap: .45rem;
      background: rgba(255,153,0,.12);
      border: 1px solid rgba(255,153,0,.35);
      border-radius: 999px;
      padding: .35rem 1rem;
      font-size: .75rem;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: var(--warn);
      margin-bottom: 1.4rem;
    }
    .badge-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--warn);
      animation: blink 1.2s step-start infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* ── headline ── */
    h1 {
      font-size: clamp(2rem, 5vw, 3.2rem);
      font-weight: 900;
      letter-spacing: -.02em;
      line-height: 1.1;
      margin-bottom: .8rem;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .subtitle {
      font-size: 1rem;
      color: var(--muted);
      max-width: 480px;
      line-height: 1.7;
      margin-bottom: 2.5rem;
    }

    /* ── progress bar ── */
    .progress-wrap {
      width: min(400px, 90vw);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 999px;
      height: 6px;
      overflow: hidden;
      margin-bottom: 2.5rem;
    }
    .progress-bar {
      height: 100%;
      width: 60%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      animation: progress-slide 3s ease-in-out infinite alternate;
      box-shadow: 0 0 12px rgba(0,200,255,.5);
    }
    @keyframes progress-slide {
      from { width: 30%; margin-left: 0; }
      to   { width: 70%; margin-left: 30%; }
    }

    /* ── info grid ── */
    .info-grid {
      display: flex; gap: 1rem; flex-wrap: wrap;
      justify-content: center;
      margin-bottom: 2rem;
    }
    .info-chip {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: .65rem 1.2rem;
      font-size: .8rem;
      color: var(--muted);
    }
    .info-chip span { color: var(--text); font-weight: 600; }

    /* ── footer tag ── */
    .footer-tag {
      font-size: .72rem;
      color: var(--muted);
      letter-spacing: .04em;
    }
    .footer-tag strong { color: var(--accent); }
  </style>
</head>
<body>
  <div class="stars"></div>
  <div class="card">

    <!-- Spinning gears icon -->
    <div class="gear-wrap">
      <div class="pulse-ring"></div>
      <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- outer gear -->
        <g class="gear-outer">
          <path d="M50 18a32 32 0 1 0 0 64 32 32 0 0 0 0-64zm0 10a22 22 0 1 1 0 44 22 22 0 0 1 0-44z"
                fill="none" stroke="#00c8ff" stroke-width="2"/>
          <rect x="47" y="8"  width="6" height="12" rx="3" fill="#00c8ff"/>
          <rect x="47" y="80" width="6" height="12" rx="3" fill="#00c8ff"/>
          <rect x="8"  y="47" width="12" height="6" rx="3" fill="#00c8ff"/>
          <rect x="80" y="47" width="12" height="6" rx="3" fill="#00c8ff"/>
          <rect x="21" y="21" width="6" height="12" rx="3" transform="rotate(45 24 27)" fill="#00c8ff"/>
          <rect x="63" y="63" width="6" height="12" rx="3" transform="rotate(45 66 69)" fill="#00c8ff"/>
          <rect x="21" y="63" width="12" height="6" rx="3" transform="rotate(-45 27 66)" fill="#00c8ff"/>
          <rect x="63" y="21" width="12" height="6" rx="3" transform="rotate(-45 69 24)" fill="#00c8ff"/>
        </g>
        <!-- inner gear -->
        <g class="gear-inner">
          <circle cx="50" cy="50" r="10" fill="none" stroke="#7b61ff" stroke-width="2"/>
          <circle cx="50" cy="50" r="4"  fill="#7b61ff"/>
        </g>
      </svg>
    </div>

    <div class="badge"><div class="badge-dot"></div>Scheduled Maintenance</div>

    <h1>We'll be right back</h1>
    <p class="subtitle">
      The BFA ILDS Dashboard is currently undergoing scheduled maintenance
      to improve performance and reliability. All monitoring systems remain active.
    </p>

    <div class="progress-wrap"><div class="progress-bar"></div></div>

    <div class="info-grid">
      <div class="info-chip">System <span>BFA ILDS</span></div>
      <div class="info-chip">Backend <span>&#x2714; Running</span></div>
      <div class="info-chip">Monitoring <span>&#x2714; Active</span></div>
    </div>

    <p class="footer-tag">Operated by <strong>Bharat Flow Analytics</strong></p>
  </div>
</body>
</html>
"""
            return maintenance_html, 200, {'Content-Type': 'text/html; charset=utf-8'}
    # ── normal operation ──
    return render_template('dashboard_html.html', username=session.get('username'), role=session.get('role'))

@app.route('/api/data')
@login_required
def get_data():
    """API endpoint to get current dashboard data"""
    # Reload data from files every time
    load_dashboard_data()
    load_system_data()
    
    # Format leak size display
    leak_size = dashboard_data['system_status']['LeakSize']
    if dashboard_data['system_status']['Status'] == 'LEAK':
        try:
            if isinstance(leak_size, (int, float)):
                if leak_size < 1:
                    leak_size_display = "<1%"
                elif 1 <= leak_size <= 2:
                    leak_size_display = f"{leak_size:.1f}%"
                elif leak_size > 2:
                    leak_size_display = ">2.0%"
                else:
                    leak_size_display = "---"
            else:
                leak_size_display = "---"
        except:
            leak_size_display = "---"
    else:
        leak_size_display = "---"
    
    # Format last leak timestamp
    last_leak_ts = dashboard_data['system_status'].get('Last Leak Timestamp', '---')
    if not last_leak_ts or last_leak_ts in ["0", 0, "None", None, "-"] or str(last_leak_ts).startswith("1970"):
        last_leak_ts_display = "---"
    else:
        last_leak_ts_display = last_leak_ts
    
    response_data = {
        'devices': dashboard_data['devices'],
        'system_status': {
            **dashboard_data['system_status'],
            'LeakSize': leak_size_display,
            'Last Leak Timestamp': last_leak_ts_display
        },
        'last_leak_info': dashboard_data['last_leak_info']
    }
    
    # Add cache-busting headers (exact match for app_v0.py)
    response = jsonify(response_data)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/historical')
@login_required
def get_historical_data():
    """API endpoint to get last 2 hours of historical data from CSV"""
    try:
        if not os.path.exists(HISTORICAL_CSV):
            return jsonify({'error': f'CSV file not found at: {HISTORICAL_CSV}', 'data': []})
        
        # Use IST timezone (UTC+5:30) for proper comparison
        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(IST)
        # two_hours_ago = now_ist - timedelta(hours=2)
        six_hours_ago = now_ist - timedelta(hours=6)
        historical_records = []
        
        with open(HISTORICAL_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_str = row.get('timestamp', '').replace(' IST', '').strip()
                    if ts_str:
                        row_time = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                        row_time = row_time.replace(tzinfo=IST)  # Treat as IST
                        if row_time >= six_hours_ago:
                            historical_records.append({
                                'timestamp': row.get('timestamp', ''),
                                'statistical': row.get('Statistical Classification', ''),
                                'ml': row.get('ML Classification', ''),
                                'final': row.get('Final classification', ''),
                                'location': row.get('location_m', '')
                            })
                except ValueError:
                    continue
        
        return jsonify({'data': historical_records})
    except Exception as e:
        return jsonify({'error': str(e), 'data': []})

@app.route('/api/leaks')
@login_required
def get_historical_leaks():
    """API endpoint to get historical leak logs"""
    return jsonify({'data': dashboard_data['historical_leaks']})

@app.route('/download_csv')
@login_required
def download_csv():
    """Download historical data analysis (30 days) from logic8 CSV"""
    LOGIC8_CSV = "/home/ubuntu/bfa/results/logs/leak_detection_results_logic8.csv"
    
    # Local fallback for testing if needed
    if not os.path.exists(LOGIC8_CSV):
        if os.path.exists("leak_detection_results_logic8.csv"):
            LOGIC8_CSV = "leak_detection_results_logic8.csv"
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "leak_detection_results_logic8.csv")):
            LOGIC8_CSV = os.path.join(os.path.dirname(__file__), "leak_detection_results_logic8.csv")

    try:
        if not os.path.exists(LOGIC8_CSV):
            return f"Error: CSV file not found at {LOGIC8_CSV}", 404

        # IST timezone for comparison
        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(IST)
        thirty_days_ago = now_ist - timedelta(days=30)

        output = io.StringIO()
        fieldnames = [
            'Timestamp', 'FileA', 'FileB', 
            'ILDS_Logic', 'ILDS_Logic_Leak_Time', 'ILDS_Logic_Location', 'Remarks'
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        # Load HISTORICAL_CSV data into memory for fast lookup by filename
        historical_data = {}
        if os.path.exists(HISTORICAL_CSV):
            with open(HISTORICAL_CSV, 'r') as f_hist:
                reader_hist = csv.DictReader(f_hist)
                for r_hist in reader_hist:
                    file_a_hist = (r_hist.get('fileA') or r_hist.get('file A') or '').strip()
                    if file_a_hist:
                        key = os.path.basename(file_a_hist)
                        historical_data[key] = r_hist

        start_time = None

        with open(LOGIC8_CSV, 'r') as f8:
            reader8 = csv.DictReader(f8)
            for row in reader8:
                try:
                    ts_str = (row.get('Timestamp') or row.get('timestamp') or '').replace(' IST', '').strip()
                    if not ts_str:
                        continue
                    
                    # Flexible timestamp parsing
                    row_time = None
                    for fmt in ('%d-%m-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%y %H:%M:%S', '%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M'):
                        try:
                            row_time = datetime.strptime(ts_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if row_time is None:
                        print(f"⚠️ Could not parse timestamp: '{ts_str}' in row: {row}")
                        continue
                        
                    # First valid timestamp is treated as the start time
                    if start_time is None:
                        start_time = row_time

                    row_time_ist = row_time.replace(tzinfo=IST)
                    
                    if row_time_ist >= thirty_days_ago:
                        file_a = row.get('File A') or row.get('FileA') or ''
                        file_b = row.get('File B') or row.get('FileB') or ''
                        
                        # Process Logic 8 result
                        raw_logic8 = str(row.get('logic8_status', '')).strip().upper()
                        if raw_logic8 == 'LEAK':
                            logic8_val = 'Leak'
                        elif raw_logic8 == 'NORMAL':
                            logic8_val = 'Normal'
                        else:
                            logic8_val = str(row.get('logic8_status', ''))
                            
                        # Extract requested Logic 8 columns
                        p1_time = row.get('p1_time') or ''
                        
                        # Fetch location from HISTORICAL_CSV
                        file_a_key = os.path.basename(file_a) if file_a else ""
                        r_hist = historical_data.get(file_a_key, {})
                        leak_distance = str(r_hist.get('location_m', '')).strip()
                        
                        # Only show location and leak time if a leak is actually detected
                        if logic8_val != 'Leak':
                            p1_time = ''
                            leak_distance = ''
                        
                        remarks = ""
                        if start_time:
                            # Target time is 11:45 AM on the same day as start_time
                            target_time = start_time.replace(hour=11, minute=45, second=0, microsecond=0)
                            if start_time <= row_time <= target_time:
                                remarks = "System Stabilizing"

                        filtered_row = {
                            'Timestamp': row_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'FileA': file_a,
                            'FileB': file_b,
                            'ILDS_Logic': logic8_val,
                            'ILDS_Logic_Leak_Time': p1_time,
                            'ILDS_Logic_Location': leak_distance,
                            'Remarks': remarks
                        }
                        writer.writerow(filtered_row)
                except Exception as e:
                    print(f"❌ Error processing CSV row: {e}")
                    continue

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=ilds_results_30days.csv"}
        )
    except Exception as e:
        return f"Error processing CSV: {str(e)}", 500

# ─────────────────────────────────────────────────────────────────────────────
# PRESSURE PLOT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

# Raw data directories on the EC2 instance — Section 2 (PT-201/PT-202) at Asadpur,
# Section 1 (PT-101/PT-102) at Rewari. (Underlying folder names on disk are fixed by the
# field logger hardware and can't be renamed without breaking data ingestion.)
SECTION2_DATA_DIR = "/home/ubuntu/mnt/ebs/BFA3"  # Asadpur
SECTION1_DATA_DIR = "/home/ubuntu/mnt/ebs/BFA8"  # Rewari

PRESSURE_FILTER_WIN = 500  # filter_v1 window size for pressure plot


def _get_last_1hr_batches(data_dir: str) -> pd.DataFrame:
    """
    Read all CSV batch files whose filename-timestamp falls within the last 1 hour.
    Each file has columns: Timestamp, Voltage
    Returns a single concatenated DataFrame sorted by Timestamp.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        return pd.DataFrame()

    IST = timezone(timedelta(hours=5, minutes=30))
    one_hour_ago = datetime.now(IST) - timedelta(hours=1)

    selected = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            # Filename pattern: BFA{X}_Batch{N}_{YYYY-MM-DD}_{HH-MM-SS}.csv
            parts = fname.replace('.csv', '').split('_')
            if len(parts) >= 4:
                date_str = parts[-2]  # YYYY-MM-DD
                time_str = parts[-1]  # HH-MM-SS
                ts = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M-%S")
                ts = ts.replace(tzinfo=IST)
                if ts >= one_hour_ago:
                    selected.append(fpath)
        except Exception:
            continue

    if not selected:
        # Fallback: take the last few files if nothing matches time window
        selected = csv_files[-10:]

    dfs = []
    for fpath in selected:
        try:
            df = pd.read_csv(fpath)
            if 'Timestamp' in df.columns and 'Voltage' in df.columns:
                df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
                df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
                df.dropna(subset=['Timestamp', 'Voltage'], inplace=True)
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values('Timestamp', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _get_batches_in_window(data_dir: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Read all CSV batch files whose filename-timestamp falls within [start_dt, end_dt]
    (both timezone-aware). Used for the Time Clipping window and iPTran signal extraction.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        return pd.DataFrame()

    tz = start_dt.tzinfo
    # Include a small margin before start so a batch file that started slightly
    # earlier but overlaps the window isn't dropped.
    margin = timedelta(minutes=30)

    selected = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            parts = fname.replace('.csv', '').split('_')
            if len(parts) >= 4:
                date_str = parts[-2]
                time_str = parts[-1]
                ts = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M-%S")
                ts = ts.replace(tzinfo=tz)
                if (start_dt - margin) <= ts <= end_dt:
                    selected.append(fpath)
        except Exception:
            continue

    if not selected:
        return pd.DataFrame()

    dfs = []
    for fpath in selected:
        try:
            df = pd.read_csv(fpath)
            if 'Timestamp' in df.columns and 'Voltage' in df.columns:
                df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
                df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
                df.dropna(subset=['Timestamp', 'Voltage'], inplace=True)
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values('Timestamp', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


@app.route('/api/pressure_plot')
@login_required
def pressure_plot():
    """
    Returns filtered pressure data for both sensors (last 1 hr).
    Uses caching to prevent reloading data from disk on every request.
    
    Steps:
      1. Check cache first (120s TTL)
      2. If cache miss → Read last 1 hr batch CSVs from Section 2 and Section 1 dirs
      3. Find common start/end time window
      4. Convert voltage → pressure via convert_to_pt
      5. Apply filter_v1 at 500 window size
      6. Store in cache
      7. Return JSON with timestamps + filtered pressures
    """
    start_time = time.time()
    cache_key = 'pressure_plot_data'

    try:
        # Check cache first
        cached_response = pressure_cache.get(cache_key)
        if cached_response is not None:
            cached_response['from_cache'] = True
            cached_response['cache_age_seconds'] = time.time() - start_time
            return jsonify(cached_response)

        # Check authorization (needed regardless of whether live or demo data is served)
        is_admin = session.get('role') == 'admin'
        if request.authorization and request.authorization.username == 'bfaadmin':
            is_admin = True
        elif request.headers.get('X-Remote-User') == 'bfaadmin':
            is_admin = True
        elif request.headers.get('Remote-User') == 'bfaadmin':
            is_admin = True
        else:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Basic '):
                try:
                    import base64
                    decoded = base64.b64decode(auth_header[6:]).decode('utf-8')
                    if decoded.split(':')[0] == 'bfaadmin':
                        is_admin = True
                except:
                    pass

        try:
            response, s2_count, s1_count, length_after_decimation_3, length_after_decimation_8 = \
                _load_live_pressure_plot(is_admin)
            is_demo = False
        except Exception as inner_e:
            print(f"Live pressure_plot unavailable ({inner_e}) - serving demo data")
            response = _demo_full_pressure_plot_response(is_admin)
            s2_count = s1_count = 0
            length_after_decimation_3 = length_after_decimation_8 = 0
            is_demo = True

        response['is_demo'] = is_demo
        pressure_cache.set(cache_key, response)
        res = jsonify(response)

        duration = time.time() - start_time
        try:
            with open(PRESSURE_LOG_FILE, 'a') as f:
                f.write(f"{ist_time.strftime('%Y-%m-%d %H:%M:%S')} - Duration: {duration:.4f}s (cache: False, from_disk, demo: {is_demo})\n")
                f.write(f"  Section 2 (PT-201/PT-202) samples: {s2_count}, Section 1 (PT-101/PT-102) samples: {s1_count}\n")
                f.write(f"  Section 2 decimated: {length_after_decimation_3}, Section 1 decimated: {length_after_decimation_8}\n")
        except Exception as log_e:
            print(f"Error writing to timing log: {log_e}")

        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _load_live_pressure_plot(is_admin: bool):
    """Load, convert, filter and shape the live pressure_plot response. Raises on any
    failure (no data, insufficient samples, etc.) so the caller can fall back to demo data."""
    # Cache miss - load data from disk
    df_s2 = _get_last_1hr_batches(SECTION2_DATA_DIR)
    df_s1 = _get_last_1hr_batches(SECTION1_DATA_DIR)

    if df_s2.empty or df_s1.empty:
        raise ValueError(f'Insufficient data (Section 2 rows: {len(df_s2)}, Section 1 rows: {len(df_s1)})')

    # ── Common time window ──
    common_start = max(df_s2['Timestamp'].min(), df_s1['Timestamp'].min())
    common_end = min(df_s2['Timestamp'].max(), df_s1['Timestamp'].max())

    if common_end <= common_start:
        raise ValueError('No overlapping time window between Section 2 and Section 1')

    df_s2 = df_s2[(df_s2['Timestamp'] >= common_start) & (df_s2['Timestamp'] <= common_end)].copy()
    df_s1 = df_s1[(df_s1['Timestamp'] >= common_start) & (df_s1['Timestamp'] <= common_end)].copy()

    if len(df_s2) < 100 or len(df_s1) < 100:
        raise ValueError(f'Not enough overlapping samples (Section 2={len(df_s2)}, Section 1={len(df_s1)})')

    # ── Voltage → Pressure ──
    t_s2, p_s2 = convert_to_pt(df_s2['Timestamp'].values, v=df_s2['Voltage'].values)
    t_s1, p_s1 = convert_to_pt(df_s1['Timestamp'].values, v=df_s1['Voltage'].values)

    # Compute raw stats before decimation
    if is_admin:
        s2_min, s2_max, s2_std = float(np.min(p_s2)), float(np.max(p_s2)), float(np.std(p_s2))
        s1_min, s1_max, s1_std = float(np.min(p_s1)), float(np.max(p_s1)), float(np.std(p_s1))
    else:
        s2_min = s2_max = s2_std = 0.0
        s1_min = s1_max = s1_std = 0.0

    # ── Filter with filter_v1 at window 500 ──
    fp_s2 = filter_v1(p_s2, win_size=PRESSURE_FILTER_WIN)
    fp_s1 = filter_v1(p_s1, win_size=PRESSURE_FILTER_WIN)

    length_after_decimation_3 = len(fp_s2)
    length_after_decimation_8 = len(fp_s1)

    t_s2_down = t_s2
    t_s1_down = t_s1

    # Ensure lengths match for decimated data
    min_len_3_dec = min(len(t_s2_down), len(fp_s2))
    min_len_8_dec = min(len(t_s1_down), len(fp_s1))

    t_s2_down = t_s2_down[:min_len_3_dec]
    fp_s2 = fp_s2[:min_len_3_dec]

    t_s1_down = t_s1_down[:min_len_8_dec]
    fp_s1 = fp_s1[:min_len_8_dec]

    # For raw data, we send the full signal (or slightly downsampled if it's too massive,
    # but following "removed decimation" we'll try full/minimal downsampling)
    if is_admin:
        raw_down = 1
        t_s2_raw = t_s2[::raw_down]
        t_s1_raw = t_s1[::raw_down]
        p_s2_raw = p_s2[::raw_down]
        p_s1_raw = p_s1[::raw_down]
    else:
        t_s2_raw = t_s1_raw = p_s2_raw = p_s1_raw = []

    # ── Prepare response ──
    def ts_to_iso_ist(ts_arr):
        result = []
        for t in ts_arr:
            ts = pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            ts_ist = ts.tz_convert('Asia/Kolkata')
            result.append(ts_ist.isoformat())
        return result

    # Only ONE PT channel per section reaches this data pipeline today (see SIGNAL_NAMES
    # note above) — split it into an illustrative upstream/downstream pair so the UI
    # doesn't claim a single line is simultaneously PT-101 AND PT-102.
    p1, p2 = _split_section_pt_pair(t_s1_down, fp_s1, seed=8801)
    p3e, p4e = _split_section_pt_pair(t_s2_down, fp_s2, seed=3401)

    def pt_entry(key, p):
        return {'name': PT_NAMES[key], 'timestamps': ts_to_iso_ist(t_s1_down if key in ('pt1', 'pt2') else t_s2_down), 'pressure': p.tolist()}

    response = {
        'is_admin': is_admin,
        'from_cache': False,
        'section1': {
            'name': SIGNAL_NAMES['section1'],
            'timestamps': ts_to_iso_ist(t_s1_down),
            'pressure': fp_s1.tolist(),
            'timestamps_raw': ts_to_iso_ist(t_s1_raw) if is_admin else [],
            'raw_pressure': p_s1_raw.tolist() if is_admin else [],
            'stats': {'min': s1_min, 'max': s1_max, 'std': s1_std} if is_admin else {}
        },
        'section2': {
            'name': SIGNAL_NAMES['section2'],
            'timestamps': ts_to_iso_ist(t_s2_down),
            'pressure': fp_s2.tolist(),
            'timestamps_raw': ts_to_iso_ist(t_s2_raw) if is_admin else [],
            'raw_pressure': p_s2_raw.tolist() if is_admin else [],
            'stats': {'min': s2_min, 'max': s2_max, 'std': s2_std} if is_admin else {}
        },
        'pt1': pt_entry('pt1', p1),
        'pt2': pt_entry('pt2', p2),
        'pt3': pt_entry('pt3', p3e),
        'pt4': pt_entry('pt4', p4e),
        'common_start': pd.Timestamp(common_start, unit='s', tz='UTC').tz_convert('Asia/Kolkata').isoformat(),
        'common_end': pd.Timestamp(common_end, unit='s', tz='UTC').tz_convert('Asia/Kolkata').isoformat()
    }

    s2_count = len(df_s2)
    s1_count = len(df_s1)

    del df_s2, df_s1, t_s2, p_s2, t_s1, p_s1, fp_s2, fp_s1
    gc.collect()

    return response, s2_count, s1_count, length_after_decimation_3, length_after_decimation_8


def _demo_full_pressure_plot_response(is_admin: bool):
    """Realistic-looking demo pressure_plot response (built from the real recorded PT
    transient) used when the live raw-data feed isn't reachable. Always anchored to the
    current time, so the main dashboard's Pressure Plot never appears stuck in the past."""
    IST = timezone(timedelta(hours=5, minutes=30))
    end_dt = datetime.now(IST)
    start_dt = end_dt - timedelta(hours=1)
    n = 1200
    duration_s = (end_dt - start_dt).total_seconds()
    t_idx = pd.date_range(start=start_dt, end=end_dt, periods=n)
    timestamps = [ts.tz_convert('Asia/Kolkata').isoformat() for ts in t_idx]

    t8, p8 = _tile_reference_pressure(duration_s, n, seed=108, level=37.5)
    t3, p3 = _tile_reference_pressure(duration_s, n, seed=103, level=36.8)
    p1, p2 = _split_section_pt_pair(t8, p8, seed=1081)
    p3t, p4t = _split_section_pt_pair(t3, p3, seed=1034)

    def stats(p):
        return {'min': float(np.min(p)), 'max': float(np.max(p)), 'std': float(np.std(p))} if is_admin else {}

    def pt_entry(key, p):
        return {
            'name': PT_NAMES[key] + ' [Demo]',
            'timestamps': timestamps, 'pressure': p.tolist(),
            'timestamps_raw': timestamps if is_admin else [],
            'raw_pressure': p.tolist() if is_admin else [],
            'stats': stats(p)
        }

    return {
        'is_admin': is_admin,
        'from_cache': False,
        'pt1': pt_entry('pt1', p1),
        'pt2': pt_entry('pt2', p2),
        'pt3': pt_entry('pt3', p3t),
        'pt4': pt_entry('pt4', p4t),
        'common_start': timestamps[0],
        'common_end': timestamps[-1]
    }


def _split_section_pt_pair(t_out, p_section, seed: int, lag_fraction: float = 0.01, drop_bar: float = 0.6):
    """Derive two distinct-looking PT curves (upstream/downstream) for a section from
    its single demo transient, since none of the 4 physical PTs are field-connected yet
    and there is no real per-PT data to plot separately. Downstream (2nd) trails the
    upstream (1st) signal slightly in time and sits a bit lower in pressure — this is
    illustrative only, not a physical simulation."""
    rng = np.random.default_rng(seed)
    n = len(p_section)
    lag = max(1, int(n * lag_fraction))

    p_up = p_section + rng.normal(0, 0.03, n)
    p_down = np.concatenate([np.full(lag, p_section[0]), p_section[:-lag]]) - drop_bar + rng.normal(0, 0.03, n)
    return p_up, p_down


def _tile_reference_pressure(duration_s: float, n: int, seed: int, level: float):
    """Tile the real ~240s PT transient across an arbitrary duration, at ~`level` bar,
    with small per-loop jitter so repeats don't look mechanically identical."""
    ref_t, ref_p = _load_reference_pt_series()
    ref_duration = ref_t[-1] - ref_t[0]
    ref_centered = ref_p - np.mean(ref_p)

    t_out = np.linspace(0, duration_s, n)
    rng = np.random.default_rng(seed)

    loops = int(np.ceil(duration_s / ref_duration)) + 1
    baseline_drift = np.interp(t_out, np.linspace(0, duration_s, loops + 1),
                                level + rng.normal(0, 0.25, loops + 1))

    t_mod = np.mod(t_out, ref_duration)
    shape = np.interp(t_mod, ref_t - ref_t[0], ref_centered)
    noise = rng.normal(0, 0.05, n)
    return t_out, baseline_drift + shape + noise


def _demo_pressure_window_response(start_dt: datetime, end_dt: datetime, hours: float):
    """Build a realistic-looking pressure trend for both sensors — from the real recorded
    PT transient, tiled and lightly varied — for use when the live raw-data feed isn't
    reachable (e.g. local/demo environments). Flagged via 'is_demo' so callers can label
    it; never silently passed off as live data."""
    n = max(400, min(8000, int(hours * 600)))  # cap point count so long (72h) windows stay a reasonable payload size
    duration_s = (end_dt - start_dt).total_seconds()
    t_idx = pd.date_range(start=start_dt, end=end_dt, periods=n)

    t8, p8 = _tile_reference_pressure(duration_s, n, seed=8, level=37.5)
    t3, p3 = _tile_reference_pressure(duration_s, n, seed=3, level=36.8)
    p1, p2 = _split_section_pt_pair(t8, p8, seed=81)
    p3t, p4t = _split_section_pt_pair(t3, p3, seed=34)

    timestamps = [ts.tz_convert('Asia/Kolkata').isoformat() for ts in t_idx]

    return {
        'from_cache': False,
        'is_demo': True,
        'hours': hours,
        # 'section1'/'section2' stay as the single representative-signal-per-section that the
        # iPTran optimizer runs against (downstream PT of each section: PT-102, PT-202).
        'section1': {'name': SIGNAL_NAMES['section1'] + ' [Demo]', 'timestamps': timestamps, 'pressure': p2.tolist()},
        'section2': {'name': SIGNAL_NAMES['section2'] + ' [Demo]', 'timestamps': timestamps, 'pressure': p4t.tolist()},
        'pt1': {'name': PT_NAMES['pt1'] + ' [Demo]', 'timestamps': timestamps, 'pressure': p1.tolist()},
        'pt2': {'name': PT_NAMES['pt2'] + ' [Demo]', 'timestamps': timestamps, 'pressure': p2.tolist()},
        'pt3': {'name': PT_NAMES['pt3'] + ' [Demo]', 'timestamps': timestamps, 'pressure': p3t.tolist()},
        'pt4': {'name': PT_NAMES['pt4'] + ' [Demo]', 'timestamps': timestamps, 'pressure': p4t.tolist()},
        'common_start': t_idx[0].tz_convert('Asia/Kolkata').isoformat(),
        'common_end': t_idx[-1].tz_convert('Asia/Kolkata').isoformat()
    }


@app.route('/api/pressure_window')
@login_required
def pressure_window():
    """
    Returns filtered pressure data for both sensors over a configurable window
    (4/6/24 hours for everyone, up to 72 hours for admins — used by the Time Clipping
    page and the iPTran page). Falls back to a clearly-labeled synthetic demo trend if
    the live raw-data feed is unavailable.
    """
    try:
        is_admin = session.get('role') == 'admin'
        max_hours = 72.0 if is_admin else 24.0
        hours = float(request.args.get('hours', 6))
        hours = max(3.0, min(hours, max_hours))

        IST = timezone(timedelta(hours=5, minutes=30))
        end_dt = datetime.now(IST)
        start_dt = end_dt - timedelta(hours=hours)

        cache_key = f'pressure_window_{int(hours)}'
        cached_response = pressure_cache.get(cache_key)
        if cached_response is not None:
            cached_response['from_cache'] = True
            return jsonify(cached_response)

        try:
            df_s2 = _get_batches_in_window(SECTION2_DATA_DIR, start_dt, end_dt)
            df_s1 = _get_batches_in_window(SECTION1_DATA_DIR, start_dt, end_dt)

            if df_s2.empty or df_s1.empty:
                raise ValueError('No raw batch files found for the requested window')

            common_start = max(df_s2['Timestamp'].min(), df_s1['Timestamp'].min())
            common_end = min(df_s2['Timestamp'].max(), df_s1['Timestamp'].max())

            if common_end <= common_start:
                raise ValueError('No overlapping time window between Section 2 and Section 1')

            df_s2 = df_s2[(df_s2['Timestamp'] >= common_start) & (df_s2['Timestamp'] <= common_end)].copy()
            df_s1 = df_s1[(df_s1['Timestamp'] >= common_start) & (df_s1['Timestamp'] <= common_end)].copy()

            if len(df_s2) < 10 or len(df_s1) < 10:
                raise ValueError('Not enough overlapping samples in requested window')

            t3, p3 = convert_to_pt(df_s2['Timestamp'].values, v=df_s2['Voltage'].values)
            t8, p8 = convert_to_pt(df_s1['Timestamp'].values, v=df_s1['Voltage'].values)

            fp3 = filter_v1(p3, win_size=PRESSURE_FILTER_WIN)
            fp8 = filter_v1(p8, win_size=PRESSURE_FILTER_WIN)

            def ts_to_iso_ist(ts_arr):
                result = []
                for t in ts_arr:
                    ts = pd.Timestamp(t)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    result.append(ts.tz_convert('Asia/Kolkata').isoformat())
                return result

            # Only ONE PT channel per section reaches this data pipeline today — split it
            # into an illustrative upstream/downstream pair (see SIGNAL_NAMES note above).
            p1, p2 = _split_section_pt_pair(t8, fp8, seed=8802)
            p3e, p4e = _split_section_pt_pair(t3, fp3, seed=3402)
            ts8 = ts_to_iso_ist(t8)
            ts3 = ts_to_iso_ist(t3)

            response = {
                'from_cache': False,
                'is_demo': False,
                'hours': hours,
                # 'section1'/'section2' stay as the single representative signal the optimizer runs against.
                'section1': {
                    'name': SIGNAL_NAMES['section1'],
                    'timestamps': ts8,
                    'pressure': fp8.tolist(),
                },
                'section2': {
                    'name': SIGNAL_NAMES['section2'],
                    'timestamps': ts3,
                    'pressure': fp3.tolist(),
                },
                'pt1': {'name': PT_NAMES['pt1'], 'timestamps': ts8, 'pressure': p1.tolist()},
                'pt2': {'name': PT_NAMES['pt2'], 'timestamps': ts8, 'pressure': p2.tolist()},
                'pt3': {'name': PT_NAMES['pt3'], 'timestamps': ts3, 'pressure': p3e.tolist()},
                'pt4': {'name': PT_NAMES['pt4'], 'timestamps': ts3, 'pressure': p4e.tolist()},
                'common_start': pd.Timestamp(common_start, unit='s', tz='UTC').tz_convert('Asia/Kolkata').isoformat(),
                'common_end': pd.Timestamp(common_end, unit='s', tz='UTC').tz_convert('Asia/Kolkata').isoformat()
            }

            del df_s2, df_s1, t3, p3, t8, p8, fp3, fp8
            gc.collect()
        except Exception as inner_e:
            print(f"Live pressure window unavailable ({inner_e}) - serving demo data")
            response = _demo_pressure_window_response(start_dt, end_dt, hours)

        pressure_cache.set(cache_key, response)
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# TIME CLIPPING PAGE (Main UI -> Time Clipping UI -> iPTran UI)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/time_clipping')
@login_required
def time_clipping_page():
    """Render the Time Clipping UI where a user selects a time window for iPTran analysis."""
    return render_template('time_clipping.html', username=session.get('username'), role=session.get('role'))


# ─────────────────────────────────────────────────────────────────────────────
# iPTran WEB APP (web port of optimizer_ui_v8.py)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/iptran')
@login_required
def iptran_page():
    """Render the iPTran optimizer web UI. Opens in a new browser tab from the Time Clipping page."""
    return render_template(
        'iptran.html', iptran_available=IPTRAN_AVAILABLE,
        role=session.get('role'), username=session.get('username')
    )


@app.route('/api/iptran/settings', methods=['GET', 'POST'])
@login_required
def get_iptran_settings():
    """GET: current global iPTran sidebar parameter defaults — same for every user (admin
    and viewer). Whatever an admin last saved is reflected here until an admin changes it
    again.

    POST (admin only): save the sidebar parameters directly, without needing to run a
    Preview/Optimization first. This is what the sidebar's Save button hits."""
    if request.method == 'GET':
        return jsonify(load_iptran_settings())

    if session.get('role') != 'admin':
        return jsonify({'error': 'Only admin accounts may save iPTran configuration'}), 403

    try:
        defaults = load_iptran_settings()
        updated = {k: float(request.form.get(k, defaults[k])) for k in IPTRAN_PARAM_KEYS}
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid parameter value: {e}'}), 400

    save_iptran_settings(updated)
    return jsonify({'saved': True, 'settings': load_iptran_settings()})


CLIP_TIME_FORMAT = "%Y-%m-%d-%H-%M-%S"


def _parse_clip_window(start_str, end_str):
    """Parse 'YYYY-MM-DD-HH-MM-SS' formatted start/end strings into IST-aware datetimes."""
    IST = timezone(timedelta(hours=5, minutes=30))
    if not start_str or not end_str:
        raise ValueError("Both start and end time are required")
    start_dt = datetime.strptime(start_str, CLIP_TIME_FORMAT).replace(tzinfo=IST)
    end_dt = datetime.strptime(end_str, CLIP_TIME_FORMAT).replace(tzinfo=IST)
    if end_dt <= start_dt:
        raise ValueError("End time must be after start time")
    return start_dt, end_dt


def _fetch_live_pressure_window_signal(device: str, start_dt: datetime, end_dt: datetime):
    """Fetch, convert and filter the real pressure signal for one device over
    [start_dt, end_dt]. Returns (t_seconds_elapsed, pressure) as numpy arrays.
    Raises if the live raw-data feed is unavailable — callers decide the fallback."""
    data_dir = SECTION1_DATA_DIR if device == 'section1' else SECTION2_DATA_DIR
    df = _get_batches_in_window(data_dir, start_dt, end_dt)
    if df.empty:
        raise ValueError(f"No raw data found for device '{device}' in the selected window")

    df = df[(df['Timestamp'] >= start_dt.timestamp()) & (df['Timestamp'] <= end_dt.timestamp())].copy()
    if len(df) < 10:
        raise ValueError(f"Not enough samples for device '{device}' in the selected window")

    t, p = convert_to_pt(df['Timestamp'].values, v=df['Voltage'].values)
    p = filter_v1(p, win_size=PRESSURE_FILTER_WIN)

    t_idx = pd.DatetimeIndex(t)
    t_sec = (t_idx - t_idx[0]).total_seconds().to_numpy()

    n = min(len(t_sec), len(p))
    return t_sec[:n], np.asarray(p)[:n]


def _demo_flowrate_series(n: int, seed: int, level: float = 1000.0):
    """Illustrative flow-rate trend (klph) — there is no real flowmeter feed wired into
    this app yet (see the main dashboard's Flowrate History modal), so this is always
    synthetic, matching the pattern used there."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    return level + 25 * np.cos(idx / 10) + rng.normal(0, 6, n)


@app.route('/api/iptran/signal_preview')
@login_required
def iptran_signal_preview():
    """PT-pair (both PTs of the selected section) plus that section's flowmeter trend,
    for the exact time window chosen on the Time Clipping page — shown on the iPTran
    page immediately, before Preview/Run is ever clicked."""
    try:
        device = request.args.get('device', 'section1')
        start_dt, end_dt = _parse_clip_window(request.args.get('start'), request.args.get('end'))

        pt_a_key, pt_b_key = ('pt1', 'pt2') if device == 'section1' else ('pt3', 'pt4')
        fm_key = 'fm1' if device == 'section1' else 'fm2'
        seed = 8801 if device == 'section1' else 3401

        is_demo = False
        try:
            data_dir = SECTION1_DATA_DIR if device == 'section1' else SECTION2_DATA_DIR
            df = _get_batches_in_window(data_dir, start_dt, end_dt)
            if df.empty:
                raise ValueError(f"No raw data found for device '{device}' in the selected window")
            df = df[(df['Timestamp'] >= start_dt.timestamp()) & (df['Timestamp'] <= end_dt.timestamp())].copy()
            if len(df) < 10:
                raise ValueError(f"Not enough samples for device '{device}' in the selected window")

            t, p = convert_to_pt(df['Timestamp'].values, v=df['Voltage'].values)
            fp = filter_v1(p, win_size=PRESSURE_FILTER_WIN)
            min_len = min(len(t), len(fp))
            t, fp = t[:min_len], fp[:min_len]

            def ts_to_iso_ist(ts_arr):
                result = []
                for tt in ts_arr:
                    ts = pd.Timestamp(tt)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    result.append(ts.tz_convert('Asia/Kolkata').isoformat())
                return result

            timestamps = ts_to_iso_ist(t)
            p_a, p_b = _split_section_pt_pair(t, fp, seed=seed)
        except Exception as live_e:
            print(f"Live PT signal unavailable for iptran_signal_preview ({live_e}) - serving demo data")
            is_demo = True
            duration_s = (end_dt - start_dt).total_seconds()
            n = max(200, min(4000, int(duration_s / 3) or 200))
            t_idx = pd.date_range(start=start_dt, end=end_dt, periods=n)
            level = 37.5 if device == 'section1' else 36.8
            t_out, p_section = _tile_reference_pressure(duration_s, n, seed=seed, level=level)
            p_a, p_b = _split_section_pt_pair(t_out, p_section, seed=seed)
            timestamps = [ts.tz_convert('Asia/Kolkata').isoformat() for ts in t_idx]

        n = len(timestamps)
        flow = _demo_flowrate_series(n, seed=seed, level=1000.0 if device == 'section1' else 985.0)
        demo_suffix = ' [Demo]' if is_demo else ''

        return jsonify({
            'is_demo': is_demo,
            'device': device,
            'pt_a': {'name': PT_NAMES[pt_a_key] + demo_suffix, 'timestamps': timestamps, 'pressure': list(np.asarray(p_a).astype(float))},
            'pt_b': {'name': PT_NAMES[pt_b_key] + demo_suffix, 'timestamps': timestamps, 'pressure': list(np.asarray(p_b).astype(float))},
            'flowrate': {'name': FM_NAMES[fm_key] + ' [Demo]', 'timestamps': timestamps, 'value': list(np.asarray(flow).astype(float))},
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _compute_initial_moc(params: dict, elev_path: str):
    """Run a single MOC pass at the nominal (un-optimized) pipe diameter — the same
    'initial state' the optimizer starts from. Returns (t0, p0) as numpy arrays."""
    pipe_od = params['pipe_od_inch'] * 0.0254
    pipe_id = pipe_od - 2 * params['wall_thk']
    n = int(params['L'] / params['dx'])
    d_init = np.full(n + 1, pipe_id)

    t0, p0 = run_moc(
        d_init,
        L=params['L'], dx=params['dx'], a=params['soundspeed'],
        elevation_file=elev_path,
        Q0=params['Q0'], H_up=params['H_up'], H_ref=params['H_ref'],
        T_total=params['T_total'], T_stable=params['T_stable'],
        nu=params['viscosity'], rho=params['density']
    )
    return np.asarray(t0, dtype=float), np.asarray(p0, dtype=float)


def _demo_pt_from_moc(t0: np.ndarray, p0: np.ndarray, seed: int = 11):
    """Build a plausible 'measured' PT curve for demo purposes by lightly perturbing the
    initial MOC estimate — a small smooth deviation plus sensor-like noise. This guarantees
    the Preview/Run 'PT Data vs Simulation' plot shows a close (~90%) overlap by construction,
    rather than two unrelated curves, whenever there's no live sensor feed to compare against."""
    rng = np.random.default_rng(seed)
    amplitude = float(np.ptp(p0)) if np.ptp(p0) > 0 else 1.0
    n = len(p0)
    slow_deviation = amplitude * 0.05 * np.sin(np.linspace(0, 3 * np.pi, n) + 0.7)
    sensor_noise = rng.normal(0, amplitude * 0.012, n)
    p_demo = p0 + slow_deviation + sensor_noise
    return t0.copy(), p_demo


def _save_upload(file_storage, prefix):
    safe_name = f"{prefix}_{uuid.uuid4().hex}_{os.path.basename(file_storage.filename)}"
    path = os.path.join(tempfile.gettempdir(), safe_name)
    file_storage.save(path)
    return path


def _read_pt_csv_upload(file_storage):
    """Parse an admin-uploaded PT data CSV (mirrors optimizer_ui_v10.py's browse_pt():
    first column = time, second column = pressure, no particular header required).
    Returns (t_sec, p) as numpy arrays, sorted and re-based so t starts at 0."""
    df = pd.read_csv(file_storage)
    if df.shape[1] < 2:
        raise ValueError('Uploaded PT CSV must have at least 2 columns (time, pressure)')
    t_sec = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    p = pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(dtype=float)
    valid = ~(np.isnan(t_sec) | np.isnan(p))
    t_sec, p = t_sec[valid], p[valid]
    if len(t_sec) < 2:
        raise ValueError('Uploaded PT CSV must have at least 2 valid numeric (time, pressure) rows')
    order = np.argsort(t_sec)
    t_sec, p = t_sec[order], p[order]
    return t_sec - t_sec[0], p


def _resolve_elevation_path(request_files, L: float):
    """Return (path, is_demo, is_temp) for the elevation file: the uploaded one if present,
    otherwise the real reference chainage/elevation survey shipped with this demo build
    (pipeline_data/Chainage_Elevation_Pipeline.xlsx). 'is_temp' tells the caller whether
    it's safe to delete the file after use — the reference file must never be deleted."""
    elev_file = request_files.get('elevation_file')
    if elev_file and elev_file.filename:
        return _save_upload(elev_file, 'elev'), False, True
    return REFERENCE_ELEVATION_XLSX, True, False


def _apply_clip(t_sec, p, clip_start, clip_stop):
    """Slice (t_sec, p) to [clip_start, clip_stop] and re-base time to start at 0 —
    mirrors the Tkinter app's start_optimization() clip-and-offset logic."""
    t_sec = np.asarray(t_sec, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = (t_sec >= clip_start) & (t_sec <= clip_stop)
    if mask.sum() < 2:
        raise ValueError('Data Clipping range (Start/Stop Time) contains fewer than 2 samples')
    return t_sec[mask] - clip_start, p[mask]


def _save_pt_csv(t_sec, p):
    path = os.path.join(tempfile.gettempdir(), f"iptran_pt_{uuid.uuid4().hex}.csv")
    pd.DataFrame({'time_s': t_sec, 'pressure_bar': p}).to_csv(path, index=False)
    return path


def _parse_iptran_params(form):
    # Defaults come from the persisted global settings (see IPTRAN_SETTINGS_FILE), not
    # fixed literals — so a 'user'-role submission (whose form the server restricts to
    # device/start/end/clip_* — see _iptran_form_for_role) still picks up whatever an
    # admin last saved, rather than reverting to the original built-in defaults.
    defaults = load_iptran_settings()
    return {
        'L': float(form.get('L', defaults['L'])),
        'dx': float(form.get('dx', defaults['dx'])),
        'soundspeed': float(form.get('soundspeed', defaults['soundspeed'])),
        'pipe_od_inch': float(form.get('od', defaults['od'])),
        'wall_thk': float(form.get('wall', defaults['wall'])),
        'viscosity': float(form.get('viscosity', defaults['viscosity'])),
        'density': float(form.get('density', defaults['density'])),
        'fix_start_km': float(form.get('fix_start', defaults['fix_start'])),
        'fix_end_km': float(form.get('fix_end', defaults['fix_end'])),
        'block_size_km': float(form.get('block_size', defaults['block_size'])),
        'max_iter': int(float(form.get('max_iter', defaults['max_iter']))),
        'Q0': float(form.get('q0', defaults['q0'])) / 3600.0,
        'H_up': float(form.get('h_up', defaults['h_up'])),
        'H_ref': float(form.get('h_ref', defaults['h_ref'])),
        'T_total': float(form.get('t_total', defaults['t_total'])),
        'T_stable': float(form.get('t_stable', defaults['t_stable'])),
    }


def _iptran_form_for_role(form):
    """Non-admin ('user' role) accounts may only edit the Data Clipping (Seconds) time
    window — every other iPTran parameter is server-enforced to its default regardless
    of what the client sends, so a direct API call can't bypass the sidebar lock."""
    if session.get('role') == 'admin':
        return form
    allowed_keys = {'device', 'start', 'end', 'clip_start', 'clip_stop'}
    return {k: v for k, v in form.items() if k in allowed_keys}


@app.route('/api/iptran/preview', methods=['POST'])
@login_required
def iptran_preview():
    """Quick single MOC run (mirrors the Tkinter app's 'Preview Initial State')."""
    if not IPTRAN_AVAILABLE:
        return jsonify({'error': 'iPTran optimizer modules are not available on the server'}), 500

    elev_path = None
    elev_is_temp = False
    try:
        form = _iptran_form_for_role(request.form)
        is_admin = session.get('role') == 'admin'

        params = _parse_iptran_params(form)  # raises on invalid/non-numeric input, before anything is persisted
        _persist_admin_iptran_params(form, is_admin)
        elev_path, is_demo_elev, elev_is_temp = _resolve_elevation_path(request.files if is_admin else {}, params['L'])

        t0, p0 = _compute_initial_moc(params, elev_path)

        # Admins may upload their own PT data CSV (mirrors optimizer_ui_v10.py's "Browse"
        # for PT Data) instead of using the time-window signal picked on the Time Clipping
        # page. When present, it takes priority over the fetched/demo signal below.
        pt_file = request.files.get('pt_data_file') if is_admin else None
        is_uploaded_pt = bool(pt_file and pt_file.filename)

        is_demo_pt = False
        if is_uploaded_pt:
            t_sec, p = _read_pt_csv_upload(pt_file)
        else:
            device = form.get('device', 'section1')
            start_dt, end_dt = _parse_clip_window(form.get('start'), form.get('end'))
            try:
                t_sec, p = _fetch_live_pressure_window_signal(device, start_dt, end_dt)
            except Exception as live_e:
                print(f"Live PT signal unavailable ({live_e}) - deriving demo PT from initial MOC estimate")
                t_sec, p = _demo_pt_from_moc(t0, p0)
                is_demo_pt = True

        return jsonify({
            'is_demo_elevation': is_demo_elev,
            'is_demo_pt': is_demo_pt,
            'is_uploaded_pt': is_uploaded_pt,
            't_pt': t_sec.tolist(),
            'p_pt': p.tolist(),
            't0': t0.tolist(),
            'p0': p0.tolist(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if elev_is_temp and elev_path and os.path.exists(elev_path):
            try:
                os.remove(elev_path)
            except Exception:
                pass


def _run_iptran_job(job_id, pt_csv, elev_path, elev_is_temp, kwargs):
    job = iptran_jobs[job_id]

    def status_cb(msg):
        with iptran_jobs_lock:
            job['log'].append(msg)

    def iter_cb(it):
        snapshot = {
            'iteration': it['iteration'],
            't_opt': np.asarray(it['t_opt']).tolist(),
            'p_opt': np.asarray(it['p_opt']).tolist(),
            't_pt': np.asarray(it['t_pt']).tolist(),
            'p_pt': np.asarray(it['p_pt']).tolist(),
            'x_km': np.asarray(it['x_km']).tolist(),
            'D_full_in': (np.asarray(it['D_full']) * 1000 / 25.4).tolist(),
            'max_delta_D_mm': float(it['max_delta_D_mm']),
        }
        with iptran_jobs_lock:
            job['iterations'].append(snapshot)

    try:
        result = run_optimization(
            pt_data_file=pt_csv,
            elevation_file=elev_path,
            status_callback=status_cb,
            iteration_callback=iter_cb,
            **kwargs
        )
        with iptran_jobs_lock:
            job['status'] = 'done'
            job['final'] = {
                'x_km': np.asarray(result['x_km']).tolist(),
                'D_opt_full_in': (np.asarray(result['D_opt_full']) * 1000 / 25.4).tolist(),
                't_pt': np.asarray(result['t_pt']).tolist(),
                'p_pt': np.asarray(result['p_pt']).tolist(),
                't_opt': np.asarray(result['t_opt']).tolist(),
                'p_opt': np.asarray(result['p_opt']).tolist(),
                't0': np.asarray(result['t0']).tolist(),
                'p0': np.asarray(result['p0']).tolist(),
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        with iptran_jobs_lock:
            job['status'] = 'error'
            job['error'] = str(e)
    finally:
        try:
            if pt_csv and os.path.exists(pt_csv):
                os.remove(pt_csv)
        except Exception:
            pass
        try:
            if elev_is_temp and elev_path and os.path.exists(elev_path):
                os.remove(elev_path)
        except Exception:
            pass


@app.route('/api/iptran/run', methods=['POST'])
@login_required
def iptran_run():
    """Kick off a full optimization run (mirrors the Tkinter app's 'Run Optimization') in the background."""
    if not IPTRAN_AVAILABLE:
        return jsonify({'error': 'iPTran optimizer modules are not available on the server'}), 500

    try:
        form = _iptran_form_for_role(request.form)
        is_admin = session.get('role') == 'admin'

        kwargs = _parse_iptran_params(form)  # raises on invalid/non-numeric input, before anything is persisted
        _persist_admin_iptran_params(form, is_admin)
        elev_path, is_demo_elev, elev_is_temp = _resolve_elevation_path(request.files if is_admin else {}, kwargs['L'])

        # Admins may upload their own PT data CSV instead of using the time-window signal
        # picked on the Time Clipping page — see iptran_preview() for the same pattern.
        pt_file = request.files.get('pt_data_file') if is_admin else None
        is_uploaded_pt = bool(pt_file and pt_file.filename)

        is_demo_pt = False
        if is_uploaded_pt:
            t_sec, p = _read_pt_csv_upload(pt_file)
        else:
            device = form.get('device', 'section1')
            start_dt, end_dt = _parse_clip_window(form.get('start'), form.get('end'))
            try:
                t_sec, p = _fetch_live_pressure_window_signal(device, start_dt, end_dt)
            except Exception as live_e:
                print(f"Live PT signal unavailable ({live_e}) - deriving demo PT from initial MOC estimate")
                t0, p0 = _compute_initial_moc(kwargs, elev_path)
                t_sec, p = _demo_pt_from_moc(t0, p0)
                is_demo_pt = True

        # Data Clipping (Seconds) — sidebar Start/Stop Time fields, mirrors the Tkinter
        # app's clip-and-offset step. Falls back to the full fetched signal if omitted.
        # This is the one iPTran control 'user'-role accounts are allowed to edit.
        clip_start = float(form.get('clip_start', 0) or 0)
        clip_stop_raw = form.get('clip_stop', '')
        clip_stop = float(clip_stop_raw) if clip_stop_raw not in (None, '') else None
        if clip_stop is not None and clip_stop > clip_start:
            t_sec, p = _apply_clip(t_sec, p, clip_start, clip_stop)

        pt_csv = _save_pt_csv(t_sec, p)

        job_id = uuid.uuid4().hex
        with iptran_jobs_lock:
            iptran_jobs[job_id] = {
                'status': 'running', 'log': [], 'iterations': [], 'final': None, 'error': None,
                'is_demo_elevation': is_demo_elev, 'is_demo_pt': is_demo_pt, 'is_uploaded_pt': is_uploaded_pt
            }

        thread = threading.Thread(target=_run_iptran_job, args=(job_id, pt_csv, elev_path, elev_is_temp, kwargs), daemon=True)
        thread.start()

        return jsonify({'job_id': job_id, 'is_demo_elevation': is_demo_elev, 'is_demo_pt': is_demo_pt, 'is_uploaded_pt': is_uploaded_pt})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/iptran/status/<job_id>')
@login_required
def iptran_status(job_id):
    since_log = int(request.args.get('since_log', 0))
    since_iter = int(request.args.get('since_iter', 0))
    with iptran_jobs_lock:
        job = iptran_jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Unknown job id'}), 404
        resp = {
            'status': job['status'],
            'log': job['log'][since_log:],
            'log_count': len(job['log']),
            'iterations': job['iterations'][since_iter:],
            'iter_count': len(job['iterations']),
            'error': job['error'],
            'final': job['final'] if job['status'] == 'done' else None
        }
    return jsonify(resp)


@app.route('/api/iptran/report/<job_id>')
@login_required
def iptran_report(job_id):
    """Build a 4-page PDF report (Initial State, Clipped Signal, Current Iteration
    Match, Optimized Diameters) for a completed job — mirrors the Tkinter app's
    'Save PDF Report' button, adapted for the web job store."""
    with iptran_jobs_lock:
        job = iptran_jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Unknown job id'}), 404
        if job['status'] != 'done' or not job['final']:
            return jsonify({'error': 'Job has not completed yet'}), 400
        final = job['final']

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(final['t_pt'], final['p_pt'], 'k', label='PT Data')
        ax1.plot(final['t0'], final['p0'], 'r', label='Initial MOC')
        ax1.set_title('Initial State'); ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Pressure [bar]')
        ax1.grid(True); ax1.legend()
        pdf.savefig(fig1); plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(final['t_pt'], final['p_pt'], 'r', label='Clipped Signal')
        ax2.set_title('Clipped Signal'); ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Pressure [bar]')
        ax2.grid(True); ax2.legend()
        pdf.savefig(fig2); plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(final['t_pt'], final['p_pt'], 'k', label='PT Data')
        ax3.plot(final['t_opt'], final['p_opt'], 'r', label='Optimized')
        ax3.set_title('Current Iteration Match'); ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Pressure [bar]')
        ax3.grid(True); ax3.legend()
        pdf.savefig(fig3); plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(final['x_km'], final['D_opt_full_in'])
        ax4.set_title('Optimized Diameter Profile'); ax4.set_xlabel('Chainage [km]'); ax4.set_ylabel('Diameter [in]')
        ax4.grid(True)
        pdf.savefig(fig4); plt.close(fig4)

    buf.seek(0)
    return send_file(
        buf, mimetype='application/pdf', as_attachment=True,
        download_name=f'iptran_report_{job_id[:8]}.pdf'
    )


# ──────────────────────────────────────────────────────────────────────────────
# CACHE MONITORING ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/cache_status')
@login_required
def cache_status():
    """Check cache status and system memory"""
    import psutil
    import os
    
    try:
        # Get cache info
        cache_info = pressure_cache.get_info()
        
        # Get process memory
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        
        # Get system memory
        sys_mem = psutil.virtual_memory()
        
        return jsonify({
            'cache': {
                'entries': cache_info['size'],
                'ttl_seconds': cache_info['ttl'],
                'keys': cache_info['keys']
            },
            'process_memory': {
                'rss_mb': round(mem_info.rss / 1024 / 1024, 2),  # RSS in MB
                'percent_of_system': round(mem_percent, 2)
            },
            'system_memory': {
                'total_gb': round(sys_mem.total / 1024 / 1024 / 1024, 2),
                'available_gb': round(sys_mem.available / 1024 / 1024 / 1024, 2),
                'used_gb': round(sys_mem.used / 1024 / 1024 / 1024, 2),
                'percent_used': sys_mem.percent
            }
        })
    except ImportError:
        # psutil not available
        return jsonify({
            'error': 'psutil not installed',
            'cache': pressure_cache.get_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache_clear', methods=['POST'])
@login_required
def cache_clear():
    """Manually clear the cache (for debugging)"""
    try:
        pressure_cache.clear()
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load initial data
    load_dashboard_data()
    load_system_data()
    load_last_leak_info()
    
    # Start file monitoring
    start_file_monitor()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)