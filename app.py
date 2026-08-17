from flask import Flask, render_template, jsonify, send_file, Response, request
import io
import json
import os
import csv
import glob
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
    return render_template('dashboard_html.html')

@app.route('/api/data')
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
def get_historical_leaks():
    """API endpoint to get historical leak logs"""
    return jsonify({'data': dashboard_data['historical_leaks']})

@app.route('/download_csv')
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

# Raw data directories on the EC2 instance
BFA3_DATA_DIR = "/home/ubuntu/mnt/ebs/BFA3"  # Asadpur
BFA8_DATA_DIR = "/home/ubuntu/mnt/ebs/BFA8"  # Rewari

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


@app.route('/api/pressure_plot')
def pressure_plot():
    """
    Returns filtered pressure data for both sensors (last 1 hr).
    Uses caching to prevent reloading data from disk on every request.
    
    Steps:
      1. Check cache first (120s TTL)
      2. If cache miss → Read last 1 hr batch CSVs from BFA3 and BFA8 dirs
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
        
        # Cache miss - load data from disk
        df_bfa3 = _get_last_1hr_batches(BFA3_DATA_DIR)
        df_bfa8 = _get_last_1hr_batches(BFA8_DATA_DIR)

        if df_bfa3.empty or df_bfa8.empty:
            return jsonify({
                'error': 'Insufficient data',
                'detail': f'BFA3 rows: {len(df_bfa3)}, BFA8 rows: {len(df_bfa8)}'
            }), 400

        # ── Common time window ──
        common_start = max(df_bfa3['Timestamp'].min(), df_bfa8['Timestamp'].min())
        common_end = min(df_bfa3['Timestamp'].max(), df_bfa8['Timestamp'].max())

        if common_end <= common_start:
            return jsonify({'error': 'No overlapping time window between BFA3 and BFA8'}), 400

        df_bfa3 = df_bfa3[(df_bfa3['Timestamp'] >= common_start) & (df_bfa3['Timestamp'] <= common_end)].copy()
        df_bfa8 = df_bfa8[(df_bfa8['Timestamp'] >= common_start) & (df_bfa8['Timestamp'] <= common_end)].copy()

        if len(df_bfa3) < 100 or len(df_bfa8) < 100:
            return jsonify({'error': 'Not enough overlapping samples', 
                            'detail': f'BFA3={len(df_bfa3)}, BFA8={len(df_bfa8)}'}), 400

        # Check authorization
        is_admin = False
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

        # ── Voltage → Pressure ──
        t_bfa3, p_bfa3 = convert_to_pt(df_bfa3['Timestamp'].values, v=df_bfa3['Voltage'].values)
        t_bfa8, p_bfa8 = convert_to_pt(df_bfa8['Timestamp'].values, v=df_bfa8['Voltage'].values)

        # Compute raw stats before decimation
        if is_admin:
            bfa3_min, bfa3_max, bfa3_std = float(np.min(p_bfa3)), float(np.max(p_bfa3)), float(np.std(p_bfa3))
            bfa8_min, bfa8_max, bfa8_std = float(np.min(p_bfa8)), float(np.max(p_bfa8)), float(np.std(p_bfa8))
        else:
            bfa3_min = bfa3_max = bfa3_std = 0.0
            bfa8_min = bfa8_max = bfa8_std = 0.0

        # ── Filter with filter_v1 at window 500 ──
        fp_bfa3 = filter_v1(p_bfa3, win_size=PRESSURE_FILTER_WIN)
        fp_bfa8 = filter_v1(p_bfa8, win_size=PRESSURE_FILTER_WIN)
        # fp_bfa3 = p_bfa3
        # fp_bfa8 = p_bfa8

        # ── Decimate twice (factor 2 each → 4× total reduction) ──
        from scipy.signal import decimate as sp_decimate
        Q=10 #decimation factor for anti-aliasing filter

        # Decimating 3 times
        # fp_bfa3 = sp_decimate(x=fp_bfa3, q=Q)
        # fp_bfa3 = sp_decimate(x=fp_bfa3, q=Q)
        # fp_bfa3 = sp_decimate(x=fp_bfa3, q=Q)
        length_after_decimation_3 = len(fp_bfa3)


        # fp_bfa8 = sp_decimate(x=fp_bfa8, q=Q)
        # fp_bfa8 = sp_decimate(x=fp_bfa8, q=Q)
        # fp_bfa8 = sp_decimate(x=fp_bfa8, q=Q)
        length_after_decimation_8 = len(fp_bfa8)

        if is_admin:
            # dec_p_bfa3 = sp_decimate(x=p_bfa3, q=Q)
            # dec_p_bfa3 = sp_decimate(x=dec_p_bfa3, q=Q)
            dec_p_bfa3 = p_bfa3
            dec_p_bfa8 = p_bfa8
            
            # dec_p_bfa8 = sp_decimate(x=p_bfa8, q=Q)
            # dec_p_bfa8 = sp_decimate(x=dec_p_bfa8, q=Q)

        # # Downsample timestamps to match decimated signals (Q^3 = 1000x reduction)
        # total_decimation = Q * Q * Q
        # t_bfa3_down = t_bfa3[::total_decimation]
        # t_bfa8_down = t_bfa8[::total_decimation]

        t_bfa3_down=t_bfa3
        t_bfa8_down=t_bfa8
        
        # Ensure lengths match for decimated data
        min_len_3_dec = min(len(t_bfa3_down), len(fp_bfa3))
        min_len_8_dec = min(len(t_bfa8_down), len(fp_bfa8))

        t_bfa3_down = t_bfa3_down[:min_len_3_dec]
        fp_bfa3 = fp_bfa3[:min_len_3_dec]

        t_bfa8_down = t_bfa8_down[:min_len_8_dec]
        fp_bfa8 = fp_bfa8[:min_len_8_dec]

        # For raw data, we send the full signal (or slightly downsampled if it's too massive, 
        # but following "removed decimation" we'll try full/minimal downsampling)
        if is_admin:
            # We'll use a small slice to avoid crashing the browser if it's > 100k points
            # but keep it high-res enough to be "raw". ::5 is usually a good compromise.
            raw_down = 1 
            t_bfa3_raw = t_bfa3[::raw_down]
            t_bfa8_raw = t_bfa8[::raw_down]
            p_bfa3_raw = p_bfa3[::raw_down]
            p_bfa8_raw = p_bfa8[::raw_down]
        else:
            t_bfa3_raw = t_bfa8_raw = p_bfa3_raw = p_bfa8_raw = []

        # ── Prepare response ──
        # Convert datetime64 timestamps to IST ISO strings for JS
        IST_TZ = timezone(timedelta(hours=5, minutes=30))

        def ts_to_iso_ist(ts_arr):
            result = []
            for t in ts_arr:
                ts = pd.Timestamp(t)
                # Localize to UTC first, then convert to IST
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                ts_ist = ts.tz_convert('Asia/Kolkata')
                result.append(ts_ist.isoformat())
            return result

        response = {
            'is_admin': is_admin,
            'from_cache': False,
            'bfa8': {
                'name': 'BFA8 – Rewari',
                'timestamps': ts_to_iso_ist(t_bfa8_down),
                'pressure': fp_bfa8.tolist(),
                'timestamps_raw': ts_to_iso_ist(t_bfa8_raw) if is_admin else [],
                'raw_pressure': p_bfa8_raw.tolist() if is_admin else [],
                'stats': {'min': bfa8_min, 'max': bfa8_max, 'std': bfa8_std} if is_admin else {}
            },
            'bfa3': {
                'name': 'BFA3 – Asadpur',
                'timestamps': ts_to_iso_ist(t_bfa3_down),
                'pressure': fp_bfa3.tolist(),
                'timestamps_raw': ts_to_iso_ist(t_bfa3_raw) if is_admin else [],
                'raw_pressure': p_bfa3_raw.tolist() if is_admin else [],
                'stats': {'min': bfa3_min, 'max': bfa3_max, 'std': bfa3_std} if is_admin else {}
            },
            'common_start': pd.Timestamp(common_start, unit='s', tz='UTC').tz_convert('Asia/Kolkata').isoformat(),
            'common_end': pd.Timestamp(common_end, unit='s', tz='UTC').tz_convert('Asia/Kolkata').isoformat()
        }
        
        # Store in cache for next 120 seconds
        pressure_cache.set(cache_key, response)
        
        # Log timing (capture counts before cleanup)
        bfa3_count = len(df_bfa3)
        bfa8_count = len(df_bfa8)
        
        # Clean up DataFrames to free memory
        del df_bfa3, df_bfa8, t_bfa3, p_bfa3, t_bfa8, p_bfa8, fp_bfa3, fp_bfa8
        gc.collect()
        
        res = jsonify(response)
        
        # Log timing
        duration = time.time() - start_time
        try:
            with open(PRESSURE_LOG_FILE, 'a') as f:
                f.write(f"{ist_time.strftime('%Y-%m-%d %H:%M:%S')} - Duration: {duration:.4f}s (cache: False, from_disk)\n")
                f.write(f"  BFA3 samples: {bfa3_count}, BFA8 samples: {bfa8_count}\n")
                f.write(f"  BFA3 decimated: {length_after_decimation_3}, BFA8 decimated: {length_after_decimation_8}\n")
        except Exception as log_e:
            print(f"Error writing to timing log: {log_e}")
            
        return res

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# CACHE MONITORING ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/cache_status')
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