
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import os
import datetime
import pandas as pd
import numpy as np
import json
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Try importing PIL for image handling
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError: 
    HAS_PIL = False
 
# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from optimizer2_v2 import run_optimization
    from moc_core import run_moc
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import optimizer module: {e}")
    sys.exit(1)

class OptimizationApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("iPTran Software v10")
        self.geometry("1400x900")
        
        # Style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        
        # Highlight style for Run button
        self.style.configure('Highlight.TButton', font=('Helvetica', 12, 'bold'), foreground='blue', background="#7affa6")
        
        # Layout
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.sidebar_container = ttk.Frame(self.main_container, width=400, padding=10)
        self.sidebar_container.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_container.pack_propagate(False) # Keep width fixed

        # --- Scrollable Sidebar ---
        self.canvas = tk.Canvas(self.sidebar_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.canvas.yview)
        self.sidebar = ttk.Frame(self.canvas)

        self.sidebar.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        
        # Ensure the inner frame takes the full width of the canvas
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel support
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.content = ttk.Frame(self.main_container, padding=10)
        self.content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Variables
        self.pt_file_path = tk.StringVar()
        self.elev_file_path = tk.StringVar()
        
        # Defaults for Single Valve (Optimizer2)
        self.L_var = tk.DoubleVar(value=69500.0)
        self.dx_var = tk.DoubleVar(value=500.0)
        self.sound_speed_var = tk.DoubleVar(value=820.0)
        self.od_var = tk.DoubleVar(value=10.75)
        self.wall_var = tk.DoubleVar(value=0.0071)
        
        self.fix_start_var = tk.DoubleVar(value=5.0)
        self.fix_end_var = tk.DoubleVar(value=0.5)
        self.block_size_var = tk.DoubleVar(value=3.0)
        self.max_iter_var = tk.IntVar(value=40)
        
        self.viscosity_var = tk.DoubleVar(value=0.2e-6)
        self.density_var = tk.DoubleVar(value=531.65)
        
        # MOC simulation parameters
        self.q0_var = tk.DoubleVar(value=50.0) 
        self.h_up_var = tk.DoubleVar(value=660.0)
        self.h_ref_var = tk.DoubleVar(value=643.0)
        self.t_total_var = tk.DoubleVar(value=240.0)
        self.t_stable_var = tk.DoubleVar(value=25.0)
        
        # Clipping Variables
        self.start_t_var = tk.DoubleVar(value=0.0)
        self.stop_t_var = tk.DoubleVar(value=0.0)
        self.df_pt = None
        
        # Store figures for PDF export
        self.generated_figures = {}
        
        # Store initial data for optimization
        self.initial_data = None
        self.results_dir = None

        self.create_sidebar()
        self.create_content_area()
        
        self.running = False

    def _on_canvas_configure(self, event):
        # Update the width of the sidebar frame to match the canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_sidebar(self):
        # Logo Integration
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        if os.path.exists(logo_path):
            try:
                if HAS_PIL:
                    pil_img = Image.open(logo_path)
                    base_width = 300
                    w_percent = (base_width / float(pil_img.size[0]))
                    h_size = int((float(pil_img.size[1]) * float(w_percent)))
                    pil_img = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                    self.logo_img = ImageTk.PhotoImage(pil_img)
                else:
                    self.logo_img = tk.PhotoImage(file=logo_path)
                
                logo_lbl = ttk.Label(self.sidebar, image=self.logo_img)
                logo_lbl.pack(pady=(0, 10), anchor="center")
            except Exception as e:
                print(f"Error loading logo: {e}")
        
        # Header
        lbl = ttk.Label(self.sidebar, text="iPTran Software", font=("Helvetica", 16, "bold"), foreground="blue")
        lbl.pack(pady=(0, 20), anchor="center")
        
        # --- Files ---
        file_frame = ttk.LabelFrame(self.sidebar, text="Input Files", padding=5)
        file_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(file_frame, text="PT Data (CSV):").pack(anchor="w")
        pt_box = ttk.Frame(file_frame)
        pt_box.pack(fill=tk.X)
        ttk.Entry(pt_box, textvariable=self.pt_file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(pt_box, text="Browse", command=self.browse_pt).pack(side=tk.RIGHT)
        
        ttk.Label(file_frame, text="Elevation Data:").pack(anchor="w", pady=(2,0))
        el_box = ttk.Frame(file_frame)
        el_box.pack(fill=tk.X)
        ttk.Entry(el_box, textvariable=self.elev_file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(el_box, text="Browse", command=self.browse_elev).pack(side=tk.RIGHT)

        # --- Data Clipping ---
        clip_frame = ttk.LabelFrame(self.sidebar, text="Data Clipping (Seconds)", padding=5)
        clip_frame.pack(fill=tk.X, pady=2)
        
        self.create_entry(clip_frame, "Start Time [s]:", self.start_t_var)
        self.create_entry(clip_frame, "Stop Time [s]:", self.stop_t_var)
        ttk.Button(clip_frame, text="Update Plot Preview", command=self.plot_initial_data).pack(fill=tk.X, pady=2)

        # --- Geometry ---
        geo_frame = ttk.LabelFrame(self.sidebar, text="Geometry", padding=5)
        geo_frame.pack(fill=tk.X, pady=2)
        
        self.create_entry(geo_frame, "Length (L) [m]:", self.L_var)
        self.create_entry(geo_frame, "Pipe OD [inch]:", self.od_var)
        self.create_entry(geo_frame, "Wall Thickness [m]:", self.wall_var)

        # --- Grid ---
        grid_frame = ttk.LabelFrame(self.sidebar, text="Grid", padding=5)
        grid_frame.pack(fill=tk.X, pady=2)
        
        self.create_entry(grid_frame, "Grid Step (dx) [m]:", self.dx_var)
        self.create_entry(grid_frame, "Sound Speed [m/s]:", self.sound_speed_var)

        # --- Pipeline Properties ---
        prop_frame = ttk.LabelFrame(self.sidebar, text="Pipeline Properties", padding=5)
        prop_frame.pack(fill=tk.X, pady=2)
        
        self.create_entry(prop_frame, "Viscosity [m²/s]:", self.viscosity_var)
        self.create_entry(prop_frame, "Density (rho) [kg/m³]:", self.density_var)
        
        # --- MOC Parameters ---
        moc_frame = ttk.LabelFrame(self.sidebar, text="MOC Simulation (Single Valve)", padding=5)
        moc_frame.pack(fill=tk.X, pady=2)
        
        self.create_entry(moc_frame, "Flow Rate (Q0) [m³/h]:", self.q0_var)
        self.create_entry(moc_frame, "Upstream Head (H_up) [m]:", self.h_up_var)
        self.create_entry(moc_frame, "Reference Head (H_ref) [m]:", self.h_ref_var)
        self.create_entry(moc_frame, "Total Time (T_total) [s]:", self.t_total_var)
        self.create_entry(moc_frame, "Stabilization Time [s]:", self.t_stable_var)

        # --- Advanced Settings Toggle ---
        self.show_advanced = tk.BooleanVar(value=False)
        adv_chk = ttk.Checkbutton(self.sidebar, text="Show Advanced Settings", 
                                  variable=self.show_advanced, command=self.toggle_advanced)
        adv_chk.pack(anchor="w", pady=(10, 5))

        # --- Optimization (Advanced) ---
        self.opt_frame = ttk.LabelFrame(self.sidebar, text="Advanced Settings", padding=5)
        
        self.create_entry(self.opt_frame, "Fixed Start Length [km]:", self.fix_start_var)
        self.create_entry(self.opt_frame, "Fixed End Length [km]:", self.fix_end_var)
        self.create_entry(self.opt_frame, "Block Size [km]:", self.block_size_var)
        self.create_entry(self.opt_frame, "Max Iterations:", self.max_iter_var)

        # --- Actions ---
        self.preview_btn = ttk.Button(self.sidebar, text="INITIAL DATA PLOT", command=self.plot_initial_data)
        self.preview_btn.pack(fill=tk.X, pady=5)
        
        self.run_btn = ttk.Button(self.sidebar, text="Run iPTran Application", command=self.start_optimization, state='disabled', style='Highlight.TButton')
        self.run_btn.pack(fill=tk.X, pady=5)
        
        self.save_btn = ttk.Button(self.sidebar, text="SAVE PDF REPORT", command=self.save_report, state='disabled')
        self.save_btn.pack(fill=tk.X, pady=5)
        
        # Exit button
        self.exit_btn = ttk.Button(self.sidebar, text="EXIT", command=self.safe_exit)
        self.exit_btn.pack(fill=tk.X, pady=5)

        # Help Text
        help_lbl = ttk.Label(self.sidebar, text="Note: Single-valve logic results are saved per iteration in 'v10_results' folder.", wraplength=350, foreground="gray")
        help_lbl.pack(fill=tk.X)

    def toggle_advanced(self):
        if self.show_advanced.get():
            self.opt_frame.pack(fill=tk.X, pady=5, before=self.preview_btn)
        else:
            self.opt_frame.pack_forget()

    def create_entry(self, parent, label, variable):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=label, width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=variable, width=10).pack(side=tk.RIGHT)

    def create_content_area(self):
        # --- Plots ---
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_initial = ttk.Frame(self.notebook)
        self.tab_clipped = ttk.Frame(self.notebook)
        self.tab_final = ttk.Frame(self.notebook)
        self.tab_diameter = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_initial, text="Initial State")
        self.notebook.add(self.tab_clipped, text="Clipped Signal")
        self.notebook.add(self.tab_final, text="Current Iteration Match")
        self.notebook.add(self.tab_diameter, text="Current Optimized Diameters")
        
        for tab in [self.tab_initial, self.tab_clipped, self.tab_final, self.tab_diameter]:
            ttk.Label(tab, text="Plots will appear here after execution.", font=("Arial", 14), foreground="gray").pack(expand=True)

    def browse_pt(self):
        filename = filedialog.askopenfilename(title="Select PT Data CSV", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filename:
            self.pt_file_path.set(filename)
            try:
                self.df_pt = pd.read_csv(filename)
                t_min = float(self.df_pt.iloc[0, 0])
                t_max = float(self.df_pt.iloc[-1, 0])
                self.start_t_var.set(round(t_min, 2))
                self.stop_t_var.set(round(t_max, 2))
                self.plot_initial_data()
                self.run_btn.configure(state='normal', style='Highlight.TButton')
                self.log(f"Loaded PT data: {len(self.df_pt)} points. Time range: {t_min:.1f} to {t_max:.1f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def browse_elev(self):
        filename = filedialog.askopenfilename(title="Select Elevation File", filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")])
        if filename:
            self.elev_file_path.set(filename)

    def log(self, msg):
        # Logging disabled - removed execution log UI
        pass

    def plot_initial_data(self):
        if self.df_pt is None:
            return
            
        fig1 = Figure(figsize=(6, 4))
        ax = fig1.add_subplot(111)
        
        t = self.df_pt.iloc[:, 0].values
        p = self.df_pt.iloc[:, 1].values
        
        ax.plot(t, p, 'b-', label="Full PT Data", alpha=0.5)
        
        # Clip markers
        t_start = self.start_t_var.get()
        t_stop = self.stop_t_var.get()
        
        ax.axvline(x=t_start, color='green', linestyle='--', label="Start Clip")
        ax.axvline(x=t_stop, color='red', linestyle='--', label="Stop Clip")
        
        # Highlight clipped region
        mask = (t >= t_start) & (t <= t_stop)
        if any(mask):
            ax.plot(t[mask], p[mask], 'r-', label="Clipped Region")
            
        ax.legend()
        ax.set_title("Initial PT Data Plot (Select Clipping Range)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [bar]")
        ax.grid(True)
        
        self.display_plot_figure(self.tab_initial, fig1)
        self.generated_figures['initial'] = fig1
        
        # Plot clipped data in the new tab
        if any(mask):
            fig2 = Figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(111)
            ax2.plot(t[mask], p[mask], 'r-', label="Clipped Signal")
            ax2.set_title("Clipped Signal Only")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Pressure [bar]")
            ax2.grid(True)
            ax2.legend()
            self.display_plot_figure(self.tab_clipped, fig2)
            self.generated_figures['clipped'] = fig2
            
        self.notebook.select(self.tab_initial) # Keep focus on initial for now, or maybe clipped?
        # If the user just updated the clip, they might want to see the clipped tab.
        # But 'plot_initial_data' is also called on browse.
        # Let's stay on initial for now as per current behavior, or switch to clipped if it was an explicit 'Update'?
        # The user said "after the clipped time is selected a new tab ... is made", 
        # which implies they want to see it.
        # Maybe select it? 
        # Let's keep it as is for browse, but if they click "Update Plot Preview" they might expect it.
        # Actually, the browse also calls plot_initial_data.


    def preview_initial(self):
        # Kept for compatibility if needed, but not used by main buttons anymore
        pass

    def start_optimization(self):
        if self.running: return
        if self.df_pt is None:
            messagebox.showerror("Error", "Load PT data first.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"v10_results_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

        # Create clipped CSV
        try:
            t_start = self.start_t_var.get()
            t_end = self.stop_t_var.get()
            mask = (self.df_pt.iloc[:, 0] >= t_start) & (self.df_pt.iloc[:, 0] <= t_end)
            clipped_df = self.df_pt.loc[mask].copy()
            # Reset time to start from 0 for the MOC/Optimizer? 
            # Usually better to offset if the logic expects t=0 at start
            # But the user might want original timestamps. 
            # Let's subtract t_start to be safe for the MOC logic which usually starts at 0.
            clipped_df.iloc[:, 0] = clipped_df.iloc[:, 0] - t_start
            
            self.clipped_pt_path = os.path.join(self.results_dir, "clipped_pt_data.csv")
            clipped_df.to_csv(self.clipped_pt_path, index=False)
            self.log(f"Saved clipped data ({len(clipped_df)} points) to {self.clipped_pt_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clip data: {e}")
            return

        self.running = True
        self.run_btn.configure(state='disabled', text="Processing...", style='Highlight.TButton')
        self.preview_btn.configure(state='disabled')
        threading.Thread(target=self.run_thread, daemon=True).start()

    def run_thread(self):
        try:
            params = {
                "pt_data_file": self.clipped_pt_path,
                "elevation_file": self.elev_file_path.get(),
                "L": self.L_var.get(),
                "dx": self.dx_var.get(),
                "soundspeed": self.sound_speed_var.get(),
                "pipe_od_inch": self.od_var.get(),
                "wall_thk": self.wall_var.get(),
                "fix_start_km": self.fix_start_var.get(),
                "fix_end_km": self.fix_end_var.get(),
                "block_size_km": self.block_size_var.get(),
                "max_iter": self.max_iter_var.get(),
                "status_callback": self.thread_log,
                "iteration_callback": self.on_iteration_step,
                "Q0": self.q0_var.get()/3600.0,
                "H_up": self.h_up_var.get(),
                "H_ref": self.h_ref_var.get(),
                "T_total": self.t_total_var.get(),
                "T_stable": self.t_stable_var.get()
            }
            # Save config
            with open(os.path.join(self.results_dir, "config.json"), "w") as f:
                json.dump({k:v for k,v in params.items() if not callable(v)}, f, indent=4)

            results = run_optimization(**params)
            self.after(0, lambda: self.on_success(results))
        except Exception as e:
            self.after(0, lambda: self.on_error(str(e)))

    def thread_log(self, msg):
        self.after(0, lambda: self.log(msg))

    def on_iteration_step(self, it):
        n = it["iteration"]
        # Save CSV
        df = pd.DataFrame({"x_km": it["x_km"], "D_m": it["D_full"]})
        df.to_csv(os.path.join(self.results_dir, f"diam_iter_{n:03d}.csv"), index=False)
        
        # Save Plots
        f1 = plt.figure(figsize=(10,6)); plt.plot(it['t_pt'], it['p_pt'], 'k'); plt.plot(it['t_opt'], it['p_opt'], 'r')
        plt.title(f"Iter {n} Match"); plt.grid(True); plt.savefig(os.path.join(self.results_dir, f"match_{n:03d}.png")); plt.close(f1)
        
        # Save State JSON
        with open(os.path.join(self.results_dir, f"state_{n:03d}.json"), "w") as f:
            json.dump({"iteration": n, "max_dD_mm": it["max_delta_D_mm"]}, f)

        self.after(0, lambda: self.update_iter_plots(it))

    def update_iter_plots(self, it):
        n = it["iteration"]
        fig_m = Figure(figsize=(6, 4)); ax = fig_m.add_subplot(111)
        ax.plot(it['t_pt'], it['p_pt'], 'k', label="PT Data")
        ax.plot(it['t_opt'], it['p_opt'], 'r', label=f"Iter {n}")
        ax.set_title(f"Pressure Match (Iter {n})"); ax.grid(True); ax.legend()
        self.display_plot_figure(self.tab_final, fig_m)
        self.generated_figures['final'] = fig_m
        
        fig_d = Figure(figsize=(6, 4)); axd = fig_d.add_subplot(111)
        axd.plot(it['x_km'], it['D_full'] * 1000 / 25.4)
        axd.set_title(f"Diameter Profile (Iter {n})"); axd.grid(True)
        self.display_plot_figure(self.tab_diameter, fig_d)
        self.generated_figures['diameter'] = fig_d
        
        self.notebook.select(self.tab_final)

    def on_success(self, res):
        self.running = False
        self.run_btn.configure(state='normal', text="Run iPTran Application", style='Highlight.TButton')
        self.preview_btn.configure(state='normal', text="INITIAL DATA PLOT")
        self.save_btn.configure(state='normal')
        self.log(f"Process Finished. Results in {self.results_dir}")
        messagebox.showinfo("Success", "Process complete.")

    def on_error(self, e):
        self.running = False
        self.run_btn.configure(state='normal', text="Run iPTran Application", style='Highlight.TButton')
        self.preview_btn.configure(state='normal', text="INITIAL DATA PLOT")
        self.log(f"Error: {e}")
        messagebox.showerror("Error", e)

    def display_plot_figure(self, parent_frame, fig):
        for widget in parent_frame.winfo_children(): widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def safe_exit(self):
        if self.running:
            if not messagebox.askyesno("Exit", "Process running. Exit?"): return
        self.quit(); self.destroy()

    def save_report(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not file_path: return
        try:
            with PdfPages(file_path) as pdf:
                for k in ['initial', 'clipped', 'final', 'diameter']:
                    if k in self.generated_figures: pdf.savefig(self.generated_figures[k])
            messagebox.showinfo("Saved", "Report saved.")
        except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
