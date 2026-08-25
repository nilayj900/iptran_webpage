import numpy as np
import pandas as pd

def run_moc(D_array,
            L,
            dx,
            a=812.0,
            g=9.81,
            rho=531.65,
            nu=0.2e-6,
            eps=100e-6,
            PIPE_OD=10.75*0.0254,
            wall_thk=0.0071,
            Q0=50.0/3600.0,
            H_up=660.0,
            H_ref=643.0,
            T_total=240.0,
            T_stable=50.0,
            elevation_file = None,
            enable_upstream_valve = False):

    # ---------------- Grid ----------------
    N = int(L/dx)
    dt = dx/a
    nt = int(T_total/dt)
    x = np.linspace(0, L, N+1)

    # ---------------- Elevation ----------------
    if elevation_file is not None:
        elevation_df = pd.read_excel(elevation_file, sheet_name='Sheet1')
        chainage = elevation_df.iloc[:,0]
        elevation =  elevation_df.iloc[:,1]
        z = np.interp(x, list(chainage), list(elevation))
    else:
        z = np.zeros_like(x)


    # ---------------- Valve timing ----------------
    t_close = T_stable + 5.0
    dur_close = 5.0 #86sec 
    hold = 20.0 #
    t_open = t_close + dur_close + hold
    dur_open = 5.0
    se = 0.001
    final_pos_valve = 0.5

    def valve_tau(t):
        # Before closure
        if t < t_close:
            return 1.0

        # Closing ramp: 1.0 → se
        elif t < t_close + dur_close:
            return 1.0 - (1.0 - se) * (t - t_close) / dur_close

        # Hold at minimum opening
        elif t < t_open:
            return se

        # Opening ramp: se → final_pos_valve
        elif t < t_open + dur_open:
            return se + (final_pos_valve - se) * (t - t_open) / dur_open

        # Final steady opening
        else:
            return final_pos_valve
            
    valve_tau_up = valve_tau

    # ---------------- Friction ----------------
    def friction_factor(v, D):
        Re = abs(v)*D/nu
        if Re < 2000:
            return 64/max(Re,1)
        f = 0.25/(np.log10(eps/(3.7*D)+5.74/Re**0.9)**2)
        return 0.95*f

    # ---------------- Initial conditions (UNCHANGED) ----------------
    H = np.linspace(H_up, H_ref, N+1)
    Q = np.full(N+1, Q0)

    DELTA_H_REF = 1.0
    Cv = Q0/np.sqrt(DELTA_H_REF)

    p_down = np.zeros(nt)
    time = np.arange(nt)*dt

    # ---------------- MOC Loop ----------------
    for n in range(nt):
        t = n*dt
        Hn = H.copy()
        Qn = Q.copy()

        for i in range(1, N):
            A_L = np.pi*(D_array[i-1]**2)/4
            A_R = np.pi*(D_array[i+1]**2)/4

            vL = Q[i-1]/A_L
            vR = Q[i+1]/A_R

            fL = friction_factor(vL, D_array[i-1])
            fR = friction_factor(vR, D_array[i+1])

            RL = fL*dx/(2*g*D_array[i-1]*A_L**2)
            RR = fR*dx/(2*g*D_array[i+1]*A_R**2)

            Cp = H[i-1] + (a/(g*A_L))*Q[i-1] - RL*Q[i-1]*abs(Q[i-1])
            Cm = H[i+1] - (a/(g*A_R))*Q[i+1] + RR*Q[i+1]*abs(Q[i+1])

            A_i = np.pi*(D_array[i]**2)/4
            B_i = a/(g*A_i)

            Hn[i] = 0.5*(Cp+Cm)
            Qn[i] = (Cp - Hn[i])/B_i

        # ---------- UPSTREAM BOUNDARY ----------
        if enable_upstream_valve:
            tau_u = valve_tau_up(t)
            A0 = np.pi*(D_array[0]**2)/4
            B0 = a/(g*A0)
            Cp0 = H[1] - B0*Q[1]
            a_q = 1.0/(Cv*tau_u)**2
            b_q = B0
            c_q = H_up - Cp0
            disc = max(b_q**2 - 4*a_q*c_q, 0.0)
            Qn[0] = (-b_q + np.sqrt(disc))/(2*a_q)
            Hn[0] = Cp0 - B0*Qn[0]
        else:
            Hn[0] = H_up
            Qn[0] = Qn[1]

        # Downstream valve (correct MOC BC)
        A_N = np.pi*(D_array[N-1]**2)/4
        vN = Q[N-1]/A_N
        fN = friction_factor(vN, D_array[N-1])
        RN = fN*dx/(2*g*D_array[N-1]*A_N**2)

        Cp = H[N-1] + (a/(g*A_N))*Q[N-1] - RN*Q[N-1]*abs(Q[N-1])
        tau = valve_tau(t)

        B_N = a/(g*(np.pi*D_array[N]**2/4))
        a_q = 1.0/(Cv*tau)**2
        b_q = B_N
        c_q = H_ref - Cp

        disc = max(b_q**2 - 4*a_q*c_q, 0.0)
        Qn[N] = (-b_q + np.sqrt(disc))/(2*a_q)
        Hn[N] = Cp - B_N*Qn[N]

        Hn[0] = H_up
        Qn[0] = Qn[1]

        H, Q = Hn, Qn
        p_down[n] = (H[-1]-z[-1])*rho*g/1e5
        p_probe[n] = (H[probe_idx] - z[probe_idx])*rho*g/1e5

    return time, p_down, p_probe
