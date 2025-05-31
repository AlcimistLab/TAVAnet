import math
import numpy as np

# 1. Carrier‐Sense Range (Eq. 1)
def carrier_sense_range(p, m, A, S, beta):
    """
    p    : transmit power (linear scale)
    m    : Nakagami fading shape parameter
    A    : path‐loss constant (e.g., (4π/λ)^2)
    S    : receiver sensitivity (linear scale)
    beta : path‐loss exponent
    """
    return ((math.gamma(m + 1) / math.gamma(m)) * A * p / S) ** (1.0 / beta)

# 2. Channel Capacity (Eq. 2)
def channel_capacity(C_d, b_s, t, M, t_ps):
    """
    C_d  : coded bits per OFDM symbol
    b_s  : service+tail bits per packet
    t    : number of packets (e.g., beacon interval count)
    M    : payload bits per packet
    t_ps : PHY preamble+signal duration (s)
    """
    symbols = math.ceil((b_s * t + M) / C_d)
    return 1.0 / (C_d / symbols + t_ps)

# 3. Channel Busy Ratio (CBR, Eq. 3)
def compute_CBR(p, d, rho, b, m, A, S, beta, C_d, b_s, t, M, t_ps):
    """
    b    : beacon rate (Hz)
    rho  : vehicle density (vehicles/m)
    d    : data rate (unused directly in CBR formula but may affect C_d, S, etc.)
    """
    rCS = carrier_sense_range(p, m, A, S, beta)
    C = channel_capacity(C_d, b_s, t, M, t_ps)
    return 2 * rCS * rho * b / C

# 4. Path Loss at Safety Distance
def path_loss(distance, A, beta):
    """
    distance : safety distance (m)
    A        : path‐loss constant
    beta     : path‐loss exponent
    """
    return A * (distance ** beta)

# 5. Reward Components
def g_CBR(cbr, MBL=0.6):
    """Eq. 4: drive CBR toward target load (MBL)."""
    return -np.sign(cbr - MBL) * cbr

def reward_cbr_shaping(cbr, MBL=0.6, eps=0.025, bonus=10.0, penalty=-0.1):
    """
    Bonus if CBR within ±eps of MBL, else penalty.
    """
    return bonus if abs(cbr - MBL) <= eps else penalty

def reliability_term(p, l, S_r):
    """
    Eq. 5: reliability at safety distance.
    p   : transmit power
    l   : path‐loss at safety distance
    S_r : receiver sensitivity for current data rate
    """
    return -abs((S_r + l) - p)

def data_rate_penalty(d, omega_d=0.1, omega_e=0.8):
    """
    Negative penalty for high data rates.
    """
    return -omega_d * (d ** omega_e)

# 6. Total Reward (Eq. 6)
def total_reward(cbr, p, d, rho, 
                 m, A, S, beta, 
                 C_d, b_s, t, M, t_ps, b,
                 S_r,
                 omega_c=2.0, omega_p=0.25, omega_d=0.1, omega_e=0.8):
    """
    Combines all reward terms into a scalar reward.
    """
    # Compute CBR
    cbr = compute_CBR(p, d, rho, b, m, A, S, beta, C_d, b_s, t, M, t_ps)
    # Reward components
    r1 = omega_c * g_CBR(cbr)
    r2 = omega_p * reliability_term(p, path_loss(1.0, A, beta), S_r)  # example: safety distance=1 m
    r3 = data_rate_penalty(d, omega_d, omega_e)
    return r1 + r2 + r3

# Example usage
if __name__ == "__main__":
    # Example parameters (placeholders)
    p = 23.0        # dBm
    d = 6.0         # Mbps
    rho = 0.02      # vehicles/m
    b = 10          # Hz
    m = 1.5
    A = (4 * math.pi / 5.9e9) ** 2
    S = 1e-9        # linear scale
    beta = 2.0
    C_d = 144       # bits/symbol
    b_s = 22        # bits
    t = 1           # packets per interval
    M = 200         # bits
    t_ps = 40e-6    # seconds
    S_r = -85       # dBm sensitivity
    
    # Compute formulas
    rCS = carrier_sense_range(p, m, A, S, beta)
    C = channel_capacity(C_d, b_s, t, M, t_ps)
    cbr = compute_CBR(p, d, rho, b, m, A, S, beta, C_d, b_s, t, M, t_ps)
    reward = total_reward(cbr, p, d, rho, m, A, S, beta, C_d, b_s, t, M, t_ps, b, S_r)
    
    print(f"Carrier‐Sense Range: {rCS:.2f} m")
    print(f"Channel Capacity:   {C:.2f} msgs/sec")
    print(f"CBR:               {cbr:.3f}")
    print(f"Reward:            {reward:.3f}")
