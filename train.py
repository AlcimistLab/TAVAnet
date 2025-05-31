import socket
import torch
import numpy as np
import logging
import json
import os
import csv
from datetime import datetime
from sac_agent import SACAgent

LOG_RECEIVED_PATH = 'custom/logs/receive_data.log'
LOG_SENT_PATH = 'custom/logs/sent_data.log'
LOG_DEBUG_ACTION_PATH = 'custom/logs/action.log'
LOG_DEBUG_REWARD_PATH = 'custom/logs/reward_debug.log'  # debugging purposes
PERFORMANCE_LOG_PATH = 'custom/logs/performance_metrics.csv'

def log_data(log_path, data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")
        
def write_performance_metrics(cbr, snr, reward, batch_number, file_path=PERFORMANCE_LOG_PATH):
    # Check if the file already exists, if not, create the header
    file_exists = os.path.exists(file_path)
    
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Write header if the file is new
            writer.writerow(['Batch', 'Timestamp', 'CBR', 'SNR', 'Reward'])
        
        # Write performance metrics with batch number
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([batch_number, timestamp, cbr, snr, reward])

# Fungsi untuk menyesuaikan MCS berdasarkan SNR
def adjust_mcs_based_on_snr(snr):
    if 0 <= snr < 10:
        return 0  # MCS 0: BPSK 1/2
    elif 10 <= snr < 15:
        return 1  # MCS 1: BPSK 3/4
    elif 15 <= snr < 20:
        return 2  # MCS 2: QPSK 1/2
    elif 20 <= snr < 25:
        return 3  # MCS 3: QPSK 3/4
    elif 25 <= snr < 30:
        return 4  # MCS 4: 16-QAM 1/2
    elif 30 <= snr < 35:
        return 5  # MCS 5: 16-QAM 3/4
    elif 35 <= snr < 40:
        return 6  # MCS 6: 64-QAM 2/3
    elif 40 <= snr <= 50:
        return 7  # MCS 7: 64-QAM 3/4
    else:
        raise ValueError("SNR out of range. SNR should be between 0 and 50 dB.")
    
    
def get_data_rate_and_sr(mcs):
    """
    Mengembalikan data rate dan sensitivitas receiver (Sr) berdasarkan MCS index (0-7).

    Parameters:
    - mcs (int): Modulation and Coding Scheme index (0 to 7)

    Returns:
    - data_rate (float): Data rate dalam Mbps
    - sr (float): Receiver sensitivity (dBm)

    Raises:
    - ValueError: Jika MCS tidak dalam rentang 0-7
    """
    table = [
        (3, -85),    # MCS 0: BPSK 1/2
        (4.5, -84),  # MCS 1: BPSK 3/4
        (6, -82),    # MCS 2: QPSK 1/2
        (9, -80),    # MCS 3: QPSK 3/4
        (12, -77),   # MCS 4: 16-QAM 1/2
        (18, -73),   # MCS 5: 16-QAM 3/4
        (24, -69),   # MCS 6: 64-QAM 2/3
        (27, -68),   # MCS 7: 64-QAM 3/4
    ]

    if 0 <= mcs < len(table):
        return table[mcs]
    else:
        raise ValueError("MCS index harus antara 0 dan 7.")


# Formula reward yang baru
def calculation_reward_new(cbr, snr, prev_tx_power, tx_power):
    # CBR Reward: menggunakan fungsi berbasis log
    deviation = abs(cbr - 0.65)
    if deviation < 0.1:
        reward_cbr = 10 * (1 - deviation / 0.65)
    else:
        reward_cbr = -5 * deviation

    # SNR Reward: fungsi berbasis kuadrat yang lebih smooth
    snr_deviation = abs(snr - 25.0)
    if snr_deviation <= 5.0:
        reward_snr = 10 * (1 - (snr_deviation / 5.0) ** 2)
    else:
        reward_snr = -1 * snr_deviation 

    # Power Stability Reward: menambahkan penalti bertahap jika tx_power jauh dari nilai sebelumnya
    reward_power = -0.1 * (abs(tx_power - prev_tx_power) ** 1.5)
    
    reward = reward_cbr + reward_snr + reward_power
    return reward

def calculate_reward(cbr, sr, p_tx, data_rate, snr):
    """
    Menghitung reward berdasarkan CBR, sensitivitas, power transmisi, data rate, dan SNR (input langsung).

    Parameters:
    - cbr (float): Channel Busy Ratio saat ini.
    - sr (float): Sensitivitas penerima (dBm) untuk data rate tertentu.
    - p_tx (float): Power transmisi saat ini (dBm).
    - data_rate (float): Data rate saat ini (Mbps).
    - snr (float): Signal-to-Noise Ratio saat ini (dB).

    Returns:
    - reward (float): Nilai reward akhir.
    """

    # Konstanta tetap
    ds = 100              # Safety distance (m)
    mbl = 0.6             # Target CBR
    freq_hz = 5.9e9       # Frekuensi (Hz)
    beta = 2.5            # Path loss exponent
    snr_threshold = 5.0   # Minimum SNR required (dB)
    c = 3e8               # Kecepatan cahaya

    # Hitung Path Loss (dari p_tx ke ds)
    lambda_ = c / freq_hz
    A = (4 * np.pi / lambda_) ** 2
    path_loss_linear = A * (ds ** beta)
    path_loss_db = 10 * np.log10(path_loss_linear)

    # Bobot reward
    omega_c = 2.0
    omega_p = 0.25
    omega_d = 0.1
    omega_e = 0.8
    omega_snr = 0.3

    # Komponen reward 1: CBR
    g = -np.sign(cbr - mbl) * cbr
    reward_c = omega_c * g

    # Komponen reward 2: kecocokan p_tx terhadap sr dan path loss
    reward_p = -omega_p * abs((sr + path_loss_db) - p_tx)

    # Komponen reward 3: penalti untuk data rate tinggi
    reward_d = -omega_d * (data_rate ** omega_e)

    # Komponen reward 4: kontrol SNR
    reward_snr = omega_snr * (snr - snr_threshold) if snr >= snr_threshold else -omega_snr * abs(snr - snr_threshold)

    # Total reward
    reward = reward_c + reward_p + reward_d + reward_snr
    debug_reward = [reward_c, reward_p, reward_d, reward_snr, reward]
    log_data(LOG_DEBUG_REWARD_PATH, json.dumps(debug_reward, indent=4))
    return reward

logging.basicConfig(level=logging.INFO)

# Inisialisasi agen SAC
state_dim = 5  # state memiliki 5 komponen: power, beacon, cbr, neighbors, snr
action_dim = 2  # action: perubahan transmisi daya dan beacon rate
agent = SACAgent(state_dim, action_dim)

# Path untuk menyimpan model
model_save_path = "model/custom.pth"

# Memeriksa jika model sudah ada, dan memuatnya jika ada
if os.path.exists(model_save_path):
    agent.policy_net.load_state_dict(torch.load(model_save_path))
    logging.info("Loaded existing model.")
else:
    logging.info("Initialized new model.")

# Membuka soket untuk menerima data
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

# Loop untuk menerima dan memproses batch data
batch_counter = 0
while True:
    data = conn.recv(65536)
    if not data:
        logging.info("No data received, closing connection.")
        break
    try:
        batch_counter += 1
        # Parse incoming batch data from client
        batch_data = json.loads(data.decode())
        logging.info(f"[BATCH] Received {len(batch_data)} vehicles' data")
        log_data(LOG_RECEIVED_PATH, f"Batch {batch_counter}: {json.dumps(batch_data, indent=4)}")
        logging.info(f"Received data for batch {batch_counter}:\n{json.dumps(batch_data, indent=4)}")

        # Prepare response dictionary
        responses = {}

        for veh_id, vehicle_data in batch_data.items():
            # Extract state information from each vehicle data
            current_power = vehicle_data['transmissionPower']
            current_beacon = vehicle_data['beaconRate']
            cbr = vehicle_data['CBR']
            neighbors = vehicle_data['neighbors']
            snr = vehicle_data['SNR']
            mcs = vehicle_data['MCS']
            
            # Compose state for the model
            state = [current_power, current_beacon, cbr, neighbors, snr]
            
            # Select action using the SAC agent
            action = agent.select_action(state)
            
            # Scaling the action to the valid range [1, 20] for beaconRate, and [1, 30] for transmissionPower
            new_power = (action[0] + 1) / 2 * (30 - 1) + 1  # Transmission power range [1, 30]
            new_beacon = (action[1] + 1) / 2 * (20 - 1) + 1  # Beacon rate range [1, 20]
            action_float = tuple(map(float, [new_power, new_beacon]))
            log_data(LOG_DEBUG_ACTION_PATH, f"Batch {batch_counter}: {json.dumps(action_float, indent=4)}")
            
            # Calculate reward using the custom reward function
            prev_tx_power = current_power
            mcs = adjust_mcs_based_on_snr(snr)
            data_rate, sr = get_data_rate_and_sr(mcs)
            
            reward = calculate_reward(cbr, sr, prev_tx_power, data_rate, snr)
            # reward = calculation_reward_new(cbr, snr, prev_tx_power, new_power)
            write_performance_metrics(cbr, snr, reward, batch_number=batch_counter)
            
            # Store the transition in the agent's replay buffer
            agent.store_transition(state, action, reward, [new_power, new_beacon, cbr, neighbors, snr], False)
            agent.train()

            # Prepare response for the current vehicle
            responses[veh_id] = {
                "transmissionPower": float(new_power),
                "beaconRate": float(new_beacon),
                "MCS": mcs
            }

        # Send the response batch to the client
        response_data = json.dumps(responses).encode('utf-8')
        conn.sendall(response_data)
        formatted_response = json.dumps(responses, indent=4)
        log_data(LOG_SENT_PATH, f"Batch {batch_counter}: {formatted_response}")
        logging.info(f"Sending RL response to client for batch {batch_counter}: {formatted_response}")

    except Exception as e:
        logging.error(f"Error: {e}")
        continue

# Simpan model setelah pelatihan selesai
torch.save(agent.policy_net.state_dict(), model_save_path)
logging.info(f"Model saved to {model_save_path}")

# Menutup koneksi dan server
conn.close()
server.close()
logging.info("Server closed.")