import socket
import torch
import numpy as np
import logging
import json
import os
import csv
import struct
from datetime import datetime
from nndp_ppo_core import PPOAgentNNDP, TrajectoryBufferNNDP # Import PPO components
from scipy.special import gamma

# --- Path Constants ---
LOG_DIR = 'custom/logs'
MODEL_DIR = 'model'
LOG_RECEIVED_PATH = os.path.join(LOG_DIR, 'receive_data.log')
LOG_SENT_PATH = os.path.join(LOG_DIR, 'sent_data.log')
LOG_DEBUG_ACTION_PATH = os.path.join(LOG_DIR, 'action.log')
PERFORMANCE_LOG_PATH = os.path.join(LOG_DIR, 'performance_metrics.csv')

# --- Create Directories ---
# This ensures that the directories for logging and saving models exist before they are used.
# os.makedirs() will create the directories if they are missing and do nothing if they already exist.
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
# --- End Directory Creation ---

# --- PPO Specific Hyperparameters ---
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
PPO_EPOCHS = 10
PPO_CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
TRAJECTORY_BUFFER_CAPACITY = 1024
ACTION_SCALE_POWER = 3.0
ACTION_SCALE_MCS = 3.0

def log_data(log_path, data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")

def write_performance_metrics(timestamp, cbr, snr, reward, file_path=PERFORMANCE_LOG_PATH):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'CBR', 'SNR', 'Reward'])
        writer.writerow([timestamp, cbr, snr, reward])

def get_data_rate_and_sensitivity(mcs_index):
    table = [
        (3, -85), (4.5, -84), (6, -82), (9, -80),
        (12, -77), (18, -73), (24, -69), (27, -68),
    ]
    mcs_index = max(min(int(round(mcs_index)), 7), 0)
    data_rate, S_r = table[mcs_index]
    return data_rate, S_r

def calculate_carrier_sense_range(p_dBm, mcs_index):
    beta = 3
    m = 1
    c = 3e8
    freq = 5.9e9
    _, S_r_dBm = get_data_rate_and_sensitivity(mcs_index)
    p_linear = 10**(p_dBm/10) * 1e-3
    S_linear = 10**(S_r_dBm/10) * 1e-3
    lambda_val = c / freq
    A = (4 * np.pi / lambda_val)**2
    gamma_term1 = gamma(m + 1/beta)
    gamma_term2 = gamma(m)
    r_CS = (gamma_term1 / gamma_term2) * (S_linear * A * m / p_linear)**(-1/beta)
    return r_CS

def calculate_full_reward(CBR, p, mcs_index):
    MBL = 0.65
    omega_c = 2
    omega_p = 0.25
    omega_d = 0.1
    omega_e = 0.8
    ds = 100
    c = 3e8
    freq = 5.9e9
    data_rate, S_r = get_data_rate_and_sensitivity(mcs_index)
    lambda_val = c / freq
    A = (4 * np.pi / lambda_val)**2
    beta = 3
    path_loss_linear = A * (ds**beta)
    path_loss_db = 10 * np.log10(path_loss_linear)
    error = CBR - MBL
    baseCBRReward = -np.sign(error) * CBR
    bonus = 10 if abs(error) <= 0.025 else -0.1
    components = {}
    components['CBR_term'] = omega_c * (baseCBRReward + bonus)
    components['power_term'] = omega_p * abs((S_r + path_loss_db) - p)
    components['rate_term'] = omega_d * (data_rate)**omega_e
    reward = components['CBR_term'] - components['power_term'] - components['rate_term']
    return reward, components

logging.basicConfig(level=logging.INFO)

timestamps_to_save = [100, 200, 300, 400, 500]

# --- Agent Initialization ---
state_dim = 5
action_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ppo_agent = PPOAgentNNDP(
    state_dim, action_dim,
    lr_actor=LEARNING_RATE_ACTOR, lr_critic=LEARNING_RATE_CRITIC,
    gamma=GAMMA, ppo_epochs=PPO_EPOCHS, ppo_clip_epsilon=PPO_CLIP_EPSILON,
    gae_lambda=GAE_LAMBDA, entropy_coefficient=ENTROPY_COEFFICIENT,
    max_grad_norm=MAX_GRAD_NORM, device=device
)
trajectory_buffer = TrajectoryBufferNNDP(device=device)
logging.info(f"Initialized PPO model on {device}.")
# --- End Agent Initialization ---

steps_in_buffer = 0
last_recorded_next_state_for_gae = None

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

current_simulation_timestamp = 0

try:
    while True:
        length_header = conn.recv(4)
        if len(length_header) < 4:
            logging.warning("Failed to receive message length, closing connection.")
            break
        msg_length = struct.unpack('<I', length_header)[0]
        
        data = conn.recv(msg_length)
        if len(data) < msg_length:
            logging.warning("Incomplete data received, closing connection.")
            break

        batch_data = json.loads(data.decode())
        
        responses = {}
        batch_processed_count = 0

        for veh_id, vehicle_data in batch_data.items():
            batch_processed_count +=1
            original_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
            current_beacon = max(min(vehicle_data['beaconRate'], 20.0), 1.0)
            cbr = max(min(vehicle_data['CBR'], 1.0), 0.0)
            neighbors = vehicle_data['neighbors']
            snr = max(min(vehicle_data['SNR'], 50.0), 1.0)
            original_mcs = max(min(vehicle_data['MCS'], 7), 0)
            
            state = [original_power, current_beacon, cbr, float(neighbors), snr]

            action_deltas_normalized, log_prob, value = ppo_agent.select_action_and_evaluate(state)

            power_adjustment = action_deltas_normalized[0] * ACTION_SCALE_POWER
            mcs_adjustment = action_deltas_normalized[1] * ACTION_SCALE_MCS

            new_power = original_power + power_adjustment
            new_mcs_float = float(original_mcs) + mcs_adjustment
            
            new_power = max(min(new_power, 30.0), 1.0)
            new_mcs_int = int(round(max(min(new_mcs_float, 7.0), 0.0)))

            reward, _ = calculate_full_reward(cbr, new_power, new_mcs_int)
            
            sim_time_for_perf = vehicle_data.get('timestamp', current_simulation_timestamp)
            if isinstance(sim_time_for_perf, list):
                sim_time_for_perf = sim_time_for_perf[0] 
            write_performance_metrics(sim_time_for_perf, cbr, snr, reward)
            
            if 'timestamp' in vehicle_data:
                 current_simulation_timestamp = vehicle_data['timestamp']

            next_state = [new_power, current_beacon, cbr, float(neighbors), snr]
            
            trajectory_buffer.add(state, action_deltas_normalized, log_prob, reward, False, value)
            steps_in_buffer += 1
            last_recorded_next_state_for_gae = next_state

            responses[veh_id] = {
                "transmissionPower": float(new_power),
                "beaconRate": float(current_beacon),
                "MCS": int(new_mcs_int)
            }

            if steps_in_buffer >= TRAJECTORY_BUFFER_CAPACITY:
                logging.info(f"Buffer full ({steps_in_buffer} steps). Performing PPO update.")
                
                bootstrap_value_tensor = torch.tensor([0.0], dtype=torch.float32).to(device)
                if last_recorded_next_state_for_gae is not None:
                     with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(last_recorded_next_state_for_gae).to(device).unsqueeze(0)
                        bootstrap_value_tensor = ppo_agent.critic(next_state_tensor).squeeze()

                ppo_agent.update(trajectory_buffer, bootstrap_value_tensor)
                trajectory_buffer.clear()
                steps_in_buffer = 0
                last_recorded_next_state_for_gae = None
                logging.info("PPO Update finished. Buffer cleared.")

        if current_simulation_timestamp > 0: 
            for ts_save_point in list(timestamps_to_save):
                if int(current_simulation_timestamp) >= ts_save_point:
                    actor_path_check = os.path.join(MODEL_DIR, f'ppo_actor_ts{int(ts_save_point)}.pth')
                    if not os.path.exists(actor_path_check):
                        actor_path = os.path.join(MODEL_DIR, f'ppo_actor_ts{int(ts_save_point)}.pth')
                        critic_path = os.path.join(MODEL_DIR, f'ppo_critic_ts{int(ts_save_point)}.pth')
                        ppo_agent.save_models(actor_path, critic_path)
                        logging.info(f"Models saved for timestamp {ts_save_point}")
                        
        response_data = json.dumps(responses).encode('utf-8')
        response_length = len(response_data)
        length_header = struct.pack('<I', response_length)
        conn.sendall(length_header)
        conn.sendall(response_data)

        if batch_processed_count > 0:
             logging.info(f"Processed batch of {batch_processed_count} vehicles. Current buffer size: {steps_in_buffer}/{TRAJECTORY_BUFFER_CAPACITY}. SimTime: {current_simulation_timestamp}")

except ConnectionResetError:
    logging.warning("Client closed the connection unexpectedly.")
except socket.error as e:
    logging.error(f"Socket error: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}", exc_info=True)
finally:
    final_actor_path = os.path.join(MODEL_DIR, 'final_ppo_actor.pth')
    final_critic_path = os.path.join(MODEL_DIR, 'final_ppo_critic.pth')
    ppo_agent.save_models(final_actor_path, final_critic_path)
    logging.info(f"Final PPO models saved.")

    if 'conn' in locals() and conn:
        conn.close()
    if 'server' in locals() and server:
        server.close()
    logging.info("Server closed.")