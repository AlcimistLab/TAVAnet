import socket
import torch
import numpy as np
import logging
import json
import os
import csv
import struct
from datetime import datetime
from scipy.special import gamma # --- MODIFICATION: Added for new reward calculation
# --- MODIFICATION: Import PPO agent and buffer ---
from custom.nndp_ppo_core import PPOAgentNNDP, TrajectoryBufferNNDP

LOG_RECEIVED_PATH = 'custom/logs/training/receive_data.log'
LOG_SENT_PATH = 'custom/logs/training/sent_data.log'
LOG_DEBUG_ACTION_PATH = 'custom/logs/training/action.log'
PERFORMANCE_LOG_PATH = 'custom/logs/training/performance_metrics.csv'

# --- MODIFICATION: PPO Hyperparameters with updated state dimension ---
STATE_DIM = 5  # State: [power, MCS, cbr, vehicle_density, snr]
ACTION_DIM = 2 # Action: adjustments to power and MCS
UPDATE_TIMESTEP = 100 # Perform a PPO update every N timesteps
PPO_ACTOR_PATH = "model/ppo_actor.pth"
PPO_CRITIC_PATH = "model/ppo_critic.pth"

# --- MODIFICATION: Initialize PPO agent and its trajectory buffer ---
agent = PPOAgentNNDP(STATE_DIM, ACTION_DIM)
trajectory_buffer = TrajectoryBufferNNDP(device=agent.device)

def log_data(log_path, data):
    """Logs data to a specified file with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")
        
def write_performance_metrics(timestamp, veh_id, cbr, snr, reward, file_path=PERFORMANCE_LOG_PATH):
    """Writes performance metrics to a CSV file."""
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'veh_id', 'CBR', 'SNR', 'Reward'])
        writer.writerow([timestamp, veh_id, cbr, snr, reward])

# --- MODIFICATION: New helper functions from the provided script ---

def get_data_rate_and_sensitivity(mcs_index):
    """Returns data rate and sensitivity based on MCS index (0-7)."""
    table = [
        (3, -85), (4.5, -84), (6, -82), (9, -80),
        (12, -77), (18, -73), (24, -69), (27, -68)
    ]
    mcs_index = max(min(int(mcs_index), 7), 0)
    return table[mcs_index]

def calculate_carrier_sense_range(p_dBm, mcs_index):
    """Calculate carrier sense range based on transmission power and MCS index."""
    beta, m, c, freq = 3, 1, 3e8, 5.9e9
    _, S_r_dBm = get_data_rate_and_sensitivity(mcs_index)
    p_linear = 10**(p_dBm / 10) * 1e-3
    S_linear = 10**(S_r_dBm / 10) * 1e-3
    lambda_val = c / freq
    A = (4 * np.pi / lambda_val)**2
    gamma_term1 = gamma(m + 1 / beta)
    gamma_term2 = gamma(m)
    r_CS = (gamma_term1 / gamma_term2) * (S_linear * A * m / p_linear)**(-1 / beta)
    return r_CS

def calculate_vehicle_density(num_neighbors, power_dBm, mcs_index):
    """Calculate vehicle density based on number of neighbors and carrier sense range."""
    r_cs = calculate_carrier_sense_range(power_dBm, mcs_index)
    area = np.pi * r_cs**2
    return num_neighbors / area if area > 0 else 0

def calculate_full_reward(CBR, p, mcs_index):
    """Calculate full reward based on CBR, power, and MCS index."""
    MBL, omega_c, omega_p, omega_d, omega_e, ds, c, freq = 0.65, 2, 0.25, 0.1, 0.8, 100, 3e8, 5.9e9
    data_rate, S_r = get_data_rate_and_sensitivity(mcs_index)
    lambda_val = c / freq
    A = (4 * np.pi / lambda_val)**2
    beta = 3
    path_loss_db = 10 * np.log10(A * (ds**beta))
    error = CBR - MBL
    baseCBRReward = -np.sign(error) * CBR
    bonus = 10 if abs(error) <= 0.025 else -0.1
    cbr_term = omega_c * (baseCBRReward + bonus)
    power_term = omega_p * abs((S_r + path_loss_db) - p)
    rate_term = omega_d * (data_rate)**omega_e
    reward = cbr_term - power_term - rate_term
    return reward, {'cbr': cbr_term, 'power': -power_term, 'rate': -rate_term}


def save_model_ppo(agent, actor_path, critic_path):
    """Saves the PPO actor and critic models."""
    agent.save_models(actor_path, critic_path)

def load_model_ppo(agent, actor_path, critic_path):
    """Loads the PPO actor and critic models."""
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        agent.load_models(actor_path, critic_path)
        logging.info(f"PPO models loaded from {actor_path} and {critic_path}")
        return True
    logging.info("No existing PPO models found, initializing new ones.")
    return False

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)

# --- Load PPO models if they exist ---
load_model_ppo(agent, PPO_ACTOR_PATH, PPO_CRITIC_PATH)

# --- Server setup ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

# Dictionaries to store trajectory data between steps
prev_states = {}
prev_actions = {}
prev_log_probs = {}
prev_values = {}

# --- Main training loop ---
while True:
    try:
        # Receive data from the simulator
        length_header = conn.recv(4)
        if len(length_header) < 4: break
        msg_length = struct.unpack('<I', length_header)[0]
        data = bytearray()
        while len(data) < msg_length:
            packet = conn.recv(msg_length - len(data))
            if not packet: break
            data.extend(packet)
        if len(data) < msg_length: break

        batch_data = json.loads(data.decode())
        log_data(LOG_RECEIVED_PATH, {json.dumps(batch_data, indent=4)})
        
        current_timestamp = -1
        last_known_state = None

        # --- STEP 1: Process batch, calculate rewards, and store transitions ---
        for veh_id, vehicle_data in batch_data.items():
            current_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
            current_mcs = max(min(vehicle_data['MCS'], 7), 0)
            cbr = max(min(vehicle_data['CBR'], 1.0), 0.0)
            snr = max(min(vehicle_data['SNR'], 50.0), 1.0)
            neighbors = vehicle_data['neighbors']
            timestamp = vehicle_data['timestamp']
            current_timestamp = timestamp
            
            vehicle_density = calculate_vehicle_density(neighbors, current_power, current_mcs)
            current_state = [current_power, current_mcs, cbr, vehicle_density, snr]
            last_known_state = current_state

            if timestamp > 0 and veh_id in prev_states:
                # The reward for the previous action is calculated based on the resulting state (current_state)
                reward, _ = calculate_full_reward(cbr, current_power, current_mcs)
                write_performance_metrics(timestamp, veh_id, cbr, snr, reward)
                
                done = False
                trajectory_buffer.add(
                    state=prev_states[veh_id], action=prev_actions[veh_id],
                    log_prob=prev_log_probs[veh_id], reward=reward,
                    done=done, value=prev_values[veh_id]
                )

        # --- STEP 2: Perform PPO update if the buffer is full ---
        if len(trajectory_buffer) >= UPDATE_TIMESTEP:
            logging.info(f"Buffer full ({len(trajectory_buffer)} transitions). Updating PPO agent...")
            with torch.no_grad():
                last_state_tensor = torch.FloatTensor(last_known_state).to(agent.device).unsqueeze(0)
                last_value_tensor = agent.critic(last_state_tensor).squeeze()

            agent.update(trajectory_buffer, last_value_tensor)
            trajectory_buffer.clear()
            
            save_model_ppo(agent, PPO_ACTOR_PATH, PPO_CRITIC_PATH)
            logging.info(f"PPO models saved after update at timestamp {current_timestamp}.")

        # --- STEP 3: Select actions for the CURRENT states and prepare response ---
        responses = {}
        for veh_id, vehicle_data in batch_data.items():
            # Reconstruct the current state for action selection
            original_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
            original_mcs = max(min(vehicle_data['MCS'], 7), 0)
            cbr = max(min(vehicle_data['CBR'], 1.0), 0.0)
            snr = max(min(vehicle_data['SNR'], 50.0), 1.0)
            neighbors = vehicle_data['neighbors']
            vehicle_density = calculate_vehicle_density(neighbors, original_power, original_mcs)
            current_state = [original_power, original_mcs, cbr, vehicle_density, snr]

            action, log_prob, value = agent.select_action_and_evaluate(current_state)
            
            prev_states[veh_id] = current_state
            prev_actions[veh_id] = action
            prev_log_probs[veh_id] = log_prob
            prev_values[veh_id] = value

            # --- MODIFICATION: Apply actions as adjustments ---
            power_adjustment = action[0] * 3
            mcs_adjustment = action[1] * 3
            
            new_power = original_power + power_adjustment
            new_mcs = round(original_mcs + mcs_adjustment)
            
            # Clamp to realistic constraints
            new_power = max(min(new_power, 30), 1)
            new_mcs = max(min(new_mcs, 7), 0)
            
            log_data(LOG_DEBUG_ACTION_PATH, json.dumps({"action": action.tolist(), "new_power": new_power, "new_mcs": new_mcs}))

            responses[veh_id] = {
                "transmissionPower": float(new_power),
                "beaconRate": float(vehicle_data['beaconRate']), # Keep original beacon rate
                "MCS": int(new_mcs)
            }

        # --- STEP 4: Send actions back to the simulator ---
        response_data = json.dumps(responses).encode('utf-8')
        response_length = len(response_data)
        length_header = struct.pack('<I', response_length)
        conn.sendall(length_header)
        conn.sendall(response_data)
        
        formatted_response = json.dumps(responses, indent=4)
        log_data(LOG_SENT_PATH, {formatted_response})

    except (struct.error, ConnectionResetError, BrokenPipeError, json.JSONDecodeError) as e:
        logging.error(f"A connection error occurred: {e}. The connection may have been closed.")
        break
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        break

# --- Final save and cleanup ---
save_model_ppo(agent, "model/final_ppo_actor.pth", "model/final_ppo_critic.pth")
logging.info("Final PPO models saved.")
conn.close()
server.close()
logging.info("Server closed.")
