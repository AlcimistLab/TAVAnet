import socket
import torch
import numpy as np
import logging
import json
import os
import csv
from datetime import datetime
from nndp_ppo_core import PPOAgentNNDP, TrajectoryBufferNNDP # Import PPO components
import struct # <-- Added this import

# --- NNDP Paper Constants & Helper Data ---
# Based on Table 2 and Section III.B.2 of the NNDP paper
NNDP_DATA_RATES_INFO = {
    # Data Rate (Mbps): {"Sr_dbm": Receiver Sensitivity, "Cd_bits_per_symbol": Coded bits per symbol}
    3.0: {"Sr_dbm": -85, "Cd_bits_per_symbol": 48, "modulation": "BPSK", "cr": "1/2"},
    4.5: {"Sr_dbm": -84, "Cd_bits_per_symbol": 48, "modulation": "BPSK", "cr": "3/4"},
    6.0: {"Sr_dbm": -82, "Cd_bits_per_symbol": 96, "modulation": "QPSK", "cr": "1/2"}, # Default/Reference often
    9.0: {"Sr_dbm": -80, "Cd_bits_per_symbol": 96, "modulation": "QPSK", "cr": "3/4"},
    12.0: {"Sr_dbm": -77, "Cd_bits_per_symbol": 192, "modulation": "16-QAM", "cr": "1/2"},
    18.0: {"Sr_dbm": -73, "Cd_bits_per_symbol": 192, "modulation": "16-QAM", "cr": "3/4"},
    24.0: {"Sr_dbm": -69, "Cd_bits_per_symbol": 288, "modulation": "64-QAM", "cr": "2/3"},
    27.0: {"Sr_dbm": -68, "Cd_bits_per_symbol": 288, "modulation": "64-QAM", "cr": "3/4"},
}
VALID_NNDP_DATA_RATES = sorted(NNDP_DATA_RATES_INFO.keys())
MIN_NNDP_DATA_RATE = VALID_NNDP_DATA_RATES[0]
MAX_NNDP_DATA_RATE = VALID_NNDP_DATA_RATES[-1]

MIN_TX_POWER_DBM = 1.0
MAX_TX_POWER_DBM = 30.0

# Reward function parameters from NNDP Paper (Section III.B.2, Table 4)
OMEGA_C = 2.0
OMEGA_P = 0.25
OMEGA_D = 0.1
OMEGA_E = 0.8
MBL_TARGET_CBR = 0.6
SAFETY_DISTANCE_DS_M = 100.0
PATH_LOSS_EXPONENT_BETA = 2.5 # From Table 4 (training condition)
CHANNEL_FREQUENCY_HZ = 5.9e9
SPEED_OF_LIGHT_MPS = 3.0e8

# Calculate A for path loss: A = (4*pi*f/c)^2
LAMBDA_WAVELENGTH_M = SPEED_OF_LIGHT_MPS / CHANNEL_FREQUENCY_HZ
PATH_LOSS_A_CONST = (4 * np.pi / LAMBDA_WAVELENGTH_M)**2

# Action scaling parameters (hyperparameters, might need tuning)
# Actor outputs means in [-1, 1]. These are scaled to become deltas.
MAX_POWER_DELTA_SCALE = 5.0  # e.g., maps [-1,1] to [-5dBm, +5dBm] delta
MAX_DATA_RATE_DELTA_SCALE = 3.0 # e.g., maps [-1,1] to [-3Mbps, +3Mbps] delta

# --- Logging Setup (similar to template) ---
LOG_DIR = 'custom/logs/nndp_ppo'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_RECEIVED_PATH = os.path.join(LOG_DIR, 'receive_data.log')
LOG_SENT_PATH = os.path.join(LOG_DIR, 'sent_data.log')
LOG_ACTION_REWARD_PATH = os.path.join(LOG_DIR, 'action_reward.log')
PERFORMANCE_LOG_PATH = os.path.join(LOG_DIR, 'performance_metrics.csv')
MODEL_SAVE_DIR = "model/nndp_ppo"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
ACTOR_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "nndp_ppo_actor.pth")
CRITIC_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "nndp_ppo_critic.pth")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_data_to_file(log_path, data_str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data_str}\n")

def write_performance_metrics(cbr, vehicle_density, reward, avg_power, avg_data_rate, batch_number, file_path=PERFORMANCE_LOG_PATH):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Batch', 'Timestamp', 'AvgCBR', 'AvgVehicleDensity', 'AvgReward', 'AvgTxPower', 'AvgDataRate'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([batch_number, timestamp, cbr, vehicle_density, reward, avg_power, avg_data_rate])

# --- Helper Functions for NNDP ---
def get_nndp_discrete_data_rate_info(continuous_data_rate_mbps):
    """
    Takes a continuous data rate, clips it, finds the closest valid discrete rate 
    from NNDP Table 2, and returns its properties.
    """
    # Clip the continuous data rate to the valid range of NNDP rates
    clipped_continuous_rate = np.clip(continuous_data_rate_mbps, MIN_NNDP_DATA_RATE, MAX_NNDP_DATA_RATE)
    
    # Find the discrete rate in VALID_NNDP_DATA_RATES that is closest to the clipped continuous rate
    final_discrete_rate = min(VALID_NNDP_DATA_RATES, key=lambda x: abs(x - clipped_continuous_rate))
    
    info = NNDP_DATA_RATES_INFO[final_discrete_rate]
    return final_discrete_rate, info["Sr_dbm"], info["Cd_bits_per_symbol"]

def calculate_nndp_reward(cbr_measured, new_tx_power_dbm, new_data_rate_mbps, sr_for_new_data_rate_dbm):
    """
    Calculates reward based on Equation (6) from the NNDP paper.
    - cbr_measured: CBR value (ideally after the action, or current if next is unavailable)
    - new_tx_power_dbm: Transmission power chosen by the agent (p in the paper's eq.)
    - new_data_rate_mbps: Data rate chosen by the agent (d in the paper's eq.)
    - sr_for_new_data_rate_dbm: Receiver sensitivity for the chosen data rate (Sr in paper's eq.)
    """
    # Term 1: CBR control (g(CBR) part)
    cbr_deviation = cbr_measured - MBL_TARGET_CBR
    
    if cbr_measured <= MBL_TARGET_CBR:
        g_cbr = cbr_measured 
    else:
        g_cbr = -cbr_measured 

    reward1_base = OMEGA_C * g_cbr
    
    if abs(cbr_deviation) <= 0.025:
        reward1_bonus = 10.0
    else:
        reward1_bonus = -0.1
    reward1 = reward1_base + reward1_bonus

    # Term 2: Power adequacy (Path loss and sensitivity)
    path_loss_linear = PATH_LOSS_A_CONST * (SAFETY_DISTANCE_DS_M ** PATH_LOSS_EXPONENT_BETA)
    path_loss_db = 10 * np.log10(path_loss_linear) if path_loss_linear > 0 else float('inf')
    
    target_power_for_reliability = sr_for_new_data_rate_dbm + path_loss_db
    reward2 = -OMEGA_P * abs(target_power_for_reliability - new_tx_power_dbm)

    # Term 3: Data rate penalty
    reward3 = -OMEGA_D * (new_data_rate_mbps ** OMEGA_E)
    
    total_reward = reward1 + reward2 + reward3
    
    return total_reward

# --- Main Training Script ---
def main():
    # NNDP PPO Agent Hyperparameters
    state_dim = 3
    action_dim = 2
    lr_actor = 3e-5
    lr_critic = 1e-4
    gamma = 0.99
    ppo_epochs = 10
    ppo_clip_epsilon = 0.2
    gae_lambda = 0.95
    entropy_coefficient = 0.01
    
    update_interval_timesteps = 2048
    save_model_interval_batches = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    agent = PPOAgentNNDP(state_dim, action_dim, lr_actor, lr_critic, gamma, 
                         ppo_epochs, ppo_clip_epsilon, gae_lambda, 
                         entropy_coefficient, device=device)
    
    trajectory_buffer = TrajectoryBufferNNDP(device=device)

    if os.path.exists(ACTOR_MODEL_SAVE_PATH) and os.path.exists(CRITIC_MODEL_SAVE_PATH):
        try:
            agent.load_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)
            logging.info(f"Loaded models from {ACTOR_MODEL_SAVE_PATH} and {CRITIC_MODEL_SAVE_PATH}")
        except Exception as e:
            logging.error(f"Could not load models: {e}. Initializing new models.")
    else:
        logging.info("Initialized new PPO models.")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 5000))
    server.listen(1)
    logging.info("NNDP PPO Training Server listening on port 5000...")
    
    conn = None # Initialize conn to None
    try:
        conn, addr = server.accept()
        logging.info(f"Connected by {addr}")
        
        batch_counter = 0
        total_timesteps_collected = 0

        while True:
            # === MODIFIED DATA RECEPTION LOGIC ===
            # 1. Read the 4-byte length prefix
            length_prefix_bytes = conn.recv(4)
            if not length_prefix_bytes or len(length_prefix_bytes) < 4:
                logging.error("Failed to receive complete length prefix from client. Closing connection.")
                break

            try:
                # Assuming little-endian ('<') unsigned int ('I') for the 4-byte length
                message_length = struct.unpack('<I', length_prefix_bytes)[0]
                logging.info(f"Expecting message of {message_length} bytes.")
            except struct.error as e:
                logging.error(f"Could not unpack message length: {e}. Raw prefix: {length_prefix_bytes.hex()}. Closing connection.")
                break

            # 2. Read the full message body based on the received length
            data_chunks = []
            bytes_received = 0
            while bytes_received < message_length:
                # Read in chunks, up to 65536 or remaining bytes
                chunk_size_to_read = min(message_length - bytes_received, 65536)
                chunk = conn.recv(chunk_size_to_read)
                if not chunk:
                    logging.error(f"Connection broken while receiving message body. Expected {message_length}, got {bytes_received}.")
                    message_length = -1 # Signal error to break outer loop
                    break
                data_chunks.append(chunk)
                bytes_received += len(chunk)
            
            if message_length == -1: # Error signaled from inner reading loop
                break 
            
            data = b''.join(data_chunks)
            # === END OF MODIFIED DATA RECEPTION LOGIC ===
            
            # Initialize vehicle_input_data here for broader scope in except blocks if needed
            vehicle_input_data = {} 
            veh_id_in_error = None

            try:
                # 3. Process the received message (decode, JSON parse, etc.)
                try:
                    batch_data_str = data.decode('utf-8')
                except UnicodeDecodeError as ude:
                    logging.error(f"UnicodeDecodeError: {ude}. Raw data (first 50 bytes as hex): {data[:50].hex()}")
                    # Optionally, send an error response to the client
                    conn.sendall(json.dumps({"error": "Server received non-UTF-8 data"}).encode('utf-8'))
                    continue # Skip this message and try to receive the next

                batch_data = json.loads(batch_data_str)
                log_data_to_file(LOG_RECEIVED_PATH, f"Batch {batch_counter}: {batch_data_str}")

                responses = {}
                
                current_batch_rewards = []
                current_batch_cbrs = []
                current_batch_rhos = []
                current_batch_powers = []
                current_batch_data_rates = []

                for veh_id, vehicle_input_data_loop in batch_data.items():
                    veh_id_in_error = veh_id # Store for potential use in KeyError logging
                    vehicle_input_data = vehicle_input_data_loop # For logging in case of KeyError

                    current_power = float(vehicle_input_data['current_tx_power_dbm'])
                    current_data_rate = float(vehicle_input_data['current_data_rate_mbps'])
                    vehicle_density = float(vehicle_input_data['vehicle_density_rho'])
                    cbr = float(vehicle_input_data['cbr_measured'])

                    nndp_state = [current_power, current_data_rate, vehicle_density]
                    norm_action_deltas, log_prob, value_s = agent.select_action_and_evaluate(nndp_state)
                    
                    delta_power = norm_action_deltas[0] * MAX_POWER_DELTA_SCALE
                    delta_data_rate = norm_action_deltas[1] * MAX_DATA_RATE_DELTA_SCALE
                    
                    new_power_continuous = current_power + delta_power
                    new_power_clipped = np.clip(new_power_continuous, MIN_TX_POWER_DBM, MAX_TX_POWER_DBM)
                    
                    new_data_rate_continuous = current_data_rate + delta_data_rate
                    new_data_rate_discrete, sr_for_new_rate, _ = get_nndp_discrete_data_rate_info(new_data_rate_continuous)

                    reward = calculate_nndp_reward(cbr, new_power_clipped, new_data_rate_discrete, sr_for_new_rate)
                    
                    done_flag = False 
                    trajectory_buffer.add(nndp_state, norm_action_deltas, log_prob, reward, done_flag, value_s)
                    total_timesteps_collected += 1

                    responses[veh_id] = {
                        "new_tx_power_dbm": float(new_power_clipped),
                        "new_data_rate_mbps": float(new_data_rate_discrete)
                    }
                    
                    current_batch_rewards.append(reward)
                    current_batch_cbrs.append(cbr)
                    current_batch_rhos.append(vehicle_density)
                    current_batch_powers.append(new_power_clipped)
                    current_batch_data_rates.append(new_data_rate_discrete)

                    log_data_to_file(LOG_ACTION_REWARD_PATH, 
                                     f"Batch {batch_counter}, Veh {veh_id}: State={nndp_state}, NormActionDeltas={norm_action_deltas}, "
                                     f"ScaledDeltas=[{delta_power:.2f}, {delta_data_rate:.2f}], NewP={new_power_clipped:.2f}, NewDR={new_data_rate_discrete:.2f}, "
                                     f"CBR={cbr:.3f}, Rho={vehicle_density:.3f}, Reward={reward:.3f}, ValueS={value_s:.3f}")

                response_data_str = json.dumps(responses)
                conn.sendall(response_data_str.encode('utf-8'))
                log_data_to_file(LOG_SENT_PATH, f"Batch {batch_counter}: {response_data_str}")
                
                if current_batch_rewards:
                    avg_reward = np.mean(current_batch_rewards)
                    avg_cbr = np.mean(current_batch_cbrs)
                    avg_rho = np.mean(current_batch_rhos)
                    avg_power = np.mean(current_batch_powers)
                    avg_dr = np.mean(current_batch_data_rates)
                    logging.info(f"Batch {batch_counter}: Avg Reward: {avg_reward:.3f}, Avg CBR: {avg_cbr:.3f}, Avg Rho: {avg_rho:.3f}, Avg Power: {avg_power:.2f}, Avg DR: {avg_dr:.2f}")
                    write_performance_metrics(avg_cbr, avg_rho, avg_reward, avg_power, avg_dr, batch_counter)

                batch_counter += 1

                if total_timesteps_collected >= update_interval_timesteps:
                    logging.info(f"Collected {total_timesteps_collected} timesteps. Updating PPO agent...")
                    
                    final_done_in_buffer = trajectory_buffer.dones[-1].item() == 1.0
                    if final_done_in_buffer:
                        last_s_bootstrap_value_tensor = torch.tensor([0.0]).to(device)
                    else:
                        last_s_bootstrap_value_tensor = trajectory_buffer.values[-1]

                    agent.update(trajectory_buffer, last_s_bootstrap_value_tensor)
                    trajectory_buffer.clear()
                    total_timesteps_collected = 0
                    logging.info("PPO Agent updated.")

                if batch_counter % save_model_interval_batches == 0:
                    agent.save_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)

            except json.JSONDecodeError:
                logging.error(f"Could not decode JSON from client. Data: {data[:200]}")
                conn.sendall(json.dumps({"error": "Invalid JSON received"}).encode('utf-8'))
                continue
            except KeyError as e:
                # vehicle_input_data here refers to the one assigned just before the loop, or the last one in the loop
                logging.error(f"Missing expected key in vehicle data: {e}. Data for veh '{veh_id_in_error}': {vehicle_input_data}")
                if veh_id_in_error: # Check if veh_id was set (i.e., error happened inside or after vehicle loop)
                     responses[veh_id_in_error] = {"error": f"Missing key {e} for vehicle {veh_id_in_error}"}
                # If the error is very early, veh_id_in_error might be None.
                # Consider if a batch-level error response is needed or just skip.
                # The original code would try to use responses[veh_id] which might fail if veh_id isn't defined.
                continue
            except Exception as e:
                logging.error(f"Error processing batch {batch_counter}: {e}", exc_info=True)
                try:
                    conn.sendall(json.dumps({"error": "Server-side processing error"}).encode('utf-8'))
                except Exception as sock_e:
                    logging.error(f"Failed to send error to client: {sock_e}")
                break 

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Major error in server loop: {e}", exc_info=True)
    finally:
        logging.info("Shutting down NNDP PPO server.")
        if conn: # Check if conn was successfully assigned
            conn.close()
        if server: # Check if server socket was created
            server.close()
        # Save models on exit, ensure agent is defined
        if 'agent' in locals() and agent:
            agent.save_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)
        logging.info("Server closed.")

if __name__ == "__main__":
    main()