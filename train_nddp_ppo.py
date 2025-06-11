import socket
import torch
import numpy as np
import logging
import json
import os
import csv
from datetime import datetime
from nndp_ppo_core import PPOAgentNNDP, TrajectoryBufferNNDP # Import PPO components
import struct

# --- NNDP Paper Constants & Helper Data ---
NNDP_DATA_RATES_INFO = {
    3.0: {"Sr_dbm": -85, "Cd_bits_per_symbol": 48, "modulation": "BPSK", "cr": "1/2"},
    4.5: {"Sr_dbm": -84, "Cd_bits_per_symbol": 48, "modulation": "BPSK", "cr": "3/4"},
    6.0: {"Sr_dbm": -82, "Cd_bits_per_symbol": 96, "modulation": "QPSK", "cr": "1/2"},
    9.0: {"Sr_dbm": -80, "Cd_bits_per_symbol": 96, "modulation": "QPSK", "cr": "3/4"},
    12.0: {"Sr_dbm": -77, "Cd_bits_per_symbol": 192, "modulation": "16-QAM", "cr": "1/2"},
    18.0: {"Sr_dbm": -73, "Cd_bits_per_symbol": 192, "modulation": "16-QAM", "cr": "3/4"},
    24.0: {"Sr_dbm": -69, "Cd_bits_per_symbol": 288, "modulation": "64-QAM", "cr": "2/3"},
    27.0: {"Sr_dbm": -68, "Cd_bits_per_symbol": 288, "modulation": "64-QAM", "cr": "3/4"},
}
VALID_NNDP_DATA_RATES = sorted(NNDP_DATA_RATES_INFO.keys())
MIN_NNDP_DATA_RATE = VALID_NNDP_DATA_RATES[0]
MAX_NNDP_DATA_RATE = VALID_NNDP_DATA_RATES[-1]

# --- MCS from MATLAB to NNDP Data Rate (Mbps) Mapping ---
MCS_TO_NNDP_DATA_RATE = {
    0: 3.0,   # BPSK 1/2
    1: 6.0,   # QPSK 1/2
    2: 9.0,   # QPSK 3/4
    3: 12.0,  # 16-QAM 1/2
    4: 18.0,  # 16-QAM 3/4
    5: 24.0,  # 64-QAM 2/3
    6: 27.0,  # 64-QAM 3/4
}

# --- NNDP Data Rate (Mbps) to MCS Index Mapping (for response to MATLAB File A) ---
NNDP_DATA_RATE_TO_MCS = {v: k for k, v in MCS_TO_NNDP_DATA_RATE.items()}

MIN_TX_POWER_DBM = 1.0
MAX_TX_POWER_DBM = 30.0

OMEGA_C = 2.0
OMEGA_P = 0.25
OMEGA_D = 0.1
OMEGA_E = 0.8
MBL_TARGET_CBR = 0.6
SAFETY_DISTANCE_DS_M = 100.0
PATH_LOSS_EXPONENT_BETA = 2.5
CHANNEL_FREQUENCY_HZ = 5.9e9
SPEED_OF_LIGHT_MPS = 3.0e8

LAMBDA_WAVELENGTH_M = SPEED_OF_LIGHT_MPS / CHANNEL_FREQUENCY_HZ
PATH_LOSS_A_CONST = (4 * np.pi / LAMBDA_WAVELENGTH_M)**2

MAX_POWER_DELTA_SCALE = 5.0
MAX_DATA_RATE_DELTA_SCALE = 3.0

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

def get_nndp_discrete_data_rate_info(continuous_data_rate_mbps):
    clipped_continuous_rate = np.clip(continuous_data_rate_mbps, MIN_NNDP_DATA_RATE, MAX_NNDP_DATA_RATE)
    final_discrete_rate = min(VALID_NNDP_DATA_RATES, key=lambda x: abs(x - clipped_continuous_rate))
    info = NNDP_DATA_RATES_INFO[final_discrete_rate]
    return final_discrete_rate, info["Sr_dbm"], info["Cd_bits_per_symbol"]

def calculate_nndp_reward(cbr_measured, new_tx_power_dbm, new_data_rate_mbps, sr_for_new_data_rate_dbm):
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
    path_loss_linear = PATH_LOSS_A_CONST * (SAFETY_DISTANCE_DS_M ** PATH_LOSS_EXPONENT_BETA)
    path_loss_db = 10 * np.log10(path_loss_linear) if path_loss_linear > 0 else float('inf')
    target_power_for_reliability = sr_for_new_data_rate_dbm + path_loss_db
    reward2 = -OMEGA_P * abs(target_power_for_reliability - new_tx_power_dbm)
    reward3 = -OMEGA_D * (new_data_rate_mbps ** OMEGA_E)
    total_reward = reward1 + reward2 + reward3
    return total_reward

def main():
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
    save_model_interval_batches = 100 # This controls model saving frequency
    
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
    
    conn = None
    try:
        conn, addr = server.accept()
        logging.info(f"Connected by {addr}")
        
        batch_counter = 0
        total_timesteps_collected = 0

        while True:
            length_prefix_bytes = conn.recv(4)
            if not length_prefix_bytes or len(length_prefix_bytes) < 4:
                logging.error("Failed to receive complete length prefix from client. Closing connection.")
                break
            try:
                message_length = struct.unpack('<I', length_prefix_bytes)[0]
                logging.info(f"Expecting message of {message_length} bytes.")
            except struct.error as e:
                logging.error(f"Could not unpack message length: {e}. Raw prefix: {length_prefix_bytes.hex()}. Closing connection.")
                break

            data_chunks = []
            bytes_received = 0
            while bytes_received < message_length:
                chunk_size_to_read = min(message_length - bytes_received, 65536)
                chunk = conn.recv(chunk_size_to_read)
                if not chunk:
                    logging.error(f"Connection broken while receiving message body. Expected {message_length}, got {bytes_received}.")
                    message_length = -1 
                    break
                data_chunks.append(chunk)
                bytes_received += len(chunk)
            if message_length == -1: 
                break 
            data = b''.join(data_chunks)
            
            vehicle_input_data = {} 
            veh_id_in_error = None

            try:
                try:
                    batch_data_str = data.decode('utf-8')
                except UnicodeDecodeError as ude:
                    logging.error(f"UnicodeDecodeError: {ude}. Raw data (first 50 bytes as hex): {data[:50].hex()}")
                    conn.sendall(json.dumps({"error": "Server received non-UTF-8 data"}).encode('utf-8'))
                    continue 

                batch_data = json.loads(batch_data_str)
                log_data_to_file(LOG_RECEIVED_PATH, f"Batch {batch_counter}: {batch_data_str}")

                responses = {}
                current_batch_rewards = []
                current_batch_cbrs = []
                current_batch_rhos = []
                current_batch_powers = []
                current_batch_data_rates = []

                for veh_id, vehicle_input_data_loop in batch_data.items():
                    veh_id_in_error = veh_id 
                    vehicle_input_data = vehicle_input_data_loop

                    # Extract state from MATLAB data using MATLAB's keys
                    current_power = float(vehicle_input_data['transmissionPower'])
                    mcs_from_matlab = int(vehicle_input_data['MCS'])
                    current_data_rate = MCS_TO_NNDP_DATA_RATE.get(mcs_from_matlab)
                    if current_data_rate is None:
                        logging.warning(f"Veh {veh_id}: MCS value {mcs_from_matlab} from MATLAB not in explicit map. Defaulting data rate to {MAX_NNDP_DATA_RATE} Mbps.")
                        current_data_rate = MAX_NNDP_DATA_RATE
                    vehicle_density_rho = float(vehicle_input_data['neighbors']) + 1.0
                    cbr_measured = float(vehicle_input_data['CBR'])
                    
                    # Get current beacon rate from MATLAB's input to echo it back
                    try:
                        beacon_rate_from_matlab = float(vehicle_input_data['beaconRate'])
                    except KeyError:
                        logging.warning(f"Veh {veh_id}: 'beaconRate' not found in input from MATLAB. Defaulting to 10Hz for response.")
                        beacon_rate_from_matlab = 10.0

                    nndp_state = [current_power, current_data_rate, vehicle_density_rho]
                    norm_action_deltas, log_prob, value_s = agent.select_action_and_evaluate(nndp_state)
                    
                    delta_power = norm_action_deltas[0] * MAX_POWER_DELTA_SCALE
                    delta_data_rate = norm_action_deltas[1] * MAX_DATA_RATE_DELTA_SCALE
                    
                    new_power_continuous = current_power + delta_power
                    new_power_clipped = np.clip(new_power_continuous, MIN_TX_POWER_DBM, MAX_TX_POWER_DBM)
                    
                    new_data_rate_continuous = current_data_rate + delta_data_rate
                    new_data_rate_discrete, sr_for_new_rate, _ = get_nndp_discrete_data_rate_info(new_data_rate_continuous)

                    reward = calculate_nndp_reward(cbr_measured, new_power_clipped, new_data_rate_discrete, sr_for_new_rate)
                    
                    done_flag = False 
                    trajectory_buffer.add(nndp_state, norm_action_deltas, log_prob, reward, done_flag, value_s)
                    total_timesteps_collected += 1

                    # --- RESPONSE FOR MATLAB FILE A ---
                    new_mcs_for_matlab = NNDP_DATA_RATE_TO_MCS.get(new_data_rate_discrete)
                    if new_mcs_for_matlab is None:
                        logging.warning(f"Veh {veh_id}: Could not map new_data_rate_discrete {new_data_rate_discrete} Mbps back to MCS for response. Sending default MCS 0.")
                        new_mcs_for_matlab = 0 

                    responses[veh_id] = {
                        "transmissionPower": float(new_power_clipped),
                        "MCS": int(new_mcs_for_matlab),
                        "beaconRate": float(beacon_rate_from_matlab) 
                    }
                    # --- END OF RESPONSE MODIFICATION ---
                    
                    current_batch_rewards.append(reward)
                    current_batch_cbrs.append(cbr_measured) 
                    current_batch_rhos.append(vehicle_density_rho) 
                    current_batch_powers.append(new_power_clipped)
                    current_batch_data_rates.append(new_data_rate_discrete)

                    log_data_to_file(LOG_ACTION_REWARD_PATH, 
                                     f"Batch {batch_counter}, Veh {veh_id}: State={nndp_state}, NormActionDeltas={norm_action_deltas}, "
                                     f"RespToMATLAB=[P_tx:{new_power_clipped:.2f}, MCS:{new_mcs_for_matlab}, BR:{beacon_rate_from_matlab:.1f}], " # Updated Log
                                     f"NewDR_disc={new_data_rate_discrete:.2f}, CBR_meas={cbr_measured:.3f}, Rho_calc={vehicle_density_rho:.3f}, Reward={reward:.3f}, ValueS={value_s:.3f}")

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

                if batch_counter % save_model_interval_batches == 0: # Model saving logic
                    agent.save_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)
                    logging.info(f"Saved models at batch {batch_counter}")


            except json.JSONDecodeError:
                logging.error(f"Could not decode JSON from client. Data: {data[:200]}")
                conn.sendall(json.dumps({"error": "Invalid JSON received"}).encode('utf-8'))
                continue
            except KeyError as e:
                logging.error(f"Missing expected key in vehicle data: {e}. Data for veh '{veh_id_in_error}': {vehicle_input_data}")
                if veh_id_in_error:
                     responses[veh_id_in_error] = {"error": f"Missing key {e} for vehicle {veh_id_in_error}"} # This response might not be sent if we 'continue'
                continue # Skip this problematic batch or vehicle
            except Exception as e:
                logging.error(f"Error processing batch {batch_counter}: {e}", exc_info=True)
                try:
                    conn.sendall(json.dumps({"error": "Server-side processing error"}).encode('utf-8'))
                except Exception as sock_e:
                    logging.error(f"Failed to send error to client: {sock_e}")
                break 

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except ConnectionResetError:
        logging.error("Connection reset by peer. MATLAB may have closed.")
    except Exception as e:
        logging.error(f"Major error in server loop: {e}", exc_info=True)
    finally:
        logging.info("Shutting down NNDP PPO server.")
        if conn: 
            conn.close()
        if server: 
            server.close()
        if 'agent' in locals() and agent: # Ensure agent is defined before saving
            agent.save_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)
            logging.info("Final models saved on shutdown.")
        logging.info("Server closed.")

if __name__ == "__main__":
    main()