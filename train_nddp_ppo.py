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
    # g(x) = -sgn(x - x_T) * x, where x=CBR, x_T=MBL_TARGET_CBR
    # The paper's text says "a positive reward increase is obtained as long as the CBR approaches the target value".
    # The formula g(x) = -sgn(x-x_T)x might not directly reflect this for x < x_T.
    # Let's use a more direct approach for reward based on proximity to MBL, and the paper's bonus.
    
    # Deviation from MBL
    cbr_deviation = cbr_measured - MBL_TARGET_CBR
    
    # CBR component of reward based on paper's description:
    # "a positive reward increase is obtained as long as the CBR approaches the target value"
    # Let's use a quadratic penalty for deviation, or a shaped reward.
    # The paper's g(x) is: if x < MBL, g(x) = x. if x > MBL, g(x) = -x.
    # This means reward increases with CBR up to MBL, then decreases.
    if cbr_measured <= MBL_TARGET_CBR:
        g_cbr = cbr_measured 
    else:
        g_cbr = -cbr_measured # Penalize exceeding MBL more harshly
        # Alternative: g_cbr = MBL_TARGET_CBR - abs(cbr_deviation) # Peaks at MBL

    reward1_base = OMEGA_C * g_cbr
    
    # Additive bonus/penalty from paper (page 122073)
    if abs(cbr_deviation) <= 0.025:
        reward1_bonus = 10.0
    else:
        reward1_bonus = -0.1
    reward1 = reward1_base + reward1_bonus

    # Term 2: Power adequacy (Path loss and sensitivity)
    # l = A * (ds^beta)
    path_loss_linear = PATH_LOSS_A_CONST * (SAFETY_DISTANCE_DS_M ** PATH_LOSS_EXPONENT_BETA)
    path_loss_db = 10 * np.log10(path_loss_linear) if path_loss_linear > 0 else float('inf') # Avoid log(0)
    
    # Reward term: -omega_p * |(Sr + l) - p|
    # This penalizes deviation of p from (Sr + l)
    target_power_for_reliability = sr_for_new_data_rate_dbm + path_loss_db
    reward2 = -OMEGA_P * abs(target_power_for_reliability - new_tx_power_dbm)

    # Term 3: Data rate penalty (encourage lower data rates if possible)
    # Reward term: -omega_d * (d ^ omega_e)
    reward3 = -OMEGA_D * (new_data_rate_mbps ** OMEGA_E)
    
    total_reward = reward1 + reward2 + reward3
    
    # Log individual reward components for debugging
    # log_data_to_file(LOG_ACTION_REWARD_PATH, f"RewardCalc: CBR={cbr_measured:.3f}, P_new={new_tx_power_dbm:.2f}, D_new={new_data_rate_mbps:.2f}, Sr={sr_for_new_data_rate_dbm:.1f} | R1_base={reward1_base:.3f}, R1_bonus={reward1_bonus:.3f}, R2={reward2:.3f}, R3={reward3:.3f} | Total={total_reward:.3f}")

    return total_reward

# --- MCS from MATLAB to NNDP Data Rate (Mbps) Mapping ---
# This mapping is based on common interpretations of MCS indices
# and how they might correspond to the NNDP defined data rates.
# Adjust if your MATLAB MCS indices have a different specific mapping.
MCS_TO_NNDP_DATA_RATE = {
    0: 3.0,   # Typically BPSK 1/2
    1: 6.0,   # Typically QPSK 1/2 (NNDP 4.5 is BPSK 3/4; NNDP 6.0 is QPSK 1/2)
    2: 9.0,   # Typically QPSK 3/4
    3: 12.0,  # Typically 16-QAM 1/2
    4: 18.0,  # Typically 16-QAM 3/4
    5: 24.0,  # Typically 64-QAM 2/3
    6: 27.0,  # Typically 64-QAM 3/4
    # For MCS values from MATLAB not explicitly in this map (e.g., > 6),
    # we'll default to MAX_NNDP_DATA_RATE and log a warning.
}

# --- Main Training Script ---
def main():
    # NNDP PPO Agent Hyperparameters
    state_dim = 3  # current_power, current_data_rate, vehicle_density (rho)
    action_dim = 2 # delta_power, delta_data_rate
    lr_actor = 3e-5 # Learning rates can be sensitive
    lr_critic = 1e-4
    gamma = 0.99
    ppo_epochs = 10 # Number of epochs to train on the collected data
    ppo_clip_epsilon = 0.2
    gae_lambda = 0.95
    entropy_coefficient = 0.01 # Encourages exploration
    
    # Training control
    # PPO updates after N timesteps (experiences from all vehicles in OMNeT batches count)
    # The OMNeT client sends a batch of vehicle data. Each vehicle's data is one "timestep" or experience.
    update_interval_timesteps = 2048 # Collect this many experiences before PPO update
    save_model_interval_batches = 100 # Save model every X batches from OMNeT
    
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

    # Socket setup
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 5000)) # Use the same port as in the template
    server.listen(1)
    logging.info("NNDP PPO Training Server listening on port 5000...")
    
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

                    # --- MODIFIED STATE VARIABLE EXTRACTION ---
                    # 1. Current Power from MATLAB's 'transmissionPower'
                    current_power = float(vehicle_input_data['transmissionPower'])

                    # 2. Current Data Rate from MATLAB's 'MCS'
                    mcs_from_matlab = int(vehicle_input_data['MCS'])
                    current_data_rate = MCS_TO_NNDP_DATA_RATE.get(mcs_from_matlab)
                    if current_data_rate is None:
                        logging.warning(f"Veh {veh_id}: MCS value {mcs_from_matlab} from MATLAB not in explicit map. Defaulting data rate to {MAX_NNDP_DATA_RATE} Mbps.")
                        current_data_rate = MAX_NNDP_DATA_RATE
                    
                    # 3. Vehicle Density from MATLAB's 'neighbors' count + 1 (for self)
                    #    The state variable in Python is often called vehicle_density_rho or similar.
                    vehicle_density_rho = float(vehicle_input_data['neighbors']) + 1.0

                    # CBR for reward calculation from MATLAB's 'CBR'
                    cbr_measured = float(vehicle_input_data['CBR'])
                    # --- END OF MODIFIED STATE VARIABLE EXTRACTION ---

                    # Construct the state vector for the PPO agent
                    nndp_state = [current_power, current_data_rate, vehicle_density_rho]

                    # Get action, log_prob, and value from PPO agent
                    # Action from actor is in [-1, 1] for mean, then sampled.
                    # These are normalized deltas.
                    norm_action_deltas, log_prob, value_s = agent.select_action_and_evaluate(nndp_state)
                    
                    # Scale normalized deltas to actual deltas
                    # delta_power is action_deltas[0] * MAX_POWER_DELTA_SCALE
                    # delta_data_rate is action_deltas[1] * MAX_DATA_RATE_DELTA_SCALE
                    delta_power = norm_action_deltas[0] * MAX_POWER_DELTA_SCALE
                    delta_data_rate = norm_action_deltas[1] * MAX_DATA_RATE_DELTA_SCALE
                    
                    # Apply deltas and clip to valid ranges
                    new_power_continuous = current_power + delta_power
                    new_power_clipped = np.clip(new_power_continuous, MIN_TX_POWER_DBM, MAX_TX_POWER_DBM)
                    
                    new_data_rate_continuous = current_data_rate + delta_data_rate
                    # Discretize data rate and get its Sr
                    new_data_rate_discrete, sr_for_new_rate, _ = get_nndp_discrete_data_rate_info(new_data_rate_continuous)

                    # Calculate NNDP reward using the state *after* action (new_power, new_data_rate)
                    # and the cbr from state *s* (or s' if available).
                    # The paper's reward r(s,a) implies reward is based on s and a, leading to s'.
                    # The components of reward use p (new_tx_power_dbm) and d (new_data_rate_mbps).
                    reward = calculate_nndp_reward(cbr_measured, new_power_clipped, new_data_rate_discrete, sr_for_new_rate)
                    
                    # Store transition in PPO trajectory buffer
                    # 'done' is usually False in continuous tasks unless an episode ends.
                    # Assuming OMNeT++ runs continuously for now.
                    done_flag = False 
                    trajectory_buffer.add(nndp_state, norm_action_deltas, log_prob, reward, done_flag, value_s)
                    total_timesteps_collected += 1

                    # Prepare response for this vehicle
                    responses[veh_id] = {
                        "new_tx_power_dbm": float(new_power_clipped),
                        "new_data_rate_mbps": float(new_data_rate_discrete)
                        # OMNeT++ will use these to update the vehicle for the next step
                    }
                    
                    # For logging batch averages
                    current_batch_rewards.append(reward)
                    current_batch_cbrs.append(cbr_measured)
                    current_batch_rhos.append(vehicle_density_rho)
                    current_batch_powers.append(new_power_clipped)
                    current_batch_data_rates.append(new_data_rate_discrete)

                    log_data_to_file(LOG_ACTION_REWARD_PATH, 
                                     f"Batch {batch_counter}, Veh {veh_id}: State={nndp_state}, NormActionDeltas={norm_action_deltas}, "
                                     f"ScaledDeltas=[{delta_power:.2f}, {delta_data_rate:.2f}], NewP={new_power_clipped:.2f}, NewDR={new_data_rate_discrete:.2f}, "
                                     f"CBR={cbr_measured:.3f}, Rho={vehicle_density_rho:.3f}, Reward={reward:.3f}, ValueS={value_s:.3f}")


                # Send batch of responses to client
                response_data_str = json.dumps(responses)
                conn.sendall(response_data_str.encode('utf-8'))
                log_data_to_file(LOG_SENT_PATH, f"Batch {batch_counter}: {response_data_str}")
                
                if current_batch_rewards: # if any vehicles were processed
                    avg_reward = np.mean(current_batch_rewards)
                    avg_cbr = np.mean(current_batch_cbrs)
                    avg_rho = np.mean(current_batch_rhos)
                    avg_power = np.mean(current_batch_powers)
                    avg_dr = np.mean(current_batch_data_rates)
                    logging.info(f"Batch {batch_counter}: Avg Reward: {avg_reward:.3f}, Avg CBR: {avg_cbr:.3f}, Avg Rho: {avg_rho:.3f}, Avg Power: {avg_power:.2f}, Avg DR: {avg_dr:.2f}")
                    write_performance_metrics(avg_cbr, avg_rho, avg_reward, avg_power, avg_dr, batch_counter)

                batch_counter += 1

                # PPO Update logic
                if total_timesteps_collected >= update_interval_timesteps:
                    logging.info(f"Collected {total_timesteps_collected} timesteps. Updating PPO agent...")
                    # To compute GAE correctly for the last trajectory, we need V(s_last_next_state).
                    # This requires getting the *next* state from OMNeT++ for the last items in the buffer,
                    # or if an episode ends, the value is 0.
                    # For simplicity in this structure, if the buffer ends mid-episode,
                    # we can estimate V(s_N) using the current critic for the *last observed state* in the buffer.
                    # However, a more standard approach is to get V(s_N) for the *actual next state* s_N.
                    # Let's assume for now that the last state in the buffer is not necessarily terminal.
                    # We need the value of the state that *would have followed* the last state in the buffer.
                    # This is complex without knowing that next state.
                    # A common simplification: if the trajectory is cut short (not 'done'),
                    # bootstrap with the current critic's value of the last state in the trajectory.
                    # The `compute_gae` in `PPOAgentNNDP` expects `next_value_tensor` for V(s_T).
                    # This `s_T` is the state *after* the last action in the buffer.
                    # If we don't have it, we can use V(last_state_in_buffer) as an approximation,
                    # or set it to 0 if the last `done` was true.

                    # Simplified: If the last transition in buffer is (s_L, a_L, r_L, ..., V(s_L)),
                    # and it's not 'done', GAE needs V(s_L+1).
                    # If we don't have s_L+1, we can use V(s_L) from critic as a proxy if not done.
                    # Or, if the buffer always collects full episodes, or fixed length, this is handled.
                    
                    # For now, let's assume if the last 'done' in buffer is False, we use critic's estimate
                    # of the last state *in the buffer* as the bootstrap value.
                    # This is not ideal but a common simplification if next state is unknown.
                    # A better way: the `select_action_and_evaluate` already gives V(s).
                    # The GAE calculation needs V(s_t) for all t, and V(s_N) for the state *after* the last action.
                    # The buffer stores V(s_t). If the last `done` is false, we need V(s_N).
                    # We can get this by passing the *next state* (if we knew it) to critic.
                    # Or, if the last state in buffer is s_N-1, then `last_state_value_tensor` should be V(s_N).
                    # The current buffer stores V(s_t) for each s_t.
                    
                    # Let's refine: the `compute_gae` uses `values` (which are V(s_0)...V(s_T-1))
                    # and `next_value_tensor` (which is V(s_T)).
                    # If the last `done` in the buffer is true, `next_value_tensor` should be 0.
                    # Otherwise, we need an estimate for V(s_T).
                    # This `s_T` is the state that results from the last action in the buffer.
                    # We don't have it directly from OMNeT++ at this point of update.
                    #
                    # Simplification: if the last `dones` entry is False, estimate V(s_T) using the
                    # critic on the *last state recorded in the buffer* if we assume s_T ~ s_T-1 for value.
                    # Or, more simply, if not done, the value function itself provides the bootstrap.
                    # The `values_from_rollout` are V(s_0)...V(s_N-1). GAE needs V(s_N).
                    # If last `dones[-1]` is true, `last_bootstrap_value = 0`.
                    # Else, `last_bootstrap_value = agent.critic(trajectory_buffer.states[-1].unsqueeze(0)).item()`.
                    # This is V(s_N-1), not V(s_N).
                    
                    # A common way is to ensure trajectories are collected up to a 'done' or max length,
                    # then the last value is either 0 or V(s_final_next).
                    # Given the OMNeT interaction, we might not have s_final_next.
                    # Let `last_state_value_tensor` be None for now, `compute_gae` will handle it.
                    # The `PPOAgentNNDP.compute_gae` is set up to take `next_value_tensor`.
                    # If the last `done` in the buffer is true, this should be 0.
                    # Otherwise, it should be `agent.critic(next_state_of_last_action)`.
                    # Since we don't have `next_state_of_last_action` at update time easily,
                    # we can pass the value of the *last state in the buffer* if not done.
                    
                    final_done_in_buffer = trajectory_buffer.dones[-1].item() == 1.0
                    if final_done_in_buffer:
                        last_s_bootstrap_value_tensor = torch.tensor([0.0]).to(device)
                    else:
                        # Use the V(s) that was already computed for the last state during rollout
                        last_s_bootstrap_value_tensor = trajectory_buffer.values[-1]


                    agent.update(trajectory_buffer, last_s_bootstrap_value_tensor)
                    trajectory_buffer.clear()
                    total_timesteps_collected = 0
                    logging.info("PPO Agent updated.")

                if batch_counter % save_model_interval_batches == 0:
                    agent.save_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)

            except json.JSONDecodeError:
                logging.error(f"Could not decode JSON from client. Data: {data[:200]}") # Log first 200 chars
                # Potentially send an error response or close connection
                conn.sendall(json.dumps({"error": "Invalid JSON received"}).encode('utf-8'))
                continue # Or break, depending on desired error handling
            except KeyError as e:
                logging.error(f"Missing expected key in vehicle data: {e}. Data: {vehicle_input_data}")
                # Send error for this specific vehicle or skip
                responses[veh_id] = {"error": f"Missing key {e}"} # Example
                continue
            except Exception as e:
                logging.error(f"Error processing batch {batch_counter}: {e}", exc_info=True)
                # Potentially send a general error response
                try:
                    conn.sendall(json.dumps({"error": "Server-side processing error"}).encode('utf-8'))
                except Exception as sock_e:
                    logging.error(f"Failed to send error to client: {sock_e}")
                break # Critical error, stop server or try to recover

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Major error in server loop: {e}", exc_info=True)
    finally:
        logging.info("Shutting down NNDP PPO server.")
        if conn: 
            conn.close()
        if server: 
            server.close()
        if 'agent' in locals() and agent:
            agent.save_models(ACTOR_MODEL_SAVE_PATH, CRITIC_MODEL_SAVE_PATH)
        logging.info("Server closed.")

if __name__ == "__main__":
    main()
