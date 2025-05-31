import socket
import torch
import numpy as np
import logging
import json
import os
import csv
from datetime import datetime
from ppo_agent import PPOAgent

# Constants for logging and data collection
LOG_RECEIVED_PATH = 'custom/logs/receive_data.log'
LOG_SENT_PATH = 'custom/logs/sent_data.log'
LOG_DEBUG_ACTION_PATH = 'custom/logs/action.log'
LOG_DEBUG_REWARD_PATH = 'custom/logs/reward_debug.log'
PERFORMANCE_LOG_PATH = 'custom/logs/performance_metrics.csv'

# PPO specific constants
PPO_ROLLOUT_STEPS = 2048  # Number of steps to collect before training

def ensure_log_dirs():
    """Ensure all log directories exist."""
    log_dirs = ['custom/logs', 'model']
    for dir_path in log_dirs:
        os.makedirs(dir_path, exist_ok=True)

def log_data(log_path, data):
    """Log data with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")

def write_performance_metrics(cbr, snr, reward, batch_number):
    """Write performance metrics to CSV file."""
    file_exists = os.path.exists(PERFORMANCE_LOG_PATH)
    
    with open(PERFORMANCE_LOG_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Batch', 'Timestamp', 'CBR', 'SNR', 'Reward'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([batch_number, timestamp, cbr, snr, reward])

def adjust_mcs_based_on_snr(snr):
    """Adjust MCS based on SNR value."""
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
    """Get data rate and receiver sensitivity for given MCS."""
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
        raise ValueError("MCS index must be between 0 and 7.")

def calculate_reward(cbr, snr, prev_tx_power, tx_power):
    """Calculate reward based on CBR, SNR, and power changes."""
    # CBR Reward
    deviation = abs(cbr - 0.65)
    if deviation < 0.1:
        reward_cbr = 10 * (1 - deviation / 0.65)
    else:
        reward_cbr = -5 * deviation

    # SNR Reward
    snr_deviation = abs(snr - 25.0)
    if snr_deviation <= 5.0:
        reward_snr = 10 * (1 - (snr_deviation / 5.0) ** 2)
    else:
        reward_snr = -1 * snr_deviation 

    # Power Stability Reward
    reward_power = -0.1 * (abs(tx_power - prev_tx_power) ** 1.5)
    
    return reward_cbr + reward_snr + reward_power

class PPOTrainer:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        ensure_log_dirs()
        
        # Initialize PPO agent
        self.state_dim = 5  # power, beacon, cbr, neighbors, snr
        self.action_dim = 2  # transmission power, beacon rate
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            clip_epsilon=0.2,
            ppo_epochs=10,
            batch_size=64,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5
        )
        
        # Model paths
        self.actor_model_path = "model/ppo_actor.pth"
        self.critic_model_path = "model/ppo_critic.pth"
        self.load_models()
        
        # Training state
        self.total_steps = 0
        self.batch_counter = 0
        self.last_state = None

    def load_models(self):
        """Load existing models if available."""
        if os.path.exists(self.actor_model_path) and os.path.exists(self.critic_model_path):
            self.agent.load_models(self.actor_model_path, self.critic_model_path)
            logging.info("Loaded existing PPO models.")
        else:
            logging.info("Initialized new PPO models.")

    def save_models(self):
        """Save current models."""
        self.agent.save_models(self.actor_model_path, self.critic_model_path)
        logging.info("Saved PPO models.")

    def process_vehicle(self, veh_id, vehicle_data):
        """Process data for a single vehicle."""
        # Extract state information
        current_power = vehicle_data['transmissionPower']
        current_data = vehicle_data['dataRate']
        cbr = vehicle_data['CBR']
        neighbors = vehicle_data['neighbors']
        snr = vehicle_data['SNR']
        
        # Create state array
        state = np.array([current_power, current_beacon, cbr, neighbors, snr], dtype=np.float32)
        
        # Get action from PPO agent
        action_raw, log_prob, value_old = self.agent.select_action_for_rollout(state)
        
        # Scale actions to environment ranges
        new_power = (action_raw[0] + 1) / 2 * (30 - 1) + 1  # [1, 30]
        new_data = (action_raw[1] + 1) / 2 * (20 - 1) + 1  # [1, 20]
        
        # Calculate reward
        reward = calculate_reward(cbr, snr, current_power, new_power)
        
        # Store transition in PPO buffer
        done = False  # Assuming continuous operation
        self.agent.store_transition(state, action_raw, reward, None, done, log_prob, value_old)
        
        # Update last state for GAE calculation
        self.last_state = state
        
        # Determine MCS based on current SNR
        mcs = adjust_mcs_based_on_snr(snr)
        
        # Log performance metrics
        write_performance_metrics(cbr, snr, reward, self.batch_counter)
        
        return {
            "transmissionPower": float(new_power),
            "beaconRate": float(new_beacon),
            "MCS": mcs
        }

    def should_train(self):
        """Check if we should perform PPO training."""
        return self.total_steps >= PPO_ROLLOUT_STEPS

    def train(self):
        """Perform PPO training if enough steps collected."""
        if not self.should_train():
            return
            
        logging.info(f"Starting PPO training after {self.total_steps} steps...")
        
        # Calculate last value estimate for GAE
        last_value = 0.0
        if self.last_state is not None:
            state_tensor = torch.FloatTensor(self.last_state).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                last_value = self.agent.value_net(state_tensor).cpu().item()
        
        # Train PPO
        self.agent.train(last_value)
        
        # Reset step counter and save models
        self.total_steps = 0
        self.save_models()
        logging.info("PPO training completed.")

    def run(self):
        """Main training loop."""
        # Setup server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("localhost", 5000))
        server.listen(1)
        logging.info("Listening on port 5000...")
        
        try:
            conn, addr = server.accept()
            logging.info(f"Connected by {addr}")
            
            while True:
                # Receive data
                data = conn.recv(65536)
                if not data:
                    break
                
                try:
                    # Process batch
                    self.batch_counter += 1
                    batch_data = json.loads(data.decode())
                    logging.info(f"[BATCH] Received data for {len(batch_data)} vehicles")
                    log_data(LOG_RECEIVED_PATH, f"Batch {self.batch_counter}: {json.dumps(batch_data, indent=4)}")
                    
                    # Process each vehicle
                    responses = {}
                    for veh_id, vehicle_data in batch_data.items():
                        responses[veh_id] = self.process_vehicle(veh_id, vehicle_data)
                        self.total_steps += 1
                    
                    # Send responses
                    response_data = json.dumps(responses).encode('utf-8')
                    conn.sendall(response_data)
                    log_data(LOG_SENT_PATH, f"Batch {self.batch_counter}: {json.dumps(responses, indent=4)}")
                    
                    # Train if needed
                    self.train()
                    
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing batch: {e}", exc_info=True)
                    continue
                    
        except Exception as e:
            logging.error(f"Server error: {e}", exc_info=True)
        finally:
            # Clean up
            if 'conn' in locals():
                conn.close()
            server.close()
            self.save_models()
            logging.info("Server closed and models saved.")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.run() 