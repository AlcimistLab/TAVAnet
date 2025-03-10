import sys
import socket
import threading
import json
import os
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces

# If SUMO is used elsewhere in your project, ensure SUMO_HOME is set.
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

#####################################
# 1) Custom Gymnasium Environment for PPO
#####################################
class VANETPPOEnv(gym.Env):
    def __init__(self, beacon_min=1, beacon_max=20, power_min=1, power_max=30):
        super(VANETPPOEnv, self).__init__()
        
        # Continuous action space: two values in [-1, 1] for beacon and power adjustments.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [beacon_rate, power_transmission, vehicle_density]
        self.observation_space = spaces.Box(
            low=np.array([beacon_min, power_min, 0.0], dtype=np.float32),
            high=np.array([beacon_max, power_max, 100.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        self.beacon_min = beacon_min
        self.beacon_max = beacon_max
        self.power_min = power_min
        self.power_max = power_max
        
        # Initial state: can be adjusted or set from real logs later.
        self.state = np.array([10.0, 15.0, 10.0], dtype=np.float32)
        self.episode_ended = False
        
    def reset(self, seed=None, options=None):
        # Reset state and return (observation, info) as required by Gymnasium.
        super().reset(seed=seed)
        self.state = np.array([10.0, 15.0, 10.0], dtype=np.float32)
        self.episode_ended = False
        return self.state, {}
    
    def step(self, action):
        """
        action: [delta_beacon, delta_power] where each delta is in [-1, 1].
        The step applies a scaling factor to update beacon_rate and power_transmission.
        Returns (observation, reward, terminated, truncated, info).
        """
        beacon_rate, power_trans, density = self.state
        
        # Update state based on action deltas (with scaling factors)
        beacon_rate += action[0] * 1.0
        power_trans += action[1] * 3.0
        
        # Clamp the updated values to valid ranges
        beacon_rate = np.clip(beacon_rate, self.beacon_min, self.beacon_max)
        power_trans = np.clip(power_trans, self.power_min, self.power_max)
        
        # Simple reward: lower vehicle density yields higher reward (placeholder)
        reward = float(-density)
        
        # Update the state; vehicle density is unchanged in this minimal example.
        self.state = np.array([beacon_rate, power_trans, density], dtype=np.float32)
        
        terminated = True   # End the episode after one step (for demonstration)
        truncated = False   # No truncation logic implemented
        
        return self.state, reward, terminated, truncated, {}
    
    def set_real_state(self, beacon_rate, power_tx, density):
        """Update the environment's state using real-time logs."""
        self.state = np.array([beacon_rate, power_tx, density], dtype=np.float32)

#####################################
# 2) PPO Server: Socket-Based Inference
#####################################
class PPOServer:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        
        # Initialize the custom environment and check its compatibility.
        self.env = VANETPPOEnv()
        check_env(self.env, warn=True)
        
        # Create a new PPO model or load a pre-trained model.
        self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=1e-4, n_steps=2048, batch_size=64)
        # To load an existing model, uncomment the following line:
        # self.model = PPO.load("my_ppo_model.zip", env=self.env)
    
    def start_server(self):
        """Starts the socket server to listen for incoming vehicular logs."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"[PPO Server] Listening on {self.host}:{self.port}...")
        
        # Handle each client connection concurrently in a new thread.
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"[PPO Server] Connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()
    
    def handle_client(self, client_socket):
        try:
            data = client_socket.recv(4096)
            if not data:
                return
            decoded_data = json.loads(data.decode('utf-8'))
            
            # Extract parameters from the received JSON log.
            beacon_rate = decoded_data.get('beacon_rate', 10)
            power_trans = decoded_data.get('power_transmission', 15)
            density = decoded_data.get('vehicle_density', 10)
            car_id = decoded_data.get('car_id', 'unknown_car')
            
            # Update the environment's state with real data.
            self.env.set_real_state(beacon_rate, power_trans, density)
            
            # Predict action using the PPO model.
            obs = self.env.state
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Perform one step in the environment with the predicted action.
            new_state, reward, terminated, truncated, info = self.env.step(action)
            new_beacon_rate = float(new_state[0])
            new_power_trans = float(new_state[1])
            
            print(f"[PPO] Car {car_id}: old beacon={beacon_rate}, new beacon={new_beacon_rate}, old power={power_trans}, new power={new_power_trans}")
            
            # Send updated parameters back as a JSON response.
            response_data = {
                "car_id": car_id,
                "beacon_rate": new_beacon_rate,
                "power_transmission": new_power_trans
            }
            client_socket.sendall(json.dumps(response_data).encode('utf-8'))
        except Exception as e:
            print(f"[PPO] Error handling client: {e}")
        finally:
            client_socket.close()

#####################################
# 3) Main: Start the PPO Server
#####################################
def main():
    """Main function to start the PPO socket server."""
    ppo_server = PPOServer()
    ppo_server.start_server()

if __name__ == "__main__":
    main()
