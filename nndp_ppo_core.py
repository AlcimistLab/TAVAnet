import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class TrajectoryBufferNNDP:
    """
    Buffer to store trajectories for PPO.
    PPO is on-policy, so the buffer is typically filled for a certain number of steps
    and then cleared after the policy update.
    """
    def __init__(self, device):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = [] # Stores V(s_t) from critic during rollout
        self.device = device

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(torch.FloatTensor(state).to(self.device))
        self.actions.append(torch.FloatTensor(action).to(self.device)) # Actions are deltas
        self.log_probs.append(torch.FloatTensor([log_prob]).to(self.device))
        self.rewards.append(torch.FloatTensor([reward]).to(self.device))
        self.dones.append(torch.FloatTensor([1.0 if done else 0.0]).to(self.device))
        self.values.append(torch.FloatTensor([value]).to(self.device))

    def get_trajectories(self):
        """
        Returns all stored trajectories as tensors and clears the buffer.
        """
        # Convert lists of tensors to single tensors
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.stack(self.actions)
        log_probs_tensor = torch.stack(self.log_probs).squeeze(-1) # Remove last dim if it's 1
        rewards_tensor = torch.stack(self.rewards).squeeze(-1)
        dones_tensor = torch.stack(self.dones).squeeze(-1)
        values_tensor = torch.stack(self.values).squeeze(-1)
        
        return states_tensor, actions_tensor, log_probs_tensor, rewards_tensor, dones_tensor, values_tensor

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)

class ActorNetworkNNDP(nn.Module):
    """
    Actor Network for NNDP PPO.
    Input: state (current_power, current_data_rate, vehicle_density)
    Output: mean and log_std for a Gaussian distribution of action deltas (delta_power, delta_data_rate)
    Architecture: MLP (input_dim -> 64 -> 64 -> output_dim_mean/log_std)
    Activation: ReLU for hidden, Tanh for mean output (scaled later), Clamp for log_std.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_min=-20, log_std_max=2):
        super(ActorNetworkNNDP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Initialize weights for stability (optional, but can help)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # For mean layer, smaller weights might be good if actions are scaled by tanh
        # nn.init.orthogonal_(self.mean_layer.weight, 0.01)
        # nn.init.constant_(self.mean_layer.bias, 0)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.mean_layer(x)) # Output mean in [-1, 1]
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

class CriticNetworkNNDP(nn.Module):
    """
    Critic Network for NNDP PPO.
    Input: state (current_power, current_data_rate, vehicle_density)
    Output: state value (scalar)
    Architecture: MLP (input_dim -> 64 -> 64 -> 1)
    Activation: ReLU for hidden layers.
    """
    def __init__(self, state_dim, hidden_dim=64):
        super(CriticNetworkNNDP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Output a single value

        # Initialize weights (optional)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # nn.init.orthogonal_(self.fc3.weight, 1.0) # Value head
        # nn.init.constant_(self.fc3.bias, 0)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPOAgentNNDP:
    """
    PPO Agent for NNDP.
    Implements the Proximal Policy Optimization algorithm.
    """
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, ppo_epochs=10, ppo_clip_epsilon=0.2, 
                 gae_lambda=0.95, entropy_coefficient=0.01,
                 max_grad_norm=0.5, device=None,
                 action_std_init=0.5): # Initial std for actions
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm

        self.actor = ActorNetworkNNDP(state_dim, action_dim).to(self.device)
        self.critic = CriticNetworkNNDP(state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # The paper mentions "A hyperbolic tangent activation function is employed for PPO by default"
        # This is implemented in the ActorNetworkNNDP for the mean.
        # The paper also mentions "2 layers of 64 nodes" which is used.

    def select_action_and_evaluate(self, state):
        """
        Selects an action for the current state for rollout, and gets its log_prob and value.
        state: A single state, numpy array or list.
        Returns:
            action: Sampled action (numpy array).
            log_prob: Log probability of the action (scalar).
            value: Critic's value of the state (scalar).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0) # Add batch dim
            action_mean, action_log_std = self.actor(state_tensor)
            action_std = torch.exp(action_log_std)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1) # Sum over action dimensions
            
            value = self.critic(state_tensor)
            
        return action.squeeze(0).cpu().numpy(), action_log_prob.item(), value.item()

    def evaluate_trajectory_batch(self, states_batch, actions_batch):
        """
        Evaluates a batch of states and actions using the current policy.
        Used during the PPO update phase.
        states_batch: Tensor of states.
        actions_batch: Tensor of actions.
        Returns:
            log_probs: Log probabilities of the actions under the current policy.
            state_values: Critic's values for the states.
            entropy: Entropy of the action distribution.
        """
        action_mean, action_log_std = self.actor(states_batch)
        action_std = torch.exp(action_log_std)
        
        dist = Normal(action_mean, action_std)
        
        action_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1) # Sum entropy over action dimensions
        
        state_values = self.critic(states_batch).squeeze(-1) # Remove last dim
        
        return action_log_probs, state_values, entropy

    def compute_gae(self, rewards, dones, values, next_value_tensor=None):
        """
        Computes Generalized Advantage Estimation (GAE).
        rewards, dones, values are tensors from the trajectory buffer.
        next_value_tensor: Value of the state after the last state in the trajectory.
                           If the last state was terminal (done=1), this should be 0.
                           If trajectory doesn't end with 'done', this is V(s_T+1).
        """
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        
        # If trajectory buffer doesn't include the value of the *next* state for the last transition,
        # we need it. If the episode ended, next_value is 0. Otherwise, it's V(s_last_next).
        # For simplicity, the buffer stores V(s_t). We need V(s_t+1).
        # The 'values' tensor from buffer is V(s_0), V(s_1), ..., V(s_T-1).
        # We need V(s_T) for the last delta.

        num_steps = len(rewards)
        full_values = torch.cat((values, next_value_tensor.unsqueeze(0)), dim=0) if next_value_tensor is not None else values
        # If next_value_tensor is None, it means the last step was terminal or we are not bootstrapping
        # from V(s_T+1). For GAE, we typically need V(s_t+1).

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # If next_value_tensor is provided, it's V(s_T) where T is num_steps.
                # rewards[t] is r_T-1, dones[t] is done_T-1, values[t] is V(s_T-1).
                # We need V(s_T) for delta_T-1.
                next_non_terminal = 1.0 - dones[t] # if dones[t] is 1, next_non_terminal is 0
                if next_value_tensor is not None:
                    next_val = next_value_tensor
                else: # If no next_value_tensor, assume V(s_T) is 0 if done, or critic(s_T) if not.
                      # This case is tricky if s_T is not available.
                      # Let's assume buffer provides V(s0)...V(s_N-1) and one V(s_N) as next_value
                    next_val = torch.zeros_like(rewards[t]) # Fallback if not provided and last step is terminal
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t+1] # V(s_t+1)

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values # Q-values approx by GAE + V(s)
        return advantages, returns

    def update(self, trajectory_buffer, last_state_value_tensor=None):
        """
        Updates the policy and value function using collected trajectories.
        last_state_value_tensor: V(s_N) where s_N is the state after the last action in the buffer.
                                 Used for GAE calculation if the trajectory didn't end with 'done'.
                                 If the last state in buffer was terminal, this should be zero.
        """
        states, actions, old_log_probs, rewards, dones, values_from_rollout = trajectory_buffer.get_trajectories()
        
        # Compute GAE and returns (targets for value function)
        # If the trajectory ended because it was 'done', last_state_value_tensor should be 0.
        # If it ended due to reaching max trajectory length, last_state_value_tensor is V(s_N).
        advantages, returns = self.compute_gae(rewards, dones, values_from_rollout, last_state_value_tensor)
        
        # Normalize advantages (optional but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update loop
        for _ in range(self.ppo_epochs):
            # Recalculate log_probs, values, and entropy for the current policy
            current_log_probs, current_values, entropy = self.evaluate_trajectory_batch(states, actions)
            
            # Ratio of probabilities
            ratios = torch.exp(current_log_probs - old_log_probs.detach())
            
            # Actor loss (Clipped Surrogate Objective)
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * entropy.mean()
            
            # Critic loss (MSE)
            # current_values are V(s_t) from current critic, returns are GAE_t + V_old(s_t)
            # Or, returns can be just the calculated returns from rewards (Bellman targets)
            # Standard PPO uses returns = advantages + values_from_rollout (where values_from_rollout are V_target)
            critic_loss = F.mse_loss(current_values, returns.detach())
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def save_models(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Saved Actor model to {actor_path}")
        print(f"Saved Critic model to {critic_path}")

    def load_models(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        print(f"Loaded Actor model from {actor_path}")
        print(f"Loaded Critic model from {critic_path}")
        self.actor.eval() # Set to evaluation mode
        self.critic.eval()

