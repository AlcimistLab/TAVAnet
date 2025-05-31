import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Reusing PolicyNetwork from SAC, ensuring it returns action and log_prob
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x) # No tanh here, PPO often uses raw action from Gaussian then clips
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, return_log_prob=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample() # Raw action
        
        # Optional: Apply tanh squashing if environment action space is bounded like [-1, 1]
        # This matches SAC more closely, but PPO can also work without it
        # and clip actions later. For consistency with SAC example, let's keep it for now.
        # However, in "pure" PPO, often this squashing is omitted and actions are clipped,
        # or a Beta distribution is used for bounded actions.
        squashed_action = torch.tanh(action)

        if not return_log_prob:
            return squashed_action

        # Calculate log_prob of squashed_action
        # log_prob = normal.log_prob(action) - torch.log(1 - squashed_action.pow(2) + 1e-6)
        # log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Correct log_prob for Gaussian + tanh
        log_prob = normal.log_prob(action)
        # The log-probability of the action under the Tanh-transformed distribution
        # log π(a|s) = log ρ(u|s) - Σ_{i=1}^{D} log(1 - tanh(u_i)^2)
        # where u is the pre-tanh action, a is tanh(u)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return squashed_action, log_prob

    def evaluate_actions(self, state, action_squashed):
        # Inverse tanh to get pre-squashed action
        # action_squashed = torch.clamp(action_squashed, -1 + 1e-6, 1 - 1e-6) # for numerical stability
        # pre_action = 0.5 * (torch.log(1 + action_squashed + 1e-6) - torch.log(1 - action_squashed + 1e-6))
        
        # A more stable way to get pre_action for PPO when actions are already squashed:
        # If actions are produced by `sample` which includes tanh, and we stored those squashed actions.
        # We need to evaluate their log_prob under the current policy.
        # The original 'action' (pre-tanh) is needed to compute its log_prob under the Gaussian.
        # This is tricky if 'action_squashed' is the only thing available.
        # PPO typically stores log_probs from the *behavioral* policy.
        # When evaluating under the *current* policy, we re-compute mean, std from state.
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # To evaluate the log_prob of `action_squashed`, we need the `pre_action` that led to it.
        # This is `atanh(action_squashed)`.
        # Clamping is important for atanh.
        action_squashed_clamped = torch.clamp(action_squashed, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_action = torch.atanh(action_squashed_clamped)

        log_prob = normal.log_prob(pre_action)
        log_prob -= torch.log(1 - action_squashed_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        entropy = normal.entropy().sum(dim=-1, keepdim=True) # Entropy of the Gaussian part
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs_old = []
        self.values_old = [] # V(s_t) from the critic at the time of collection
        self.advantages = []
        self.returns = [] # Target for value function V_target = A_t + V(s_t)

    def add(self, state, action, reward, next_state, done, log_prob_old, value_old):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs_old.append(log_prob_old)
        self.values_old.append(value_old)

    def _to_torch_tensors(self):
        # Convert lists to numpy arrays first for efficiency if they are not already
        # This is safer if elements are single numbers/small lists before converting to tensor
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1)
        log_probs_old = torch.FloatTensor(np.array(self.log_probs_old))
        values_old = torch.FloatTensor(np.array(self.values_old)) # Ensure this is already a tensor or array of scalars
        
        # These will be computed later and then converted
        advantages = torch.FloatTensor(np.array(self.advantages))
        returns = torch.FloatTensor(np.array(self.returns))

        return states, actions, rewards, next_states, dones, log_probs_old, values_old, advantages, returns

    def compute_advantages_and_returns(self, last_value, gamma, gae_lambda):
        """
        Computes GAE and returns for the collected trajectory.
        last_value: V(s_T) estimate from the critic for the last next_state in the rollout.
        """
        num_steps = len(self.rewards)
        self.advantages = [0.0] * num_steps
        self.returns = [0.0] * num_steps
        
        gae = 0
        # Ensure values_old contains V(s_t) for t=0..T-1
        # last_value is V(s_T)
        
        # Iterate backwards from T-1 to 0
        for t in reversed(range(num_steps)):
            # If done, next_value is 0, unless it's a truncation not a terminal state.
            # For PPO, if it's a terminal done, V(s_{t+1}) is 0.
            # If it's a truncation (episode didn't end but rollout limit reached),
            # V(s_{t+1}) is bootstrapped using last_value if t == num_steps -1,
            # or self.values_old[t+1] if not done[t+1]
            
            if self.dones[t]: # if s_{t+1} is a terminal state
                next_value = 0.0
            elif t == num_steps - 1: # if s_{t+1} is the state after the last collected state
                next_value = last_value.item() if isinstance(last_value, torch.Tensor) else last_value
            else: # if s_{t+1} is an intermediate state in the buffer
                next_value = self.values_old[t+1].item() if isinstance(self.values_old[t+1], torch.Tensor) else self.values_old[t+1]


            # Delta_t = R_t + gamma * V(S_{t+1}) * (1-done_t) - V(S_t)
            # Here, V(S_{t+1}) is represented by `next_value`
            # V(S_t) is self.values_old[t]
            # (1-done_t) is implicitly handled by next_value = 0 if self.dones[t] is true
            # For delta:
            # if self.dones[t] is true, then reward_t + gamma * 0 - V(s_t)
            # else reward_t + gamma * V(s_{t+1}) - V(s_t)
            
            # Ensure rewards, values_old are scalars here or correctly indexed
            current_reward = self.rewards[t]
            current_value_old = self.values_old[t].item() if isinstance(self.values_old[t], torch.Tensor) else self.values_old[t]

            delta = current_reward + gamma * next_value * (1 - self.dones[t]) - current_value_old
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + current_value_old # R_t = A_t + V(s_t)
            
        # Normalize advantages (optional but often helps)
        # self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)
        # This should be done after converting to tensor if advantages are stored as list of scalars

    def get_batch_generator(self, batch_size):
        states, actions, _, _, _, log_probs_old, _, advantages, returns = self._to_torch_tensors()
        
        # Normalize advantages at the batch level before creating mini-batches
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(states)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs_old[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
            )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs_old.clear()
        self.values_old.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self):
        return len(self.states)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99,
                 clip_epsilon=0.2, ppo_epochs=10, batch_size=64, gae_lambda=0.95,
                 ent_coef=0.01, vf_coef=0.5, hidden_dim=256, log_std_min=-20, log_std_max=2):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, log_std_min, log_std_max).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size # This is mini-batch size for updates
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        self.rollout_buffer = RolloutBuffer()

    def select_action_for_rollout(self, state):
        """ Selects action, gets log_prob and value for rollout collection. """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy_net.sample(state_tensor)
            value = self.value_net(state_tensor)
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def store_transition(self, state, action, reward, next_state, done, log_prob_old, value_old):
        self.rollout_buffer.add(state, action, reward, next_state, done, log_prob_old, value_old)

    def train(self, last_value_estimate_for_rollout):
        """
        last_value_estimate_for_rollout: V(s_T) for the state after the last collected step.
                                         Should be 0 if the last step was terminal.
        """
        # 1. Compute advantages and returns
        self.rollout_buffer.compute_advantages_and_returns(
            last_value_estimate_for_rollout, self.gamma, self.gae_lambda
        )
        
        # 2. PPO Update (K epochs)
        for _ in range(self.ppo_epochs):
            batch_generator = self.rollout_buffer.get_batch_generator(self.batch_size)
            for states_b, actions_b, log_probs_old_b, advantages_b, returns_b in batch_generator:
                
                states_b = states_b.to(self.device)
                actions_b = actions_b.to(self.device)
                log_probs_old_b = log_probs_old_b.to(self.device)
                advantages_b = advantages_b.to(self.device)
                returns_b = returns_b.to(self.device)

                # Evaluate current policy on collected actions
                # For PPO, actions_b are the actions taken by the old policy
                # We need their log_prob under the current policy pi_theta(a_t | s_t)
                # And the entropy of the current policy's distribution
                current_log_probs, entropy = self.policy_net.evaluate_actions(states_b, actions_b)

                # ---- Policy (Actor) Loss ----
                # Ratio r_t(theta) = exp(log_prob_current - log_prob_old)
                ratios = torch.exp(current_log_probs - log_probs_old_b)
                
                surr1 = ratios * advantages_b
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus (maximization, so subtract from loss)
                entropy_bonus = entropy.mean()
                actor_loss = policy_loss - self.ent_coef * entropy_bonus
                
                # ---- Value (Critic) Loss ----
                # V_phi(s_t) predicted by current value network
                current_values = self.value_net(states_b)
                # MSE against returns_b (which are GAE + V_old(s_t))
                value_loss = F.mse_loss(current_values, returns_b)
                
                # ---- Update Actor ----
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5) # Optional grad clipping
                self.actor_optimizer.step()
                
                # ---- Update Critic ----
                self.critic_optimizer.zero_grad()
                # Total loss can also be combined: total_loss = actor_loss + self.vf_coef * value_loss
                # But often value_loss is calculated and optimized separately or its gradient detached for actor
                critic_loss = self.vf_coef * value_loss
                critic_loss.backward()
                # nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5) # Optional grad clipping
                self.critic_optimizer.step()

        # 3. Clear rollout buffer for next set of rollouts
        self.rollout_buffer.clear()

    def save_models(self, filepath_actor, filepath_critic):
        torch.save(self.policy_net.state_dict(), filepath_actor)
        torch.save(self.value_net.state_dict(), filepath_critic)
        print(f"Actor model saved to {filepath_actor}")
        print(f"Critic model saved to {filepath_critic}")

    def load_models(self, filepath_actor, filepath_critic):
        self.policy_net.load_state_dict(torch.load(filepath_actor, map_location=self.device))
        self.value_net.load_state_dict(torch.load(filepath_critic, map_location=self.device))
        print(f"Actor model loaded from {filepath_actor}")
        print(f"Critic model loaded from {filepath_critic}")
        self.policy_net.eval() # Set to evaluation mode if only for inference
        self.value_net.eval()


# Example usage (conceptual)
if __name__ == '__main__':
    # These are placeholders, replace with your environment's specifics
    STATE_DIM = 4 
    ACTION_DIM = 2
    
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    # --- Rollout Phase ---
    # Simulating a rollout of, say, 2048 steps
    # In a real scenario, this loop interacts with an environment
    num_rollout_steps = 2048 
    
    # Dummy current state for the start of the rollout
    current_state = np.random.rand(STATE_DIM) 
    
    for step in range(num_rollout_steps):
        # 1. Select action, get log_prob and value
        action, log_prob, value_old = agent.select_action_for_rollout(current_state)
        
        # 2. Simulate environment step (replace with actual env.step(action))
        # next_state, reward, done, _ = env.step(action)
        next_state = np.random.rand(STATE_DIM) # Dummy
        reward = np.random.rand()             # Dummy
        done = False                          # Dummy
        if step == num_rollout_steps - 1:     # Simulate end of rollout
            # done = True # or False if episode continues beyond rollout
            pass

        # 3. Store transition
        agent.store_transition(current_state, action, reward, next_state, done, log_prob, value_old)
        
        current_state = next_state
        if done and step < num_rollout_steps -1 : # If episode ends before rollout is full
            # current_state = env.reset() # Reset env and get new state
            current_state = np.random.rand(STATE_DIM) # Dummy reset
            print(f"Episode ended at step {step}, environment reset.")
            # If you want to collect exactly num_rollout_steps, you might continue filling
            # or handle partial rollouts. PPO typically collects a fixed number of steps.

    # --- Training Phase ---
    # After collecting num_rollout_steps, calculate the value of the *final next_state*
    # This is V(s_T) used for GAE calculation if the last step was not terminal
    last_next_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        last_value_estimate = agent.value_net(last_next_state_tensor).cpu().item()
    
    # If the very last transition resulted in 'done=True', then last_value_estimate should be 0
    # The GAE calculation within RolloutBuffer handles 'done' for intermediate steps.
    # This `last_value_estimate` is specifically for the state *after* the final collected transition s_{T-1} -> r_T, s_T
    # If `dones[T-1]` (the done signal for s_T) is True, then GAE will use 0. Otherwise, it uses this estimate.
    if agent.rollout_buffer.dones[-1]: # if the last collected transition's next_state was terminal
        last_value_estimate_for_gae = 0.0
    else:
        last_value_estimate_for_gae = last_value_estimate

    print(f"Collected {len(agent.rollout_buffer)} transitions. Starting training...")
    agent.train(last_value_estimate_for_gae)
    print("Training finished.")

    # Example: save models
    # agent.save_models("ppo_actor.pth", "ppo_critic.pth")
    # agent.load_models("ppo_actor.pth", "ppo_critic.pth")
    # new_action, _, _ = agent.select_action_for_rollout(np.random.rand(STATE_DIM))
    # print("Action from loaded model:", new_action) 