import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from collections import deque
import pickle

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """Configuration for PPO agent"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 2048
    hidden_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, obs_shape: Tuple[int, ...], action_dims: List[int], config: PPOConfig):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dims = action_dims
        self.config = config
        
        # Calculate input size (flatten the observation)
        input_size = np.prod(obs_shape)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )
        
        # Actor heads (one for each action dimension)
        self.actor_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, action_dim) 
            for action_dim in action_dims
        ])
        
        # Critic head
        self.critic = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through the network"""
        # Flatten observation
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        
        # Extract features
        features = self.feature_extractor(obs_flat)
        
        # Get action logits for each action dimension
        action_logits = [head(features) for head in self.actor_heads]
        
        # Get value estimate
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action probabilities and value estimates"""
        action_logits, value = self.forward(obs)
        
        # Create categorical distributions for each action dimension
        action_dists = [Categorical(logits=logits) for logits in action_logits]
        
        if action is None:
            # Sample actions
            actions = torch.stack([dist.sample() for dist in action_dists], dim=1)
        else:
            actions = action
        
        # Calculate log probabilities
        log_probs = torch.stack([
            dist.log_prob(actions[:, i]) for i, dist in enumerate(action_dists)
        ], dim=1).sum(dim=1)
        
        # Calculate entropy
        entropy = torch.stack([dist.entropy() for dist in action_dists], dim=1).sum(dim=1)
        
        return actions, log_probs, entropy, value.squeeze(-1)

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        # Initialize buffers
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, obs, action, reward, value, log_prob, done):
        """Store a transition in the buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self, gamma: float, gae_lambda: float, last_value: float = 0.0):
        """Get all stored transitions with computed advantages"""
        assert self.size == self.buffer_size, "Buffer must be full before getting data"
        
        # Compute advantages using GAE
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.rewards)
        
        gae = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = advantages[step] + self.values[step]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'returns': returns,
            'advantages': advantages,
            'values': self.values
        }
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """Proximal Policy Optimization agent for trading"""
    
    def __init__(self, obs_shape: Tuple[int, ...], action_dims: List[int], config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.obs_shape = obs_shape
        self.action_dims = action_dims
        self.device = torch.device(self.config.device)
        
        # Initialize network
        self.network = ActorCriticNetwork(obs_shape, action_dims, self.config).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Initialize buffer
        action_shape = (len(action_dims),)
        self.buffer = PPOBuffer(self.config.buffer_size, obs_shape, action_shape)
        
        # Training metrics
        self.training_metrics = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
        
        logger.info(f"PPO Agent initialized with observation shape: {obs_shape}, "
                   f"action dimensions: {action_dims}, device: {self.device}")
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Get action from the current policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if deterministic:
                # Get deterministic action (argmax)
                action_logits, value = self.network(obs_tensor)
                actions = torch.stack([torch.argmax(logits, dim=-1) for logits in action_logits], dim=1)
                log_prob = torch.tensor(0.0)  # Not used in deterministic mode
            else:
                # Sample action
                actions, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            
            return actions.cpu().numpy()[0], log_prob.cpu().numpy(), value.cpu().numpy()[0]
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store a transition in the buffer"""
        self.buffer.store(obs, action, reward, value, log_prob, done)
    
    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """Update the policy using PPO"""
        # Get last value for GAE computation
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_obs_tensor)
            last_value = last_value.cpu().numpy()[0]
        
        # Get buffer data
        buffer_data = self.buffer.get(self.config.gamma, self.config.gae_lambda, last_value)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(buffer_data['observations']).to(self.device)
        actions_tensor = torch.LongTensor(buffer_data['actions']).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(buffer_data['log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(buffer_data['returns']).to(self.device)
        advantages_tensor = torch.FloatTensor(buffer_data['advantages']).to(self.device)
        old_values_tensor = torch.FloatTensor(buffer_data['values']).to(self.device)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = np.random.permutation(self.config.buffer_size)
            
            for start in range(0, self.config.buffer_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.config.clip_epsilon, self.config.clip_epsilon
                )
                value_loss1 = (new_values - batch_returns).pow(2)
                value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
        
        # Calculate explained variance
        explained_var = 1 - torch.var(returns_tensor - new_values.detach()) / torch.var(returns_tensor)
        
        # Store metrics
        num_updates = self.config.ppo_epochs * (self.config.buffer_size // self.config.batch_size)
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'total_loss': total_loss / num_updates,
            'explained_variance': explained_var.item()
        }
        
        for key, value in metrics.items():
            self.training_metrics[key].append(value)
        
        # Clear buffer
        self.buffer.clear()
        
        return metrics
    
    def save(self, filepath: str):
        """Save the agent"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'obs_shape': self.obs_shape,
            'action_dims': self.action_dims,
            'training_metrics': dict(self.training_metrics)
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Agent loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get recent training metrics"""
        return {
            key: np.mean(values) if values else 0.0 
            for key, values in self.training_metrics.items()
        }
