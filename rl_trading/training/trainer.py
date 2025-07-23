import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Callable
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..environments.trading_env import TradingEnv
from ..agents.ppo_agent import PPOAgent, PPOConfig
from ...core.metrics import TRAINING_METRICS

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000
    log_freq: int = 1_000
    n_eval_episodes: int = 5
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.01
    model_save_path: str = "models/rl_trading"
    tensorboard_log: Optional[str] = None

class RLTrainer:
    """Trainer for RL trading agents"""
    
    def __init__(
        self,
        env: TradingEnv,
        eval_env: TradingEnv,
        agent: PPOAgent,
        config: TrainingConfig = None
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.config = config or TrainingConfig()
        
        # Create save
        self.config = config or TrainingConfig()
        
        # Create save paths
        self.model_dir = Path(self.config.model_save_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.total_timesteps_trained = 0
        self.best_eval_return = -float('inf')
        self.patience_counter = 0
        
        # For logging (if tensorboard is enabled, though current scope doesn't include full tensorboard setup)
        # self.writer = SummaryWriter(self.config.tensorboard_log) if self.config.tensorboard_log else None
        
        logger.info(f"RLTrainer initialized. Model save path: {self.model_dir}")

    def train(self) -> None:
        """
        Main training loop for the RL agent.
        Collects experiences, updates the agent, and performs periodic evaluations.
        """
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_len = 0
        episode_count = 0
        
        start_time = time.time()
        
        logger.info(f"Starting training for {self.config.total_timesteps} timesteps...")

        with tqdm(total=self.config.total_timesteps, desc="Training RL Agent") as pbar:
            while self.total_timesteps_trained < self.config.total_timesteps:
                # Collect experience rollout
                rollout_start_time = time.time()
                current_obs = obs
                current_episode_reward = 0
                current_episode_len = 0
                
                # Collect `buffer_size` timesteps or until episode ends
                for _ in range(self.agent.config.buffer_size):
                    action, log_prob, value = self.agent.get_action(current_obs)
                    
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    self.agent.store_transition(current_obs, action, reward, value, log_prob, done)
                    
                    current_obs = next_obs
                    current_episode_reward += reward
                    current_episode_len += 1
                    self.total_timesteps_trained += 1
                    
                    pbar.update(1) # Update progress bar
                    
                    if done:
                        # Log episode metrics
                        logger.debug(f"Episode {episode_count} finished. Reward: {current_episode_reward:.2f}, Length: {current_episode_len}")
                        
                        # Reset environment for next episode
                        current_obs, _ = self.env.reset()
                        episode_count += 1
                        current_episode_reward = 0
                        current_episode_len = 0
                
                # Update the agent using the collected rollout
                last_obs_for_value = current_obs
                
                # If the episode just terminated/truncated, the last_value should be 0.
                # Otherwise, it's the value estimate for the next state.
                if terminated or truncated:
                    last_value = 0.0
                else:
                    _, _, last_value_estimate = self.agent.get_action(last_obs_for_value)
                    last_value = last_value_estimate
                
                # Pass the effective last_value to the update function
                agent_metrics = self.agent.update(last_obs=last_obs_for_value if not (terminated or truncated) else current_obs)
                
                # Log training metrics
                if self.total_timesteps_trained % self.config.log_freq < self.agent.config.buffer_size:
                    avg_metrics = self.agent.get_training_metrics()
                    logger.info(f"Timestep {self.total_timesteps_trained}/{self.config.total_timesteps} - "
                                f"Policy Loss: {avg_metrics['policy_loss']:.4f}, "
                                f"Value Loss: {avg_metrics['value_loss']:.4f}, "
                                f"Entropy Loss: {avg_metrics['entropy_loss']:.4f}, "
                                f"Total Loss: {avg_metrics['total_loss']:.4f}, "
                                f"Explained Var: {avg_metrics['explained_variance']:.4f}")
                    # if self.writer:
                    #     for k, v in avg_metrics.items():
                    #         self.writer.add_scalar(f"train/{k}", v, self.total_timesteps_trained)
                
                # Evaluate agent
                if self.total_timesteps_trained % self.config.eval_freq < self.agent.config.buffer_size:
                    eval_avg_return, eval_avg_length, eval_info = self._evaluate_agent()
                    logger.info(f"Evaluation at Timestep {self.total_timesteps_trained}: "
                                f"Avg Return: {eval_avg_return:.2f}, Avg Length: {eval_avg_length:.0f}")
                    # if self.writer:
                    #     self.writer.add_scalar("eval/avg_return", eval_avg_return, self.total_timesteps_trained)
                    #     self.writer.add_scalar("eval/avg_episode_length", eval_avg_length, self.total_timesteps_trained)

                    # Check for improvement and save best model
                    if eval_avg_return > self.best_eval_return + self.config.early_stopping_threshold:
                        self.best_eval_return = eval_avg_return
                        self.patience_counter = 0
                        self._save_model(f"best_model_timestep_{self.total_timesteps_trained}.pth")
                        logger.info(f"New best model saved with average return: {self.best_eval_return:.2f}")
                    else:
                        self.patience_counter += 1
                        logger.info(f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
                        if self.patience_counter >= self.config.early_stopping_patience:
                            logger.info("Early stopping triggered: No significant improvement.")
                            break
                
                # Save model periodically
                if self.total_timesteps_trained % self.config.save_freq < self.agent.config.buffer_size:
                    self._save_model(f"checkpoint_timestep_{self.total_timesteps_trained}.pth")
                    
        end_time = time.time()
        total_training_duration = (end_time - start_time) / 3600
        logger.info(f"Training finished after {self.total_timesteps_trained} timesteps "
                    f"in {total_training_duration:.2f} hours.")
        self._save_model("final_model.pth")
        # if self.writer:
        #     self.writer.close()

    def _evaluate_agent(self) -> Tuple[float, float, Dict]:
        """Evaluates the agent's performance over a few episodes."""
        eval_returns = []
        eval_lengths = []
        eval_info_list = []
        
        for i in range(self.config.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.get_action(obs, deterministic=True) # Use deterministic policy for evaluation
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
            eval_info_list.append(info) # Store last info of episode
            
        avg_return = np.mean(eval_returns)
        avg_length = np.mean(eval_lengths)
        
        return avg_return, avg_length, eval_info_list
    
    def _save_model(self, filename: str) -> None:
        """Saves the agent's model to a file."""
        filepath = self.model_dir / filename
        self.agent.save(str(filepath))
        logger.debug(f"Model saved to {filepath}")
                  
