import os
import sys
import yaml
import numpy as np
import torch
import wandb
import argparse
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environments.soft_robot_env import SoftRobotEnv
from src.agents.safe_ppo_agent import SafePPOAgent
from src.safety.safety_monitor import SafetyWrapper
from src.utils.logger import setup_logger
from src.utils.visualization import TrainingVisualizer


class SafeRLTrainer:
    """
    Main training class for safe reinforcement learning with soft robots
    """
    
    def __init__(self, config_path: str, experiment_name: str = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup experiment name
        self.experiment_name = experiment_name or f"safe_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.logger = setup_logger('SafeRLTrainer', self.log_dir)
        
        # Initialize environment
        self.env = self.create_environment()
        
        # Initialize agent
        self.agent = SafePPOAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            config=self.config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Training parameters
        self.total_timesteps = self.config['training']['total_timesteps']
        self.eval_freq = self.config['training']['eval_freq']
        self.save_freq = self.config['training']['save_freq']
        self.n_eval_episodes = self.config['training']['n_eval_episodes']
        self.log_interval = self.config['training']['log_interval']
        
        # Training state
        self.timestep = 0
        self.episode = 0
        self.best_reward = float('-inf')
        self.best_safety_score = 0.0
        
        # Statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_safety_violations = []
        
        # Setup visualization
        self.visualizer = TrainingVisualizer(self.results_dir)
        
        # Initialize Weights & Biases
        if self.config['logging']['use_wandb']:
            self.init_wandb()
    
    def setup_directories(self):
        """Create necessary directories for the experiment"""
        base_dir = f"experiments/results/{self.experiment_name}"
        
        self.experiment_dir = base_dir
        self.log_dir = f"{base_dir}/logs"
        self.models_dir = f"{base_dir}/models"
        self.results_dir = f"{base_dir}/results"
        self.videos_dir = f"{base_dir}/videos"
        
        for directory in [self.log_dir, self.models_dir, self.results_dir, self.videos_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config['logging']['project_name'],
            name=self.experiment_name,
            config=self.config,
            sync_tensorboard=True
        )
    
    def create_environment(self):
        """Create and configure the training environment"""
        # Create base environment
        env = SoftRobotEnv(config=self.config)
        
        # Wrap with safety monitoring
        env = SafetyWrapper(env, self.config)
        
        return env
    
    def collect_rollout(self, max_steps: int = 2048) -> Dict[str, List]:
        """Collect a rollout of experience"""
        rollout_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'costs': [],
            'cost_values': [],
            'safety_violations': []
        }
        
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_violations = 0
        
        for step in range(max_steps):
            # Select action
            action, action_info = self.agent.select_action(state)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            rollout_data['states'].append(state)
            rollout_data['actions'].append(action)
            rollout_data['rewards'].append(reward)
            rollout_data['dones'].append(done)
            rollout_data['log_probs'].append(action_info['log_prob'])
            rollout_data['values'].append(action_info['value'])
            rollout_data['costs'].append(info.get('total_violations', 0))
            rollout_data['cost_values'].append(action_info['cost'])
            
            # Track safety violations
            safety_violations = info.get('violations', {})
            violation_count = len(safety_violations)
            rollout_data['safety_violations'].append(violation_count)
            episode_violations += violation_count
            
            episode_reward += reward
            episode_length += 1
            self.timestep += 1
            
            state = next_state
            
            if done:
                # Log episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_safety_violations.append(episode_violations)
                self.episode += 1
                
                # Reset environment
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_violations = 0
                
                # Log periodically
                if self.episode % self.log_interval == 0:
                    self.log_training_progress()
            
            if self.timestep >= self.total_timesteps:
                break
        
        return rollout_data
    
    def evaluate_agent(self) -> Dict[str, float]:
        """Evaluate the current agent"""
        eval_rewards = []
        eval_lengths = []
        eval_violations = []
        eval_success_rate = 0
        
        for episode in range(self.n_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_violation_count = 0
            
            done = False
            while not done and episode_length < self.config['environment']['max_episode_steps']:
                action, _ = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Count violations
                violations = info.get('violations', {})
                episode_violation_count += len(violations)
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_violations.append(episode_violation_count)
            
            # Check if episode was successful (task completed without emergency stop)
            if episode_reward > -50 and not info.get('emergency_stop', False):
                eval_success_rate += 1
        
        eval_success_rate /= self.n_eval_episodes
        
        eval_stats = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_length_mean': np.mean(eval_lengths),
            'eval_violations_mean': np.mean(eval_violations),
            'eval_success_rate': eval_success_rate
        }
        
        # Calculate safety score
        safety_score = max(0, 1.0 - np.mean(eval_violations) / 10.0)
        eval_stats['eval_safety_score'] = safety_score
        
        return eval_stats
    
    def log_training_progress(self):
        """Log training progress and statistics"""
        if not self.episode_rewards:
            return
        
        # Calculate recent statistics
        recent_window = min(100, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_window:]
        recent_violations = self.episode_safety_violations[-recent_window:]
        
        # Training statistics
        train_stats = {
            'train/episode': self.episode,
            'train/timestep': self.timestep,
            'train/reward_mean': np.mean(recent_rewards),
            'train/reward_std': np.std(recent_rewards),
            'train/violations_mean': np.mean(recent_violations),
            'train/safety_score': max(0, 1.0 - np.mean(recent_violations) / 10.0)
        }
        
        # Agent statistics
        agent_stats = self.agent.get_training_stats()
        agent_stats = {f'agent/{k}': v for k, v in agent_stats.items()}
        
        # Combine all statistics
        all_stats = {**train_stats, **agent_stats}
        
        # Log to console
        self.logger.info(
            f"Episode {self.episode}, Timestep {self.timestep}: "
            f"Reward={train_stats['train/reward_mean']:.2f}, "
            f"Safety Score={train_stats['train/safety_score']:.3f}, "
            f"Violations={train_stats['train/violations_mean']:.1f}"
        )
        
        # Log to wandb
        if self.config['logging']['use_wandb']:
            wandb.log(all_stats, step=self.timestep)
    
    def save_model(self, suffix: str = ""):
        """Save the current model"""
        if suffix:
            filename = f"agent_{suffix}.pth"
        else:
            filename = f"agent_timestep_{self.timestep}.pth"
        
        filepath = os.path.join(self.models_dir, filename)
        self.agent.save(filepath)
        
        # Also save config
        config_filepath = os.path.join(self.models_dir, f"config_{suffix}.yaml" if suffix else f"config_timestep_{self.timestep}.yaml")
        with open(config_filepath, 'w') as f:
            yaml.dump(self.config, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.total_timesteps} timesteps")
        self.logger.info(f"Environment: {self.config['environment']['robot_type']} robot")
        self.logger.info(f"Algorithm: Safe PPO with CPO constraints")
        
        try:
            while self.timestep < self.total_timesteps:
                # Collect rollout data
                rollout_data = self.collect_rollout()
                
                # Update agent
                if len(rollout_data['states']) > 0:
                    update_stats = self.agent.update(rollout_data)
                    
                    # Log update statistics
                    if self.config['logging']['use_wandb']:
                        update_log = {f'update/{k}': v for k, v in update_stats.items()}
                        wandb.log(update_log, step=self.timestep)
                
                # Evaluation
                if self.timestep % self.eval_freq == 0 or self.timestep >= self.total_timesteps:
                    eval_stats = self.evaluate_agent()
                    
                    # Log evaluation results
                    self.logger.info(
                        f"Evaluation at timestep {self.timestep}: "
                        f"Reward={eval_stats['eval_reward_mean']:.2f}±{eval_stats['eval_reward_std']:.2f}, "
                        f"Success Rate={eval_stats['eval_success_rate']:.2f}, "
                        f"Safety Score={eval_stats['eval_safety_score']:.3f}"
                    )
                    
                    if self.config['logging']['use_wandb']:
                        wandb.log(eval_stats, step=self.timestep)
                    
                    # Save best model
                    if eval_stats['eval_reward_mean'] > self.best_reward:
                        self.best_reward = eval_stats['eval_reward_mean']
                        self.save_model("best_reward")
                    
                    if eval_stats['eval_safety_score'] > self.best_safety_score:
                        self.best_safety_score = eval_stats['eval_safety_score']
                        self.save_model("best_safety")
                
                # Periodic model saving
                if self.timestep % self.save_freq == 0:
                    self.save_model()
                
                # Generate visualizations
                if self.timestep % (self.eval_freq * 5) == 0:
                    self.create_visualizations()
            
            # Final save
            self.save_model("final")
            self.create_visualizations()
            
            self.logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_model("interrupted")
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Cleanup
            self.env.close()
            if self.config['logging']['use_wandb']:
                wandb.finish()
    
    def create_visualizations(self):
        """Create and save training visualizations"""
        try:
            # Plot training curves
            self.visualizer.plot_training_curves(
                self.episode_rewards,
                self.episode_safety_violations,
                save_path=f"{self.results_dir}/training_curves.png"
            )
            
            # Plot safety analysis
            if hasattr(self.env, 'safety_monitor'):
                safety_metrics = self.env.safety_monitor.get_safety_metrics()
                self.visualizer.plot_safety_analysis(
                    safety_metrics,
                    save_path=f"{self.results_dir}/safety_analysis.png"
                )
            
            # Save training statistics
            stats_data = {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'episode_safety_violations': self.episode_safety_violations,
                'timesteps': self.timestep,
                'episodes': self.episode
            }
            
            np.save(f"{self.results_dir}/training_stats.npy", stats_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to create visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train Safe RL Agent for Soft Robots')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create trainer and start training
    trainer = SafeRLTrainer(
        config_path=args.config,
        experiment_name=args.name
    )
    
    trainer.train()


if __name__ == "__main__":
    main()