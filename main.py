"""
Main entry point for Safe Reinforcement Learning with Soft Robots

This script provides a unified interface for training, evaluation, and testing
of safe RL agents on soft robot environments.

Usage:
    python main.py train --config experiments/configs/default_config.yaml
    python main.py evaluate --model data/models/best_model.pth
    python main.py test --model data/models/best_model.pth --episodes 10
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environments.soft_robot_env import SoftRobotEnv
from agents.safe_ppo_agent import SafePPOAgent
from safety.safety_monitor import SafetyWrapper
from utils.logger import setup_experiment_logging
from utils.visualization import TrainingVisualizer
from scripts.training.train_safe_agent import SafeRLTrainer


class SafeRLRunner:
    """Main runner class for safe RL experiments"""
    
    def __init__(self, args):
        self.args = args
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration if provided
        if hasattr(args, 'config') and args.config:
            with open(args.config, 'r') as f:
                self.config = yaml.safe_load(f)
    
    def train(self):
        """Train a safe RL agent"""
        if not self.config:
            raise ValueError("Configuration file is required for training")
        
        print("="*50)
        print("SAFE REINFORCEMENT LEARNING TRAINING")
        print("="*50)
        print(f"Robot Type: {self.config['environment']['robot_type']}")
        print(f"Algorithm: Safe PPO with CPO constraints")
        print(f"Total Timesteps: {self.config['training']['total_timesteps']:,}")
        print(f"Device: {self.device}")
        print("="*50)
        
        # Create trainer
        trainer = SafeRLTrainer(
            config_path=self.args.config,
            experiment_name=self.args.name
        )
        
        # Start training
        trainer.train()
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {trainer.experiment_dir}")
    
    def evaluate(self):
        """Evaluate a trained agent"""
        if not os.path.exists(self.args.model):
            raise FileError(f"Model file not found: {self.args.model}")
        
        print("="*50)
        print("AGENT EVALUATION")
        print("="*50)
        
        # Load model and config
        checkpoint = torch.load(self.args.model, map_location=self.device)
        config = checkpoint.get('config', self._get_default_config())
        
        # Create environment and agent
        env = SoftRobotEnv(config=config)
        env = SafetyWrapper(env, config)
        
        agent = SafePPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=config,
            device=self.device
        )
        agent.load(self.args.model)
        
        # Run evaluation
        n_episodes = getattr(self.args, 'episodes', 10)
        results = self._run_evaluation(env, agent, n_episodes)
        
        # Print results
        self._print_evaluation_results(results)
        
        env.close()
    
    def test(self):
        """Test agent with specific scenarios"""
        if not os.path.exists(self.args.model):
            raise FileError(f"Model file not found: {self.args.model}")
        
        print("="*50)
        print("AGENT TESTING")
        print("="*50)
        
        # Load model and config
        checkpoint = torch.load(self.args.model, map_location=self.device)
        config = checkpoint.get('config', self._get_default_config())
        
        # Create environment and agent
        env = SoftRobotEnv(config=config, render_mode='human')
        env = SafetyWrapper(env, config)
        
        agent = SafePPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=config,
            device=self.device
        )
        agent.load(self.args.model)
        
        # Run test episodes
        n_episodes = getattr(self.args, 'episodes', 5)
        
        for episode in range(n_episodes):
            print(f"\nRunning test episode {episode + 1}/{n_episodes}")
            
            state = env.reset()
            total_reward = 0
            steps = 0
            violations = 0
            
            done = False
            while not done and steps < config['environment']['max_episode_steps']:
                # Get action with uncertainty
                action, info = agent.select_action(state, deterministic=True)
                
                # Step environment
                next_state, reward, done, step_info = env.step(action)
                
                total_reward += reward
                steps += 1
                violations += len(step_info.get('violations', {}))
                
                # Render
                env.render()
                
                state = next_state
                
                # Print step info
                if steps % 50 == 0:
                    print(f"  Step {steps}: Reward={reward:.3f}, "
                          f"Violations={len(step_info.get('violations', {}))}, "
                          f"Value={info.get('value', 0):.3f}")
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Safety Violations: {violations}")
            print(f"  Emergency Stop: {step_info.get('emergency_stop', False)}")
        
        env.close()
    
    def benchmark(self):
        """Run comprehensive benchmarking"""
        if not os.path.exists(self.args.model):
            raise FileError(f"Model file not found: {self.args.model}")
        
        print("="*50)
        print("COMPREHENSIVE BENCHMARKING")
        print("="*50)
        
        # Load model and config
        checkpoint = torch.load(self.args.model, map_location=self.device)
        config = checkpoint.get('config', self._get_default_config())
        
        # Test different robot configurations
        robot_types = ['tentacle', 'gripper', 'locomotion']
        results = {}
        
        for robot_type in robot_types:
            print(f"\nTesting on {robot_type} robot...")
            
            # Modify config for this robot type
            test_config = config.copy()
            test_config['environment']['robot_type'] = robot_type
            
            # Create environment and agent
            env = SoftRobotEnv(config=test_config)
            env = SafetyWrapper(env, test_config)
            
            agent = SafePPOAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                config=test_config,
                device=self.device
            )
            agent.load(self.args.model)
            
            # Run evaluation
            robot_results = self._run_evaluation(env, agent, 20)
            results[robot_type] = robot_results
            
            env.close()
        
        # Print benchmark results
        self._print_benchmark_results(results)
    
    def _run_evaluation(self, env, agent, n_episodes):
        """Run evaluation episodes"""
        rewards = []
        lengths = []
        violations = []
        success_count = 0
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_violations = 0
            
            done = False
            while not done and episode_length < env.max_episode_steps:
                action, _ = agent.select_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_violations += len(info.get('violations', {}))
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            violations.append(episode_violations)
            
            # Check success (no emergency stop and reasonable reward)
            if episode_reward > -50 and not info.get('emergency_stop', False):
                success_count += 1
        
        return {
            'rewards': rewards,
            'lengths': lengths,
            'violations': violations,
            'success_rate': success_count / n_episodes,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_violations': np.mean(violations),
            'safety_score': max(0, 1.0 - np.mean(violations) / 10.0)
        }
    
    def _print_evaluation_results(self, results):
        """Print formatted evaluation results"""
        print(f"\nEVALUATION RESULTS:")
        print(f"  Episodes: {len(results['rewards'])}")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Success Rate: {results['success_rate']:.2f}")
        print(f"  Mean Violations: {results['mean_violations']:.1f}")
        print(f"  Safety Score: {results['safety_score']:.3f}")
        print(f"  Best Episode Reward: {max(results['rewards']):.2f}")
        print(f"  Worst Episode Reward: {min(results['rewards']):.2f}")
    
    def _print_benchmark_results(self, results):
        """Print formatted benchmark results"""
        print(f"\nBENCHMARK RESULTS:")
        print("-"*60)
        print(f"{'Robot Type':<15} {'Mean Reward':<12} {'Success Rate':<12} {'Safety Score':<12}")
        print("-"*60)
        
        for robot_type, result in results.items():
            print(f"{robot_type.title():<15} "
                  f"{result['mean_reward']:<12.2f} "
                  f"{result['success_rate']:<12.2f} "
                  f"{result['safety_score']:<12.3f}")
        
        print("-"*60)
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'environment': {
                'robot_type': 'tentacle',
                'action_dim': 8,
                'observation_dim': 24,
                'max_episode_steps': 1000,
                'control_frequency': 50
            },
            'robot': {
                'segments': 4,
                'segment_length': 0.1,
                'radius': 0.02,
                'mass_per_segment': 0.05,
                'stiffness': 1000.0,
                'damping': 10.0,
                'max_force': 10.0
            },
            'safety': {
                'max_deformation': 0.5,
                'collision_threshold': 0.01,
                'force_limit': 15.0,
                'velocity_limit': 2.0,
                'emergency_stop_threshold': 0.8
            },
            'algorithm': {
                'learning_rate': 3e-4,
                'clip_range': 0.2,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_epochs': 10,
                'batch_size': 64
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Safe RL for Soft Robots')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a safe RL agent')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to configuration file')
    train_parser.add_argument('--name', type=str, default=None,
                             help='Experiment name')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to model file')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test agent interactively')
    test_parser.add_argument('--model', type=str, required=True,
                            help='Path to model file')
    test_parser.add_argument('--episodes', type=int, default=5,
                            help='Number of test episodes')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmarks')
    bench_parser.add_argument('--model', type=str, required=True,
                             help='Path to model file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create runner and execute command
    runner = SafeRLRunner(args)
    
    if args.command == 'train':
        runner.train()
    elif args.command == 'evaluate':
        runner.evaluate()
    elif args.command == 'test':
        runner.test()
    elif args.command == 'benchmark':
        runner.benchmark()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()