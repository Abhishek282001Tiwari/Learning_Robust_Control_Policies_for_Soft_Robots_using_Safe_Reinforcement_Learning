import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance


class PolicyNetwork(nn.Module):
    """
    Policy network with uncertainty quantification using dropout
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 dropout_rate: float = 0.1):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # Build policy network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layers
        layers.append(nn.Linear(input_dim, action_dim))
        self.policy_net = nn.Sequential(*layers)
        
        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        mean = self.policy_net(state)
        std = torch.exp(self.log_std.clamp(min=-20, max=2))
        return mean, std
    
    def get_action_and_log_prob(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        # Apply tanh squashing
        action_squashed = torch.tanh(action)
        
        # Adjust log probability for squashing
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6).sum(dim=-1)
        
        return action_squashed, log_prob
    
    def get_uncertainty(self, state, n_samples=10):
        """Estimate uncertainty using dropout sampling"""
        self.train()  # Enable dropout
        
        actions = []
        for _ in range(n_samples):
            mean, _ = self.forward(state)
            actions.append(mean.detach())
        
        actions = torch.stack(actions)
        mean_action = actions.mean(dim=0)
        uncertainty = actions.std(dim=0)
        
        return mean_action, uncertainty


class ValueNetwork(nn.Module):
    """Value function network"""
    
    def __init__(self, 
                 state_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 dropout_rate: float = 0.1):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.value_net = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.value_net(state).squeeze(-1)


class CostNetwork(nn.Module):
    """Cost function network for safety constraints"""
    
    def __init__(self, 
                 state_dim: int, 
                 hidden_dims: List[int] = [128, 128]):
        super(CostNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.cost_net = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.cost_net(state).squeeze(-1)


class SafePPOAgent:
    """
    Safe PPO agent with Constrained Policy Optimization (CPO) elements
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        
        # Algorithm hyperparameters
        self.lr = config['algorithm']['learning_rate']
        self.clip_range = config['algorithm']['clip_range']
        self.entropy_coef = config['algorithm']['entropy_coef']
        self.value_loss_coef = config['algorithm']['value_loss_coef']
        self.max_grad_norm = config['algorithm']['max_grad_norm']
        self.n_epochs = config['algorithm']['n_epochs']
        self.batch_size = config['algorithm']['batch_size']
        
        # Safety parameters
        self.cost_limit = config['safe_rl']['cost_limit']
        self.lagrange_lr = config['safe_rl']['lagrange_multiplier_lr']
        self.safety_budget = config['safe_rl']['safety_budget']
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_func = ValueNetwork(state_dim).to(self.device)
        self.cost_func = CostNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_func.parameters(), lr=self.lr)
        self.cost_optimizer = optim.Adam(self.cost_func.parameters(), lr=self.lr)
        
        # Lagrange multiplier for cost constraint
        self.lagrange_multiplier = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.lagrange_optimizer = optim.Adam([self.lagrange_multiplier], lr=self.lagrange_lr)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'cost_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'safety_violations': deque(maxlen=100),
            'lagrange_multiplier': deque(maxlen=100)
        }
        
        # Setup logging
        self.logger = logging.getLogger('SafePPOAgent')
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action with uncertainty estimation"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.get_action_and_log_prob(state_tensor, deterministic)
            value = self.value_func(state_tensor)
            cost = self.cost_func(state_tensor)
            
            # Get uncertainty estimate
            if not deterministic:
                _, uncertainty = self.policy.get_uncertainty(state_tensor)
                uncertainty = uncertainty.cpu().numpy().flatten()
            else:
                uncertainty = np.zeros(self.action_dim)
        
        action_np = action.cpu().numpy().flatten()
        
        info = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'cost': cost.item(),
            'uncertainty': uncertainty
        }
        
        return action_np, info
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step with safety constraints"""
        
        states = batch_data['states']
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        advantages = batch_data['advantages']
        returns = batch_data['returns']
        costs = batch_data['costs']
        cost_advantages = batch_data['cost_advantages']
        cost_returns = batch_data['cost_returns']
        
        # Get current policy outputs
        means, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Calculate importance sampling ratios
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        # Policy loss with clipping
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy regularization
        policy_loss -= self.entropy_coef * entropy
        
        # Safety constraint: add cost penalty
        cost_surr1 = ratios * cost_advantages
        cost_surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * cost_advantages
        cost_penalty = torch.max(cost_surr1, cost_surr2).mean()
        
        # Total policy loss with safety constraint
        total_policy_loss = policy_loss + self.lagrange_multiplier.detach() * cost_penalty
        
        # Value function loss
        current_values = self.value_func(states)
        value_loss = nn.MSELoss()(current_values, returns)
        
        # Cost function loss
        current_costs = self.cost_func(states)
        cost_loss = nn.MSELoss()(current_costs, cost_returns)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_func.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        # Update cost function
        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_func.parameters(), self.max_grad_norm)
        self.cost_optimizer.step()
        
        # Update Lagrange multiplier
        constraint_violation = cost_penalty.detach() - self.cost_limit
        lagrange_loss = -self.lagrange_multiplier * constraint_violation
        
        self.lagrange_optimizer.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optimizer.step()
        
        # Clamp Lagrange multiplier to be non-negative
        with torch.no_grad():
            self.lagrange_multiplier.clamp_(min=0.0)
        
        # Update statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'cost_loss': cost_loss.item(),
            'entropy': entropy.item(),
            'cost_penalty': cost_penalty.item(),
            'lagrange_multiplier': self.lagrange_multiplier.item(),
            'constraint_violation': constraint_violation.item()
        }
        
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return stats
    
    def compute_gae(self, 
                   rewards: List[float], 
                   values: List[float], 
                   dones: List[bool],
                   costs: List[float] = None,
                   cost_values: List[float] = None,
                   gamma: float = 0.99, 
                   lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) for both reward and cost
        """
        advantages = []
        cost_advantages = []
        returns = []
        cost_returns = []
        
        gae = 0
        cost_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_cost_value = 0
            else:
                next_value = values[t + 1]
                next_cost_value = cost_values[t + 1] if cost_values else 0
            
            # Reward advantage
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            # Cost advantage
            if costs and cost_values:
                cost_delta = costs[t] + gamma * next_cost_value * (1 - dones[t]) - cost_values[t]
                cost_gae = cost_delta + gamma * lam * (1 - dones[t]) * cost_gae
                cost_advantages.insert(0, cost_gae)
                cost_returns.insert(0, cost_gae + cost_values[t])
            else:
                cost_advantages.insert(0, 0)
                cost_returns.insert(0, 0)
        
        advantages = np.array(advantages)
        cost_advantages = np.array(cost_advantages)
        returns = np.array(returns)
        cost_returns = np.array(cost_returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if len(cost_advantages) > 1 and cost_advantages.std() > 0:
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        return advantages, returns, cost_advantages, cost_returns
    
    def update(self, rollout_data: Dict[str, List]) -> Dict[str, float]:
        """Update agent using collected rollout data"""
        
        states = np.array(rollout_data['states'])
        actions = np.array(rollout_data['actions'])
        rewards = rollout_data['rewards']
        dones = rollout_data['dones']
        log_probs = np.array(rollout_data['log_probs'])
        values = np.array(rollout_data['values'])
        costs = rollout_data.get('costs', [0] * len(rewards))
        cost_values = rollout_data.get('cost_values', values)  # Use values if no cost values
        
        # Compute advantages and returns
        advantages, returns, cost_advantages, cost_returns = self.compute_gae(
            rewards, values, dones, costs, cost_values
        )
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        cost_advantages_tensor = torch.FloatTensor(cost_advantages).to(self.device)
        cost_returns_tensor = torch.FloatTensor(cost_returns).to(self.device)
        costs_tensor = torch.FloatTensor(costs).to(self.device)
        
        # Prepare batch data
        batch_data = {
            'states': states_tensor,
            'actions': actions_tensor,
            'log_probs': old_log_probs_tensor,
            'advantages': advantages_tensor,
            'returns': returns_tensor,
            'costs': costs_tensor,
            'cost_advantages': cost_advantages_tensor,
            'cost_returns': cost_returns_tensor
        }
        
        # Multiple epochs of updates
        total_stats = {}
        for epoch in range(self.n_epochs):
            epoch_stats = self.train_step(batch_data)
            
            for key, value in epoch_stats.items():
                if key not in total_stats:
                    total_stats[key] = []
                total_stats[key].append(value)
        
        # Average statistics
        avg_stats = {key: np.mean(values) for key, values in total_stats.items()}
        
        return avg_stats
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_func.state_dict(),
            'cost_state_dict': self.cost_func.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'cost_optimizer_state_dict': self.cost_optimizer.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier.item(),
            'training_stats': dict(self.training_stats),
            'config': self.config
        }, filepath)
        
        self.logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_func.load_state_dict(checkpoint['value_state_dict'])
        self.cost_func.load_state_dict(checkpoint['cost_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.cost_optimizer.load_state_dict(checkpoint['cost_optimizer_state_dict'])
        
        self.lagrange_multiplier = torch.tensor(
            checkpoint['lagrange_multiplier'], 
            requires_grad=True, 
            device=self.device
        )
        
        self.logger.info(f"Agent loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'recent_{key}'] = values[-1] if values else 0
        
        return stats