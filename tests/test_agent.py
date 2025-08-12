import unittest
import numpy as np
import torch
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.safe_ppo_agent import SafePPOAgent, PolicyNetwork, ValueNetwork, CostNetwork


class TestNeuralNetworks(unittest.TestCase):
    """Test cases for neural network components"""
    
    def setUp(self):
        """Set up test parameters"""
        self.state_dim = 24
        self.action_dim = 8
        self.hidden_dims = [64, 64]
        self.dropout_rate = 0.1
    
    def test_policy_network_initialization(self):
        """Test policy network initialization"""
        policy = PolicyNetwork(
            self.state_dim, 
            self.action_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        # Check network structure
        self.assertEqual(policy.state_dim, self.state_dim)
        self.assertEqual(policy.action_dim, self.action_dim)
        self.assertEqual(policy.dropout_rate, self.dropout_rate)
        
        # Check that log_std is learnable parameter
        self.assertTrue(policy.log_std.requires_grad)
        self.assertEqual(policy.log_std.shape[0], self.action_dim)
    
    def test_policy_network_forward(self):
        """Test policy network forward pass"""
        policy = PolicyNetwork(self.state_dim, self.action_dim)
        
        batch_size = 32
        state = torch.randn(batch_size, self.state_dim)
        
        mean, std = policy(state)
        
        # Check output shapes
        self.assertEqual(mean.shape, (batch_size, self.action_dim))
        self.assertEqual(std.shape, (batch_size, self.action_dim))
        
        # Check that std is positive
        self.assertTrue(torch.all(std > 0))
    
    def test_policy_network_action_sampling(self):
        """Test action sampling with log probabilities"""
        policy = PolicyNetwork(self.state_dim, self.action_dim)
        
        state = torch.randn(1, self.state_dim)
        
        # Test stochastic action
        action, log_prob = policy.get_action_and_log_prob(state, deterministic=False)
        
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertEqual(log_prob.shape, (1,))
        
        # Actions should be in [-1, 1] due to tanh squashing
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))
        
        # Test deterministic action
        det_action, det_log_prob = policy.get_action_and_log_prob(state, deterministic=True)
        
        self.assertEqual(det_action.shape, (1, self.action_dim))
        self.assertEqual(det_log_prob.shape, (1,))
        
        # Deterministic log prob should be zero
        self.assertTrue(torch.allclose(det_log_prob, torch.zeros_like(det_log_prob)))
    
    def test_policy_uncertainty_estimation(self):
        """Test uncertainty estimation using dropout"""
        policy = PolicyNetwork(self.state_dim, self.action_dim, dropout_rate=0.2)
        
        state = torch.randn(1, self.state_dim)
        
        mean_action, uncertainty = policy.get_uncertainty(state, n_samples=10)
        
        self.assertEqual(mean_action.shape, (1, self.action_dim))
        self.assertEqual(uncertainty.shape, (1, self.action_dim))
        
        # Uncertainty should be non-negative
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_value_network(self):
        """Test value network"""
        value_net = ValueNetwork(self.state_dim, hidden_dims=[32, 32])
        
        batch_size = 16
        state = torch.randn(batch_size, self.state_dim)
        
        value = value_net(state)
        
        # Check output shape (single value per state)
        self.assertEqual(value.shape, (batch_size,))
    
    def test_cost_network(self):
        """Test cost network"""
        cost_net = CostNetwork(self.state_dim, hidden_dims=[32, 32])
        
        batch_size = 16
        state = torch.randn(batch_size, self.state_dim)
        
        cost = cost_net(state)
        
        # Check output shape
        self.assertEqual(cost.shape, (batch_size,))


class TestSafePPOAgent(unittest.TestCase):
    """Test cases for Safe PPO agent"""
    
    def setUp(self):
        """Set up test agent"""
        self.state_dim = 24
        self.action_dim = 8
        self.config = {
            'algorithm': {
                'learning_rate': 3e-4,
                'clip_range': 0.2,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_epochs': 4,
                'batch_size': 32
            },
            'safe_rl': {
                'cost_limit': 25.0,
                'lagrange_multiplier_lr': 1e-3,
                'safety_budget': 0.05
            }
        }
        self.device = 'cpu'
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        # Check basic properties
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertEqual(agent.device.type, self.device)
        
        # Check networks exist
        self.assertIsInstance(agent.policy, PolicyNetwork)
        self.assertIsInstance(agent.value_func, ValueNetwork)
        self.assertIsInstance(agent.cost_func, CostNetwork)
        
        # Check optimizers exist
        self.assertIsNotNone(agent.policy_optimizer)
        self.assertIsNotNone(agent.value_optimizer)
        self.assertIsNotNone(agent.cost_optimizer)
        self.assertIsNotNone(agent.lagrange_optimizer)
        
        # Check Lagrange multiplier is properly initialized
        self.assertTrue(agent.lagrange_multiplier.requires_grad)
        self.assertEqual(agent.lagrange_multiplier.item(), 1.0)
    
    def test_action_selection(self):
        """Test action selection with and without determinism"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        state = np.random.randn(self.state_dim)
        
        # Test stochastic action selection
        action, info = agent.select_action(state, deterministic=False)
        
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertIn('log_prob', info)
        self.assertIn('value', info)
        self.assertIn('cost', info)
        self.assertIn('uncertainty', info)
        
        # Check action bounds
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        
        # Test deterministic action selection
        det_action, det_info = agent.select_action(state, deterministic=True)
        
        self.assertEqual(det_action.shape, (self.action_dim,))
        self.assertEqual(det_info['uncertainty'].shape, (self.action_dim,))
    
    def test_gae_computation(self):
        """Test Generalized Advantage Estimation"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        # Create sample trajectory
        rewards = [1.0, 0.5, -0.1, 0.8, -0.5]
        values = [0.9, 0.7, 0.2, 0.6, 0.1]
        dones = [False, False, False, False, True]
        costs = [0.1, 0.0, 0.3, 0.0, 0.2]
        cost_values = [0.05, 0.02, 0.25, 0.01, 0.15]
        
        advantages, returns, cost_advantages, cost_returns = agent.compute_gae(
            rewards, values, dones, costs, cost_values
        )
        
        # Check shapes
        self.assertEqual(len(advantages), len(rewards))
        self.assertEqual(len(returns), len(rewards))
        self.assertEqual(len(cost_advantages), len(rewards))
        self.assertEqual(len(cost_returns), len(rewards))
        
        # Check normalization (advantages should have zero mean, unit variance)
        self.assertAlmostEqual(np.mean(advantages), 0.0, places=5)
        self.assertAlmostEqual(np.std(advantages), 1.0, places=5)
    
    def test_training_step(self):
        """Test single training step"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        batch_size = 64
        batch_data = {
            'states': torch.randn(batch_size, self.state_dim),
            'actions': torch.randn(batch_size, self.action_dim),
            'log_probs': torch.randn(batch_size),
            'advantages': torch.randn(batch_size),
            'returns': torch.randn(batch_size),
            'costs': torch.randn(batch_size),
            'cost_advantages': torch.randn(batch_size),
            'cost_returns': torch.randn(batch_size)
        }
        
        # Get initial parameters for comparison
        initial_policy_params = list(agent.policy.parameters())[0].clone()
        
        stats = agent.train_step(batch_data)
        
        # Check that parameters were updated
        updated_policy_params = list(agent.policy.parameters())[0]
        self.assertFalse(torch.allclose(initial_policy_params, updated_policy_params))
        
        # Check returned statistics
        required_stats = ['policy_loss', 'value_loss', 'cost_loss', 'entropy', 
                         'cost_penalty', 'lagrange_multiplier', 'constraint_violation']
        
        for stat in required_stats:
            self.assertIn(stat, stats)
            self.assertIsInstance(stats[stat], (int, float))
    
    def test_update_with_rollout(self):
        """Test update with rollout data"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        # Create sample rollout data
        trajectory_length = 100
        rollout_data = {
            'states': [np.random.randn(self.state_dim) for _ in range(trajectory_length)],
            'actions': [np.random.randn(self.action_dim) for _ in range(trajectory_length)],
            'rewards': np.random.randn(trajectory_length).tolist(),
            'dones': [False] * (trajectory_length - 1) + [True],
            'log_probs': np.random.randn(trajectory_length).tolist(),
            'values': np.random.randn(trajectory_length).tolist(),
            'costs': np.random.randn(trajectory_length).tolist(),
            'cost_values': np.random.randn(trajectory_length).tolist()
        }
        
        # Perform update
        avg_stats = agent.update(rollout_data)
        
        # Check that statistics are returned
        self.assertIsInstance(avg_stats, dict)
        self.assertIn('policy_loss', avg_stats)
        self.assertIn('value_loss', avg_stats)
        
        # Check that Lagrange multiplier is non-negative
        self.assertGreaterEqual(agent.lagrange_multiplier.item(), 0.0)
    
    def test_save_and_load(self):
        """Test saving and loading agent state"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        # Get initial state
        initial_policy_params = list(agent.policy.parameters())[0].clone()
        initial_lagrange = agent.lagrange_multiplier.item()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            filepath = f.name
        
        try:
            agent.save(filepath)
            
            # Modify agent state
            with torch.no_grad():
                for param in agent.policy.parameters():
                    param.fill_(999.0)
                agent.lagrange_multiplier.fill_(999.0)
            
            # Load saved state
            agent.load(filepath)
            
            # Check that state was restored
            loaded_policy_params = list(agent.policy.parameters())[0]
            loaded_lagrange = agent.lagrange_multiplier.item()
            
            self.assertTrue(torch.allclose(initial_policy_params, loaded_policy_params))
            self.assertAlmostEqual(initial_lagrange, loaded_lagrange, places=5)
            
        finally:
            os.unlink(filepath)
    
    def test_training_statistics(self):
        """Test training statistics tracking"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=self.device
        )
        
        # Add some statistics
        agent.training_stats['policy_loss'].extend([1.0, 0.8, 0.6])
        agent.training_stats['value_loss'].extend([2.0, 1.5, 1.2])
        
        stats = agent.get_training_stats()
        
        # Check averaged statistics
        self.assertIn('avg_policy_loss', stats)
        self.assertIn('avg_value_loss', stats)
        self.assertIn('recent_policy_loss', stats)
        self.assertIn('recent_value_loss', stats)
        
        self.assertAlmostEqual(stats['avg_policy_loss'], 0.8, places=5)
        self.assertAlmostEqual(stats['recent_policy_loss'], 0.6, places=5)


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agent components"""
    
    def setUp(self):
        """Set up integration test"""
        self.state_dim = 12
        self.action_dim = 4
        self.config = {
            'algorithm': {
                'learning_rate': 1e-3,
                'clip_range': 0.1,
                'entropy_coef': 0.001,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_epochs': 2,
                'batch_size': 16
            },
            'safe_rl': {
                'cost_limit': 10.0,
                'lagrange_multiplier_lr': 1e-3,
                'safety_budget': 0.1
            }
        }
    
    def test_complete_training_loop(self):
        """Test complete training loop integration"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device='cpu'
        )
        
        # Simulate multiple episodes
        all_rollout_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'costs': [],
            'cost_values': []
        }
        
        # Generate episodes
        for episode in range(3):
            episode_length = 50
            
            for step in range(episode_length):
                state = np.random.randn(self.state_dim)
                action, info = agent.select_action(state)
                
                all_rollout_data['states'].append(state)
                all_rollout_data['actions'].append(action)
                all_rollout_data['rewards'].append(np.random.randn())
                all_rollout_data['dones'].append(step == episode_length - 1)
                all_rollout_data['log_probs'].append(info['log_prob'])
                all_rollout_data['values'].append(info['value'])
                all_rollout_data['costs'].append(abs(np.random.randn()) * 0.1)
                all_rollout_data['cost_values'].append(info['cost'])
        
        # Perform training update
        initial_loss = None
        for update in range(3):
            stats = agent.update(all_rollout_data)
            
            if initial_loss is None:
                initial_loss = stats['policy_loss']
            
            # Check that training produces reasonable statistics
            self.assertIsInstance(stats['policy_loss'], (int, float))
            self.assertIsInstance(stats['value_loss'], (int, float))
            self.assertIsInstance(stats['entropy'], (int, float))
        
        # Training should generally reduce loss (not always guaranteed due to randomness)
        # But we can at least check that the process completes without errors
        final_stats = agent.get_training_stats()
        self.assertIsInstance(final_stats, dict)
    
    def test_constraint_satisfaction(self):
        """Test that agent respects safety constraints during training"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device='cpu'
        )
        
        # Create rollout with high costs (constraint violations)
        rollout_data = {
            'states': [np.random.randn(self.state_dim) for _ in range(64)],
            'actions': [np.random.randn(self.action_dim) for _ in range(64)],
            'rewards': np.random.randn(64).tolist(),
            'dones': [False] * 63 + [True],
            'log_probs': np.random.randn(64).tolist(),
            'values': np.random.randn(64).tolist(),
            'costs': [5.0] * 64,  # High costs (above limit)
            'cost_values': [4.0] * 64
        }
        
        initial_lagrange = agent.lagrange_multiplier.item()
        
        # Update with high-cost rollout
        for _ in range(5):
            stats = agent.update(rollout_data)
        
        final_lagrange = agent.lagrange_multiplier.item()
        
        # Lagrange multiplier should increase to penalize constraint violations
        self.assertGreater(final_lagrange, initial_lagrange)
    
    def test_policy_gradient_flow(self):
        """Test that gradients flow properly through the network"""
        agent = SafePPOAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device='cpu'
        )
        
        # Create batch data
        batch_size = 32
        batch_data = {
            'states': torch.randn(batch_size, self.state_dim, requires_grad=False),
            'actions': torch.randn(batch_size, self.action_dim, requires_grad=False),
            'log_probs': torch.randn(batch_size, requires_grad=False),
            'advantages': torch.randn(batch_size, requires_grad=False),
            'returns': torch.randn(batch_size, requires_grad=False),
            'costs': torch.randn(batch_size, requires_grad=False),
            'cost_advantages': torch.randn(batch_size, requires_grad=False),
            'cost_returns': torch.randn(batch_size, requires_grad=False)
        }
        
        # Check that all networks have gradients after training step
        stats = agent.train_step(batch_data)
        
        # Check policy network gradients
        policy_has_grad = any(param.grad is not None and param.grad.sum() != 0 
                             for param in agent.policy.parameters())
        self.assertTrue(policy_has_grad, "Policy network should have non-zero gradients")
        
        # Check value network gradients
        value_has_grad = any(param.grad is not None and param.grad.sum() != 0 
                            for param in agent.value_func.parameters())
        self.assertTrue(value_has_grad, "Value network should have non-zero gradients")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all tests
    unittest.main(verbosity=2)