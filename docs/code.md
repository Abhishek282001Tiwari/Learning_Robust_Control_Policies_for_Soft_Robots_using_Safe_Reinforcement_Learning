---
layout: page
title: Code
permalink: /code/
---

# Implementation and Code Documentation

This page provides comprehensive documentation of our Safe RL implementation for soft robots, including code structure, usage examples, and API documentation.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Learning_Robust_Control_Policies_for_Soft_Robots_using_Safe_Reinforcement_Learning.git
cd Learning_Robust_Control_Policies_for_Soft_Robots_using_Safe_Reinforcement_Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train a Safe PPO agent
python main.py train --config experiments/configs/default_config.yaml --name my_experiment

# Evaluate a trained model  
python main.py evaluate --model data/models/safe_ppo_tentacle_best.pth --episodes 20

# Run interactive testing
python main.py test --model data/models/safe_ppo_tentacle_best.pth
```

## Project Structure

```
src/
├── environments/           # Robot simulation environments
│   └── soft_robot_env.py  # Main PyBullet environment
├── agents/                # RL agents and algorithms  
│   └── safe_ppo_agent.py  # Safe PPO implementation
├── safety/                # Safety monitoring framework
│   └── safety_monitor.py  # Constraint violation detection
├── utils/                 # Utility functions
│   ├── logger.py          # Logging utilities
│   └── visualization.py   # Plotting and visualization
└── models/                # Neural network architectures

experiments/
├── configs/               # YAML configuration files
└── results/              # Training results and logs

data/
├── models/               # Saved model checkpoints
├── experimental_results/ # Training data and metrics  
└── analysis/             # Safety and performance analysis

tests/                    # Comprehensive unit tests
├── test_environment.py   # Environment tests
├── test_agent.py         # Agent tests
├── test_safety.py        # Safety framework tests
└── test_utils.py         # Utility tests
```

## Core Components

### 1. Soft Robot Environment

The `SoftRobotEnv` class provides a unified interface for different soft robot configurations using PyBullet physics simulation.

#### Key Features

- **Multi-robot Support**: Tentacle, gripper, and locomotion robots
- **Realistic Physics**: Deformable dynamics with spring-damper models
- **Safety Integration**: Real-time constraint monitoring
- **Domain Randomization**: Parameter variation for robustness

#### Example Usage

```python
from src.environments.soft_robot_env import SoftRobotEnv

# Create environment with custom configuration
config = {
    'environment': {'robot_type': 'tentacle', 'max_episode_steps': 500},
    'robot': {'segments': 6, 'stiffness': 800.0},
    'safety': {'max_deformation': 0.6, 'collision_threshold': 0.005}
}

env = SoftRobotEnv(config=config, render_mode='human')
```

#### API Reference

**Constructor Parameters**:
- `config_path` (str, optional): Path to YAML configuration file
- `config` (dict, optional): Configuration dictionary
- `render_mode` (str): Rendering mode ('human', 'rgb_array')

**Key Methods**:
- `reset()`: Reset environment and return initial observation
- `step(action)`: Execute action and return (obs, reward, done, info)
- `render(mode)`: Render the environment
- `close()`: Clean up resources

**Observation Space**: Box(24,) containing robot pose, joint angles, and velocities
**Action Space**: Box(8,) containing normalized actuator commands

### 2. Safe PPO Agent

The `SafePPOAgent` implements our safe reinforcement learning algorithm combining PPO with Constrained Policy Optimization.

#### Key Features

- **Dual Value Functions**: Separate reward and cost value estimation
- **Safety Constraints**: Lagrange multiplier-based constraint optimization
- **Uncertainty Quantification**: Monte Carlo dropout for epistemic uncertainty
- **Robust Training**: Gradient clipping and adaptive learning rates

#### Example Usage

```python
from src.agents.safe_ppo_agent import SafePPOAgent

# Initialize agent
agent = SafePPOAgent(
    state_dim=24,
    action_dim=8,
    config=config,
    device='cuda'
)

# Training loop
for episode in range(1000):
    state = env.reset()
    rollout_data = collect_rollout(env, agent)
    update_stats = agent.update(rollout_data)
```

#### Architecture Details

**Policy Network**:
- Input: 24-dimensional state vector
- Hidden layers: 256→256 with ReLU activation and dropout
- Output: 8-dimensional action mean and learned log-std

**Value Networks**:
- Reward value function: Estimates expected future rewards
- Cost value function: Estimates expected future constraint violations
- Architecture: 256→256→1 with ReLU activation

**Training Algorithm**:
1. Collect rollout data using current policy
2. Compute GAE for both reward and cost advantages  
3. Update policy using clipped PPO objective with cost penalty
4. Update value functions using MSE loss
5. Adapt Lagrange multiplier based on constraint satisfaction

### 3. Safety Monitor

The `SafetyMonitor` class provides comprehensive safety constraint monitoring and violation detection.

#### Constraint Types

1. **Collision Constraints**: Minimum distance to obstacles
2. **Deformation Constraints**: Maximum joint angle deviations
3. **Force Constraints**: Actuator force limits
4. **Velocity Constraints**: Maximum joint velocities
5. **Emergency Stops**: Critical safety interventions

#### Example Usage

```python
from src.safety.safety_monitor import SafetyMonitor, SafetyWrapper

# Create safety monitor
monitor = SafetyMonitor(config)

# Wrap environment with safety monitoring
safe_env = SafetyWrapper(env, config)

# Safety monitoring happens automatically during env.step()
```

#### Safety Metrics

The safety monitor tracks comprehensive metrics:

- **Violation Counts**: By type and severity
- **Safety Score**: Normalized safety performance (0-1)
- **Violation Timeline**: Historical trend analysis
- **Emergency Events**: Critical failure logging

### 4. Training Framework

Our training framework provides a complete pipeline for safe RL experiments.

#### Configuration System

Training is controlled through YAML configuration files:

```yaml
# Example configuration
environment:
  robot_type: "tentacle"
  max_episode_steps: 1000
  
robot:
  segments: 4
  stiffness: 1000.0
  damping: 10.0
  
safety:
  max_deformation: 0.5
  collision_threshold: 0.01
  force_limit: 15.0
  
algorithm:
  learning_rate: 3e-4
  clip_range: 0.2
  cost_limit: 25.0
  
training:
  total_timesteps: 1000000
  eval_freq: 10000
  save_freq: 50000
```

#### Training Script

The main training script (`scripts/training/train_safe_agent.py`) provides:

- **Experiment Management**: Automatic logging and checkpointing
- **Evaluation**: Periodic assessment during training
- **Visualization**: Real-time training curves
- **Safety Monitoring**: Continuous constraint tracking

#### Example Training Session

```python
from scripts.training.train_safe_agent import SafeRLTrainer

# Create trainer
trainer = SafeRLTrainer(
    config_path='experiments/configs/default_config.yaml',
    experiment_name='safe_ppo_experiment'
)

# Start training
trainer.train()
```

## Advanced Usage

### Custom Robot Configuration

Create new robot types by extending the base environment:

```python
class CustomSoftRobot(SoftRobotEnv):
    def _create_soft_robot(self):
        # Implement custom robot creation logic
        # Return robot body ID
        pass
    
    def _get_observation(self):
        # Define custom observation space
        pass
```

### Custom Safety Constraints

Add domain-specific safety constraints:

```python
class CustomSafetyMonitor(SafetyMonitor):
    def check_custom_constraint(self, robot_state):
        # Implement custom constraint logic
        violation = check_condition(robot_state)
        severity = calculate_severity(violation)
        return violation, severity
    
    def monitor_step(self, robot_state, action, contact_points, applied_forces):
        # Call parent monitoring
        safety_info = super().monitor_step(robot_state, action, contact_points, applied_forces)
        
        # Add custom constraint
        violation, severity = self.check_custom_constraint(robot_state)
        if violation:
            safety_info['violations']['custom'] = severity
        
        return safety_info
```

### Hyperparameter Optimization

Use the configuration system for systematic hyperparameter search:

```python
import itertools
from copy import deepcopy

# Define hyperparameter grid
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'cost_limit': [10, 25, 50],
    'clip_range': [0.1, 0.2, 0.3]
}

# Grid search
best_score = -float('inf')
for params in itertools.product(*param_grid.values()):
    config = deepcopy(base_config)
    
    # Update configuration
    config['algorithm']['learning_rate'] = params[0]
    config['safe_rl']['cost_limit'] = params[1] 
    config['algorithm']['clip_range'] = params[2]
    
    # Train and evaluate
    trainer = SafeRLTrainer(config=config)
    results = trainer.train()
    
    if results['final_safety_score'] > best_score:
        best_score = results['final_safety_score']
        best_params = params
```

## Testing and Validation

### Running Unit Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test modules
python -m pytest tests/test_environment.py -v
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_safety.py -v
python -m pytest tests/test_utils.py -v
```

### Test Coverage

Our test suite provides comprehensive coverage:

- **Environment Tests**: Physics simulation, observation spaces, robot creation
- **Agent Tests**: Neural networks, policy updates, training loops
- **Safety Tests**: Constraint monitoring, violation detection, emergency stops
- **Utility Tests**: Logging, visualization, data processing

### Continuous Integration

The project includes GitHub Actions workflow for automated testing:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python tests/run_tests.py
```

## Performance Optimization

### Memory Optimization

For large-scale training, consider these optimizations:

```python
# Use gradient checkpointing for memory efficiency
import torch.utils.checkpoint as checkpoint

class MemoryEfficientPolicy(nn.Module):
    def forward(self, x):
        return checkpoint.checkpoint(self.policy_net, x)

# Reduce batch size and increase update frequency
config['algorithm']['batch_size'] = 32
config['algorithm']['n_epochs'] = 20
```

### GPU Acceleration

Optimize GPU usage for faster training:

```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training step with mixed precision
with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Parallel Training

Scale training across multiple GPUs:

```python
# Multi-GPU training setup
import torch.nn.parallel as parallel

if torch.cuda.device_count() > 1:
    agent.policy = parallel.DataParallel(agent.policy)
    agent.value_func = parallel.DataParallel(agent.value_func)
```

## Troubleshooting

### Common Issues

**Environment Creation Fails**:
- Check PyBullet installation: `pip install pybullet`
- Verify URDF file paths in configuration
- Ensure adequate GPU memory for physics simulation

**Training Instability**:
- Reduce learning rate: Try `1e-4` instead of `3e-4`
- Increase batch size for more stable gradients
- Check for NaN values in loss computation

**Safety Violations Not Detected**:
- Verify constraint thresholds in configuration
- Check robot state dimensions match expected values
- Enable debug logging for detailed violation information

**Memory Issues**:
- Reduce batch size and episode length
- Clear GPU cache periodically: `torch.cuda.empty_cache()`
- Use CPU for evaluation if GPU memory is limited

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable PyBullet debug visualization
env = SoftRobotEnv(config=config, render_mode='human')
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
```

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

### Code Style

We follow PEP 8 with these modifications:
- Line length: 100 characters
- Use type hints for public APIs
- Document all classes and functions
- Include comprehensive docstrings

### Pull Request Process

1. Fork the repository and create a feature branch
2. Implement changes with appropriate tests
3. Ensure all tests pass and code is formatted
4. Submit pull request with detailed description
5. Address review feedback and maintainer comments

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{safe_soft_robot_rl_2024,
  title={Learning Robust Control Policies for Soft Robots using Safe Reinforcement Learning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/Learning_Robust_Control_Policies_for_Soft_Robots_using_Safe_Reinforcement_Learning}},
  note={Research project on safe RL for soft robotics}
}
```

## Support and Contact

For questions, issues, or contributions:

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-username/Learning_Robust_Control_Policies_for_Soft_Robots_using_Safe_Reinforcement_Learning/issues)
- **Email**: your.email@example.com
- **Documentation**: [Technical Documentation](methodology.html)

---

*This codebase represents ongoing research in safe reinforcement learning for soft robotics. We welcome contributions from the research community to advance this important area of study.*