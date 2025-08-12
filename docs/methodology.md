---
layout: page
title: Methodology
permalink: /methodology/
---

# Technical Methodology

## System Architecture Overview

Our safe reinforcement learning framework consists of four integrated components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │    │   Safe Agent    │    │ Safety Monitor  │
│                 │    │                 │    │                 │
│  • Soft Robot   │◄──►│  • Policy Net   │◄──►│ • Constraints   │
│  • Physics Sim  │    │  • Value Net    │    │ • Violations    │
│  • Sensors      │    │  • Cost Net     │    │ • Emergency     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Robust Control  │    │ Training Loop   │    │ Visualization   │
│                 │    │                 │    │                 │
│ • Uncertainty   │    │ • PPO + CPO     │    │ • Metrics       │
│ • Adaptation    │    │ • Experience    │    │ • Plots         │
│ • Disturbances  │    │ • Updates       │    │ • Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 1. Soft Robot Environment

### Physics Simulation

We utilize PyBullet for realistic soft body simulation with the following key components:

**Deformable Body Modeling**
- Multi-segment rigid body approximation of soft structures
- Spring-damper connections between segments for compliance
- Realistic material properties (stiffness, damping, mass distribution)

```python
# Example: Tentacle robot with 4 segments
segments = 4
stiffness = 1000.0  # N/m
damping = 10.0      # Ns/m
```

**Contact Dynamics**
- Collision detection with environment objects
- Contact force computation and distribution
- Surface friction and material interaction modeling

**Sensor Modeling**
- Joint angle and velocity measurements
- Contact force sensors
- Pose estimation with realistic noise models

### Robot Configurations

**Tentacle Robot**
- 4-8 flexible segments connected by spherical joints
- Pneumatic or cable actuation model
- Reaching and manipulation tasks

**Soft Gripper**
- Two-finger gripper with deformable fingertips
- Variable stiffness control
- Grasping and manipulation tasks

**Locomotion Robot**
- Worm-like multi-segment body
- Peristaltic motion generation
- Navigation and mobility tasks

## 2. Safe Reinforcement Learning Algorithm

### Core Algorithm: Safe PPO

Our approach extends Proximal Policy Optimization (PPO) with Constrained Policy Optimization (CPO) elements:

**Policy Network**
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_rate=0.1):
        # Network layers with dropout for uncertainty
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # ... more layers
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
```

**Dual Value Functions**
- Reward Value Function: $V^R_\phi(s)$ estimates expected future rewards
- Cost Value Function: $V^C_\phi(s)$ estimates expected future constraint violations

**Constrained Objective**
The policy optimization objective combines reward maximization with constraint satisfaction:

$$L(\theta) = L^{PPO}(\theta) - \lambda \cdot \max(0, J^C(\pi_\theta) - d)$$

Where:
- $L^{PPO}(\theta)$ is the standard PPO loss
- $J^C(\pi_\theta)$ is the expected constraint violation
- $d$ is the constraint threshold
- $\lambda$ is the adaptive Lagrange multiplier

### Uncertainty Quantification

**Monte Carlo Dropout**
During policy execution, we estimate epistemic uncertainty by:
1. Running multiple forward passes with dropout enabled
2. Computing action mean and variance across samples
3. Using uncertainty for risk-aware action selection

**Ensemble Methods**
For critical applications, we maintain an ensemble of policies:
```python
def get_ensemble_action(self, state, n_models=5):
    actions = []
    for model in self.ensemble:
        action = model.get_action(state)
        actions.append(action)
    
    mean_action = np.mean(actions, axis=0)
    uncertainty = np.std(actions, axis=0)
    return mean_action, uncertainty
```

## 3. Safety Framework

### Constraint Definition

**Physical Constraints**
- Maximum deformation limits: $|\theta_i| < \theta_{max}$
- Force limits: $|F_i| < F_{max}$
- Velocity limits: $|\dot{q}_i| < v_{max}$
- Collision avoidance: $d_{min} > d_{threshold}$

**Safety Zones**
We define multiple safety zones with increasing intervention levels:
- **Safe Zone**: Normal operation, no restrictions
- **Warning Zone**: Reduced action space, increased monitoring  
- **Danger Zone**: Emergency protocols, conservative actions
- **Critical Zone**: Immediate stop, system shutdown

### Real-Time Monitoring

**Predictive Safety Checks**
Before executing any action, we predict its safety consequences:

```python
def is_safe_action(self, action, state):
    # Predict next state
    next_state = self.dynamics_model.predict(state, action)
    
    # Check all constraints
    for constraint in self.constraints:
        if constraint.will_violate(state, action, next_state):
            return False
    return True
```

**Constraint Violation Detection**
- Real-time monitoring of all safety constraints
- Severity scoring based on constraint margin
- Automatic logging and alerting systems

**Emergency Stop Mechanisms**
- Hardware-level emergency stops for critical violations
- Software-level action modification for minor violations
- Graceful degradation with continued operation when possible

## 4. Robust Control Elements

### Domain Randomization

During training, we systematically vary:

**Robot Parameters**
- Segment mass: ±30% variation
- Stiffness: ±50% variation  
- Damping: ±40% variation
- Actuator limits: ±20% variation

**Environmental Conditions**
- Ground friction: 0.1 to 0.9
- Air resistance: 0% to 10% of motion
- External disturbances: random force/torque pulses
- Sensor noise: Gaussian with σ = 5% of signal

**Task Variations**
- Target positions: randomized within workspace
- Obstacle configurations: random placement and size
- Initial conditions: varied robot poses and velocities

### Adaptive Control

**Online Parameter Estimation**
- Recursive least squares for dynamics identification
- Bayesian inference for uncertainty quantification
- Real-time model adaptation based on observed performance

**Meta-Learning Approaches**
- Model-Agnostic Meta-Learning (MAML) for fast adaptation
- Context-dependent policy selection
- Transfer learning across robot configurations

## 5. Training Protocol

### Experience Collection

**Rollout Generation**
```python
def collect_rollout(self, env, agent, max_steps=2048):
    states, actions, rewards, costs = [], [], [], []
    
    state = env.reset()
    for step in range(max_steps):
        action, info = agent.select_action(state)
        next_state, reward, done, step_info = env.step(action)
        
        # Track both reward and safety cost
        cost = sum(step_info.get('violations', {}).values())
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        costs.append(cost)
        
        state = next_state
        if done:
            state = env.reset()
    
    return states, actions, rewards, costs
```

**Generalized Advantage Estimation (GAE)**
We compute advantages for both reward and cost functions using GAE with λ = 0.95:

$$A_t^R = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^R$$
$$A_t^C = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^C$$

### Policy Updates

**PPO Clipping**
Standard PPO clipping for the reward objective:
$$L^{PPO} = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

**Constraint Penalty**
Additional penalty term for constraint violations:
$$L^{CPO} = \mathbb{E}[\max(r_t(\theta)\hat{A}_t^C, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^C)]$$

**Lagrange Multiplier Update**
Adaptive adjustment of constraint weighting:
$$\lambda_{k+1} = \max(0, \lambda_k + \alpha_\lambda(J^C(\pi_k) - d))$$

## 6. Evaluation Metrics

### Performance Metrics
- **Task Success Rate**: Percentage of episodes completing the objective
- **Average Reward**: Mean cumulative reward per episode
- **Sample Efficiency**: Steps required to reach performance threshold
- **Computational Cost**: Training time and resource utilization

### Safety Metrics
- **Constraint Violation Rate**: Violations per episode
- **Safety Score**: $1 - \frac{\text{violations}}{\text{max violations}}$
- **Emergency Stop Frequency**: Critical failures requiring intervention
- **Recovery Time**: Steps needed to return to safe operation

### Robustness Metrics
- **Performance Variance**: Standard deviation across test conditions
- **Worst-Case Performance**: Minimum performance across all scenarios
- **Adaptation Speed**: Time to adjust to new conditions
- **Transfer Success**: Performance when applied to new robot configurations

## 7. Implementation Details

### Software Architecture
- **Python 3.8+** with PyTorch for neural networks
- **PyBullet** for physics simulation
- **Stable-Baselines3** as RL foundation (extended for safety)
- **Weights & Biases** for experiment tracking
- **Matplotlib/Plotly** for visualization

### Hardware Requirements
- **Training**: NVIDIA GPU with 8GB+ VRAM recommended
- **Inference**: CPU-only operation possible for real-time control
- **Storage**: 10GB+ for experiment logs and model checkpoints

### Hyperparameter Configuration
Key hyperparameters with default values:
```yaml
algorithm:
  learning_rate: 3e-4
  clip_range: 0.2
  entropy_coef: 0.01
  batch_size: 64
  n_epochs: 10

safety:
  cost_limit: 25.0
  lagrange_lr: 1e-3
  emergency_threshold: 0.8
```

This methodology provides a comprehensive framework for developing safe, robust control policies for soft robots through the integration of advanced reinforcement learning techniques with rigorous safety monitoring and constraint satisfaction mechanisms.