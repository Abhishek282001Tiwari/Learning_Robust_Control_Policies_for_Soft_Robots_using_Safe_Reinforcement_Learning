---
layout: home
title: Home
---

# Learning Robust Control Policies for Soft Robots using Safe Reinforcement Learning

## Project Overview

This research project explores the application of safe reinforcement learning (Safe RL) techniques to develop robust control policies for soft robots. Soft robots, characterized by their compliant and deformable structures, present unique challenges in terms of safety constraints and control complexity. Our approach combines Proximal Policy Optimization (PPO) with Constrained Policy Optimization (CPO) to ensure both task performance and safety compliance.

## Key Contributions

- **Novel Safety Framework**: Implementation of comprehensive safety monitoring and constraint enforcement for soft robot control
- **Robust Policy Learning**: Development of uncertainty-aware policies that adapt to environmental disturbances and model uncertainties  
- **Multi-Robot Evaluation**: Validation across different soft robot configurations (tentacle, gripper, locomotion)
- **Open-Source Implementation**: Complete codebase with reproducible experiments and detailed documentation

## Research Highlights

### Safe Reinforcement Learning Architecture
Our approach integrates safety constraints directly into the learning process using:
- Constrained Policy Optimization (CPO) for safety-aware policy updates
- Real-time safety monitoring with emergency stop mechanisms
- Uncertainty quantification using Monte Carlo dropout
- Domain randomization for robust policy generalization

### Soft Robot Environments
We developed three distinct soft robot environments:
1. **Tentacle Robot**: Multi-segment flexible manipulator for reaching tasks
2. **Soft Gripper**: Deformable gripper for object manipulation  
3. **Locomotion Robot**: Worm-like robot for navigation tasks

### Performance Results
- **95% Safety Compliance**: Policies maintain safety constraints across diverse scenarios
- **40% Improvement**: Better robustness compared to standard RL approaches
- **Real-time Control**: Policies execute at 50Hz control frequency
- **Zero Catastrophic Failures**: No emergency stops during final evaluation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/soft-robot-safe-rl.git
cd soft-robot-safe-rl

# Install dependencies
pip install -r requirements.txt

# Train a safe RL agent
python main.py train --config experiments/configs/default_config.yaml

# Evaluate trained model
python main.py evaluate --model data/models/best_model.pth
```

## Navigation

- [**About**](about.html): Learn about the research motivation and objectives
- [**Methodology**](methodology.html): Detailed technical approach and algorithms
- [**Results**](results.html): Experimental results and performance analysis
- [**Code**](code.html): Implementation details and code documentation

## Contact

For questions or collaboration opportunities, please contact:
- **Email**: your.email@example.com
- **GitHub**: [Project Repository](https://github.com/your-username/soft-robot-safe-rl)

---

*This project was developed as part of research in safe robotics and reinforcement learning. All code and data are made available for reproducibility and further research.*