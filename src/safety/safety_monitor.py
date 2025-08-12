import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class SafetyViolation:
    """Data class for storing safety violation information"""
    timestamp: float
    violation_type: str
    severity: float
    robot_state: np.ndarray
    action: np.ndarray
    description: str


class SafetyMonitor:
    """
    Comprehensive safety monitoring system for soft robot environments.
    Tracks violations, implements emergency stops, and maintains safety logs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_config = config.get('safety', {})
        
        # Safety thresholds
        self.max_deformation = self.safety_config.get('max_deformation', 0.5)
        self.collision_threshold = self.safety_config.get('collision_threshold', 0.01)
        self.force_limit = self.safety_config.get('force_limit', 15.0)
        self.velocity_limit = self.safety_config.get('velocity_limit', 2.0)
        self.emergency_threshold = self.safety_config.get('emergency_stop_threshold', 0.8)
        
        # Safety monitoring state
        self.violations_history: List[SafetyViolation] = []
        self.violation_counts: Dict[str, int] = {}
        self.safety_buffer_zones = {}
        self.emergency_stop_active = False
        
        # Statistics tracking
        self.episode_violations: deque = deque(maxlen=100)
        self.total_episodes = 0
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize safety buffer zones
        self._initialize_safety_buffers()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup safety monitoring logger"""
        logger = logging.getLogger('SafetyMonitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_safety_buffers(self):
        """Initialize safety buffer zones around constraints"""
        self.safety_buffer_zones = {
            'collision': self.collision_threshold * 1.5,
            'deformation': self.max_deformation * 0.8,
            'force': self.force_limit * 0.9,
            'velocity': self.velocity_limit * 0.9
        }
    
    def check_collision_safety(self, contact_points: List) -> Tuple[bool, float]:
        """Check collision safety constraints"""
        if not contact_points:
            return False, 0.0
        
        min_distance = min([pt[8] for pt in contact_points if len(pt) > 8])
        
        if min_distance < self.collision_threshold:
            severity = (self.collision_threshold - min_distance) / self.collision_threshold
            return True, severity
        
        return False, 0.0
    
    def check_deformation_safety(self, robot_state: np.ndarray) -> Tuple[bool, float]:
        """Check deformation limits"""
        # Calculate deformation from joint angles (simplified)
        if len(robot_state) < 8:
            return False, 0.0
        
        joint_angles = robot_state[7:]  # Assuming first 7 are base pose/vel
        max_angle_deviation = np.max(np.abs(joint_angles))
        normalized_deformation = max_angle_deviation / (np.pi / 2)  # Normalize to 0-1
        
        if normalized_deformation > self.max_deformation:
            severity = (normalized_deformation - self.max_deformation) / self.max_deformation
            return True, min(severity, 1.0)
        
        return False, 0.0
    
    def check_force_safety(self, applied_forces: np.ndarray) -> Tuple[bool, float]:
        """Check force limits"""
        total_force = np.sum(np.abs(applied_forces))
        
        if total_force > self.force_limit:
            severity = (total_force - self.force_limit) / self.force_limit
            return True, min(severity, 1.0)
        
        return False, 0.0
    
    def check_velocity_safety(self, velocities: np.ndarray) -> Tuple[bool, float]:
        """Check velocity limits"""
        max_velocity = np.max(np.abs(velocities))
        
        if max_velocity > self.velocity_limit:
            severity = (max_velocity - self.velocity_limit) / self.velocity_limit
            return True, min(severity, 1.0)
        
        return False, 0.0
    
    def monitor_step(self, 
                    robot_state: np.ndarray,
                    action: np.ndarray,
                    contact_points: List,
                    applied_forces: np.ndarray) -> Dict[str, Any]:
        """
        Monitor safety constraints for a single step
        
        Returns:
            Dict containing safety status, violations, and emergency stop flag
        """
        current_time = time.time()
        violations = {}
        emergency_stop = False
        
        # Check collision safety
        collision_violation, collision_severity = self.check_collision_safety(contact_points)
        if collision_violation:
            violations['collision'] = collision_severity
            self._log_violation('collision', collision_severity, robot_state, action)
        
        # Check deformation safety
        deformation_violation, deformation_severity = self.check_deformation_safety(robot_state)
        if deformation_violation:
            violations['deformation'] = deformation_severity
            self._log_violation('deformation', deformation_severity, robot_state, action)
        
        # Check force safety
        force_violation, force_severity = self.check_force_safety(applied_forces)
        if force_violation:
            violations['force'] = force_severity
            self._log_violation('force', force_severity, robot_state, action)
        
        # Check velocity safety
        if len(robot_state) >= 6:
            velocities = robot_state[3:6]  # Assuming positions [0:3], velocities [3:6]
            velocity_violation, velocity_severity = self.check_velocity_safety(velocities)
            if velocity_violation:
                violations['velocity'] = velocity_severity
                self._log_violation('velocity', velocity_severity, robot_state, action)
        
        # Check for emergency stop
        max_severity = max(violations.values()) if violations else 0.0
        if max_severity > self.emergency_threshold:
            emergency_stop = True
            self.emergency_stop_active = True
            self.logger.critical(f"Emergency stop activated! Max severity: {max_severity}")
        
        # Update violation counts
        for violation_type in violations:
            self.violation_counts[violation_type] = self.violation_counts.get(violation_type, 0) + 1
        
        return {
            'violations': violations,
            'emergency_stop': emergency_stop,
            'max_severity': max_severity,
            'total_violations': len(violations)
        }
    
    def _log_violation(self, 
                      violation_type: str, 
                      severity: float, 
                      robot_state: np.ndarray, 
                      action: np.ndarray):
        """Log a safety violation"""
        violation = SafetyViolation(
            timestamp=time.time(),
            violation_type=violation_type,
            severity=severity,
            robot_state=robot_state.copy(),
            action=action.copy(),
            description=f"{violation_type} violation with severity {severity:.3f}"
        )
        
        self.violations_history.append(violation)
        
        # Log based on severity
        if severity > self.emergency_threshold:
            self.logger.critical(violation.description)
        elif severity > 0.5:
            self.logger.warning(violation.description)
        else:
            self.logger.info(violation.description)
    
    def reset_episode(self):
        """Reset safety monitor for new episode"""
        episode_violation_count = len([v for v in self.violations_history 
                                     if time.time() - v.timestamp < 3600])  # Last hour
        
        self.episode_violations.append(episode_violation_count)
        self.total_episodes += 1
        self.emergency_stop_active = False
        
        self.logger.info(f"Episode {self.total_episodes} completed with {episode_violation_count} violations")
    
    def get_safety_metrics(self) -> Dict[str, float]:
        """Get comprehensive safety metrics"""
        if not self.violations_history:
            return {
                'total_violations': 0,
                'violation_rate': 0.0,
                'avg_severity': 0.0,
                'emergency_stops': 0,
                'safety_score': 1.0
            }
        
        recent_violations = [v for v in self.violations_history 
                           if time.time() - v.timestamp < 3600]
        
        violation_types = {}
        total_severity = 0.0
        emergency_stops = 0
        
        for violation in recent_violations:
            violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1
            total_severity += violation.severity
            if violation.severity > self.emergency_threshold:
                emergency_stops += 1
        
        avg_severity = total_severity / len(recent_violations) if recent_violations else 0.0
        violation_rate = len(recent_violations) / max(self.total_episodes, 1)
        
        # Calculate safety score (0-1, higher is safer)
        safety_score = max(0.0, 1.0 - (violation_rate * avg_severity))
        
        metrics = {
            'total_violations': len(recent_violations),
            'violation_rate': violation_rate,
            'avg_severity': avg_severity,
            'emergency_stops': emergency_stops,
            'safety_score': safety_score,
            **{f'{vtype}_count': count for vtype, count in violation_types.items()}
        }
        
        return metrics
    
    def is_safe_action(self, action: np.ndarray, robot_state: np.ndarray) -> bool:
        """
        Predict if an action is safe before execution
        Simple heuristic-based check
        """
        # Check if action magnitude is within reasonable bounds
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 2.0:  # Arbitrary threshold
            return False
        
        # Check if robot is already in violation
        if self.emergency_stop_active:
            return False
        
        # Simple velocity prediction
        if len(robot_state) >= 6:
            predicted_vel = robot_state[3:6] + action[:3] * 0.1  # Simple integration
            if np.max(np.abs(predicted_vel)) > self.velocity_limit * 0.9:
                return False
        
        return True
    
    def get_safe_action(self, 
                       unsafe_action: np.ndarray, 
                       robot_state: np.ndarray) -> np.ndarray:
        """
        Modify unsafe action to make it safer
        """
        safe_action = unsafe_action.copy()
        
        # Clip action magnitude
        action_magnitude = np.linalg.norm(safe_action)
        if action_magnitude > 1.5:
            safe_action = safe_action / action_magnitude * 1.5
        
        # Reduce action if near safety boundaries
        safety_factor = 1.0
        
        # Check current violations and reduce action accordingly
        recent_violations = [v for v in self.violations_history 
                           if time.time() - v.timestamp < 1.0]
        
        if recent_violations:
            avg_recent_severity = np.mean([v.severity for v in recent_violations])
            safety_factor = max(0.1, 1.0 - avg_recent_severity)
        
        safe_action *= safety_factor
        
        return safe_action
    
    def export_safety_log(self, filename: str):
        """Export safety violations to file"""
        import json
        
        log_data = {
            'config': self.safety_config,
            'total_episodes': self.total_episodes,
            'violation_counts': self.violation_counts,
            'violations': [
                {
                    'timestamp': v.timestamp,
                    'type': v.violation_type,
                    'severity': v.severity,
                    'description': v.description
                } for v in self.violations_history
            ],
            'safety_metrics': self.get_safety_metrics()
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Safety log exported to {filename}")


class SafetyWrapper:
    """
    Wrapper class to integrate safety monitoring with RL environments
    """
    
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.safety_monitor = SafetyMonitor(config)
        self.safety_enabled = True
        
    def step(self, action):
        # Check if action is safe
        obs = self.env._get_observation() if hasattr(self.env, '_get_observation') else None
        
        if self.safety_enabled and obs is not None:
            if not self.safety_monitor.is_safe_action(action, obs):
                action = self.safety_monitor.get_safe_action(action, obs)
        
        # Execute environment step
        observation, reward, done, info = self.env.step(action)
        
        # Monitor safety after step
        if self.safety_enabled:
            contact_points = getattr(info, 'contact_points', [])
            applied_forces = action  # Simplified assumption
            
            safety_info = self.safety_monitor.monitor_step(
                observation, action, contact_points, applied_forces
            )
            
            info.update(safety_info)
            
            # Override done if emergency stop
            if safety_info.get('emergency_stop', False):
                done = True
                reward -= 100.0  # Large penalty for emergency stop
        
        return observation, reward, done, info
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.safety_monitor.reset_episode()
        return observation
    
    def __getattr__(self, name):
        return getattr(self.env, name)