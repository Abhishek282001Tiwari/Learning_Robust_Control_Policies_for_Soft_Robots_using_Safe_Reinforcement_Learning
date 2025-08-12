import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with both file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler for all logs
    log_file = os.path.join(log_dir, f'{name.lower()}_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Error file handler
    error_file = os.path.join(log_dir, f'{name.lower()}_errors_{datetime.now().strftime("%Y%m%d")}.log')
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logger '{name}' initialized. Logs will be saved to {log_dir}")
    
    return logger


class PerformanceLogger:
    """
    Logger specifically for tracking training performance metrics
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Setup logger
        self.logger = setup_logger(f'Performance_{experiment_name}', log_dir)
        
        # Performance tracking
        self.metrics_history = {
            'timestep': [],
            'episode': [],
            'reward': [],
            'length': [],
            'safety_violations': [],
            'safety_score': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
    
    def log_episode(self, 
                   timestep: int,
                   episode: int,
                   reward: float,
                   length: int,
                   safety_violations: int,
                   safety_score: float):
        """Log episode-level metrics"""
        
        self.metrics_history['timestep'].append(timestep)
        self.metrics_history['episode'].append(episode)
        self.metrics_history['reward'].append(reward)
        self.metrics_history['length'].append(length)
        self.metrics_history['safety_violations'].append(safety_violations)
        self.metrics_history['safety_score'].append(safety_score)
        
        self.logger.info(
            f"Episode {episode:6d} | "
            f"Timestep {timestep:8d} | "
            f"Reward {reward:8.2f} | "
            f"Length {length:4d} | "
            f"Violations {safety_violations:3d} | "
            f"Safety {safety_score:.3f}"
        )
    
    def log_training_update(self,
                          policy_loss: float,
                          value_loss: float,
                          entropy: float):
        """Log training update metrics"""
        
        self.metrics_history['policy_loss'].append(policy_loss)
        self.metrics_history['value_loss'].append(value_loss)
        self.metrics_history['entropy'].append(entropy)
    
    def log_evaluation(self,
                      timestep: int,
                      eval_reward_mean: float,
                      eval_reward_std: float,
                      eval_success_rate: float,
                      eval_safety_score: float):
        """Log evaluation metrics"""
        
        self.logger.info(
            f"=== EVALUATION at timestep {timestep} ===\n"
            f"Reward: {eval_reward_mean:.2f} ± {eval_reward_std:.2f}\n"
            f"Success Rate: {eval_success_rate:.2f}\n"
            f"Safety Score: {eval_safety_score:.3f}\n"
            f"=================================="
        )
    
    def save_metrics(self):
        """Save all metrics to file"""
        import json
        
        metrics_file = os.path.join(self.log_dir, f'{self.experiment_name}_metrics.json')
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")


class SafetyLogger:
    """
    Specialized logger for safety monitoring and violations
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Setup logger
        self.logger = setup_logger(f'Safety_{experiment_name}', log_dir)
        
        # Safety tracking
        self.safety_events = []
        self.violation_counts = {
            'collision': 0,
            'deformation': 0,
            'force': 0,
            'velocity': 0,
            'emergency_stop': 0
        }
    
    def log_violation(self,
                     timestep: int,
                     episode: int,
                     violation_type: str,
                     severity: float,
                     robot_state: str = "",
                     description: str = ""):
        """Log a safety violation"""
        
        event = {
            'timestep': timestep,
            'episode': episode,
            'type': violation_type,
            'severity': severity,
            'robot_state': robot_state,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        self.safety_events.append(event)
        self.violation_counts[violation_type] = self.violation_counts.get(violation_type, 0) + 1
        
        # Log based on severity
        if severity > 0.8:
            self.logger.critical(
                f"CRITICAL VIOLATION | Episode {episode} | "
                f"Type: {violation_type} | Severity: {severity:.3f} | "
                f"Description: {description}"
            )
        elif severity > 0.5:
            self.logger.warning(
                f"WARNING VIOLATION | Episode {episode} | "
                f"Type: {violation_type} | Severity: {severity:.3f}"
            )
        else:
            self.logger.info(
                f"Minor violation | Episode {episode} | "
                f"Type: {violation_type} | Severity: {severity:.3f}"
            )
    
    def log_emergency_stop(self,
                          timestep: int,
                          episode: int,
                          trigger_violations: list,
                          robot_state: str = ""):
        """Log emergency stop event"""
        
        self.violation_counts['emergency_stop'] += 1
        
        self.logger.critical(
            f"EMERGENCY STOP ACTIVATED | Episode {episode} | Timestep {timestep} | "
            f"Triggered by: {', '.join(trigger_violations)}"
        )
        
        # Log as safety event
        self.log_violation(
            timestep, episode, 'emergency_stop', 1.0,
            robot_state, f"Emergency stop triggered by: {', '.join(trigger_violations)}"
        )
    
    def get_safety_summary(self) -> dict:
        """Get summary of safety statistics"""
        total_violations = sum(self.violation_counts.values())
        
        if total_violations == 0:
            return {
                'total_violations': 0,
                'violation_rate': 0.0,
                'most_common_violation': 'None',
                'safety_score': 1.0
            }
        
        most_common = max(self.violation_counts, key=self.violation_counts.get)
        
        # Calculate safety score (0-1, higher is better)
        recent_events = [e for e in self.safety_events[-100:]]  # Last 100 events
        avg_severity = sum(e['severity'] for e in recent_events) / max(len(recent_events), 1)
        safety_score = max(0.0, 1.0 - avg_severity)
        
        return {
            'total_violations': total_violations,
            'violation_counts': self.violation_counts.copy(),
            'most_common_violation': most_common,
            'safety_score': safety_score,
            'recent_violations': len(recent_events)
        }
    
    def save_safety_log(self):
        """Save safety events to file"""
        import json
        
        safety_file = os.path.join(self.log_dir, f'{self.experiment_name}_safety.json')
        
        safety_data = {
            'experiment': self.experiment_name,
            'summary': self.get_safety_summary(),
            'events': self.safety_events
        }
        
        with open(safety_file, 'w') as f:
            json.dump(safety_data, f, indent=2)
        
        self.logger.info(f"Safety log saved to {safety_file}")


def setup_experiment_logging(experiment_name: str, 
                           base_log_dir: str = "experiments/logs") -> tuple:
    """
    Setup all loggers for an experiment
    
    Returns:
        Tuple of (main_logger, performance_logger, safety_logger)
    """
    log_dir = os.path.join(base_log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    main_logger = setup_logger(f'Main_{experiment_name}', log_dir)
    performance_logger = PerformanceLogger(log_dir, experiment_name)
    safety_logger = SafetyLogger(log_dir, experiment_name)
    
    main_logger.info(f"Experiment logging initialized for: {experiment_name}")
    main_logger.info(f"Log directory: {log_dir}")
    
    return main_logger, performance_logger, safety_logger