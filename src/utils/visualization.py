import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


class TrainingVisualizer:
    """
    Visualization utilities for training progress and results
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subdirectories
        self.plots_dir = os.path.join(save_dir, 'plots')
        self.interactive_dir = os.path.join(save_dir, 'interactive')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.interactive_dir, exist_ok=True)
    
    def plot_training_curves(self, 
                           episode_rewards: List[float],
                           episode_safety_violations: List[float],
                           save_path: Optional[str] = None) -> None:
        """Plot training curves for rewards and safety violations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(episode_rewards) + 1)
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(episodes, episode_rewards, alpha=0.7, color='blue', linewidth=1)
        
        # Add moving average
        if len(episode_rewards) > 10:
            window = min(100, len(episode_rewards) // 10)
            moving_avg = pd.Series(episode_rewards).rolling(window=window, center=True).mean()
            axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2, label=f'MA({window})')
            axes[0, 0].legend()
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Safety Violations
        axes[0, 1].plot(episodes, episode_safety_violations, alpha=0.7, color='red', linewidth=1)
        
        # Add moving average
        if len(episode_safety_violations) > 10:
            window = min(100, len(episode_safety_violations) // 10)
            moving_avg = pd.Series(episode_safety_violations).rolling(window=window, center=True).mean()
            axes[0, 1].plot(episodes, moving_avg, color='darkred', linewidth=2, label=f'MA({window})')
            axes[0, 1].legend()
        
        axes[0, 1].set_title('Safety Violations per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Violations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reward Distribution
        if len(episode_rewards) > 10:
            recent_rewards = episode_rewards[-min(200, len(episode_rewards)):]
            axes[1, 0].hist(recent_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].axvline(np.mean(recent_rewards), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(recent_rewards):.2f}')
            axes[1, 0].axvline(np.median(recent_rewards), color='blue', linestyle='--', 
                              label=f'Median: {np.median(recent_rewards):.2f}')
            axes[1, 0].legend()
        
        axes[1, 0].set_title('Recent Reward Distribution')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Safety Score
        safety_scores = [max(0, 1 - v/10) for v in episode_safety_violations]
        axes[1, 1].plot(episodes, safety_scores, alpha=0.7, color='green', linewidth=1)
        
        # Add moving average
        if len(safety_scores) > 10:
            window = min(100, len(safety_scores) // 10)
            moving_avg = pd.Series(safety_scores).rolling(window=window, center=True).mean()
            axes[1, 1].plot(episodes, moving_avg, color='darkgreen', linewidth=2, label=f'MA({window})')
            axes[1, 1].legend()
        
        axes[1, 1].set_title('Safety Score (Higher is Better)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Safety Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_safety_analysis(self, 
                           safety_metrics: Dict[str, Any],
                           save_path: Optional[str] = None) -> None:
        """Plot detailed safety analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Safety Analysis', fontsize=16, fontweight='bold')
        
        # Extract violation types and counts
        violation_types = []
        violation_counts = []
        
        for key, value in safety_metrics.items():
            if key.endswith('_count') and isinstance(value, (int, float)):
                vtype = key.replace('_count', '')
                violation_types.append(vtype.replace('_', ' ').title())
                violation_counts.append(value)
        
        if violation_types and violation_counts:
            # Plot 1: Violation Types Bar Chart
            axes[0, 0].bar(violation_types, violation_counts, color='red', alpha=0.7)
            axes[0, 0].set_title('Violations by Type')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Violation Types Pie Chart
            if sum(violation_counts) > 0:
                axes[0, 1].pie(violation_counts, labels=violation_types, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Violation Distribution')
        
        # Plot 3: Safety Metrics
        metric_names = []
        metric_values = []
        
        key_metrics = ['safety_score', 'violation_rate', 'avg_severity', 'emergency_stops']
        for metric in key_metrics:
            if metric in safety_metrics:
                metric_names.append(metric.replace('_', ' ').title())
                metric_values.append(safety_metrics[metric])
        
        if metric_names and metric_values:
            colors = ['green' if 'safety' in name.lower() else 'red' for name in metric_names]
            axes[1, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
            axes[1, 0].set_title('Key Safety Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Safety Score Gauge
        safety_score = safety_metrics.get('safety_score', 0)
        
        # Create a simple gauge using bar plot
        axes[1, 1].barh(['Safety Score'], [safety_score], 
                       color='green' if safety_score > 0.7 else 'orange' if safety_score > 0.3 else 'red')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_title(f'Overall Safety Score: {safety_score:.3f}')
        axes[1, 1].set_xlabel('Score (0-1)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.savefig(os.path.join(self.plots_dir, 'safety_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, 
                                   training_data: Dict[str, List],
                                   safety_data: Dict[str, Any]) -> str:
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Rewards', 'Safety Violations', 
                          'Reward Distribution', 'Safety Score',
                          'Policy Loss', 'Value Loss'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        episodes = list(range(1, len(training_data.get('episode_rewards', [])) + 1))
        
        # Training rewards
        if 'episode_rewards' in training_data:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_data['episode_rewards'],
                          mode='lines', name='Episode Rewards',
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # Safety violations
        if 'episode_safety_violations' in training_data:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_data['episode_safety_violations'],
                          mode='lines', name='Safety Violations',
                          line=dict(color='red', width=1)),
                row=1, col=2
            )
        
        # Reward distribution
        if 'episode_rewards' in training_data and len(training_data['episode_rewards']) > 10:
            recent_rewards = training_data['episode_rewards'][-200:]
            fig.add_trace(
                go.Histogram(x=recent_rewards, name='Reward Distribution',
                           marker=dict(color='green', opacity=0.7)),
                row=2, col=1
            )
        
        # Safety score
        if 'episode_safety_violations' in training_data:
            safety_scores = [max(0, 1 - v/10) for v in training_data['episode_safety_violations']]
            fig.add_trace(
                go.Scatter(x=episodes, y=safety_scores,
                          mode='lines', name='Safety Score',
                          line=dict(color='green', width=2)),
                row=2, col=2
            )
        
        # Policy and value losses (if available)
        if 'policy_loss' in training_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(training_data['policy_loss']))), 
                          y=training_data['policy_loss'],
                          mode='lines', name='Policy Loss',
                          line=dict(color='orange', width=1)),
                row=3, col=1
            )
        
        if 'value_loss' in training_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(training_data['value_loss']))), 
                          y=training_data['value_loss'],
                          mode='lines', name='Value Loss',
                          line=dict(color='purple', width=1)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Safe RL Training Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive plot
        html_path = os.path.join(self.interactive_dir, 'training_dashboard.html')
        fig.write_html(html_path)
        
        return html_path
    
    def plot_robustness_analysis(self, 
                               robustness_results: Dict[str, List[float]],
                               save_path: Optional[str] = None) -> None:
        """Plot robustness analysis across different conditions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Robustness Analysis', fontsize=16, fontweight='bold')
        
        conditions = list(robustness_results.keys())
        
        if not conditions:
            return
        
        # Plot 1: Performance across conditions
        condition_means = []
        condition_stds = []
        
        for condition in conditions:
            results = robustness_results[condition]
            condition_means.append(np.mean(results))
            condition_stds.append(np.std(results))
        
        axes[0, 0].bar(conditions, condition_means, yerr=condition_stds, 
                      capsize=5, alpha=0.7, color='blue')
        axes[0, 0].set_title('Performance Across Conditions')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of performance
        data_for_box = [robustness_results[cond] for cond in conditions]
        axes[0, 1].boxplot(data_for_box, labels=conditions)
        axes[0, 1].set_title('Performance Distribution')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Coefficient of Variation
        cv_values = [np.std(robustness_results[cond]) / np.abs(np.mean(robustness_results[cond]) + 1e-8) 
                    for cond in conditions]
        
        axes[1, 0].bar(conditions, cv_values, alpha=0.7, color='orange')
        axes[1, 0].set_title('Coefficient of Variation (Lower is More Robust)')
        axes[1, 0].set_ylabel('CV')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Success Rate (assuming negative rewards indicate failure)
        success_rates = []
        for condition in conditions:
            results = robustness_results[condition]
            success_rate = np.mean([1 if r > -50 else 0 for r in results])  # Threshold-based success
            success_rates.append(success_rate)
        
        axes[1, 1].bar(conditions, success_rates, alpha=0.7, color='green')
        axes[1, 1].set_title('Success Rate Across Conditions')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.savefig(os.path.join(self.plots_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, 
                       algorithms_data: Dict[str, Dict[str, List[float]]],
                       save_path: Optional[str] = None) -> None:
        """Compare multiple algorithms"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
        
        algorithm_names = list(algorithms_data.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithm_names)))
        
        # Plot 1: Learning curves
        for i, (alg_name, alg_data) in enumerate(algorithms_data.items()):
            if 'episode_rewards' in alg_data:
                episodes = range(1, len(alg_data['episode_rewards']) + 1)
                
                # Moving average
                window = min(50, len(alg_data['episode_rewards']) // 5)
                if window > 1:
                    moving_avg = pd.Series(alg_data['episode_rewards']).rolling(window=window).mean()
                    axes[0].plot(episodes, moving_avg, color=colors[i], 
                               linewidth=2, label=alg_name, alpha=0.8)
        
        axes[0].set_title('Learning Curves')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Final performance comparison
        final_performances = []
        for alg_name in algorithm_names:
            if 'episode_rewards' in algorithms_data[alg_name]:
                recent_rewards = algorithms_data[alg_name]['episode_rewards'][-50:]  # Last 50 episodes
                final_performances.append(np.mean(recent_rewards))
            else:
                final_performances.append(0)
        
        bars = axes[1].bar(algorithm_names, final_performances, color=colors, alpha=0.7)
        axes[1].set_title('Final Performance')
        axes[1].set_ylabel('Mean Reward (Last 50 Episodes)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_performances):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 3: Safety comparison
        safety_scores = []
        for alg_name in algorithm_names:
            if 'episode_safety_violations' in algorithms_data[alg_name]:
                recent_violations = algorithms_data[alg_name]['episode_safety_violations'][-50:]
                safety_score = np.mean([max(0, 1 - v/10) for v in recent_violations])
                safety_scores.append(safety_score)
            else:
                safety_scores.append(0)
        
        bars = axes[2].bar(algorithm_names, safety_scores, color=colors, alpha=0.7)
        axes[2].set_title('Safety Score')
        axes[2].set_ylabel('Safety Score (Higher is Better)')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, safety_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.savefig(os.path.join(self.plots_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()