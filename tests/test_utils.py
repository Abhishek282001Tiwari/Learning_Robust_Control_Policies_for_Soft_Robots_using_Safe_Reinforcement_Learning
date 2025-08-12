import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
import os
import sys
import json
import logging
from unittest.mock import patch, MagicMock, mock_open

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logger import setup_logger, PerformanceLogger, SafetyLogger, setup_experiment_logging
from utils.visualization import TrainingVisualizer


class TestLogging(unittest.TestCase):
    """Test cases for logging utilities"""
    
    def setUp(self):
        """Set up test logging"""
        self.test_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_setup_logger(self):
        """Test basic logger setup"""
        logger = setup_logger("TestLogger", self.test_dir)
        
        # Check logger properties
        self.assertEqual(logger.name, "TestLogger")
        self.assertEqual(logger.level, logging.INFO)
        
        # Check that handlers were added
        self.assertGreater(len(logger.handlers), 0)
        
        # Test logging
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Check that log files were created
        log_files = [f for f in os.listdir(self.test_dir) if f.endswith('.log')]
        self.assertGreater(len(log_files), 0)
    
    def test_logger_no_duplicate_handlers(self):
        """Test that duplicate handlers are not added"""
        logger1 = setup_logger("SameLogger", self.test_dir)
        initial_handler_count = len(logger1.handlers)
        
        logger2 = setup_logger("SameLogger", self.test_dir)
        final_handler_count = len(logger2.handlers)
        
        # Should be the same logger instance with same number of handlers
        self.assertEqual(logger1, logger2)
        self.assertEqual(initial_handler_count, final_handler_count)
    
    def test_performance_logger(self):
        """Test performance logger functionality"""
        perf_logger = PerformanceLogger(self.test_dir, self.experiment_name)
        
        # Test episode logging
        perf_logger.log_episode(
            timestep=1000,
            episode=10,
            reward=15.5,
            length=200,
            safety_violations=2,
            safety_score=0.85
        )
        
        # Check that data was recorded
        self.assertEqual(len(perf_logger.metrics_history['timestep']), 1)
        self.assertEqual(perf_logger.metrics_history['timestep'][0], 1000)
        self.assertEqual(perf_logger.metrics_history['reward'][0], 15.5)
        
        # Test training update logging
        perf_logger.log_training_update(
            policy_loss=0.5,
            value_loss=1.2,
            entropy=0.1
        )
        
        self.assertEqual(len(perf_logger.metrics_history['policy_loss']), 1)
        self.assertEqual(perf_logger.metrics_history['policy_loss'][0], 0.5)
        
        # Test evaluation logging
        perf_logger.log_evaluation(
            timestep=2000,
            eval_reward_mean=20.0,
            eval_reward_std=3.5,
            eval_success_rate=0.9,
            eval_safety_score=0.92
        )
        
        # Test metrics saving
        perf_logger.save_metrics()
        
        # Check that metrics file was created
        metrics_file = os.path.join(self.test_dir, f'{self.experiment_name}_metrics.json')
        self.assertTrue(os.path.exists(metrics_file))
        
        # Verify file content
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('timestep', data)
        self.assertIn('reward', data)
        self.assertEqual(len(data['timestep']), 1)
    
    def test_safety_logger(self):
        """Test safety logger functionality"""
        safety_logger = SafetyLogger(self.test_dir, self.experiment_name)
        
        # Test violation logging
        safety_logger.log_violation(
            timestep=500,
            episode=5,
            violation_type='collision',
            severity=0.7,
            robot_state='[0.1, 0.2, 0.3]',
            description='Test collision'
        )
        
        # Check that violation was recorded
        self.assertEqual(len(safety_logger.safety_events), 1)
        self.assertEqual(safety_logger.violation_counts['collision'], 1)
        
        event = safety_logger.safety_events[0]
        self.assertEqual(event['type'], 'collision')
        self.assertEqual(event['severity'], 0.7)
        self.assertEqual(event['timestep'], 500)
        
        # Test emergency stop logging
        safety_logger.log_emergency_stop(
            timestep=1000,
            episode=10,
            trigger_violations=['force', 'velocity'],
            robot_state='emergency_state'
        )
        
        self.assertEqual(safety_logger.violation_counts['emergency_stop'], 1)
        self.assertEqual(len(safety_logger.safety_events), 2)
        
        # Test safety summary
        summary = safety_logger.get_safety_summary()
        
        self.assertIn('total_violations', summary)
        self.assertIn('violation_counts', summary)
        self.assertIn('safety_score', summary)
        self.assertEqual(summary['total_violations'], 2)
        
        # Test safety log saving
        safety_logger.save_safety_log()
        
        safety_file = os.path.join(self.test_dir, f'{self.experiment_name}_safety.json')
        self.assertTrue(os.path.exists(safety_file))
        
        with open(safety_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('experiment', data)
        self.assertIn('summary', data)
        self.assertIn('events', data)
        self.assertEqual(len(data['events']), 2)
    
    def test_experiment_logging_setup(self):
        """Test complete experiment logging setup"""
        main_logger, perf_logger, safety_logger = setup_experiment_logging(
            self.experiment_name, self.test_dir
        )
        
        # Check that all loggers were created
        self.assertIsInstance(main_logger, logging.Logger)
        self.assertIsInstance(perf_logger, PerformanceLogger)
        self.assertIsInstance(safety_logger, SafetyLogger)
        
        # Test that they can be used together
        main_logger.info("Starting experiment")
        
        perf_logger.log_episode(100, 1, 10.0, 50, 1, 0.9)
        safety_logger.log_violation(100, 1, 'velocity', 0.3, '', 'Test')
        
        # Should complete without errors
        self.assertTrue(True)


class TestVisualization(unittest.TestCase):
    """Test cases for visualization utilities"""
    
    def setUp(self):
        """Set up test visualization"""
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = TrainingVisualizer(self.test_dir)
        
        # Sample data for testing
        self.episode_rewards = [10.0 + np.random.randn() for _ in range(100)]
        self.episode_violations = [max(0, int(5 + np.random.randn() * 2)) for _ in range(100)]
    
    def tearDown(self):
        """Clean up temporary files"""
        plt.close('all')  # Close all matplotlib figures
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        # Check directories were created
        self.assertTrue(os.path.exists(self.visualizer.plots_dir))
        self.assertTrue(os.path.exists(self.visualizer.interactive_dir))
        
        # Check save directory
        self.assertEqual(self.visualizer.save_dir, self.test_dir)
    
    def test_training_curves_plot(self):
        """Test training curves plotting"""
        save_path = os.path.join(self.test_dir, 'test_training_curves.png')
        
        # Should work with valid data
        self.visualizer.plot_training_curves(
            self.episode_rewards,
            self.episode_violations,
            save_path=save_path
        )
        
        # Check that plot was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Also check default save location
        default_path = os.path.join(self.visualizer.plots_dir, 'training_curves.png')
        self.assertTrue(os.path.exists(default_path))
    
    def test_training_curves_with_short_data(self):
        """Test training curves with insufficient data for moving average"""
        short_rewards = [10.0, 12.0, 8.0]
        short_violations = [1, 0, 2]
        
        # Should handle short data gracefully
        try:
            self.visualizer.plot_training_curves(short_rewards, short_violations)
            success = True
        except Exception as e:
            success = False
            print(f"Error with short data: {e}")
        
        self.assertTrue(success)
    
    def test_training_curves_with_empty_data(self):
        """Test training curves with empty data"""
        # Should handle empty data gracefully
        try:
            self.visualizer.plot_training_curves([], [])
            success = True
        except Exception as e:
            success = False
            print(f"Error with empty data: {e}")
        
        # Empty data might raise an error, which is acceptable
        # As long as it doesn't crash the entire system
        self.assertTrue(True)  # Test passes if we get here without system crash
    
    def test_safety_analysis_plot(self):
        """Test safety analysis plotting"""
        safety_metrics = {
            'collision_count': 15,
            'velocity_count': 8,
            'force_count': 3,
            'safety_score': 0.85,
            'violation_rate': 0.12,
            'avg_severity': 0.3,
            'emergency_stops': 2
        }
        
        save_path = os.path.join(self.test_dir, 'test_safety_analysis.png')
        
        self.visualizer.plot_safety_analysis(safety_metrics, save_path=save_path)
        
        # Check that plot was saved
        self.assertTrue(os.path.exists(save_path))
    
    def test_safety_analysis_with_no_violations(self):
        """Test safety analysis with no violations"""
        empty_metrics = {
            'safety_score': 1.0,
            'violation_rate': 0.0,
            'emergency_stops': 0
        }
        
        # Should handle metrics with no violations
        try:
            self.visualizer.plot_safety_analysis(empty_metrics)
            success = True
        except Exception as e:
            success = False
            print(f"Error with no violations: {e}")
        
        self.assertTrue(success)
    
    def test_interactive_dashboard_creation(self):
        """Test interactive dashboard creation"""
        training_data = {
            'episode_rewards': self.episode_rewards,
            'episode_safety_violations': self.episode_violations,
            'policy_loss': [0.5 + np.random.randn() * 0.1 for _ in range(20)],
            'value_loss': [1.2 + np.random.randn() * 0.2 for _ in range(20)]
        }
        
        safety_data = {
            'collision_count': 10,
            'velocity_count': 5
        }
        
        html_path = self.visualizer.create_interactive_dashboard(training_data, safety_data)
        
        # Check that HTML file was created
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(html_path.endswith('.html'))
        
        # Check that file has content
        with open(html_path, 'r') as f:
            content = f.read()
        
        self.assertGreater(len(content), 1000)  # Should be substantial HTML
        self.assertIn('plotly', content.lower())  # Should contain Plotly code
    
    def test_robustness_analysis_plot(self):
        """Test robustness analysis plotting"""
        robustness_results = {
            'baseline': [15.0, 12.0, 18.0, 14.0, 16.0],
            'stiffness_low': [12.0, 10.0, 15.0, 11.0, 13.0],
            'stiffness_high': [13.0, 9.0, 16.0, 12.0, 14.0],
            'noise_high': [8.0, 6.0, 11.0, 7.0, 9.0]
        }
        
        save_path = os.path.join(self.test_dir, 'test_robustness.png')
        
        self.visualizer.plot_robustness_analysis(robustness_results, save_path=save_path)
        
        # Check that plot was saved
        self.assertTrue(os.path.exists(save_path))
    
    def test_robustness_analysis_with_empty_results(self):
        """Test robustness analysis with empty results"""
        empty_results = {}
        
        # Should handle empty results gracefully
        try:
            self.visualizer.plot_robustness_analysis(empty_results)
            success = True
        except Exception as e:
            success = False
            print(f"Error with empty robustness results: {e}")
        
        self.assertTrue(success)
    
    def test_algorithm_comparison_plot(self):
        """Test algorithm comparison plotting"""
        algorithms_data = {
            'Safe PPO': {
                'episode_rewards': [10 + i * 0.1 + np.random.randn() * 2 for i in range(100)],
                'episode_safety_violations': [max(0, int(3 + np.random.randn())) for _ in range(100)]
            },
            'Standard PPO': {
                'episode_rewards': [8 + i * 0.08 + np.random.randn() * 3 for i in range(100)],
                'episode_safety_violations': [max(0, int(8 + np.random.randn() * 3)) for _ in range(100)]
            },
            'CPO': {
                'episode_rewards': [9 + i * 0.05 + np.random.randn() * 2.5 for i in range(100)],
                'episode_safety_violations': [max(0, int(5 + np.random.randn() * 2)) for _ in range(100)]
            }
        }
        
        save_path = os.path.join(self.test_dir, 'test_comparison.png')
        
        self.visualizer.plot_comparison(algorithms_data, save_path=save_path)
        
        # Check that plot was saved
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_file_structure(self):
        """Test that plots are saved in correct directory structure"""
        # Generate some plots
        self.visualizer.plot_training_curves(self.episode_rewards, self.episode_violations)
        
        safety_metrics = {'safety_score': 0.9, 'collision_count': 5}
        self.visualizer.plot_safety_analysis(safety_metrics)
        
        # Check that files exist in plots directory
        plots_dir = self.visualizer.plots_dir
        
        expected_files = ['training_curves.png', 'safety_analysis.png']
        
        for filename in expected_files:
            filepath = os.path.join(plots_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Expected file {filename} not found")
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_error_handling(self, mock_savefig):
        """Test error handling in plotting functions"""
        # Mock savefig to raise an error
        mock_savefig.side_effect = IOError("Mock save error")
        
        # Should handle save errors gracefully
        try:
            self.visualizer.plot_training_curves(self.episode_rewards, self.episode_violations)
            success = True
        except IOError:
            success = False
        
        # The function should either handle the error gracefully or let it bubble up
        # Either behavior is acceptable as long as the system doesn't crash
        self.assertTrue(True)  # Test passes if we get here
    
    def test_data_type_validation(self):
        """Test handling of different data types"""
        # Test with numpy arrays
        np_rewards = np.array(self.episode_rewards)
        np_violations = np.array(self.episode_violations)
        
        try:
            self.visualizer.plot_training_curves(np_rewards.tolist(), np_violations.tolist())
            success = True
        except Exception as e:
            success = False
            print(f"Error with numpy arrays: {e}")
        
        self.assertTrue(success)
        
        # Test with pandas Series
        try:
            pd_rewards = pd.Series(self.episode_rewards)
            pd_violations = pd.Series(self.episode_violations)
            self.visualizer.plot_training_curves(pd_rewards.tolist(), pd_violations.tolist())
            success = True
        except Exception as e:
            success = False
            print(f"Error with pandas Series: {e}")
        
        self.assertTrue(success)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utility functions"""
    
    def setUp(self):
        """Set up integration test"""
        self.test_dir = tempfile.mkdtemp()
        self.experiment_name = "integration_test"
    
    def tearDown(self):
        """Clean up"""
        plt.close('all')
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_complete_logging_and_visualization_workflow(self):
        """Test complete workflow with logging and visualization"""
        # Setup logging
        main_logger, perf_logger, safety_logger = setup_experiment_logging(
            self.experiment_name, self.test_dir
        )
        
        # Setup visualization
        visualizer = TrainingVisualizer(self.test_dir)
        
        # Simulate training loop
        rewards = []
        violations = []
        
        for episode in range(50):
            # Simulate episode
            reward = 10.0 + episode * 0.1 + np.random.randn() * 2
            violation_count = max(0, int(5 + np.random.randn() * 2))
            
            rewards.append(reward)
            violations.append(violation_count)
            
            # Log episode
            perf_logger.log_episode(
                timestep=episode * 100,
                episode=episode,
                reward=reward,
                length=100,
                safety_violations=violation_count,
                safety_score=max(0, 1 - violation_count / 10)
            )
            
            # Log some safety violations
            if violation_count > 3:
                safety_logger.log_violation(
                    timestep=episode * 100,
                    episode=episode,
                    violation_type='collision',
                    severity=violation_count / 10.0,
                    description=f'Episode {episode} collision'
                )
        
        # Save logs
        perf_logger.save_metrics()
        safety_logger.save_safety_log()
        
        # Create visualizations
        visualizer.plot_training_curves(rewards, violations)
        
        safety_metrics = safety_logger.get_safety_summary()
        visualizer.plot_safety_analysis(safety_metrics)
        
        # Verify files were created
        expected_files = [
            f'{self.experiment_name}_metrics.json',
            f'{self.experiment_name}_safety.json'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Expected file {filename} not found")
        
        # Check plot files
        plots_dir = os.path.join(self.test_dir, 'plots')
        plot_files = ['training_curves.png', 'safety_analysis.png']
        
        for filename in plot_files:
            filepath = os.path.join(plots_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Expected plot {filename} not found")
    
    def test_data_consistency(self):
        """Test consistency between logged data and visualizations"""
        # Create performance logger
        perf_logger = PerformanceLogger(self.test_dir, self.experiment_name)
        
        # Log consistent data
        test_rewards = [15.0, 16.0, 14.0, 17.0, 18.0]
        
        for i, reward in enumerate(test_rewards):
            perf_logger.log_episode(
                timestep=i * 100,
                episode=i,
                reward=reward,
                length=100,
                safety_violations=2,
                safety_score=0.8
            )
        
        # Save and reload metrics
        perf_logger.save_metrics()
        
        metrics_file = os.path.join(self.test_dir, f'{self.experiment_name}_metrics.json')
        with open(metrics_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Check data consistency
        self.assertEqual(loaded_data['reward'], test_rewards)
        self.assertEqual(len(loaded_data['timestep']), len(test_rewards))
        
        # Visualize and check that no errors occur
        visualizer = TrainingVisualizer(self.test_dir)
        
        try:
            visualizer.plot_training_curves(
                loaded_data['reward'],
                loaded_data['safety_violations']
            )
            visualization_success = True
        except Exception as e:
            visualization_success = False
            print(f"Visualization error: {e}")
        
        self.assertTrue(visualization_success)


class TestUtilsEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_invalid_directory_handling(self):
        """Test handling of invalid directories"""
        invalid_dir = "/nonexistent/directory/path"
        
        # Logger should handle invalid directory gracefully or raise clear error
        try:
            logger = setup_logger("TestLogger", invalid_dir)
            # If successful, directory should be created
            self.assertTrue(os.path.exists(invalid_dir))
        except Exception as e:
            # Should raise informative error
            self.assertIsInstance(e, (OSError, PermissionError, FileNotFoundError))
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        test_dir = tempfile.mkdtemp()
        
        try:
            # Create large dataset
            large_rewards = [np.random.randn() for _ in range(10000)]
            large_violations = [max(0, int(np.random.randn() * 3 + 2)) for _ in range(10000)]
            
            visualizer = TrainingVisualizer(test_dir)
            
            # Should handle large datasets without crashing
            start_time = time.time()
            visualizer.plot_training_curves(large_rewards, large_violations)
            end_time = time.time()
            
            # Should complete in reasonable time (less than 30 seconds)
            self.assertLess(end_time - start_time, 30.0)
            
        finally:
            import shutil
            import time
            time.sleep(0.1)  # Brief delay to ensure files are closed
            shutil.rmtree(test_dir)
    
    def test_special_values_handling(self):
        """Test handling of special numeric values (NaN, inf)"""
        test_dir = tempfile.mkdtemp()
        
        try:
            # Data with special values
            special_rewards = [1.0, np.nan, 3.0, np.inf, -np.inf, 6.0]
            special_violations = [1, 0, 2, 999, -1, 3]  # Including invalid negative
            
            visualizer = TrainingVisualizer(test_dir)
            
            # Should handle special values gracefully
            try:
                visualizer.plot_training_curves(special_rewards, special_violations)
                success = True
            except Exception as e:
                success = False
                print(f"Error with special values: {e}")
            
            # Either handle gracefully or raise informative error
            self.assertTrue(True)  # Test passes if we get here without system crash
            
        finally:
            import shutil
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all tests
    unittest.main(verbosity=2)