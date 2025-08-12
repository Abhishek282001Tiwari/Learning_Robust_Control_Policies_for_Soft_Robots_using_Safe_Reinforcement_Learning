import unittest
import numpy as np
import time
import tempfile
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety.safety_monitor import SafetyMonitor, SafetyViolation, SafetyWrapper


class TestSafetyViolation(unittest.TestCase):
    """Test cases for SafetyViolation data class"""
    
    def test_safety_violation_creation(self):
        """Test creating SafetyViolation objects"""
        timestamp = time.time()
        robot_state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.1, 0.2])
        
        violation = SafetyViolation(
            timestamp=timestamp,
            violation_type="collision",
            severity=0.8,
            robot_state=robot_state,
            action=action,
            description="Test collision violation"
        )
        
        self.assertEqual(violation.timestamp, timestamp)
        self.assertEqual(violation.violation_type, "collision")
        self.assertEqual(violation.severity, 0.8)
        self.assertTrue(np.array_equal(violation.robot_state, robot_state))
        self.assertTrue(np.array_equal(violation.action, action))
        self.assertEqual(violation.description, "Test collision violation")


class TestSafetyMonitor(unittest.TestCase):
    """Test cases for SafetyMonitor class"""
    
    def setUp(self):
        """Set up test safety monitor"""
        self.config = {
            'safety': {
                'max_deformation': 0.5,
                'collision_threshold': 0.01,
                'force_limit': 15.0,
                'velocity_limit': 2.0,
                'emergency_stop_threshold': 0.8
            }
        }
        self.monitor = SafetyMonitor(self.config)
    
    def test_safety_monitor_initialization(self):
        """Test safety monitor initialization"""
        # Check configuration loading
        self.assertEqual(self.monitor.max_deformation, 0.5)
        self.assertEqual(self.monitor.collision_threshold, 0.01)
        self.assertEqual(self.monitor.force_limit, 15.0)
        self.assertEqual(self.monitor.velocity_limit, 2.0)
        self.assertEqual(self.monitor.emergency_threshold, 0.8)
        
        # Check initial state
        self.assertEqual(len(self.monitor.violations_history), 0)
        self.assertEqual(len(self.monitor.violation_counts), 0)
        self.assertFalse(self.monitor.emergency_stop_active)
        
        # Check safety buffer zones
        self.assertIn('collision', self.monitor.safety_buffer_zones)
        self.assertIn('deformation', self.monitor.safety_buffer_zones)
        self.assertIn('force', self.monitor.safety_buffer_zones)
        self.assertIn('velocity', self.monitor.safety_buffer_zones)
    
    def test_collision_safety_check(self):
        """Test collision safety constraint checking"""
        # No contacts - should be safe
        contact_points = []
        violation, severity = self.monitor.check_collision_safety(contact_points)
        self.assertFalse(violation)
        self.assertEqual(severity, 0.0)
        
        # Contact within threshold - should be violation
        contact_points = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0.005, 0]  # contact distance = 0.005 < 0.01
        ]
        violation, severity = self.monitor.check_collision_safety(contact_points)
        self.assertTrue(violation)
        self.assertGreater(severity, 0.0)
        
        # Contact outside threshold - should be safe
        contact_points = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0]  # contact distance = 0.02 > 0.01
        ]
        violation, severity = self.monitor.check_collision_safety(contact_points)
        self.assertFalse(violation)
        self.assertEqual(severity, 0.0)
    
    def test_deformation_safety_check(self):
        """Test deformation safety constraint checking"""
        # Small deformation - should be safe
        robot_state = np.array([0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1])  # Small joint angles
        violation, severity = self.monitor.check_deformation_safety(robot_state)
        self.assertFalse(violation)
        
        # Large deformation - should be violation
        robot_state = np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 1.2, 1.5])  # Large joint angles
        violation, severity = self.monitor.check_deformation_safety(robot_state)
        self.assertTrue(violation)
        self.assertGreater(severity, 0.0)
        
        # Edge case: insufficient state data
        robot_state = np.array([0, 0, 0])  # Too few elements
        violation, severity = self.monitor.check_deformation_safety(robot_state)
        self.assertFalse(violation)  # Should handle gracefully
    
    def test_force_safety_check(self):
        """Test force safety constraint checking"""
        # Low forces - should be safe
        applied_forces = np.array([1.0, 2.0, 1.5, 0.5])
        violation, severity = self.monitor.check_force_safety(applied_forces)
        self.assertFalse(violation)
        
        # High forces - should be violation
        applied_forces = np.array([10.0, 8.0, 5.0, 3.0])  # Total = 26.0 > 15.0
        violation, severity = self.monitor.check_force_safety(applied_forces)
        self.assertTrue(violation)
        self.assertGreater(severity, 0.0)
        
        # Edge case: negative forces
        applied_forces = np.array([-5.0, 3.0, -2.0, 1.0])  # Total absolute = 11.0
        violation, severity = self.monitor.check_force_safety(applied_forces)
        self.assertFalse(violation)
    
    def test_velocity_safety_check(self):
        """Test velocity safety constraint checking"""
        # Low velocities - should be safe
        velocities = np.array([0.5, 0.3, 0.8])
        violation, severity = self.monitor.check_velocity_safety(velocities)
        self.assertFalse(violation)
        
        # High velocities - should be violation
        velocities = np.array([3.0, 1.0, 2.5])  # Max = 3.0 > 2.0
        violation, severity = self.monitor.check_velocity_safety(velocities)
        self.assertTrue(violation)
        self.assertGreater(severity, 0.0)
        
        # Edge case: negative velocities
        velocities = np.array([-2.5, 1.0, 0.5])  # Max absolute = 2.5 > 2.0
        violation, severity = self.monitor.check_velocity_safety(velocities)
        self.assertTrue(violation)
    
    def test_comprehensive_monitoring_step(self):
        """Test complete monitoring step"""
        robot_state = np.array([0, 0, 0, 0.5, 0.3, 0.8, 0, 0.2, 0.3, 0.1])
        action = np.array([1.0, 1.5, 2.0, 0.5])
        contact_points = []
        applied_forces = np.array([2.0, 3.0, 1.5, 1.0])
        
        safety_info = self.monitor.monitor_step(
            robot_state, action, contact_points, applied_forces
        )
        
        # Check return structure
        self.assertIn('violations', safety_info)
        self.assertIn('emergency_stop', safety_info)
        self.assertIn('max_severity', safety_info)
        self.assertIn('total_violations', safety_info)
        
        # Should be safe with these parameters
        self.assertFalse(safety_info['emergency_stop'])
        self.assertEqual(len(safety_info['violations']), 0)
    
    def test_emergency_stop_activation(self):
        """Test emergency stop activation"""
        # Create scenario with severe violations
        robot_state = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0, 2.0, 2.5, 3.0])  # High velocities and deformation
        action = np.array([1.0, 1.0, 1.0, 1.0])
        contact_points = []
        applied_forces = np.array([20.0, 15.0, 10.0, 5.0])  # Total = 50.0 >> 15.0
        
        safety_info = self.monitor.monitor_step(
            robot_state, action, contact_points, applied_forces
        )
        
        # Should trigger emergency stop due to high severity
        self.assertTrue(safety_info['emergency_stop'])
        self.assertTrue(self.monitor.emergency_stop_active)
        self.assertGreater(safety_info['max_severity'], self.monitor.emergency_threshold)
    
    def test_violation_logging(self):
        """Test violation logging and history"""
        initial_count = len(self.monitor.violations_history)
        
        # Trigger a violation
        robot_state = np.array([0, 0, 0, 3.0, 2.5, 1.0])  # High velocities
        action = np.array([1.0, 1.0, 1.0, 1.0])
        
        self.monitor._log_violation(
            'velocity', 0.6, robot_state, action
        )
        
        # Check that violation was logged
        self.assertEqual(len(self.monitor.violations_history), initial_count + 1)
        self.assertIn('velocity', self.monitor.violation_counts)
        self.assertEqual(self.monitor.violation_counts['velocity'], 1)
        
        # Check violation details
        violation = self.monitor.violations_history[-1]
        self.assertEqual(violation.violation_type, 'velocity')
        self.assertEqual(violation.severity, 0.6)
        self.assertTrue(np.array_equal(violation.robot_state, robot_state))
        self.assertTrue(np.array_equal(violation.action, action))
    
    def test_episode_reset(self):
        """Test episode reset functionality"""
        # Add some violations
        self.monitor.violation_counts['collision'] = 5
        self.monitor.violation_counts['velocity'] = 3
        self.monitor.emergency_stop_active = True
        
        # Reset episode
        self.monitor.reset_episode()
        
        # Emergency stop should be cleared
        self.assertFalse(self.monitor.emergency_stop_active)
        
        # Episode count should increment
        self.assertEqual(self.monitor.total_episodes, 1)
    
    def test_safety_metrics(self):
        """Test safety metrics calculation"""
        # Add some test violations
        current_time = time.time()
        for i in range(5):
            violation = SafetyViolation(
                timestamp=current_time - i * 60,  # Spread over last 5 minutes
                violation_type='collision',
                severity=0.3 + i * 0.1,
                robot_state=np.zeros(10),
                action=np.zeros(4),
                description=f'Test violation {i}'
            )
            self.monitor.violations_history.append(violation)
            self.monitor.violation_counts['collision'] = self.monitor.violation_counts.get('collision', 0) + 1
        
        self.monitor.total_episodes = 10
        
        metrics = self.monitor.get_safety_metrics()
        
        # Check metric structure
        self.assertIn('total_violations', metrics)
        self.assertIn('violation_rate', metrics)
        self.assertIn('avg_severity', metrics)
        self.assertIn('emergency_stops', metrics)
        self.assertIn('safety_score', metrics)
        self.assertIn('collision_count', metrics)
        
        # Check values
        self.assertEqual(metrics['total_violations'], 5)
        self.assertGreater(metrics['violation_rate'], 0)
        self.assertGreater(metrics['avg_severity'], 0)
        self.assertGreater(metrics['safety_score'], 0)
        self.assertLessEqual(metrics['safety_score'], 1.0)
    
    def test_safe_action_prediction(self):
        """Test safe action prediction"""
        action = np.array([0.5, 0.3, 0.8, 0.2])
        robot_state = np.zeros(10)
        
        # Should be safe with reasonable action
        self.assertTrue(self.monitor.is_safe_action(action, robot_state))
        
        # Should be unsafe with very large action
        large_action = np.array([5.0, 5.0, 5.0, 5.0])
        self.assertFalse(self.monitor.is_safe_action(large_action, robot_state))
        
        # Should be unsafe if emergency stop is active
        self.monitor.emergency_stop_active = True
        self.assertFalse(self.monitor.is_safe_action(action, robot_state))
    
    def test_safe_action_modification(self):
        """Test safe action modification"""
        unsafe_action = np.array([3.0, 2.5, 4.0, 1.5])
        robot_state = np.zeros(10)
        
        safe_action = self.monitor.get_safe_action(unsafe_action, robot_state)
        
        # Safe action should have smaller magnitude
        unsafe_magnitude = np.linalg.norm(unsafe_action)
        safe_magnitude = np.linalg.norm(safe_action)
        self.assertLess(safe_magnitude, unsafe_magnitude)
        
        # Safe action should maintain direction (roughly)
        unsafe_direction = unsafe_action / unsafe_magnitude
        safe_direction = safe_action / safe_magnitude
        cosine_similarity = np.dot(unsafe_direction, safe_direction)
        self.assertGreater(cosine_similarity, 0.5)  # Should be reasonably aligned
    
    def test_safety_log_export(self):
        """Test safety log export functionality"""
        # Add some test data
        self.monitor.violation_counts = {'collision': 3, 'velocity': 2}
        self.monitor.total_episodes = 15
        
        violation = SafetyViolation(
            timestamp=time.time(),
            violation_type='force',
            severity=0.7,
            robot_state=np.array([1, 2, 3]),
            action=np.array([0.1, 0.2]),
            description='Test force violation'
        )
        self.monitor.violations_history.append(violation)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.monitor.export_safety_log(filepath)
            
            # Check that file was created and contains expected data
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIn('config', data)
            self.assertIn('total_episodes', data)
            self.assertIn('violation_counts', data)
            self.assertIn('violations', data)
            self.assertIn('safety_metrics', data)
            
            self.assertEqual(data['total_episodes'], 15)
            self.assertEqual(data['violation_counts']['collision'], 3)
            self.assertEqual(len(data['violations']), 1)
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestSafetyWrapperIntegration(unittest.TestCase):
    """Integration tests for SafetyWrapper"""
    
    def setUp(self):
        """Set up test environment wrapper"""
        self.config = {
            'safety': {
                'max_deformation': 0.5,
                'collision_threshold': 0.01,
                'force_limit': 15.0,
                'velocity_limit': 2.0,
                'emergency_stop_threshold': 0.8
            }
        }
    
    def test_wrapper_initialization(self):
        """Test safety wrapper initialization"""
        mock_env = MagicMock()
        mock_env.action_space.shape = (4,)
        mock_env.observation_space.shape = (12,)
        
        wrapper = SafetyWrapper(mock_env, self.config)
        
        # Check wrapper properties
        self.assertIsNotNone(wrapper.safety_monitor)
        self.assertTrue(wrapper.safety_enabled)
        self.assertEqual(wrapper.env, mock_env)
    
    def test_safe_step_execution(self):
        """Test step execution with safety monitoring"""
        mock_env = MagicMock()
        mock_env._get_observation.return_value = np.zeros(12)
        mock_env.step.return_value = (np.zeros(12), 1.0, False, {})
        
        wrapper = SafetyWrapper(mock_env, self.config)
        
        action = np.array([0.1, 0.2, 0.3, 0.4])
        
        obs, reward, done, info = wrapper.step(action)
        
        # Check that step was executed
        mock_env.step.assert_called_once()
        
        # Check that safety info was added
        self.assertIn('violations', info)
        self.assertIn('emergency_stop', info)
        self.assertIn('max_severity', info)
    
    def test_unsafe_action_modification(self):
        """Test modification of unsafe actions"""
        mock_env = MagicMock()
        mock_env._get_observation.return_value = np.zeros(12)
        mock_env.step.return_value = (np.zeros(12), 1.0, False, {})
        
        wrapper = SafetyWrapper(mock_env, self.config)
        
        # Mock unsafe action detection
        with patch.object(wrapper.safety_monitor, 'is_safe_action', return_value=False):
            with patch.object(wrapper.safety_monitor, 'get_safe_action') as mock_safe_action:
                mock_safe_action.return_value = np.array([0.1, 0.1, 0.1, 0.1])
                
                unsafe_action = np.array([5.0, 5.0, 5.0, 5.0])
                obs, reward, done, info = wrapper.step(unsafe_action)
                
                # Check that safe action was requested
                mock_safe_action.assert_called_once()
    
    def test_emergency_stop_override(self):
        """Test emergency stop override behavior"""
        mock_env = MagicMock()
        mock_env._get_observation.return_value = np.zeros(12)
        mock_env.step.return_value = (np.zeros(12), 10.0, False, {})
        
        wrapper = SafetyWrapper(mock_env, self.config)
        
        # Mock emergency stop scenario
        with patch.object(wrapper.safety_monitor, 'monitor_step') as mock_monitor:
            mock_monitor.return_value = {
                'violations': {'force': 0.9},
                'emergency_stop': True,
                'max_severity': 0.9,
                'total_violations': 1
            }
            
            action = np.array([1.0, 1.0, 1.0, 1.0])
            obs, reward, done, info = wrapper.step(action)
            
            # Should override done=True and penalize reward
            self.assertTrue(done)
            self.assertLess(reward, 10.0)  # Should be penalized
            self.assertTrue(info['emergency_stop'])
    
    def test_reset_integration(self):
        """Test reset with safety monitor integration"""
        mock_env = MagicMock()
        mock_env.reset.return_value = np.zeros(12)
        
        wrapper = SafetyWrapper(mock_env, self.config)
        
        with patch.object(wrapper.safety_monitor, 'reset_episode') as mock_reset:
            obs = wrapper.reset()
            
            # Check that both env and safety monitor were reset
            mock_env.reset.assert_called_once()
            mock_reset.assert_called_once()
            
            self.assertEqual(obs.shape[0], 12)
    
    def test_attribute_forwarding(self):
        """Test that wrapper forwards attributes to base environment"""
        mock_env = MagicMock()
        mock_env.some_property = "test_value"
        mock_env.some_method.return_value = 42
        
        wrapper = SafetyWrapper(mock_env, self.config)
        
        # Test property access
        self.assertEqual(wrapper.some_property, "test_value")
        
        # Test method call
        result = wrapper.some_method()
        self.assertEqual(result, 42)
        mock_env.some_method.assert_called_once()


class TestSafetyEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_config_handling(self):
        """Test handling of empty or missing safety config"""
        empty_config = {'safety': {}}
        
        # Should use default values
        monitor = SafetyMonitor(empty_config)
        
        # Check that defaults are reasonable
        self.assertGreater(monitor.max_deformation, 0)
        self.assertGreater(monitor.force_limit, 0)
        self.assertGreater(monitor.velocity_limit, 0)
    
    def test_malformed_contact_points(self):
        """Test handling of malformed contact point data"""
        config = {'safety': {'collision_threshold': 0.01}}
        monitor = SafetyMonitor(config)
        
        # Test empty list
        violation, severity = monitor.check_collision_safety([])
        self.assertFalse(violation)
        
        # Test malformed contact points
        malformed_contacts = [[1, 2, 3]]  # Too few elements
        
        try:
            violation, severity = monitor.check_collision_safety(malformed_contacts)
            # Should handle gracefully
            self.assertIsInstance(violation, bool)
            self.assertIsInstance(severity, (int, float))
        except Exception as e:
            # If exception occurs, should be informative
            self.assertIsInstance(e, (ValueError, IndexError))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        config = {'safety': {
            'max_deformation': 1e-10,
            'force_limit': 1e10,
            'velocity_limit': 1e-5
        }}
        monitor = SafetyMonitor(config)
        
        # Test with very small values
        tiny_forces = np.array([1e-12, 1e-12, 1e-12])
        violation, severity = monitor.check_force_safety(tiny_forces)
        self.assertFalse(violation)
        
        # Test with very large values
        huge_velocities = np.array([1e6, 1e6, 1e6])
        violation, severity = monitor.check_velocity_safety(huge_velocities)
        self.assertTrue(violation)
        self.assertLessEqual(severity, 1.0)  # Should be clamped


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)