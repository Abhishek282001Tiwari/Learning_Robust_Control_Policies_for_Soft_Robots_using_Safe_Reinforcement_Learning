import unittest
import numpy as np
import tempfile
import os
import sys
import yaml
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.soft_robot_env import SoftRobotEnv
from safety.safety_monitor import SafetyWrapper


class TestSoftRobotEnvironment(unittest.TestCase):
    """Test cases for the soft robot environment"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'environment': {
                'robot_type': 'tentacle',
                'action_dim': 8,
                'observation_dim': 24,
                'max_episode_steps': 100,
                'control_frequency': 50
            },
            'robot': {
                'segments': 4,
                'segment_length': 0.1,
                'radius': 0.02,
                'mass_per_segment': 0.05,
                'stiffness': 1000.0,
                'damping': 10.0,
                'max_force': 10.0
            },
            'safety': {
                'max_deformation': 0.5,
                'collision_threshold': 0.01,
                'force_limit': 15.0,
                'velocity_limit': 2.0,
                'emergency_stop_threshold': 0.8
            }
        }
    
    def tearDown(self):
        """Clean up after tests"""
        # Close any open environments
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except:
                pass
    
    @patch('pybullet.connect')
    @patch('pybullet.setGravity')
    @patch('pybullet.setTimeStep')
    def test_environment_initialization(self, mock_timestep, mock_gravity, mock_connect):
        """Test environment initialization"""
        mock_connect.return_value = 0
        
        env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
        
        # Check basic properties
        self.assertEqual(env.robot_type, 'tentacle')
        self.assertEqual(env.segments, 4)
        self.assertEqual(env.max_episode_steps, 100)
        
        # Check action and observation spaces
        self.assertEqual(env.action_space.shape[0], 8)
        self.assertEqual(env.observation_space.shape[0], 24)
        
        # Verify PyBullet calls
        mock_connect.assert_called_once()
        mock_gravity.assert_called_once_with(0, 0, -9.81)
        mock_timestep.assert_called_once_with(1.0 / 50)
        
        env.close()
    
    def test_default_config_loading(self):
        """Test default configuration when no config provided"""
        with patch('pybullet.connect'), patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv()
            
            # Check default values are loaded
            self.assertEqual(env.robot_type, 'tentacle')
            self.assertEqual(env.segments, 4)
            self.assertIsNotNone(env.config)
            
            env.close()
    
    def test_config_file_loading(self):
        """Test loading configuration from file"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.config, f)
            config_path = f.name
        
        try:
            with patch('pybullet.connect'), patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
                env = SoftRobotEnv(config_path=config_path)
                
                # Check config was loaded correctly
                self.assertEqual(env.robot_type, 'tentacle')
                self.assertEqual(env.segments, 4)
                self.assertEqual(env.stiffness, 1000.0)
                
                env.close()
        finally:
            os.unlink(config_path)
    
    @patch('pybullet.connect')
    @patch('pybullet.createMultiBody')
    @patch('pybullet.createCollisionShape')
    @patch('pybullet.createVisualShape')
    def test_robot_creation(self, mock_visual, mock_collision, mock_multibody, mock_connect):
        """Test robot creation for different types"""
        mock_connect.return_value = 0
        mock_collision.return_value = 1
        mock_visual.return_value = 2
        mock_multibody.return_value = 3
        
        robot_types = ['tentacle', 'gripper', 'locomotion']
        
        for robot_type in robot_types:
            config = self.config.copy()
            config['environment']['robot_type'] = robot_type
            
            with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
                env = SoftRobotEnv(config=config, render_mode='rgb_array')
                robot_id = env._create_soft_robot()
                
                self.assertIsNotNone(robot_id)
                mock_multibody.assert_called()
                
                env.close()
    
    @patch('pybullet.connect')
    @patch('pybullet.getBasePositionAndOrientation')
    @patch('pybullet.getBaseVelocity')
    @patch('pybullet.getNumJoints')
    @patch('pybullet.getJointState')
    def test_observation_generation(self, mock_joint_state, mock_num_joints, 
                                  mock_base_vel, mock_base_pos, mock_connect):
        """Test observation space generation"""
        mock_connect.return_value = 0
        mock_base_pos.return_value = ([0, 0, 0], [0, 0, 0, 1])
        mock_base_vel.return_value = ([0, 0, 0], [0, 0, 0])
        mock_num_joints.return_value = 3
        mock_joint_state.return_value = (0.1, 0.2, [0, 0, 0], 0)  # pos, vel, reaction, applied
        
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
            env.robot_id = 1  # Mock robot ID
            
            obs = env._get_observation()
            
            # Check observation shape
            self.assertEqual(obs.shape[0], 24)
            self.assertTrue(np.all(np.isfinite(obs)))
            
            env.close()
    
    @patch('pybullet.connect')
    @patch('pybullet.getContactPoints')
    @patch('pybullet.getBaseVelocity')
    def test_safety_constraint_checking(self, mock_base_vel, mock_contacts, mock_connect):
        """Test safety constraint checking"""
        mock_connect.return_value = 0
        mock_contacts.return_value = []  # No contacts
        mock_base_vel.return_value = ([0, 0, 0], [0.1, 0.1, 0.1])
        
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
            env.robot_id = 1
            
            emergency, violations = env._check_safety_constraints()
            
            # Should be safe with low velocities and no contacts
            self.assertFalse(emergency)
            self.assertEqual(len(violations), 0)
            
            # Test with high velocity
            mock_base_vel.return_value = ([0, 0, 0], [5.0, 5.0, 5.0])
            emergency, violations = env._check_safety_constraints()
            
            self.assertIn('velocity', violations)
            
            env.close()
    
    @patch('pybullet.connect')
    @patch('pybullet.stepSimulation')
    @patch('pybullet.setJointMotorControl2')
    def test_environment_step(self, mock_motor_control, mock_step_sim, mock_connect):
        """Test environment stepping"""
        mock_connect.return_value = 0
        
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            with patch.object(SoftRobotEnv, '_get_observation') as mock_obs:
                with patch.object(SoftRobotEnv, '_check_safety_constraints') as mock_safety:
                    mock_obs.return_value = np.zeros(24)
                    mock_safety.return_value = (False, {})
                    
                    env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
                    env.robot_id = 1
                    env.reset()
                    
                    action = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.2])
                    obs, reward, done, info = env.step(action)
                    
                    # Check return values
                    self.assertEqual(obs.shape[0], 24)
                    self.assertIsInstance(reward, (int, float))
                    self.assertIsInstance(done, bool)
                    self.assertIsInstance(info, dict)
                    
                    # Check that motor control was called
                    mock_motor_control.assert_called()
                    mock_step_sim.assert_called_once()
                    
                    env.close()
    
    @patch('pybullet.connect')
    @patch('pybullet.removeBody')
    def test_environment_reset(self, mock_remove, mock_connect):
        """Test environment reset functionality"""
        mock_connect.return_value = 0
        
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            with patch.object(SoftRobotEnv, '_create_soft_robot') as mock_create:
                with patch.object(SoftRobotEnv, '_get_observation') as mock_obs:
                    mock_create.return_value = 1
                    mock_obs.return_value = np.zeros(24)
                    
                    env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
                    
                    # First reset
                    obs1 = env.reset()
                    self.assertEqual(obs1.shape[0], 24)
                    
                    # Second reset should remove old robot
                    env.robot_id = 1
                    obs2 = env.reset()
                    
                    mock_remove.assert_called_with(1)
                    self.assertEqual(obs2.shape[0], 24)
                    
                    env.close()
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        with patch('pybullet.connect'), patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            with patch('pybullet.getBasePositionAndOrientation') as mock_pos:
                mock_pos.return_value = ([0.2, 0.1, 0.05], [0, 0, 0, 1])
                
                env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
                env.robot_id = 1
                env.target_pos = np.array([0.3, 0.0, 0.1])
                
                action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                safety_violations = {'collision': 0.5}
                
                reward = env._calculate_reward(action, safety_violations)
                
                # Reward should be negative (distance penalty + safety penalty + action penalty)
                self.assertLess(reward, 0)
                
                env.close()
    
    @patch('pybullet.connect')
    def test_render_modes(self, mock_connect):
        """Test different render modes"""
        mock_connect.return_value = 0
        
        # Test human render mode
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv(config=self.config, render_mode='human')
            self.assertEqual(env.render_mode, 'human')
            env.close()
        
        # Test rgb_array render mode
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            with patch('pybullet.getCameraImage') as mock_camera:
                mock_camera.return_value = (640, 480, np.zeros((480, 640, 4)), None, None)
                
                env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
                rgb_array = env.render(mode='rgb_array')
                
                self.assertIsInstance(rgb_array, np.ndarray)
                self.assertEqual(rgb_array.shape[2], 3)  # RGB channels
                
                env.close()


class TestSafetyWrapper(unittest.TestCase):
    """Test cases for the safety wrapper"""
    
    def setUp(self):
        """Set up test environment with safety wrapper"""
        self.config = {
            'environment': {
                'robot_type': 'tentacle',
                'action_dim': 8,
                'observation_dim': 24,
                'max_episode_steps': 100,
                'control_frequency': 50
            },
            'robot': {
                'segments': 4,
                'segment_length': 0.1,
                'radius': 0.02,
                'mass_per_segment': 0.05,
                'stiffness': 1000.0,
                'damping': 10.0,
                'max_force': 10.0
            },
            'safety': {
                'max_deformation': 0.5,
                'collision_threshold': 0.01,
                'force_limit': 15.0,
                'velocity_limit': 2.0,
                'emergency_stop_threshold': 0.8
            }
        }
    
    @patch('pybullet.connect')
    def test_safety_wrapper_initialization(self, mock_connect):
        """Test safety wrapper initialization"""
        mock_connect.return_value = 0
        
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            base_env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
            wrapped_env = SafetyWrapper(base_env, self.config)
            
            # Check that wrapper has safety monitor
            self.assertIsNotNone(wrapped_env.safety_monitor)
            self.assertTrue(wrapped_env.safety_enabled)
            
            # Check that base environment methods are accessible
            self.assertEqual(wrapped_env.action_space.shape[0], 8)
            self.assertEqual(wrapped_env.observation_space.shape[0], 24)
            
            wrapped_env.close()
    
    @patch('pybullet.connect')
    def test_safe_action_modification(self, mock_connect):
        """Test that unsafe actions are modified"""
        mock_connect.return_value = 0
        
        with patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            base_env = SoftRobotEnv(config=self.config, render_mode='rgb_array')
            wrapped_env = SafetyWrapper(base_env, self.config)
            
            # Mock unsafe action
            with patch.object(wrapped_env.safety_monitor, 'is_safe_action', return_value=False):
                with patch.object(wrapped_env.safety_monitor, 'get_safe_action') as mock_safe_action:
                    with patch.object(base_env, 'step') as mock_step:
                        with patch.object(base_env, '_get_observation', return_value=np.zeros(24)):
                            mock_safe_action.return_value = np.array([0.1] * 8)
                            mock_step.return_value = (np.zeros(24), 0, False, {})
                            
                            unsafe_action = np.array([10.0] * 8)  # Very large action
                            obs, reward, done, info = wrapped_env.step(unsafe_action)
                            
                            # Check that safe action was used
                            mock_safe_action.assert_called_once()
            
            wrapped_env.close()


class TestEnvironmentEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_invalid_robot_type(self):
        """Test handling of invalid robot type"""
        config = {
            'environment': {'robot_type': 'invalid_robot'},
            'robot': {'segments': 4},
            'safety': {}
        }
        
        with patch('pybullet.connect'), patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv(config=config, render_mode='rgb_array')
            
            # Should raise ValueError for invalid robot type
            with self.assertRaises(ValueError):
                env._create_soft_robot()
            
            env.close()
    
    def test_action_bounds_clipping(self):
        """Test that actions are properly bounded"""
        config = {
            'environment': {
                'robot_type': 'tentacle',
                'action_dim': 4,
                'observation_dim': 12,
                'max_episode_steps': 10,
                'control_frequency': 50
            },
            'robot': {'segments': 2, 'max_force': 1.0},
            'safety': {}
        }
        
        with patch('pybullet.connect'), patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv(config=config, render_mode='rgb_array')
            
            # Test that action space is properly bounded
            self.assertEqual(env.action_space.low[0], -1.0)
            self.assertEqual(env.action_space.high[0], 1.0)
            
            env.close()
    
    def test_zero_segments_error(self):
        """Test handling of zero segments"""
        config = {
            'environment': {'robot_type': 'tentacle'},
            'robot': {'segments': 0},
            'safety': {}
        }
        
        with patch('pybullet.connect'), patch('pybullet.setGravity'), patch('pybullet.setTimeStep'):
            env = SoftRobotEnv(config=config, render_mode='rgb_array')
            
            # Should handle zero segments gracefully
            try:
                robot_id = env._create_soft_robot()
                # Should either create valid robot or raise informative error
                self.assertTrue(True)  # If we get here, no crash occurred
            except Exception as e:
                # Should be informative error, not generic crash
                self.assertIsInstance(e, (ValueError, RuntimeError))
            
            env.close()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)