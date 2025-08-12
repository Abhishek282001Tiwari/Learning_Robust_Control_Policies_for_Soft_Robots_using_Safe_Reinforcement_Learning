import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import time
from typing import Tuple, Dict, Any, Optional
import yaml


class SoftRobotEnv(gym.Env):
    """
    Soft robot environment using PyBullet for safe reinforcement learning.
    Supports tentacle, gripper, and locomotion robot configurations.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config_path: Optional[str] = None, render_mode: str = 'human'):
        super(SoftRobotEnv, self).__init__()
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.render_mode = render_mode
        self.robot_type = self.config['environment']['robot_type']
        self.max_episode_steps = self.config['environment']['max_episode_steps']
        self.control_freq = self.config['environment']['control_frequency']
        
        # Robot parameters
        self.segments = self.config['robot']['segments']
        self.segment_length = self.config['robot']['segment_length']
        self.radius = self.config['robot']['radius']
        self.stiffness = self.config['robot']['stiffness']
        self.damping = self.config['robot']['damping']
        self.max_force = self.config['robot']['max_force']
        
        # Safety constraints
        self.safety_config = self.config['safety']
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.config['environment']['action_dim'],), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.config['environment']['observation_dim'],),
            dtype=np.float32
        )
        
        # Initialize PyBullet
        self._init_physics()
        
        # Environment state
        self.step_count = 0
        self.robot_id = None
        self.target_pos = np.array([0.3, 0.0, 0.1])
        self.safety_violations = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if no config file provided"""
        return {
            'environment': {
                'robot_type': 'tentacle',
                'action_dim': 8,
                'observation_dim': 24,
                'max_episode_steps': 1000,
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
    
    def _init_physics(self):
        """Initialize PyBullet physics simulation"""
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.control_freq)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Setup camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.1]
        )
    
    def _create_soft_robot(self):
        """Create soft robot based on configuration"""
        if self.robot_type == 'tentacle':
            return self._create_tentacle_robot()
        elif self.robot_type == 'gripper':
            return self._create_gripper_robot()
        elif self.robot_type == 'locomotion':
            return self._create_locomotion_robot()
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")
    
    def _create_tentacle_robot(self):
        """Create a soft tentacle robot using connected rigid bodies"""
        segment_positions = []
        segment_orientations = []
        
        # Create segments along z-axis
        for i in range(self.segments):
            pos = [0, 0, (i + 0.5) * self.segment_length]
            orn = [0, 0, 0, 1]
            segment_positions.append(pos)
            segment_orientations.append(orn)
        
        # Create collision and visual shapes
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.segment_length
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.segment_length,
            rgbaColor=[0.7, 0.3, 0.3, 1.0]
        )
        
        # Create multi-body
        robot_id = p.createMultiBody(
            baseMass=self.config['robot']['mass_per_segment'],
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=segment_positions[0],
            baseOrientation=segment_orientations[0],
            linkMasses=[self.config['robot']['mass_per_segment']] * (self.segments - 1),
            linkCollisionShapeIndices=[collision_shape] * (self.segments - 1),
            linkVisualShapeIndices=[visual_shape] * (self.segments - 1),
            linkPositions=segment_positions[1:],
            linkOrientations=segment_orientations[1:],
            linkInertialFramePositions=[[0, 0, 0]] * (self.segments - 1),
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * (self.segments - 1),
            linkParentIndices=list(range(self.segments - 1)),
            linkJointTypes=[p.JOINT_SPHERICAL] * (self.segments - 1),
            linkJointAxis=[[0, 0, 1]] * (self.segments - 1)
        )
        
        return robot_id
    
    def _create_gripper_robot(self):
        """Create a soft gripper robot"""
        # Simplified gripper with two fingers
        base_pos = [0, 0, 0.05]
        
        # Create base
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0.3, 0.7, 0.3, 1.0])
        
        # Create fingers
        finger_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.03])
        finger_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.03], rgbaColor=[0.3, 0.3, 0.7, 1.0])
        
        robot_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_pos,
            linkMasses=[0.05, 0.05],
            linkCollisionShapeIndices=[finger_collision, finger_collision],
            linkVisualShapeIndices=[finger_visual, finger_visual],
            linkPositions=[[0, 0.03, 0.03], [0, -0.03, 0.03]],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC],
            linkJointAxis=[[0, 1, 0], [0, 1, 0]]
        )
        
        return robot_id
    
    def _create_locomotion_robot(self):
        """Create a soft locomotion robot"""
        # Simple worm-like robot
        return self._create_tentacle_robot()  # Use tentacle as base for now
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from the environment"""
        if self.robot_id is None:
            return np.zeros(self.observation_space.shape[0])
        
        # Get robot state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Get joint states
        num_joints = p.getNumJoints(self.robot_id)
        joint_states = []
        for i in range(num_joints):
            joint_info = p.getJointState(self.robot_id, i)
            joint_states.extend([joint_info[0], joint_info[1]])  # position, velocity
        
        # Pad or truncate to match observation dimension
        obs = np.array(list(base_pos) + list(base_orn) + list(base_vel) + list(base_ang_vel) + joint_states)
        obs = obs[:self.observation_space.shape[0]]
        if len(obs) < self.observation_space.shape[0]:
            obs = np.pad(obs, (0, self.observation_space.shape[0] - len(obs)))
        
        return obs.astype(np.float32)
    
    def _check_safety_constraints(self) -> Tuple[bool, Dict[str, float]]:
        """Check safety constraints and return violation status"""
        violations = {}
        emergency_stop = False
        
        if self.robot_id is None:
            return emergency_stop, violations
        
        # Check collision constraints
        contact_points = p.getContactPoints(self.robot_id)
        if contact_points:
            min_distance = min([pt[8] for pt in contact_points])  # contact distance
            if min_distance < self.safety_config['collision_threshold']:
                violations['collision'] = abs(min_distance)
        
        # Check velocity constraints
        _, base_ang_vel = p.getBaseVelocity(self.robot_id)
        max_vel = max([abs(v) for v in base_ang_vel])
        if max_vel > self.safety_config['velocity_limit']:
            violations['velocity'] = max_vel - self.safety_config['velocity_limit']
        
        # Check for emergency stop
        if any(v > self.safety_config['emergency_stop_threshold'] for v in violations.values()):
            emergency_stop = True
        
        return emergency_stop, violations
    
    def _calculate_reward(self, action: np.ndarray, safety_violations: Dict[str, float]) -> float:
        """Calculate reward based on task completion and safety"""
        if self.robot_id is None:
            return 0.0
        
        # Task reward (reach target)
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(np.array(base_pos) - self.target_pos)
        task_reward = -distance_to_target
        
        # Safety penalty
        safety_penalty = -sum(safety_violations.values()) * 10.0
        
        # Action smoothness reward
        action_penalty = -0.01 * np.sum(np.square(action))
        
        total_reward = task_reward + safety_penalty + action_penalty
        
        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step"""
        self.step_count += 1
        
        # Apply action to robot
        if self.robot_id is not None:
            # Scale action to force limits
            scaled_action = action * self.max_force
            
            # Apply forces to joints
            num_joints = min(p.getNumJoints(self.robot_id), len(scaled_action))
            for i in range(num_joints):
                p.setJointMotorControl2(
                    self.robot_id, i,
                    p.TORQUE_CONTROL,
                    force=scaled_action[i]
                )
        
        # Step physics simulation
        p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Check safety constraints
        emergency_stop, safety_violations = self._check_safety_constraints()
        self.safety_violations.extend(list(safety_violations.keys()))
        
        # Calculate reward
        reward = self._calculate_reward(action, safety_violations)
        
        # Check termination conditions
        done = (
            self.step_count >= self.max_episode_steps or
            emergency_stop
        )
        
        # Info dictionary
        info = {
            'safety_violations': safety_violations,
            'emergency_stop': emergency_stop,
            'step_count': self.step_count,
            'total_violations': len(self.safety_violations)
        }
        
        return observation, reward, done, info
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.safety_violations = []
        
        # Remove existing robot
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        
        # Create new robot
        self.robot_id = self._create_soft_robot()
        
        # Reset target position with some randomization
        if hasattr(self, 'np_random'):
            self.target_pos = self.np_random.uniform([0.2, -0.2, 0.05], [0.4, 0.2, 0.15])
        
        # Create target visualization
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1.0, 0.0, 0.0, 0.8]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'rgb_array':
            width, height = 640, 480
            view_matrix = p.computeViewMatrix([1, 1, 1], [0, 0, 0], [0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100)
            
            _, _, rgb_img, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix
            )
            
            return np.array(rgb_img)[:, :, :3]  # Remove alpha channel
        elif mode == 'human':
            time.sleep(1.0 / self.control_freq)
    
    def close(self):
        """Clean up the environment"""
        p.disconnect(self.physics_client)