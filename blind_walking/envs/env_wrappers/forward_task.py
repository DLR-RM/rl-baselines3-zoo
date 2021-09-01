import numpy as np


class ForwardTask(object):
    def __init__(self, num_legs=4, num_motors=12):
        """Initializes the task."""
        self._num_legs = num_legs
        self._num_motors = num_motors

        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)
        self.current_motor_velocities = np.zeros(num_motors)
        self.last_motor_velocities = np.zeros(num_motors)
        self.current_motor_torques = np.zeros(num_motors)
        self.last_motor_torques = np.zeros(num_motors)
        self.current_base_orientation = np.zeros(3)
        self.last_base_orientation = np.zeros(3)
        self.current_foot_contacts = np.zeros(num_legs)
        self.last_foot_contacts = np.zeros(num_legs)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env

        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos
        self.last_motor_velocities = env.robot.GetMotorVelocities()
        self.current_motor_velocities = self.last_motor_velocities
        self.last_motor_torques = env.robot.GetMotorTorques()
        self.current_motor_torques = self.last_motor_torques
        self.last_base_orientation = env.robot.GetBaseOrientation()
        self.current_base_orientation = self.last_base_orientation
        self.last_foot_contacts = env.robot.GetFootContacts()
        self.current_foot_contacts = self.last_foot_contacts

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()
        self.last_motor_velocities = self.current_motor_velocities
        self.current_motor_velocities = env.robot.GetMotorVelocities()
        self.last_motor_torques = self.current_motor_torques
        self.current_motor_torques = env.robot.GetMotorTorques()
        self.last_base_orientation = self.current_base_orientation
        self.current_base_orientation = env.robot.GetBaseOrientation()
        self.Last_foot_contacts = self.current_foot_contacts
        self.current_foot_contacts = env.robot.GetFootContacts()

    def done(self, env):
        """Checks if the episode is over.
        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.85

    def reward(self, env):
        """Get the reward without side effects."""
        del env

        forward_reward = self.current_base_pos[0] - self.last_base_pos[0]
        # Cap the forward reward if a cap is set.
        forward_reward = min(forward_reward, 0.03)
        # Penalty for sideways translation.
        drift_reward = -abs(self.current_base_pos[1] - self.last_base_pos[1])
        # Penalty for upways translation.
        jump_reward = -abs(self.current_base_pos[2] - self.last_base_pos[2])
        # Penalty for sideways rotation of the body.
        orientation = self.current_base_orientation
        rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # Penalty for energy usage.
        energy_reward = -np.abs(
            np.dot(self.current_motor_torques,
                   self.current_motor_velocities)) * self._env._sim_time_step
        # Penalty for lost of more than two foot contacts
        contact_reward = min(sum(self.current_foot_contacts), 2) - 2

        objectives = [forward_reward, energy_reward, drift_reward, shake_reward, jump_reward, contact_reward]
        objective_weights = [1.0, 0.005, 0.001, 0.001, 0.001, 0.005]
        weighted_objectives = [o * w for o, w in zip(objectives, objective_weights)]
        reward = sum(weighted_objectives)
        return reward
