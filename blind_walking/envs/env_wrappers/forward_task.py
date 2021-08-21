import numpy as np


class ForwardTask(object):
    def __init__(self, num_legs=4, num_motors=12):
        """Initializes the task."""
        self._num_legs = num_legs
        self._num_motors = num_motors

        self._weight = 1
        self._forward_weight = 20
        self._lateral_weight = 21
        self._work_weight = 0.002
        self._ground_impact_weight = 0.02
        self._smoothness_weight = 0.001
        self._action_magnitude_weight = 0.07
        self._joint_speed_weight = 0.002
        self._orientation_weight = 1.5
        self._z_acceleration_weight = 2.0
        self._foot_slip_weight = 0.8

        self.current_base_velocity = np.zeros(3)
        self.last_base_velocity = np.zeros(3)
        self.current_base_rpy = np.zeros(3)
        self.last_base_rpy = np.zeros(3)
        self.current_base_rpy_rate = np.zeros(3)
        self.last_base_rpy_rate = np.zeros(3)
        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)
        self.current_motor_angles = np.zeros(num_motors)
        self.last_motor_angles = np.zeros(num_motors)
        self.current_motor_velocities = np.zeros(num_motors)
        self.last_motor_velocities = np.zeros(num_motors)
        self.current_foot_forces = np.zeros(num_legs)
        self.last_foot_forces = np.zeros(num_legs)
        self.current_motor_torques = np.zeros(num_motors)
        self.last_motor_torques = np.zeros(num_motors)
        self.current_foot_contacts = np.zeros(num_legs)
        self.last_foot_contacts = np.zeros(num_legs)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env

        self.last_base_velocity = env.robot.GetBaseVelocity()
        self.current_base_velocity = self.last_base_velocity
        self.last_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.current_base_rpy = self.last_base_rpy
        self.last_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.current_base_rpy_rate = self.last_base_rpy_rate
        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos
        self.last_motor_angles = env.robot.GetMotorAngles()
        self.current_motor_angles = self.last_motor_angles
        self.last_motor_velocities = env.robot.GetMotorVelocities()
        self.current_motor_velocities = self.last_motor_velocities
        self.last_foot_forces = env.robot.GetFootForces()
        self.current_foot_forces = self.last_foot_forces
        self.last_motor_torques = env.robot.GetMotorTorques()
        self.current_motor_torques = self.last_motor_torques
        self.last_foot_contacts = env.robot.GetFootContacts()
        self.current_foot_contacts = self.last_foot_contacts

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_velocity = self.current_base_velocity
        self.current_base_velocity = env.robot.GetBaseVelocity()
        self.last_base_rpy = self.current_base_rpy
        self.current_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.last_base_rpy_rate = self.current_base_rpy_rate
        self.current_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()
        self.last_motor_angles = self.current_motor_angles
        self.current_motor_angles = env.robot.GetMotorAngles()
        self.last_motor_velocities = self.current_motor_velocities
        self.current_motor_velocities = env.robot.GetMotorVelocities()
        self.last_foot_forces = self.current_foot_forces
        self.current_foot_forces = env.robot.GetFootForces()
        self.last_motor_torques = self.current_motor_torques
        self.current_motor_torques = env.robot.GetMotorTorques()
        self.last_foot_contacts = self.current_foot_contacts
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

        forward_reward = self._calc_reward_forward()
        lateral_reward = self._calc_reward_lateral()
        work_reward = self._calc_reward_work()
        ground_impact_reward = self._calc_reward_ground_impact()
        smoothness_reward = self._calc_reward_smoothness()
        action_magnitude_reward = self._calc_reward_action_magnitude()
        joint_speed_reward = self._calc_reward_joint_speed()
        orientation_reward = self._calc_reward_orientation()
        z_acceleration_reward = self._calc_reward_z_acceleration()
        foot_slip_reward = self._calc_reward_foot_slip()

        reward = self._forward_weight * forward_reward \
            + self._lateral_weight * lateral_reward \
            + self._work_weight * work_reward \
            + self._ground_impact_weight * ground_impact_reward \
            + self._smoothness_weight * smoothness_reward \
            + self._action_magnitude_weight * action_magnitude_reward \
            + self._joint_speed_weight * joint_speed_reward \
            + self._orientation_weight * orientation_reward \
            + self._z_acceleration_weight * z_acceleration_reward \
            + self._foot_slip_weight * foot_slip_reward

        return reward * self._weight

    def _calc_reward_forward(self, max_speed=0.35):
        """Get the forward speed reward"""
        base_velocity = self.current_base_velocity
        return min(max_speed, base_velocity[0])

    def _calc_reward_lateral(self):
        """Get the lateral movement and rotation reward"""
        lateral_movement = self.current_base_velocity[1]
        lateral_rotation = self.current_base_rpy_rate[2]
        return - pow(lateral_movement, 2) - pow(lateral_rotation, 2)

    def _calc_reward_work(self):
        """Get the work reward"""
        curr_motor_torques = self.current_motor_torques
        curr_motor_angles = self.current_motor_angles
        prev_motor_angles = self.last_motor_angles
        return - abs(np.dot(curr_motor_torques, curr_motor_angles-prev_motor_angles))

    def _calc_reward_ground_impact(self):
        """Get the ground impact reward"""
        curr_ground_impact = np.array(self.current_foot_forces)
        prev_ground_impact = np.array(self.last_foot_forces)
        diff_ground_impact = np.subtract(curr_ground_impact, prev_ground_impact)
        return - pow(np.linalg.norm(diff_ground_impact, 2), 2)

    def _calc_reward_smoothness(self):
        """Get the smoothness reward"""
        curr_motor_torques = np.array(self.current_motor_torques)
        prev_motor_torques = np.array(self.last_motor_torques)
        diff_motor_torques = np.subtract(curr_motor_torques, prev_motor_torques)
        return - pow(np.linalg.norm(diff_motor_torques, 2), 2)

    def _calc_reward_action_magnitude(self):
        """Get the action magnitude reward"""
        env = self._env
        robot = env.robot
        return - pow(np.linalg.norm(env._last_action, 2), 2)

    def _calc_reward_joint_speed(self):
        """Get the joint speed reward"""
        motor_velocities = self.current_motor_velocities
        return - pow(np.linalg.norm(motor_velocities, 2), 2)

    def _calc_reward_orientation(self):
        """Get the orientation reward"""
        orientation = self.current_base_rpy[:2]
        return - pow(np.linalg.norm(orientation, 2), 2)

    def _calc_reward_z_acceleration(self):
        """Get the z acceleration reward"""
        z_acceleration = self.current_base_velocity[2]
        return - pow(z_acceleration, 2)

    def _calc_reward_foot_slip(self):
        """Get the foot slip reward"""
        env = self._env
        robot = env.robot
        foot_contacts = self.current_foot_contacts
        foot_velocities = [0] * self._num_legs
        for leg_id in range(self._num_legs):
            jacobian = robot.ComputeJacobian(leg_id)
            # Only pick the jacobian related to joint motors
            joint_velocities = self.current_motor_velocities[leg_id *
                                                             3:(leg_id + 1) * 3]
            leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
            foot_velocities[leg_id] = leg_velocity_in_base_frame

        foot_slip = np.dot(np.diag(foot_contacts), np.array(foot_velocities))
        return - pow(np.linalg.norm(foot_slip, 2), 2)
