"""Set up the zeroth-order QP problem for stance leg control.

For details, please refer to this paper:
https://arxiv.org/abs/2009.10019
"""

import numpy as np
import quadprog  # pytype:disable=import-error
np.set_printoptions(precision=3, suppress=True)

ACC_WEIGHT = np.array([1., 1., 1., 10., 10, 1.])


class QPTorqueOptimizer():
  """QP Torque Optimizer Class."""
  def __init__(self,
               robot_mass,
               robot_inertia,
               friction_coef=0.45,
               f_min_ratio=0.1,
               f_max_ratio=10.):
    self.mpc_body_mass = robot_mass
    self.inv_mass = np.eye(3) / robot_mass
    self.inv_inertia = np.linalg.inv(robot_inertia.reshape((3, 3)))
    self.friction_coef = friction_coef
    self.f_min_ratio = f_min_ratio
    self.f_max_ratio = f_max_ratio

    # Precompute constraint matrix A
    self.A = np.zeros((24, 12))
    for leg_id in range(4):
      self.A[leg_id * 2, leg_id * 3 + 2] = 1
      self.A[leg_id * 2 + 1, leg_id * 3 + 2] = -1

    # Friction constraints
    for leg_id in range(4):
      row_id = 8 + leg_id * 4
      col_id = leg_id * 3
      self.A[row_id, col_id:col_id + 3] = np.array([1, 0, self.friction_coef])
      self.A[row_id + 1,
             col_id:col_id + 3] = np.array([-1, 0, self.friction_coef])
      self.A[row_id + 2,
             col_id:col_id + 3] = np.array([0, 1, self.friction_coef])
      self.A[row_id + 3,
             col_id:col_id + 3] = np.array([0, -1, self.friction_coef])

  def compute_mass_matrix(self, foot_positions):
    mass_mat = np.zeros((6, 12))
    mass_mat[:3] = np.concatenate([self.inv_mass] * 4, axis=1)

    for leg_id in range(4):
      x = foot_positions[leg_id]
      foot_position_skew = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]],
                                     [-x[1], x[0], 0]])
      mass_mat[3:6, leg_id * 3:leg_id * 3 +
               3] = self.inv_inertia.dot(foot_position_skew)
    return mass_mat

  def compute_constraint_matrix(self, contacts):
    f_min = self.f_min_ratio * self.mpc_body_mass * 9.8
    f_max = self.f_max_ratio * self.mpc_body_mass * 9.8
    lb = np.ones(24) * (-1e-7)
    contact_ids = np.nonzero(contacts)[0]
    lb[contact_ids * 2] = f_min
    lb[contact_ids * 2 + 1] = -f_max
    return self.A.T, lb

  def compute_objective_matrix(self, mass_matrix, desired_acc, acc_weight,
                               reg_weight):
    g = np.array([0., 0., 9.8, 0., 0., 0.])
    Q = np.diag(acc_weight)
    R = np.ones(12) * reg_weight

    quad_term = mass_matrix.T.dot(Q).dot(mass_matrix) + R
    linear_term = 1 * (g + desired_acc).T.dot(Q).dot(mass_matrix)
    return quad_term, linear_term

  def compute_contact_force(self,
                            foot_positions,
                            desired_acc,
                            contacts,
                            acc_weight=ACC_WEIGHT,
                            reg_weight=1e-4):
    mass_matrix = self.compute_mass_matrix(foot_positions)
    G, a = self.compute_objective_matrix(mass_matrix, desired_acc, acc_weight,
                                         reg_weight)
    C, b = self.compute_constraint_matrix(contacts)
    G += 1e-4 * np.eye(12)
    result = quadprog.solve_qp(G, a, C, b)
    return -result[0].reshape((4, 3))
