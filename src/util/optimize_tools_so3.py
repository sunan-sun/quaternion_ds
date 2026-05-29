import numpy as np


def compute_ang_vel_so3(q_k, q_kp1, dt=0.01):
    """Compute world-frame angular velocity between two Rotation objects."""

    dq = q_kp1 * q_k.inv()
    return dq.as_rotvec() / dt


def _stack_error_vectors(q_in, q_att):
    return np.vstack([(q * q_att.inv()).as_rotvec() for q in q_in])


def _stack_angular_velocities(q_in, q_out, dt):
    return np.vstack([compute_ang_vel_so3(q_k, q_kp1, dt) for q_k, q_kp1 in zip(q_in, q_out)])


def optimize_ori_so3(q_in, q_out, q_att, postProb, dt=0.01):
    """Fit SO(3)-native orientation dynamics matrices.

    The fitted policy predicts world-frame angular velocity as
    sum_k gamma_k(q) A_k log_SO3(q q_att^{-1}).
    """

    if len(q_in) != len(q_out):
        raise ValueError("q_in and q_out must contain the same number of samples")

    postProb = np.asarray(postProb, dtype=float)
    if postProb.ndim == 1:
        postProb = postProb.reshape(1, -1)
    if postProb.shape[1] != len(q_in):
        raise ValueError("postProb must have shape (K, len(q_in))")

    K, _ = postProb.shape
    N = 3
    A_res = np.zeros((K, N, N))
    success = False

    e_train = _stack_error_vectors(q_in, q_att)
    omega_train = _stack_angular_velocities(q_in, q_out, dt)

    try:
        import cvxpy as cp

        A_vars = []
        constraints = []
        for k in range(K):
            A_var = cp.Variable((N, N), symmetric=False)
            A_vars.append(A_var)
            constraints += [A_var + A_var.T << np.zeros((N, N))]

        omega_pred = 0
        for k in range(K):
            omega_pred_k = e_train @ A_vars[k].T
            weights = postProb[k, :].reshape(-1, 1)
            omega_pred += cp.multiply(weights, omega_pred_k)

        objective = cp.sum_squares(omega_train - omega_pred)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)

        success = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        success = success and all(A_var.value is not None for A_var in A_vars)
    except Exception:
        print("Retrying orientation optimization")
        return A_res, success

    if success:
        for k in range(K):
            A_res[k, :, :] = A_vars[k].value
            print("A_norm_ori", np.linalg.norm(A_res[k], "fro"))

    return A_res, success
