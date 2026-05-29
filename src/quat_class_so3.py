import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from .gmm_class import gmm_class
from .riemannian_gmm import RiemannianGMM, rotations_to_wxyz, wxyz_to_xyzw
from .util import optimize_tools_so3


def _write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def _first_cover_gamma(gamma, K):
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim == 1:
        gamma = gamma.reshape(-1, 1)
    return gamma[:K, :]


def _normalize_cluster_method(cluster_method):
    aliases = {
        "rgmm": "rgmm",
        "riemannian": "rgmm",
        "riemannian_gmm": "rgmm",
        "legacy": "legacy_gmm",
        "legacy_gmm": "legacy_gmm",
        "old": "legacy_gmm",
        "old_gmm": "legacy_gmm",
        "fast": "legacy_gmm",
    }
    try:
        return aliases[cluster_method.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown cluster_method '{cluster_method}'. Expected one of: {valid}") from exc


class quat_class_so3:
    def __init__(self, q_in: list, q_out: list, q_att: R, dt, K_init: int, cluster_method="rgmm") -> None:
        """
        Parameters:
        ----------
            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR

            dt:                     TIME DIFFERENCE in differentiating ORIENTATION

            K_init:                 Initial number of GAUSSIAN COMPONENTS

            cluster_method:         "rgmm" for intrinsic S^3 clustering,
                                    or "legacy_gmm"/"old_gmm" for the fast
                                    single-chart legacy clustering.
        """

        self.q_in = q_in
        self.q_out = q_out
        self.q_att = q_att

        self.dt = dt
        self.K_init = K_init
        self.cluster_method = _normalize_cluster_method(cluster_method)
        self.M = len(q_in)
        self.N = 3

        self.tol = 10E-2

        file_path = os.path.dirname(os.path.realpath(__file__))
        self.output_path = os.path.join(os.path.dirname(file_path), "output_ori_so3.json")

    def _cluster(self):
        if self.cluster_method == "rgmm":
            q_arr = rotations_to_wxyz(self.q_in)
            gmm = RiemannianGMM(self.K_init, max_iter=50, tol=1e-6, reg=1e-6, random_state=2)
            gmm.fit(q_arr)

            self.K = gmm.n_components
            self.gamma = gmm.predict_proba(q_arr).T
            gmm.K = self.K
            gmm.assignment_arr = gmm.predict(q_arr)
            gmm.gaussian_list = [
                {
                    "prior": gmm.weights_[k],
                    "mu": R.from_quat(wxyz_to_xyzw(gmm.means_[k])),
                    "sigma": gmm.covariances_[k],
                }
                for k in range(self.K)
            ]
            self.gmm = gmm
            return

        gmm = gmm_class(self.q_in, self.q_att, self.K_init)
        gamma = gmm.fit()
        self.K = gmm.K
        self.gamma = _first_cover_gamma(gamma, self.K)
        self.gmm = gmm

    def _optimize(self):
        A_ori, _ = optimize_tools_so3.optimize_ori_so3(
            self.q_in,
            self.q_out,
            self.q_att,
            self.gamma,
            self.dt,
        )
        self.A_ori = A_ori

    def begin(self):
        self._cluster()
        self._optimize()

    def sim(self, q_init, step_size, duration):
        q_test = [q_init]
        gamma_test = []
        omega_test = []

        i = 0
        max_iter = int(duration / step_size)
        while i < max_iter:
            q_in = q_test[i]

            q_next, gamma, omega = self._step(q_in, step_size)

            q_test.append(q_next)
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)

            i += 1

        final_error = np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec())
        if final_error <= self.tol:
            print("Converged within max iteration")
        else:
            print("Did not converge within max iteration")
            print("Ori norm: ", final_error)

        return q_test, np.array(gamma_test), np.array(omega_test)

    def _step(self, q_in, step_size):
        """Integrate forward by one step using SO(3) log/exp coordinates."""

        e = (q_in * self.q_att.inv()).as_rotvec()
        if abs(np.linalg.norm(e) - np.pi) < 1e-6:
            # near cut locus: scipy has already chosen a canonical log representative.
            pass

        if self.gmm is None:
            gamma = np.ones((self.K, 1)) / self.K
        else:
            gamma = self._predict_gamma(q_in)

        omega = np.zeros(3)
        for k in range(self.K):
            omega += gamma[k, 0] * self.A_ori[k] @ e

        # The Lie policy predicts omega directly, so integration uses step_size only.
        q_next = R.from_rotvec(omega * step_size) * q_in

        return q_next, gamma, omega

    def _predict_gamma(self, q_in):
        if hasattr(self.gmm, "predict_proba"):
            return self.gmm.predict_proba(rotations_to_wxyz(q_in)).T
        return _first_cover_gamma(self.gmm.logProb(q_in), self.K)

    def _logOut(self, write_json, *args):
        if hasattr(self.gmm, "weights_"):
            Prior = np.asarray(self.gmm.weights_, dtype=float).tolist()
            Mu = self.gmm.means_
            Sigma = self.gmm.covariances_
            Mu_arr = np.zeros((self.K, 4))
            Sigma_arr = np.zeros((self.K, 3, 3), dtype=np.float32)
            for k in range(self.K):
                Mu_arr[k, :] = wxyz_to_xyzw(Mu[k])
                Sigma_arr[k, :, :] = Sigma[k]
        else:
            Prior = np.asarray(self.gmm.Prior[:self.K], dtype=float)
            prior_sum = np.sum(Prior)
            if prior_sum > 0:
                Prior = Prior / prior_sum
            Prior = Prior.tolist()
            Mu = self.gmm.Mu[:self.K]
            Sigma = self.gmm.Sigma[:self.K]
            Mu_arr = np.zeros((self.K, 4))
            Sigma_arr = np.zeros((self.K, 4, 4), dtype=np.float32)
            for k in range(self.K):
                Mu_arr[k, :] = Mu[k].as_quat()
                Sigma_arr[k, :, :] = Sigma[k]

        json_output = {
            "name": "Quaternion-DS",
            "formulation": "so3_lie",
            "cluster_method": self.cluster_method,
            "K": self.K,
            "M": 3,
            "gmm_M": int(Sigma_arr.shape[1]),
            "Prior": Prior,
            "Mu": Mu_arr.ravel().tolist(),
            "Sigma": Sigma_arr.ravel().tolist(),
            "A_ori": self.A_ori.ravel().tolist(),
            "att_ori": self.q_att.as_quat().ravel().tolist(),
            "dt": self.dt,
            "q_0": self.q_in[0].as_quat().ravel().tolist(),
            "gripper_open": 0,
        }

        if write_json:
            if len(args) == 0:
                _write_json(json_output, self.output_path)
            else:
                _write_json(json_output, os.path.join(args[0], "1.json"))

        return json_output

    @classmethod
    def single_ds(cls, q_att):
        """Return a one-component SO(3) DS instance without fitting a model."""

        instance = cls.__new__(cls)

        instance.q_in = None
        instance.q_out = None
        instance.K_init = None
        instance.M = None
        instance.gamma = None
        instance.gmm = None
        instance.cluster_method = "manual"

        instance.N = 3
        instance.q_att = q_att
        instance.tol = 10E-2
        instance.K = 1
        instance.dt = 0.01
        instance.A_ori = np.tile(-0.1 * np.eye(instance.N), (instance.K, 1, 1))

        file_path = os.path.dirname(os.path.realpath(__file__))
        instance.output_path = os.path.join(os.path.dirname(file_path), "output_ori_so3.json")

        return instance
