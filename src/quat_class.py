import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space

from .util import optimize_tools, quat_tools
from .gmm_class import gmm_class



def _write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)



def compute_ang_vel(q_k, q_kp1, dt=0.01):
    """ Compute angular velocity """

    # dq = q_k.inv() * q_kp1    # from q_k to q_kp1 in body frame
    dq = q_kp1 * q_k.inv()    # from q_k to q_kp1 in fixed frame

    dq = dq.as_rotvec() 
    w  = dq / dt
    
    return w


def map(att_vec, q):
    #4d to 3d
    att_basis = null_space(att_vec.as_quat().reshape(1, -1))
    v, _, _, _ = np.linalg.lstsq(att_basis, q, rcond=None)
    return v

def inv_map(att_vec, v):
    #3d to 4d
    att_basis = null_space(att_vec.as_quat().reshape(1, -1))
    q_tangent = att_basis @ v               
    return q_tangent

def modulation_matrix(mu_phi, kappa):
    M = (1+kappa) * R.from_rotvec(mu_phi).as_matrix() 
    return M


class quat_class:
    def __init__(self, q_in:list,  q_out:list, q_att:R, dt, K_init:int) -> None:
        """
        Parameters:
        ----------
            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR

            dt:                     TIME DIFFERENCE in differentiating ORIENTATION
            
            K_init:                 Inital number of GAUSSIAN COMPONENTS

            M:                      OBSERVATION size

            N:                      OBSERVATION dimenstion
        """

        # store parameters
        self.q_in  = q_in
        self.q_out = q_out
        self.q_att = q_att

        self.dt = dt
        self.K_init = K_init
        self.M = len(q_in)
        self.N = 4

        # simulation parameters
        self.tol = 10E-2
        # self.max_iter = 5000

        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output_ori.json')



    def _cluster(self):
        gmm = gmm_class(self.q_in, self.q_att, self.K_init)  

        self.gamma    = gmm.fit()  # (K, M)
        self.K        = gmm.K
        self.gmm      = gmm



    def _optimize(self):
        A_ori, _ = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)

        q_in_dual   = [R.from_quat(-q.as_quat()) for q in self.q_in]
        q_out_dual  = [R.from_quat(-q.as_quat()) for q in self.q_out]
        q_att_dual =  R.from_quat(-self.q_att.as_quat())
        A_ori_dual, _ = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, self.gamma)

        self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)



    def begin(self):
        self._cluster()
        self._optimize()
        # self._logOut()


    def elasticUpdate(self, new_q_in, new_q_out, gmm_struct_ori, att_ori_new):
        self.q_att = att_ori_new

        gamma = self.gmm.elasticUpdate(new_q_in, gmm_struct_ori)

        A_ori, success = optimize_tools.optimize_ori(new_q_in, new_q_out, self.q_att, gamma)

        if success:
            q_in_dual   = [R.from_quat(-q.as_quat()) for q in new_q_in]
            q_out_dual  = [R.from_quat(-q.as_quat()) for q in new_q_out]
            q_att_dual =  R.from_quat(-self.q_att.as_quat())
            A_ori_dual, success = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, gamma)
            self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)

        return success


    def sim(self, q_init, step_size, duration):
        q_test = [q_init]
        gamma_test = []
        omega_test = []
        
        # while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            # if i > self.max_iter:
            #     print("Exceed max iteration")
            #     break
        
        i = 0
        max_iter = int(duration / step_size)
        while i < max_iter:
            q_in  = q_test[i]

            q_next, gamma, omega = self._step(q_in, step_size)

            q_test.append(q_next)
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)

            i += 1
        if np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) <= self.tol:
            print("Converged within max iteration")
        else:
            print("Did not converge within max iteration")
            print("Ori norm: ", np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()))

        return  q_test, np.array(gamma_test), np.array(omega_test)
        


    def _step(self, q_in, step_size):
        """ Integrate forward by one time step """

        # read parameters
        A_ori = self.A_ori  # (2K, N, N)
        q_att = self.q_att
        K     = self.K
        gmm   = self.gmm


        # compute gamma
        if gmm is None:
            gamma = np.zeros((2*K, 1))
            dot_product = np.dot(q_in.as_quat(), q_att.as_quat())
            if dot_product >= 0:
                gamma[:K, 0] = 1.0 / K
            else:
                gamma[K:, 0] = 1.0 / K
        else:
            gamma = gmm.logProb(q_in)   # (2K, 1)


        # first cover 
        q_out_att = np.zeros((4, 1))
        q_diff  = quat_tools.riem_log(q_att, q_in)
        for k in range(K):
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = quat_tools.riem_exp(q_in, q_out_body) 
        q_out      = R.from_quat(q_out_q.reshape(4,))
        omega      = compute_ang_vel(q_in, q_out, self.dt)  


        # dual cover
        q_att_dual = R.from_quat(-q_att.as_quat())
        q_out_att_dual = np.zeros((4, 1))
        q_diff_dual  = quat_tools.riem_log(q_att_dual, q_in)
        for k in range(K):
            q_out_att_dual += gamma[self.K+k, 0] * A_ori[self.K+k] @ q_diff_dual.T
        q_out_body_dual = quat_tools.parallel_transport(q_att_dual, q_in, q_out_att_dual.T)
        q_out_q_dual    = quat_tools.riem_exp(q_in, q_out_body_dual) 
        q_out_dual      = R.from_quat(q_out_q_dual.reshape(4,))
        omega           += compute_ang_vel(q_in, q_out_dual, self.dt)  
        
        
        # propagate forward
        q_next     = R.from_rotvec(omega * step_size) * q_in  #compose in world frame
        # q_next     = q_in * R.from_rotvec(w_out * step_size)   #compose in body frame

        return q_next, gamma, omega
            
    
    def sim2(self, q_init, step_size, duration, gpr, modulation_matrix, map, inv_map):
        q_test = [q_init]
        gamma_test = []
        omega_test = []
        q_out_att_3d_list = []
        y_mean_list = []
        # while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
        #     if i > self.max_iter:
        #         print("Exceed max iteration")
        #         break
        
        i = 0
        max_iter = int(duration / step_size)
        while i < max_iter:
            q_in  = q_test[i]

            q_next, gamma, omega, q_out_att_3d, y_mean= self._step2(q_in, step_size, gpr, modulation_matrix, map, inv_map)

            q_test.append(q_next)
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)
            q_out_att_3d_list.append(q_out_att_3d)
            y_mean_list.append(y_mean[0])

            i += 1
        if np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) <= self.tol:
            print("Converged within max iteration")
        else:
            print("Did not converge within max iteration")
        
        return  q_test, np.array(gamma_test), np.array(omega_test), np.array(q_out_att_3d_list), np.array(y_mean_list)
    


    def _step2(self, q_in, step_size, gpr):
        """ Used for orientation modulation """

        # read parameters
        A_ori = self.A_ori  # (2K, N, N)
        q_att = self.q_att
        K     = self.K
        gmm   = self.gmm


        # compute gamma
        if gmm is None:
            gamma = np.zeros((2*K, 1))
            dot_product = np.dot(q_in.as_quat(), q_att.as_quat())
            if dot_product >= 0:
                gamma[:K, 0] = 1.0 / K
            else:
                gamma[K:, 0] = 1.0 / K
        else:
            gamma = gmm.logProb(q_in)   # (2K, 1)
        if np.argmax(gamma) >= K:
            print("flipping new_ori")
            q_in = R.from_quat(-q_in.as_quat())

        if gpr is None:
            M = np.eye(3)
            y_mean = np.zeros((1,3))
        elif gpr=='test':
            y_mean = np.ones((1,4))
            M = modulation_matrix(y_mean[0, 1:4], y_mean[0, 0])
        else:
            X = quat_tools.riem_log(q_att, q_in)
            y_mean, y_std = gpr.predict(X, return_std=True)
            M = modulation_matrix(y_mean[0, 1:4], y_mean[0, 0])


        # first cover 
        q_out_att = np.zeros((4, 1))
        q_diff  = quat_tools.riem_log(q_att, q_in)
        for k in range(K):
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
        # print("Original intermediate output in 4D: ", q_out_att[:, 0])
        q_out_att_3d = map(q_att, q_out_att.T[0])
        # print("Original intermediate output in 3D: ", q_out_att_3d)
        q_out_att_3d = M @ q_out_att_3d
        # print("Modified intermediate output in 3D: ", q_out_att_3d)
        q_out_att = inv_map(q_att, q_out_att_3d)
        # print("Modified intermediate output in 4D: ", q_out_att)

        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = quat_tools.riem_exp(q_in, q_out_body) 
        q_out      = R.from_quat(q_out_q.reshape(4,))
        omega      = compute_ang_vel(q_in, q_out, self.dt)  


        # dual cover
        # q_att_dual = R.from_quat(-q_att.as_quat())
        # q_out_att_dual = np.zeros((4, 1))
        # q_diff_dual  = quat_tools.riem_log(q_att_dual, q_in)
        # for k in range(K):
        #     q_out_att_dual += gamma[self.K+k, 0] * A_ori[self.K+k] @ q_diff_dual.T
        # q_out_body_dual = quat_tools.parallel_transport(q_att_dual, q_in, q_out_att_dual.T)
        # q_out_q_dual    = quat_tools.riem_exp(q_in, q_out_body_dual) 
        # q_out_dual      = R.from_quat(q_out_q_dual.reshape(4,))
        # omega           += compute_ang_vel(q_in, q_out_dual, self.dt)  
        
        
        # propagate forward
        q_next     = R.from_rotvec(omega * step_size) * q_in  #compose in world frame
        # q_next     = q_in * R.from_rotvec(w_out * step_size)   #compose in body frame


        # if np.argmax(gamma) >= K:
        #     q_next = R.from_quat(-q_next.as_quat())

        return q_next, gamma, omega, y_mean
    


    def _logOut(self, write_json, *args): 

        Prior = self.gmm.Prior
        Mu    = self.gmm.Mu
        Mu_rollout = [q_mean.as_quat() for q_mean in Mu]
        Sigma = self.gmm.Sigma

        Mu_arr      = np.zeros((2 * self.K, self.N)) 
        Sigma_arr   = np.zeros((2 * self.K, self.N, self.N), dtype=np.float32)

        for k in range(2 * self.K):
            Mu_arr[k, :] = Mu_rollout[k]
            Sigma_arr[k, :, :] = Sigma[k]

        json_output = {
            "name": "Quaternion-DS",

            "K": self.K,
            "M": 4,
            "Prior": Prior,
            "Mu": Mu_arr.ravel().tolist(),
            "Sigma": Sigma_arr.ravel().tolist(),

            'A_ori': self.A_ori.ravel().tolist(),
            'att_ori': self.q_att.as_quat().ravel().tolist(),
            "dt": self.dt,
            'q_0': self.q_in[0].as_quat().ravel().tolist(),
            "gripper_open": 0
        }

        if write_json:  
            if len(args) == 0:
                _write_json(json_output, self.output_path)
            else:
                _write_json(json_output, os.path.join(args[0], '1.json'))

        return json_output


    @classmethod
    def single_ds(cls, q_att):
        """ Returns an instance of quat_class with all attributes initialized to None """
        # Create the instance without calling __init__
        instance = cls.__new__(cls)
        
        # Manually initialize the expected attributes
        instance.q_in = None
        instance.q_out = None
        instance.K_init = None
        instance.M = None
        instance.output_path = None
        instance.gamma = None
        instance.gmm = None


        instance.N = 4  # You might keep constants like N=4
        instance.q_att = q_att
        instance.tol = 10E-2
        instance.K = 1
        instance.dt = 0.01 # modify to control the angular velocity
        instance.A_ori =  np.tile(-0.1 * np.eye(instance.N), (2 * instance.K, 1, 1)) # (2K, N, N)
        
        return instance
