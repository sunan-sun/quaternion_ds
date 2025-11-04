import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .util.quat_tools import *
from .util.plot_tools import *




def adjust_cov(cov, rel_scale=0.7, total_scale=1.5):
    vals, vecs = np.linalg.eigh(cov)
    # print("Old eigenvalues:", vals)
    mean_val = np.mean(vals)
    new_vals = (1 - rel_scale) * vals + rel_scale * mean_val
    new_vals *= total_scale
    # print("New eigenvalues:", new_vals)
    return vecs @ np.diag(new_vals) @ vecs.T



class gmm_class:
    def __init__(self, q_in:list, q_att:R, K_init:int):
        """
        Initialize a GMM class

        Parameters:
        ----------
            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
        """

        # store parameters
        self.q_in     = q_in
        self.q_att    = q_att
        self.K_init   = K_init

        self.M = len(q_in)
        self.N = 4

        # form projected states
        self.q_in_att    = riem_log(q_att, q_in)



    def fit(self):
        """ 
        Fit model to data; 
        predict and store assignment label;
        extract and store Gaussian component 
        """

        gmm = BayesianGaussianMixture(n_components=self.K_init, n_init=3, random_state=2).fit(self.q_in_att)
        assignment_arr = gmm.predict(self.q_in_att)
        unique_elements, counts = np.unique(assignment_arr, return_counts=True)
        for element, count in zip(unique_elements, counts):
            print("Current ori_element", element)
            print("has number", count)

        self._rearrange_array(assignment_arr)
        self._extract_gaussian()

        dual_gamma = self.logProb(self.q_in) # 2K by M

        return dual_gamma[:self.K, :] # K by M; always remain the first half



    def _rearrange_array(self, assignment_arr):
        """ Remove empty components and arrange components in order """
        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)   
        
        self.K = int(assignment_arr.max()+1)
        self.assignment_arr = assignment_arr



    def _extract_gaussian(self):
        """
        Extract Gaussian components from assignment labels and data

        Parameters:
        ----------
            Priors(list): K-length list of priors

            Mu(list):     K-length list of tuple: ([3, ] NumPy array, Rotation)

            Sigma(list):  K-length list of [N, N] NumPy array 
        """

        assignment_arr = self.assignment_arr

        Prior   = [0] *  (2 * self.K)
        Mu      = [R.identity()] * (2 * self.K)
        Sigma   = [np.zeros((self.N, self.N), dtype=np.float32)] * (2 * self.K)
        Sigma_gt = [np.zeros((self.N, self.N), dtype=np.float32)] * (self.K)  # for elastic update only; only first half cover needed

        gaussian_list = [] 
        dual_gaussian_list = []
        for k in range(self.K):
            q_k      = [q for index, q in enumerate(self.q_in) if assignment_arr[index]==k] 
            q_k_mean = quat_mean(q_k)

            q_diff = riem_log(q_k_mean, q_k)

            Prior[k]  = len(q_k)/(2 * self.M)
            Mu[k]     = q_k_mean
            Sigma_k   = q_diff.T @ q_diff / (len(q_k)-1)  # + 10E-6 * np.eye(self.N)
            Sigma[k]  = adjust_cov(Sigma_k)
            # Sigma[k]  = Sigma_k
            Sigma_gt[k] = q_diff.T @ q_diff / (len(q_k)-1) 

            gaussian_list.append(
                {   
                    "prior" : Prior[k],
                    "mu"    : Mu[k],
                    "sigma" : Sigma[k],
                    "rv"    : multivariate_normal(np.zeros(4), Sigma[k], allow_singular=True)
                }
            )

            # q_k_dual  = [R.from_quat(-q.as_quat()) for q in q_k]
            q_k_mean_dual     = R.from_quat(-q_k_mean.as_quat())

            # q_diff_dual = riem_log(q_k_mean_dual, q_k_dual) 
            Prior[self.K + k] = Prior[k]
            Mu[self.K + k]     = q_k_mean_dual
            # Sigma_k_dual = q_diff_dual.T @ q_diff_dual / (len(q_k_dual)-1)  + 10E-6 * np.eye(self.N)
            Sigma_k_dual = Sigma_k
            Sigma[self.K+k]  = adjust_cov(Sigma_k_dual)
            # Sigma[self.K+k]  = Sigma_k_dual


            dual_gaussian_list.append(
                {   
                    "prior" : Prior[self.K + k],
                    "mu"    : Mu[self.K + k],
                    "sigma" : Sigma[self.K+k],
                    "rv"    : multivariate_normal(np.zeros(4), Sigma[self.K+k], allow_singular=True)
                }
            )


        self.gaussian_list = gaussian_list
        self.dual_gaussian_list = dual_gaussian_list


        self.Prior  = Prior
        self.Mu     = Mu
        self.Sigma  = Sigma

        self.Sigma_gt = Sigma_gt
        

    def elasticUpdate(self, new_ori, gmm_struct):
        # self.q_in = new_ori
        # self.M = len(self.q_in)

        # Prior   = [0] *  (2 * self.K)
        # Mu      = [R.identity()] * (2 * self.K)
        # Sigma   = [np.zeros((self.N, self.N), dtype=np.float32)] * (2 * self.K)
        # Sigma_new = self.Sigma
        # Mu_new = self.Mu
        
        Prior = gmm_struct["Prior"].tolist()
        Mu = gmm_struct["Mu"]
        Sigma = gmm_struct["Sigma"]
        K = len(Prior)
    

        gaussian_list = [] 
        dual_gaussian_list = []
        for k in range(K):

            # Prior[k]  = Prior_new[k]/(2)
            # Mu[k]     = Mu_new[k]
            # Sigma_k   = Sigma_new[k]
            # Sigma[k]  = adjust_cov(Sigma_k)
            # Sigma[k]  = Sigma_k

            Sigma_k = Sigma[k]
            Sigma_k = adjust_cov(Sigma[k])
            gaussian_list.append(
                {   
                    "prior" : Prior[k]/(2.0),
                    "mu"    : Mu[k], # Rotation object
                    "sigma" : Sigma_k,
                    "rv"    : multivariate_normal(np.zeros(4), Sigma_k, allow_singular=True)
                }
            )

            # q_k_dual  = [R.from_quat(-q.as_quat()) for q in q_k]
            # q_k_mean_dual     = R.from_quat(-Mu[k].as_quat())
            # q_diff_dual = riem_log(q_k_mean_dual, q_k_dual) 
            # Prior[self.K + k] = Prior[k]
            # Mu[self.K + k]    = q_k_mean_dual
            # Sigma_k_dual     = Sigma_k
            # Sigma[self.K+k]  = adjust_cov(Sigma_k_dual)
            # Sigma[self.K+k]  = Sigma_k_dual


            dual_gaussian_list.append({   
                    "prior" : Prior[k]/(2.0),
                    "mu"    : R.from_quat(-Mu[k].as_quat()),
                    "sigma" : Sigma_k,
                    "rv"    : multivariate_normal(np.zeros(4), Sigma_k, allow_singular=True)
                }
            )


        self.gaussian_list = gaussian_list
        self.dual_gaussian_list = dual_gaussian_list

        dual_gamma = self.logProb(new_ori) # 2K by M
        assignment_arr = np.argmax(dual_gamma, axis = 0) # reverse order that we are assigning given the new gmm parameters; hence there's chance some component being empty
        unique_elements, counts = np.unique(assignment_arr, return_counts=True)
        for element, count in zip(unique_elements, counts):
            print("Current ori_element", element)
            print("has number", count)


        # to_keep = [k for k, count in zip(unique_elements, counts) if count >= 15]
        # removed_count = self.K - len(to_keep)
        # if removed_count > 0:
        #     print(f"Removing {removed_count} Gaussian components with counts less than 10.")
        #     # Filter gaussian_lists
        #     self.gaussian_list = [self.gaussian_list[k] for k in to_keep]
        #     self.dual_gaussian_list = [self.dual_gaussian_list[k] for k in to_keep]


        #     # Filter Prior, Mu, Sigma
        #     Prior = [Prior[k] for k in to_keep]
        #     Mu = Mu[to_keep]
        #     Sigma = Sigma[to_keep]
        #     K = len(to_keep)
        #     gamma = gamma[to_keep]

        #     # Update assignment_arr to map old indices to new indices
        #     mapping = {old_k: new_k for new_k, old_k in enumerate(to_keep)}
        #     assignment_arr = np.array([mapping.get(a, -1) for a in assignment_arr])
        #     # Reassign any assignments that mapped to -1 and gamma values accordingly
        #     if np.any(assignment_arr == -1):
        #         orphan_idx = np.where(assignment_arr == -1)[0]
        #         print(f"Reassigning {len(orphan_idx)} orphan points to nearest Gaussian.")
        #         x_orphans = new_ori[orphan_idx]

        #         # Compute Mahalanobis distances for each orphan to all remaining components
        #         dists = np.zeros((len(x_orphans), K))
        #         for k in range(K):
        #             mu_k = Mu[k]
        #             sigma_k = Sigma[k]
        #             inv_sigma = np.linalg.inv(sigma_k)
        #             diff = x_orphans - mu_k
        #             dists[:, k] = np.einsum('ij,ij->i', diff @ inv_sigma, diff)  # Mahalanobis

        #         nearest = np.argmin(dists, axis=1)
        #         assignment_arr[orphan_idx] = nearest

        #         gamma[:, orphan_idx] = 0.0
        #         for idx, comp in zip(orphan_idx, nearest):
        #             gamma[comp, idx] = 1.0

        # self.Prior  = Prior
        # self.Mu     = Mu
        # self.Sigma  = Sigma


        return dual_gamma[:self.K, :] # K by M; always return the first half only




    def logProb(self, q_in):
        """ Compute log probability"""
        if isinstance(q_in, list):
            logProb = np.zeros((self.K * 2, len(q_in)))
        else:
            logProb = np.zeros((self.K * 2, 1))


        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.gaussian_list[k].values())

            q_k  = riem_log(mu_k, q_in)

            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(q_k)


        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.dual_gaussian_list[k].values())

            q_k  = riem_log(mu_k, q_in)

            logProb[k+self.K, :] = np.log(prior_k) + normal_k.logpdf(q_k)


        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (self.K * 2, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

        return postProb
    