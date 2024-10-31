import torch
from gpdm import GPDM
import numpy as np

class GPDM_UKF:
    """
    Unscented Kalman Filter implementation for Gaussian Process Dynamical Model (GPDM)

    For the most part, the implementation follows the approach described in the paper: https://ieeexplore.ieee.org/document/4651188
    However, this paper doesn't specifically cover GPDMs, so there are some differences in the implementation of covariance matrices.
    """

    def __init__(self, gpdm: GPDM, alpha=1e-2, beta=2, kappa=1):
        """
        Initialize GP-UKF with a GPDM model and UKF parameters.

        Parameters:
        - gpdm (GPDM): an instance of the GPDM class
        - alpha (float): UKF spread parameter (usually a small positive number)
        - beta (float): usually 2 
        - kappa (float): secondary scaling parameter
        """
        self.gpdm = gpdm
        self.gpdm.set_evaluation_mode()

        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._lambda = alpha ** 2 * (self.latent_dim + kappa) - self.latent_dim
        self._gamma = torch.sqrt(torch.tensor(self.latent_dim + self._lambda))

        # Compute weights
        # wm and wc are the weights for the mean and covariance respectively
        n = self.latent_dim
        self.wm = torch.zeros(2 * n + 1)
        self.wc = torch.zeros(2 * n + 1)
        self.wm[0] = self._lambda / (n + self._lambda)
        self.wc[0] = self._lambda / (n + self._lambda) + (1 - self.alpha ** 2 + self.beta)
        self.wm[1:] = self.wc[1:] = 1 / (2 * (n + self._lambda))

        self.reset()

    def reset(self):
        """ 
        Reset the filter to the initial state 
        
        The initial state is the mean of the latent space and with a very high covariance.
        """
        self._mu = self.gpdm.X.mean(axis=0).clone().detach()
        self._sigma = torch.eye(self.latent_dim) * 1000

        self._z = None
        self._z_pred = None
        self._Q_k = None
        self._R_k = None
        self._S_k = None

    def _mean_pred_x(self, x_star):
        """
        Predict latent space mean of x_t+1 given x_t

        Parameters
        ----------

        x_star : input latent state matrix 

        Return
        ------

        mean_Xout_pred : mean of Xout prediction
        """    
        Xin, Xout, _ = self.gpdm.get_Xin_Xout_matrices()
        Kx_star = self.gpdm.get_x_kernel(Xin, x_star, False)
        mean_Xout_pred = torch.linalg.multi_dot([Xout.t(), self.gpdm.Kx_inv, Kx_star]).t()
        return mean_Xout_pred 

    def _cov_pred_x(self, x_star):
        """
        Get the covariance of the predicted latent space

        Parameters
        ----------

        x_star : tensor(dtype)
            input latent state matrix 

        Return
        ------

        diag_var_Xout_pred : variance of Xout prediction

        """

        Xin, Xout, _ = self.gpdm.get_Xin_Xout_matrices()

        # TODO: check if this is correct
        # Flg noise is set to False, but maybe it should be True
        # See the comment in the _cov_pred_y function
        Kx_star = self.gpdm.get_x_kernel(Xin, x_star,False)
   
        diag_var_Xout_pred_common = self.gpdm.get_x_diag_kernel(x_star, True) - \
            torch.sum(torch.matmul(Kx_star.t(), self.gpdm.Kx_inv) * Kx_star.t(), dim = 1)
        x_log_lambdas = torch.exp(self.gpdm.x_log_lambdas)**-2
        diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1) * x_log_lambdas.unsqueeze(0)

        return diag_var_Xout_pred
   
    def _mean_pred_y(self, x):
        """
        Predict observation space mean of z_t given x_t

        z is used interchangeably with y

        Parameters
        ----------

        x : input latent state matrix 

        Return
        ------

        mean_Y_pred : mean of Yout prediction
        """     

        Y_obs = self.gpdm.get_Y()
        Y_obs = torch.tensor(Y_obs, dtype = self.gpdm.dtype, device = self.gpdm.device)
        Ky_star = self.gpdm.get_y_kernel(self.gpdm.X, x, False)
        mean_Y_pred = torch.linalg.multi_dot([Y_obs.t(), self.gpdm.Ky_inv, Ky_star]).t()
        return mean_Y_pred 

    def _cov_pred_y(self, x_star):
        """
        Get the covariance of the predicted observation space

        Parameters
        ----------

        x_star : tensor(dtype)
            input latent state matrix 

        Return
        ------

        diag_var_Y_pred : variance of Y prediction


        """

        Y_obs = self.gpdm.get_Y()
        Y_obs = torch.tensor(Y_obs, dtype = self.gpdm.dtype, device = self.gpdm.device)

        # TODO: check if this is correct
        # Is it correct to use self.gpdm.X instead of self.gpdm.get_Xin_Xout_matrices()[0]?
        # Ideally we should be able to use flg_noise = True, however:
        # When flg_noise is True, Ky_star is a tensor of shape (n_train, n_train). This is wrong.
        # When flg_noise is False, Ky_star is a tensor of shape (n_train, 1). This is as expected.

        Ky_star = self.gpdm.get_y_kernel(self.gpdm.X, x_star, False)

        diag_var_Y_pred_common = self.gpdm.get_y_diag_kernel(x_star, True) - \
            torch.sum(torch.matmul(Ky_star.t(), self.gpdm.Ky_inv) * Ky_star.t(), dim = 1)
        y_log_lambdas = torch.exp(self.gpdm.y_log_lambdas)**-2
        diag_var_Y_pred = diag_var_Y_pred_common.unsqueeze(1) * y_log_lambdas.unsqueeze(0)

        return diag_var_Y_pred

    def _compute_sigma_points(self, mu, sigma):
        """ 
        Compute sigma points for the given mean and covariance matrix

        There are 2n+1 sigma points for n-dimensional state space.
        """        
        # Cholesky decomposition for square root of sigma
        sqrt_sigma = torch.linalg.cholesky(sigma)
        
        # Compute sigma points
        sigma_points = [mu]
        for i in range(self.latent_dim):
            sigma_points.append(mu + self._gamma * sqrt_sigma[:, i])
            sigma_points.append(mu - self._gamma * sqrt_sigma[:, i])
        return torch.stack(sigma_points)

    def update(self, z):
        """ 
        Update the filter with the new observation z

        Parameters:
        - z (tensor): observation vector

        The update steps are as follows:
        1. Compute the weights
        2. Generate sigma points
        3. Propagate sigma points through GP latent dynamics model
        4. Compute process noise using GP covariance
        5. Compute mean of the predicted latent state
        6. Compute covariance of the predicted latent state
        7. Generate new sigma points for the predicted mean latent state
        8. Propagate sigma points through the observation model
        9. Compute measurement/observation noise using GP's predicted covariance
        10. Compute the predicted observation mean
        11. Compute the observation covariance
        12. Compute the cross-covariance between latent state and observation
        13. Compute the Kalman gain
        14. Update the latent mean
        15. Update the latent covariance

        To access the updated latent mean and covariance, use the properties mu and sigma.
        To access the likelihood of the observation, use the log_likelihood function.
        """

        # If z is a not a tensor, convert it to a tensor
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=self.gpdm.dtype, device=self.gpdm.device)

        wm = self.wm
        wc = self.wc

        mu_prev = self._mu
        sigma_prev = self._sigma

        # Step 2: Generate sigma points
        sigma_points = self._compute_sigma_points(mu_prev, sigma_prev)

        # Step 3: Propagate sigma points through GP dynamics model
        propagated_points = self._mean_pred_x(sigma_points)

        # TODO: check if step 4 is correct
        # self._cov_pred_x(sigma_points) returns a tensor of shape (1, d)
        # diag_embed converts it to a diagonal matrix of shape (d, d)

        # Step 4: Compute process noise using gp covariance
        cov_x = self._cov_pred_x(mu_prev.unsqueeze(0))
        # create a matrix with covs as diagonal
        Q_k = torch.diag_embed(cov_x.squeeze(0))
    
        # Step 5: Compute mean of the predicted state
        mu_pred = torch.sum(wm.view(-1, 1) * propagated_points, dim=0)

        # Step 6: Compute covariance of the predicted state
        sigma_pred = torch.sum(wc.view(-1, 1, 1) * (propagated_points - mu_pred).unsqueeze(-1) @ (propagated_points - mu_pred).unsqueeze(1), dim=0)
        sigma_pred += Q_k 
        
        # Step 7: Generate new sigma points for the predicted mean
        sigma_points_pred = self._compute_sigma_points(mu_pred, sigma_pred)

        # Step 8: Propagate sigma points through the measurement model
        z_sigma_points = self._mean_pred_y(sigma_points_pred)

        # TODO: check if step 9 is correct
        # self._cov_pred_y(sigma_points_pred) returns a tensor of shape (n_train, D)
        # we take the mean along n_train to get a tensor of shape (1, D)
        # diag_embed converts it to a diagonal matrix of shape (D, D)

        # Step 9: Compute measurement noise R_k using GP's predicted covariance
        cov_y = self._cov_pred_y(mu_pred.unsqueeze(0))
        R_k = torch.diag_embed(cov_y.squeeze(0))

        # Step 10: Compute the predicted measurement mean
        z_pred = torch.sum(wm.view(-1, 1) * z_sigma_points, dim=0)

        # Step 11: Compute the measurement covariance S_k
        S_k = torch.sum(wc.view(-1, 1, 1) * (z_sigma_points - z_pred).unsqueeze(-1) @ (z_sigma_points - z_pred).unsqueeze(1), dim=0)
        S_k += R_k  # Add measurement noise

        # Step 12: Compute the cross-covariance between state and measurement
        sigma_xz = torch.sum(wc.view(-1, 1, 1) * (propagated_points - mu_pred).unsqueeze(-1) @ (z_sigma_points - z_pred).unsqueeze(1), dim=0)

        # Step 13: Compute the Kalman gain
        K_k = sigma_xz @ torch.inverse(S_k)

        # Step 14: Update the mean
        mu = mu_pred + K_k @ (z - z_pred)

        # Step 15: Update the covariance
        sigma = sigma_pred - K_k @ S_k @ K_k.T

        # Update values
        self._z = z
        self._mu = mu
        self._sigma = sigma
        self._z_pred = z_pred
        self._Q_k = Q_k
        self._R_k = R_k
        self._S_k = S_k

    def log_likelihood(self):
        """
        Compute the log likelihood of the most recent observation
        """
        ll = -0.5 * (torch.log(torch.det(self.S)) + (self.residual).T @ torch.inverse(self.S) @ (self.residual))
        return ll.item()
    
    @property
    def latent_dim(self):
        return self.gpdm.d

    @property
    def observation_dim(self):
        return self.gpdm.D

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def kappa(self):
        return self._kappa


    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma
    
    @property
    def z(self):
        return self._z

    @property
    def z_pred(self):
        return self._z_pred
    
    @property
    def residual(self):
        return self.z - self.z_pred

    @property
    def Q(self):
        return self._Q_k

    @property
    def R(self):
        return self._R_k

    @property
    def S(self):
        return self._S_k