import torch
from gpdm import GPDM

_LOG_2PI = torch.log(torch.tensor(2 * 3.14159265358979323846))

class GPDM_PF:
    """
    GPDM Particle Filter

    This implementation is based on https://ieeexplore.ieee.org/document/4651188
    """

    def __init__(self, gpdm: GPDM, num_particles: int = 100):
        """ Given a trained GPDM model, 
            this builds a model to compute likelihood of a sequence of observations
        """

        self.gpdm = gpdm
        self.gpdm.set_evaluation_mode()

        self.num_particles = num_particles
        self._init_particles()

    def update(self, z):
        """ Update the particle filter with a new observation
            Returns the log likelihood of the observation given the model

            Parameters:
                z: np.array of shape (D,) where D is the observation dimension
        """
        z = torch.tensor(z, dtype=self.gpdm.dtype, device=self.gpdm.device)
        self._update(z)

    def _update(self, z):
        """ Update the particle filter with a new observation
            Returns the log likelihood of the observation given the model

            Parameters:
                z: np.array of shape (D,) where D is the observation dimension
        """
        self._propogate_dynamics()
        self._update_weights(z)
        self._resample()

    def _init_particles(self):
        """ 
        Initialize particles by randomly sampling latent points from training data.
        Initialize weights uniformly.
        """
        indices = torch.randperm(self.gpdm.X.size(0))[:self.num_particles]
        self.particles = self.gpdm.X[indices].detach().clone()
        self.log_likelihoods = torch.zeros(self.num_particles, dtype=self.gpdm.dtype, device=self.gpdm.device)
        self.weights = torch.ones(self.num_particles, dtype=self.gpdm.dtype, device=self.gpdm.device) / self.num_particles

    def _propogate_dynamics(self):
        """
        Propogate the particles through the dynamics model to get the next latent state

        Parameters
        ----------
        x_star : tensor(dtype)
            input latent state matrix

        Return
        ------
        mean_Xout_pred : mean of Xout prediction
        diag_var_Xout_pred : variance of Xout prediction
        
        """ 
        mean_Xout_pred = self._mean_pred_x(self.particles)
        diag_var_Xout_pred = self._var_pred_x(self.particles)
    
        for i in range(self.num_particles):
            self.particles[i] = torch.normal(mean_Xout_pred[i], torch.sqrt(diag_var_Xout_pred[i]))

    def _update_weights(self, z):
        """
        Update the weights of the particles based on the observation

        Parameters
        ----------
        z : tensor(dtype)
            observation vector

        Return
        ------
        weights : tensor(dtype)
            updated weights of the particles
        """
        
        # propogate particles through measurement model
        mean_Y_pred = self._mean_pred_y(self.particles)
        diag_var_Y_pred = self._var_pred_y(self.particles)

        # compute log likelihood of the observation for each particle
        # since the covariance matrix is diagonal, we can compute the likelihood as a product of likelihoods
        # product of likelihoods is equivalent to sum of log likelihoods
        max_ll = torch.tensor(float('-inf'))
        for i in range(self.num_particles):

            log_likelihood_mu_term = -0.5 * torch.sum((z - mean_Y_pred[i])**2 / diag_var_Y_pred[i] + torch.log(diag_var_Y_pred[i]))
            log_likelihood_sigma_term = torch.sum(-torch.log(torch.sqrt(diag_var_Y_pred[i])))
            log_likelihood = log_likelihood_mu_term + log_likelihood_sigma_term - 0.5 * self.gpdm.D * _LOG_2PI
            self.log_likelihoods[i] = log_likelihood
            max_ll = torch.max(max_ll, log_likelihood)

        # center the weights to avoid numerical instability
        self.weights = torch.exp(self.log_likelihoods - max_ll)
        self.weights /= torch.sum(self.weights)

    def _resample(self):
        """ 
        Resample particles based on the weights
        """
        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices]

    def log_likelihood(self):
        """
        Compute the log likelihood of the most recent observation given the model

        This is computed as a weighted average of the log likelihood of each particle
        """
        log_weights = torch.log(self.weights)
        log_weighted_likelihoods = log_weights + self.log_likelihoods
        max_ll = torch.max(log_weighted_likelihoods)
        return max_ll + torch.log(torch.sum(torch.exp(log_weighted_likelihoods - max_ll)))

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

    def _var_pred_x(self, x_star):
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

    def _var_pred_y(self, x_star):
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

        Ky_star = self.gpdm.get_y_kernel(self.gpdm.X, x_star, False)

        diag_var_Y_pred_common = self.gpdm.get_y_diag_kernel(x_star, True) - \
            torch.sum(torch.matmul(Ky_star.t(), self.gpdm.Ky_inv) * Ky_star.t(), dim = 1)
        y_log_lambdas = torch.exp(self.gpdm.y_log_lambdas)**-2
        diag_var_Y_pred = diag_var_Y_pred_common.unsqueeze(1) * y_log_lambdas.unsqueeze(0)

        return diag_var_Y_pred
    
    def reset(self):
        self._init_particles()

    @property
    def latent_dim(self):
        return self.gpdm.d

    @property
    def observation_dim(self):
        return self.gpdm.D