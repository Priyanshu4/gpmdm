import numpy as np
from gpdm import GPDM
import torch
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

class GPDM_Filter:

    def __init__(self, gpdm: GPDM, dt: float = 1.0):
        """ Given a trained GPDM model, 
            this builds a model to compute likelihood of a sequence of observations
        """

        self.gpdm = gpdm
        self.gpdm.set_evaluation_mode()

        self.sigma_points = MerweScaledSigmaPoints(self.latent_dim, alpha=0.1, beta=2., kappa=1.)

        self.ukf = UKF( dim_x=self.latent_dim, 
                        dim_z=self.observation_dim, 
                        dt=1,
                        points=self.sigma_points,
                        fx=self.mean_pred_x,
                        hx=self.mean_pred_y
                       )
        
        self.reset()

    def update(self, y):
        """ Update the kalman filter with a new observation
            Returns the log likelihood of the observation given the model

            Parameters:
                y: np.array of shape (D,) where D is the observation dimension

            Returns:
                log_likelihood: float
        """

        self.ukf.update(y)
        log_likelihood = self.ukf.log_likelihood
        return log_likelihood
    
    def reset(self):
        """ Reset the filter to the initial state """

        self.ukf.x = self.gpdm.X.mean(axis=0).detach().numpy()  # Initial latent space state is the mean of the GPDM
        self.ukf.P = np.eye(self.latent_dim) * 1000             # Initial covariance is high
        self.ukf.R = np.eye(self.observation_dim) * 3           # Measurement noise (3 degrees per joint is a reasonable guess)
        self.ukf.Q = np.eye(self.latent_dim)                    # Process noise (TODO: tune this)


    def mean_pred_x(self, x):
        """
        Predict latent space mean of x_t+1 given x_t

        Parameters
        ----------

        x : input latent state matrix 

        Return
        ------

        mean_Xout_pred : mean of Xout prediction
        """    
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype = self.gpdm.dtype, device = self.gpdm.device, requires_grad=False)

            # If x is a single point, add batch dimension
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

        Xin, Xout, _ = self.gpdm.get_Xin_Xout_matrices()
        Kx_star = self.gpdm.get_x_kernel(Xin, x,False)
        mean_Xout_pred = torch.linalg.multi_dot([Xout.t(), self.gpdm.Kx_inv, Kx_star]).t()

        mean_Xout_pred = mean_Xout_pred.squeeze().cpu().detach().numpy()

        print(mean_Xout_pred)

        return mean_Xout_pred 
    
    def mean_pred_y(self, x):
        """
        Predict observation space mean of y_t given x_t

        Parameters
        ----------

        x : input latent state matrix 

        Return
        ------

        mean_Y_pred : mean of Xout prediction
        """     
        # if x is np, convert to tensor 
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype = self.gpdm.dtype, device = self.gpdm.device, requires_grad=False)

            # If x is a single point, add batch dimension
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

        Y_obs = self.gpdm.get_Y()
        Y_obs = torch.tensor(Y_obs, dtype = self.gpdm.dtype, device = self.gpdm.device)

        Ky_star = self.gpdm.get_y_kernel(self.gpdm.X, x, False)

        mean_Y_pred = torch.linalg.multi_dot([Y_obs.t(), self.gpdm.Ky_inv, Ky_star]).t()

        mean_Y_pred = mean_Y_pred.squeeze().cpu().detach().numpy()

        return mean_Y_pred + self.gpdm.meanY
    
    @property
    def latent_dim(self):
        return self.gpdm.d

    @property
    def observation_dim(self):
        return self.gpdm.D







