import numpy as np
from gpdm import GPDM
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.stats import multivariate_normal

class GDPM_Filter:

    def __init__(self, gpdm: GPDM, dt: float = 1.0):
        """ Given a trained GPDM model, 
            this builds a model to compute likelihood of a sequence of observations
        """

        self.gpdm = gpdm

        self.sigma_points = MerweScaledSigmaPoints(self.latent_dim, alpha=0.1, beta=2., kappa=1.)

        self.ukf = UKF( dim_x=self.latent_dim, 
                        dim_z=self.observation_dim, 
                        dt=1,
                        sigma_points=self.sigma_points,
                        fx=self.gpdm.mean_pred_x,
                        hx=self.gpdm.mean_pred_y
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
        
        residual = self.ukf.y
        cov = self.ukf.S

        log_likelihood = multivariate_normal.logpdf(residual, mean=np.zeros(len(residual)), cov=cov)

        return log_likelihood
    
    def reset(self):
        """ Reset the filter to the initial state """

        self.ukf.x = self.gpdm.X.mean(axis=0)               # Initial latent space state is the mean of the GPDM
        self.ukf.P = np.eye(self.latent_dim) * 1000         # Initial covariance is high
        self.ukf.R = np.eye(self.observation_dim) * 3       # Measurement noise (3 degrees per joint is a reasonable guess)
        self.ukf.Q = np.eye(self.latent_dim)                # Process noise (TODO: tune this)



    @property
    def latent_dim(self):
        return self.gpdm.d

    @property
    def observation_dim(self):
        return self.gpdm.D






