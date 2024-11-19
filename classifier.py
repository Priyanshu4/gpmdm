from gpdm import GPDM
from gpdm_filter import GPDM_Filter
import numpy as np

class ActionClassifier:

    def __init__(self, actions: list[str], dofs: int, latent_dim: int, ):
        """ Given a list of actions and a list of GPDM filters, 
            this builds a model to classify a sequence of observations
        """

        self.actions = actions

        self.latent_dim = latent_dim
        self.dofs = dofs

        dyn_back_step = 1 # Number of time steps to look back in the dynamics GP

        # Initial values for hyperparameters
        y_lambdas_init = np.ones(dofs)  # Signal standard deviation for observation GP
        y_lengthscales_init = np.ones(latent_dim)  # Lengthscales for observation GP
        y_sigma_n_init = 1e-2  # Noise standard deviation for observation GP

        x_lambdas_init = np.ones(latent_dim)  # Signal standard deviation for latent dynamics GP
        x_lengthscales_init = np.ones(dyn_back_step*latent_dim)  # Lengthscales for latent dynamics GP
        x_sigma_n_init = 1e-2  # Noise standard deviation for latent dynamics GP
        x_lin_coeff_init = np.ones(dyn_back_step*latent_dim + 1)  # Linear coefficients for latent dynamics GP


        self.gpdms : list[GPDM] = []

        for action in actions:
            gpdm = GPDM(
                D=self.dofs,
                d=self.latent_dim,
                dyn_target='full',
                dyn_back_step=dyn_back_step,
                y_lambdas_init=y_lambdas_init,
                y_lengthscales_init=y_lengthscales_init,
                y_sigma_n_init=y_sigma_n_init,
                x_lambdas_init=x_lambdas_init,
                x_lengthscales_init=x_lengthscales_init,
                x_sigma_n_init=x_sigma_n_init,
                x_lin_coeff_init=x_lin_coeff_init
            )
            self.gpdms.append(gpdm)
        
    
    def add_training_sequence(self, action: str | int, y: np.array):
        """ Add a sequence of observations to the GPDM model for a given action
        """

        if isinstance(action, str):
            action = self.actions.index(action)

        self.gpdms[action].add_data(y)

    def train_adam(self, action: str | int, epochs: int = 100):
        """ Train the GPDM model for a given action using Adam optimizer
        """

        if isinstance(action, str):
            action = self.actions.index(action)

        self.gpdms[action].train_adam(epochs=epochs)