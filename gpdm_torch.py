import torch
import gpytorch
import numpy as np
from sklearn.decomposition import PCA
from torch.optim import LBFGS

class LatentDynamicsGP(gpytorch.models.ExactGP):
    """ The latent space GP model for GPDM.
        Multi-input and multi-output GP with RBF + linear kernel.
    """

    def __init__(self, train_x, train_y, likelihood, num_outputs):
        super(LatentDynamicsGP, self).__init__(train_x, train_y, likelihood)
        
       # Mean and covariance modules for the input space
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Define the RBF kernel for the input covariance
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])  # RBF kernel for input space
        )
        
        # Define a MultitaskKernel to combine input and task covariance
        self.multitask_kernel = gpytorch.kernels.MultitaskKernel(
            self.covar_module.base_kernel, num_tasks=num_outputs, rank=1
        )
        
        # Number of output dimensions (tasks)
        self.num_outputs = num_outputs

    def forward(self, x):
        # Compute the mean over the input space
        mean_x = self.mean_module(x)
        
        # Compute the covariance using the MultitaskKernel
        # This will combine the input covariance (RBF) and task covariance (modeled by the MultitaskKernel)
        covar_x = self.multitask_kernel(x)
        
        # Return the MultitaskMultivariateNormal distribution
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class ObservationGP(gpytorch.models.ExactGP):
    """ 
    The observation GP model for GPDM.
    Multi-input and multi-output GP with RBF kernel.
    Each output dimension (joint angle) has its own GP model, 
    and the outputs are correlated.
    """

    def __init__(self, train_x, train_y, likelihood, num_outputs):
        super(ObservationGP, self).__init__(train_x, train_y, likelihood)
        
        # Mean and covariance modules for the input space
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Define the RBF kernel for the input covariance
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])  # RBF kernel for input space
        )
        
        # Define a MultitaskKernel to combine input and task covariance
        self.multitask_kernel = gpytorch.kernels.MultitaskKernel(
            self.covar_module.base_kernel, num_tasks=num_outputs, rank=1
        )
        
        # Number of output dimensions (tasks)
        self.num_outputs = num_outputs

    def forward(self, x):
        # Compute the mean over the input space
        mean_x = self.mean_module(x)
        
        # Compute the covariance using the MultitaskKernel
        # This will combine the input covariance (RBF) and task covariance (modeled by the MultitaskKernel)
        covar_x = self.multitask_kernel(x)
        
        # Return the MultitaskMultivariateNormal distribution
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class GPDM:
    def __init__(self, Y, latent_dim=3, device='cpu'):
        """
        Gaussian Process Dynamical Model (GPDM).
        :param Y: High-dimensional observed data [T x Y_dim]
        :param latent_dim: Dimensionality of the latent space
        :param device: 'cpu' or 'cuda' depending on availability
        """
        self.device = torch.device(device)
        self.Y = torch.tensor(Y, dtype=torch.float).to(self.device)
        self.T, self.Y_dim = self.Y.shape
        self.latent_dim = latent_dim

        # Step 1: Initialize latent space using PCA
        pca = PCA(n_components=self.latent_dim)
        X_init = pca.fit_transform(self.Y.cpu().numpy())  # Initialize X using PCA
        self.X = torch.tensor(X_init, dtype=torch.float).to(self.device).requires_grad_(True)  # Latent variables

        # Step 2: Define observation GPs (multi-output) and dynamics GP using gpytorch
        self.observation_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.Y_dim).to(self.device)
        self.observation_gp = ObservationGP(self.X, self.Y, self.observation_likelihood, num_outputs=self.Y_dim).to(self.device)

        self.dynamics_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.latent_dim).to(self.device)
        self.dynamics_gp = LatentDynamicsGP(self.X[:-1], self.X[1:], self.dynamics_likelihood, num_outputs=self.latent_dim).to(self.device)

        # Combine parameters: Include self.X and GP parameters in the optimizer
        self.parameters = [{'params': self.X}] + \
                          [{'params': self.observation_gp.parameters()}] + \
                          [{'params': self.dynamics_gp.parameters()}]

        # Initialize the Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.01)

    def _objective_function(self):
        """
        Objective function for MAP estimation.
        Computes the negative log-posterior.
        """
        # Update the GP models with current latent variables X
        self.observation_gp.set_train_data(self.X, self.Y, strict=False)
        self.dynamics_gp.set_train_data(self.X[:-1], self.X[1:], strict=False)

        # Set models to training mode
        self.observation_gp.train()
        self.dynamics_gp.train()

        # Compute the log marginal likelihood for both GPs
        observation_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.observation_likelihood, self.observation_gp)
        total_observation_ll = observation_mll(self.observation_gp(self.X), self.Y)

        dynamics_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.dynamics_likelihood, self.dynamics_gp)
        dynamics_ll = dynamics_mll(self.dynamics_gp(self.X[:-1]), self.X[1:])

        # Optional: Prior on the latent space to encourage smoothness
        smoothness_prior = torch.sum((self.X[1:] - self.X[:-1]) ** 2)

        # Return the negative log-posterior (minimization objective)
        return -(total_observation_ll + dynamics_ll - 0.1 * smoothness_prior)  # Adjust prior weight as needed

    def train(self, num_iters=100):
        """
        Train the GPDM model using L-BFGS optimization.
        :param num_iters: Number of iterations for training
        """
        # Closure function required for L-BFGS optimizer
        def closure():
            self.optimizer.zero_grad()
            loss = self._objective_function()  # Compute loss
            loss.backward()  # Backpropagate gradients
            return loss

        # Run optimization for num_iters
        for i in range(num_iters):
            # Perform a single optimization step
            loss = self.optimizer.step(closure)

            # Print loss every 10 iterations to track training progress
            if i % 10 == 0:
                print(f"Iteration {i}/{num_iters} - Loss: {loss.item():.4f}")

    def extract_theta_feature(self):
        """
        Extract the theta feature vector (hyperparameters) from both the observation and dynamics GPs.
        :return: 7-dimensional feature vector
        """
        observation_hyperparams = self._extract_gp_hyperparameters(self.observation_gp)
        dynamics_hyperparams = self._extract_gp_hyperparameters(self.dynamics_gp)

        # Concatenate the hyperparameters from both GPs to form the Theta feature
        theta_feature = torch.cat((observation_hyperparams, dynamics_hyperparams))
        return theta_feature.cpu().detach().numpy()  # Return as numpy array for further analysis

    def _extract_gp_hyperparameters(self, gp_model):
        """
        Extracts the kernel hyperparameters (e.g., length scale, variance) from a GP model.
        :param gp_model: GP model (observation or dynamics GP)
        :return: Tensor of hyperparameters
        """
        lengthscale = gp_model.covar_module.base_kernel.lengthscale  # Extract lengthscale of RBF kernel
        variance = gp_model.covar_module.outputscale  # Extract output scale (variance)
        noise = gp_model.likelihood.noise  # Extract noise variance
        return torch.cat([lengthscale.flatten(), variance.view(1), noise.view(1)])  # Flatten and concatenate
