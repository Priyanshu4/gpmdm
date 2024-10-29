import torch
import gpytorch
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution, LMCVariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO


class LMCMultitaskGP(ApproximateGP):
    """
    Implements a Multi-Task GP with Linear Model of Coregionalization (LMC)
    """
    def __init__(self, inducing_points, num_tasks, num_latents):
        # Define the variational distribution and base variational strategy
        variational_distribution = MeanFieldVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_latents])
        )
        base_variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        # Use LMCVariationalStrategy to model the linear combination of latent GPs
        variational_strategy = LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1  # The batch dimension corresponding to latent functions
        )
        
        # Initialize the parent class with the LMC variational strategy
        super().__init__(variational_strategy)
        
        # Define kernel and mean for each latent GP (with shared structure)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
        
    def forward(self, x):
        # Evaluate the latent GPs and apply coregionalization to get the multitask output
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class PoseGP:
    """ Encapsulates training and prediction of LMC GP model for pose data. 
        Stochastic Variational Inference (SVI) is used for training.
    """

    def __init__(self, num_dofs: int, num_latents: int, num_inducing_points: int, learning_rate: float = 0.01):
        """
        Initialize the PoseGP model.
        Parameters:
        - num_dofs (int): The number of degrees of freedom in the pose data
        - num_latents (int): The number of latent functions in the LMC GP model
        - num_inducing_points (int): The number of inducing points to use in the GP model
        - learning_rate (float): The learning rate to use for optimization
        """

        # Set up the inducing points
        inducing_points = torch.randn(num_inducing_points, num_dofs)

        # Create the GP model
        self.model = LMCMultitaskGP(inducing_points, num_dofs, num_latents)
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_dofs)
        
        # Set up the optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ],  lr=learning_rate)
                                          
                   
    def fit(self, train_x, train_y, iterations: int = 200):
        """
        Fit the GP model to the training data using SVI.

        Parameters:
        - train_x (Tensor): The training input data
        - train_y (Tensor): The training target data
        - iterations (int): The number of iterations to train
        """

        self.model.train()
        self.likelihood.train()

        # Create the mll (Variational ELBO) with the num_data parameter
        self.mll = VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))
        
        for i in range(iterations):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations} - Loss: {loss.item():.4f}")


    def predict(self, x):

        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x))
        
        return preds
    

    def log_likelihood(self, x, y):
        """
        Compute the log likelihood of the data under the model.
        
        Parameters:
        - x (Tensor): The input data
        - y (Tensor): The target data
        
        Returns:
        - log_likelihood: The log likelihood of the test data

        """

        self.model.eval()
        self.likelihood.eval()
        
        # Forward pass: Get the model's posterior distribution for the test data
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x))
        
        # Calculate the log likelihood of the observed data under the predicted distribution
        log_likelihood = preds.log_prob(y)

        # Return the log likelihood
        return log_likelihood.item()
            



