import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class LMCMultitaskGP(ExactGP):
    """
    Implements a Multi-Task GP with Linear Model of Coregionalization (LMC)
    for exact inference.
    """
    def __init__(self, train_x, train_y, likelihood, num_tasks, num_latents):
        # Initialize the ExactGP model
        super(LMCMultitaskGP, self).__init__(train_x, train_y, likelihood)
        
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
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x),
            task_dim=-1
        )


class PoseGP:
    """ Encapsulates training and prediction of LMC GP model for pose data.
        Uses exact inference.
    """

    def __init__(self, num_dofs: int, num_latents: int, learning_rate: float = 0.01):
        """
        Initialize the PoseGP model.
        Parameters:
        - num_dofs (int): The number of degrees of freedom in the pose data
        - num_latents (int): The number of latent functions in the LMC GP model
        - learning_rate (float): The learning rate to use for optimization
        """
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_dofs)
        self.model = None
        self.optimizer = None
        self.mll = None
        self.learning_rate = learning_rate

    def fit(self, train_x, train_y, iterations: int = 200):
        """
        Fit the GP model to the training data using exact inference.

        Parameters:
        - train_x (Tensor): The training input data
        - train_y (Tensor): The training target data
        - iterations (int): The number of iterations to train
        """
        # Initialize the model and the marginal log likelihood
        num_dofs = train_y.size(-1)  # The number of degrees of freedom (output dimensions)
        num_latents = train_y.size(-1)  # Use the number of outputs as latent processes (can be modified)
        
        self.model = LMCMultitaskGP(train_x, train_y, self.likelihood, num_dofs, num_latents)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Set up the optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        self.likelihood.train()

        for i in range(iterations):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i}/{iterations} - Loss: {loss.item():.4f}")

    def predict(self, x):
        """
        Predict outputs for the test data.

        Parameters:
        - x (Tensor): The test input data

        Returns:
        - preds: The predicted outputs
        """
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

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x))

        log_likelihood = preds.log_prob(y)
        return log_likelihood.item()
