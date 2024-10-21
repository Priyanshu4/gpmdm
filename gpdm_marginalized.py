import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
from sklearn.decomposition import PCA

class GPDM_Marginalized:
    """ 
    The original GPDM model with marginalized hyperparameters.
    """


    def __init__(self, Y, latent_dim=3, pca_init=True):
        """
        Initialize GPDM model.
        
        Parameters:
        Y : array-like, shape (T, D)
            The observed data (T time steps, D dimensions).
        latent_dim : int, optional
            The dimensionality of the latent space X.
        pca_init : bool, optional
            If True, initialize latent space X0 using PCA. Otherwise, random initialization.
        """
        self.Y = Y
        self.T, self.D = Y.shape
        self.latent_dim = latent_dim
        self.beta0 = np.array([1, 1, 1e-6])
        self.alpha0 = np.array([1, 1, 1e-6, 1e-6])

        # Initialize X0 (latent space)
        self.X0 = self.initialize_X0(pca_init)

        self.X_map = None
        
    def initialize_X0(self, pca_init=True):
        """
        Initialize latent space X0 using PCA or random initialization.
        
        Parameters:
        pca_init : bool, optional
            If True, use PCA to initialize X0. Otherwise, use random initialization.
            
        Returns:
        X0 : array-like, shape (T, latent_dim)
            Initialized latent space.
        """
        if pca_init:
            pca = PCA(n_components=self.latent_dim)
            X0 = pca.fit_transform(self.Y)
        else:
            X0 = np.random.randn(self.T, self.latent_dim)
        return X0

    def rbf_kernel(self, X, length_scale, var, diag):
        """RBF Kernel function."""
        dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        K = var * np.exp(-0.5 * dists / length_scale**2)
        K += diag * np.eye(X.shape[0])  # Add diagonal noise
        return K

    def rbf_linear_kernel(self, X, var, length_scale, diag1, diag2):
        """RBF + Linear Kernel."""
        rbf = self.rbf_kernel(X, length_scale, var, diag1)
        linear = diag2 * X @ X.T
        return rbf + linear

    def log_posterior(self, Y, X, beta, alpha):
        """
        Compute the log-posterior of the GPDM model.
        
        Parameters:
        Y : array-like, shape (T, D)
            Observed data.
        X : array-like, shape (T, latent_dim)
            Latent variables.
        beta : array-like, shape (3,)
            Kernel hyperparameters for the observation kernel.
        alpha : array-like, shape (4,)
            Kernel hyperparameters for the latent dynamics kernel.
        
        Returns:
        log_posterior : float
            The value of the log-posterior.
        """
        _, J = Y.shape
        
        # Log likelihood (observation model)
        K_Y = self.rbf_kernel(X, *beta)
        det_term_Y = -J/2 * np.prod(np.linalg.slogdet(K_Y))
        tr_term_Y = -1/2 * np.trace(np.linalg.inv(K_Y) @ Y @ Y.T)
        LL = det_term_Y + tr_term_Y

        # Log prior (latent dynamics model)
        K_X = self.rbf_linear_kernel(X[:-1], *alpha)
        X_bar = X[1:]
        det_term_X = -self.latent_dim / 2 * np.prod(np.linalg.slogdet(K_X))
        tr_term_X = -1/2 * np.trace(np.linalg.inv(K_X) @ X_bar @ X_bar.T)
        LP = det_term_X + tr_term_X

        return LL + LP

    def _neg_f(self, params):
        """
        Negative log-posterior function used for optimization.
        
        Parameters:
        params : array-like
            Flattened array of latent variables and kernel hyperparameters.
        
        Returns:
        neg_log_posterior : float
            Negative log-posterior.
        """
        X = params[:self.T*self.latent_dim].reshape(self.X0.shape)
        beta = params[self.T*self.latent_dim:self.T*self.latent_dim + 3]
        alpha = params[self.T*self.latent_dim + 3:]
        return -1 * self.log_posterior(self.Y, X, beta, alpha)

    def optimize_gpdm(self):
        """
        Optimize the GPDM model using L-BFGS-B.
        
        Returns:
        X_map : array-like, shape (T, latent_dim)
            The optimized latent variables (MAP estimate).
        """
        # Compute gradients of the negative log-posterior
        _neg_fp = grad(self._neg_f)

        # Combined objective function and gradient for L-BFGS-B
        def f_fp(params):
            return self._neg_f(params), _neg_fp(params)

        # Initial values (latent variables and hyperparameters)
        x0 = np.concatenate([self.X0.flatten(), self.beta0, self.alpha0])

        # Optimize using L-BFGS-B
        res = fmin_l_bfgs_b(f_fp, x0)
        self.X_map = res[0][:self.T*self.latent_dim].reshape(self.X0.shape)
        
        return self.X_map
    
    def fit(self):
        """Alias for optimize_gpdm."""
        return self.optimize_gpdm()

