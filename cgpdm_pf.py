import torch
from cgpdm import CGPDM

_LOG_2PI = torch.log(torch.tensor(2 * 3.14159265358979323846))

class CGPDM_PF:
    """
    CGPDM Particle Filter
    """

    def __init__(self, cgpdm: CGPDM, 
                 num_particles: int = 100, 
                 num_particles_reset_per_observation: int = 30):
        """ Given a trained class aware GPDM model, 
            this builds a model to compute likelihood of a sequence of observations
        """

        self.cgpdm = cgpdm
        self.cgpdm.set_evaluation_mode()

        self.num_particles = num_particles
        self.num_particles_reset_per_observation = num_particles_reset_per_observation
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

    def _sample_particles_from_training_data(self, n_particles: int, class_index: int):
        """
        Sample n_particles from the training data of a given class
        """
        class_data = self.cgpdm.get_X_for_class(class_index)
        indices = torch.randperm(class_data.size(0))[:n_particles]
        return class_data[indices].detach().clone()


    def _init_particles(self):
        """ 
        Initialize particles by randomly sampling latent points from each class of training data.
        Initialize weights uniformly.
        """
        self.particles = []
        self.class_labels = []
        particles_per_class = self.num_particles // self.cgpdm.num_classes
        for i in range(self.cgpdm.num_classes):
            class_particles = self._sample_particles_from_training_data(particles_per_class, i)
            self.particles.append(class_particles)
            self.class_labels.extend([i] * particles_per_class)

        self.particles = torch.cat(self.particles)

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
        # convert particles to tensor
        mean_Xout_pred, diag_var_Xout_pred = self.cgpdm.map_x_dynamics(self.particles)
    
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
        n_resample = self.num_particles - self.num_particles_reset_per_observation
        particle_indices = torch.multinomial(self.weights, n_resample, replacement=True)
        resampled_particles = self.particles[particle_indices]
        resampled_class_labels = self.class_labels[particle_indices]

        reset_particles = []
        reset_class_labels = []
        for i in range(self.cgpdm.num_classes):
            class_particles = self._sample_particles_from_training_data(self.num_particles_reset_per_observation, i)
            reset_particles.append(class_particles)
            reset_class_labels.extend([i] * self.num_particles_reset_per_observation)
            
        self.particles = torch.cat([resampled_particles, torch.cat(reset_particles)])
        self.class_labels = resampled_class_labels + reset_class_labels

    def log_likelihood(self) -> float:
        """
        Compute the log likelihood of the most recent observation given the model

        This is computed as a weighted average of the log likelihood of each particle
        """
        log_weights = torch.log(self.weights)
        log_weighted_likelihoods = log_weights + self.log_likelihoods
        max_ll = torch.max(log_weighted_likelihoods)
        log_likelihood = max_ll + torch.log(torch.sum(torch.exp(log_weighted_likelihoods - max_ll)))
        return log_likelihood.item()
   
    def log_likelihood_classwise(self):
        """
        Compute the log likelihood of the most recent observation given the model
        for each class

        This is computed as a weighted average of the log likelihood of each particle
        """
        log_likelihoods = []
        for i in range(self.cgpdm.num_classes):
            class_indices = torch.tensor([j for j in range(self.num_particles) if self.class_labels[j] == i])
            class_log_weights = torch.log(self.weights[class_indices])
            class_log_weighted_likelihoods = class_log_weights + self.log_likelihoods[class_indices]
            max_ll = torch.max(class_log_weighted_likelihoods)
            log_likelihood = max_ll + torch.log(torch.sum(torch.exp(class_log_weighted_likelihoods - max_ll)))
            log_likelihoods.append(log_likelihood.item())
        return log_likelihoods

    def reset(self):
        self._init_particles()

    @property
    def latent_dim(self):
        return self.gpdm.d

    @property
    def observation_dim(self):
        return self.gpdm.D