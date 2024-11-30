import torch
import numpy as np
from cgpdm_dynamics import CGPDM

_LOG_2PI = torch.log(torch.tensor(2 * 3.14159265358979323846))

class CGPDM_PF:
    """
    CGPDM Particle Filter
    """

    def __init__(self, cgpdm: CGPDM, 
                 num_particles: int = 100):
        """ Given a trained class aware GPDM model with different dynamics for each class, 
            this builds a model to compute likelihood of a sequence of observations
        """

        self.cgpdm = cgpdm
        self.cgpdm.set_evaluation_mode()

        self.num_particles = num_particles
        self._init_particles()

    def _init_particles(self):
        """ 
        Initialize particles by randomly sampling latent points from each class of training data.
        Initialize weights uniformly.
        """

        # Particles is a list of particles for each class
        self.particles = []
        particles_per_class = self.num_particles // self.cgpdm.n_classes
        for i in range(self.cgpdm.n_classes):
            class_particles = self._sample_particles_from_training_data(particles_per_class, i)
            self.particles.append(class_particles)
            
        self.log_likelihoods = torch.zeros(self.num_particles, dtype=self.cgpdm.dtype, device=self.cgpdm.device)
        self.weights = torch.ones(self.num_particles, dtype=self.cgpdm.dtype, device=self.cgpdm.device) / self.num_particles

    def _sample_particles_from_training_data(self, n_particles: int, class_index: int):
        """
        Sample n_particles from the training data of a given class
        """
        class_data = self.cgpdm.get_X_for_class(class_index)
        indices = torch.randperm(class_data.size(0))[:n_particles]
        return class_data[indices].detach().clone()

    def update(self, z):
        """ Update the particle filter with a new observation

            Parameters:
                z: np.array of shape (D,) where D is the observation dimension
        """
        z = torch.tensor(z, dtype=self.dtype, device=self.device)
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

    def _propogate_dynamics(self):
        """
        Propogate the particles through the dynamics model to get the next latent state
        """ 

        # for each class, propogate particles through the dynamics model
        for i in range(self.cgpdm.n_classes):
            class_particles = self.particles[i]
            mean_Xout_pred, diag_var_Xout_pred = self.cgpdm.map_x_dynamics_for_class(class_particles, class_index=i)
            self.particles[i] = torch.normal(mean_Xout_pred, torch.sqrt(diag_var_Xout_pred))
            
    def _update_weights(self, z):
        """
        Update the weights of the particles based on the observation

        Parameters
        ----------
        z : tensor(dtype)
            observation vector
        """

        # put the particles into one tensor
        all_particles = torch.cat(self.particles)
        
        # propogate particles through measurement model
        mean_Y_pred, diag_var_Y_pred = self.cgpdm.map_x_to_y(all_particles)
        

        # compute log likelihood of the observation for each particle
        # since the covariance matrix is diagonal, we can compute the likelihood as a product of likelihoods
        # product of likelihoods is equivalent to sum of log likelihoods
        max_ll = torch.tensor(float('-inf'))
        for i in range(self.particles.size(0)):

            log_likelihood_mu_term = -0.5 * torch.sum((z - mean_Y_pred[i])**2 / diag_var_Y_pred[i] + torch.log(diag_var_Y_pred[i]))
            log_likelihood_sigma_term = torch.sum(-torch.log(torch.sqrt(diag_var_Y_pred[i])))
            log_likelihood = log_likelihood_mu_term + log_likelihood_sigma_term - 0.5 * self.cgpdm.D * _LOG_2PI
            self.log_likelihoods[i] = log_likelihood
            max_ll = torch.max(max_ll, log_likelihood)

        # center the weights to avoid numerical instability
        self.weights = torch.exp(self.log_likelihoods - max_ll)
        self.weights /= torch.sum(self.weights)

    def _resample(self):
        """ 
        Resample particles based on the weights
        """
        class_end_index = -1
        for i in range(self.num_classes):
            class_start_index = class_end_index + 1
            class_end_index = self.particles[i].size(0) + class_start_index
            class_total_weight = torch.sum(self.weights[class_start_index:class_end_index])
            n_particles_to_resample = round(class_total_weight.item() * self.num_particles)
            if n_particles_to_resample == 0:
                if self.particles[i].size(0) > 0:
                    n_particles_to_resample = 1
                    one_normalized_class_weights = torch.ones(self.particles[i].size(0))
                else:
                    continue
            else:
                one_normalized_class_weights = self.weights[class_start_index:class_end_index] / class_total_weight

            # resample particles from this class
            # make a multinomial distribution with the weights
            resampled_indices = torch.multinomial(one_normalized_class_weights, n_particles_to_resample, replacement=True)
            self.particles[i] = self.particles[i][resampled_indices]

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
        class_end_index = -1
        for i in range(self.num_classes):
            class_start_index = class_end_index + 1
            class_end_index = self.particles[i].size(0) + class_start_index
            
            class_weights = self.weights[class_start_index:class_end_index]
            class_log_weights = torch.log(class_weights)
            class_log_weighted_likelihoods = class_log_weights + self.log_likelihoods[class_start_index:class_end_index]
            max_ll = torch.max(class_log_weighted_likelihoods)
            log_likelihood = max_ll + torch.log(torch.sum(torch.exp(class_log_weighted_likelihoods - max_ll)))
            log_likelihoods.append(log_likelihood.item())
        return log_likelihoods
    
    def current_state_mean(self):
        """
        Compute the mean of the state distribution estimated by the particle filter

        This is computed as a weighted average of the particles
        """
        return torch.sum(self.particles * self.weights.unsqueeze(-1), dim=0)
    
    def get_most_likely_class(self):
        """
        Returns the class with the highest likelihood
        """
        ll = self.log_likelihood_classwise()
        return np.argmax(ll)

    def reset(self):
        self._init_particles()

    @property
    def latent_dim(self):
        return self.cgpdm.d

    @property
    def observation_dim(self):
        return self.cgpdm.D
    
    @property
    def num_classes(self):
        return self.cgpdm.n_classes
    
    @property
    def dtype(self):
        return self.cgpdm.dtype
    
    @property
    def device(self):
        return self.cgpdm.device