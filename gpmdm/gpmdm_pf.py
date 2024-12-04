import torch
from torchtyping import TensorType
from .gpmdm import GPMDM

_LOG_2PI = torch.log(torch.tensor(2 * 3.14159265358979323846))

class GPMDM_PF:
    """
    GPMDM Particle Filter

    This class implements a particle filter for a GPMDM model. 
    The particle filter is used to estimate the latent state of the model given observations.

    Each particle represents a hypothesis of the latent state and class of the system.
    Each particle i at time t is represented by tuple (x_t, c_t, w_t) where:

        x_t: latent state of the system
        c_t: class (mode) of the system
        w_t: weight of the particle

    At each time step, the particle filter is updated with a new observation z_t.
    Particles are propogated by sampling from the predicted distribution of the next state.
    Update Steps:
    1. Particles are propogated through the Markov transition matrix to update the class of each particle.
    2. Particles are propogated through the corresponding dynamics model to update the latent state of each particle.
    3. Weights of each particle are updated:
        a) Particles are propogated through the measurement model.
        b) For each particle, compute the log likelihood of the observation given the particle.
        c) For each class, the likelihood is the weighted sum of the likelihoods of the particles in that class.
        d) Weights are updated according to w_k = p(z | x_k, c_k) * w_{k-1}
    4. Particles are resampled.
        a) Particles are resampled with replacement based on their weights.
        b) A user-defined number of fresh particles are sampled from the training data of each class.
        c) In total, the number of particles remains constant.

    This class uses shape annotations to ensure that the tensors have the correct shapes.

    Shapes:
        P: number of particles
        L: number of latent dimensions
        D: number of observation dimensions
        C: number of classes
    """

    def __init__(self, 
                 gpmdm: GPMDM, 
                 markov_switching_model: TensorType["C", "C", torch.float],
                 num_particles: int,
                 n_fresh_particles_at_each_timestep: int):
        """ 
        Initialize the particle filter

        Parameters
        ----------
        gpmdm : GPMDM
            GPMDM model

        markov_switching_model : tensor of shape (C, C)
            Markov transition matrix between classes (C is the number of classes)

        num_particles : int
            Number of particles to use in the particle filter
        """

        # Set the GPMDM model and set it to evaluation mode
        self._gpmdm = gpmdm
        self._gpmdm.set_evaluation_mode()

        # Set member variables
        self._markov_switching_model = markov_switching_model.type(self.dtype)
        self._num_particles = num_particles
        self._n_fresh_particles_at_each_timestep = n_fresh_particles_at_each_timestep

        if self._gpmdm.n_classes != self._markov_switching_model.size(0):
            raise ValueError("Number of classes in the GPMDM model and the Markov model do not match")
        
        if self._num_particles <= self._n_fresh_particles_at_each_timestep:
            raise ValueError("Number of particles should be greater than the number of fresh particles at each timestep")

        # Type annotations
        self._particle_states : TensorType["P", "L"] = ...       # latent states of the particles
        self._particle_classes : TensorType["P", int] = ...      # class of each particle
        self._log_weights : TensorType["P", float] = ...         # log weights of each particle
        self._weights : TensorType["P", float] = ...             # weights of each particle
        self._log_likelihoods : TensorType["P", float] = ...     # log likelihood of each particle

        # Initialize the particle filter
        self._init_particles()

    def _init_particles(self):
        """ 
        Initialize particles by randomly sampling latent points from each class of training data.
        """
        n_particles_per_class = self._divide_into_n_parts(self._num_particles, self.num_classes)

        particles_of_each_class = []
        particle_classes = []
        for i in range(self.num_classes):
            class_particles = self._sample_particles_from_training_data(n_particles_per_class[i], i)
            particles_of_each_class.append(class_particles)
            particle_classes += [i] * n_particles_per_class[i]

        self._particle_states = torch.cat(particles_of_each_class, dim=0) 
        self._particle_classes = torch.tensor(particle_classes, dtype=torch.int64, device=self.device)
        self._log_likelihoods = torch.zeros(self._num_particles, dtype=self.dtype, device=self.device)
        self._log_weights = torch.zeros(self._num_particles, dtype=self.dtype, device=self.device)
        self._weights = torch.ones(self._num_particles, dtype=self.dtype, device=self.device) / self._num_particles
    
    def _sample_particles_from_training_data(self, n_particles: int, class_index: int) -> TensorType["P", "L"]:
        """
        Sample n_particles from the training data of a given class
        """
        class_data = self._gpmdm.get_X_for_class(class_index)

        # Sample indices with replacement
        indices = torch.randint(0, class_data.size(0), (n_particles,), device=self.device)
        
        return class_data[indices].detach().clone()

    def update(self, z: TensorType["D"]):
        """ Update the particle filter with a new observation

            Parameters:
                z: np.array of shape (D,) where D is the observation dimension
        """
        z = torch.tensor(z, dtype=self.dtype, device=self.device)
        self._update(z)

    def _update(self, z: TensorType["D"]):
        """ Update the particle filter with a new observation

            Parameters:
                z: np.array of shape (D,) where D is the observation dimension
        """
        self._propogate_markov_switching()
        self._propogate_dynamics()
        self._update_weights(z)
        self._resample()

    def _propogate_markov_switching(self):
        """
        Propogate the particles through the Markov transition matrix to get the next class.
        We sample the new class of each particle from the categorical distribution of the markov model.
        """
        # Get the current class of each particle as a one-hot vector
        one_hot_classes = self._one_hot(self._particle_classes, self.num_classes)
        one_hot_classes = one_hot_classes.type(self.dtype)

        # Propogate the one-hot vectors through the markov transition matrix
        class_distribution = torch.matmul(one_hot_classes, self._markov_switching_model)

        # Sample the new class of each particle from the categorical distribution
        new_classes = torch.multinomial(class_distribution, 1, replacement=True)
        self._particle_classes = new_classes.squeeze()

    def _propogate_dynamics(self):
        """
        Propogate the particles through the dynamics model of their class to get the next latent state.
        We sample the new latent state of each particle from the predicted gaussian distribution.
        """ 
        for i in range(self.num_classes):

            # get the particle states of just this class
            class_particles = self._particle_states[self._particle_classes == i]

            # propogate the particles through the dynamics model of their class
            mean_Xout_pred, diag_var_Xout_pred = self._gpmdm.map_x_dynamics_for_class(class_particles, class_index=i)

            # sample the new latent state of each particle from the predicted gaussian distribution
            self._particle_states[self._particle_classes == i] = \
                torch.normal(mean_Xout_pred, torch.sqrt(diag_var_Xout_pred))
            
    def _update_weights(self, z):
        """
        Update the weights of the particles based on the observation.
        Weight w_k of particle k is updated as:
            w_k = p(z | x_k, c_k) * w_{k-1}

        Parameters
        ----------
        z : tensor(dtype)
            observation vector
        """

        # propogate particle states through measurement model
        mean_Y_pred, diag_var_Y_pred = self._gpmdm.map_x_to_y(self._particle_states)
        
        # compute log likelihood of the observation for each particle
        # since the covariance matrix is diagonal, we can compute the likelihood as a product of likelihoods
        # product of likelihoods is equivalent to sum of log likelihoods
        for i in range(self._num_particles):
            log_likelihood_mu_term = -0.5 * torch.sum((z - mean_Y_pred[i])**2 / diag_var_Y_pred[i] + torch.log(diag_var_Y_pred[i]))
            log_likelihood_sigma_term = torch.sum(-torch.log(torch.sqrt(diag_var_Y_pred[i])))
            log_likelihood = log_likelihood_mu_term + log_likelihood_sigma_term - 0.5 * self._gpmdm.D * _LOG_2PI
            self._log_likelihoods[i] = log_likelihood

        # update the log weights
        # for each particle
        # w_t = p(z_t | x_t, c_t) * w_{t-1}
        # log(w_t) = log(p(z_t | x_t, c_t)) + log(w_{t-1})
        # self._log_weights = self._log_likelihoods + self._log_weights

        self._log_weights = self._log_likelihoods
        self._log_weights = self._log_weights - torch.max(self._log_weights)
  
        self._weights = torch.exp(self._log_weights)
        self._weights = self._weights / torch.sum(self._weights)

    def _resample(self):
        """ 
        Resample particles with replacement based on the weights. 
        Also sample fresh particles from the training data of each class.
        """
        
        # Compute the number of fresh particles to sample from each class
        n_fresh = self._n_fresh_particles_at_each_timestep
        n_fresh_per_class = self._divide_into_n_parts(n_fresh, self.num_classes)

        # Compute the number of particles to resample
        n_resample = self._num_particles - n_fresh 

        # Sample with replacement from the multinomial distribution according to the weights
        resampled_indices = torch.multinomial(self._weights, n_resample, replacement=True)
        resampled_states = self._particle_states[resampled_indices]
        resampled_classes = self._particle_classes[resampled_indices]

        # For each class, sample some fresh particles
        fresh_particles = []
        fresh_classes = []
        for i in range(self.num_classes):
            class_particles = self._sample_particles_from_training_data(n_fresh_per_class[i], i)
            fresh_particles.append(class_particles)
            fresh_classes += [i] * n_fresh_per_class[i]

        # Concatenate the resampled and fresh particles
        self._particle_states = torch.cat([resampled_states, torch.cat(fresh_particles)], dim=0)
        self._particle_classes = torch.cat([resampled_classes, 
                                           torch.tensor(fresh_classes, dtype=torch.int, device=self.device)], 
                                           dim=0)   

    def log_likelihood(self) -> float:
        """
        Compute the log likelihood of the most recent observation given the model

        This is computed as a weighted average of the likelihoods of each particle
        """
        log_likelihood = self._weighted_sum_from_log_space(self._log_likelihoods, self._log_weights)
        return log_likelihood.item()
   
    def class_probabilities(self) -> TensorType["C", float]:
        """
        Compute the probability of each class given the most recent observation

        P(z_t | c_t == i) = sum_{particles} P(z_t | x_t, c_t == i) * w_t
        """
        class_likelilhoods = torch.zeros(self.num_classes, dtype=float, device=self.device)
     
        # Compute log P(z_t | x_t) + log (w_t) for each particle
        log_weighted_likelihoods = self._log_likelihoods + self._log_weights

        # Subtract the max value to avoid numerical instability
        log_weighted_likelihoods -= torch.max(log_weighted_likelihoods)

        # Exponentiate to get P(z_t | x_t, c_t) * w_t
        weighted_likelihoods = torch.exp(log_weighted_likelihoods)

        # Sum the likelihoods for each class
        for i in range(self.num_classes):
            class_likelilhoods[i] = torch.sum(weighted_likelihoods[self._particle_classes == i])
            
        # Normalize to 1 to get the probabilities
        class_probabilities = class_likelilhoods / torch.sum(class_likelilhoods)

        return class_probabilities

    def get_most_likely_class(self) -> int:
        """
        Returns the class with the highest likelihood
        """
        return torch.argmax(self.class_probabilities()).item()

    def current_state_mean(self):
        """
        Compute the mean of the state distribution estimated by the particle filter

        This is computed as a weighted average of the particles
        """
        return torch.sum(self._particle_states * self._weights.unsqueeze(-1), dim=0)
    
    def reset(self):
        self._init_particles()

    @property
    def latent_dim(self):
        return self._gpmdm.d

    @property
    def observation_dim(self):
        return self._gpmdm.D
    
    @property
    def num_classes(self):
        return self._gpmdm.n_classes
    
    @property
    def dtype(self):
        return self._gpmdm.dtype
    
    @property
    def device(self):
        return self._gpmdm.device
    
    def _divide_into_n_parts(self, x: int, n: int) -> list[int]:
        """
        Divide the integer x into n parts
        """
        groupSize, remainder = divmod(x, n)
        return [groupSize + (1 if x < remainder else 0) for x in range(n)]
    
    def _one_hot(self, indices: TensorType['X', torch.int64], num_classes: int) -> TensorType['X', 'num_classes', torch.int64]:
        """
        Convert a tensor of indices to a one-hot tensor
        """
        one_hot_encoded = torch.zeros(indices.size(0), num_classes, dtype=torch.int, device=self.device)
        one_hot_encoded.scatter_(1, indices.unsqueeze(1), 1)
        return one_hot_encoded
    
    def _weighted_sum_from_log_space(self, 
                                     log_summands: TensorType['X', float],
                                     log_weights: TensorType['X', float]) -> TensorType['X', float]:
        """
        Compute the weighted sum of the summands and weights and return the result in standard space
        Only use this when you plan to normalize the result into a probability distribution.
        If all your log likelihoods are negative, the max value outputed will be 1.
        """
        log_weighted_summands = log_weights + log_summands
        max_weighted_summand = torch.max(log_weighted_summands)
        return torch.sum(torch.exp(log_weighted_summands - max_weighted_summand))