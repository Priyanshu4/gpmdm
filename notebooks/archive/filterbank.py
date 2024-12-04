from gpdm import GPDM
from gpdm_pf import GPDM_PF
import numpy as np

class FilterBank:

    def __init__(self, gpdm_filters: list[GPDM_PF], window_size: int):
        """ Given a list of GPDM filters, this builds a model to classify a sequence of observations
        """

        self.gpdm_filters_ = gpdm_filters
        self.window_size_ = window_size

        self.log_likelihoods_ = np.zeros((len(gpdm_filters), self.window_size_), dtype=float)
        self.ll_pointer_ = 0
        self.total_updates_ = 0

    def reset(self):
        for gpdm in self.gpdm_filters_:
            gpdm.reset()

        self.log_likelihoods_ = np.zeros((len(self.gpdm_filters_), self.window_size_), dtype=float)
        self.ll_pointer_ = 0

    def update(self, y: np.array):
        """ Add a measurement to the filter bank
        """

        for i, gpdm_filter in enumerate(self.gpdm_filters_):
            gpdm_filter.update(y)
            self.log_likelihoods_[i, self.ll_pointer_] = gpdm_filter.log_likelihood()
            
        self.ll_pointer_ = (self.ll_pointer_ + 1) % self.window_size_
        self.total_updates_ += 1

    def get_latest_log_likelihood(self):
        return self.log_likelihoods_[:, self.ll_pointer_-1]
    
    def get_sum_log_likelihood(self):
        if self.total_updates_ < self.window_size_:
            return np.sum(self.log_likelihoods_[:self.total_updates_, :], axis=1)
        return np.sum(self.log_likelihoods_, axis=1)
    
    def get_probabilities(self):
        sum_ll = self.get_sum_log_likelihood()
        
        # likelihoods will be very small, so shift to avoid numerical issues
        shifted_ll = sum_ll - np.max(sum_ll)

        # exponentiate and normalize to get probabilities
        return np.exp(shifted_ll) / np.sum(np.exp(shifted_ll))
    
    def get_most_likely(self):
        return np.argmax(self.get_sum_log_likelihood())

