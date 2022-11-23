# -*- coding: utf-8 -*-

import numpy as np
import torch

reweighting_dict = dict()


def reweighting_decorator(reweighting):
    reweighting_dict[reweighting.__name__] = reweighting
    return reweighting


class BasicReWeighting:

    def __init__(self, reweighting_params=None):
        self.reweighting_params = reweighting_params

    def sample_weights(self, indices, scores):
        raise NotImplementedError


@reweighting_decorator
class NoReWeighting(BasicReWeighting):

    def __init__(self, reweighting_parmas=None):
        super(NoReWeighting, self).__init__(reweighting_parmas)

    def sample_weights(self, indxs, scores):
        """
        sample weight = 1
        """
        return torch.ones(len(indxs))


@reweighting_decorator
class BiasedReWeighting(BasicReWeighting):

    def __init__(self, reweighting_params=None):
        """
        ref:
        A. Katharopoulos, F. Fleuret.
        Biased importance sampling for deep neural network training[J].
        arXiv preprint arXiv:1706.00043, 2017.
        """
        super(BiasedReWeighting, self).__init__(reweighting_params)
        self.k_zero = self.reweighting_params["k_init"]
        self.k_end = self.reweighting_params["k_final"]
        self.max_step = self.reweighting_params["iter_n"]
        self.decay_step = int(self.max_step * 0.25)
        self.decrease_step = self.max_step - self.decay_step
        self.step_count = 0

    def sample_weights(self, indxs, scores):

        # rate
        if self.step_count <= self.decay_step:
            k = self.k_zero
        else:
            k = self.k_zero + (self.k_end - self.k_zero) * (self.step_count / self.decrease_step)

        # reweighting
        samples_len = len(scores)
        samples_scores = scores[indxs]
        samplers_weight = np.sum(scores) / (samples_len * samples_scores)
        samplers_weight = samplers_weight ** k

        self.step_count += 1
        return torch.from_numpy(samplers_weight)

