# -*- coding: utf-8 -*-

import os

import torch
import logging
import numpy as np

from scipy import spatial, interpolate
from .plot_utils import mesh_plotter_2d
from .models import model_saver


sampler_dict = dict()


def sampler_decorator(sampler):

    sampler_dict[sampler.__name__] = sampler
    return sampler


@sampler_decorator
class BaseSampler:
    """
    Abstract class of sampler
    """

    def __init__(self, data, reweighting):
        self.data = data
        self.reweighting = reweighting

        self.data_n = len(self.data)
        self.indices = torch.arange(self.data_n)
        self.log = logging.getLogger("sampler")

    def __len__(self):
        return self.data_n

    def compute_scores(self):
        raise NotImplementedError

    def sampler(self, batch_size, replace=True):
        # sample weights
        indices, scores = self.compute_scores()

        # sampling
        sample_p = scores / np.sum(scores)
        sample_indices = np.random.choice(len(indices), batch_size, p=sample_p, replace=replace)

        sample_weights = self.reweighting.sample_weights(sample_indices, scores)

        batch_dict = dict()
        batch_dict["data"] = self.data[sample_indices]
        batch_dict["weights"] = sample_weights.to(torch.device("cuda"))

        return batch_dict


@sampler_decorator
class UniformSampler(BaseSampler):
    """
    Implement of PINN-O
    """

    def __init__(self, data, reweighting, *args, **kwargs):
        super(UniformSampler, self).__init__(data, reweighting)

        self.scores = np.ones(len(self))

    def compute_scores(self):
        return self.indices, self.scores


class InterpolationSampler(BaseSampler):

    def __init__(self, data, reweighting, *args, **kwargs):
        super(InterpolationSampler, self).__init__(data, reweighting)

        # mesh simplex
        self.interp_simplex_result = None
        # interpolation weights
        self.interp_bary_weights = None
        # mesh update flag
        self.mesh_update_flag = False
        # set of mesh points
        self.seed_indxs = None

        # interpolation result
        self.interp_data = None

    def compute_scores(self):
        raise NotImplementedError

    def mesh_update(self):
        """
        update mesh
        1. Delaunay
        2. Compute the triangular of each point
        """

        seed_points = self.interp_data[self.seed_indxs]
        n_dim = self.interp_data.shape[1]

        # Delaunay
        tri = spatial.Delaunay(seed_points)

        # Compute the triangular of each point
        simplex = tri.find_simplex(self.data_numpy)

        self.interp_simplex_result = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.data_numpy - temp[:, n_dim]
        bary = np.einsum('njk,nk->nj', temp[:, :n_dim, :], delta)
        self.interp_bary_weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def seed_update(self):
        raise NotImplementedError


@sampler_decorator
class SamplerWithDMIS(InterpolationSampler):
    """
    Implement of PINN-DMIS
    """

    def __init__(
            self,
            data,
            reweighting,
            *args,
            **kwargs
    ):
        super(SamplerWithDMIS, self).__init__(data, reweighting)

        # model
        self.model = kwargs["model"]
        # loss function
        self.loss_func = kwargs["loss_func"]
        # mesh update threshold
        self.mesh_update_thres = kwargs["mesh_update_thres"]
        # batch size of computing sample weights
        self.forward_batch_size = kwargs["forward_batch_size"]

        # create data
        addon_points = torch.tensor(list(kwargs["addon_points"])).to(device=torch.device("cuda"), dtype=torch.float)  # 获取额外边界点
        self.data = torch.concat([self.data, addon_points], dim=0)
        self.data_np = self.data.detach().cpu().numpy()

        # total number
        self.addon_n = len(addon_points)
        self.seed_n = kwargs["seed_n"] - self.addon_n

        # init recoder
        self.seed_scores_t0 = np.ones(kwargs["seed_n"])
        self.seed_scores_t = np.ones(kwargs["seed_n"])
        self.step_count = 0
        if not os.path.exists("./mesh_data"):
            os.mkdir("./mesh_data")

        # init set of mesh points
        self._build_seed_indices()
        # init the triangular mesh
        self.mesh_update()

    def compute_scores(self):

        # --------------
        # mesh update
        # --------------
        if self.mesh_update_flag:
            self.seed_update()
            self.mesh_update()

        # ------------------
        # compute weight of mesh points
        # ------------------
        seed_data = self.data[self.seed_indxs]
        seed_len = len(seed_data)
        self.model.eval()
        _index = 0
        scores_tensor = torch.tensor([]).to(torch.device("cuda"))
        while True:
            if _index == seed_len:
                break

            last_index = min(_index + self.forward_batch_size, seed_len)

            input_tensor = seed_data[_index:last_index, :]
            _loss = self.loss_func(self.model, input_tensor)
            scores_tensor = torch.cat([scores_tensor, _loss], dim=0)
            _index = last_index

        self.model.train()
        seed_scores = scores_tensor.detach().cpu().numpy().reshape(-1)

        # ------------------
        # update weight recorders
        # ------------------
        if self.mesh_update_flag:
            self.seed_scores_t = seed_scores
            self.seed_scores_t0 = seed_scores.copy()
            self.mesh_update_flag = False
        else:
            self.seed_scores_t = seed_scores

        # ---------------
        # interpolation
        # ---------------
        interp_scores = np.einsum(
            'nj,nj->n',
            np.take(self.seed_scores_t, self.interp_simplex_result),
            self.interp_bary_weights
        )
        interp_scores -= np.min(interp_scores)
        interp_scores += 1e-15

        # check update
        self.mesh_update_check()
        self.step_count += 1

        # return the list of datapoints and sample weights
        return self.indices, interp_scores

    def mesh_update_check(self):
        # cosine similarity
        norm_scores = np.linalg.norm(self.seed_scores_t)
        norm_history_scores = np.linalg.norm(self.seed_scores_t0)
        cos_sim = self.seed_scores_t0.dot(self.seed_scores_t) / (norm_scores * norm_history_scores)

        if cos_sim < self.mesh_update_thres:
            self.mesh_update_flag = True
            self.log.info("change mesh")

    def seed_update(self):
        """update the set of mesh points"""
        scores_differance = np.abs(self.seed_scores_t - self.seed_scores_t0)

        interp_differance = np.einsum(
            'nj,nj->n',
            np.take(scores_differance, self.interp_simplex_result),
            self.interp_bary_weights
        )

        p = interp_differance / np.sum(interp_differance)

        # re-select mesh points
        self._build_seed_indices(p)

    def _build_seed_indices(self, p=None):
        """create the set of mesh points"""
        self.seed_indxs = np.random.choice(
            self.data_n,
            self.seed_n,
            p=p,
            replace=False
        )
        self.seed_indxs = np.append(self.seed_indxs, np.arange(self.data_n, self.data_n + self.addon_n))

    def mesh_update(self):
        """create triangular according to the set of mesh points"""

        seed_points = self.data_np[self.seed_indxs]
        n_dim = self.data_np.shape[1]

        # Delaunay
        tri = spatial.Delaunay(seed_points)
        simplex = tri.find_simplex(self.data_np[:self.data_n, :])
        self.interp_simplex_result = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.data_np[:self.data_n, :] - temp[:, n_dim]
        bary = np.einsum('njk,nk->nj', temp[:, :n_dim, :], delta)
        self.interp_bary_weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

        # save data
        save_dict = {
            "seeds": seed_points,
            "data": self.data_np[:self.data_n, :]
        }

        model_saver(
            save_folder="./mesh_data",
            model=self.model,
            save_name="mesh",
            step=self.step_count
        )

        output_full_path = os.path.join("./mesh_data", "{}_{}.npz".format("mesh", self.step_count))
        np.savez(output_full_path, **save_dict)

        # quick plot
        mesh_plotter_2d(seed_points, tri.simplices, self.step_count)


@sampler_decorator
class SamplerWithBasicIS(InterpolationSampler):
    """
    Implement of PINN-BasicIS
    """

    def __init__(
            self,
            data,
            reweighting,
            *args,
            **kwargs
    ):
        super(SamplerWithBasicIS, self).__init__(data, reweighting)

        # model
        self.model = kwargs["model"]
        # loss function
        self.loss_func = kwargs["loss_func"]
        # batch size of computing sample weights
        self.forward_batch_size = kwargs["forward_batch_size"]

        # create data
        addon_points = torch.tensor(list(kwargs["addon_points"])).to(device=torch.device("cuda"), dtype=torch.float)  # 获取额外边界点
        self.data = torch.concat([self.data, addon_points], dim=0)
        self.data_np = self.data.detach().cpu().numpy()

        # total number
        self.addon_n = len(addon_points)
        self.seed_n = kwargs["seed_n"] - self.addon_n

        # init recoder
        self.seed_scores_t0 = np.ones(kwargs["seed_n"])
        self.seed_scores_t = np.ones(kwargs["seed_n"])
        self.step_count = 0
        if not os.path.exists("./mesh_data"):
            os.mkdir("./mesh_data")

        # init the set of mesh points
        self._build_seed_indices()
        # init the triangular mesh
        self.mesh_update()

    def compute_scores(self):

        # ------------------
        # compute weight of mesh points
        # ------------------
        seed_data = self.data[self.seed_indxs]
        seed_len = len(seed_data)
        self.model.eval()
        _index = 0
        scores_tensor = torch.tensor([]).to(torch.device("cuda"))
        while True:
            if _index == seed_len:
                break

            last_index = min(_index + self.forward_batch_size, seed_len)

            input_tensor = seed_data[_index:last_index, :]
            _loss = self.loss_func(self.model, input_tensor)
            scores_tensor = torch.cat([scores_tensor, _loss], dim=0)
            _index = last_index

        self.model.train()
        seed_scores = scores_tensor.detach().cpu().numpy().reshape(-1)

        # ------------------
        # update weight recorders
        # ------------------
        if self.mesh_update_flag:
            self.seed_scores_t = seed_scores
            self.seed_scores_t0 = seed_scores.copy()
            self.mesh_update_flag = False
        else:
            self.seed_scores_t = seed_scores

        # ---------------
        # interpolation
        # ---------------
        interp_scores = np.einsum(
            'nj,nj->n',
            np.take(self.seed_scores_t, self.interp_simplex_result),
            self.interp_bary_weights
        )
        interp_scores -= np.min(interp_scores)
        interp_scores += 1e-15

        # check update
        self.mesh_update_check()
        self.step_count += 1

        # return the list of datapoints and sample weights
        return self.indices, interp_scores

    def _build_seed_indices(self, p=None):
        self.seed_indxs = np.random.choice(
            self.data_n,
            self.seed_n,
            p=p,
            replace=False
        )
        self.seed_indxs = np.append(self.seed_indxs, np.arange(self.data_n, self.data_n + self.addon_n))

    def seed_update(self):
        pass

    def mesh_update(self):
        """create triangular according to the set of mesh points"""

        seed_points = self.data_np[self.seed_indxs]
        n_dim = self.data_np.shape[1]

        # Delaunay
        tri = spatial.Delaunay(seed_points)
        simplex = tri.find_simplex(self.data_np[:self.data_n, :])
        self.interp_simplex_result = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.data_np[:self.data_n, :] - temp[:, n_dim]
        bary = np.einsum('njk,nk->nj', temp[:, :n_dim, :], delta)
        self.interp_bary_weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

        # save data
        save_dict = {
            "seeds": seed_points,
            "data": self.data_np[:self.data_n, :]
        }

        model_saver(
            save_folder="./mesh_data",
            model=self.model,
            save_name="mesh",
            step=self.step_count
        )

        output_full_path = os.path.join("./mesh_data", "{}_{}.npz".format("mesh", self.step_count))
        np.savez(output_full_path, **save_dict)

        # quick plot
        mesh_plotter_2d(seed_points, tri.simplices, self.step_count)


@sampler_decorator
class SamplerWithNabianMethod(InterpolationSampler):
    """
    Implement of PINN-N
    Nabian, M. A.; Gladstone, R. J.; and Meidani, H. 2021.
    Efficient training of physics-informed neural networks via importance sampling.
    Computer-Aided Civil and Infrastructure Engineering, 36(8): 962–977.
    """

    def __init__(
            self,
            data,
            reweighting,
            *args,
            **kwargs
    ):
        super(SamplerWithNabianMethod, self).__init__(data, reweighting)

        self.model = kwargs["model"]
        self.loss_func = kwargs["loss_func"]

        self.forward_batch_size = kwargs["forward_batch_size"]

        addon_points = torch.tensor(list(kwargs["addon_points"])).to(device=torch.device("cuda"), dtype=torch.float)  # 获取额外边界点
        self.data = torch.concat([self.data, addon_points], dim=0)
        self.data_np = self.data.detach().cpu().numpy()
        self.addon_n = len(addon_points)
        self.seed_n = kwargs["seed_n"] - self.addon_n

        self.seed_scores_t = np.ones(kwargs["seed_n"])
        self.step_count = 0
        if not os.path.exists("./mesh_data"):
            os.mkdir("./mesh_data")

        self._build_seed_indices()

    def compute_scores(self):

        # ------------------
        # compute sample weights
        # ------------------
        seed_data = self.data[self.seed_indxs]
        seed_len = len(seed_data)
        self.model.eval()
        _index = 0
        scores_tensor = torch.tensor([]).to(torch.device("cuda"))
        while True:
            if _index == seed_len:
                break

            last_index = min(_index + self.forward_batch_size, seed_len)

            input_tensor = seed_data[_index:last_index, :]
            _loss = self.loss_func(self.model, input_tensor)
            scores_tensor = torch.cat([scores_tensor, _loss], dim=0)
            _index = last_index

        self.model.train()
        seed_scores = scores_tensor.detach().cpu().numpy().reshape(-1)

        self.seed_scores_t = seed_scores

        # ---------------
        # interpolation
        # ---------------
        interp_scores = interpolate.griddata(self.data_np[self.seed_indxs],
                                             self.seed_scores_t,
                                             self.data_np[:self.data_n],
                                             method="nearest")
        interp_scores -= np.min(interp_scores)
        interp_scores += 1e-15

        self.step_count += 1

        return self.indices, interp_scores

    def _build_seed_indices(self, p=None):
        self.seed_indxs = np.random.choice(
            self.data_n,
            self.seed_n,
            p=p,
            replace=False
        )
        self.seed_indxs = np.append(self.seed_indxs, np.arange(self.data_n, self.data_n + self.addon_n))

    def seed_update(self):
        """no need to update the set of mesh points"""
        pass

    def mesh_update(self):
        """no need to update the triangular mesh"""
        pass
