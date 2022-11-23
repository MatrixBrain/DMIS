# -*- coding: utf-8 -*-

import random
import numpy as np

import torch

from ..pde_utils import fwd_gradients
from .basic_define import problem_decorator, ProblemDefine2d


@problem_decorator
class ACEquation(ProblemDefine2d):

    def __init__(self, problem_conf, data_conf):
        super(ACEquation, self).__init__(problem_conf, data_conf)

    def data_generator(self, random_state=None):
        """
        create data from domain
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.create_boundary_data()
        self.create_initial_data()
        self.create_pde_data()

    def create_boundary_data(self):
        self._create_boundary_data_periodic()

    def pde_loss(self, pred, input_tensor):
        df_dt_dx = fwd_gradients(pred, input_tensor)
        df_dt = df_dt_dx[:, 0:1]
        df_dx = df_dt_dx[:, 1:2]

        df_dxx = fwd_gradients(df_dx, input_tensor)[:, 1:2]
        pde_output = df_dt - 0.0001 * df_dxx + 5 * pred ** 3 - 5 * pred
        return pde_output

    def boundary_loss(self, input_lower, input_upper, pred_lower, pred_upper):
        df_dx_lower = fwd_gradients(pred_lower, input_lower)[:, 1:2]
        df_dx_upper = fwd_gradients(pred_upper, input_upper)[:, 1:2]

        boundary_value_loss = torch.mean((pred_lower - pred_upper) ** 2)
        boundary_gradient_loss = torch.mean((df_dx_upper - df_dx_lower) ** 2)

        return boundary_gradient_loss + boundary_value_loss

    def compute_loss(self, model, pde_data, initial_data, boundary_data, state="train"):
        loss_dict = dict()
        if state == "train":

            # -------------
            # initial conditions
            # -------------
            _initial_data = initial_data["data"]
            initial_input = _initial_data[:, 0:2]
            initial_ground_true = _initial_data[:, 2:3]
            initial_pred = model(initial_input)
            initial_loss = torch.mean((initial_pred - initial_ground_true)**2)

            # -------------
            # boundary conditions
            # -------------
            _boundary_data = boundary_data["data"]
            boundary_input_lower = _boundary_data[:, [0, 1]]
            boundary_input_upper = _boundary_data[:, [0, 2]]
            boundary_pred_lower = model(boundary_input_lower)
            boundary_pred_upper = model(boundary_input_upper)
            boundary_loss = self.boundary_loss(
                boundary_input_lower,
                boundary_input_upper,
                boundary_pred_lower,
                boundary_pred_upper
            )
            # -------------
            # pde
            # -------------
            _pde_data = pde_data["data"]
            _pde_weight = pde_data["weights"] / torch.sum(pde_data["weights"])
            pde_pred = model(_pde_data)
            _pde_weight = torch.reshape(_pde_weight, pde_pred.shape)
            pde_loss = torch.sum((self.pde_loss(pde_pred, _pde_data)**2).mul(_pde_weight))

            total_loss = initial_loss + boundary_loss + pde_loss
            total_loss.backward()

            loss_dict["pde"] = pde_loss.item()
            loss_dict["initial"] = initial_loss.item()
            loss_dict["boundary"] = boundary_loss.item()
            loss_dict["total"] = total_loss.item()

        elif state == "eval":
            boundary_input = boundary_data[:, 0:2]
            boundary_ground_true = boundary_data[:, 2:3]
            boundary_pred = model(boundary_input)
            boundary_loss = torch.mean((boundary_pred - boundary_ground_true) ** 2)

            pde_pred = model(pde_data)
            pde_loss = torch.mean(self.pde_loss(pde_pred, pde_data) ** 2)

            total_loss = boundary_loss + pde_loss

            loss_dict["pde"] = pde_loss.item()
            loss_dict["boundary"] = boundary_loss.item()
            loss_dict["total"] = total_loss.item()

        return loss_dict

    def compute_loss_basic_weights(self, model, data):
        pde_pred = model(data)
        pde_loss = torch.abs(self.pde_loss(pde_pred, data))
        return pde_loss