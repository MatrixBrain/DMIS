# -*- coding: utf-8 -*-

import random
import numpy as np
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
from ..data_utils import uniform_sampling

import torch

from ..pde_utils import fwd_gradients
from .basic_define import problem_decorator, ProblemDefine2d


@problem_decorator
class Schrodinger(ProblemDefine2d):

    def __init__(self, problem_conf, data_conf):
        super(Schrodinger, self).__init__(problem_conf, data_conf)

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

    def create_initial_data(self):
        x = sym.symbols("x")
        t = sym.symbols("t")
        _temp_x_data = uniform_sampling(list(self.x_range), self.initial_data_n)
        _temp_t_data = uniform_sampling(self.t_range[0], self.initial_data_n)

        _expr_real = parse_expr(self.initial_condition, evaluate=False)
        _expr_real = sym.lambdify((x, t), _expr_real, "numpy")
        _ground_true_real = _expr_real(_temp_x_data, _temp_t_data)
        _ground_true_imaginary = np.zeros_like(_ground_true_real)
        self.initial_data = np.concatenate(
            (
                _temp_t_data,
                _temp_x_data,
                _ground_true_real,
                _ground_true_imaginary
            ),
            axis=1
        )

    def pde_loss(self, pred, input_tensor):

        pred_real = pred[:, 0:1]
        pred_imag = pred[:, 1:2]

        df_dt_dx_real = fwd_gradients(pred_real, input_tensor)
        df_dt_real = df_dt_dx_real[:, 0:1]
        df_dx_real = df_dt_dx_real[:, 1:2]

        df_dt_dx_imag = fwd_gradients(pred_imag, input_tensor)
        df_dt_imag = df_dt_dx_imag[:, 0:1]
        df_dx_imag = df_dt_dx_imag[:, 1:2]

        df_dxx_real = fwd_gradients(df_dx_real, input_tensor)[:, 1:2]
        df_dxx_imag = fwd_gradients(df_dx_imag, input_tensor)[:, 1:2]

        pde_output_real = -df_dt_imag + 0.5 * df_dxx_real + (pred_real ** 2 + pred_imag ** 2) * pred_real
        pde_output_imag = df_dt_real + 0.5 * df_dxx_imag + (pred_real ** 2 + pred_imag ** 2) * pred_imag
        return pde_output_real, pde_output_imag

    def boundary_loss(self, input_lower, input_upper, pred_lower, pred_upper):

        df_dx_lower_real = fwd_gradients(pred_lower[:, 0:1], input_lower)[:, 1:2]
        df_dx_lower_imag = fwd_gradients(pred_lower[:, 1:2], input_lower)[:, 1:2]

        df_dx_upper_real = fwd_gradients(pred_upper[:, 0:1], input_upper)[:, 1:2]
        df_dx_upper_imag = fwd_gradients(pred_upper[:, 1:2], input_upper)[:, 1:2]

        boundary_value_loss_real = torch.mean((pred_lower[:, 0:1] - pred_upper[:, 0:1]) ** 2)
        boundary_value_loss_imag = torch.mean((pred_lower[:, 1:2] - pred_upper[:, 1:2]) ** 2)
        boundary_gradient_loss_real = torch.mean((df_dx_lower_real - df_dx_upper_real) ** 2)
        boundary_gradient_loss_imag = torch.mean((df_dx_lower_imag - df_dx_upper_imag) ** 2)

        total_boundary_loss = boundary_value_loss_real + boundary_value_loss_imag +\
                              boundary_gradient_loss_real + boundary_gradient_loss_imag

        return total_boundary_loss

    def compute_loss(self, model, pde_data, initial_data, boundary_data, state="train"):
        loss_dict = dict()
        if state == "train":

            # -------------
            # initial conditions
            # -------------
            _initial_data = initial_data["data"]
            initial_input = _initial_data[:, 0:2]
            initial_ground_true = _initial_data[:, 2:4]
            initial_pred = model(initial_input)
            initial_loss = torch.mean((initial_pred - initial_ground_true) ** 2)

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
            _pde_weight = torch.reshape(_pde_weight, (pde_pred.shape[0], 1))
            _pde_real, _pde_imag = self.pde_loss(pde_pred, _pde_data)
            pde_loss = torch.sum((_pde_real ** 2 + _pde_imag ** 2).mul(_pde_weight))

            total_loss = initial_loss + boundary_loss + pde_loss
            total_loss.backward()

            loss_dict["pde"] = pde_loss.item()
            loss_dict["initial"] = initial_loss.item()
            loss_dict["boundary"] = boundary_loss.item()
            loss_dict["total"] = total_loss.item()

        return loss_dict

    def compute_loss_basic_weights(self, model, data):
        pde_pred = model(data)
        _pde_real, _pde_imag = self.pde_loss(pde_pred, data)
        pde_loss = torch.sqrt(_pde_real ** 2 + _pde_imag ** 2)
        return pde_loss

