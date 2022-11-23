# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

from ..data_utils import uniform_sampling


equation_dict = dict()


def problem_decorator(problem):
    equation_dict[problem.__name__] = problem
    return problem


class ProblemDefine2d:

    def __init__(self, problem_conf, data_conf):
        self.x_range = problem_conf["x_range"]
        self.t_range = problem_conf["t_range"]
        self.initial_condition = problem_conf["initial_cond"]
        self.boundary_condition = problem_conf["boundary_cond"]

        self.initial_data_n = data_conf["initial_data_n"]
        self.boundary_data_n = data_conf["boundary_data_n"]
        self.pde_data_n = data_conf["pde_data_n"]

        self.initial_data = None
        self.boundary_data = None
        self.pde_data = None
        self.data_dict = dict()

    def create_initial_data(self):
        self._create_initial_data_basic()

    def create_boundary_data(self):
        self._create_boundary_data_basic()

    def create_pde_data(self):
        self._create_pde_data_basic()

    def _create_initial_data_basic(self):
        x = sym.symbols("x")
        t = sym.symbols("t")
        _temp_x_data = uniform_sampling(list(self.x_range), self.initial_data_n)
        _temp_t_data = uniform_sampling(self.t_range[0], self.initial_data_n)
        _expr = parse_expr(self.initial_condition, evaluate=False)
        _expr = sym.lambdify((x, t), _expr, "numpy")
        _ground_true = _expr(_temp_x_data, _temp_t_data)
        self.initial_data = np.concatenate((_temp_t_data, _temp_x_data, _ground_true), axis=1)
        self.data_dict["initial"] = self.initial_data

    def _create_boundary_data_basic(self):
        x = sym.symbols("x")
        t = sym.symbols("t")
        _temp_x_data = uniform_sampling(tuple(self.x_range), self.boundary_data_n)
        _temp_t_data = uniform_sampling(list(self.t_range), self.boundary_data_n)
        _expr = parse_expr(self.boundary_condition, evaluate=False)
        _expr = sym.lambdify((x, t), _expr, "numpy")
        _ground_true = _expr(_temp_x_data, _temp_t_data)
        self.boundary_data = np.concatenate((_temp_t_data, _temp_x_data, _ground_true), axis=1)
        self.data_dict["boundary"] = self.boundary_data

    def _create_boundary_data_periodic(self):
        _temp_t_data = uniform_sampling(list(self.t_range), self.boundary_data_n)
        _lower_x_data = np.ones_like(_temp_t_data) * self.x_range[0]
        _upper_x_data = np.ones_like(_temp_t_data) * self.x_range[1]
        self.boundary_data = np.concatenate((_temp_t_data, _lower_x_data, _upper_x_data), axis=1)
        self.data_dict["boundary"] = self.boundary_data

    def _create_pde_data_basic(self):
        _temp_x_data = uniform_sampling(list(self.x_range), self.pde_data_n)
        _temp_t_data = uniform_sampling(list(self.t_range), self.pde_data_n)
        self.pde_data = np.concatenate((_temp_t_data, _temp_x_data), axis=1)
        self.data_dict["pde"] = self.pde_data

    def pde_loss(self, pred, input_tensor):
        raise NotImplementedError

    def compute_loss(self, model, pde_data, initial_data, boundary_data, state="train"):
        raise NotImplementedError

    def plot_samples(self):
        plt.figure()
        plt.scatter(self.boundary_data[:, 0], self.boundary_data[:, 1], label="boundary", s=2)
        plt.scatter(self.initial_data[:, 0], self.initial_data[:, 1], label="initial", s=2)
        plt.scatter(self.pde_data[:, 0], self.pde_data[:, 1], label="pde", s=2)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.legend()
        plt.show()
