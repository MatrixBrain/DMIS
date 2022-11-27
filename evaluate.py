# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : evaluation_metrics_result.py
# @Time       : 2022/7/28 0:56
import logging
import os.path

import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch

from utils.models import FullyConnectedNetwork


@hydra.main(version_base=None, config_path="./conf", config_name="evaluation_conf")
def evaluation_setup(cfg):
    log = logging.getLogger("evaluation")
    project_root = get_original_cwd()

    model_conf = cfg["model_conf"]
    equation_conf = cfg["equation_conf"]
    evaluation_metrics = cfg["evaluation_metrics"]

    device = torch.device("cuda")

    for equation_key, conf in equation_conf.items():
        log.info(equation_key)

        log.info("create model and load weight")
        model_dict = dict()
        for model_key, weight_path in conf["weight_dict"].items():
            if weight_path != "":
                weight_path = os.path.join(project_root,
                                           "pretrain/{}".format(equation_key),
                                           weight_path)
                model_conf["layer"]["layer_n"] = conf["layer_n"]
                model_conf["layer"]["layer_size"] = conf["layer_size"]
                model_conf["dim"]["output_dim"] = conf["output_dim"]
                model_dict[model_key] = FullyConnectedNetwork(model_conf).to(device)
                model_dict[model_key].load_state_dict(torch.load(weight_path))

        log.info("load ground true")
        ground_true_path = os.path.join(project_root, "ground_true/{}.npz".format(equation_key))
        ground_true_numpy = np.load(ground_true_path)

        x_input = ground_true_numpy["input_x"]
        t_input = ground_true_numpy["input_t"]
        max_t = np.max(t_input)
        output = ground_true_numpy["output"]
        test_data_indices = np.argwhere(t_input > max_t * 0.75).reshape(-1)

        test_data_x = x_input[test_data_indices].reshape(-1, 1)
        test_data_t = t_input[test_data_indices].reshape(-1, 1)
        test_data_ground_true = output[test_data_indices].reshape(-1, 1)

        test_data = np.concatenate([
            test_data_t,
            test_data_x,
            test_data_ground_true
        ], axis=1)

        log.info("pred")
        input_tensor = torch.from_numpy(test_data[:, :2]).to(device=device, dtype=torch.float)
        pred_dict = dict()

        for model_key, model in model_dict.items():
            _pred = model(input_tensor).detach().cpu().numpy()
            if _pred.shape[1] == 1:
                _pred = _pred.reshape(-1)
            elif _pred.shape[1] == 2:
                _pred = np.sqrt(_pred[:, 0] ** 2 + _pred[:, 1] ** 2)
            pred_dict[model_key] = _pred

        log.info("evaluation")
        evaluation_dict = dict()
        for model_key, pred_result in pred_dict.items():
            for metric in evaluation_metrics:
                if metric == "max error":
                    evaluation_result = np.max(np.abs(pred_result - test_data[:, -1]))
                elif metric == "l2 norm":
                    evaluation_result = np.linalg.norm(pred_result - test_data[:, -1])
                elif metric == "RMSE":
                    evaluation_result = np.sqrt(np.mean((pred_result - test_data[:, -1]) ** 2))
                elif metric == "mean absolute error":
                    evaluation_result = np.mean(np.abs(pred_result - test_data[:, -1]))
                else:
                    raise KeyError
                evaluation_dict["{}_{}".format(model_key, metric)] = evaluation_result

        log.info("print evaluation result")
        for key, value in evaluation_dict.items():
            log.info("{}: {:.5e}".format(key, value))


if __name__ == "__main__":
    evaluation_setup()
