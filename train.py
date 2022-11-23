# -*- coding: utf-8 -*-

import os
import queue
import hydra
import random
import logging
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.equations import equation_dict
from utils.data_utils import split_data
from utils.samplers import sampler_dict
from utils.reweightings import reweighting_dict
from utils.models import FullyConnectedNetwork, model_saver
from test import test2d


@hydra.main(version_base=None, config_path="./conf", config_name="ACEquation")
def train_setup(cfg):
    log = logging.getLogger("Train")
    problem_conf = cfg["problem_conf"]
    global_conf = cfg["global_conf"]
    model_conf = cfg["model_conf"]
    train_conf = cfg["train_conf"]
    data_conf = cfg["data_conf"]
    tensorboard_writer = SummaryWriter(cfg["global_conf"]["tensorboard_path"])
    log.info(OmegaConf.to_yaml(cfg))

    # ---------------
    # global
    # ---------------
    if global_conf["seed"]:
        np.random.seed(global_conf["seed"])
        random.seed(global_conf["seed"])
        torch.manual_seed(global_conf["seed"])
        torch.cuda.manual_seed(global_conf["seed"])

    device = torch.device(global_conf["device"])
    log.info(f"device: {device}")

    # -------------
    # model
    # -------------
    log.info("create model...")
    model = FullyConnectedNetwork(model_conf)
    model.to(device)
    log.info(model)
    if model_conf.load_model:
        log.info("load weights")
        model.load_state_dict(torch.load(model_conf.model_path))
        log.info("load done...")

    # ------------
    # create data
    # ------------
    problem_define = equation_dict[cfg["name"]](problem_conf, data_conf)  # create data_manager
    problem_define.data_generator(global_conf["seed"])  # create dataset
    log.info("create problem data successful...")

    # ---------------------
    # split training, validating, testing
    # ---------------------
    split_t_dict = {
        "train": train_conf["train_t_range"],
        "eval": train_conf["eval_t_range"],
        "test": train_conf["test_t_range"]
    }
    boundary_data_split_result = split_data(problem_define.boundary_data, split_t_dict, 0)
    pde_data_split_result = split_data(problem_define.pde_data, split_t_dict, 0)

    log.info("split dataset successful...")

    # ---------
    # create sampler
    # ---------
    # train data sampler
    train_initial_tensor = torch.from_numpy(problem_define.initial_data).to(device=device, dtype=torch.float)
    train_boundary_tensor = torch.from_numpy(boundary_data_split_result["train"]).to(device=device, dtype=torch.float)
    train_pde_tensor = torch.from_numpy(pde_data_split_result["train"]).to(device=device, dtype=torch.float)
    train_pde_tensor.requires_grad = True
    if problem_conf["boundary_cond"] == "periodic":
        train_boundary_tensor.requires_grad = True

    train_pde_sampler = sampler_dict[train_conf["pde_sampler"]](
        train_pde_tensor, reweighting_dict[train_conf["pde_reweighting"]](train_conf["reweighting_params"]),
        model=model,
        loss_func=problem_define.compute_loss_basic_weights,
        **train_conf["sampler_conf"]
    )
    train_initial_sampler = sampler_dict["UniformSampler"](train_initial_tensor, reweighting_dict["NoReWeighting"]())
    train_boundary_sampler = sampler_dict["UniformSampler"](train_boundary_tensor, reweighting_dict["NoReWeighting"]())

    # validate data
    project_root = get_original_cwd()
    ground_true_numpy = np.load("{}/ground_true/{}.npz".format(project_root, cfg["name"]))

    x_input = ground_true_numpy["input_x"].reshape(-1, 1)
    t_input = ground_true_numpy["input_t"].reshape(-1, 1)
    output = ground_true_numpy["output"].reshape(-1, 1)
    ground_true = np.concatenate([t_input, x_input, output], axis=1)

    # ground true
    ground_true_split_data = split_data(ground_true, split_t_dict, 0)
    for key, data in ground_true_split_data.items():
        ground_true_split_data[key] = torch.from_numpy(data).to(device=torch.device("cuda"), dtype=torch.float)

    # -------------
    # optimizer
    # -------------
    optim = torch.optim.Adam(model.parameters(), **train_conf["optim_conf"])

    # -------------
    # main loop
    # -------------
    best_eval_loss = 1e6
    best_model_save_path = None
    train_main_conf = train_conf["main_conf"]
    model_save_queue = queue.Queue(maxsize=5)
    for step in range(train_main_conf["max_steps"]):

        train_pde_data = train_pde_sampler.sampler(train_main_conf["pde_batch_size"])
        train_initial_data = train_initial_sampler.sampler(train_main_conf["initial_batch_size"])
        train_boundary_data = train_boundary_sampler.sampler(train_main_conf["boundary_batch_size"])

        optim.zero_grad()
        loss_dict = problem_define.compute_loss(model, train_pde_data, train_initial_data, train_boundary_data, "train")
        optim.step()

        if step % train_main_conf["print_frequency"] == 0:
            log.info(f"step: {step}")
            for key, value in loss_dict.items():
                log.info("{} loss: {:.5e}".format(key, value))
                tensorboard_writer.add_scalar(f"TrainLoss/{key}", value, step)

        if step % train_main_conf["eval_frequency"] == 0:
            log.info("evaluation")
            model.eval()

            # evaluation
            loss_dict = dict()
            for key, data in ground_true_split_data.items():
                _pred = model(data[:, 0:2])
                if _pred.shape[1] == 2:
                    _pred = torch.sqrt(_pred[:, 0:1] ** 2 + _pred[:, 1:2] ** 2)
                _error = torch.abs(_pred - data[:, 2:3])
                _absolute_error = torch.mean(_error).item()
                _l2_error = torch.mean(_error**2).item()
                _peak_error = torch.max(_error).item()
                log.info("{} area: peak error:{:.4e}, "
                         "absolute error:{:.4e}, "
                         "l2 error:{:.4e}".format(key, _peak_error, _absolute_error, _l2_error))

                tensorboard_writer.add_scalar(f"Error/{key} peak", _peak_error, step)
                tensorboard_writer.add_scalar(f"Error/{key} l2", _l2_error, step)
                tensorboard_writer.add_scalar(f"Error/{key} absolute", _absolute_error, step)

                loss_dict[key] = _l2_error

            if best_eval_loss > loss_dict["eval"]:
                best_eval_loss = loss_dict["eval"]
                best_model_save_path = model_saver(
                    save_folder=train_main_conf["model_save_folder"],
                    model=model,
                    save_name=train_main_conf["model_basic_save_name"],
                    step=step
                )

                if model_save_queue.full():
                    del_step = model_save_queue.get()
                    del_path = os.path.join(train_main_conf["model_save_folder"],
                                            "{}_{}.pth".format(train_main_conf["model_basic_save_name"], del_step))
                    os.remove(del_path)

                model_save_queue.put(step)

            model.train()

    log.info("train done...")

    # ---------
    # testing
    # ---------
    log.info("begin test...")
    model.load_state_dict(torch.load(best_model_save_path))

    if problem_conf["dims"] == 2:

        test2d(model, problem_conf["t_range"], problem_conf["x_range"], ground_true=ground_true)
    log.info("test done...")


if __name__ == "__main__":
    train_setup()
