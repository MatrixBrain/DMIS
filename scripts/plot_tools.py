# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_train_process(data_dict: dict, if_log=True):
    """
    plot convergence curves
    """
    keys = list(data_dict.keys())
    model_name_list = set([key.split("_")[0] for key in keys])
    plot_data_dict = dict()
    for model_name in model_name_list:
        plot_data_dict["{}_step".format(model_name)] = None
        plot_data_dict["{}_time".format(model_name)] = None
        plot_data_dict["{}_data".format(model_name)] = list()

    for key, data in data_dict.items():
        model_name = key.split("_")[0]
        if plot_data_dict["{}_step".format(model_name)] is None:
            plot_data_dict["{}_step".format(model_name)] = data["step"]
            plot_data_dict["{}_time".format(model_name)] = data["walltime"] - np.min(data["walltime"])
        plot_data_dict["{}_data".format(model_name)].append(data["value"])

    for model_name in model_name_list:
        key = "{}_data".format(model_name)
        model_name = key.split("_")[0]
        plot_data_dict[key] = np.array(plot_data_dict[key])
        upper_list = np.max(plot_data_dict[key], axis=0)
        lower_list = np.min(plot_data_dict[key], axis=0)
        if if_log:
            mean_list = np.power(10, (np.log10(upper_list) + np.log10(lower_list)) / 2)
        else:
            mean_list = (upper_list + lower_list) / 2
        plot_data_dict["{}_mean".format(model_name)] = mean_list
        plot_data_dict["{}_upper".format(model_name)] = upper_list
        plot_data_dict["{}_lower".format(model_name)] = lower_list

    plt.figure(figsize=(16, 8))
    iter_subplot = plt.subplot(1, 2, 1)
    if if_log:
        iter_subplot.set_yscale("log")
    for model_name in model_name_list:
        step_list = plot_data_dict["{}_step".format(model_name)]
        mean_list = plot_data_dict["{}_mean".format(model_name)]
        upper_list = plot_data_dict["{}_upper".format(model_name)]
        lower_list = plot_data_dict["{}_lower".format(model_name)]

        iter_subplot.plot(step_list, mean_list, label=model_name, linewidth=2)
        iter_subplot.fill_between(step_list, upper_list, lower_list, alpha=0.3)

    iter_subplot.set_xlabel("iterations")
    iter_subplot.set_ylabel("loss")
    iter_subplot.legend()

    time_subplot = plt.subplot(1, 2, 2)
    if if_log:
        time_subplot.set_yscale("log")
    for model_name in model_name_list:
        time_list = plot_data_dict["{}_time".format(model_name)]
        mean_list = plot_data_dict["{}_mean".format(model_name)]
        upper_list = plot_data_dict["{}_upper".format(model_name)]
        lower_list = plot_data_dict["{}_lower".format(model_name)]

        time_subplot.plot(time_list, mean_list, label=model_name, linewidth=2)
        time_subplot.fill_between(time_list, upper_list, lower_list, alpha=0.3)
    time_subplot.set_xlabel("time")
    time_subplot.set_ylabel("loss")
    time_subplot.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # testing
    # curve 1: y = e^(-x) - 1
    # curve 2: y = e^(-2x) - 1

    test_data_dict = dict()
    test_basic_input = np.arange(200) * 0.1
    for i in range(10):
        test_model1_data = np.exp(-0.05*test_basic_input) - 1 + np.random.random(200) * np.exp(-0.07*test_basic_input)
        test_model1_data = (test_model1_data - np.min(test_model1_data) + 1e-1) * 1000
        test_model1_step = np.arange(200) * 50
        test_model1_time = np.arange(200) * 1.1 + np.random.random(200)

        test_data = {
            "value": test_model1_data,
            "step": test_model1_step,
            "walltime": test_model1_time
        }

        test_data_dict["model1_{}".format(i)] = test_data

        test_model2_data = np.exp(-0.2*test_basic_input) - 1 + np.random.random(200) * np.exp(-0.1*test_basic_input)
        test_model2_data = (test_model2_data - np.min(test_model2_data) + 1e-1) * 1000
        test_model2_step = np.arange(200) * 50
        test_model2_time = np.arange(200) * 1.3 + np.random.random(200)

        test_data = {
            "value": test_model2_data,
            "step": test_model2_step,
            "walltime": test_model2_time
        }
        test_data_dict["model2_{}".format(i)] = test_data

    plot_train_process(test_data_dict, if_log=True)


