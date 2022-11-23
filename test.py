# -*- coding: utf-8 -*-

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.plot_utils import interpolate_2d


def test2d(model, t_range, x_range, *, delta_t=0.01, delta_x=0.01, ground_true=None):

    model.eval()
    x_input = np.arange(x_range[0], x_range[1], delta_x)
    t_input = np.arange(t_range[0], t_range[1], delta_t)

    xx, tt = np.meshgrid(x_input, t_input)
    input_array = np.concatenate([tt.reshape(-1, 1), xx.reshape(-1, 1)], axis=1)
    input_tensor = torch.from_numpy(input_array).to(device=torch.device("cuda"), dtype=torch.float)

    pred = model(input_tensor).detach().cpu().numpy()
    if pred.shape[1] == 1:
        pred = pred.reshape(-1)
    elif pred.shape[1] == 2:
        pred = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
    else:
        raise ValueError("don't support {} dims".format(pred.shape[1]))

    pred_extent, pred_image, pred_mesh = interpolate_2d(input_array, pred)

    plt.figure(figsize=(8, 8))
    plt.pcolor(pred_mesh[0], pred_mesh[1], pred_image, cmap="rainbow")
    plt.xlabel("t")
    plt.ylabel("x")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad="3%")
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.savefig("./pred.png")
    plt.close()

    if ground_true is not None:
        gt_extent, gt_image, gt_mesh = interpolate_2d(ground_true[:, 0:2], ground_true[:, 2])

        plt.figure(figsize=(8, 8))
        plt.pcolor(gt_mesh[0], gt_mesh[1], gt_image, cmap="rainbow")
        # plt.imshow(ground_true_mesh.T, origin="lower", extent=ground_true_extent)
        plt.xlabel("t")
        plt.ylabel("x")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad="3%")
        plt.colorbar(cax=cax)
        plt.tight_layout()
        plt.savefig("./ground_true.png")
        plt.close()

        difference_image = gt_image - pred_image
        plt.figure(figsize=(8, 8))
        plt.pcolor(gt_mesh[0], gt_mesh[1], difference_image, cmap="rainbow")
        # plt.imshow(difference_mesh.T, origin="lower", extent=ground_true_extent)
        plt.xlabel("t")
        plt.ylabel("x")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad="3%")
        plt.colorbar(cax=cax)
        plt.tight_layout()
        plt.savefig("./difference.png")
        plt.close()


