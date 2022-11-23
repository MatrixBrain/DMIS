# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def interpolate_2d(invar, outvar, plotsize_x=100, plotsize_y=100):

    # create grid
    extent = (invar[:, 0].min(), invar[:, 0].max(), invar[:, 1].min(), invar[:, 1].max())
    _plot_mesh = np.meshgrid(
        np.linspace(extent[0], extent[1], plotsize_x),
        np.linspace(extent[2], extent[3], plotsize_y),
        indexing="ij"
    )

    outvar_interp = griddata(
        invar, outvar, tuple(_plot_mesh)
    )
    return extent, outvar_interp, _plot_mesh


def mesh_plotter_2d(coords, simplices, step=None, ex_path="./mesh_data", name="mesh"):
    """
    function to plot triangular meshes
    """
    assert coords.shape[1] == 2
    plt.figure(figsize=(20, 20), dpi=100)

    plt.triplot(coords[:, 0], coords[:, 1], simplices)

    if step is not None:
        plt.savefig(os.path.join(ex_path, "{}_{}.png".format(name, step)))
    else:
        plt.savefig(os.path.join(ex_path, "{}.png".format(name)))
    plt.close()

