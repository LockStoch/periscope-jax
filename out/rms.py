
import time
import sys
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], ".."))

from stb import strtobool

from msh import load_mesh
from ops import operators

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

def extract(args, data, maxstr, rmsstr):
#-- extract various norms from time series data
    max_ = np.zeros(data.shape[0], dtype=np.float32)
    rms_ = np.zeros(data.shape[0], dtype=np.float32)
    for step in range(data.shape[0]):
        max_[step] = np.max(np.abs(data[step, :]))
        rms_[step] = np.sqrt(np.mean(data[step, :] ** 2))

    save = nc.Dataset(args.stat_file, "a")
    if ("step" not in save.dimensions.keys()):
        save.createDimension("step", data.shape[0])
    if (maxstr not in save.variables.keys()):
        save.createVariable(maxstr, "f4", ("step"))
    if (rmsstr not in save.variables.keys()):
        save.createVariable(rmsstr, "f4", ("step"))
    save[maxstr][:] = max_
    save[rmsstr][:] = rms_
    save.close()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--soln-file", dest="soln_file", type=str,
        required=True, help="Path to user soln file.")

    parser.add_argument(
        "--stat-file", dest="stat_file", type=str,
        default="",
        required=False, help="Path to save RMS stat.")

    parser.add_argument(
        "--save-vars", dest="save_vars", type=str,
        default="ke_cell, du_cell, rv_dual",
        required=False,
        help="Selected ouput variables to save to file.")

    parser.add_argument(
        "--init-step", dest="init_step", type=int,
        default=-1,
        required=False, help="1st step in analysis.")

    parser.add_argument(
        "--stop-step", dest="stop_step", type=int,
        default=-1,
        required=False, help="End step in analysis.")

    parser.add_argument(
        "--show-plot", dest="show_plot",
        type=lambda x: bool(strtobool(str(x.strip()))),
        default=False,
        required=False, help="TRUE to display plots.")

    args = parser.parse_args()

    if ("rv_dual" in args.save_vars.lower()):
        try:
            flow = nc.Dataset(args.soln_file, "r")
            data = np.squeeze(np.asarray(
                flow["rv_dual"][:], dtype=np.float32))
            flow.close()
            extract(args, data, "rot_max", "rot_rms")
        except:
            print("Vorticity not found.")

    if ("du_cell" in args.save_vars.lower()):
        try:
            flow = nc.Dataset(args.soln_file, "r")
            data = np.squeeze(np.asarray(
                flow["du_cell"][:], dtype=np.float32))
            flow.close()
            extract(args, data, "div_max", "div_rms")
        except:
            print("Divergence not found.")
    
    if ("ke_cell" in args.save_vars.lower()):
        try:
            flow = nc.Dataset(args.soln_file, "r")
            data = np.squeeze(np.asarray(
                flow["ke_cell"][:], dtype=np.float32))
            flow.close()
            extract(args, data, "_ke_max", "_ke_rms")
        except:
            print("Kinetic energy not found.")


