
import time
import sys
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse

from scipy.ndimage import gaussian_filter

sys.path.insert(
    1, os.path.join(sys.path[0], ".."))

from stb import strtobool

from msh import load_mesh
from ops import operators
from map import idw_remap

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

def spectra(udiv, urot, rsph):
#-- build kinetic energy spectra from div, rot
#-- also return the enstrophy spectra
#-- apt install libfftw3-dev
    import shtns

    nlat, nlon = udiv.shape

    lmax = nlat - 1
    mmax = nlat - 1

    sh = shtns.sht(lmax, mmax)

    sh.set_grid(nlat=nlat, nphi=nlon)

    sdiv = sh.analys(udiv)
    srot = sh.analys(urot)

    wave_n = np.arange(lmax + 1)

    factor = ((rsph ** 2) / 
        np.maximum(1, (2 * wave_n * (wave_n + 1))))
    factor[0] = 0.0

    ke_rot = factor * np.bincount(
        sh.l, minlength=lmax+1,
        weights=srot.real ** 2 + srot.imag ** 2)

    ke_div = factor * np.bincount(
        sh.l, minlength=lmax+1, 
        weights=sdiv.real ** 2 + sdiv.imag ** 2)
    
    factor[0] = 1.0
    en_tot = ke_rot / factor * 0.5 * (rsph ** 2)

    ke_tot = ke_rot + ke_div

    return wave_n, en_tot, ke_tot, ke_rot, ke_div


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--soln-file", dest="soln_file", type=str,
        required=True, help="Path to user soln file.")

    parser.add_argument(
        "--spec-file", dest="spec_file", type=str,
        default="",
        required=False, help="Path to save spectrum.")

    parser.add_argument(
        "--numthread", dest="numthread", type=int,
        default=1,
        required=False, help="Number of cpu threads.")
    
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

    os.environ["OMP_NUM_THREADS"] = str(args.numthread)

    print("Loading the mesh file...")
    
    mesh = load_mesh(args.soln_file)

    print("Creating the analysis...")

    try:
        flow = nc.Dataset(args.soln_file, "r")
        rotu = np.squeeze(np.asarray(
            flow["rv_dual"][:], dtype=np.float32))
        divu = np.squeeze(np.asarray(
            flow["du_cell"][:], dtype=np.float32))
        flow.close()
    except:
        raise KeyError("Soln data not found.")

    xlen = np.sort(np.sqrt(mesh.cell.area))
    xlen = np.mean(xlen[0:(xlen.size // 10)]) * 8. / 7.
    nlon = np.floor(2. * np.pi * mesh.rsph / xlen)

    nlon, nlat = int(nlon), int(nlon) // 2

    lats = np.linspace(
        -90.0, 90.0, nlat) * np.pi / 180.0
    lons = np.linspace(
        +0.0, 360.0, nlon) * np.pi / 180.0

    xlon, ylat = np.meshgrid(lons, lats)

#-- remap from MPAS mesh to lon-lat grid
    xpos = mesh.rsph * np.cos(xlon) * np.cos(ylat)
    ypos = mesh.rsph * np.sin(xlon) * np.cos(ylat)
    zpos = mesh.rsph * np.sin(ylat)

    qpos = np.vstack((
        xpos.ravel(), ypos.ravel(), zpos.ravel()
                    )).T

    ppos = np.vstack((mesh.cell.xpos.ravel(), 
                      mesh.cell.ypos.ravel(), 
                      mesh.cell.zpos.ravel()
                    )).T

    cmap, __ = idw_remap(ppos, qpos, halo= 7, dpow=4)

    ppos = np.vstack((mesh.vert.xpos.ravel(), 
                      mesh.vert.ypos.ravel(), 
                      mesh.vert.zpos.ravel()
                    )).T

    vmap, __ = idw_remap(ppos, qpos, halo=10, dpow=4)

#-- build spectra and average over steps
    head = 0; tail = divu.shape[0] - 1
    if (args.init_step >= 0): head = args.init_step
    if (args.stop_step >= 0): tail = args.stop_step

    ttic = time.time()
    nout = tail - head + 1 
    oinc = nout // 50
    next = head + oinc
    en_mean_tot = None
    ke_mean_tot = None
    ke_mean_rot = None
    ke_mean_div = None
    for step in range(head, tail + 1):
        if (step >= next):
            next += oinc
            print(".", end="", flush=True)
            
        udiv = np.reshape(cmap * divu[step, :], (nlat, nlon))
        urot = np.reshape(vmap * rotu[step, :], (nlat, nlon))

        udiv = gaussian_filter(
            udiv, sigma=1./2., mode=("reflect", "wrap"))
        urot = gaussian_filter(
            urot, sigma=1./2., mode=("reflect", "wrap"))
        
        udiv = np.flipud(udiv)  # north=>south for shtns
        urot = np.flipud(urot)

        bins, en_tot, ke_tot, ke_rot, ke_div = \
            spectra(udiv, urot, mesh.rsph)

        if (en_mean_tot is None):
            en_mean_tot = en_tot / nout
        else:
            en_mean_tot+= en_tot / nout

        if (ke_mean_tot is None):
            ke_mean_tot = ke_tot / nout
        else:
            ke_mean_tot+= ke_tot / nout

        if (ke_mean_rot is None):
            ke_mean_rot = ke_rot / nout
        else:
            ke_mean_rot+= ke_rot / nout

        if (ke_mean_div is None):
            ke_mean_div = ke_div / nout
        else:
            ke_mean_div+= ke_div / nout

    ttoc = time.time()

    print()
    print("Spectrum calc. complete. (", round(ttoc-ttic, 1), "sec )")

    """
    data = nc.Dataset("_ke.nc", "w")
    data.createDimension("nlon", nlon)
    data.createDimension("nlat", nlat)

    if ("div" not in data.variables.keys()):
        data.createVariable("div", "f4", ("nlat", "nlon"))
    data["div"][:, :] = np.flipud(udiv)

    if ("rot" not in data.variables.keys()):
        data.createVariable("rot", "f4", ("nlat", "nlon"))
    data["rot"][:, :] = np.flipud(urot)
    data.close()
    """

    if (args.spec_file != ""):
        data = nc.Dataset(args.spec_file, "w")
        data.createDimension("nwav", bins.size)
        data.createVariable("wave_n", "f4", ("nwav"))
        data["wave_n"][:] = bins
        data.createVariable("ke_tot", "f4", ("nwav"))
        data["ke_tot"][:] = ke_mean_tot
        data.createVariable("ke_rot", "f4", ("nwav"))
        data["ke_rot"][:] = ke_mean_rot
        data.createVariable("ke_div", "f4", ("nwav"))
        data["ke_div"][:] = ke_mean_div

    plt.figure()
    plt.loglog(bins[4:-1], ke_mean_tot[4:-1], 
               color="blue", linewidth=+2.0)
    """
    plt.loglog(bins[4:-1], ke_mean_rot[4:-1], 
               color="blue", linewidth=+2.0, linestyle=":")
    plt.loglog(bins[4:-1], ke_mean_div[4:-1], 
               color= "red", linewidth=+2.0, linestyle=":")
    """
    plt.loglog(bins[20:100], 8.0e+03 * (bins[20:100] ** -3.0), 
               color="gray", linestyle="--")    
    plt.xlabel(r"wavenumber $[n]$")
    plt.ylabel(r"$\mathrm{E}[n]$")
    plt.grid(True, which="both", ls="-", alpha=0.25)

    plt.figure()
    plt.loglog(bins[4:-1], en_mean_tot[4:-1], 
               color="blue", linewidth=+2.0)
    plt.loglog(bins[20:100], 4.0e+03 * (bins[20:100] ** -1.0), 
               color="gray", linestyle="--")    
    plt.xlabel(r"wavenumber $[n]$")
    plt.ylabel(r"$\mathrm{Z}[n]$")
    plt.grid(True, which="both", ls="-", alpha=0.25)

    if (args.show_plot): plt.show()


