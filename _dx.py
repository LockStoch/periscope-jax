
import time
import math
import numpy as np

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables

from ops import gather_apply

def calc_vars(mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                      qq_cell):

#-- compute diagnostic variables from the current state

    ff_dual = variables.ff_vert
    ff_edge = variables.ff_edge
    ff_cell = variables.ff_cell
    
    Xi_tide = variables.Xi_tide  # lagged values
    Xi_self = variables.Xi_self

    uu_filt = variables.uu_filt

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)

    hh_dual, hh_edge, hh_quad, hh_bias = calc_hmap(
        mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge)

    """
    ke_cell, ke_bias = calc_u_ke(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, r2_dual, p2_dual, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)
    """    

    ke_cell = variables.ke_cell
    ke_bias = variables.ke_bias

    rv_dual = variables.rv_dual
    pv_dual = variables.pv_dual
    rv_cell = variables.rv_cell
    pv_cell = variables.pv_cell
    pv_edge = variables.pv_edge
    pv_bias = variables.pv_bias

    nu_turb = variables.nu_turb  # lagged values

    nu_thin = variables.nu_thin

    nu_wave = variables.nu_wave
    os_wave = variables.os_wave

    nu_shoc = variables.nu_shoc
    os_shoc = variables.os_shoc

    return hh_edge, hh_dual, hh_bias, \
           ke_cell, ke_bias, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, \
           pv_edge, pv_bias, \
           vv_edge, nu_turb, \
           nu_wave, os_wave, \
           nu_shoc, os_shoc, \
           nu_thin, uu_filt, \
           Xi_tide, Xi_self


def invariant(mesh, hh_cell, uu_edge):

#-- compute basic scalar diagnostics for the reduced (PGF +
#-- continuity) physics: total volume (should be conserved by
#-- the flux-form continuity scheme up to boundary effects) and
#-- domain rms thickness (a bounded-ness sanity check).

    kp_sums = np.sum(hh_cell * mesh.cell.area)
    hr_sums = np.sqrt(np.mean(hh_cell ** 2))

    return kp_sums, hr_sums


def calc_hmap(mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge):

#-- compute discrete thickness

    ttic = time.time()
    
    hh_dual = variables.hh_dual
    hh_edge = variables.hh_edge
    hh_quad = variables.hh_quad
    hh_bias = variables.hh_bias

    hh_dual[:] = mats.dual_kite_sums * hh_cell
    hh_dual[:]/= mesh.vert.area

    hh_edge[:] = mats.edge_wing_sums * hh_cell
    hh_edge[:]/= mesh.edge.area

    # don't worry about hh_quad or hh_bias for now

    ttoc = time.time()
    tcpu.calc_hmap = tcpu.calc_hmap + (ttoc - ttic)

    return hh_dual, hh_edge, hh_quad, hh_bias
              
              
def calc_perp(mesh, mats, cnfg, uu_edge):

#-- get tangential velocity

    ttic = time.time()

    vv_edge = variables.vv_edge

    vv_edge[:] = mats.edge_lsqr_perp * uu_edge

    ttoc = time.time()
    tcpu.calc_perp = tcpu.calc_perp + (ttoc - ttic)

    return vv_edge
              
              
def tend_hadv(ops, hh_cell, uu_edge, hh_tend):

#-- div. for thickness flux -- JAX, GPU-resident, called from the
#-- jit-compiled RK step in _dt.py. OPS is an ops.JaxOps bundle
#-- (see ops.to_jax); replaces calc_hmap's edge-remap + the old
#-- csr-matrix divergence with padded-gather equivalents.

    hh_edge = gather_apply(
        ops.wing_idx, ops.wing_wgt, hh_cell) / ops.edge_area

    uh_flux = uu_edge * hh_edge

    hh_tend = hh_tend + gather_apply(
        ops.div_idx, ops.div_wgt, uh_flux) / ops.cell_area

    return hh_tend


def tend_upgf(ops, hh_cell, zb_cell, gravity, uu_tend):

#-- get z pressure gradient -- JAX, GPU-resident, see tend_hadv above.

    zt_cell = zb_cell + hh_cell

    uu_tend = uu_tend + gravity * gather_apply(
        ops.grad_idx, ops.grad_wgt, zt_cell)

    return uu_tend

