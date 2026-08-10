
import time
import numpy as np

""" SWE time integration via various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda, Jeremy Lilly
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables

from rhs import rhs_tde_d, rhs_all_d, \
                rhs_all_u, rhs_slw_u, rhs_fst_u, \
                rhs_pgf_u, \
                rhs_all_h, rhs_slw_h, rhs_fst_h

def mark_time(cnfg, flow, time):

#-- Update simulation time and interp. on forc. tendencies

    if (flow.xx_time is not None):
    
    #-- linear interp. between prev and next
        prev = flow.xx_time [flow.prev.step]
        next = flow.xx_time [flow.next.step]
        cnfg.timeisnow = time
        cnfg.frc_blend = min(1.0, (time - prev) / 
                                  (next - prev) )

    else:
    
    #-- piecewise const. data, so do nothing
        cnfg.timeisnow = time
        cnfg.frc_blend = reals_t(0.0)
    
    if (cnfg.forc_ramp > 0.0):
        cnfg.frc_start = min(1.0, time / cnfg.forc_ramp)
    else:
        cnfg.frc_start = reals_t(1.0)

    return cnfg


def init_step(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,
              qq_cell):

#-- Set initial estimate of time step size, via CFL bounds

    if (cnfg.time_step <= 0.0):
        raise Exception("Must set fixed DT")

    return cnfg
    

def init_RKFB(cnfg):

#-- Initialise coefficients for user time-stepping schemes

    if   ("RK33" in cnfg.integrate):
        cnfg.fb_weight = np.array([
            0.301666666666667, 0.316666666666667,
            0.366666666666667
            ] )

    """
    else:#"RK43" in cnfg.integrate):
        cnfg.fb_weight = np.array([
            0.000000000000000, 0.500000000000000,
            0.500000000000000, 0.000000000000000
            ] )
    """

    return cnfg


def step_eqns(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              qq_cell):

#-- A single time-step - via user-defined method of choice

    for ssub in range(cnfg.dt_cycles):

        hh_cell, uu_edge, qq_cell = step_try_(
            mesh, mats, 
            flow, cnfg, hh_cell, uu_edge, qq_cell)

        if (cnfg.next_step < 0.0):
            cnfg.time_step = -cnfg.next_step
            print (f">REDO: {-cnfg.next_step:+.2E}")
        else: 
            break

    return hh_cell, uu_edge, qq_cell


def step_try_(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              qq_cell):

#-- A single time-step - via user-defined method of choice

    if   ("RK33" in cnfg.integrate):

        hh_cell, uu_edge, hb_cell = step_RK33(
            mesh, mats, 
            flow, cnfg, hh_cell, uu_edge, None, None)

    """
    elif ("RK43" in cnfg.integrate):

        hh_cell, uu_edge, hb_cell = step_RK43(
            mesh, mats, 
            flow, cnfg, hh_cell, uu_edge, None, None)
    """

    return hh_cell, uu_edge, qq_cell
    
    
def step_bnds(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              qq_cell,
              hh_min_, hh_max_,     # up/lo bounds
              uu_min_, uu_max_,
              qq_min_, qq_max_,
              zt_rms_,
     ke_ave_, ke_rms_, ke_max_,
     dk_ave_, dk_rms_, dk_max_):
              
#-- Expand the min./max. status for each degree of freedom
    
    # don't do this for now re: jax simplification
    """
    if ("hh_cell" in cnfg.stat_vars): \
    hh_min_, hh_max_ = \
        bnd_x_vec(cnfg, hh_cell, hh_min_, hh_max_)

    if ("uu_edge" in cnfg.stat_vars): \
    uu_min_, uu_max_ = \
        bnd_x_vec(cnfg, uu_edge, uu_min_, uu_max_)

    if ("qq_cell" in cnfg.stat_vars): \
    qq_min_, qq_max_ = \
        bnd_x_vec(cnfg, qq_cell, qq_min_, qq_max_)

    zb_cell = variables.zb_cell

    if ("zt_cell" in cnfg.stat_vars): \
    zt_rms_ = \
        nrm_z_vec(cnfg, zb_cell, hh_cell, zt_rms_)

    ke_cell = variables.ke_cell  # from previous

    if ("ke_cell" in cnfg.stat_vars): \
    ke_ave_, ke_rms_, ke_max_ = \
        nrm_x_vec(cnfg, ke_cell, ke_ave_, ke_rms_, 
                                 ke_max_)

    ke_diss = variables.ke_diss

    if ("cd_diss" in cnfg.stat_vars or 
        "nu_diss" in cnfg.stat_vars): \
    dk_ave_, dk_rms_, dk_max_ = \
        nrm_x_vec(cnfg, ke_diss, dk_ave_, dk_rms_, 
                                 dk_max_)
    """
   
    return hh_min_, hh_max_, uu_min_, uu_max_, \
           qq_min_, qq_max_, zt_rms_, \
           ke_ave_, ke_rms_, ke_max_, \
           dk_ave_, dk_rms_, dk_max_


def step_RK33(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              Rh_cell, Ru_edge):    # slow tend.

#-- A 3-stage 3rd/2nd-order RK scheme:
#-- D. Engwirda (2025): 3- and 4-stage forward-backward
#-- Runge-Kutta methods for geophysical flows

#-- drag included via a 2nd-order IMEX scheme

    start_t = cnfg.timeisnow

    k1_step = 1.0 / 3.0 * cnfg.time_step
    k2_step = 2.0 / 3.0 * cnfg.time_step
    k3_step = 1.0 / 1.0 * cnfg.time_step
    dt_step = 1.0 / 1.0 * cnfg.time_step

#-- 1st RK + FB stage

    h0_tend = np.zeros(hh_cell.shape, dtype=hh_cell.dtype)
    u0_tend = np.zeros(uu_edge.shape, dtype=uu_edge.dtype)

    uk_edge = uu_edge.copy()

    cnfg.rhs_stage = 1
    cnfg.time_step = k1_step
    cnfg = mark_time(
        cnfg, flow, start_t + 0. / 1.0 * dt_step)

    BETA = cnfg.fb_weight[0]
    
    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, hh_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, hh_cell, uk_edge)

    h0_tend = rhs_all_h(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, h0_tend)

    h1_cell = hh_cell - k1_step * h0_tend 

    u0_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, u0_tend)
    u0_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, u0_tend)

    hb_cell = (0.0 + 1.0 * BETA) * h1_cell + \
              (1.0 - 1.0 * BETA) * hh_cell

    uk_tend = u0_tend.copy()
    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = uu_edge - k1_step * uk_tend
    

#-- 2nd RK + FB stage

    hk_tend = np.zeros(hh_cell.shape, dtype=hh_cell.dtype)
    uk_tend = np.zeros(uu_edge.shape, dtype=uu_edge.dtype)

    cnfg.rhs_stage = 2
    cnfg.time_step = k2_step
    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 3.0 * dt_step)

    BETA = cnfg.fb_weight[1]
    
    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, h1_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h1_cell, uk_edge)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, hk_tend)

    h2_cell = hh_cell - k2_step * hk_tend

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, uk_tend)    
    
    hb_cell = (0.0 + 1.0 * BETA) * h2_cell + \
              (1.0 - 1.0 * BETA) * hh_cell

    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = uu_edge - k2_step * uk_tend


#-- 3rd RK + FB stage

    hk_tend = np.zeros(hh_cell.shape, dtype=hh_cell.dtype)
    uk_tend = np.zeros(uu_edge.shape, dtype=uu_edge.dtype)

    cnfg.rhs_stage = 3
    cnfg.time_step = k3_step
    cnfg = mark_time(
        cnfg, flow, start_t + 2. / 3.0 * dt_step)

    BETA = cnfg.fb_weight[2]
    
    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, h2_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h2_cell, uk_edge)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, hk_tend)

    hk_tend = +1./4. * h0_tend + 3./4. * hk_tend
    
    h3_cell = hh_cell - k3_step * hk_tend

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, uk_tend)

    uk_tend = +1./4. * u0_tend + 3./4. * uk_tend
    
    hb_cell = (0.0 + 1.0 * BETA) * h3_cell + \
        +3.0 / 4.0 * (1.0 - 2.0 * BETA) * h2_cell + \
        +1.0 / 2.0 * (1.0 / 2.0 + BETA) * hh_cell

    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = uu_edge - k3_step * uk_tend
 

    # disable adaptive time-step fox jaxification
    cnfg.next_step = cnfg.time_step

    return  h3_cell, uk_edge, hb_cell


